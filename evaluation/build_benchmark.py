"""Build a frozen evaluation benchmark file for a user.

Two-phase design: build-once / run-many. All randomness (slate composition,
shuffle order, Task C scenario instantiation, C1 probe selection) is resolved
here and written to `evaluation/benchmarks/{user_id}/benchmark.json`.
`run_inference.py` consumes that file and performs no runtime RNG, so:
  - two runs of the same config produce the same instances,
  - mode-A vs mode-B comparisons see identical inputs,
  - different models can be compared apples-to-apples.

Per-instance seeding: each test item derives its RNG from
`(rng_seed, source_object_id)`. Adding or removing one item does NOT cascade-
shift every other slate.

Rebuild when the underlying backend data changes — the file records a
`backend_hash` so staleness is detectable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.backend_query import APPS, BackendQuery
from evaluation.inference_utils import TestItem, build_gt_slice, load_test_items
from evaluation import scenarios as scenarios_mod
from evaluation.tasks import chatbot_response as cb_task

BENCHMARK_VERSION = "v2"
SOCIAL_APPS = ("instagram", "facebook", "threads")


# --- Per-instance RNG ------------------------------------------------------

def _instance_rng(global_seed: int, instance_key: str) -> random.Random:
    """Derive an independent RNG per instance — stable across item additions."""
    return random.Random(f"{global_seed}:{instance_key}")


# --- Backend hash (for staleness detection) --------------------------------

def compute_backend_hash(backend_dir: str | Path, user_id: str) -> str:
    h = hashlib.sha256()
    base = Path(backend_dir) / user_id
    for name in sorted(["profile.json", "instagram.json", "facebook.json", "threads.json", "chatbot.json"]):
        p = base / name
        if p.exists():
            with p.open("rb") as f:
                h.update(f.read())
    return h.hexdigest()[:16]


# --- Task A: slate instances -----------------------------------------------

def _content_to_item(event_content: dict, hashtags: list, content_type: str) -> dict:
    return {
        "title": event_content.get("title") or "",
        "caption": event_content.get("caption") or "",
        "hashtags": hashtags,
        "content_type": content_type,
    }


def _preference_to_item(pref: dict) -> dict:
    persona = pref.get("persona_item") or ""
    return {
        "title": persona[:80],
        "caption": persona,
        "hashtags": pref.get("source_hashtags") or [],
        "content_type": "text",
    }


def build_slate_instance(test: TestItem, bq: BackendQuery, rng: random.Random) -> dict:
    candidates: list[dict] = []
    t = test.source_timestamp

    # 1x held-out positive
    held_out = _content_to_item(test.content, test.source_hashtags, test.content.get("content_type") or "text")
    held_out["_origin"] = "held_out"
    candidates.append(held_out)

    # 3x topically-irrelevant
    irrels = test.over_personalization_irrelevant[:]
    rng.shuffle(irrels)
    for p in irrels[:3]:
        c = _preference_to_item(p)
        c["_origin"] = "irrelevant"
        candidates.append(c)

    # 3x known-disliked
    neg_prefs = bq.get_preferences(user_id=test.user_id, since_timestamp=t, polarity="negative")
    rng.shuffle(neg_prefs)
    for p in neg_prefs[:3]:
        c = _preference_to_item(p)
        c["_origin"] = "negative"
        candidates.append(c)

    # 3x plausible-random from unused hashtags
    used = {h.lower() for p in bq.get_preferences(user_id=test.user_id, since_timestamp=t) for h in p.get("source_hashtags", [])}
    unused_hashtags: list[str] = []
    for app in APPS:
        for e in bq.get_events(user_id=test.user_id, app=app, since_timestamp=t):
            for h in e.get("source_hashtags", []):
                if h.lower() not in used:
                    unused_hashtags.append(h)
    rng.shuffle(unused_hashtags)
    for h in unused_hashtags[:3]:
        candidates.append({
            "title": f"Trending content about {h}",
            "caption": f"A popular post mentioning {h}.",
            "hashtags": [h],
            "content_type": "text",
            "_origin": "random",
        })

    while len(candidates) < 10:
        candidates.append({
            "title": "General content",
            "caption": "Unspecified item.",
            "hashtags": [],
            "content_type": "text",
            "_origin": "filler",
        })

    rng.shuffle(candidates)
    slate = []
    held_out_idx = 0
    origin_by_idx: list[str] = []
    for idx, c in enumerate(candidates):
        slate.append({
            "idx": idx,
            "app": test.app,
            "title": c["title"],
            "caption": c["caption"],
            "hashtags": c["hashtags"],
            "content_type": c["content_type"],
        })
        origin_by_idx.append(c["_origin"])
        if c["_origin"] == "held_out":
            held_out_idx = idx

    return {
        "test_id": test.source_object_id,
        "app": test.app,
        "source_timestamp": test.source_timestamp,
        "formatted_timestamp": test.formatted_timestamp,
        "query_hashtags": test.source_hashtags,
        "slate": slate,
        "held_out_idx": held_out_idx,
        "origin_by_idx": origin_by_idx,
    }


# --- Task B: chatbot instances ---------------------------------------------

def build_chatbot_instance(bq: BackendQuery, test: TestItem) -> dict | None:
    user_query, prior = cb_task._extract_query_and_prior(test)
    if not user_query:
        return None
    action = (test.interaction_format or {}).get("action", "")
    # Freeze the same-day TARGET/AVOID slice so scoring is fully reproducible.
    gt_slice = build_gt_slice(bq, test)
    return {
        "test_id": test.source_object_id,
        "source_timestamp": test.source_timestamp,
        "formatted_timestamp": test.formatted_timestamp,
        "user_query": user_query,
        "prior_conversation": prior,
        "polarity": test.polarity,
        "action": action,
        "held_out_preference": {
            "persona_item": test.preference.get("persona_item"),
            "category": test.preference.get("category"),
        },
        "source_hashtags": test.source_hashtags,
        "gt_slice": gt_slice,
    }


# --- Task C1: repetition-fatigue probes ------------------------------------

def build_c1_instances(bq: BackendQuery, user_id: str, t_probe: int, min_positive_count: int = 10) -> list[dict]:
    hashtag_rows = bq.hashtag_summary(user_id=user_id, since_timestamp=t_probe)
    out: list[dict] = []
    for row in hashtag_rows:
        if row["positive"] < min_positive_count:
            continue
        for app in SOCIAL_APPS:
            probe = scenarios_mod.build_repetition_probe(bq, user_id, t_probe, app, row["hashtag"])
            if probe:
                out.append({
                    "probe_id": f"{user_id}_{app}_{row['hashtag'].lstrip('#')}",
                    "t_probe": t_probe,
                    **probe,
                })
                break
    return out


# --- Task C2: scenario instances -------------------------------------------

def build_c2_instances(bq: BackendQuery, user_id: str, t_probe: int, rng_seed: int) -> list[dict]:
    scs = scenarios_mod.build_all_scenarios(bq, user_id, t_probe, seed=rng_seed)
    return [
        {
            "scenario_id": f"{user_id}_{s['name']}",
            "t_probe": t_probe,
            **s,
        }
        for s in scs
    ]


# --- Task C3: restraint instances ------------------------------------------

def build_c3_instance(test: TestItem, rng: random.Random) -> dict | None:
    if not test.over_personalization_irrelevant:
        return None
    irrels = list(test.over_personalization_irrelevant)
    candidates_raw = [
        {
            "persona_item": test.preference.get("persona_item"),
            "category": test.preference.get("category"),
            "_origin": "held_out",
        }
    ] + [{**p, "_origin": "irrelevant"} for p in irrels]
    rng.shuffle(candidates_raw)
    candidates = []
    origin_by_idx = []
    for i, c in enumerate(candidates_raw):
        candidates.append({"idx": i, "persona_item": c.get("persona_item"), "category": c.get("category")})
        origin_by_idx.append(c["_origin"])
    return {
        "test_id": test.source_object_id,
        "app": test.app,
        "source_timestamp": test.source_timestamp,
        "parent_event": {
            "source_hashtags": test.source_hashtags,
            "content": test.content,
        },
        "candidates": candidates,
        "origin_by_idx": origin_by_idx,
        "held_out_persona_item": test.preference.get("persona_item", ""),
        "irrelevant_persona_items": [p.get("persona_item", "") for p in irrels],
    }


# --- Top-level build -------------------------------------------------------

def build_benchmark(
    backend_dir: str | Path,
    user_id: str,
    rng_seed: int = 0,
) -> dict:
    bq = BackendQuery(backend_dir)
    test_items = load_test_items(backend_dir, user_id)
    if not test_items:
        raise SystemExit(f"No test items found for user {user_id} under {backend_dir}/")

    # Task A slates — per-item seeded.
    slate_instances = []
    for t in test_items:
        if t.app not in SOCIAL_APPS:
            continue
        rng = _instance_rng(rng_seed, f"slate:{t.source_object_id}")
        slate_instances.append(build_slate_instance(t, bq, rng))

    # Task B chatbot instances — no randomness, but frozen query/prior + GT slice.
    chatbot_instances = []
    for t in test_items:
        if t.app != "chatbot":
            continue
        inst = build_chatbot_instance(bq, t)
        if inst is not None:
            chatbot_instances.append(inst)

    # Task C1/C2: one probe set per user, anchored at the latest test timestamp.
    t_probe = max(t.source_timestamp for t in test_items)
    c1_instances = build_c1_instances(bq, user_id, t_probe)
    c2_instances = build_c2_instances(bq, user_id, t_probe, rng_seed=rng_seed)

    # Task C3 — per-item seeded, only social-app tests with irrelevant distractors.
    c3_instances = []
    for t in test_items:
        if t.app not in SOCIAL_APPS or not t.over_personalization_irrelevant:
            continue
        rng = _instance_rng(rng_seed, f"c3:{t.source_object_id}")
        inst = build_c3_instance(t, rng)
        if inst is not None:
            c3_instances.append(inst)

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "user_id": user_id,
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rng_seed": rng_seed,
        "backend_hash": compute_backend_hash(backend_dir, user_id),
        "counts": {
            "test_items": len(test_items),
            "slate_ranking": len(slate_instances),
            "chatbot_response": len(chatbot_instances),
            "c1_fatigue": len(c1_instances),
            "c2_scenarios": len(c2_instances),
            "c3_restraint": len(c3_instances),
        },
        "slate_ranking": slate_instances,
        "chatbot_response": chatbot_instances,
        "c1_fatigue": c1_instances,
        "c2_scenarios": c2_instances,
        "c3_restraint": c3_instances,
    }


def default_benchmark_path(user_id: str) -> Path:
    return Path("benchmark") / user_id / "benchmark.json"


def main():
    parser = argparse.ArgumentParser(description="Build a frozen eval benchmark for a user.")
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--backend_dir", default="backend")
    parser.add_argument("--rng_seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="Output path (default: evaluation/benchmarks/{user_id}/benchmark.json)")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else default_benchmark_path(args.user_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bm = build_benchmark(args.backend_dir, args.user_id, rng_seed=args.rng_seed)
    with out_path.open("w") as f:
        json.dump(bm, f, ensure_ascii=False, indent=2)
    print(f"[build_benchmark] wrote {out_path}")
    print(f"[build_benchmark] counts: {bm['counts']}")
    print(f"[build_benchmark] backend_hash: {bm['backend_hash']}")


if __name__ == "__main__":
    main()
