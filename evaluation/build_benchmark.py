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
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.backend_query import APPS, BackendQuery
from evaluation.inference_utils import TestItem, build_gt_slice, load_test_items, DAY_SECONDS
from evaluation import scenarios as scenarios_mod
from evaluation import metrics as metrics_mod
from evaluation import prompts as prompts_mod
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


# --- Task B: proactive chatbot instances (with curation + control arm) ----

# Actions that move a query out of B (into C) — user is explicitly asking for
# or against personalization, so it no longer tests proactive capability.
_ACTION_EXCLUDE = {"asked_not_to_personalize", "asked_to_forget", "asked_to_personalize", "asked_for_recommendation"}

# Regex patterns detecting in-text personalization asks.
_EXPLICIT_ASK_RE = re.compile(
    r"\b(recommend (?:me|for me|something)|what (?:do|would) i (?:like|enjoy)|based on my|personalize for me|tailored to me)\b",
    re.IGNORECASE,
)

# Conversation-continuation pronouns that indicate the query references a prior
# assistant turn — these are instruction-following tests, not proactive tests.
_CONTINUATION_RE = re.compile(
    r"\b(that (?:last|one|answer|line|part|bit|down|thing|stuff|one|paragraph|bullet)|the (?:above|prior|previous|one you|last)|your (?:response|answer|reply))\b",
    re.IGNORECASE,
)

# Queries starting with these discourse markers are almost always follow-ups.
_CONTINUATION_STARTERS = (
    "yeah", "ok ", "okay", "good", "perfect", "great", "nice", "actually",
    "and ", "but ", "also", "so ", "hmm", "uh ", "um ", "wait", "right",
    "sure", "got it", "makes sense", "huh",
)


def _load_all_chatbot_events(backend_dir: str | Path, user_id: str) -> list[dict]:
    path = Path(backend_dir) / user_id / "chatbot.json"
    if not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def _candidate_from_event(event: dict, held_out_preference: dict | None = None) -> dict | None:
    """Extract a proactive-query candidate from a chatbot event. Returns None
    if no usable user query is present.
    """
    fmt = event.get("interaction_format") or {}
    user_msg = fmt.get("user_message") or ""
    convo = event.get("conversation") or []
    # Fallback: if no user_message, take the last user turn in the convo.
    if not user_msg and convo:
        for m in reversed(convo):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break
    if not user_msg:
        return None
    # Build prior conversation = everything before the user's final message.
    prior: list[dict] = []
    if convo:
        for m in convo:
            if m.get("role") == "user" and m.get("content", "").strip() == user_msg.strip():
                break
            prior.append(m)
    return {
        "source_object_id": str(event.get("source_object_id", "")),
        "source_timestamp": int(event.get("source_timestamp", 0)),
        "formatted_timestamp": event.get("formatted_timestamp", ""),
        "action": fmt.get("action", ""),
        "user_query": user_msg,
        "prior_conversation": prior,
        "source_hashtags": event.get("source_hashtags", []),
        "held_out_preference": held_out_preference,
    }


def _fresh_start_ok(candidate: dict) -> bool:
    """Fresh-start filter: a *proactive* benchmark query must stand alone.

    Strict: the user query must be the very first message of a session — no
    prior assistant turns. Also rejects continuations-by-syntax as defensive
    check.
    """
    q = (candidate.get("user_query") or "").strip()
    if not q:
        return False
    q_low = q.lower()
    if any(q_low.startswith(s) for s in _CONTINUATION_STARTERS):
        return False
    if _CONTINUATION_RE.search(q_low):
        return False
    # Hard requirement: no prior assistant turn. Prior user-only turns are fine
    # (some conversations open with the user restating themselves before the
    # assistant has responded), but any assistant turn means we're mid-conversation.
    if any(m.get("role") == "assistant" for m in (candidate.get("prior_conversation") or [])):
        return False
    return True


def _proactive_filter_ok(candidate: dict) -> bool:
    """Drop candidates that explicitly ask for personalization or are carve-outs."""
    if candidate["action"] in _ACTION_EXCLUDE:
        return False
    if _EXPLICIT_ASK_RE.search(candidate["user_query"] or ""):
        return False
    return True


def _blind_check(query: str, judge_llm) -> dict:
    """Build-time LLM call: estimate how much personalization would help.

    Returns `{blind_score: int 0-3, generic_answer: str, reasoning: str}`.
    On failure, returns score = 2 (moderate — conservative default keeps query).
    """
    if judge_llm is None:
        return {"blind_score": 2, "generic_answer": "", "reasoning": "blind-check skipped (--skip_blind_check)"}
    prompt = prompts_mod.query_blind_check_prompt(query)
    try:
        resp = judge_llm(prompt)
        from data_preparation.utils import extract_json_from_response
        parsed = extract_json_from_response(resp) or {}
        score = int(parsed.get("personalization_value", 2) or 2)
        return {
            "blind_score": max(0, min(3, score)),
            "generic_answer": parsed.get("generic_answer", "") or "",
            "reasoning": parsed.get("reasoning", "") or "",
        }
    except Exception as exc:
        return {"blind_score": 2, "generic_answer": "", "reasoning": f"blind-check error: {exc}"}


def _dedup_candidates(cands: list[dict]) -> list[dict]:
    """Drop near-identical queries (normalized string equality; simple but effective)."""
    seen: set[str] = set()
    out = []
    for c in cands:
        key = " ".join((c["user_query"] or "").lower().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _dedup_user_prefs(bq: BackendQuery, user_id: str, t_test: int) -> list[dict]:
    """Collect structured, deduplicated user preferences from all app events
    before T_test. profile.json.preferences is a flat list of strings; app-event
    preferences are the structured form with persona_item / category / source_hashtags.
    Dedup key: persona_item text.
    """
    all_prefs = bq.get_preferences(user_id=user_id, since_timestamp=t_test)
    seen: set[str] = set()
    out: list[dict] = []
    for p in all_prefs:
        item = p.get("persona_item") or ""
        if not item or item in seen:
            continue
        seen.add(item)
        out.append({
            "persona_item": item,
            "category": p.get("category", ""),
            "source_hashtags": p.get("source_hashtags") or [],
            "source_app": p.get("source_app", ""),
            "source_interaction_type": p.get("source_interaction_type", ""),
        })
    return out


def _build_top_k_relevant_prefs(
    all_prefs: list[dict],
    query: str,
    query_hashtags: list[str],
    k: int = 5,
) -> list[dict]:
    if not all_prefs:
        return []
    q_tokens = metrics_mod.tokenize(query)
    q_hash = {h.lower().lstrip("#") for h in (query_hashtags or [])}

    def score(p: dict) -> float:
        txt = (p.get("persona_item") or "") + " " + (p.get("category") or "")
        s = metrics_mod.jaccard(query, txt)
        cat = (p.get("category") or "").lower()
        if cat and any(tok in cat for tok in q_tokens):
            s += 0.5
        for h in (p.get("source_hashtags") or []):
            if h.lower().lstrip("#") in q_hash:
                s += 1.0
        return s

    scored = sorted(all_prefs, key=score, reverse=True)
    top = scored[:k]
    return [
        {
            "persona_item": p["persona_item"],
            "category": p["category"],
            "source_hashtags": p["source_hashtags"],
            "relevance_score": round(score(p), 3),
        }
        for p in top
    ]


def _build_privacy_flagged_prefs(
    profile: dict,
    all_prefs: list[dict],
) -> list[dict]:
    hidden = (profile or {}).get("hidden_personas", []) or []
    privacy_types = {"intimate_interest", "covert_concern", "compensatory_need"}
    sensitive_tags: set[str] = set()
    sensitive_cats: set[str] = set()
    for h in hidden:
        if (h.get("privacy_ratio") or 0) > 0.7 or h.get("type") in privacy_types:
            for tag in (h.get("evidence_hashtags") or []):
                sensitive_tags.add(tag.lower().lstrip("#"))
            lbl = (h.get("label") or "").lower()
            if lbl:
                sensitive_cats.add(lbl)
    out = []
    for p in all_prefs:
        p_tags = {h.lower().lstrip("#") for h in (p.get("source_hashtags") or [])}
        p_cat = (p.get("category") or "").lower()
        if p_tags & sensitive_tags or p_cat in sensitive_cats:
            out.append({
                "persona_item": p["persona_item"],
                "category": p["category"],
                "source_hashtags": p["source_hashtags"],
            })
    return out


def _build_post_test_window(bq: BackendQuery, user_id: str, t_test: int, window_hours: int = 48) -> dict:
    """Source B — events in [T_test, T_test + window]. The agent never sees this.

    Loads raw events (bypassing time-mask) since this is for scoring only.
    """
    lo = t_test
    hi = t_test + window_hours * 3600
    base = Path(bq.base) / user_id
    pos_prefs: list[dict] = []
    neg_prefs: list[dict] = []
    engagements: list[dict] = []
    for app in APPS:
        p = base / f"{app}.json"
        if not p.exists():
            continue
        with p.open() as f:
            events = json.load(f)
        for e in events:
            ts = int(e.get("source_timestamp", 0))
            if ts < lo or ts > hi:
                continue
            it = e.get("source_interaction_type", "")
            engagements.append({
                "event_id": str(e.get("source_object_id", "")),
                "app": app,
                "source_timestamp": ts,
                "source_hashtags": e.get("source_hashtags", []),
                "source_interaction_type": it,
                "persona_items": [p.get("persona_item") for p in (e.get("preferences") or []) if p.get("persona_item")],
            })
            for pref in (e.get("preferences") or []):
                item = {
                    "persona_item": pref.get("persona_item"),
                    "category": pref.get("category"),
                    "source_hashtags": e.get("source_hashtags", []),
                }
                if "positive" in it:
                    pos_prefs.append(item)
                elif "negative" in it:
                    neg_prefs.append(item)
    return {
        "window_hours": window_hours,
        "post_test_engagements": engagements,
        "post_test_positives": pos_prefs,
        "post_test_negatives": neg_prefs,
    }


def build_task_b_arms(
    backend_dir: str | Path,
    bq: BackendQuery,
    user_id: str,
    test_items: list[TestItem],
    blind_check_llm=None,
    blind_check_limit: int | None = None,
) -> dict:
    """Build the two B arms: proactive + control.

    Pulls from ALL chatbot events (not just test events), applies the 4-stage
    filter, labels by arm. Each instance carries same-day slice + top-K prefs
    + privacy-flagged slice + post-T_test Source B.
    """
    profile = bq.get_full_profile(user_id)
    all_events = _load_all_chatbot_events(backend_dir, user_id)

    # Map source_object_id → held-out preference for test events (used when
    # the candidate corresponds to a test event, to preserve continuity with v1).
    test_index: dict[str, dict] = {}
    for t in test_items:
        if t.app == "chatbot":
            test_index[t.source_object_id] = {
                "persona_item": t.preference.get("persona_item"),
                "category": t.preference.get("category"),
            }

    # Stage 1: extract raw candidates.
    candidates = []
    for e in all_events:
        held_out = test_index.get(str(e.get("source_object_id", "")))
        c = _candidate_from_event(e, held_out_preference=held_out)
        if c is not None:
            candidates.append(c)

    # Stage 2: fresh-start filter.
    candidates = [c for c in candidates if _fresh_start_ok(c)]
    # Stage 3: proactive filter.
    candidates = [c for c in candidates if _proactive_filter_ok(c)]
    # Stage 4: dedup.
    candidates = _dedup_candidates(candidates)

    # Stage 5: blind-check → score each, split into proactive vs control.
    if blind_check_limit is not None:
        candidates = candidates[:blind_check_limit]
    proactive: list[dict] = []
    control: list[dict] = []
    for c in candidates:
        bc = _blind_check(c["user_query"], blind_check_llm)
        c["blind_check_score"] = bc["blind_score"]
        c["blind_check_generic_answer"] = bc["generic_answer"]
        if bc["blind_score"] >= 2:
            proactive.append(c)
        else:
            control.append(c)

    # Fallback: if control arm is empty, grab the 3 lowest-scoring candidates.
    if not control and len(candidates) >= 3:
        control_picks = sorted(candidates, key=lambda x: x["blind_check_score"])[:3]
        control = control_picks
        proactive_ids = {c["source_object_id"] for c in control_picks}
        proactive = [c for c in proactive if c["source_object_id"] not in proactive_ids]

    # Enrich instances with ground truth (TARGET/AVOID + top-K + privacy + Source B).
    def _finalize(c: dict, arm: str) -> dict:
        t_test = c["source_timestamp"]
        # Dedup'd preferences from all events before this candidate's timestamp.
        all_prefs = _dedup_user_prefs(bq, user_id, t_test)
        # Same-day slice anchored on this candidate's timestamp.
        gt_slice = _build_gt_slice_for_candidate(bq, user_id, t_test, c["held_out_preference"], c["source_hashtags"])
        top_k = _build_top_k_relevant_prefs(all_prefs, c["user_query"], c["source_hashtags"])
        privacy_flagged = _build_privacy_flagged_prefs(profile, all_prefs)
        source_b = _build_post_test_window(bq, user_id, t_test)
        return {
            "test_id": c["source_object_id"],
            "arm": arm,
            "source_timestamp": t_test,
            "formatted_timestamp": c["formatted_timestamp"],
            "user_query": c["user_query"],
            "prior_conversation": c["prior_conversation"],
            "action": c["action"],
            "source_hashtags": c["source_hashtags"],
            "held_out_preference": c.get("held_out_preference"),
            "blind_check_score": c["blind_check_score"],
            "blind_check_generic_answer": c["blind_check_generic_answer"],
            "gt_slice": gt_slice,
            "top_k_relevant_prefs": top_k,
            "privacy_flagged_prefs": privacy_flagged,
            "post_test_window": source_b,
        }

    return {
        "chatbot_response_proactive": [_finalize(c, "proactive") for c in proactive],
        "chatbot_response_control":   [_finalize(c, "control") for c in control],
    }


def _build_gt_slice_for_candidate(
    bq: BackendQuery,
    user_id: str,
    t_test: int,
    held_out_preference: dict | None,
    source_hashtags: list[str],
) -> dict:
    """Same-day TARGET/AVOID slice anchored on an arbitrary chatbot candidate
    (not necessarily a test event). Mirrors build_gt_slice but takes looser inputs.
    """
    lo, hi = t_test - DAY_SECONDS, t_test + DAY_SECONDS
    base = Path(bq.base) / user_id
    target: list[dict] = []
    avoid: list[dict] = []
    for app in APPS:
        p = base / f"{app}.json"
        if not p.exists():
            continue
        with p.open() as f:
            events = json.load(f)
        for e in events:
            ts = int(e.get("source_timestamp", 0))
            if ts < lo or ts > hi:
                continue
            it = e.get("source_interaction_type", "")
            for pref in (e.get("preferences") or []):
                item = {
                    "persona_item": pref.get("persona_item"),
                    "category": pref.get("category"),
                    "source_hashtags": e.get("source_hashtags", []),
                    "polarity": "positive" if "positive" in it else "negative" if "negative" in it else "other",
                    "source_app": app,
                    "source_timestamp": ts,
                    "source_interaction_type": it,
                }
                if item["polarity"] == "positive":
                    target.append(item)
                elif item["polarity"] == "negative":
                    avoid.append(item)

    if held_out_preference and held_out_preference.get("persona_item"):
        held = {
            "persona_item": held_out_preference.get("persona_item"),
            "category": held_out_preference.get("category"),
            "source_hashtags": source_hashtags,
            "polarity": "positive",
            "source_app": "chatbot",
            "source_timestamp": t_test,
            "is_held_out": True,
        }
        seen = {held["persona_item"]}
        target_dedup = [held]
        for it in target:
            if it["persona_item"] not in seen:
                target_dedup.append(it)
                seen.add(it["persona_item"])
        target = target_dedup

    return {
        "t_test": t_test,
        "window_seconds": DAY_SECONDS,
        "target": target,
        "avoid": avoid,
    }


# --- Legacy Task B builder (kept for schema back-compat; unused in v2) ------

def build_chatbot_instance(bq: BackendQuery, test: TestItem) -> dict | None:
    user_query, prior = cb_task._extract_query_and_prior(test)
    if not user_query:
        return None
    action = (test.interaction_format or {}).get("action", "")
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


# --- Task C1a: counterfactual history-diff pairs ---------------------------

def build_c1a_pairs(
    bq: BackendQuery,
    user_id: str,
    test_items: list[TestItem],
    max_pairs: int = 5,
    window_hours: int = 24,
    min_diff_events: int = 3,
) -> list[dict]:
    """Find pairs of (T_early, T_late) such that events in [T_early, T_late]
    include ≥ min_diff_events, and the diff's hashtag center differs from
    the user's long-term dominant category (shifts topical interest).

    Each pair freezes:
    - `t_early`, `t_late`, `target_app`
    - `diff_events`: summaries of the events added between the two moments
    - `shared_context_size`: event count at T_early (rough indicator)
    - `dominant_category_pre`: user's top category before T_early
    - `shift_category`: dominant category of the diff events

    Returns up to `max_pairs` pairs spanning diverse (app, category) slots.
    """
    if not test_items:
        return []
    # Anchor candidate moments at social-app test item timestamps (we already
    # know the user engages meaningfully at these times).
    anchors = sorted(
        {(t.app, t.source_timestamp) for t in test_items if t.app in SOCIAL_APPS},
        key=lambda x: x[1],
    )
    if len(anchors) < 2:
        return []

    # Dominant category = most-frequent positive preference category across history.
    base = Path(bq.base) / user_id
    cat_counts: dict[str, int] = {}
    for app in APPS:
        p = base / f"{app}.json"
        if not p.exists():
            continue
        with p.open() as f:
            events = json.load(f)
        for e in events:
            if "positive" not in e.get("source_interaction_type", ""):
                continue
            for pref in (e.get("preferences") or []):
                c = pref.get("category")
                if c:
                    cat_counts[c] = cat_counts.get(c, 0) + 1
    dominant_pre = max(cat_counts.items(), key=lambda kv: kv[1])[0] if cat_counts else None

    pairs: list[dict] = []
    for i, (app_early, t_early) in enumerate(anchors[:-1]):
        for app_late, t_late in anchors[i + 1:]:
            if t_late - t_early > window_hours * 3600 or t_late - t_early <= 0:
                continue
            # Collect diff events: events in (t_early, t_late] for any app.
            diff_events: list[dict] = []
            for app in APPS:
                p = base / f"{app}.json"
                if not p.exists():
                    continue
                with p.open() as f:
                    events = json.load(f)
                for e in events:
                    ts = int(e.get("source_timestamp", 0))
                    if t_early < ts <= t_late:
                        diff_events.append({
                            "event_id": str(e.get("source_object_id", "")),
                            "app": app,
                            "t": ts,
                            "hashtags": e.get("source_hashtags", []),
                            "categories": [pref.get("category") for pref in (e.get("preferences") or []) if pref.get("category")],
                        })
            if len(diff_events) < min_diff_events:
                continue
            # Shift-check: does the diff introduce a category that's not dominant_pre?
            diff_cats = [c for e in diff_events for c in e["categories"]]
            non_dominant_diff = [c for c in diff_cats if c != dominant_pre]
            if not non_dominant_diff:
                continue
            shift_category = max(set(non_dominant_diff), key=non_dominant_diff.count)
            pair_id = f"{user_id}_c1a_{t_early}_{t_late}"
            pairs.append({
                "pair_id": pair_id,
                "target_app": app_late,
                "t_early": t_early,
                "t_late": t_late,
                "diff_events": diff_events[:15],
                "dominant_category_pre": dominant_pre,
                "shift_category": shift_category,
                "query": f"Recommend something on {app_late} this user would engage with next.",
            })
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


# --- Task C1b: chatbot-sequence preference repetition ----------------------

def build_c1b_sequence(
    b_proactive_instances: list[dict],
    max_seq_len: int = 5,
    min_distinct_categories: int = 3,
) -> list[dict]:
    """Assemble one or more sequences from diverse-topic B-proactive queries.

    Each sequence requires queries spanning ≥ min_distinct_categories distinct
    top-1-preference categories (computed from each instance's top_k_relevant_prefs).
    Returns a list of sequence instances (usually 1–2 per user).
    """
    if not b_proactive_instances:
        return []
    # Bucket B-proactive instances by top-1 category (their strongest preference signal).
    by_cat: dict[str, list[dict]] = {}
    for inst in b_proactive_instances:
        top = inst.get("top_k_relevant_prefs") or []
        if not top:
            continue
        cat = top[0].get("category") or "uncategorized"
        by_cat.setdefault(cat, []).append(inst)

    if len(by_cat) < min_distinct_categories:
        return []

    # Take one instance per category, up to max_seq_len, in timestamp order.
    picks: list[dict] = []
    for cat, insts in list(by_cat.items())[:max_seq_len]:
        picks.append(sorted(insts, key=lambda x: x.get("source_timestamp", 0))[0])
    picks.sort(key=lambda x: x.get("source_timestamp", 0))

    if len({p["top_k_relevant_prefs"][0]["category"] for p in picks if p.get("top_k_relevant_prefs")}) < min_distinct_categories:
        return []

    sequence = {
        "sequence_id": f"c1b_seq_0",
        "queries": [
            {
                "source_test_id": p["test_id"],
                "source_timestamp": p["source_timestamp"],
                "user_query": p["user_query"],
                "top_k_relevant_prefs": p["top_k_relevant_prefs"],
            }
            for p in picks
        ],
    }
    return [sequence]


# --- Task C4: do-not-personalize button regeneration ----------------------

def build_c4_instances(b_proactive_instances: list[dict]) -> list[dict]:
    """Convert each B-proactive instance into a two-turn C4 regen probe.

    At eval time, the driver first gets the B-proactive response (or uses a
    held-out "original personalized response" field if we've cached one), then
    sends the button-click regen prompt and scores the regen.
    """
    out = []
    for b in b_proactive_instances:
        if not (b.get("held_out_preference") or {}).get("persona_item"):
            continue
        out.append({
            "test_id": f"{b['test_id']}_c4",
            "source_test_id": b["test_id"],
            "source_timestamp": b["source_timestamp"],
            "user_query": b["user_query"],
            "prior_conversation": b["prior_conversation"],
            "held_out_preference": b["held_out_preference"],
            "top_k_relevant_prefs": b.get("top_k_relevant_prefs") or [],
            "blind_check_generic_answer": b.get("blind_check_generic_answer", ""),
        })
    return out


# --- Task C1: repetition-fatigue probes (LEGACY — kept but deprecated) -----

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
    blind_check_llm=None,
    blind_check_limit: int | None = None,
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

    # Task B (v2) — proactive + control arms with build-time curation.
    b_arms = build_task_b_arms(
        backend_dir=backend_dir,
        bq=bq,
        user_id=user_id,
        test_items=test_items,
        blind_check_llm=blind_check_llm,
        blind_check_limit=blind_check_limit,
    )

    # Task C1a/C1b/C2/C3/C4.
    t_probe = max(t.source_timestamp for t in test_items)
    c1a_pairs = build_c1a_pairs(bq, user_id, test_items)
    c1b_sequences = build_c1b_sequence(b_arms["chatbot_response_proactive"])
    c2_instances = build_c2_instances(bq, user_id, t_probe, rng_seed=rng_seed)
    c4_instances = build_c4_instances(b_arms["chatbot_response_proactive"])

    # Agentic tasks T6-T19 — all share t_probe.
    from evaluation.tasks.agentic_tasks import ALL_BUILDERS as _AGENTIC_BUILDERS
    agentic_buckets: dict[str, list[dict]] = {}
    for task_id, builder in _AGENTIC_BUILDERS.items():
        try:
            agentic_buckets[task_id] = builder(bq, user_id, t_probe)
        except Exception as exc:
            agentic_buckets[task_id] = []
            print(f"[build_benchmark] WARN: {task_id} builder failed: {exc}")

    # Task E2 — @ai proactive recommendation (R9 addition)
    try:
        from evaluation.tasks.e2_at_ai_followup import build_e2_at_ai_followup
        e2_instances = build_e2_at_ai_followup(bq, user_id, rng_seed=rng_seed)
    except Exception as exc:
        e2_instances = []
        print(f"[build_benchmark] WARN: e2_at_ai_followup builder failed: {exc}")

    # Task E3 — multi-day proactive daily briefing
    try:
        from evaluation.tasks.e3_daily_briefing_multi import build_e3_daily_briefing_multi
        e3_instances = build_e3_daily_briefing_multi(bq, user_id, t_probe)
    except Exception as exc:
        e3_instances = []
        print(f"[build_benchmark] WARN: e3_daily_briefing_multi builder failed: {exc}")

    # Task E4 — Google Search personalization (opt-in at run time; always built)
    try:
        from evaluation.tasks.e4_google_search import build_e4_google_search
        e4_instances = build_e4_google_search(bq, user_id, t_probe)
    except Exception as exc:
        e4_instances = []
        print(f"[build_benchmark] WARN: e4_google_search builder failed: {exc}")

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
        "blind_check_enabled": blind_check_llm is not None,
        "counts": {
            "test_items": len(test_items),
            "slate_ranking": len(slate_instances),
            "chatbot_response_proactive": len(b_arms["chatbot_response_proactive"]),
            "chatbot_response_control":   len(b_arms["chatbot_response_control"]),
            "c1a_pairs": len(c1a_pairs),
            "c1b_sequences": len(c1b_sequences),
            "c2_scenarios": len(c2_instances),
            "c3_restraint": len(c3_instances),
            "c4_button_regen": len(c4_instances),
            "e2_at_ai_followup": len(e2_instances),
            "e3_daily_briefing_multi": len(e3_instances),
            "e4_google_search": len(e4_instances),
            **{k: len(v) for k, v in agentic_buckets.items()},
        },
        "slate_ranking": slate_instances,
        "chatbot_response_proactive": b_arms["chatbot_response_proactive"],
        "chatbot_response_control":   b_arms["chatbot_response_control"],
        "c1a_pairs": c1a_pairs,
        "c1b_sequences": c1b_sequences,
        "c2_scenarios": c2_instances,
        "c3_restraint": c3_instances,
        "c4_button_regen": c4_instances,
        "e2_at_ai_followup": e2_instances,
        "e3_daily_briefing_multi": e3_instances,
        "e4_google_search": e4_instances,
        **agentic_buckets,
    }


def default_benchmark_path(user_id: str) -> Path:
    return Path("benchmark") / user_id / "benchmark.json"


def default_benchmark_csv_path(user_id: str) -> Path:
    return Path("benchmark") / user_id / "benchmark.csv"


def export_benchmark_csv(benchmark: dict, user_id: str, out_path: Path | None = None) -> Path:
    """Project a `benchmark.json` blob into a flat CSV for HuggingFace publication.

    Columns (stable, narrow):
      instance_id, task, user_id, t_test, t_test_iso, query, query_type,
      candidates_json, ground_truth_json, carveout_json, metadata_json

    Task-specific fields that don't fit the narrow schema are JSON-packed
    into `metadata_json`. The runner continues to consume the structured
    JSON; the CSV is a publication-friendly projection.
    """
    import csv
    import datetime as _dt

    out_path = Path(out_path) if out_path else default_benchmark_csv_path(user_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Iterate task buckets. A benchmark dict has two kinds of top-level keys:
    # metadata (counts, backend_hash, etc.) and task buckets (lists of instances).
    columns = [
        "instance_id", "task", "user_id", "t_test", "t_test_iso",
        "query", "query_type", "candidates_json", "ground_truth_json",
        "carveout_json", "metadata_json",
    ]

    def _isoformat_ts(ts) -> str:
        if not isinstance(ts, (int, float)) or ts <= 0:
            return ""
        try:
            return _dt.datetime.fromtimestamp(int(ts), tz=_dt.timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return ""

    def _project_row(task: str, inst: dict) -> dict | None:
        if not isinstance(inst, dict):
            return None
        t_test = inst.get("t_test") or inst.get("test_timestamp") or inst.get("source_timestamp") or 0
        query = inst.get("query") or inst.get("user_message") or ""
        query_type = inst.get("query_type") or inst.get("entry_point") or ""
        candidates = inst.get("candidates") or inst.get("slate") or []
        # Ground truth — task-specific; collect the commonly named ones
        gt = {
            k: inst[k] for k in (
                "held_out_preference", "held_out_indices", "positive_indices",
                "target_match", "target_ids", "forbidden_items", "matching_indices",
                "post_test_positives", "post_test_negatives", "post_test_engagements",
            )
            if k in inst
        }
        carveout = {
            k: inst[k] for k in ("carveout_indices", "carve_out_topic", "carveout_topic")
            if k in inst
        }
        # Everything else goes into metadata_json
        known_keys = {
            "instance_id", "task_id", "t_test", "test_timestamp", "source_timestamp",
            "query", "user_message", "query_type", "entry_point",
            "candidates", "slate",
        } | set(gt.keys()) | set(carveout.keys())
        metadata = {k: v for k, v in inst.items() if k not in known_keys}
        return {
            "instance_id": str(inst.get("instance_id") or ""),
            "task": task,
            "user_id": str(user_id),
            "t_test": int(t_test) if isinstance(t_test, (int, float)) else 0,
            "t_test_iso": _isoformat_ts(t_test),
            "query": str(query) if query is not None else "",
            "query_type": str(query_type) if query_type else "",
            "candidates_json": json.dumps(candidates, ensure_ascii=False) if candidates else "",
            "ground_truth_json": json.dumps(gt, ensure_ascii=False) if gt else "",
            "carveout_json": json.dumps(carveout, ensure_ascii=False) if carveout else "",
            "metadata_json": json.dumps(metadata, ensure_ascii=False) if metadata else "",
        }

    rows: list[dict] = []
    for task_key, bucket in benchmark.items():
        if not isinstance(bucket, list):
            continue  # skip metadata keys
        for inst in bucket:
            row = _project_row(task_key, inst)
            if row is not None:
                rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return out_path


def _make_blind_check_llm(model: str):
    """Return a callable `llm(prompt) -> str` that goes through the Claude Code
    subscription via `claude -p`. No API key, no QueryLLM — subscription-covered.
    """
    from evaluation.claude_subagent import run_subagent
    from pathlib import Path as _P

    # For the blind check we don't need a snapshot — the prompt carries everything.
    # But run_subagent wants a scope-dir; use a throwaway empty dir.
    scope = _P("/tmp/pm3_blind_check_scope")
    scope.mkdir(exist_ok=True)

    def _call(prompt: str) -> str:
        res = run_subagent(
            prompt=prompt,
            snapshot_dir=scope,
            model=model,
            allowed_tools=(),    # pure LLM, no tools
            timeout_seconds=60,
        )
        return res.text or ""
    return _call


def main():
    parser = argparse.ArgumentParser(description="Build a frozen eval benchmark for a user.")
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--backend_dir", default="backend")
    parser.add_argument("--rng_seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="Output path (default: benchmark/{user_id}/benchmark.json)")
    parser.add_argument("--skip_blind_check", action="store_true", help="Skip LLM blind-check for Task B curation (use default score 2)")
    parser.add_argument("--blind_check_limit", type=int, default=None, help="Cap how many candidate queries get blind-checked (for fast iteration)")
    parser.add_argument("--blind_check_model", default="haiku", help="Claude Code subagent model used for blind-check (default: haiku)")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else default_benchmark_path(args.user_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blind_llm = None if args.skip_blind_check else _make_blind_check_llm(args.blind_check_model)

    bm = build_benchmark(
        args.backend_dir,
        args.user_id,
        rng_seed=args.rng_seed,
        blind_check_llm=blind_llm,
        blind_check_limit=args.blind_check_limit,
    )
    with out_path.open("w") as f:
        json.dump(bm, f, ensure_ascii=False, indent=2)
    print(f"[build_benchmark] wrote {out_path}")
    print(f"[build_benchmark] counts: {bm['counts']}")
    print(f"[build_benchmark] backend_hash: {bm['backend_hash']}")
    print(f"[build_benchmark] blind_check_enabled: {bm['blind_check_enabled']}")

    # R9: also export a flat benchmark.csv projection for HuggingFace publication
    csv_path = export_benchmark_csv(bm, args.user_id)
    print(f"[build_benchmark] wrote {csv_path} (CSV projection for HF publication)")


if __name__ == "__main__":
    main()
