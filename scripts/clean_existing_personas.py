"""Drop topically-disjoint (canonical, event) pairings from already-emitted
backend/{user_id}/*.json — no pipeline regen, no Step-1 LLM replay.

Two-stage filter:
  Stage A — modal hashtag overlap (free): for each unique persona_item,
            collect the top-K most-frequent hashtags across every event
            it's attached to. Reject pairings whose event hashtags share
            zero with the canonical's modal top-K (excluding the current
            event's own contribution to the modal set so a single bogus
            attachment can't self-justify).
  Stage B — LLM judge (mini-tier, batched): re-check Stage-A failures
            with `pref_event_grounding_check_prompt`. Lenient toward
            name/genre relationships ("#kaicenat" for a comedy canonical),
            strict on clear semantic mismatches ("#smokedhog" for a candy
            canonical). Only Stage-B `grounded=false` pairings are dropped.

After dropping, recompute each canonical's `confidence_cross_referenced`
from the surviving evidence rows. Events that lose all preferences are
omitted unless they are `implicit_negative`.

Writes back in place to `backend/{user_id}/*.json` and refreshes
`profile.json` so its preference list + scores reflect the cleaned events.

Usage:
    python scripts/clean_existing_personas.py --user-id 115
                                              [--llm-model haiku]
                                              [--dry-run]
                                              [--no-llm]   # skip Stage B,
                                                           # drop all Stage-A flags
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation.prompts import pref_event_grounding_check_prompt
from data_preparation.utils import extract_json_from_response
from scripts.prepare_eval_data import _make_blind_check_llm

_APPS = ("instagram", "facebook", "threads", "chatbot")
_MODAL_TOP_K = 5
_LLM_BATCH = 20


def _normalize_tag(t: str) -> str:
    return (t or "").lower().lstrip("#").strip()


def _tag_set(event: dict) -> set[str]:
    return {_normalize_tag(h) for h in (event.get("source_hashtags") or [])
            if _normalize_tag(h)}


def _content_snippet(event: dict) -> str:
    c = event.get("content") or {}
    parts = [
        c.get("title") or "",
        c.get("caption") or "",
        c.get("overall_description") or "",
    ]
    return " ".join(p for p in parts if p)


def _modal_top_k(cohort_tagsets: list[set[str]],
                 exclude_idx: int | None,
                 top_k: int = _MODAL_TOP_K) -> set[str]:
    counter: Counter = Counter()
    for i, ts in enumerate(cohort_tagsets):
        if exclude_idx is not None and i == exclude_idx:
            continue
        for t in ts:
            counter[t] += 1
    return {t for t, _ in counter.most_common(top_k)}


def _interaction_weight(itype: str) -> float:
    """Same weighting used by the pipeline's cross_reference scorer:
    explicit rows count 1.0, implicit rows 0.5. Used to recompute
    `confidence_cross_referenced` after dropping bogus contributors.
    Base score is 1.0 (anchored); each surviving row adds its weight."""
    if itype.startswith("explicit"):
        return 1.0
    return 0.5


def _score_canonical(rows_seen: list[dict]) -> float:
    """Recompute confidence_cross_referenced. Each row contributes its
    interaction-type weight; floor is 1.0 (canonical exists)."""
    total = 1.0  # base
    for r in rows_seen:
        total += _interaction_weight(r.get("source_interaction_type", ""))
    return round(total, 2)


def _is_implicit_negative_event(event: dict) -> bool:
    return event.get("source_interaction_type", "") == "implicit_negative"


def _llm_grounding_judge(llm, batch: list[dict]) -> dict[int, tuple[bool, str]]:
    """Returns {pair_id: (grounded, reason)} for the input batch.
    On parse failure, defaults to grounded=True (preserve signal)."""
    if not llm or not batch:
        return {p["pair_id"]: (True, "no_llm") for p in batch}
    prompt = pref_event_grounding_check_prompt(batch)
    raw = llm(prompt)
    parsed = extract_json_from_response(raw)
    out: dict[int, tuple[bool, str]] = {}
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            pid = item.get("pair_id")
            grounded = bool(item.get("grounded", True))
            reason = str(item.get("reason", ""))[:200]
            if isinstance(pid, int):
                out[pid] = (grounded, reason)
    # Fill in missing pair_ids with grounded=True (lenient default).
    for p in batch:
        out.setdefault(p["pair_id"], (True, "missing_from_response"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", default="115")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--llm-model", default="haiku",
                        help="model for Stage-B grounding judge (default: haiku → gpt-5.4-mini)")
    parser.add_argument("--no-llm", action="store_true",
                        help="skip Stage B (drop all Stage-A flags)")
    parser.add_argument("--dry-run", action="store_true",
                        help="don't write changes; just report counts")
    parser.add_argument("--top-k", type=int, default=_MODAL_TOP_K)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    user_dir = repo_root / "backend" / args.user_id
    if not user_dir.exists():
        raise SystemExit(f"backend/{args.user_id} not found")

    # Load all 4 app JSONs.
    events_by_app: dict[str, list[dict]] = {}
    for app in _APPS:
        path = user_dir / f"{app}.json"
        if path.exists():
            with path.open() as f:
                events_by_app[app] = json.load(f)
    print(f"[{args.user_id}] loaded "
          f"{sum(len(v) for v in events_by_app.values())} events across "
          f"{len(events_by_app)} apps")

    # Build cohort index: persona_item → list[(app, event_idx, hashtag_set)].
    cohort: dict[str, list[tuple[str, int, set[str]]]] = defaultdict(list)
    for app, events in events_by_app.items():
        for idx, event in enumerate(events):
            tags = _tag_set(event)
            for pref in (event.get("preferences") or []):
                pi = pref.get("persona_item") or ""
                if pi:
                    cohort[pi].append((app, idx, tags))

    # Pass 1 — Stage A modal-overlap check.
    stage_a_pass: list[tuple[str, int, int]] = []   # (app, event_idx, pref_idx)
    stage_a_flag: list[tuple[str, int, int, str]] = []  # +persona_item
    n_pairs = 0
    for app, events in events_by_app.items():
        for idx, event in enumerate(events):
            event_tags = _tag_set(event)
            for p_idx, pref in enumerate(event.get("preferences") or []):
                pi = pref.get("persona_item") or ""
                if not pi:
                    continue
                n_pairs += 1
                cohort_entries = cohort[pi]
                tagsets = [ts for (_a, _i, ts) in cohort_entries]
                # Find this attachment's index in the cohort to exclude it.
                self_idx = next(
                    (j for j, (a, i, _ts) in enumerate(cohort_entries)
                     if a == app and i == idx),
                    None,
                )
                # Singletons (cohort size 1) skip Stage A entirely — there's
                # no cohort to compare against. Keep them.
                if len(tagsets) < 2:
                    stage_a_pass.append((app, idx, p_idx))
                    continue
                modal = _modal_top_k(tagsets, exclude_idx=self_idx, top_k=args.top_k)
                if event_tags & modal:
                    stage_a_pass.append((app, idx, p_idx))
                else:
                    stage_a_flag.append((app, idx, p_idx, pi))

    print(f"[{args.user_id}] Stage A: {len(stage_a_pass)} pass, "
          f"{len(stage_a_flag)} flagged for Stage B")

    # Pass 2 — Stage B LLM judge (skipped with --no-llm).
    stage_b_keep: set[tuple[str, int, int]] = set()
    stage_b_drop: list[tuple[str, int, int, str, str]] = []  # +reason

    if args.no_llm:
        print(f"[{args.user_id}] Stage B SKIPPED (--no-llm) — dropping all "
              f"{len(stage_a_flag)} Stage-A flags")
        for (app, idx, p_idx, pi) in stage_a_flag:
            stage_b_drop.append((app, idx, p_idx, pi, "stage_a_modal_overlap_zero"))
    else:
        llm = None if args.dry_run else _make_blind_check_llm(args.llm_model)
        # Build batched judge inputs.
        batches: list[list[dict]] = []
        flag_lookup: list[tuple[str, int, int, str]] = list(stage_a_flag)
        for i in range(0, len(flag_lookup), _LLM_BATCH):
            chunk = flag_lookup[i:i + _LLM_BATCH]
            batch = []
            for j, (app, idx, p_idx, pi) in enumerate(chunk):
                event = events_by_app[app][idx]
                batch.append({
                    "pair_id": j,
                    "persona_item": pi,
                    "event_hashtags": ["#" + t for t in sorted(_tag_set(event))],
                    "event_content": _content_snippet(event)[:500],
                })
            batches.append(batch)

        if args.dry_run:
            # Mock all-grounded so we can count Stage-A flag distribution.
            for (app, idx, p_idx, _pi) in stage_a_flag:
                stage_b_keep.add((app, idx, p_idx))
        else:
            for bi, batch in enumerate(batches):
                verdicts = _llm_grounding_judge(llm, batch)
                for j, (app, idx, p_idx, pi) in enumerate(
                        flag_lookup[bi * _LLM_BATCH: bi * _LLM_BATCH + len(batch)]):
                    grounded, reason = verdicts.get(j, (True, "missing"))
                    if grounded:
                        stage_b_keep.add((app, idx, p_idx))
                    else:
                        stage_b_drop.append((app, idx, p_idx, pi, reason))
                if (bi + 1) % 5 == 0 or bi + 1 == len(batches):
                    print(f"[{args.user_id}] Stage B batch {bi+1}/{len(batches)} done "
                          f"(running keep={len(stage_b_keep)}, drop={len(stage_b_drop)})")

    print(f"[{args.user_id}] Stage B: kept {len(stage_b_keep)}, "
          f"dropped {len(stage_b_drop)}")

    # Apply drops.
    drop_set: set[tuple[str, int, int]] = {(a, i, p) for (a, i, p, _pi, _r)
                                           in stage_b_drop}
    if not drop_set and not args.dry_run:
        print(f"[{args.user_id}] no pairs to drop — exiting clean")
        return 0

    # Build cleaned events per app: drop flagged preference indices, then
    # omit events whose preferences list is now empty (unless implicit_negative).
    cleaned: dict[str, list[dict]] = {}
    n_events_dropped = 0
    n_prefs_dropped = 0
    for app, events in events_by_app.items():
        out_events: list[dict] = []
        for idx, event in enumerate(events):
            new_event = deepcopy(event)
            new_prefs: list[dict] = []
            for p_idx, pref in enumerate(event.get("preferences") or []):
                if (app, idx, p_idx) in drop_set:
                    n_prefs_dropped += 1
                    continue
                new_prefs.append(pref)
            new_event["preferences"] = new_prefs
            if not new_prefs and not _is_implicit_negative_event(new_event):
                n_events_dropped += 1
                continue
            out_events.append(new_event)
        cleaned[app] = out_events

    # Recompute confidence_cross_referenced per persona_item from
    # surviving evidence rows.
    canonical_rows: dict[str, list[dict]] = defaultdict(list)
    for app, events in cleaned.items():
        for event in events:
            for pref in (event.get("preferences") or []):
                pi = pref.get("persona_item") or ""
                if pi:
                    canonical_rows[pi].append({
                        "source_interaction_type": event.get("source_interaction_type", ""),
                        "app": app,
                    })
    new_scores: dict[str, float] = {pi: _score_canonical(rows)
                                    for pi, rows in canonical_rows.items()}

    # Patch each surviving preference's confidence_cross_referenced.
    for app, events in cleaned.items():
        for event in events:
            for pref in (event.get("preferences") or []):
                pi = pref.get("persona_item") or ""
                if pi in new_scores:
                    pref["confidence_cross_referenced"] = new_scores[pi]

    # Refresh profile.json's preference list + scores. Only keep entries
    # whose canonicals still survive after cleanup.
    profile_path = user_dir / "profile.json"
    profile_changed = False
    if profile_path.exists():
        with profile_path.open() as f:
            profile = json.load(f)
        # The schema typically holds a flat unique list under one of these
        # keys; handle a few common shapes defensively.
        for key in ("preferences", "personas", "unique_preferences"):
            arr = profile.get(key)
            if not isinstance(arr, list):
                continue
            new_arr: list[dict] = []
            for item in arr:
                if not isinstance(item, dict):
                    new_arr.append(item)
                    continue
                pi = item.get("persona_item") or ""
                if pi in new_scores:
                    item = dict(item)
                    item["confidence_cross_referenced"] = new_scores[pi]
                    new_arr.append(item)
                # else: drop — canonical no longer survives
            if len(new_arr) != len(arr):
                profile[key] = new_arr
                profile_changed = True

    print(f"[{args.user_id}] cleanup summary:")
    print(f"  pairs total                   : {n_pairs}")
    print(f"  pairs dropped                 : {n_prefs_dropped}")
    print(f"  events dropped (no surviving) : {n_events_dropped}")
    print(f"  unique canonicals after       : {len(new_scores)}")
    if not new_scores:
        print(f"  WARNING: cleanup left zero canonicals — aborting write")
        return 1

    if args.dry_run:
        print()
        print("(dry-run — no files written)")
        return 0

    # Write back.
    for app, events in cleaned.items():
        path = user_dir / f"{app}.json"
        with path.open("w") as f:
            json.dump(events, f, indent=2, ensure_ascii=False)
        print(f"  wrote {path} ({len(events)} events)")
    if profile_changed:
        with profile_path.open("w") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        print(f"  wrote {profile_path}")

    # Drop log for traceability.
    drop_log = Path(f"/tmp/persona_{args.user_id}_drops.tsv")
    with drop_log.open("w") as f:
        f.write("app\tevent_idx\tpref_idx\tpersona_item\treason\n")
        for (app, idx, p_idx, pi, reason) in stage_b_drop:
            f.write(f"{app}\t{idx}\t{p_idx}\t{pi}\t{reason}\n")
    print(f"  wrote {drop_log}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
