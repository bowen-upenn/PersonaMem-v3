"""Task E2 — proactive recommendation after an @ai directive.

The user posted `@ai recommend_more X`, `@ai stop_recommending Y`,
`@ai focus_topic Z`, `@ai not_interested`, or `@ai feels_off` in an
Instagram/Facebook/Threads event's comment. Cut the timeline at that
event and ask the agent to rank the upcoming 10-15 candidate feed items.
The top-1 must align with the directive (recommend_more → matching;
stop_recommending/not_interested → NOT matching, carve-out check).

Builder (build_e2_at_ai_followup) produces frozen instances; runner
(run_e2_at_ai_followup) mirrors the slate_ranking shape — the agent
receives candidates without preferences or labels, just raw content.
"""

from __future__ import annotations

import json
from pathlib import Path

from data_preparation.utils import extract_json_from_response
from evaluation import metrics, prompts
from evaluation.backend_query import _LEAK_FIELDS_EVENT, _LEAK_FIELDS_PREF, BackendQuery

# Hashtag Jaccard threshold for "matches directive". Tightened from 0.25 so a
# candidate only counts as a "directive match" on meaningful hashtag overlap,
# not loose surface match (e.g. one shared generic tag).
_E2_MATCH_THRESHOLD: float = 0.15
# Minimum candidate-pool size; instance is dropped if fewer than this. Raised
# from 6 to enforce a uniform floor across all lag buckets.
_E2_MIN_POOL: int = 12
# Post-T_test lookahead window (hours) for candidate content
_E2_LOOKAHEAD_HOURS: int = 72
# Target pool size (cap)
_E2_TARGET_POOL: int = 12
# Stratified follow-up lags: each directive yields one instance per lag.
# 24h / 72h / 7d expose a recall-vs-lag curve so we can tell memory recall
# apart from raw recency. The previous behavior (t_test = t_ai + 1s) made
# the task trivially solvable as a "list the next post that matches the
# directive's hashtags" exercise.
_E2_FOLLOWUP_LAGS_SECONDS: tuple[tuple[str, int], ...] = (
    ("24h", 24 * 3600),
    ("72h", 72 * 3600),
    ("7d", 7 * 24 * 3600),
)
# Number of hard distractors per pool — events the user engaged with on
# *adjacent* but non-matching hashtag clusters, so loose Jaccard alone
# can't pick the target.
_E2_HARD_DISTRACTORS: int = 2
# Minimum overlap (Jaccard) for a hashtag set to count as "adjacent" to
# the directive — i.e. shares some context but doesn't match outright.
_E2_HARD_DISTRACTOR_MIN_JACCARD: float = 0.05

_SOCIAL_APPS: tuple[str, ...] = ("instagram", "facebook", "threads")

# Actions whose candidate pool must MATCH the directive hashtags (recommend more)
_ACTION_POSITIVE_WANTS_MATCH: set[str] = {
    "at_ai_recommend_more",
    "at_ai_focus_topic",
}
# Actions whose candidate pool must AVOID the directive hashtags (stop recommending)
_ACTION_POSITIVE_WANTS_NON_MATCH: set[str] = {
    "at_ai_stop_recommending",
    "at_ai_not_interested",
    "at_ai_feels_off",
}


def _hashtag_jaccard(a: list[str], b: list[str]) -> float:
    sa = {h.lstrip("#").lower() for h in (a or []) if h}
    sb = {h.lstrip("#").lower() for h in (b or []) if h}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _strip_candidate(event: dict) -> dict:
    """Project a raw event into a candidate item: NO preferences, NO labels.

    Mirrors `backend_query._strip_event` + extra pruning to keep the
    candidate narrow (title/caption/hashtags/content_type/content). All
    harness-internal fields are dropped.
    """
    # The candidate is PURE content from the agent's perspective
    content = event.get("content") or {}
    hashtags = event.get("source_hashtags") or []
    content_type = event.get("content_type") or content.get("content_type") or "text"
    item = {
        "content_type": content_type,
        "hashtags": list(hashtags),
    }
    for key in ("title", "caption", "overall_description"):
        val = content.get(key)
        if val:
            item[key] = val
    # Keep `is_ad` so the agent can see it's a sponsored post (but not the
    # sponsor_name — that's not needed for ranking)
    if event.get("is_ad"):
        item["is_sponsored"] = True
    return item


def _load_raw_events(bq: BackendQuery, user_id: str, app: str) -> list[dict]:
    """Bypass BackendQuery's strip_event to access raw action + hashtags."""
    path = bq.base / user_id / f"{app}.json"
    if not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def build_e2_at_ai_followup(
    bq: BackendQuery,
    user_id: str,
    rng_seed: int = 0,
) -> list[dict]:
    """Walk social-app events, collect @ai directive events, build one
    instance per directive with a post-T_test candidate pool.
    """
    import random as _random

    # Step 1: collect all events across social apps with their app tag
    all_social: list[tuple[str, dict]] = []  # (app, event)
    for app in _SOCIAL_APPS:
        for ev in _load_raw_events(bq, user_id, app):
            if isinstance(ev, dict):
                all_social.append((app, ev))
    all_social.sort(key=lambda ae: ae[1].get("source_timestamp", 0))

    # Step 2: find events with @ai actions
    directive_events: list[tuple[str, dict]] = []
    for app, ev in all_social:
        fmt = ev.get("interaction_format") or {}
        action = fmt.get("action", "")
        if action in _ACTION_POSITIVE_WANTS_MATCH or action in _ACTION_POSITIVE_WANTS_NON_MATCH:
            directive_events.append((app, ev))

    if not directive_events:
        return []

    instances: list[dict] = []
    lookahead_sec = _E2_LOOKAHEAD_HOURS * 3600

    for app, dev in directive_events:
        t_ai = int(dev.get("source_timestamp") or 0)
        if t_ai <= 0:
            continue
        directive_action = (dev.get("interaction_format") or {}).get("action", "")
        directive_hashtags = list(dev.get("source_hashtags") or [])
        if not directive_hashtags:
            continue
        directive_oid = dev.get("source_object_id")

        # Pre-build a hard-distractor pool: events anywhere in the timeline
        # whose hashtags overlap the directive's hashtags by a non-trivial
        # but sub-threshold amount. These are "adjacent" content the user
        # engages with — Jaccard alone can't tell them apart from the
        # target, so the agent has to actually reason about the directive.
        hard_pool: list[tuple[str, dict, float]] = []
        for a, ev in all_social:
            if ev.get("source_object_id") == directive_oid:
                continue
            j = _hashtag_jaccard(ev.get("source_hashtags") or [], directive_hashtags)
            if _E2_HARD_DISTRACTOR_MIN_JACCARD <= j < _E2_MATCH_THRESHOLD:
                hard_pool.append((a, ev, j))

        # Stratified-lag emission: one instance per (24h / 72h / 7d) lag.
        for lag_label, lag_seconds in _E2_FOLLOWUP_LAGS_SECONDS:
            t_test = t_ai + lag_seconds
            window_lo = t_test
            window_hi = t_test + lookahead_sec

            # Step 3: gather candidate events in (t_test, t_test + 72h]
            cand_events: list[tuple[str, dict]] = []
            for a, ev in all_social:
                ts = int(ev.get("source_timestamp") or 0)
                if ts <= window_lo or ts > window_hi:
                    continue
                if ev.get("source_object_id") == directive_oid:
                    continue
                cand_events.append((a, ev))

            # Deterministic per-(directive, lag) shuffle
            rng = _random.Random(
                f"{rng_seed}:e2:{directive_oid}:{lag_label}"
            )
            rng.shuffle(cand_events)
            # Reserve room for hard distractors so the final pool size
            # doesn't blow past the cap.
            soft_slots = max(0, _E2_TARGET_POOL - _E2_HARD_DISTRACTORS)
            cand_events = cand_events[:soft_slots]

            # Append hard distractors (adjacent-Jaccard items pulled from
            # the user's broader timeline). Skip ones already in the
            # candidate set by source_object_id.
            existing_oids = {ev.get("source_object_id") for _, ev in cand_events}
            hard_shuffled = list(hard_pool)
            rng.shuffle(hard_shuffled)
            hard_added = 0
            for a, ev, _j in hard_shuffled:
                if hard_added >= _E2_HARD_DISTRACTORS:
                    break
                if ev.get("source_object_id") in existing_oids:
                    continue
                cand_events.append((a, ev))
                existing_oids.add(ev.get("source_object_id"))
                hard_added += 1

            if len(cand_events) < _E2_MIN_POOL:
                continue

            # Re-shuffle so hard distractors aren't always at the tail.
            rng.shuffle(cand_events)
            cand_events = cand_events[:_E2_TARGET_POOL]

            # Step 4: label each candidate by hashtag Jaccard vs directive
            candidates: list[dict] = []
            match_flags: list[bool] = []
            for a, ev in cand_events:
                item = _strip_candidate(ev)
                candidates.append(item)
                match_flags.append(
                    _hashtag_jaccard(ev.get("source_hashtags") or [], directive_hashtags)
                    >= _E2_MATCH_THRESHOLD
                )

            if directive_action in _ACTION_POSITIVE_WANTS_MATCH:
                positive_indices = [i for i, m in enumerate(match_flags) if m]
                carveout_indices: list[int] = []
            else:
                # recommend-AWAY: positives are non-matching; matching candidates
                # are carve-outs (hard fail if any appears in top-1)
                positive_indices = [i for i, m in enumerate(match_flags) if not m]
                carveout_indices = [i for i, m in enumerate(match_flags) if m]

            if not positive_indices:
                continue

            instances.append({
                "instance_id": f"e2_{directive_oid or 'unk'}_{lag_label}",
                "task_id": "e2_at_ai_followup",
                "user_id": str(user_id),
                "t_test": t_test,
                "source_timestamp": t_test,
                "lag_bucket": lag_label,
                "lag_seconds": lag_seconds,
                "directive_app": app,
                "directive_action": directive_action,
                "directive_hashtags": directive_hashtags,
                "directive_user_message": (dev.get("interaction_format") or {}).get("user_message") or "",
                "candidates": candidates,
                "positive_indices": positive_indices,
                "carveout_indices": carveout_indices,
            })

    return instances


def compute_e2_metrics(ranked: list[int], instance: dict) -> dict:
    positives = set(instance.get("positive_indices") or [])
    carveouts = set(instance.get("carveout_indices") or [])
    k = len(instance.get("candidates") or [])
    top1 = ranked[0] if ranked else -1
    top3 = set(ranked[:3]) if ranked else set()
    top5 = set(ranked[:5]) if ranked else set()
    out = {
        "hit@1": int(top1 in positives),
        "recall@3": len(positives & top3) / max(len(positives), 1) if positives else 0.0,
        "recall@5": len(positives & top5) / max(len(positives), 1) if positives else 0.0,
        "mrr": metrics.mrr(ranked, positives) if positives else 0.0,
        "directive_respect@1": int(top1 in positives),
        "carveout_violation@1": int(top1 in carveouts),
        "carveout_violation@3": int(bool(top3 & carveouts)),
    }
    lag = instance.get("lag_bucket")
    if lag:
        out["lag_bucket"] = lag
    return out


def run_e2_at_ai_followup(
    instances,
    user_id,
    bq: BackendQuery,
    llm_client,
    judge_client,
    mode: str,
    snapshot_cache,
    model_name: str | None,
    claude_model: str,
    context_budget: int | None,
    enable_llm_judge: bool,
    dry_run: bool,
    limit: int | None = None,
) -> list[dict]:
    """Runner for E2. Mirrors run_task_a shape."""
    from evaluation.inference_utils import dispatch_agent_run

    if limit is not None:
        instances = instances[:limit]

    results: list[dict] = []
    for inst in instances:
        t = inst["t_test"]
        candidates = inst.get("candidates") or []
        history_block = None
        history_tokens = 0
        if mode in ("agent_longctx", "llm_longctx"):
            history_block, stats = snapshot_cache.get_or_build(bq, user_id, t, model_name, context_budget)
            history_tokens = stats["total_tokens"]

        prompt = prompts.e2_at_ai_followup_prompt(
            directive_action=inst["directive_action"],
            directive_hashtags=inst["directive_hashtags"],
            directive_user_message=inst.get("directive_user_message", ""),
            candidates=candidates,
            history_block=history_block,
        )

        if dry_run:
            results.append({
                "task": "e2_at_ai_followup",
                "user_id": user_id,
                "instance_id": inst["instance_id"],
                "mode": mode,
                "history_tokens": history_tokens,
                "metrics": None,
            })
            continue

        raw_response, tool_call_count, subagent_stats = dispatch_agent_run(
            mode, prompt, bq=bq, user_id=user_id, t=t,
            claude_model=claude_model, llm_client=llm_client,
        )
        parsed = extract_json_from_response(raw_response) or {}
        ranked = parsed.get("ranked_indices") or []
        if not isinstance(ranked, list) or sorted(set(ranked)) != list(range(len(candidates))):
            ranked = list(range(len(candidates)))

        scored = compute_e2_metrics(ranked, inst)

        # Phase L.B.1: directive intent-alignment judge.
        # Hashtag-Jaccard alone (positive_indices/carveout_indices) doesn't
        # capture the user's actual ask — e.g. "@ai recommend more posts about
        # hiking with my dog" matches #hiking but should not surface solo
        # hiking. The judge reads the full directive_user_message + the
        # agent's top-3 picks and scores semantic intent alignment.
        # `directive_score` is the headline; defaults to hit@1 when judge is
        # disabled or fails so the aggregator always finds the key.
        scored["directive_score"] = float(scored.get("hit@1", 0))
        if enable_llm_judge and judge_client and inst.get("directive_user_message"):
            from evaluation import judges as _judges
            top3_objs = [candidates[i] for i in ranked[:3] if 0 <= i < len(candidates)]
            j = _judges.judge_at_ai_directive(
                judge_client,
                inst["directive_user_message"],
                inst["directive_action"],
                top3_objs,
            )
            if j.get("intent_alignment_score") is not None:
                scored["intent_alignment_score"] = j["intent_alignment_score"]
                scored["directive_score"] = 0.5 * float(scored.get("hit@1", 0)) + 0.5 * j["intent_alignment_score"]
            scored["judge_reasoning_at_ai"] = j.get("judge_reasoning") or ""

        results.append({
            "task": "e2_at_ai_followup",
            "user_id": user_id,
            "instance_id": inst["instance_id"],
            "mode": mode,
            "ranked_indices": ranked,
            "directive_action": inst["directive_action"],
            "metrics": scored,
            "agent_response": raw_response,
            "history_tokens": history_tokens,
            "tool_call_count": tool_call_count,
        })

    return results
