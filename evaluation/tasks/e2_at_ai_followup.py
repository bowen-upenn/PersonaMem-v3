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

# Hashtag Jaccard threshold for "matches directive". 0.05 is permissive
# enough to surface the high-overlap tail for users with broad / non-
# overlapping hashtag clusters (e.g. user 115's max-observed Jaccard
# between an @ai directive's tag set and a candidate event is ~0.04),
# while still excluding rows that share zero hashtags.
_E2_MATCH_THRESHOLD: float = 0.05
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


def _soft_tag_match(t1: str, t2: str, min_len: int = 5) -> bool:
    """Exact match, OR one tag is a substring of the other with the shorter
    tag >= min_len chars. So '#wildlife' matches '#wildlifephotography' (a real
    on-directive candidate that exact Jaccard mislabels 'filler'), while short
    tags like '#cat' do NOT spuriously match '#category' (audit 2026-07-20, T2-7)."""
    if t1 == t2:
        return True
    short, long = (t1, t2) if len(t1) <= len(t2) else (t2, t1)
    return len(short) >= min_len and short in long


def _soft_overlap_frac(ev_set: set, ai_set: set) -> float:
    """Lexical-variant-tolerant Jaccard: an ev tag overlaps if it soft-matches
    ANY ai tag; denominator stays the strict union so exact-match semantics are
    unchanged and the existing _E2_MATCH_THRESHOLD still applies."""
    if not ev_set or not ai_set:
        return 0.0
    matched = sum(1 for e in ev_set if any(_soft_tag_match(e, a) for a in ai_set))
    return matched / len(ev_set | ai_set)


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
        # Real engagement timestamp — the visualizer renders a `±Xd`
        # delta vs t_test so a reviewer can see how recent each candidate is.
        "source_timestamp": int(event.get("source_timestamp") or 0),
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
    instance per directive whose candidate pool draws from the user's
    ENTIRE @ai-mention history (Option B):

      - 1 target  : event matching ANY past positive @ai mention
                    (recommend_more / focus_topic), but NOT any negative
                    @ai mention. The agent should rank this top-1.
      - N carve-outs : events matching ANY past negative @ai mention
                    (feels_off / stop_recommending / not_interested),
                    but NOT any positive @ai mention. Must rank LAST.
      - fillers   : events with no overlap to either polarity.

    The instance still names ONE specific directive (the one the prompt
    surfaces as "the latest @ai comment"). The pool tests cross-directive
    memory: does the agent honor every @ai signal in history, not just
    the most recent one?
    """
    import random as _random

    # Step 1: collect all events across social apps with their app tag
    all_social: list[tuple[str, dict]] = []  # (app, event)
    for app in _SOCIAL_APPS:
        for ev in _load_raw_events(bq, user_id, app):
            if isinstance(ev, dict):
                all_social.append((app, ev))
    all_social.sort(key=lambda ae: ae[1].get("source_timestamp", 0))

    # Step 2: partition @ai mentions by polarity and build hashtag sets
    pos_directives: list[tuple[str, dict]] = []
    neg_directives: list[tuple[str, dict]] = []
    for app, ev in all_social:
        action = (ev.get("interaction_format") or {}).get("action", "")
        if action in _ACTION_POSITIVE_WANTS_MATCH:
            pos_directives.append((app, ev))
        elif action in _ACTION_POSITIVE_WANTS_NON_MATCH:
            neg_directives.append((app, ev))
    if not pos_directives and not neg_directives:
        return []

    pos_ai_hashtags: set[str] = set()
    for _, ev in pos_directives:
        for h in (ev.get("source_hashtags") or []):
            if h:
                pos_ai_hashtags.add(h.lstrip("#").lower())
    neg_ai_hashtags: set[str] = set()
    for _, ev in neg_directives:
        for h in (ev.get("source_hashtags") or []):
            if h:
                neg_ai_hashtags.add(h.lstrip("#").lower())

    # Step 3: pre-classify every non-@ai event into target / carveout / filler.
    # Use the existing _E2_MATCH_THRESHOLD Jaccard floor so a candidate only
    # qualifies on meaningful overlap. Events that match BOTH polarities
    # (ambiguous) are dropped — they'd test ranking under contradiction.
    target_pool: list[tuple[str, dict]] = []
    carveout_pool: list[tuple[str, dict]] = []
    filler_pool: list[tuple[str, dict]] = []
    _ai_actions = _ACTION_POSITIVE_WANTS_MATCH | _ACTION_POSITIVE_WANTS_NON_MATCH
    for app, ev in all_social:
        if (ev.get("interaction_format") or {}).get("action", "") in _ai_actions:
            continue  # don't show @ai events themselves as candidates
        ev_tags = [h for h in (ev.get("source_hashtags") or []) if h]
        if not ev_tags:
            filler_pool.append((app, ev))
            continue
        ev_set = {h.lstrip("#").lower() for h in ev_tags}
        match_pos = bool(pos_ai_hashtags) and (
            _soft_overlap_frac(ev_set, pos_ai_hashtags) >= _E2_MATCH_THRESHOLD
        )
        match_neg = bool(neg_ai_hashtags) and (
            _soft_overlap_frac(ev_set, neg_ai_hashtags) >= _E2_MATCH_THRESHOLD
        )
        if match_pos and not match_neg:
            target_pool.append((app, ev))
        elif match_neg and not match_pos:
            carveout_pool.append((app, ev))
        elif not match_pos and not match_neg:
            filler_pool.append((app, ev))
        # both → skip

    # Without a target, the test has nothing for the agent to rank top-1.
    if not target_pool:
        return []

    instances: list[dict] = []
    directive_events = pos_directives + neg_directives
    # Carve-out budget: aim for ~3 per pool when available; allow 0 if user
    # has no negative @ai history.
    n_carveouts_target = 3

    for app, dev in directive_events:
        t_ai = int(dev.get("source_timestamp") or 0)
        if t_ai <= 0:
            continue
        directive_action = (dev.get("interaction_format") or {}).get("action", "")
        directive_hashtags = list(dev.get("source_hashtags") or [])
        if not directive_hashtags:
            continue
        directive_oid = dev.get("source_object_id")

        for lag_label, lag_seconds in _E2_FOLLOWUP_LAGS_SECONDS:
            t_test = t_ai + lag_seconds
            rng = _random.Random(f"{rng_seed}:e2:{directive_oid}:{lag_label}")

            # Candidates must be content the agent has NOT already seen at
            # directive time — source each lag's pool from the FORWARD window
            # (t_test, t_test + lookahead]. This (a) excludes pre-directive
            # events (which can't test whether the agent REMEMBERED the directive)
            # and (b) gives each lag a distinct window instead of every lag
            # sharing one timeline-wide pool. Previously `_E2_LOOKAHEAD_HOURS` was
            # defined but never applied — the headline confounded recall with
            # raw recency and let pre-directive events be the ranking target.
            _win_hi = t_test + _E2_LOOKAHEAD_HOURS * 3600

            def _in_window(pool):
                return [(a, e) for (a, e) in pool
                        if t_test < int(e.get("source_timestamp") or 0) <= _win_hi]

            tgt = _in_window(target_pool)
            cvs = _in_window(carveout_pool)
            fls = _in_window(filler_pool)
            rng.shuffle(tgt)
            rng.shuffle(cvs)
            rng.shuffle(fls)

            if not tgt:
                continue  # no directive-matching content in this lag's window
            target_pick = tgt[0]
            n_cv = min(n_carveouts_target, len(cvs))
            cv_pick = cvs[:n_cv]
            n_fl = _E2_TARGET_POOL - 1 - n_cv
            fl_pick = fls[:max(0, n_fl)]

            cand_events = [target_pick] + list(cv_pick) + list(fl_pick)
            if len(cand_events) < _E2_MIN_POOL:
                continue

            # Shuffle so target/carve-outs aren't always at fixed positions.
            order = list(range(len(cand_events)))
            rng.shuffle(order)

            candidates: list[dict] = []
            target_idx = -1
            carveout_idxs: list[int] = []
            for new_pos, old_idx in enumerate(order):
                _a, ev = cand_events[old_idx]
                candidates.append(_strip_candidate(ev))
                if old_idx == 0:
                    target_idx = new_pos
                elif 1 <= old_idx <= n_cv:
                    carveout_idxs.append(new_pos)

            if target_idx < 0:
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
                "positive_indices": [target_idx],
                "carveout_indices": sorted(carveout_idxs),
                # Cross-directive context for the visualizer + judge: the
                # union of hashtags across all positive/negative @ai
                # mentions in the user's history. Drives the per-candidate
                # rationale and lets the judge reason about the FULL @ai
                # signal, not just this directive's slice.
                "positive_directive_hashtags": sorted(pos_ai_hashtags),
                "negative_directive_hashtags": sorted(neg_ai_hashtags),
            })

    return instances


def compute_e2_metrics(ranked: list[int], instance: dict) -> dict:
    from evaluation.tasks.personalized_recommendation import _graded_ndcg_at_k
    positives = set(instance.get("positive_indices") or [])
    carveouts = set(instance.get("carveout_indices") or [])
    k = len(instance.get("candidates") or [])
    top1 = ranked[0] if ranked else -1
    top3 = set(ranked[:3]) if ranked else set()
    top5 = set(ranked[:5]) if ranked else set()
    # Negative-directive compliance: carveout items (topics the user
    # explicitly said to stop recommending) must ALL rank BELOW every
    # positive item. If any carveout appears before the last positive
    # in the agent's ranking, the agent failed to respect the directive.
    carveout_before_positives = 0
    if carveouts and positives and ranked:
        rank_of = {idx: rank for rank, idx in enumerate(ranked)}
        last_pos_rank = max(
            (rank_of.get(p, len(ranked)) for p in positives),
            default=len(ranked),
        )
        first_carveout_rank = min(
            (rank_of.get(c, len(ranked)) for c in carveouts),
            default=len(ranked),
        )
        carveout_before_positives = int(first_carveout_rank < last_pos_rank)

    out = {
        "hit@1": int(top1 in positives),
        "recall@3": len(positives & top3) / max(len(positives), 1) if positives else 0.0,
        "recall@5": len(positives & top5) / max(len(positives), 1) if positives else 0.0,
        "mrr": metrics.mrr(ranked, positives) if positives else 0.0,
        "directive_respect@1": int(top1 in positives),
        "carveout_violation@1": int(top1 in carveouts),
        "carveout_violation@3": int(bool(top3 & carveouts)),
        "carveout_violation@5": int(bool(top5 & carveouts)),
        "carveout_before_all_positives": carveout_before_positives,
        # Graded NDCG — shared headline with the other two ranking tasks
        # (personalized_recommendation, hidden_persona_recommendation).
        # directive-matching positives = +2, carve-outs (must-avoid) = hard
        # negatives = -2, everything else = +1: rewards surfacing the directive's
        # items up top AND burying the carve-outs.
        "ndcg_at_3": round(_graded_ndcg_at_k(ranked, positives, carveouts, 3), 4),
        "ndcg_at_5": round(_graded_ndcg_at_k(ranked, positives, carveouts, 5), 4),
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
        if mode in ("llm_longctx", "llm_memory", "mem0"):
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
            "subagent_stats": subagent_stats,
            "history_tokens": history_tokens,
            "tool_call_count": tool_call_count,
        })

    return results
