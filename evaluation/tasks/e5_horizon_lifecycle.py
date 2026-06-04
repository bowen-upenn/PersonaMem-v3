"""Task E5 — short-term horizon lifecycle.

For each surviving short-term canonical with an `expected_stop_ts`,
build TWO instances: one `pre` probe at the intent's mid-window (active
phase) and one `post` probe after the stop timestamp (expired phase).
The agent is asked to rank candidates; metrics compare top-K matching
rate before vs. after expiry.

lifecycle_score = pre.match_rate_at_3 − post.persistence_rate_at_3
  +1 = perfect horizon compliance (recommends during active window,
        drops recommendation after expiry)
  −1 = inverted (recommends AFTER expiry but not during)

Geo + calendar context from Phase 4 is injected into the post-probe
prompt so the agent has some reason to believe the intent has ended
(user returned home, calendar event passed, etc.) even when the
post-T_test moment is past the observed window.
"""

from __future__ import annotations

import json

from data_preparation.utils import extract_json_from_response
from evaluation import metrics
from evaluation.backend_query import BackendQuery


E5_MIN_MATCHING_CANDIDATES: int = 2
E5_POOL_TARGET: int = 10


def _jaccard(a: list[str], b: list[str]) -> float:
    sa = {h.lstrip("#").lower() for h in (a or []) if h}
    sb = {h.lstrip("#").lower() for h in (b or []) if h}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _collect_short_term_canonicals(bq: BackendQuery, user_id: str) -> list[dict]:
    """Harvest one record per short-term canonical with a non-null
    `expected_stop_ts`, merging data across all four apps.
    """
    from collections import defaultdict as _dd
    records: dict[str, dict] = _dd(lambda: {
        "persona_item": None,
        "category": None,
        "stop_condition": None,
        "row_timestamps": [],
        "row_hashtags": [],
        "event_location_cities": [],
    })
    for app in ("instagram", "facebook", "threads", "chatbot"):
        for e in bq._load_events(user_id, app):  # unstripped
            ts = e.get("source_timestamp") or 0
            hashtags = e.get("source_hashtags") or []
            loc = e.get("event_location") or {}
            city = loc.get("city") if isinstance(loc, dict) else None
            for p in e.get("preferences", []):
                if not isinstance(p, dict):
                    continue
                if p.get("time_horizon") != "short_term":
                    continue
                sc = p.get("stop_condition") or {}
                if not isinstance(sc, dict) or not sc.get("expected_stop_ts"):
                    continue
                pi = p.get("persona_item") or ""
                if not pi:
                    continue
                rec = records[pi]
                rec["persona_item"] = pi
                rec["category"] = p.get("category", "")
                rec["stop_condition"] = sc
                if ts:
                    rec["row_timestamps"].append(ts)
                for h in hashtags:
                    rec["row_hashtags"].append(h)
                if city:
                    rec["event_location_cities"].append(city)

    out: list[dict] = []
    for pi, rec in records.items():
        if len(rec["row_timestamps"]) < 2:
            continue
        # De-dup hashtags preserving order
        rec["row_hashtags"] = list(dict.fromkeys(rec["row_hashtags"]))
        rec["row_timestamps"].sort()
        out.append(rec)
    return out


def _gather_candidate_events(
    bq: BackendQuery, user_id: str, window_start: int, window_end: int
) -> list[dict]:
    """Collect all events in [window_start, window_end] across social apps."""
    pool: list[dict] = []
    for app in ("instagram", "facebook", "threads"):
        for e in bq._load_events(user_id, app):
            ts = e.get("source_timestamp") or 0
            if window_start <= ts <= window_end:
                pool.append(e)
    return pool


def _project_candidate(e: dict) -> dict:
    """Strip leaks; keep only raw content (mirrors e2 stripping)."""
    content = e.get("content") or {}
    item = {
        "content_type": e.get("content_type") or content.get("content_type") or "text",
        "hashtags": list(e.get("source_hashtags") or []),
    }
    for key in ("title", "caption", "overall_description"):
        val = content.get(key)
        if val:
            item[key] = val
    if e.get("is_ad"):
        item["is_sponsored"] = True
    return item


def build_e5_horizon_lifecycle(
    bq: BackendQuery,
    user_id: str,
    rng_seed: int = 0,
) -> list[dict]:
    """Emit (pre, post) paired instances per qualifying short-term canonical."""
    import random as _random

    canonicals = _collect_short_term_canonicals(bq, user_id)
    if not canonicals:
        return []

    # Max observed timestamp across user's events; used to bound post-probe
    max_ts = 0
    for app in ("instagram", "facebook", "threads", "chatbot"):
        for e in bq._load_events(user_id, app):
            max_ts = max(max_ts, e.get("source_timestamp") or 0)

    instances: list[dict] = []
    for rec in canonicals:
        row_tss = rec["row_timestamps"]
        first_ts, last_ts = row_tss[0], row_tss[-1]
        sc = rec["stop_condition"] or {}
        expected_stop_ts = int(sc.get("expected_stop_ts") or 0)
        if expected_stop_ts <= first_ts:
            continue
        t_active = int(first_ts + 0.6 * (last_ts - first_ts))
        if t_active >= expected_stop_ts:
            continue
        # Post-probe: clamp to max_ts+1h if expected_stop_ts is past the window
        t_post = max(max_ts + 3600, expected_stop_ts + 2 * 3600)

        # Shared candidate pool around the active window
        directive_hashtags = rec["row_hashtags"]
        if not directive_hashtags:
            continue
        # PRE pool: within active window ± 48h
        pre_events = _gather_candidate_events(bq, user_id,
                                              window_start=max(0, t_active - 48 * 3600),
                                              window_end=t_active + 24 * 3600)
        # POST pool: around post-probe (or clamped to window)
        post_events = _gather_candidate_events(bq, user_id,
                                               window_start=max(0, t_post - 48 * 3600),
                                               window_end=min(max_ts, t_post + 24 * 3600))
        # If the post pool is tiny (post-probe past window), fall back to pre pool
        if len(post_events) < 3:
            post_events = pre_events

        rng = _random.Random(f"{rng_seed}:e5:{rec['persona_item'][:40]}")

        def _build_phase(events_list: list[dict], t_test: int, phase: str) -> dict | None:
            # Sample up to E5_POOL_TARGET candidates
            sampled = events_list[:]
            rng.shuffle(sampled)
            sampled = sampled[:E5_POOL_TARGET]
            candidates = [_project_candidate(e) for e in sampled]
            matching_indices = [
                i for i, e in enumerate(sampled)
                if _jaccard(e.get("source_hashtags") or [], directive_hashtags) >= 0.3
            ]
            if len(matching_indices) < E5_MIN_MATCHING_CANDIDATES:
                return None
            # Compose geo + calendar snapshot for the prompt
            # (the agent sees "city_at_t_test", plus the post-fold calendar list)
            city_at_t = None
            for e in bq._load_events(user_id, "instagram") + bq._load_events(user_id, "facebook") + bq._load_events(user_id, "threads"):
                if (e.get("source_timestamp") or 0) >= t_test:
                    continue
                loc = e.get("event_location")
                if isinstance(loc, dict):
                    city_at_t = loc.get("city")
            calendar_snapshot = list(bq.get_calendar_state(user_id, as_of_timestamp=t_test).values())
            return {
                "instance_id": f"e5_{rec['persona_item'][:40]}_{phase}",
                "task_id": "e5_horizon_lifecycle",
                "canonical_id": rec["persona_item"],
                "persona_item": rec["persona_item"],
                "time_horizon": "short_term",
                "stop_condition_type": sc.get("type", "event"),
                "stop_condition_description": sc.get("description", ""),
                "expected_stop_ts": expected_stop_ts,
                "t_test": t_test,
                "phase": phase,
                "directive_hashtags": directive_hashtags,
                "candidates": candidates,
                "matching_indices": matching_indices,
                "geo_context": {
                    "city_at_t_test": city_at_t or "",
                    "same_as_intent_city": bool(city_at_t and rec["event_location_cities"] and city_at_t in rec["event_location_cities"]),
                },
                "calendar_context": calendar_snapshot,
            }

        pre_inst = _build_phase(pre_events, t_active, "pre")
        post_inst = _build_phase(post_events, t_post, "post")
        if pre_inst and post_inst:
            instances.extend([pre_inst, post_inst])
    return instances


def e5_prompt(instance: dict, history_block: str | None) -> str:
    phase = instance.get("phase", "pre")
    history = f"\n## User history (time-masked)\n{history_block}\n" if history_block else ""
    candidates = instance.get("candidates") or []
    cand_lines = "\n".join(
        f"- idx {i}: hashtags={c.get('hashtags', [])} | title={c.get('title', '')!r} | caption={c.get('caption', '')!r}"
        + (" | sponsored" if c.get('is_sponsored') else "")
        for i, c in enumerate(candidates)
    )
    geo = instance.get("geo_context") or {}
    cal = instance.get("calendar_context") or []
    cal_json = json.dumps([
        {k: v for k, v in (e or {}).items() if k in ("title", "start_ts", "type", "location")}
        for e in cal[:10]
    ], ensure_ascii=False)
    # The prompt deliberately does NOT name the short-term preference, its stop
    # condition, the lifecycle "phase" (ACTIVE/AFTER), or the scoring rule —
    # leaking any of those collapsed this into instruction-following. The agent
    # must INFER from history + current context whether a past interest is live.
    return f"""# Task: rank candidates for what to surface right now

You are choosing what to put in this user's feed. Some interests in a user's
history are time-bound (a trip, an event, a one-off project) and stop being
relevant once that moment passes; others are durable. Use the user's history
and present context to judge which still matter RIGHT NOW.

## User context at current moment
- City: {geo.get('city_at_t_test', '(unknown)')}
- In a city matching a past interest: {geo.get('same_as_intent_city', False)}
- Upcoming/recent calendar snapshot: {cal_json}
{history}
## Candidates (order is random)
{cand_lines}

## Your job
Rank the candidates from most to least worth surfacing for this user RIGHT NOW,
based on their history and present context (location, calendar, recency of
engagement) — weighing, for each relevant interest, whether it still appears
active or has likely passed.

## Output
Respond with ONE fenced ```json block:
```json
{{
  "ranked_indices": [<idx>, <idx>, ...],
  "stopped_due_to": "<=1 sentence; empty string in the active phase>",
  "reasoning": "<=2 sentences"
}}
```
The list must be a permutation of every idx above.
"""


def compute_e5_metrics(ranked: list[int], instance: dict) -> dict:
    matching = set(instance.get("matching_indices") or [])
    k_candidates = len(instance.get("candidates") or [])
    top3 = set(ranked[:3]) if ranked else set()
    top1 = ranked[0] if ranked else -1
    match_rate_3 = len(matching & top3) / 3.0 if matching else 0.0
    hard_top1 = int(top1 in matching)
    return {
        "match_rate_at_3": match_rate_3,
        "matching_in_top1": hard_top1,
        "mrr_matching": metrics.mrr(ranked, matching) if matching else 0.0,
    }


def run_e5_horizon_lifecycle(
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
    from evaluation.inference_utils import dispatch_agent_run

    if limit is not None:
        instances = instances[:limit]

    # First pass: run each instance
    per_instance: list[dict] = []
    for inst in instances:
        t = inst["t_test"]
        history_block = None
        history_tokens = 0
        if mode in ("llm_longctx", "llm_memory", "mem0"):
            history_block, stats = snapshot_cache.get_or_build(bq, user_id, t, model_name, context_budget)
            history_tokens = stats["total_tokens"]
        prompt = e5_prompt(inst, history_block)

        if dry_run:
            per_instance.append({
                "task": "e5_horizon_lifecycle",
                "user_id": user_id,
                "instance_id": inst["instance_id"],
                "canonical_id": inst["canonical_id"],
                "phase": inst["phase"],
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
        candidates = inst.get("candidates") or []
        if not isinstance(ranked, list) or sorted(set(ranked)) != list(range(len(candidates))):
            ranked = list(range(len(candidates)))

        scored = compute_e5_metrics(ranked, inst)
        per_instance.append({
            "task": "e5_horizon_lifecycle",
            "user_id": user_id,
            "instance_id": inst["instance_id"],
            "canonical_id": inst["canonical_id"],
            "phase": inst["phase"],
            "t_test": t,
            "mode": mode,
            "ranked_indices": ranked,
            "metrics": scored,
            "agent_response": raw_response,
            "stopped_due_to": parsed.get("stopped_due_to", ""),
            "history_tokens": history_tokens,
            "tool_call_count": tool_call_count,
        })

    # Second pass: pair pre/post per canonical and emit lifecycle_score rows
    from collections import defaultdict as _dd
    paired: dict[str, dict[str, dict]] = _dd(dict)
    for r in per_instance:
        phase = r.get("phase")
        if phase in ("pre", "post") and r.get("metrics"):
            paired[r["canonical_id"]][phase] = r

    for cid, sides in paired.items():
        if "pre" in sides and "post" in sides:
            pre_m = sides["pre"]["metrics"].get("match_rate_at_3", 0.0)
            post_m = sides["post"]["metrics"].get("match_rate_at_3", 0.0)
            lifecycle_score = pre_m - post_m
            per_instance.append({
                "task": "e5_horizon_lifecycle",
                "user_id": user_id,
                "canonical_id": cid,
                "instance_id": f"e5_pair_{cid[:40]}",
                "phase": "paired",
                "metrics": {
                    "pre.match_rate_at_3": pre_m,
                    "post.match_rate_at_3": post_m,
                    "lifecycle_score": lifecycle_score,
                    "post.hard_violation_at_1": sides["post"]["metrics"].get("matching_in_top1", 0),
                },
            })
    return per_instance
