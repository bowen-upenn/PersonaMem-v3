"""Task E3 — multi-day proactive daily briefing.

Extends T18 (proactive_daily) into N stratified day-midpoints per user.
Same query ("what should I catch up on today"), different `t_test` moments,
measured per-instance plus an aggregate cross-day consistency drift metric.
"""

from __future__ import annotations

from data_preparation.utils import extract_json_from_response
from evaluation.backend_query import BackendQuery

E3_DEFAULT_N_DAYS: int = 8


_POSITIVE_INTERACTION_TYPES = {"explicit_positive", "implicit_positive"}
_NEGATIVE_INTERACTION_TYPES = {"explicit_negative", "implicit_negative"}


def _collect_day_buckets(bq: BackendQuery, user_id: str,
                         positive_only: bool = False) -> dict[str, list[dict]]:
    """Group events by UTC calendar day string (YYYY-MM-DD).

    With ``positive_only=True`` only events whose source_interaction_type
    is explicit_positive or implicit_positive are kept (workstream B —
    the briefing must be grounded in real positive engagement, not
    raw event volume which includes negatives)."""
    import datetime as _dt
    buckets: dict[str, list[dict]] = {}
    for app in ("instagram", "facebook", "threads", "chatbot"):
        for e in bq._load_events(user_id, app):
            ts = e.get("source_timestamp") or 0
            if not ts:
                continue
            if positive_only and (e.get("source_interaction_type") or "") not in _POSITIVE_INTERACTION_TYPES:
                continue
            day = _dt.datetime.fromtimestamp(int(ts), tz=_dt.timezone.utc).strftime("%Y-%m-%d")
            row = dict(e)
            row.setdefault("_app", app)
            buckets.setdefault(day, []).append(row)
    return buckets


def _events_in_window(buckets: dict[str, list[dict]], day: str,
                      t_test: int, t_end: int) -> list[dict]:
    """Return rows in ``buckets[day]`` with t_test ≤ ts < t_end."""
    rows = buckets.get(day) or []
    return [e for e in rows
            if t_test <= int(e.get("source_timestamp") or 0) < t_end]


def _summarize_engagement(rows: list[dict], limit: int = 6) -> list[dict]:
    """Compact projection of an engagement event for the GT block."""
    out: list[dict] = []
    for e in rows[:limit]:
        out.append({
            "ts": int(e.get("source_timestamp") or 0),
            "app": e.get("_app", ""),
            "hashtags": (e.get("source_hashtags") or [])[:6],
            "interaction_type": e.get("source_interaction_type", ""),
            "content_snippet": (
                (e.get("content") or {}).get("caption")
                or (e.get("content") or {}).get("title")
                or ""
            )[:140],
        })
    return out


def build_e3_daily_briefing_multi(
    bq: BackendQuery,
    user_id: str,
    t_anchor: int,
    n_days: int = E3_DEFAULT_N_DAYS,
) -> list[dict]:
    """Workstream B: pick days that have ≥3 positive engagements;
    insert the briefing at 8 AM UTC; attach forward-looking GT
    (real positive engagements that day after t_test, plus any
    negatives in the same window).
    """
    import datetime as _dt

    pos_buckets = _collect_day_buckets(bq, user_id, positive_only=True)
    all_buckets = _collect_day_buckets(bq, user_id, positive_only=False)
    # Days with at least 3 positive events — these are eligible.
    eligible = [d for d, rows in pos_buckets.items() if len(rows) >= 3]
    if not eligible:
        return []
    eligible.sort()

    # Enforce a 24h guard on both ends so the prior-day window + post-test
    # 24h window both fall inside the user's data range.
    if len(eligible) >= 3:
        eligible = eligible[1:-1]
    if not eligible:
        return []

    # Pick up to n_days days with the most positive events, then sort
    # chronologically.
    by_pos_volume_desc = sorted(eligible, key=lambda d: -len(pos_buckets[d]))
    picks = sorted(by_pos_volume_desc[:n_days])

    DAY = 24 * 3600
    instances: list[dict] = []
    for i, day in enumerate(picks):
        dt0 = _dt.datetime.strptime(day, "%Y-%m-%d").replace(
            hour=8, minute=0, second=0, tzinfo=_dt.timezone.utc
        )
        morning_ts = int(dt0.timestamp())
        end_ts = morning_ts + DAY
        prior_day = (dt0 - _dt.timedelta(days=1)).strftime("%Y-%m-%d")

        # Forward-looking GT: the user's actual positive engagements
        # AFTER the briefing fired (8 AM → end of day) — what the
        # briefing should have surfaced. Source B-style.
        pos_after = _events_in_window(pos_buckets, day, morning_ts, end_ts)
        # Avoid list: real disliked rows in the same forward window.
        all_after = _events_in_window(all_buckets, day, morning_ts, end_ts)
        neg_after = [e for e in all_after
                     if (e.get("source_interaction_type") or "") in _NEGATIVE_INTERACTION_TYPES]

        instances.append({
            "instance_id": f"e3_day_{i}",
            "task_id": "e3_daily_briefing_multi",
            "entry_point": "chatbot_routed",
            "t_test": morning_ts,
            "day_index": i,
            "day_label": day,
            "prior_day_label": prior_day,
            # Workstream B: forward-looking GT lists. Both lists carry
            # real rows (timestamps + hashtags + content snippets) so
            # the rubric can match the briefing against actual
            # post-t_test engagement.
            "gt_positive_engagements": _summarize_engagement(pos_after, limit=8),
            "gt_avoid_engagements":    _summarize_engagement(neg_after, limit=4),
            "tool_call_rules": [
                "count('instagram_create_post') == 0",
                "count('facebook_create_post') == 0",
                "count('threads_create_post') == 0",
                "count('instagram_send_dm') == 0",
            ],
        })
    return instances


def e3_prompt(instance: dict, history_block: str | None) -> str:
    """Prompt mirrors T18 (`what should I catch up on today`) but cleanly
    anchors the current moment to the instance's `day_label`.
    """
    history = f"\n## User history (time-masked to before {instance['day_label']} 12:00 UTC)\n{history_block}\n" if history_block else ""
    return f"""# Task: proactive daily briefing for {instance['day_label']}

It's noon on {instance['day_label']} (UTC). The user hasn't asked anything;
you're proactively suggesting what they'd want to catch up on from their
feeds today. Surface a small, diverse set (3-5 items max) tied to their
actual preferences visible in the time-masked history.
{history}
Hard rules:
- READS ONLY — do not create posts, send DMs, or perform any write action.
- Avoid anything the user has explicitly disliked or asked not to see in recent history.
- Align with recent (prior-day) engagement when possible — consistency across days matters.

## Output
Respond with ONE fenced ```json block:
```json
{{
  "suggestions": [
    {{"title": "...", "why": "<=1 sentence grounded in history"}},
    ...
  ],
  "reasoning": "<=2 sentences"
}}
```"""


_HASHTAG_RE = __import__("re").compile(r"#([A-Za-z0-9_]{2,})")


def _extract_briefing_topics(suggestions: list[dict]) -> set[str]:
    """Pull topical signal tokens out of a briefing's suggestions list.
    Combines hashtags + lowercased title content tokens (≥ 4 chars,
    stop-word filtered) so the comparison works whether the agent emits
    explicit hashtags or just descriptive titles.
    """
    if not suggestions:
        return set()
    _STOP = {"the", "and", "for", "with", "your", "you", "from", "this", "that",
             "today", "what", "about", "have", "their", "some", "more", "into",
             "over", "than", "when", "where", "look", "want", "just"}
    out: set[str] = set()
    for s in suggestions:
        if not isinstance(s, dict):
            continue
        text = " ".join(str(s.get(k, "")) for k in ("title", "why")).lower()
        for m in _HASHTAG_RE.findall(text):
            out.add(m.lower())
        for tok in text.split():
            tok = "".join(c for c in tok if c.isalpha())
            if len(tok) >= 4 and tok not in _STOP:
                out.add(tok)
    return out


def _prior_day_top_hashtags(bq: BackendQuery, user_id: str, t_test: int,
                            window_seconds: int = 86400, top_n: int = 8) -> set[str]:
    """Return the top N hashtags the user engaged with in the 24h before t_test."""
    from collections import Counter
    lo = t_test - window_seconds
    hi = t_test
    counts: Counter = Counter()
    for app in ("instagram", "facebook", "threads", "chatbot"):
        for e in bq._load_events(user_id, app):
            ts = int(e.get("source_timestamp") or 0)
            if not (lo <= ts < hi):
                continue
            for h in (e.get("source_hashtags") or []):
                counts[h.lower().lstrip("#")] += 1
    return {h for h, _ in counts.most_common(top_n)}


def compute_e3_metrics(result_parsed: dict, instance: dict, bq: BackendQuery,
                       user_id: str) -> dict:
    """Per-instance metrics — Phase L.B.2 implements the cross-day signal.

    `briefing_personalization_score`: jaccard between this briefing's topical
    tokens and the user's top-engaged hashtags in the prior 24h. High = the
    agent's briefing aligns with what the user actually engaged with
    yesterday (good personalization). Low = generic / brittle / off-topic.
    `briefing_topics` is the extracted token set, used downstream for
    cross-day drift comparison in the runner.
    """
    suggestions = (result_parsed or {}).get("suggestions") or []
    topics = _extract_briefing_topics(suggestions)
    prior_top = _prior_day_top_hashtags(bq, user_id, instance.get("t_test", 0))
    if topics and prior_top:
        inter = len(topics & prior_top)
        union = len(topics | prior_top)
        align = inter / union if union else 0.0
    else:
        align = 0.0
    return {
        "n_suggestions": len(suggestions),
        "has_structured_output": int(bool(suggestions)),
        "briefing_topics": sorted(topics)[:20],
        "briefing_personalization_score": round(align, 3),
        "n_prior_day_top_hashtags": len(prior_top),
    }


def run_e3_daily_briefing_multi(
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

    results: list[dict] = []
    for inst in instances:
        t = inst["t_test"]
        history_block = None
        history_tokens = 0
        if mode in ("llm_longctx", "memory"):
            history_block, stats = snapshot_cache.get_or_build(bq, user_id, t, model_name, context_budget)
            history_tokens = stats["total_tokens"]

        prompt = e3_prompt(inst, history_block)
        if dry_run:
            results.append({
                "task": "e3_daily_briefing_multi",
                "user_id": user_id,
                "instance_id": inst["instance_id"],
                "day_label": inst["day_label"],
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
        m = compute_e3_metrics(parsed, inst, bq, user_id)
        results.append({
            "task": "e3_daily_briefing_multi",
            "user_id": user_id,
            "instance_id": inst["instance_id"],
            "day_label": inst["day_label"],
            "mode": mode,
            "metrics": m,
            "agent_response": raw_response,
            "history_tokens": history_tokens,
            "tool_call_count": tool_call_count,
        })

    # Phase L.B.2: cross-day drift (post-loop). For each adjacent (day_N,
    # day_{N+1}) pair, compute jaccard(briefing_topics_N, briefing_topics_{N+1}).
    # Low jaccard = the agent adapts the briefing day-to-day; very high (≥0.7)
    # = brittle repetition (same set of suggestions regardless of new history).
    sorted_results = sorted(results, key=lambda r: r.get("day_label", ""))
    for i in range(1, len(sorted_results)):
        prev_topics = set(sorted_results[i - 1]["metrics"].get("briefing_topics") or [])
        curr_topics = set(sorted_results[i]["metrics"].get("briefing_topics") or [])
        if prev_topics and curr_topics:
            inter = len(prev_topics & curr_topics)
            union = len(prev_topics | curr_topics)
            jacc = inter / union if union else 0.0
        else:
            jacc = 0.0
        sorted_results[i]["metrics"]["cross_day_repetition_rate"] = round(jacc, 3)
    return results
