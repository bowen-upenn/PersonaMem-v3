"""Task E4 — Google Search personalization evaluation (opt-in).

The agent is asked to search for content relevant to the user's recent
preferences via the `search_google` MCP tool. Results are judged for
both `personalization_fit` (does the ranking reflect recent prefs) and
`news_honesty` (does the ranking remain faithful to real-world events).

Three-level gating:
- `--enable_e4` (default off): master switch; E4 is entirely skipped otherwise
- `--e4_allow_live` (default off): enables live Google API calls (via MCP env)
- `--e4_skip_on_missing_key` (default on): skip instances when no API key
  AND no cache entry exists

This task is NOT registered in the default `all` task alias because of
its external-API coupling; users must opt in explicitly.
"""

from __future__ import annotations

import datetime as _dt

from evaluation.backend_query import BackendQuery


E4_DEFAULT_N_DAYS: int = 3


def build_e4_google_search(
    bq: BackendQuery,
    user_id: str,
    t_anchor: int,
    n_days: int = E4_DEFAULT_N_DAYS,
) -> list[dict]:
    """Reuse E3's day sampler: one instance per stratified day. Each
    instance carries a `recent_pref_summary` derived from the 24h window
    before `t_test` (contradiction-aware via BackendQuery.get_preferences).
    """
    from evaluation.tasks.e3_daily_briefing_multi import _collect_day_buckets

    buckets = _collect_day_buckets(bq, user_id)
    if len(buckets) < n_days:
        return []
    sorted_days = sorted(buckets.keys())
    if len(sorted_days) < 3:
        return []

    eligible = sorted_days[1:-1]
    if not eligible:
        return []

    by_volume = sorted(eligible, key=lambda d: len(buckets[d]))
    n = len(by_volume)
    tertile = max(1, n // 3)
    picks: list[str] = []
    if n_days >= 1 and by_volume[2 * tertile:]:
        picks.append(by_volume[2 * tertile:][-1])
    if n_days >= 2 and by_volume[tertile:2 * tertile]:
        mid = by_volume[tertile:2 * tertile]
        picks.append(mid[len(mid) // 2])
    if n_days >= 3 and by_volume[:tertile]:
        picks.append(by_volume[:tertile][0])
    picks = picks[:n_days]
    picks.sort()

    instances: list[dict] = []
    for i, day in enumerate(picks):
        dt0 = _dt.datetime.strptime(day, "%Y-%m-%d").replace(
            hour=12, minute=0, second=0, tzinfo=_dt.timezone.utc
        )
        noon_ts = int(dt0.timestamp())
        # Recent prefs in the 24h window before t_test, contradiction-aware
        recent_window_start = noon_ts - 24 * 3600
        all_prefs = bq.get_preferences(user_id=user_id, since_timestamp=noon_ts)
        recent_prefs = [p for p in all_prefs if (p.get("source_timestamp") or 0) >= recent_window_start]
        # Rank by frequency (more occurrences in the window = stronger recent signal)
        from collections import Counter as _Counter
        counts = _Counter(p.get("persona_item", "") for p in recent_prefs if p.get("persona_item"))
        top_prefs = [{"persona_item": pi, "count": c} for pi, c in counts.most_common(8)]
        instances.append({
            "instance_id": f"e4_day_{i}",
            "task_id": "e4_google_search",
            "entry_point": "chatbot_routed",
            "t_test": noon_ts,
            "day_index": i,
            "day_label": day,
            "recent_pref_summary": top_prefs,
        })
    return instances


def e4_prompt(instance: dict, history_block: str | None) -> str:
    recent = "\n".join(
        f"- {p.get('persona_item', '')} ({p.get('count', 0)}x)"
        for p in (instance.get("recent_pref_summary") or [])
    ) or "(no strong recent signals)"
    history = f"\n## User history (time-masked)\n{history_block}\n" if history_block else ""
    return f"""# Task: personalized Google search

It's {instance['day_label']} 12:00 UTC. Use the `search_google(query, num_results)` tool
to search for content the user is likely to engage with TODAY, given their
recent preferences.

## User's recent preferences (last 24h, contradiction-aware)
{recent}
{history}
## Rules
- Issue 1-3 search queries via the tool. Keep queries crisp and topical.
- Personalize the choice of query terms based on recent prefs, but remain
  honest about what's actually happening in the world — do NOT suppress
  major news just because it doesn't match the persona.
- Return the final ranked list of items the user would care about today.

## Output
Respond with ONE fenced ```json block:
```json
{{
  "queries": ["...", "..."],
  "ranked_results": [
    {{"title": "...", "link": "...", "why": "<=1 sentence grounded in recent prefs"}},
    ...
  ],
  "reasoning": "<=2 sentences"
}}
```"""


def run_e4_google_search(
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
    """Runner for E4. Requires `mcp_agent` mode AND an environment where
    the google_search MCP server is wired in.

    This runner is intentionally simple: it produces a structured
    `skipped` result when (a) mode is not `mcp_agent`, (b) the enable
    flag is off (checked via env), or (c) the tool returns
    `cache_miss_live_disabled` and we're running in replay mode.
    """
    from data_preparation.utils import extract_json_from_response
    from evaluation.inference_utils import dispatch_agent_run
    import os as _os

    if limit is not None:
        instances = instances[:limit]

    results: list[dict] = []
    e4_enabled = _os.environ.get("PM3_E4_ENABLED") == "1"

    for inst in instances:
        if not e4_enabled:
            results.append({
                "task": "e4_google_search",
                "user_id": user_id,
                "instance_id": inst["instance_id"],
                "skipped": "e4_not_enabled",
                "day_label": inst.get("day_label"),
            })
            continue

        if mode != "mcp_agent":
            results.append({
                "task": "e4_google_search",
                "user_id": user_id,
                "instance_id": inst["instance_id"],
                "skipped": f"e4_requires_mcp_agent_mode (got {mode})",
                "day_label": inst.get("day_label"),
            })
            continue

        t = inst["t_test"]
        history_block = None
        history_tokens = 0
        prompt = e4_prompt(inst, history_block)

        if dry_run:
            results.append({
                "task": "e4_google_search",
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
        results.append({
            "task": "e4_google_search",
            "user_id": user_id,
            "instance_id": inst["instance_id"],
            "day_label": inst["day_label"],
            "mode": mode,
            "n_queries": len(parsed.get("queries") or []),
            "n_ranked_results": len(parsed.get("ranked_results") or []),
            "agent_response": raw_response,
            "tool_call_count": tool_call_count,
        })
    return results
