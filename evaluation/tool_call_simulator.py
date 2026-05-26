"""Pure-Python dry-run of MCP read tools against the local backend.

Mirrors the read-side semantics of `evaluation/mcp_servers/_social_server.py`
and `evaluation/mcp_servers/chatbot_mcp_server.py` 1:1, but without the
overlay (no agent writes) — for audit-time validation that the data the
agent would have read at `t_test` actually exists.

Used by `evaluation/audit_query_quality.py:_dim_tool_call_validity`.
"""

from __future__ import annotations

from typing import Any

from evaluation.backend_query import APPS, BackendQuery


# Plain-Python re-implementation of the lightweight projections that the
# real MCP tools apply on top of `BackendQuery` rows. Kept narrow — the
# audit only needs enough fields for the supportability LLM judge.


def _project_feed_event(e: dict) -> dict:
    item = {
        "post_id": str(e.get("source_object_id", "")),
        "timestamp": e.get("source_timestamp"),
        "hashtags": e.get("source_hashtags", []),
        "content_type": e.get("content_type"),
        "title": (e.get("content") or {}).get("title"),
        "caption": (e.get("content") or {}).get("caption"),
        "author_id": e.get("author_id", "unknown"),
        "is_self_authored": bool(e.get("is_self_authored")),
        "is_dm": bool(e.get("is_dm")),
        "interaction_type": e.get("source_interaction_type", ""),
    }
    if e.get("is_ad"):
        item["is_ad"] = True
    if e.get("is_trending"):
        item["is_trending"] = True
        item["trending_topic"] = e.get("trending_topic", "")
    loc = e.get("event_location")
    if isinstance(loc, dict) and loc.get("city"):
        item["location"] = f"{loc.get('city', '')}, {loc.get('region', '')}".rstrip(", ")
    return item


def simulate_get_feed(
    bq: BackendQuery, user_id: str, app: str, t_test: int,
    cursor: str | None = None, limit: int = 20,
) -> dict:
    """Mirror `_social_server.py:get_feed`."""
    if app not in APPS:
        return {"results": [], "_error": f"unknown app {app!r}"}
    events = bq.get_events(user_id=user_id, app=app, since_timestamp=t_test)
    events.sort(key=lambda x: x.get("source_timestamp", 0), reverse=True)
    items = [_project_feed_event(e) for e in events]
    start = int(cursor) if (cursor and cursor.isdigit()) else 0
    page = items[start:start + int(limit or 20)]
    out = {"results": page}
    if start + int(limit or 20) < len(items):
        out["nextCursor"] = str(start + int(limit or 20))
    return out


def simulate_get_post(
    bq: BackendQuery, user_id: str, app: str, t_test: int,
    post_id: str | None = None,
) -> dict | None:
    """Mirror `_social_server.py:get_post`. Time-mask via t_test."""
    if not post_id:
        return None
    e = bq.get_event_by_id(user_id=user_id, app=app, event_id=str(post_id))
    if e is None:
        return None
    if (e.get("source_timestamp") or 0) >= t_test:
        return None  # not yet visible at t_test
    return e


def simulate_search(
    bq: BackendQuery, user_id: str, app: str, t_test: int,
    query: str = "", search_type: str = "post",
    cursor: str | None = None, limit: int = 20,
) -> dict:
    """Mirror `_social_server.py:search`."""
    return bq.search_events(
        user_id=user_id, app=app, query=query or "",
        search_type=search_type or "post",
        since_timestamp=t_test, cursor=cursor, limit=int(limit or 20),
    )


def simulate_list_dms(
    bq: BackendQuery, user_id: str, app: str, t_test: int,
    cursor: str | None = None, limit: int = 20,
) -> dict:
    """Mirror `_social_server.py:list_dms`."""
    return bq.list_dm_threads(
        user_id=user_id, app=app,
        since_timestamp=t_test, cursor=cursor, limit=int(limit or 20),
    )


def simulate_get_dm_thread(
    bq: BackendQuery, user_id: str, app: str, t_test: int,
    thread_id: str | None = None,
    cursor: str | None = None, limit: int = 50,
) -> dict | None:
    """Mirror `_social_server.py:get_dm_thread`."""
    if not thread_id:
        return None
    return bq.get_dm_thread(
        user_id=user_id, app=app, thread_id=str(thread_id),
        since_timestamp=t_test, cursor=cursor, limit=int(limit or 50),
    )


def simulate_get_history(
    bq: BackendQuery, user_id: str, t_test: int,
    cursor: str | None = None, limit: int = 20, **_ignored,
) -> dict:
    """Mirror `chatbot_mcp_server.py:get_history`."""
    events = bq.get_events(user_id=user_id, app="chatbot", since_timestamp=t_test)
    events.sort(key=lambda x: x.get("source_timestamp", 0), reverse=True)
    page = [{
        "conversation_id": str(e.get("source_object_id", "")),
        "timestamp": e.get("source_timestamp"),
        "conversation_type": e.get("conversation_type"),
        "conversation": e.get("conversation", []),
        "interaction_format": e.get("interaction_format", {}),
    } for e in events]
    start = int(cursor) if (cursor and cursor.isdigit()) else 0
    window = page[start:start + int(limit or 20)]
    out = {"results": window}
    if start + int(limit or 20) < len(page):
        out["nextCursor"] = str(start + int(limit or 20))
    return out


def simulate_search_history(
    bq: BackendQuery, user_id: str, t_test: int,
    query: str = "", limit: int = 10, **_ignored,
) -> dict:
    """Mirror `chatbot_mcp_server.py:search_history`."""
    q = (query or "").lower()
    hits: list[dict] = []
    for e in bq.get_events(user_id=user_id, app="chatbot", since_timestamp=t_test):
        for m in (e.get("conversation") or []):
            if q in (m.get("content", "") or "").lower():
                hits.append({
                    "conversation_id": str(e.get("source_object_id", "")),
                    "timestamp": e.get("source_timestamp"),
                    "role": m.get("role"),
                    "snippet": (m.get("content") or "")[:200],
                })
                if len(hits) >= int(limit or 10):
                    return {"results": hits}
    return {"results": hits}


def simulate_summarize_inbox(
    bq: BackendQuery, user_id: str, t_test: int,
    target_app: str = "", window_hours: int = 24, **_ignored,
) -> dict:
    """Mirror `chatbot_mcp_server.py:summarize_inbox`."""
    target_app = (target_app or "").lower()
    if target_app not in ("instagram", "facebook", "threads"):
        return {"threads": [], "_error": f"target_app must be a social app, got {target_app!r}"}
    since_ts = max(0, int(t_test) - int(window_hours or 24) * 3600)
    page = bq.list_dm_threads(user_id=user_id, app=target_app,
                               since_timestamp=t_test, limit=50)
    out: list[dict] = []
    for thread_summary in page.get("results", []):
        tid = thread_summary.get("thread_id")
        if (thread_summary.get("latest_ts") or 0) < since_ts:
            continue
        thread = bq.get_dm_thread(user_id=user_id, app=target_app,
                                   thread_id=tid, since_timestamp=t_test)
        if thread:
            recent = [m for m in thread.get("results", [])
                      if (m.get("timestamp") or 0) >= since_ts]
            if recent:
                out.append({
                    "thread_id": tid,
                    "participants": thread.get("participants"),
                    "is_group": thread.get("is_group"),
                    "messages": recent,
                })
    return {"threads": out, "window_hours": int(window_hours or 24),
            "target_app": target_app}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# Map registry `simulator` keys to the functions defined above.
_SIMULATORS = {
    "get_feed":        simulate_get_feed,
    "get_post":        simulate_get_post,
    "search":          simulate_search,
    "list_dms":        simulate_list_dms,
    "get_dm_thread":   simulate_get_dm_thread,
    "get_history":     simulate_get_history,
    "search_history":  simulate_search_history,
    "summarize_inbox": simulate_summarize_inbox,
}


def simulate_tool_call(
    name: str,
    bq: BackendQuery,
    user_id: str,
    t_test: int,
    args: dict | None = None,
) -> Any:
    """Dispatch `name` to its simulator. Returns whatever the real MCP read
    tool would have returned at `t_test`. For write tools, returns
    ``{"_simulated": False, "reason": "write tool — not dry-runnable"}``.
    For unknown tools returns ``{"_error": "unknown tool"}``.
    """
    from evaluation.mcp_tool_registry import TOOL_REGISTRY
    meta = TOOL_REGISTRY.get(name)
    if meta is None:
        return {"_error": f"unknown tool {name!r}"}
    if meta.get("kind") != "read":
        return {"_simulated": False,
                "reason": f"{name} is a write tool — not dry-runnable"}
    sim_key = meta.get("simulator")
    fn = _SIMULATORS.get(sim_key)
    if fn is None:
        return {"_error": f"no simulator wired for {name!r} (sim={sim_key!r})"}
    args = dict(args or {})
    # Auto-fill `app` arg from the tool name prefix for social tools.
    app = meta.get("app")
    try:
        if app != "chatbot":
            return fn(bq, user_id, app, int(t_test), **args)
        return fn(bq, user_id, int(t_test), **args)
    except TypeError as exc:
        return {"_error": f"simulator arg mismatch for {name!r}: {exc}"}
    except Exception as exc:
        return {"_error": f"simulator runtime error for {name!r}: {exc}"}


def is_nonempty(result: Any) -> bool:
    """Conservative: True if a simulator returned at least one row."""
    if result is None:
        return False
    if isinstance(result, dict):
        if result.get("_error") or result.get("_simulated") is False:
            return False
        # Tools return either {results:[...]} or {threads:[...]} or a single
        # event dict (get_post). Accept any non-empty case.
        if "results" in result:
            return bool(result["results"])
        if "threads" in result:
            return bool(result["threads"])
        # get_post: single dict — treat presence of source_timestamp as nonempty
        return bool(result.get("source_timestamp"))
    if isinstance(result, list):
        return bool(result)
    return False


def project_for_judge(result: Any, max_chars: int = 600) -> str:
    """Compact JSON dump of a simulated tool return, capped at `max_chars`
    so the audit LLM judge prompt stays small."""
    import json
    try:
        s = json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        s = str(result)
    if len(s) > max_chars:
        return s[:max_chars] + f"…[truncated, {len(s)} chars total]"
    return s


__all__ = [
    "simulate_tool_call",
    "is_nonempty",
    "project_for_judge",
    "simulate_get_feed",
    "simulate_get_post",
    "simulate_search",
    "simulate_list_dms",
    "simulate_get_dm_thread",
    "simulate_get_history",
    "simulate_search_history",
    "simulate_summarize_inbox",
]
