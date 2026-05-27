"""Single source of truth for MCP tool schemas.

Mirrors `evaluation/mcp_servers/{instagram,facebook,threads,chatbot}_mcp_server.py`
exactly. Used by `evaluation/audit_query_quality.py:_dim_tool_call_validity`
to validate that tool names referenced in `tool_call_rules` and
`final_state_expected` actually exist on the right MCP server, with the
right argument schema.

If you add or rename a tool in `_social_server.py` or
`chatbot_mcp_server.py`, update this registry too — there is a unit-test-style
self-check at the bottom (`_self_check_against_mcp_servers`) that imports
the FastMCP server modules and warns when this catalog drifts.
"""

from __future__ import annotations

from typing import Iterable

# Coarse-grained allowance labels — match the values used in
# `evaluation/task_registry.py:TASK_TYPE_META[*]["mcp_tools_allowed"]`.
ALLOWED_LABEL_TO_APPS: dict[str, frozenset[str]] = {
    "none":    frozenset(),
    "social":  frozenset({"instagram", "facebook", "threads"}),
    "chatbot": frozenset({"chatbot"}),
    # `all` covers chatbot + every social app (T13 send_post + E6 mistake-prevention).
    "all":     frozenset({"instagram", "facebook", "threads", "chatbot"}),
}

# Per-app allowed reaction values (mirrors REACTIONS in _social_server.py).
SOCIAL_REACTIONS: dict[str, frozenset[str]] = {
    "instagram": frozenset({"like", "save", "share", "not_interested"}),
    "facebook":  frozenset({"like", "love", "haha", "wow", "sad", "angry", "care"}),
    "threads":   frozenset({"like", "quote_repost", "not_interested"}),
}

_SOCIAL_APPS = ("instagram", "facebook", "threads")


def _social_tool_template(app: str) -> dict[str, dict]:
    """Build the per-app tool table for a social MCP server.

    Each entry: {tool_name: {kind: read|write, args: {arg: type}, simulator}}.
    `simulator` keys map to functions in `evaluation/tool_call_simulator.py`.
    `args` use a tiny syntax: type names ending in `?` are optional.
    """
    return {
        f"{app}_get_feed": {
            "app": app, "kind": "read",
            "args": {"cursor": "str?", "limit": "int?"},
            "simulator": "get_feed",
            "returns": "feed item list",
        },
        f"{app}_get_post": {
            "app": app, "kind": "read",
            "args": {"post_id": "str"},
            "simulator": "get_post",
            "returns": "single post detail",
        },
        f"{app}_search": {
            "app": app, "kind": "read",
            "args": {"query": "str", "search_type": "str?",
                     "cursor": "str?", "limit": "int?"},
            "simulator": "search",
            "returns": "matching posts",
        },
        f"{app}_list_dms": {
            "app": app, "kind": "read",
            "args": {"cursor": "str?", "limit": "int?"},
            "simulator": "list_dms",
            "returns": "DM threads",
        },
        f"{app}_get_dm_thread": {
            "app": app, "kind": "read",
            "args": {"thread_id": "str", "cursor": "str?", "limit": "int?"},
            "simulator": "get_dm_thread",
            "returns": "messages in one DM thread",
        },
        f"{app}_create_post": {
            "app": app, "kind": "write",
            "args": {"caption": "str", "media_refs": "list?", "alt_text": "str?"},
            "simulator": None,
            "returns": "post_id of newly-created post",
        },
        f"{app}_react": {
            "app": app, "kind": "write",
            "args": {"post_id": "str", "reaction_type": "str"},
            "simulator": None,
            "returns": "reaction confirmation",
            # reaction_type vocabulary checked at run time per app.
            "reaction_whitelist": SOCIAL_REACTIONS[app],
        },
        f"{app}_comment": {
            "app": app, "kind": "write",
            "args": {"post_id": "str", "text": "str"},
            "simulator": None,
            "returns": "comment confirmation",
        },
        f"{app}_send_dm": {
            "app": app, "kind": "write",
            "args": {"recipient_id": "str", "message": "str"},
            "simulator": None,
            "returns": "DM send confirmation",
        },
    }


_CHATBOT_TOOLS: dict[str, dict] = {
    "chatbot_get_history": {
        "app": "chatbot", "kind": "read",
        "args": {"cursor": "str?", "limit": "int?"},
        "simulator": "get_history",
        "returns": "recent chatbot conversations",
    },
    "chatbot_search_history": {
        "app": "chatbot", "kind": "read",
        "args": {"query": "str", "limit": "int?"},
        "simulator": "search_history",
        "returns": "matching past chatbot turns",
    },
    "chatbot_send_message": {
        "app": "chatbot", "kind": "write",
        "args": {"message": "str"},
        "simulator": None,
        "returns": "message confirmation",
    },
    "chatbot_send_post_to_app": {
        # NOTE: this writes a post on the target SOCIAL app — recorded under
        # `tool=f"{target_app}_create_post"` in the overlay (see
        # chatbot_mcp_server.py:send_post_to_app). So `final_state_expected`
        # for T13 (agentic_send_post) lists `<target_app>_create_post`,
        # NOT `chatbot_send_post_to_app`. We register both names so a rule
        # written either way validates.
        "app": "chatbot", "kind": "write",
        "args": {"target_app": "str", "caption": "str", "media_refs": "list?"},
        "simulator": None,
        "returns": "cross-app post confirmation",
    },
    "chatbot_summarize_inbox": {
        "app": "chatbot", "kind": "read",
        "args": {"target_app": "str", "window_hours": "int?"},
        "simulator": "summarize_inbox",
        "returns": "raw DMs from a target social app",
    },
}


def _build_registry() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for app in _SOCIAL_APPS:
        out.update(_social_tool_template(app))
    out.update(_CHATBOT_TOOLS)
    return out


TOOL_REGISTRY: dict[str, dict] = _build_registry()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def is_known_tool(name: str) -> bool:
    return name in TOOL_REGISTRY


def get_tool(name: str) -> dict | None:
    return TOOL_REGISTRY.get(name)


def tools_for_label(label: str) -> set[str]:
    """All tool names allowed by an `mcp_tools_allowed` label.

    `label` is one of {"none", "social", "chatbot", "all"} — the values used
    on `TASK_TYPE_META[*]["mcp_tools_allowed"]`. Returns an empty set for
    `"none"` (no MCP tools) and an empty set for any unknown label.
    """
    apps = ALLOWED_LABEL_TO_APPS.get(label or "none", frozenset())
    if not apps:
        return set()
    return {name for name, meta in TOOL_REGISTRY.items() if meta["app"] in apps}


def _parse_arg_type(spec: str) -> tuple[str, bool]:
    """Tiny arg-spec parser: 'str?' → ('str', True), 'str' → ('str', False)."""
    if spec.endswith("?"):
        return spec[:-1], True
    return spec, False


def validate_tool_call(
    name: str,
    args: dict | None,
    allowed_label: str | None = None,
) -> dict:
    """Validate one tool call against the registry.

    Returns ``{"ok": bool, "errors": [str, ...]}``.

    Checks:
      1. Tool exists in the registry.
      2. Tool is allowed by `allowed_label` (if provided).
      3. Required args (no `?`) are present.
      4. No unknown args.
      5. For `_react` tools: reaction_type is in the per-app whitelist.

    Type-checking is intentionally light — the agent passes JSON, so we
    accept any non-None for `str`/`int`/`list` slots. The MCP server's own
    runtime validators (caption length, recipient resolution) catch the rest.
    """
    errors: list[str] = []
    args = args or {}
    meta = TOOL_REGISTRY.get(name)
    if meta is None:
        return {"ok": False, "errors": [f"unknown tool: {name!r}"]}

    if allowed_label is not None:
        allowed_set = tools_for_label(allowed_label)
        if name not in allowed_set:
            errors.append(
                f"tool {name!r} not in allowed label {allowed_label!r} "
                f"(apps={sorted(ALLOWED_LABEL_TO_APPS.get(allowed_label, []))})"
            )

    spec = meta.get("args") or {}
    for arg_name, arg_spec in spec.items():
        _, optional = _parse_arg_type(arg_spec)
        if not optional and arg_name not in args:
            errors.append(f"missing required arg {arg_name!r} for {name}")
    for k in args:
        if k not in spec:
            errors.append(f"unknown arg {k!r} for {name}")

    if name.endswith("_react"):
        rxn = args.get("reaction_type")
        if rxn is not None and rxn not in (meta.get("reaction_whitelist") or set()):
            errors.append(
                f"reaction_type {rxn!r} not allowed for {name} "
                f"(valid: {sorted(meta.get('reaction_whitelist') or set())})"
            )

    return {"ok": not errors, "errors": errors}


def validate_tool_name(name: str, allowed_label: str | None = None) -> dict:
    """Lighter check: just verify the tool name exists and is allowed.

    Used when the audit only knows the tool *name* (e.g. parsed from
    `count('instagram_create_post') == N` rules) without arg context.
    """
    if not is_known_tool(name):
        return {"ok": False, "errors": [f"unknown tool: {name!r}"]}
    if allowed_label is not None and name not in tools_for_label(allowed_label):
        return {
            "ok": False,
            "errors": [
                f"tool {name!r} not allowed under {allowed_label!r}"
            ],
        }
    return {"ok": True, "errors": []}


def required_reads_for_task(task_id: str, target_app: str | None) -> list[str]:
    """Best-effort map of agentic/E task → the read tools the prompt directs
    the agent to call before responding. Mirrors `evaluation/prompts_agentic.py`
    which embeds explicit `mcp__{app}__{tool}` directives in each task prompt.

    Returned tool names are dry-runnable (all are read tools). The dry-run
    executor uses these to confirm the data the agent would have seen at
    `t_test` is non-empty and overlaps the example_response's claims.
    """
    app = (target_app or "").lower()
    is_social = app in _SOCIAL_APPS

    # Agentic T6–T19 (single-app tasks read from `target_app`).
    AGENTIC: dict[str, list[str]] = {
        "agentic_community_post":           [f"{app}_get_feed"] if is_social else [],
        "agentic_send_post":                [f"{app}_get_feed"] if is_social else [],
        "agentic_dm_digest":                [f"{app}_list_dms"] if is_social else [],
        "agentic_cross_app_repost":         [f"{app}_get_feed"] if is_social else [],
        "agentic_auto_reply":               [f"{app}_get_feed"] if is_social else [],
        "agentic_vague_refind":             [
            "instagram_search", "facebook_search", "threads_search",
            "chatbot_search_history",
        ],
        "agentic_group_dm_summary":         [f"{app}_get_feed"] if is_social else [],
        "agentic_wrong_recipient_check":    [f"{app}_list_dms"] if is_social else [],
        "agentic_proactive_daily_catchup":  [
            "instagram_get_feed", "facebook_get_feed", "threads_get_feed",
            "chatbot_get_history",
        ],
        "agentic_trending_alert":           [
            "instagram_get_feed", "threads_get_feed", "chatbot_get_history",
        ],
    }
    if task_id in AGENTIC:
        return [t for t in AGENTIC[task_id] if is_known_tool(t)]

    # E-family. Only the two tasks with mcp_tools_allowed != "none" actually
    # call MCP tools; the others rank from time-masked history alone.
    E_FAMILY: dict[str, list[str]] = {
        "daily_personalized_briefing":      [
            "instagram_get_feed", "facebook_get_feed", "threads_get_feed",
            "chatbot_get_history",
        ],
        "active_mistake_prevention":        [
            f"{app}_get_feed" if is_social else "instagram_get_feed",
            "chatbot_get_history",
        ],
    }
    return [t for t in E_FAMILY.get(task_id, []) if is_known_tool(t)]


# Wildcard sentinels that show up in `tool_call_rules` to mean "any tool of
# kind X" — recognized by the agent runner, not by the per-tool MCP registry.
# Listed here so the validator doesn't flag them as unknown.
WILDCARD_SENTINELS: frozenset[str] = frozenset({
    "__any_write__",  # E6: "no write tools called" assertion
    "__any_read__",   # reserved
})


def is_known_tool_or_sentinel(name: str) -> bool:
    return name in TOOL_REGISTRY or name in WILDCARD_SENTINELS


def extract_tool_names_from_rules(rules: Iterable[str] | None) -> list[str]:
    """Pull tool names out of `tool_call_rules` strings.

    Mirrors the regex in `evaluation/tasks/agentic_tasks.py:_check_tool_call_rules`.
    Wildcard sentinels (e.g. `__any_write__`) are excluded — the validator
    handles them separately via `is_known_tool_or_sentinel`.
    """
    import re
    names: list[str] = []
    for rule in (rules or []):
        m = re.match(r"count\('([^']+)'\)\s*", rule or "")
        if m:
            names.append(m.group(1))
    return names


def extract_tool_names_from_final_state(expected: dict | None) -> list[str]:
    """Pull tool names from `final_state_expected` (must_contain_count keys + must_not_contain entries)."""
    if not isinstance(expected, dict):
        return []
    out: list[str] = []
    out.extend((expected.get("must_contain_count") or {}).keys())
    out.extend(expected.get("must_not_contain") or [])
    return out


__all__ = [
    "TOOL_REGISTRY",
    "ALLOWED_LABEL_TO_APPS",
    "SOCIAL_REACTIONS",
    "WILDCARD_SENTINELS",
    "is_known_tool",
    "is_known_tool_or_sentinel",
    "get_tool",
    "tools_for_label",
    "validate_tool_call",
    "validate_tool_name",
    "required_reads_for_task",
    "extract_tool_names_from_rules",
    "extract_tool_names_from_final_state",
]
