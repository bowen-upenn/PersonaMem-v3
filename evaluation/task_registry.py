"""Static metadata for every task_type emitted by build_benchmark.

This registry is the single source of truth for the eval harness's
dispatch decisions: which MCP servers to spin up per query, whether
the agent is allowed to write state, what kind of response to expect,
and which rubric dimensions apply. Keeping this as a static dict lets
the CSV-building script populate per-row dispatch columns without
instance inspection and lets the runner look up behavior by string.

`TASK_TYPE_META[task_type]` returns a dict with fields:

    task_family            : str   -- slate_ranking / chatbot_response /
                                       over_personalization / agentic / e_followup
    mcp_tools_allowed      : str   -- "none" / "social" / "chatbot" / "all"
    state_write_policy     : str   -- "read_only" / "writes_ok"
    expected_response_kind : str   -- "ranking" / "text" /
                                       "text_with_tool_calls" / "agentic_writes"
    rubric_tags            : list[str]  -- applicable rubric dimensions

Unknown task_types fall through `get()` with `DEFAULT_META` — the
prepare script logs a warning so new tasks get registered promptly.
"""

from __future__ import annotations


QUERIES_CSV_VERSION: str = "1"


# Rubric-dimension vocabulary (the canonical set used in evaluation/
# personalization_rubric.py plus per-task-family additions from e2–e6).
# Kept here as documentation only — not enforced at runtime.
_RUBRIC_CORE = (
    "preference_alignment",
    "avoid_leak",            # hard-fail on app-native / third-party-visible tasks
    "privacy_leak",          # hard-fail on app-native; demoted to response_respectfulness on chatbot_routed
    "response_respectfulness",
    "over_personalization",
    "stale_preference_use",
    "relationship_aware",
    "voice_match",
    "restraint",
    "tool_call_rules",
    "final_state_diff",
    "behavioral_hit",
    "temporal_boundedness",
)
_RUBRIC_E6 = (
    "mistake_prevention_recall",
    "false_alarm_emission",
    "cross_signal_attribution",
    "actionable_specificity",
    "local_context_awareness",
    "intervention_minimality",
    "warning_respectfulness",
)


DEFAULT_META: dict = {
    "task_family": "unknown",
    "mcp_tools_allowed": "none",
    "state_write_policy": "read_only",
    "expected_response_kind": "text",
    "rubric_tags": [],
}


TASK_TYPE_META: dict[str, dict] = {
    # ------------------------------------------------------------------
    # Task A — slate ranking
    # ------------------------------------------------------------------
    "slate_ranking": {
        "task_family": "slate_ranking",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": [
            "preference_alignment", "avoid_leak", "privacy_leak",
            "over_personalization", "stale_preference_use",
            "behavioral_hit",
        ],
    },

    # ------------------------------------------------------------------
    # Task B — chatbot response (two arms)
    # ------------------------------------------------------------------
    "chatbot_response_proactive": {
        "task_family": "chatbot_response",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "text",
        "rubric_tags": [
            "preference_alignment", "avoid_leak", "over_personalization",
            "stale_preference_use", "response_respectfulness",
        ],
    },
    "chatbot_response_control": {
        "task_family": "chatbot_response",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "text",
        "rubric_tags": [
            "restraint", "avoid_leak", "over_personalization",
            "response_respectfulness",
        ],
    },

    # ------------------------------------------------------------------
    # Task C — over-personalization probes
    # ------------------------------------------------------------------
    "c1a_pairs": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": ["over_personalization"],
    },
    "c1b_sequences": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": ["over_personalization"],
    },
    "c2_scenarios": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "text",
        "rubric_tags": [
            "restraint", "avoid_leak", "privacy_leak",
            "over_personalization", "relationship_aware",
        ],
    },
    "c3_restraint": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": ["privacy_leak", "over_personalization"],
    },
    "c4_button_regen": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": ["privacy_leak", "over_personalization"],
    },

    # ------------------------------------------------------------------
    # Agentic T6–T19
    # Write-policy rule: any task whose canonical tool-call rules require
    # a `create_post` / `send_dm` count > 0 is writes_ok. Pure audit /
    # read-only surfaces are read_only.
    # ------------------------------------------------------------------
    "t6_community_digest": {
        "task_family": "agentic",
        "mcp_tools_allowed": "social",
        "state_write_policy": "writes_ok",       # exactly 1 create_post
        "expected_response_kind": "agentic_writes",
        "rubric_tags": [
            "preference_alignment", "avoid_leak", "privacy_leak",
            "over_personalization", "stale_preference_use",
            "voice_match", "tool_call_rules", "final_state_diff",
            "behavioral_hit",
        ],
    },
    "t7_moment_recommendation": {
        "task_family": "agentic",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",        # no DM sends
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "preference_alignment", "avoid_leak", "privacy_leak",
            "over_personalization", "stale_preference_use",
            "temporal_boundedness", "behavioral_hit",
        ],
    },
    "t8_dm_digest": {
        "task_family": "agentic",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",        # list_dms + no sends
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "preference_alignment", "avoid_leak", "privacy_leak",
            "over_personalization", "relationship_aware",
            "tool_call_rules",
        ],
    },
    "t9_cross_app_repost": {
        "task_family": "agentic",
        "mcp_tools_allowed": "social",
        "state_write_policy": "writes_ok",        # exactly 1 threads_create_post
        "expected_response_kind": "agentic_writes",
        "rubric_tags": [
            "preference_alignment", "avoid_leak", "privacy_leak",
            "over_personalization", "voice_match",
            "tool_call_rules", "final_state_diff", "behavioral_hit",
        ],
    },
    "t10_auto_reply": {
        "task_family": "agentic",
        "mcp_tools_allowed": "social",
        "state_write_policy": "writes_ok",        # exactly 1 send_dm
        "expected_response_kind": "agentic_writes",
        "rubric_tags": [
            "preference_alignment", "privacy_leak",
            "over_personalization", "relationship_aware", "voice_match",
            "tool_call_rules", "final_state_diff",
        ],
    },
    "t11_vague_refind": {
        "task_family": "agentic",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",        # zero create_post
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "preference_alignment", "privacy_leak",
            "stale_preference_use", "tool_call_rules", "behavioral_hit",
        ],
    },
    "t12_agent_composed_post": {
        "task_family": "agentic",
        "mcp_tools_allowed": "social",
        "state_write_policy": "writes_ok",        # exactly 1 create_post per app
        "expected_response_kind": "agentic_writes",
        "rubric_tags": [
            "preference_alignment", "avoid_leak", "privacy_leak",
            "over_personalization", "stale_preference_use",
            "voice_match", "tool_call_rules", "final_state_diff",
        ],
    },
    "t13_chatbot_dispatch": {
        "task_family": "agentic",
        "mcp_tools_allowed": "all",                # chatbot + target social app
        "state_write_policy": "writes_ok",         # 1 create_post on target
        "expected_response_kind": "agentic_writes",
        "rubric_tags": [
            "preference_alignment", "avoid_leak", "privacy_leak",
            "over_personalization", "voice_match",
            "tool_call_rules", "final_state_diff",
        ],
    },
    "t14_draft_audit": {
        "task_family": "agentic",
        "mcp_tools_allowed": "social",
        "state_write_policy": "read_only",          # audit only — zero writes
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "privacy_leak", "over_personalization",
            "stale_preference_use", "tool_call_rules",
        ],
    },
    "t15_collection_curation": {
        "task_family": "agentic",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "preference_alignment", "privacy_leak",
            "over_personalization", "stale_preference_use",
            "behavioral_hit",
        ],
    },
    "t16_group_dm_summary": {
        "task_family": "agentic",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",          # get_dm_thread only
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "preference_alignment", "avoid_leak", "privacy_leak",
            "over_personalization", "relationship_aware",
            "tool_call_rules",
        ],
    },
    "t17_wrong_recipient": {
        "task_family": "agentic",
        "mcp_tools_allowed": "social",
        "state_write_policy": "writes_ok",          # ≤1 send_dm, must ask first
        "expected_response_kind": "agentic_writes",
        "rubric_tags": [
            "preference_alignment", "privacy_leak",
            "relationship_aware", "tool_call_rules", "final_state_diff",
        ],
    },
    "t18_proactive_daily": {
        "task_family": "agentic",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "preference_alignment", "avoid_leak", "privacy_leak",
            "over_personalization", "stale_preference_use",
            "temporal_boundedness", "behavioral_hit",
        ],
    },
    "t19_trending_alert": {
        "task_family": "agentic",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "preference_alignment", "avoid_leak",
            "over_personalization", "behavioral_hit",
        ],
    },

    # ------------------------------------------------------------------
    # E-family (E2–E6)
    # ------------------------------------------------------------------
    "e2_at_ai_followup": {
        "task_family": "e_followup",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": ["preference_alignment", "stale_preference_use"],
    },
    "e3_daily_briefing_multi": {
        "task_family": "e_followup",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "preference_alignment", "avoid_leak", "privacy_leak",
            "over_personalization", "stale_preference_use",
            "temporal_boundedness", "behavioral_hit",
        ],
    },
    "e4_google_search": {
        "task_family": "e_followup",
        "mcp_tools_allowed": "all",                 # includes google_search MCP
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": ["preference_alignment", "over_personalization"],
    },
    "e5_horizon_lifecycle": {
        "task_family": "e_followup",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": [
            "preference_alignment", "stale_preference_use",
            "temporal_boundedness",
        ],
    },
    "e6_active_mistake_prevention": {
        "task_family": "e_followup",
        "mcp_tools_allowed": "all",
        "state_write_policy": "writes_ok",
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": list(_RUBRIC_E6) + [
            "avoid_leak", "response_respectfulness",
        ],
    },
}


def get_meta(task_type: str) -> dict:
    """Look up metadata for a task_type, falling back to DEFAULT_META."""
    return TASK_TYPE_META.get(task_type, DEFAULT_META)
