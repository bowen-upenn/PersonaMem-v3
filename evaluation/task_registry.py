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


QUERIES_CSV_VERSION: str = "2"


# Rename map for the v1 → v2 task-type taxonomy. Kept here so old benchmarks
# / saved results can be translated by callers that need backwards compatibility
# (the runner refuses CSVs whose version header doesn't match QUERIES_CSV_VERSION,
# so the only legit consumer is `scripts/migrate_results_csv_v1_to_v2.py`).
OLD_TO_NEW: dict[str, str] = {
    "slate_ranking":                 "personalized_feed_ranking",
    "chatbot_response_proactive":    "chatbot_proactive_personalization",
    "chatbot_response_control":      "chatbot_restraint_control",
    "c1a_pairs":                     "repetition_fatigue_pairs",
    "c1b_sequences":                 "repetition_fatigue_sequences",
    "c2_scenarios":                  "context_shift_scenarios",
    "c3_restraint":                  "irrelevant_query_restraint",
    "c4_button_regen":               "preference_removal_regen",
    "e2_at_ai_followup":             "at_ai_directive_followup",
    "e3_daily_briefing_multi":       "daily_personalized_briefing",
    "e4_google_search":              "personalized_search_ranking",
    "e5_horizon_lifecycle":          "short_vs_long_term_lifecycle",
    "e6_active_mistake_prevention":  "active_mistake_prevention",
    "t6_community_digest":           "agentic_community_digest",
    "t7_moment_recommendation":      "agentic_moment_recommendation",
    "t8_dm_digest":                  "agentic_dm_digest",
    "t9_cross_app_repost":           "agentic_cross_app_repost",
    "t10_auto_reply":                "agentic_auto_reply",
    "t11_vague_refind":              "agentic_vague_refind",
    "t12_agent_composed_post":       "agentic_composed_post",
    "t13_chatbot_dispatch":          "agentic_chatbot_dispatch",
    "t14_draft_audit":               "agentic_draft_audit",
    "t15_collection_curation":       "agentic_collection_curation",
    "t16_group_dm_summary":          "agentic_group_dm_summary",
    "t17_wrong_recipient":           "agentic_wrong_recipient_check",
    "t18_proactive_daily":           "agentic_proactive_daily_catchup",
    "t19_trending_alert":            "agentic_trending_alert",
}


def normalize_task_type(name: str) -> str:
    """Translate a possibly-old task_type to the canonical v2 name."""
    return OLD_TO_NEW.get(name, name)


# Rubric-dimension vocabulary (the canonical set used in evaluation/
# personalization_rubric.py plus per-task-family additions from e2–e6).
# Kept here as documentation only — not enforced at runtime.
_RUBRIC_CORE = (
    "preference_alignment",
    "avoid_leak",            # hard-fail on app-native / third-party-visible tasks
    "privacy_leak",          # hard-fail on app-native; demoted to response_respectfulness on chatbot_routed
    "response_respectfulness",
    "over_personalization",
    "subtle_personalization",
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
    # Personalized feed ranking (was: slate_ranking)
    # ------------------------------------------------------------------
    "personalized_feed_ranking": {
        "task_family": "personalized_feed_ranking",
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
    # Chatbot response (two arms — was: chatbot_response_proactive / _control)
    # ------------------------------------------------------------------
    "chatbot_proactive_personalization": {
        "task_family": "chatbot_response",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "text",
        "rubric_tags": [
            "preference_alignment", "avoid_leak", "over_personalization",
            "subtle_personalization", "stale_preference_use",
            "response_respectfulness",
        ],
    },
    "chatbot_restraint_control": {
        "task_family": "chatbot_response",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "text",
        "rubric_tags": [
            "restraint", "avoid_leak", "over_personalization",
            "subtle_personalization", "response_respectfulness",
        ],
    },

    # ------------------------------------------------------------------
    # Restraint / over-personalization probes (was: c1a/c1b/c2/c3/c4)
    # ------------------------------------------------------------------
    "repetition_fatigue_pairs": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": ["over_personalization"],
    },
    "repetition_fatigue_sequences": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": ["over_personalization"],
    },
    "context_shift_scenarios": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "text",
        "rubric_tags": [
            "restraint", "avoid_leak", "privacy_leak",
            "over_personalization", "relationship_aware",
        ],
    },
    "irrelevant_query_restraint": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": ["privacy_leak", "over_personalization"],
    },
    "preference_removal_regen": {
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
    "agentic_community_digest": {
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
    "agentic_moment_recommendation": {
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
    "agentic_dm_digest": {
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
    "agentic_cross_app_repost": {
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
    "agentic_auto_reply": {
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
    "agentic_vague_refind": {
        "task_family": "agentic",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",        # zero create_post
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "preference_alignment", "privacy_leak",
            "stale_preference_use", "tool_call_rules", "behavioral_hit",
        ],
    },
    "agentic_composed_post": {
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
    "agentic_chatbot_dispatch": {
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
    "agentic_draft_audit": {
        "task_family": "agentic",
        "mcp_tools_allowed": "social",
        "state_write_policy": "read_only",          # audit only — zero writes
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "privacy_leak", "over_personalization",
            "stale_preference_use", "tool_call_rules",
        ],
    },
    "agentic_collection_curation": {
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
    "agentic_group_dm_summary": {
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
    "agentic_wrong_recipient_check": {
        "task_family": "agentic",
        "mcp_tools_allowed": "social",
        "state_write_policy": "writes_ok",          # ≤1 send_dm, must ask first
        "expected_response_kind": "agentic_writes",
        "rubric_tags": [
            "preference_alignment", "privacy_leak",
            "relationship_aware", "tool_call_rules", "final_state_diff",
        ],
    },
    "agentic_proactive_daily_catchup": {
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
    "agentic_trending_alert": {
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
    # E-family (was: e2/e3/e4/e5/e6)
    # ------------------------------------------------------------------
    "at_ai_directive_followup": {
        "task_family": "e_followup",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": ["preference_alignment", "stale_preference_use"],
    },
    "daily_personalized_briefing": {
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
    "personalized_search_ranking": {
        "task_family": "e_followup",
        "mcp_tools_allowed": "all",                 # includes google_search MCP
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": ["preference_alignment", "over_personalization"],
    },
    "short_vs_long_term_lifecycle": {
        "task_family": "e_followup",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": [
            "preference_alignment", "stale_preference_use",
            "temporal_boundedness",
        ],
    },
    "active_mistake_prevention": {
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


# ---------------------------------------------------------------------------
# Primary-metric registry — picks the headline accuracy per task for the
# token-vs-accuracy table emitted by `scripts/aggregate_eval.py`.
#
# Each entry is `(metric_key, kind)` where:
#   kind == "fraction"           -> accuracy_pct = 100 * value
#   kind == "inverted_fraction"  -> accuracy_pct = 100 * (1 - value)  (lower=better)
#   kind == "agentic_pass_rate"  -> computed in aggregator from
#       (tool_pass + final_state_passed + output_quality_passed) /
#       (those + their failures); rows with status="failed_writes"
#       or "failed_quality" count as 0.
#   kind == "paired_correct"     -> for active_mistake_prevention; computed
#       in aggregator's _paired_f1 helper from pair_id + polarity grouping.
# ---------------------------------------------------------------------------

PRIMARY_METRIC: dict[str, tuple[str, str]] = {
    # Ranking tasks — graded distractors, accuracy = top-1 match
    "personalized_feed_ranking":         ("accuracy", "fraction"),
    "at_ai_directive_followup":          ("hit@1", "fraction"),
    "personalized_search_ranking":       ("recall@1", "fraction"),
    "short_vs_long_term_lifecycle":      ("recall@1", "fraction"),
    "repetition_fatigue_pairs":          ("response_divergence", "fraction"),
    "repetition_fatigue_sequences":      ("preference_repetition_rate", "inverted_fraction"),
    # Chatbot response — held-out preference alignment for proactive arm,
    # restraint for control arm. Both metrics actually emitted by chatbot_response.py.
    "chatbot_proactive_personalization": ("held_out_score", "fraction"),
    "chatbot_restraint_control":         ("personalization_leak_rate", "inverted_fraction"),
    "context_shift_scenarios":           ("pr_personalization_hard_fail_count", "inverted_fraction"),
    "irrelevant_query_restraint":        ("irrelevant_rejection_precision", "fraction"),
    "preference_removal_regen":          ("removal_success", "fraction"),
    "daily_personalized_briefing":       ("has_structured_output", "fraction"),
    # E6 — paired warn/foil; aggregator computes paired-correct
    "active_mistake_prevention":         ("paired_correct", "paired_correct"),
    # Agentic — composite pass rate over tool_call + final_state + output_quality
    "agentic_community_digest":          ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_moment_recommendation":     ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_dm_digest":                 ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_cross_app_repost":          ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_auto_reply":                ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_vague_refind":              ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_composed_post":             ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_chatbot_dispatch":          ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_draft_audit":               ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_collection_curation":       ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_group_dm_summary":          ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_wrong_recipient_check":     ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_proactive_daily_catchup":   ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_trending_alert":            ("agentic_pass_rate", "agentic_pass_rate"),
}
