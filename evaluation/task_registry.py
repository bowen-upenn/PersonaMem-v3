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


# Rename map for the v1 → v2 → v3 task-type taxonomy. Kept here so old
# benchmarks / saved results can be translated by callers that need
# backwards compatibility (the runner refuses CSVs whose version header
# doesn't match QUERIES_CSV_VERSION, so the only legit consumer is
# `scripts/migrate_results_csv_v1_to_v2.py`). v3 unified the two
# over-personalization tasks (chatbot_restraint_control,
# irrelevant_query_restraint) under the over_personalization_* prefix so
# their family is obvious from the task_type alone — the aggregator
# headline number was being read as "100 % restraint" when in fact the
# tasks were testing the same capability with different surfaces.
OLD_TO_NEW: dict[str, str] = {
    "slate_ranking":                 "personalized_feed_ranking",
    "chatbot_response_proactive":    "chatbot_proactive_personalization",
    "chatbot_response_control":      "over_personalization_chatbot_text",
    "chatbot_restraint_control":     "over_personalization_chatbot_text",
    "c1a_pairs":                     "repetition_fatigue_pairs",
    "c1b_sequences":                 "repetition_fatigue_sequences",
    "c2_scenarios":                  "context_shift_scenarios",
    "c3_restraint":                  "over_personalization_distractor_reject",
    "irrelevant_query_restraint":    "over_personalization_distractor_reject",
    "c4_button_regen":               "preference_removal_regen",
    "e2_at_ai_followup":             "at_ai_directive_followup",
    "e3_daily_briefing_multi":       "daily_personalized_briefing",
    "e4_google_search":              "personalized_search_ranking",
    "e5_horizon_lifecycle":          "short_vs_long_term_lifecycle",
    "e6_active_mistake_prevention":  "active_mistake_prevention",
    # Phase L.B.4: renamed — task is "compose an advisory post in the user's
    # voice", not a community trends summary. Old aliases keep working.
    "t6_community_digest":           "agentic_user_voice_post",
    "agentic_community_digest":      "agentic_user_voice_post",
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
    "over_personalization_chatbot_text": {
        "task_family": "over_personalization",
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
    "over_personalization_distractor_reject": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        # Phase I.3: converted from ranking to open-ended chatbot text response.
        "expected_response_kind": "text",
        "rubric_tags": ["privacy_leak", "over_personalization", "restraint"],
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
    "agentic_user_voice_post": {
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
# Audit-side classification: how each task_type appears to the user and what
# the agent is supposed to do. Kept separate from TASK_TYPE_META so the
# dispatch fields stay focused. Used by `data_preparation/visualize.py`'s
# test.json dump and by `scripts/audit_test_queries.py`.
#
# query_kind:
#   user_query             — there is a literal user-typed message
#   agentic_task           — task-driven (no user message; trigger context)
#   proactive_recommendation — system pushes content (no user query)
#   proactive_assistance   — system intervenes (audit/warn/dispatch)
#
# expected_behavior:
#   personalize            — agent should weave in held-out preference
#   restrain               — agent should NOT surface user-specific context
#   proactive_recommend    — agent picks/ranks content for the user
#   proactive_assist       — agent flags/asks/audits before user acts
#   agentic_action         — agent executes a tool-call sequence
# ---------------------------------------------------------------------------

QUERY_KIND_BY_TASK: dict[str, str] = {
    "personalized_feed_ranking":              "proactive_recommendation",
    "chatbot_proactive_personalization":      "user_query",
    "over_personalization_chatbot_text":      "user_query",
    "repetition_fatigue_pairs":               "user_query",
    "repetition_fatigue_sequences":           "user_query",
    "context_shift_scenarios":                "user_query",
    "over_personalization_distractor_reject": "user_query",
    "preference_removal_regen":               "user_query",
    "at_ai_directive_followup":               "user_query",
    "daily_personalized_briefing":            "proactive_recommendation",
    # personalized_search_ranking carries no literal user query in the
    # current builder — the test asks the agent to rank candidate search
    # results given the user's recent prefs. It's a system-side
    # recommendation surface, not a user-typed query.
    "personalized_search_ranking":            "proactive_recommendation",
    "short_vs_long_term_lifecycle":           "proactive_recommendation",
    "active_mistake_prevention":              "proactive_assistance",
    "agentic_user_voice_post":                "agentic_task",
    "agentic_moment_recommendation":          "agentic_task",
    "agentic_dm_digest":                      "agentic_task",
    "agentic_cross_app_repost":               "agentic_task",
    "agentic_auto_reply":                     "agentic_task",
    "agentic_vague_refind":                   "user_query",
    "agentic_composed_post":                  "agentic_task",
    "agentic_chatbot_dispatch":               "user_query",
    "agentic_draft_audit":                    "proactive_assistance",
    "agentic_collection_curation":            "agentic_task",
    "agentic_group_dm_summary":               "agentic_task",
    "agentic_wrong_recipient_check":          "proactive_assistance",
    "agentic_proactive_daily_catchup":        "proactive_recommendation",
    "agentic_trending_alert":                 "proactive_recommendation",
}


EXPECTED_BEHAVIOR_BY_TASK: dict[str, str] = {
    "personalized_feed_ranking":              "proactive_recommend",
    "chatbot_proactive_personalization":      "personalize",
    "over_personalization_chatbot_text":      "restrain",
    "repetition_fatigue_pairs":               "restrain",
    "repetition_fatigue_sequences":           "restrain",
    "context_shift_scenarios":                "restrain",
    "over_personalization_distractor_reject": "restrain",
    "preference_removal_regen":               "restrain",
    "at_ai_directive_followup":               "proactive_recommend",
    "daily_personalized_briefing":            "proactive_recommend",
    "personalized_search_ranking":            "proactive_recommend",
    "short_vs_long_term_lifecycle":           "proactive_recommend",
    "active_mistake_prevention":              "proactive_assist",
    "agentic_user_voice_post":                "agentic_action",
    "agentic_moment_recommendation":          "agentic_action",
    "agentic_dm_digest":                      "agentic_action",
    "agentic_cross_app_repost":               "agentic_action",
    "agentic_auto_reply":                     "agentic_action",
    "agentic_vague_refind":                   "agentic_action",
    "agentic_composed_post":                  "agentic_action",
    "agentic_chatbot_dispatch":               "agentic_action",
    "agentic_draft_audit":                    "proactive_assist",
    "agentic_collection_curation":            "agentic_action",
    "agentic_group_dm_summary":               "agentic_action",
    "agentic_wrong_recipient_check":          "proactive_assist",
    "agentic_proactive_daily_catchup":        "proactive_recommend",
    "agentic_trending_alert":                 "proactive_recommend",
}


def get_query_kind(task_type: str) -> str:
    return QUERY_KIND_BY_TASK.get(task_type, "user_query")


def get_expected_behavior(task_type: str) -> str:
    return EXPECTED_BEHAVIOR_BY_TASK.get(task_type, "personalize")


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
    # Phase L.B.1: blended hit@1 + judge intent-alignment. Falls back to
    # hit@1 alone when judge is disabled (directive_score key absent).
    "at_ai_directive_followup":          ("directive_score", "fraction"),
    # Phase L.B.3: real personalization scorer — top-3 result alignment with
    # the user's recent_pref_summary. Was previously `recall@1` against an
    # absent ground-truth (no scorer existed; metric was never populated).
    "personalized_search_ranking":       ("top3_alignment_rate", "fraction"),
    "short_vs_long_term_lifecycle":      ("recall@1", "fraction"),
    "repetition_fatigue_pairs":          ("response_divergence", "fraction"),
    "repetition_fatigue_sequences":      ("preference_repetition_rate", "inverted_fraction"),
    # Chatbot response — held-out preference alignment for proactive arm,
    # restraint for control arm. Both metrics actually emitted by chatbot_response.py.
    "chatbot_proactive_personalization":           ("held_out_score", "fraction"),
    "over_personalization_chatbot_text":           ("personalization_leak_rate", "inverted_fraction"),
    "context_shift_scenarios":                     ("pr_personalization_hard_fail_count", "inverted_fraction"),
    # F1 over (precision, recall) — gameable-by-rejecting-nothing precision was
    # the headline before; F1 punishes both always-accept and always-reject.
    # Phase I.3: now an open-ended chatbot task — graded by leak rate
    # (lower personalization_leak_rate = better restraint).
    "over_personalization_distractor_reject":      ("personalization_leak_rate", "inverted_fraction"),
    "preference_removal_regen":          ("removal_success", "fraction"),
    # Phase L.B.2: real personalization metric — jaccard(briefing topics,
    # user's prior-24h top hashtags). Was just `has_structured_output` (yes/no
    # JSON), which any non-empty response trivially passed.
    "daily_personalized_briefing":       ("briefing_personalization_score", "fraction"),
    # E6 — paired warn/foil; aggregator computes paired-correct
    "active_mistake_prevention":         ("paired_correct", "paired_correct"),
    # Agentic — composite pass rate over tool_call + final_state + output_quality
    "agentic_user_voice_post":           ("agentic_pass_rate", "agentic_pass_rate"),
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
