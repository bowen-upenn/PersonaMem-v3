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
    "slate_ranking":                 "personalized_recommendation",
    "personalized_feed_ranking":     "personalized_recommendation",
    "chatbot_response_proactive":    "chatbot_personalized_response",
    "chatbot_response_control":      "over_personalization_chatbot_text",
    "chatbot_restraint_control":     "over_personalization_chatbot_text",
    # Sensitive-event arm of run_task_b is graded under its own task_type so
    # the rubric APPLICABILITY entry / aggregator headline are isolated from
    # the generic chatbot restraint metric.
    "chatbot_response_sensitive_event": "over_personalization_sensitive_event",
    "c1c_same_preference_cluster":   "over_personalization_repetition_recsys",
    "c1d_chatbot_same_pref_cluster": "over_personalization_repetition_chatbot",
    "c1e_new_suggestions_recsys":    "new_suggestions_recsys",
    "c1f_new_suggestions_chatbot":   "new_suggestions_chatbot",
    "c2_scenarios":                  "over_personalization_context_shift",
    # Workstream cleanup: context_shift_scenarios was always part of the
    # over_personalization family; renamed so the membership is obvious
    # from the task_type alone.
    "context_shift_scenarios":       "over_personalization_context_shift",
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
    "t6_community_digest":           "agentic_user_tone_post",
    "agentic_community_digest":      "agentic_user_tone_post",
    # t7_moment_recommendation merged into personalized_recommendation
    # (slate-based ranking). Old CSV rows resolve to the new type so
    # aggregators still parse historical benchmarks.
    "t7_moment_recommendation":      "personalized_recommendation",
    "agentic_moment_recommendation": "personalized_recommendation",
    "t8_dm_digest":                  "agentic_dm_digest",
    "t9_cross_app_repost":           "agentic_cross_app_repost",
    "t10_auto_reply":                "agentic_auto_reply",
    "t11_vague_refind":              "agentic_vague_refind",
    "t12_agent_composed_post":       "agentic_composed_post",
    "t13_chatbot_dispatch":          "agentic_send_post",
    # agentic_draft_audit dropped — old strings still resolve so historical
    # CSVs parse, but the task type is no longer in TASK_TYPE_META.
    "t14_draft_audit":               "agentic_draft_audit",
    "t16_group_dm_summary":          "agentic_group_dm_summary",
    "t17_wrong_recipient":           "agentic_wrong_recipient_check",
    "t18_proactive_daily":           "agentic_proactive_daily_catchup",
    "t19_trending_alert":            "agentic_trending_alert",
    # Workstream D: e4 renamed to clarify it's social-media recommendation,
    # not Google search. Old strings still resolve.
    "personalized_search_ranking":   "personalized_recommendation",
    "e4_google_search":              "personalized_recommendation",
}

# Task types that have been removed entirely. Aggregators / runners
# should drop rows whose task_type lands in this set after
# `normalize_task_type`.
DROPPED_TASK_TYPES: set[str] = {
    "agentic_draft_audit",
}


def normalize_task_type(name: str) -> str:
    """Translate a possibly-old task_type to the canonical v2 name."""
    return OLD_TO_NEW.get(name, name)


# Rubric-dimension vocabulary — consolidated from the original 21 dims
# down to 12 after the workstream-A audit. See the plan in
# /vast/home/b/bwjiang/.claude/plans/ for the full mapping. Old
# strings (over_personalization, restraint, privacy_leak, …) are
# normalized via RUBRIC_ALIAS so saved CSVs stay readable.
_RUBRIC_CORE = (
    # LLM-judge dims
    "preference_alignment",        # was: + subtle_personalization
    "avoid_overpersonalization",   # was: over_personalization + restraint;
                                   # UNIVERSAL — every personalization task carries it
    "voice_match",                 # covers relationship_aware in DM tasks
    # Hard-rule dims
    "negative_leakage",            # was: avoid_leak — agent surfaced something
                                   # the user explicitly disliked recently
    "stale_preference_use",
    "behavioral_hit",
    "tool_call_match",             # was: tool_call_rules + final_state_diff;
                                   # the agent's tool-call sequence matches the
                                   # gold sequence — derived count + end-state checks
)
_RUBRIC_E6 = (
    "mistake_prevention_recall",
    "false_alarm_emission",
    "warning_quality",             # was: actionable_specificity +
                                   # local_context_awareness +
                                   # intervention_minimality + warning_respectfulness
)
_RUBRIC_RANKING = (
    # Deterministic ranking metrics for personalized_recommendation only
    "recall_at_k",
    "ndcg_at_k",
    "mrr",
    "hit_at_k",
)


# Old → new dimension names. Used by aggregator + audit so saved
# results.csv files remain readable after the consolidation.
RUBRIC_ALIAS: dict[str, str] = {
    "subtle_personalization":     "preference_alignment",
    "over_personalization":       "avoid_overpersonalization",
    "restraint":                  "avoid_overpersonalization",
    "relationship_aware":         "voice_match",
    "avoid_leak":                 "negative_leakage",
    "tool_call_rules":            "tool_call_match",
    "final_state_diff":           "tool_call_match",
    # The following had no real implementation — they map to a noop
    # so callers can detect-and-skip rather than crash.
    "privacy_leak":               "_removed_no_actual_private_data",
    "preference_respect":         "_removed_folded_into_negative_leakage",
    "response_respectfulness":    "_removed_no_implementation",
    "temporal_boundedness":       "stale_preference_use",
    "cross_signal_attribution":   "_removed_no_implementation",
    "actionable_specificity":     "warning_quality",
    "local_context_awareness":    "warning_quality",
    "intervention_minimality":    "warning_quality",
    "warning_respectfulness":     "warning_quality",
}


def normalize_rubric_tag(tag: str) -> str:
    """Map an old rubric-dim name to its current canonical name.
    Tags starting with `_removed_` indicate a dim that's been deleted
    entirely — callers should drop these from their per-row scoring."""
    return RUBRIC_ALIAS.get(tag, tag)


DEFAULT_META: dict = {
    "task_family": "unknown",
    "mcp_tools_allowed": "none",
    "state_write_policy": "read_only",
    "expected_response_kind": "text",
    "rubric_tags": [],
}


TASK_TYPE_META: dict[str, dict] = {
    # ------------------------------------------------------------------
    # Chatbot response (two arms — was: chatbot_response_proactive / _control)
    # ------------------------------------------------------------------
    "chatbot_personalized_response": {
        "task_family": "chatbot_response",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "text",
        "rubric_tags": [
            "preference_alignment", "avoid_overpersonalization",
            "negative_leakage", "stale_preference_use",
        ],
    },
    "over_personalization_chatbot_text": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "text",
        "rubric_tags": [
            "avoid_overpersonalization", "negative_leakage",
        ],
    },

    # ------------------------------------------------------------------
    # Over-personalization probes
    # ------------------------------------------------------------------
    "over_personalization_repetition_recsys": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "freeform",
        "rubric_tags": ["avoid_overpersonalization"],
    },
    "over_personalization_repetition_chatbot": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "freeform",
        "rubric_tags": ["avoid_overpersonalization"],
    },
    # ------------------------------------------------------------------
    # New suggestions — explorative recommendation anchored on hidden
    # personas. Sibling to the over-personalization repetition family
    # but POSITIVE: agent must propose a fresh topic / item, not just
    # restrain. Three trigger patterns (post_fatigue / chatbot_ask /
    # at_ai_directive) and two flavors of gold (LLM-generated /
    # future-truth). Recsys = 16-item slate + recall@1; chatbot =
    # free-form recommendation + leak-set + LLM-judge alignment.
    # ------------------------------------------------------------------
    "new_suggestions_recsys": {
        "task_family": "new_suggestions",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": list(_RUBRIC_RANKING) + ["avoid_overpersonalization"],
    },
    "new_suggestions_chatbot": {
        "task_family": "new_suggestions",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "text",
        "rubric_tags": ["preference_alignment", "avoid_overpersonalization"],
    },
    "over_personalization_context_shift": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "text",
        "rubric_tags": [
            "avoid_overpersonalization", "negative_leakage", "voice_match",
        ],
    },
    "over_personalization_distractor_reject": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        # Phase I.3: converted from ranking to open-ended chatbot text response.
        "expected_response_kind": "text",
        "rubric_tags": ["avoid_overpersonalization"],
    },
    # R10: sensitive-life-event over-personalization. Driven by the synthetic
    # sensitive_life_event hidden persona (1–3 LLM-personalized episodes per
    # user, each with an active window). The agent gets a benign chatbot
    # query inside one event's active window; surfacing the sensitive topic
    # is a hard privacy_leak failure.
    "over_personalization_sensitive_event": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "text",
        "rubric_tags": ["avoid_overpersonalization"],
    },
    "preference_removal_regen": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": ["avoid_overpersonalization"],
    },

    # ------------------------------------------------------------------
    # Agentic T6–T19  (T14 agentic_draft_audit dropped per workstream F)
    # ------------------------------------------------------------------
    "agentic_user_tone_post": {
        "task_family": "agentic",
        "mcp_tools_allowed": "social",
        "state_write_policy": "writes_ok",       # exactly 1 create_post
        "expected_response_kind": "agentic_writes",
        "rubric_tags": [
            "preference_alignment", "avoid_overpersonalization",
            "negative_leakage", "stale_preference_use", "voice_match",
            "tool_call_match", "behavioral_hit",
        ],
    },
    # agentic_moment_recommendation removed — merged into
    # personalized_recommendation (slate-based ranking).
    "agentic_dm_digest": {
        "task_family": "agentic",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",        # list_dms + no sends
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "preference_alignment", "avoid_overpersonalization",
            "voice_match", "tool_call_match",
        ],
    },
    "agentic_cross_app_repost": {
        "task_family": "agentic",
        "mcp_tools_allowed": "social",
        "state_write_policy": "writes_ok",        # exactly 1 threads_create_post
        "expected_response_kind": "agentic_writes",
        "rubric_tags": [
            "preference_alignment", "avoid_overpersonalization",
            "voice_match", "tool_call_match", "behavioral_hit",
        ],
    },
    "agentic_auto_reply": {
        "task_family": "agentic",
        "mcp_tools_allowed": "social",
        "state_write_policy": "writes_ok",        # exactly 1 send_dm
        "expected_response_kind": "agentic_writes",
        "rubric_tags": [
            "preference_alignment", "avoid_overpersonalization",
            "voice_match", "tool_call_match",
        ],
    },
    "agentic_vague_refind": {
        "task_family": "agentic",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",        # zero create_post
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "preference_alignment", "stale_preference_use",
            "tool_call_match", "behavioral_hit",
        ],
    },
    "agentic_composed_post": {
        "task_family": "agentic",
        "mcp_tools_allowed": "social",
        "state_write_policy": "writes_ok",        # exactly 1 create_post per app
        "expected_response_kind": "agentic_writes",
        "rubric_tags": [
            "preference_alignment", "avoid_overpersonalization",
            "negative_leakage", "stale_preference_use", "voice_match",
            "tool_call_match",
        ],
    },
    "agentic_send_post": {
        "task_family": "agentic",
        "mcp_tools_allowed": "all",                # chatbot + target social app
        "state_write_policy": "writes_ok",         # 1 create_post on target
        "expected_response_kind": "agentic_writes",
        "rubric_tags": [
            "preference_alignment", "avoid_overpersonalization",
            "voice_match", "tool_call_match",
        ],
    },
    # agentic_draft_audit removed — too subjective for benchmark grading
    "agentic_group_dm_summary": {
        "task_family": "agentic",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",          # get_dm_thread only
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "preference_alignment", "avoid_overpersonalization",
            "voice_match", "tool_call_match",
        ],
    },
    "agentic_wrong_recipient_check": {
        "task_family": "agentic",
        "mcp_tools_allowed": "social",
        "state_write_policy": "writes_ok",          # ≤1 send_dm, must ask first
        "expected_response_kind": "agentic_writes",
        "rubric_tags": [
            "preference_alignment", "voice_match", "tool_call_match",
        ],
    },
    "agentic_proactive_daily_catchup": {
        "task_family": "agentic",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "preference_alignment", "avoid_overpersonalization",
            "negative_leakage", "stale_preference_use", "behavioral_hit",
        ],
    },
    "agentic_trending_alert": {
        "task_family": "agentic",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "preference_alignment", "avoid_overpersonalization",
            "behavioral_hit",
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
            "preference_alignment", "avoid_overpersonalization",
            "negative_leakage", "stale_preference_use", "behavioral_hit",
        ],
    },
    # personalized_search_ranking renamed → personalized_recommendation
    # (workstream D). Old name still resolved via OLD_TO_NEW.
    "personalized_recommendation": {
        "task_family": "e_followup",
        "mcp_tools_allowed": "none",                # ranks the slate from time-masked history alone
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": list(_RUBRIC_RANKING),       # recall_at_k, ndcg_at_k, mrr, hit_at_k
    },
    "short_vs_long_term_lifecycle": {
        "task_family": "e_followup",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "rubric_tags": [
            "preference_alignment", "stale_preference_use",
        ],
    },
    "active_mistake_prevention": {
        "task_family": "e_followup",
        "mcp_tools_allowed": "all",
        "state_write_policy": "writes_ok",
        "expected_response_kind": "text_with_tool_calls",
        # Task-specific axes (mistake_prevention_recall, false_alarm_emission,
        # warning_quality) PLUS the universal personalization dimensions used
        # by chatbot Q&A, over-personalization, and proactive_actions. Same
        # pattern as yuan's 98a33c1: task-specific + universal in one bundle.
        # Polarity (warn vs foil) is carried by `polarity` on the instance,
        # not by tag presence.
        "rubric_tags": list(_RUBRIC_E6) + [
            "preference_alignment",
            "voice_match",
            "negative_leakage",
            "stale_preference_use",
        ],
    },
    # Silent geo-shift local recommendation. The agent must infer the user's
    # current city from the most-recent `event_location.city` in their
    # time-masked history (no city named in the user's query) and produce a
    # local recommendation that's grounded in the *current* city while still
    # aligning with the user's general persona profile. Inferior response
    # = anchoring on the prior/home city. Sits in `e_followup` because it's a
    # cross-cutting context-grounding probe, not a restraint test.
    "local_recommendation_geo_shift": {
        "task_family": "e_followup",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "text",
        "rubric_tags": ["preference_alignment", "stale_preference_use"],
    },

    # ------------------------------------------------------------------
    # Proactive Actions (Phase 1) — agent decides whether to initiate
    # contact at a moment the user did NOT explicitly open. Three trigger
    # types; agent surfaces only inside the chatbot (subtlety constraint).
    # See plan: /lcars/home/y/yyuan86/.claude/plans/rippling-honking-donut.md
    # ------------------------------------------------------------------
    "proactive_unfulfilled_stated_need": {
        "task_family": "proactive_actions",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "trigger_detection_correctness",
            "preference_alignment", "avoid_overpersonalization", "voice_match",
            "negative_leakage", "stale_preference_use",
        ],
    },
    "proactive_close_friend_update": {
        "task_family": "proactive_actions",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "trigger_detection_correctness",
            "preference_alignment", "avoid_overpersonalization", "voice_match",
            "negative_leakage", "stale_preference_use",
        ],
    },
    "restraint_sensitive_event_silence": {
        "task_family": "proactive_actions",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "trigger_detection_correctness",
            "preference_alignment", "avoid_overpersonalization", "voice_match",
            "negative_leakage", "stale_preference_use",
        ],
    },
    # Phase 2 — feed-react tasks (friend self-posts + platform trending)
    # plus the overactive-check negative control. Same prompt + grader as
    # the Phase 1 proactive entries; only the data source differs.
    "proactive_friend_feed_react": {
        "task_family": "proactive_actions",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "trigger_detection_correctness",
            "preference_alignment", "avoid_overpersonalization", "voice_match",
            "negative_leakage", "stale_preference_use",
        ],
    },
    "proactive_trending_feed_react": {
        "task_family": "proactive_actions",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "trigger_detection_correctness",
            "preference_alignment", "avoid_overpersonalization", "voice_match",
            "negative_leakage", "stale_preference_use",
        ],
    },
    "proactive_overactive_check": {
        "task_family": "proactive_actions",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",
        "expected_response_kind": "text_with_tool_calls",
        "rubric_tags": [
            "trigger_detection_correctness",
            "preference_alignment", "avoid_overpersonalization", "voice_match",
            "negative_leakage", "stale_preference_use",
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
    "chatbot_personalized_response":           "user_query",
    "over_personalization_chatbot_text":       "user_query",
    "over_personalization_repetition_recsys":  "user_query",
    "over_personalization_repetition_chatbot": "user_query",
    # new_suggestions: post_fatigue trigger has no explicit user query (the
    # framing is "user is saturated, recsys must pivot"); chatbot_ask /
    # at_ai_directive triggers carry an explicit user message. Classify the
    # tasks by their default surface — both are user-facing.
    "new_suggestions_recsys":                  "proactive_recommendation",
    "new_suggestions_chatbot":                 "user_query",
    "over_personalization_context_shift":                "user_query",
    "over_personalization_distractor_reject": "user_query",
    "over_personalization_sensitive_event":   "user_query",
    "preference_removal_regen":               "user_query",
    "at_ai_directive_followup":               "user_query",
    "daily_personalized_briefing":            "proactive_recommendation",
    # personalized_recommendation (renamed from personalized_search_ranking)
    # carries an empty query_text — system-side recommendation surface, no
    # live user message.
    "personalized_recommendation":            "proactive_recommendation",
    "short_vs_long_term_lifecycle":           "proactive_recommendation",
    "active_mistake_prevention":              "proactive_assistance",
    "local_recommendation_geo_shift":         "user_query",
    "agentic_user_tone_post":                "agentic_task",
    # agentic_moment_recommendation removed (merged into personalized_recommendation)
    "agentic_dm_digest":                      "agentic_task",
    "agentic_cross_app_repost":               "agentic_task",
    "agentic_auto_reply":                     "agentic_task",
    "agentic_vague_refind":                   "user_query",
    "agentic_composed_post":                  "agentic_task",
    "agentic_send_post":                      "user_query",
    # agentic_draft_audit removed — workstream F.
    "agentic_group_dm_summary":               "agentic_task",
    "agentic_wrong_recipient_check":          "proactive_assistance",
    "agentic_proactive_daily_catchup":        "proactive_recommendation",
    "agentic_trending_alert":                 "proactive_recommendation",
    # Proactive Actions (Phase 1) — system decides whether to initiate.
    "proactive_unfulfilled_stated_need":      "proactive_assistance",
    "proactive_close_friend_update":          "proactive_assistance",
    "restraint_sensitive_event_silence":      "proactive_assistance",
}


EXPECTED_BEHAVIOR_BY_TASK: dict[str, str] = {
    "chatbot_personalized_response":          "personalize",
    "over_personalization_chatbot_text":       "avoid_overpersonalization",
    "over_personalization_repetition_recsys":  "avoid_overpersonalization",
    "over_personalization_repetition_chatbot": "avoid_overpersonalization",
    "new_suggestions_recsys":                  "proactive_recommend",
    "new_suggestions_chatbot":                 "proactive_recommend",
    "over_personalization_context_shift":                "avoid_overpersonalization",
    "over_personalization_distractor_reject": "avoid_overpersonalization",
    "over_personalization_sensitive_event":   "avoid_overpersonalization",
    "preference_removal_regen":               "avoid_overpersonalization",
    "at_ai_directive_followup":               "proactive_recommend",
    "daily_personalized_briefing":            "proactive_recommend",
    "personalized_recommendation":            "proactive_recommend",
    "short_vs_long_term_lifecycle":           "proactive_recommend",
    "active_mistake_prevention":              "proactive_assist",
    "local_recommendation_geo_shift":         "personalize",
    "agentic_user_tone_post":                "agentic_action",
    # agentic_moment_recommendation removed (merged into personalized_recommendation)
    "agentic_dm_digest":                      "agentic_action",
    "agentic_cross_app_repost":               "agentic_action",
    "agentic_auto_reply":                     "agentic_action",
    "agentic_vague_refind":                   "agentic_action",
    "agentic_composed_post":                  "agentic_action",
    "agentic_send_post":                      "agentic_action",
    "agentic_group_dm_summary":               "agentic_action",
    "agentic_wrong_recipient_check":          "proactive_assist",
    "agentic_proactive_daily_catchup":        "proactive_recommend",
    "agentic_trending_alert":                 "proactive_recommend",
    # Proactive Actions: act on user evidence, OR stay silent (restraint).
    "proactive_unfulfilled_stated_need":      "proactive_assist",
    "proactive_close_friend_update":          "proactive_assist",
    "restraint_sensitive_event_silence":      "restrain",
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
    # Phase L.B.1: blended hit@1 + judge intent-alignment. Falls back to
    # hit@1 alone when judge is disabled (directive_score key absent).
    "at_ai_directive_followup":          ("directive_score", "fraction"),
    # Phase L.B.3: real personalization scorer — top-3 result alignment with
    # the user's recent_pref_summary. Was previously `recall@1` against an
    # absent ground-truth (no scorer existed; metric was never populated).
    "personalized_search_ranking":       ("top3_alignment_rate", "fraction"),
    "short_vs_long_term_lifecycle":      ("recall@1", "fraction"),
    # Silent geo-shift probe — composite headline that combines the
    # current-city / prior-city / neutral branches into a 0/0.5/1 score.
    "local_recommendation_geo_shift":    ("geo_shift_correctness", "fraction"),
    "over_personalization_repetition_recsys":  ("tail_passed", "boolean"),
    "over_personalization_repetition_chatbot": ("tail_passed", "boolean"),
    # new_suggestions — recsys uses recall@1 (renamed `passed` so the
    # aggregator reads a uniform headline column); chatbot uses the
    # leak-set + judge composite `passed` flag emitted by the runner.
    "new_suggestions_recsys":                  ("passed", "fraction"),
    "new_suggestions_chatbot":                 ("passed", "fraction"),
    # Chatbot response — held-out preference alignment for proactive arm,
    # restraint for control arm. Both metrics actually emitted by chatbot_response.py.
    "chatbot_personalized_response":               ("held_out_score", "fraction"),
    "over_personalization_chatbot_text":           ("personalization_leak_rate", "inverted_fraction"),
    "over_personalization_context_shift":                     ("pr_personalization_hard_fail_count", "inverted_fraction"),
    # F1 over (precision, recall) — gameable-by-rejecting-nothing precision was
    # the headline before; F1 punishes both always-accept and always-reject.
    # Phase I.3: now an open-ended chatbot task — graded by leak rate
    # (lower personalization_leak_rate = better restraint).
    "over_personalization_distractor_reject":      ("personalization_leak_rate", "inverted_fraction"),
    # Same headline metric as the distractor-reject task: lower leak rate
    # = better restraint around the user's private/sensitive episode.
    "over_personalization_sensitive_event":        ("personalization_leak_rate", "inverted_fraction"),
    "preference_removal_regen":          ("removal_success", "fraction"),
    # Phase L.B.2: real personalization metric — jaccard(briefing topics,
    # user's prior-24h top hashtags). Was just `has_structured_output` (yes/no
    # JSON), which any non-empty response trivially passed.
    "daily_personalized_briefing":       ("briefing_personalization_score", "fraction"),
    # E6 — paired warn/foil; aggregator computes paired-correct
    "active_mistake_prevention":         ("paired_correct", "paired_correct"),
    # Agentic — composite pass rate over tool_call + final_state + output_quality
    "agentic_user_tone_post":           ("agentic_pass_rate", "agentic_pass_rate"),
    # agentic_moment_recommendation removed (merged into personalized_recommendation)
    "agentic_dm_digest":                 ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_cross_app_repost":          ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_auto_reply":                ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_vague_refind":              ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_composed_post":             ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_send_post":                 ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_draft_audit":               ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_group_dm_summary":          ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_wrong_recipient_check":     ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_proactive_daily_catchup":   ("agentic_pass_rate", "agentic_pass_rate"),
    "agentic_trending_alert":            ("agentic_pass_rate", "agentic_pass_rate"),
    # Proactive Actions (Phase 1): composite proactive_action_score in [0,1]
    # produced by judge_proactive_action averaged across the 5 rubric dims.
    "proactive_unfulfilled_stated_need": ("proactive_action_score", "fraction"),
    "proactive_close_friend_update":     ("proactive_action_score", "fraction"),
    "restraint_sensitive_event_silence": ("proactive_action_score", "fraction"),
}
