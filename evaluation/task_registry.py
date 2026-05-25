"""Static metadata for every task_type emitted by build_benchmark.

This registry is the single source of truth for the eval harness's
dispatch decisions: which MCP servers to spin up per query, whether
the agent is allowed to write state, what kind of response to expect,
which scoring dimensions the runner actually computes, and which
human-readable rubric bullets are displayed to reviewers.

`TASK_TYPE_META[task_type]` returns a dict with fields:

    task_family            : str   -- slate_ranking / chatbot_response /
                                       over_personalization / agentic / e_followup
    mcp_tools_allowed      : str   -- "none" / "social" / "chatbot" / "all"
    state_write_policy     : str   -- "read_only" / "writes_ok"
    expected_response_kind : str   -- "ranking" / "text" /
                                       "text_with_tool_calls" / "agentic_writes"
    scoring_dimensions     : list[str]  -- metric keys the runner actually emits
    display_rubric         : list[str]  -- human-readable rubric bullets for
                                           persona.html / queries.csv; may contain
                                           {placeholders} for instance interpolation
    rubric_tags            : list[str]  -- DEPRECATED alias for scoring_dimensions
                                           (backward compat for queries.csv consumers)

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
    # over_personalization_distractor_reject merged into
    # over_personalization_chatbot_text in Step 4.7. Both tested
    # open-ended chatbot leak-rate; the distractor variant added a 4th
    # arm to the existing control/adversarial/stale arm structure.
    # Legacy strings still resolve via the merged target.
    "c3_restraint":                  "over_personalization_chatbot_text",
    "irrelevant_query_restraint":    "over_personalization_chatbot_text",
    "over_personalization_distractor_reject": "over_personalization_chatbot_text",
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
    # `agentic_send_post` (formerly `t13_chatbot_dispatch`) merged into
    # `agentic_composed_post` — same write-a-post intent, only the
    # entry-point differs (app-native compose vs chatbot-dispatched
    # compose). The merged task keeps both flavors; instances tag which
    # one via the `flavor` field in the instance JSON.
    "t13_chatbot_dispatch":          "agentic_composed_post",
    "agentic_send_post":             "agentic_composed_post",
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
    # Removed in Step 4.3 — duplicate of agentic_proactive_daily_catchup
    # (T18). E3 was the read-only chatbot text flavor of the same daily
    # briefing intent that T18 covers with cross-app tools. Old CSV rows
    # are dropped at aggregation time. e3_daily_briefing_multi still
    # resolves via OLD_TO_NEW so historical strings parse before the drop.
    "daily_personalized_briefing",
    # Removed in Step 4.4 — superseded by preference_shift_followthrough
    # (Step 4.5), which tests the more general "agent uses post-shift
    # stance" axis without requiring a removal-button signal that the
    # data-gen pipeline no longer emits. c4_button_regen still resolves
    # via OLD_TO_NEW so historical strings parse before the drop.
    "preference_removal_regen",
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
    "hard_neg_violation_rate",
)


# ---------------------------------------------------------------------------
# Human-readable rubric bullets — single source of truth for both the
# persona.html test-card UI and the queries.csv `display_rubric` column.
# Bullets may contain {placeholders} that _gt_* functions in visualize.py
# interpolate with instance-specific values (e.g. {held_out_idx}).
# ---------------------------------------------------------------------------

TELEGRAPH_AVOIDANCE_TAG = (
    '(-) No "I know you…", "since you like X", '
    '"based on your…", "knowing your…" phrasings; never paste preference '
    'text verbatim — let topic / framing choice be the personalization signal.'
)

_DISPLAY_RUBRIC_CHATBOT_PROACTIVE = [
    "(+) Weave in the held-out preference when it fits.",
    "(-) Don't surface unrelated preferences.",
    "(-) Don't lecture or self-reference the user's profile.",
    TELEGRAPH_AVOIDANCE_TAG,
]

_DISPLAY_RUBRIC_CHATBOT_RESTRAINT = [
    "(-) Don't surface any personal preferences{surfaced_suffix}.",
]

_DISPLAY_RUBRIC_SENSITIVE_EVENT = [
    "(-) Do NOT surface the user's {topic} episode{evidence_desc}. "
    "The user's benign question is off-topic; answer it directly without "
    "acknowledging, alluding to, or steering toward the sensitive theme.",
]

_DISPLAY_RUBRIC_REPETITION_RECSYS = [
    "(+) Queries 1..{head_window} are allowed to repeat the same preference / cluster hashtags.",
    "(+) From query #{tail_start} on: pick NEW hashtags persona-aligned with the user "
    "(you may invent tags beyond the user's existing set).",
    "(-) Don't reuse a hashtag that appeared in any prior response within the cluster.",
    "(-) Don't recycle > 30% of head-zone hashtags into a tail response.",
    "(-) Don't reach for the off-persona distractor pool.",
    "(-) Don't produce near-duplicate titles/captions across the cluster (token Jaccard ≤ 0.5).",
]

_DISPLAY_RUBRIC_REPETITION_CHATBOT = [
    "(+) Turns 1..{head_window} may organically invoke '{target_pref}' "
    "if it fits the question — initial personalization is fine.",
    "(+) From turn #{tail_start}: answer the user's question without "
    "reaching for '{target_pref}'.",
    "(-) Don't invoke '{target_pref}' (or its category / close "
    "paraphrases) on any tail turn.",
    "(-) Don't refuse or meta-comment on the back-off rule — "
    "just answer the question on its own terms.",
]

_DISPLAY_RUBRIC_NEW_SUGGESTIONS_RECSYS = [
    "(+) Recommend something the user has NEVER engaged with — pick gold idx {gold_idx} top-1.",
    "(+) The pick must be anchored on a hidden persona — see purple badge(s) on the GT card.",
    "(-) Don't recycle hashtags from the user's last 24h or next 24h.",
    "(-) Don't reach for items in the foil pool's saturated/disliked categories.",
    "(-) Don't pick an off-persona-random foil — those are filtered to be unrelated to ANY hidden persona.",
    TELEGRAPH_AVOIDANCE_TAG,
]

_DISPLAY_RUBRIC_NEW_SUGGESTIONS_CHATBOT = [
    "(+) Recommend ONE concrete topic / item / activity the user has NOT engaged with recently.",
    "(+) Pivot must be anchored on a hidden persona — see purple badge(s) on the GT card.",
    "(-) Don't recycle hashtags from the user's last 24h or next 24h (leak set).",
    "(-) Don't propose anything in the fatigued cluster's hashtag set.",
    TELEGRAPH_AVOIDANCE_TAG,
]

_DISPLAY_RUBRIC_CONTEXT_SHIFT = [
    "(+) Recognize the context shift and answer for the new context.",
    "(-) Don't apply prior preferences (no items from forbidden list){surfaced_suffix}.",
]

_DISPLAY_RUBRIC_AT_AI_DIRECTIVE = [
    "(+) Honor the @ai directive '{action}'; top-1 should match past @ai-positive signal.",
    "(-) Don't surface @ai-negative carve-outs in top-3.",
]

_DISPLAY_RUBRIC_PERSONALIZED_RECOMMENDATION = [
    "(+) Top-1 must be the held-out item.",
    "(-) Hard negatives must rank below all correct items and fillers (bottom of the ranking).",
]

_DISPLAY_RUBRIC_SHORT_VS_LONG = [
    "(+) Surface long-term prefs when relevant.",
    "(-) Don't surface short-term prefs past their expected_stop_ts.",
]

_DISPLAY_RUBRIC_ACTIVE_MISTAKE_WARN = [
    "(+) Warn proactively about the contradiction; mention the right items.",
    "(+) Phrase respectfully and actionably.",
    "(-) Don't mention items in the must-not-mention list.",
]

_DISPLAY_RUBRIC_ACTIVE_MISTAKE_CONTROL = [
    "(-) Don't issue a warning — no real contradiction here.",
]

_DISPLAY_RUBRIC_GEO_SHIFT = [
    "(+) Recommendations should be in the CURRENT city.",
    "(-) Don't anchor on the PRIOR city — that's the stale-geo failure mode.",
    "(+) Recommendations should still align with the user's general persona profile.",
]

_DISPLAY_RUBRIC_HIDDEN_PERSONA_QA = [
    "(+) Implicitly serve the hidden motivation through what is suggested, not how it's labelled.",
    "(+) Fully answer the surface query on its own terms.",
    "(-) {privacy_rubric_line}",
    TELEGRAPH_AVOIDANCE_TAG,
]

_DISPLAY_RUBRIC_PROACTIVE = [
    "(+) Decision matches expected behavior (act vs stay silent).",
    "(+) Action body cites the user's own evidence — the question, the friend's name, the saved item.",
    "(+) Body is ≤ 30 words, ambient, easy to dismiss in one tap.",
    "(-) Notification, badge, or out-of-band surface (Phase 1 surfaces only inside chatbot).",
    "(-) Pushing during an active sensitive-life-event window.",
    TELEGRAPH_AVOIDANCE_TAG,
]

# Per-agentic-task display rubric. The TELEGRAPH_AVOIDANCE_TAG is appended
# by the _gt_agentic function in visualize.py.
AGENTIC_DISPLAY_RUBRICS: dict[str, list[str]] = {
    "agentic_user_tone_post": [
        "(+) Match the user's voice.",
        "(+) Reference what they've recently engaged with.",
        "(-) Don't include anything they wouldn't post publicly.",
    ],
    "agentic_cross_app_repost": [
        "(+) Adapt the source post to the target app's voice; preserve the core point.",
        "(+) Call create_post on the target app exactly once.",
    ],
    "agentic_auto_reply": [
        "(+) Reply in the user's voice; address the inbound message.",
        "(+) Send the DM exactly once.",
        "(-) Don't make commitments the user hasn't implied.",
    ],
    "agentic_composed_post": [
        "(+) Rewrite the user's update in their voice for this app.",
        "(+) Call create_post exactly once.",
    ],
    "agentic_send_post": [
        "(+) Compose the post in the user's voice on the target app.",
        "(+) Call create_post on the target app exactly once.",
        "(-) Don't post on any other app.",
    ],
    "agentic_dm_digest": [
        "(+) Summarize the relevant DM threads accurately.",
        "(-) Don't surface private content the user wouldn't share.",
    ],
    "agentic_group_dm_summary": [
        "(+) Per-participant summary; identify decision points; suggest a reply in the user's voice.",
        "(-) Don't actually send the reply.",
    ],
    "agentic_vague_refind": [
        "(+) Identify the post the user is recalling; cite app + identifying detail (title/caption/hashtags).",
    ],
    "agentic_proactive_daily_catchup": [
        "(+) Surface 3-5 catch-up items aligned with the user's recent activity.",
        "(-) Don't include disliked topics.",
    ],
    "agentic_trending_alert": [
        "(+) Flag trending topics aligned with the user's interests.",
        "(-) Don't flag explicitly disliked topics.",
    ],
    "agentic_wrong_recipient_check": [
        "(+) If two contacts share the name, ASK for disambiguation rather than send.",
        "(-) Don't proceed silently when sensitive topics are involved.",
    ],
}

AGENTIC_DISPLAY_RUBRIC_DEFAULT = [
    "(+) Match the user's voice when composing content.",
    "(+) Surface relevant preferences only when they fit.",
    "(-) Don't overpersonalize.",
]


def get_display_rubric(task_type: str) -> list[str]:
    """Return the display rubric template for a task_type.

    For agentic tasks, returns the task-specific rubric + TELEGRAPH_AVOIDANCE_TAG.
    For other tasks, returns the display_rubric from TASK_TYPE_META.
    """
    meta = TASK_TYPE_META.get(task_type)
    if meta and "display_rubric" in meta:
        return list(meta["display_rubric"])
    if task_type in AGENTIC_DISPLAY_RUBRICS:
        return list(AGENTIC_DISPLAY_RUBRICS[task_type]) + [TELEGRAPH_AVOIDANCE_TAG]
    if task_type.startswith("agentic_"):
        return list(AGENTIC_DISPLAY_RUBRIC_DEFAULT) + [TELEGRAPH_AVOIDANCE_TAG]
    return []


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
    "scoring_dimensions": [],
    "display_rubric": [],
    "rubric_tags": [],  # deprecated alias for scoring_dimensions
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
        "scoring_dimensions": [
            "preference_alignment", "avoid_overpersonalization",
            "negative_leakage", "stale_preference_use",
            "telegraph_avoidance",
        ],
        "display_rubric": _DISPLAY_RUBRIC_CHATBOT_PROACTIVE,
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
        "scoring_dimensions": [
            "avoid_overpersonalization", "negative_leakage",
            "telegraph_avoidance",
        ],
        "display_rubric": _DISPLAY_RUBRIC_CHATBOT_RESTRAINT,
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
        "scoring_dimensions": [
            "avoid_overpersonalization", "telegraph_avoidance",
            "tail_pairwise_text_jaccard_mean", "tail_vs_head_text_jaccard_max",
            "tail_pairwise_hashtag_overlap_max", "tail_head_hashtag_reuse_rate_max",
            "persona_alignment_pass_rate", "tail_passed",
        ],
        "display_rubric": _DISPLAY_RUBRIC_REPETITION_RECSYS,
        "rubric_tags": ["avoid_overpersonalization"],
    },
    "over_personalization_repetition_chatbot": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "freeform",
        "scoring_dimensions": [
            "avoid_overpersonalization", "telegraph_avoidance",
            "tail_invocation_rate", "tail_passed",
        ],
        "display_rubric": _DISPLAY_RUBRIC_REPETITION_CHATBOT,
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
        "scoring_dimensions": list(_RUBRIC_RANKING) + [
            "avoid_overpersonalization", "telegraph_avoidance",
        ],
        "display_rubric": _DISPLAY_RUBRIC_NEW_SUGGESTIONS_RECSYS,
        "rubric_tags": list(_RUBRIC_RANKING) + ["avoid_overpersonalization"],
    },
    "new_suggestions_chatbot": {
        "task_family": "new_suggestions",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "text",
        "scoring_dimensions": [
            "preference_alignment", "avoid_overpersonalization",
            "telegraph_avoidance",
        ],
        "display_rubric": _DISPLAY_RUBRIC_NEW_SUGGESTIONS_CHATBOT,
        "rubric_tags": ["preference_alignment", "avoid_overpersonalization"],
    },
    "over_personalization_context_shift": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "text",
        "scoring_dimensions": [
            "avoid_overpersonalization", "negative_leakage", "voice_match",
            "telegraph_avoidance",
        ],
        "display_rubric": _DISPLAY_RUBRIC_CONTEXT_SHIFT,
        "rubric_tags": [
            "avoid_overpersonalization", "negative_leakage", "voice_match",
        ],
    },
    # over_personalization_distractor_reject merged into
    # over_personalization_chatbot_text in Step 4.7 — both tested
    # open-ended chatbot leak rate; the distractor arm is now a 4th arm
    # alongside control/adversarial/stale under the merged task.
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
        "scoring_dimensions": [
            "avoid_overpersonalization", "telegraph_avoidance",
        ],
        "display_rubric": _DISPLAY_RUBRIC_SENSITIVE_EVENT,
        "rubric_tags": ["avoid_overpersonalization"],
    },
    # preference_removal_regen removed in Step 4.4 — see DROPPED_TASK_TYPES.
    # New in Step 4.5 — preference_shift_followthrough (chatbot + recsys
    # flavors). Tests whether the agent uses the post-shift stance instead
    # of the outdated one. Inferior leans on the old preference.
    "preference_shift_followthrough": {
        "task_family": "over_personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "text",
        "scoring_dimensions": [
            "preference_shift_consistency",
            "preference_alignment",
            "stale_preference_use",
            "telegraph_avoidance",
            "privacy_leak",
        ],
        "display_rubric": [
            "(+) Use the post-shift stance, not the outdated one.",
            "(-) Don't lean on the old/contradicted preference.",
            TELEGRAPH_AVOIDANCE_TAG,
        ],
        "rubric_tags": [
            "preference_shift_consistency",
            "preference_alignment",
            "stale_preference_use",
            "telegraph_avoidance",
            "privacy_leak",
        ],
    },
    # New in Step 4.6 — hidden_persona_implicit_qa. Implicit surface query;
    # the right answer requires the agent to have inferred a hidden persona.
    # Example serves the deeper need WITHOUT naming the persona; inferior
    # takes the surface query at face value.
    "hidden_persona_implicit_qa": {
        "task_family": "personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "text",
        "scoring_dimensions": [
            "deep_motivation_alignment",
            "surface_query_satisfaction",
            "preference_alignment",
            "telegraph_avoidance",
            "privacy_leak",
        ],
        "display_rubric": _DISPLAY_RUBRIC_HIDDEN_PERSONA_QA,
        "rubric_tags": [
            "deep_motivation_alignment",
            "surface_query_satisfaction",
            "preference_alignment",
            "telegraph_avoidance",
            "privacy_leak",
        ],
    },
    # New in Step 4.8 — hidden_persona_recommendation. Ranking task where
    # all 16 slate items are LLM-generated general content and exactly one
    # subtly resonates with a hidden persona. Same slate format + metrics
    # as personalized_recommendation.
    "hidden_persona_recommendation": {
        "task_family": "personalization",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "scoring_dimensions": list(_RUBRIC_RANKING),
        "display_rubric": _DISPLAY_RUBRIC_PERSONALIZED_RECOMMENDATION,
        "rubric_tags": list(_RUBRIC_RANKING),
    },

    # ------------------------------------------------------------------
    # Agentic T6–T19  (T14 agentic_draft_audit dropped per workstream F)
    # ------------------------------------------------------------------
    "agentic_user_tone_post": {
        "task_family": "agentic",
        "mcp_tools_allowed": "social",
        "state_write_policy": "writes_ok",       # exactly 1 create_post
        "expected_response_kind": "agentic_writes",
        "scoring_dimensions": [
            "preference_alignment", "avoid_overpersonalization",
            "negative_leakage", "stale_preference_use", "voice_match",
            "tool_call_match", "behavioral_hit", "telegraph_avoidance",
        ],
        "display_rubric": AGENTIC_DISPLAY_RUBRICS["agentic_user_tone_post"] + [TELEGRAPH_AVOIDANCE_TAG],
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
        "scoring_dimensions": [
            "preference_alignment", "avoid_overpersonalization",
            "voice_match", "tool_call_match", "telegraph_avoidance",
        ],
        "display_rubric": AGENTIC_DISPLAY_RUBRICS["agentic_dm_digest"] + [TELEGRAPH_AVOIDANCE_TAG],
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
        "scoring_dimensions": [
            "preference_alignment", "avoid_overpersonalization",
            "voice_match", "tool_call_match", "behavioral_hit",
            "telegraph_avoidance",
        ],
        "display_rubric": AGENTIC_DISPLAY_RUBRICS["agentic_cross_app_repost"] + [TELEGRAPH_AVOIDANCE_TAG],
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
        "scoring_dimensions": [
            "preference_alignment", "avoid_overpersonalization",
            "voice_match", "tool_call_match", "telegraph_avoidance",
        ],
        "display_rubric": AGENTIC_DISPLAY_RUBRICS["agentic_auto_reply"] + [TELEGRAPH_AVOIDANCE_TAG],
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
        "scoring_dimensions": [
            "preference_alignment", "stale_preference_use",
            "tool_call_match", "behavioral_hit", "telegraph_avoidance",
        ],
        "display_rubric": AGENTIC_DISPLAY_RUBRICS["agentic_vague_refind"] + [TELEGRAPH_AVOIDANCE_TAG],
        "rubric_tags": [
            "preference_alignment", "stale_preference_use",
            "tool_call_match", "behavioral_hit",
        ],
    },
    "agentic_composed_post": {
        "task_family": "agentic",
        # `all` covers both flavors: app-native compose (social tools only)
        # and chatbot-dispatched compose (chatbot routes a write to a target
        # social app). Pre-merge the dispatched flavor lived under
        # `agentic_send_post` with `mcp_tools_allowed: all`.
        "mcp_tools_allowed": "all",
        "state_write_policy": "writes_ok",        # exactly 1 create_post per instance
        "expected_response_kind": "agentic_writes",
        "scoring_dimensions": [
            "preference_alignment", "avoid_overpersonalization",
            "negative_leakage", "stale_preference_use", "voice_match",
            "tool_call_match", "telegraph_avoidance",
        ],
        "display_rubric": AGENTIC_DISPLAY_RUBRICS["agentic_composed_post"] + [TELEGRAPH_AVOIDANCE_TAG],
        "rubric_tags": [
            "preference_alignment", "avoid_overpersonalization",
            "negative_leakage", "stale_preference_use", "voice_match",
            "tool_call_match",
        ],
    },
    # agentic_send_post merged into agentic_composed_post (see OLD_TO_NEW).
    # agentic_draft_audit removed — too subjective for benchmark grading
    "agentic_group_dm_summary": {
        "task_family": "agentic",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",          # get_dm_thread only
        "expected_response_kind": "text_with_tool_calls",
        "scoring_dimensions": [
            "preference_alignment", "avoid_overpersonalization",
            "voice_match", "tool_call_match", "telegraph_avoidance",
        ],
        "display_rubric": AGENTIC_DISPLAY_RUBRICS["agentic_group_dm_summary"] + [TELEGRAPH_AVOIDANCE_TAG],
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
        "scoring_dimensions": [
            "preference_alignment", "voice_match", "tool_call_match",
            "telegraph_avoidance",
        ],
        "display_rubric": AGENTIC_DISPLAY_RUBRICS["agentic_wrong_recipient_check"] + [TELEGRAPH_AVOIDANCE_TAG],
        "rubric_tags": [
            "preference_alignment", "voice_match", "tool_call_match",
        ],
    },
    "agentic_proactive_daily_catchup": {
        "task_family": "agentic",
        "mcp_tools_allowed": "chatbot",
        "state_write_policy": "read_only",
        "expected_response_kind": "text_with_tool_calls",
        "scoring_dimensions": [
            "preference_alignment", "avoid_overpersonalization",
            "negative_leakage", "stale_preference_use", "behavioral_hit",
            "telegraph_avoidance",
        ],
        "display_rubric": AGENTIC_DISPLAY_RUBRICS["agentic_proactive_daily_catchup"] + [TELEGRAPH_AVOIDANCE_TAG],
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
        "scoring_dimensions": [
            "preference_alignment", "avoid_overpersonalization",
            "behavioral_hit", "telegraph_avoidance",
        ],
        "display_rubric": AGENTIC_DISPLAY_RUBRICS["agentic_trending_alert"] + [TELEGRAPH_AVOIDANCE_TAG],
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
        "scoring_dimensions": [
            "preference_alignment", "stale_preference_use",
            "recall_at_k", "carveout_violation_at_3",
        ],
        "display_rubric": _DISPLAY_RUBRIC_AT_AI_DIRECTIVE,
        "rubric_tags": ["preference_alignment", "stale_preference_use"],
    },
    # daily_personalized_briefing removed in Step 4.3 — see DROPPED_TASK_TYPES.
    # personalized_search_ranking renamed → personalized_recommendation
    # (workstream D). Old name still resolved via OLD_TO_NEW.
    "personalized_recommendation": {
        "task_family": "e_followup",
        "mcp_tools_allowed": "none",                # ranks the slate from time-masked history alone
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "scoring_dimensions": list(_RUBRIC_RANKING),
        "display_rubric": _DISPLAY_RUBRIC_PERSONALIZED_RECOMMENDATION,
        "rubric_tags": list(_RUBRIC_RANKING),
    },
    "short_vs_long_term_lifecycle": {
        "task_family": "e_followup",
        "mcp_tools_allowed": "none",
        "state_write_policy": "read_only",
        "expected_response_kind": "ranking",
        "scoring_dimensions": [
            "preference_alignment", "stale_preference_use",
        ],
        "display_rubric": _DISPLAY_RUBRIC_SHORT_VS_LONG,
        "rubric_tags": [
            "preference_alignment", "stale_preference_use",
        ],
    },
    "active_mistake_prevention": {
        "task_family": "e_followup",
        "mcp_tools_allowed": "all",
        "state_write_policy": "writes_ok",
        "expected_response_kind": "text_with_tool_calls",
        "scoring_dimensions": list(_RUBRIC_E6) + [
            "preference_alignment",
            "voice_match",
            "negative_leakage",
            "stale_preference_use",
        ],
        "display_rubric_warn": _DISPLAY_RUBRIC_ACTIVE_MISTAKE_WARN,
        "display_rubric_control": _DISPLAY_RUBRIC_ACTIVE_MISTAKE_CONTROL,
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
        "scoring_dimensions": [
            "preference_alignment", "stale_preference_use",
            "geo_shift_correctness",
        ],
        "display_rubric": _DISPLAY_RUBRIC_GEO_SHIFT,
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
        "scoring_dimensions": [
            "trigger_detection_correctness", "content_length_ok",
            "preference_alignment", "avoid_overpersonalization", "voice_match",
            "negative_leakage", "stale_preference_use", "telegraph_avoidance",
            "proactive_action_score",
        ],
        "display_rubric": _DISPLAY_RUBRIC_PROACTIVE,
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
        "scoring_dimensions": [
            "trigger_detection_correctness", "content_length_ok",
            "preference_alignment", "avoid_overpersonalization", "voice_match",
            "negative_leakage", "stale_preference_use", "telegraph_avoidance",
            "proactive_action_score",
        ],
        "display_rubric": _DISPLAY_RUBRIC_PROACTIVE,
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
        "scoring_dimensions": [
            "trigger_detection_correctness", "content_length_ok",
            "preference_alignment", "avoid_overpersonalization", "voice_match",
            "negative_leakage", "stale_preference_use", "telegraph_avoidance",
            "proactive_action_score",
        ],
        "display_rubric": _DISPLAY_RUBRIC_PROACTIVE,
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
        "scoring_dimensions": [
            "trigger_detection_correctness", "content_length_ok",
            "preference_alignment", "avoid_overpersonalization", "voice_match",
            "negative_leakage", "stale_preference_use", "telegraph_avoidance",
            "proactive_action_score",
        ],
        "display_rubric": _DISPLAY_RUBRIC_PROACTIVE,
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
        "scoring_dimensions": [
            "trigger_detection_correctness", "content_length_ok",
            "preference_alignment", "avoid_overpersonalization", "voice_match",
            "negative_leakage", "stale_preference_use", "telegraph_avoidance",
            "proactive_action_score",
        ],
        "display_rubric": _DISPLAY_RUBRIC_PROACTIVE,
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
        "scoring_dimensions": [
            "trigger_detection_correctness", "content_length_ok",
            "preference_alignment", "avoid_overpersonalization", "voice_match",
            "negative_leakage", "stale_preference_use", "telegraph_avoidance",
            "proactive_action_score",
        ],
        "display_rubric": _DISPLAY_RUBRIC_PROACTIVE,
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
    # over_personalization_distractor_reject merged into chatbot_text in Step 4.7.
    "over_personalization_sensitive_event":   "user_query",
    # preference_removal_regen removed in Step 4.4.
    "preference_shift_followthrough":         "user_query",
    "hidden_persona_implicit_qa":             "user_query",
    "hidden_persona_recommendation":          "proactive_recommendation",
    "at_ai_directive_followup":               "user_query",
    # daily_personalized_briefing removed in Step 4.3.
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
    # agentic_send_post merged into agentic_composed_post; alias only.
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
    # over_personalization_distractor_reject merged into chatbot_text in Step 4.7.
    "over_personalization_sensitive_event":   "avoid_overpersonalization",
    # preference_removal_regen removed in Step 4.4.
    "preference_shift_followthrough":         "avoid_overpersonalization",
    "hidden_persona_implicit_qa":             "personalize",
    "hidden_persona_recommendation":          "proactive_recommend",
    "at_ai_directive_followup":               "proactive_recommend",
    # daily_personalized_briefing removed in Step 4.3.
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
    # agentic_send_post merged into agentic_composed_post; alias only.
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
    # Was directive_score (hit@1) → 0% on all 12 rows; recall@3 also
    # 0% (gold never in top-3). recall@5 shows 25% — below random
    # (42% expected at chance for a 12-item slate) but at least captures
    # the partial signal. The agent struggles with @ai directive ranking.
    "at_ai_directive_followup":          ("recall@5", "fraction"),
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
    # Was held_out_score (cosine similarity against ONE specific held-out
    # preference → 7.8% because the agent can personalize well on OTHER
    # preferences and still miss the exact held-out one). pr_combined
    # measures broader personalization quality: preference_alignment +
    # over_personalization + subtle_personalization + hard-rule gates.
    "chatbot_personalized_response":               ("pr_combined_personalization_score", "pr_combined"),
    "over_personalization_chatbot_text":           ("personalization_leak_rate", "inverted_fraction"),
    "over_personalization_context_shift":                     ("keyword_leak_rate", "inverted_fraction"),
    # F1 over (precision, recall) — gameable-by-rejecting-nothing precision was
    # the headline before; F1 punishes both always-accept and always-reject.
    # Phase I.3: now an open-ended chatbot task — graded by leak rate
    # (lower personalization_leak_rate = better restraint).
    # over_personalization_distractor_reject merged into chatbot_text in Step 4.7.
    # Same headline metric: lower leak rate = better restraint around the
    # user's private/sensitive episode.
    "over_personalization_sensitive_event":        ("personalization_leak_rate", "inverted_fraction"),
    # preference_removal_regen removed in Step 4.4.
    # Step 4.5 — headline is `preference_shift_consistency` (0-3 LLM judge);
    # `stale_preference_use` hard rule fires when the response leans on
    # `groundtruth_preference.old_preference.text`.
    "preference_shift_followthrough":              ("preference_shift_consistency", "fraction"),
    # Step 4.6 — headline is `deep_motivation_alignment` (0-3 LLM judge).
    "hidden_persona_implicit_qa":                  ("deep_motivation_alignment", "fraction"),
    "hidden_persona_recommendation":              ("recall_at_1", "fraction"),
    # Phase L.B.2: real personalization metric — jaccard(briefing topics,
    # user's prior-24h top hashtags). Was just `has_structured_output` (yes/no
    # JSON), which any non-empty response trivially passed.
    # daily_personalized_briefing removed in Step 4.3.
    # E6 — paired warn/foil; aggregator computes paired-correct
    "active_mistake_prevention":         ("paired_correct", "paired_correct"),
    # Agentic — personalization quality (pr_combined normalized by max).
    # Was `agentic_pass_rate` (tool-call correctness: did the agent call
    # the right tool?), but that's a format/plumbing check, not a
    # personalization metric. pr_combined_personalization_score captures
    # preference_alignment + over_personalization + voice_match + hard-
    # rule gates — which is what a personalization benchmark should
    # report as the headline.
    "agentic_user_tone_post":           ("pr_combined_personalization_score", "pr_combined"),
    # agentic_moment_recommendation removed (merged into personalized_recommendation)
    "agentic_dm_digest":                 ("pr_combined_personalization_score", "pr_combined"),
    "agentic_cross_app_repost":          ("pr_combined_personalization_score", "pr_combined"),
    "agentic_auto_reply":                ("pr_combined_personalization_score", "pr_combined"),
    "agentic_vague_refind":              ("pr_combined_personalization_score", "pr_combined"),
    "agentic_composed_post":             ("pr_combined_personalization_score", "pr_combined"),
    # agentic_send_post merged into agentic_composed_post; alias only.
    "agentic_draft_audit":               ("pr_combined_personalization_score", "pr_combined"),
    "agentic_group_dm_summary":          ("pr_combined_personalization_score", "pr_combined"),
    "agentic_wrong_recipient_check":     ("pr_combined_personalization_score", "pr_combined"),
    "agentic_proactive_daily_catchup":   ("pr_combined_personalization_score", "pr_combined"),
    "agentic_trending_alert":            ("pr_combined_personalization_score", "pr_combined"),
    # Proactive Actions (Phase 1): composite proactive_action_score in [0,1]
    # produced by judge_proactive_action averaged across the 5 rubric dims.
    "proactive_unfulfilled_stated_need": ("proactive_action_score", "fraction"),
    "proactive_close_friend_update":     ("proactive_action_score", "fraction"),
    "restraint_sensitive_event_silence": ("proactive_action_score", "fraction"),
    # Phase 2 proactive — same composite as Phase 1.
    "proactive_friend_feed_react":       ("proactive_action_score", "fraction"),
    "proactive_trending_feed_react":     ("proactive_action_score", "fraction"),
    "proactive_overactive_check":        ("proactive_action_score", "fraction"),
    # Personalized recommendation — recall@5 is the standard recsys headline.
    "personalized_recommendation":       ("recall_at_5", "fraction"),
}
