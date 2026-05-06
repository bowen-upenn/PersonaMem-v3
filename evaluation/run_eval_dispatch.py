"""Single-instance dispatch shim for the sequential eval harness.

Bridges one CSV row (one query) to the existing per-task runner functions
in `evaluation.tasks.*`, which all accept an `instances: list` arg. We
call them with `instances=[inst]` and take the first result.

No new scoring logic — all metrics / judges / rubric behavior is
whatever the existing runners produce. The shim's only job is to map
`task_type` → the right callable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation.tasks import (  # noqa: E402
    slate_ranking,
    chatbot_response,
    over_personalization,
    agentic_tasks,
)


AGENTIC_TASK_IDS: set[str] = set(agentic_tasks.ALL_BUILDERS.keys())


@dataclass
class DispatchContext:
    """Bundle of objects shared across all queries in one persona-run."""
    user_id: str
    bq: Any
    llm_client: Any
    judge_client: Any
    mode: str
    snapshot_cache: Any
    model_name: str
    claude_model: str
    context_budget: int | None
    enable_llm_judge: bool
    dry_run: bool

    def common(self) -> dict:
        """Kwargs shared by every run_task_X(...) callable."""
        return dict(
            user_id=self.user_id,
            bq=self.bq,
            llm_client=self.llm_client,
            judge_client=self.judge_client,
            mode=self.mode,
            snapshot_cache=self.snapshot_cache,
            model_name=self.model_name,
            claude_model=self.claude_model,
            context_budget=self.context_budget,
            enable_llm_judge=self.enable_llm_judge,
            dry_run=self.dry_run,
            limit=None,        # we're feeding one instance at a time
        )


def dispatch_single(task_type: str, inst: dict, ctx: DispatchContext) -> dict | None:
    """Run one instance of `task_type` through its task-specific runner.

    Returns the single result row produced by the runner, or None if the
    task type is unknown / the runner produced no rows.
    """
    common = ctx.common()
    common["instances"] = [inst]

    # Translate any v1 task_type that snuck in (defensive — runner refuses
    # CSVs whose version header doesn't match QUERIES_CSV_VERSION).
    from evaluation.task_registry import normalize_task_type
    task_type = normalize_task_type(task_type)

    if task_type == "personalized_feed_ranking":
        rows = slate_ranking.run_task_a(**common)
    elif task_type in ("chatbot_proactive_personalization", "over_personalization_chatbot_text",
                       "over_personalization_distractor_reject",
                       "over_personalization_sensitive_event"):
        # Phase I.3: distractor-reject converted from a 4-way ranking task to
        # an open-ended chatbot text task — same runner as the other chatbot
        # arms, judged by personalization_leak_rate against the irrelevant
        # persona-items (passed in via privacy_flagged_prefs).
        # R10: sensitive_event runs through the same path with arm="sensitive_event"
        # and a leak pool sourced from the synthetic sensitive_life_event persona.
        rows = chatbot_response.run_task_b(**common)
    elif task_type == "repetition_fatigue_pairs":
        rows = over_personalization.run_task_c1a(**common)
    elif task_type == "repetition_fatigue_sequences":
        rows = over_personalization.run_task_c1b(**common)
    elif task_type == "repetition_fatigue_same_preference":
        rows = over_personalization.run_task_c1c(**common)
    elif task_type == "over_personalization_context_shift":
        rows = over_personalization.run_task_c2(**common)
    elif task_type == "over_personalization_distractor_reject":
        rows = over_personalization.run_task_c3(**common)
    elif task_type == "preference_removal_regen":
        rows = over_personalization.run_task_c4(**common)
    elif task_type == "at_ai_directive_followup":
        from evaluation.tasks import e2_at_ai_followup as _e2
        rows = _e2.run_e2_at_ai_followup(**common)
    elif task_type == "daily_personalized_briefing":
        from evaluation.tasks import e3_daily_briefing_multi as _e3
        rows = _e3.run_e3_daily_briefing_multi(**common)
    elif task_type in ("personalized_recommendation", "personalized_search_ranking"):
        from evaluation.tasks import personalized_recommendation as _pr
        rows = _pr.run_personalized_recommendation(**common)
    elif task_type == "short_vs_long_term_lifecycle":
        from evaluation.tasks import e5_horizon_lifecycle as _e5
        rows = _e5.run_e5_horizon_lifecycle(**common)
    elif task_type == "active_mistake_prevention":
        from evaluation.tasks import e6_active_mistake_prevention as _e6
        rows = _e6.run_e6_active_mistake_prevention(**common)
    elif task_type in ("proactive_unfulfilled_stated_need",
                       "proactive_close_friend_update",
                       "restraint_sensitive_event_silence"):
        from evaluation.tasks import proactive_actions as _proactive
        rows = _proactive.run_proactive_task(**common)
    elif task_type in AGENTIC_TASK_IDS:
        rows = agentic_tasks.run_task(task_id=task_type, **common)
    else:
        return {
            "task": task_type,
            "instance_id": inst.get("instance_id", ""),
            "metrics": {},
            "status": "unknown_task_type",
            "error": f"no dispatcher for task_type={task_type!r}",
        }

    if not rows:
        return None
    return rows[0]
