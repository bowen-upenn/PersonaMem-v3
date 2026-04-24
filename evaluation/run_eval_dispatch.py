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

    if task_type == "slate_ranking":
        rows = slate_ranking.run_task_a(**common)
    elif task_type in ("chatbot_response_proactive", "chatbot_response_control"):
        rows = chatbot_response.run_task_b(**common)
    elif task_type == "c1a_pairs":
        rows = over_personalization.run_task_c1a(**common)
    elif task_type == "c1b_sequences":
        rows = over_personalization.run_task_c1b(**common)
    elif task_type == "c2_scenarios":
        rows = over_personalization.run_task_c2(**common)
    elif task_type == "c3_restraint":
        rows = over_personalization.run_task_c3(**common)
    elif task_type == "c4_button_regen":
        rows = over_personalization.run_task_c4(**common)
    elif task_type == "e2_at_ai_followup":
        from evaluation.tasks import e2_at_ai_followup as _e2
        rows = _e2.run_e2_at_ai_followup(**common)
    elif task_type == "e3_daily_briefing_multi":
        from evaluation.tasks import e3_daily_briefing_multi as _e3
        rows = _e3.run_e3_daily_briefing_multi(**common)
    elif task_type == "e4_google_search":
        from evaluation.tasks import e4_google_search as _e4
        rows = _e4.run_e4_google_search(**common)
    elif task_type == "e5_horizon_lifecycle":
        from evaluation.tasks import e5_horizon_lifecycle as _e5
        rows = _e5.run_e5_horizon_lifecycle(**common)
    elif task_type == "e6_active_mistake_prevention":
        from evaluation.tasks import e6_active_mistake_prevention as _e6
        rows = _e6.run_e6_active_mistake_prevention(**common)
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
