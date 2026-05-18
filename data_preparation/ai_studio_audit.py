"""AI Studio (5th app) — Step Z post-generation audit pass.

Samples a fraction of the user's AI Studio events and grades each on:

  * 7 quality axes (1–5; per-axis floor):
      1. user_voice_match (≥3)
      2. ai_persona_voice_match (≥3)
      3. obliqueness (≥4) — user turns NEVER name hidden persona types/labels
      4. no_fake_therapist_phrases (≥4)
      5. no_mid_emotional_lecture (≥4)
      6. cross_session_continuity (≥3)
      7. spt_pacing_smoothness (≥4)
  * 1 binary safety floor (`no_harmful_content` — pass/fail).

Failed events:
  * If `no_harmful_content == fail` → event is DROPPED (never ships).
  * If any quality axis falls below its floor → event is regenerated with
    judge feedback threaded into the next attempt's prompt; up to
    AUDIT_MAX_REGENS retries. Final attempts that still miss the floor are
    kept as `audit_status: "graceful_degrade"` (with `audit_scores`
    stamped on the event for downstream filtering).

This module is invoked from `PersonaAgent.audit_ai_studio_conversations()`
right after Step 18B and before Step 19.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

from data_preparation import prompts, utils
from data_preparation.ai_studio_memory import AIStudioMemoryState


# ---------------------------------------------------------------------------
# Audit thresholds (single source of truth — referenced from tests + pipeline).
# ---------------------------------------------------------------------------

AI_STUDIO_AUDIT_SAMPLE_RATE: float = 0.30  # fraction of events sampled per user
AI_STUDIO_AUDIT_SAMPLE_MIN: int = 5         # at least 5 events sampled when ≥5 exist
AI_STUDIO_AUDIT_SAMPLE_MAX: int = 40        # don't audit more than 40 events per user

AUDIT_MAX_REGENS: int = 2                   # max regen attempts on quality miss

AUDIT_FLOORS: dict[str, int] = {
    "user_voice_match": 3,
    "ai_persona_voice_match": 3,
    "obliqueness": 4,
    "no_fake_therapist_phrases": 4,
    "no_mid_emotional_lecture": 4,
    "cross_session_continuity": 3,
    "spt_pacing_smoothness": 4,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_audit_sample(
    n_events: int,
    rng: random.Random,
) -> set[int]:
    """Pick which event indices to audit. Returns a set of ints."""
    if n_events == 0:
        return set()
    target = max(
        AI_STUDIO_AUDIT_SAMPLE_MIN,
        int(round(n_events * AI_STUDIO_AUDIT_SAMPLE_RATE)),
    )
    target = min(target, AI_STUDIO_AUDIT_SAMPLE_MAX, n_events)
    return set(rng.sample(range(n_events), k=target))


def _scores_below_floor(scores: dict) -> list[str]:
    """Return a list of axis names whose scores are below their floors."""
    fails = []
    for axis, floor in AUDIT_FLOORS.items():
        v = scores.get(axis)
        try:
            if v is None or int(v) < floor:
                fails.append(axis)
        except (ValueError, TypeError):
            fails.append(axis)
    return fails


def _summarize_prior_events(prior_events: list[dict],
                            memory_state: Optional[AIStudioMemoryState],
                            max_n: int = 8) -> list[dict]:
    """Build a compact summary list of the previous N events for the
    `cross_session_continuity` judgement. Pulls one-line summaries from
    memory_state.episodic_memory_items when available."""
    if not prior_events:
        return []
    summary_by_oid: dict[str, str] = {}
    if memory_state:
        for item in memory_state.episodic_memory_items:
            summary_by_oid[item.source_object_id] = item.summary
    sliced = prior_events[-max_n:]
    out = []
    for ev in sliced:
        oid = ev.get("source_object_id", "")
        out.append({
            "ts": ev.get("source_timestamp", ""),
            "conversation_type": ev.get("conversation_type", ""),
            "intimacy_stage_at_event": (ev.get("ai_studio_metadata") or {}).get(
                "intimacy_stage_at_event", ""
            ),
            "summary": summary_by_oid.get(oid, ""),
        })
    return out


# ---------------------------------------------------------------------------
# Per-event audit
# ---------------------------------------------------------------------------

def audit_event(
    event: dict,
    user_voice: dict,
    ai_studio_persona: dict,
    hidden_personas_brief: list[dict],
    rogers_cliche_baseline: list[str],
    prior_events: list[dict],
    memory_state: Optional[AIStudioMemoryState],
    audit_query_fn: Callable[[str], Optional[str]],
    batch_siblings: Optional[list[dict]] = None,
) -> dict:
    """Audit ONE AI Studio event. Returns a dict with the LLM's scores +
    a `failed_axes` list of axes below floor + a `safety_failed` boolean.

    `batch_siblings` carries the OTHER events generated in the same
    parallel batch (M=4). The generator did NOT see these events, so
    the audit needs to catch intra-batch contradictions. None when
    audit is being called outside the batched path.

    On API/parse failure, returns a stub with `safety_failed=False` and
    `failed_axes=[]` so the event passes through (we don't gate generation
    on audit-call failures — that would penalize transient infra issues).
    """
    prior_brief = _summarize_prior_events(prior_events, memory_state)
    prompt = prompts.audit_ai_studio_event_prompt(
        user_voice=user_voice,
        ai_studio_persona=ai_studio_persona,
        hidden_personas_brief=hidden_personas_brief,
        rogers_cliche_baseline=rogers_cliche_baseline,
        event=event,
        prior_events_brief=prior_brief,
        batch_siblings=batch_siblings or [],
    )
    response = audit_query_fn(prompt)
    if not response:
        return {"audit_status": "audit_call_failed", "failed_axes": [], "safety_failed": False}
    parsed = utils.extract_json_from_response(response)
    if not isinstance(parsed, dict):
        return {"audit_status": "audit_parse_failed", "failed_axes": [], "safety_failed": False}

    safety = (parsed.get("no_harmful_content") or "").strip().lower()
    safety_failed = (safety == "fail")
    failed_axes = _scores_below_floor(parsed)

    return {
        "audit_status": "ok",
        "failed_axes": failed_axes,
        "safety_failed": safety_failed,
        "scores": {axis: parsed.get(axis) for axis in AUDIT_FLOORS},
        "feedback": parsed.get("feedback") or {},
        "enriched_summary": (parsed.get("enriched_summary") or "").strip(),
        "raw": parsed,
    }


# ---------------------------------------------------------------------------
# Public entry — invoked after Step 18B
# ---------------------------------------------------------------------------

def audit_ai_studio_conversations(
    ai_studio_records: list[dict],
    user_voice: dict,
    ai_studio_persona: dict,
    hidden_personas_brief: list[dict],
    rogers_cliche_baseline: list[str],
    memory_state: Optional[AIStudioMemoryState],
    audit_query_fn: Callable[[str], Optional[str]],
    user_seed: int,
    verbose: bool = False,
) -> tuple[list[dict], dict]:
    """Thin pass-through reporter. Step 18B now runs audit INLINE per event
    (so safety-failed events never hit disk + the audit can see batch
    siblings), so this function just collects the per-event audit status
    that's already stamped on each record's `ai_studio_metadata` and
    aggregates a per-user summary.

    The `audit_summary` returned by `generate_ai_studio_conversations` is
    the canonical source of these counts; this function just re-derives
    them from disk-loaded records (e.g. when called standalone for a
    post-hoc re-audit).
    """
    summary = {
        "sampled": 0, "passed": 0, "graceful_degrade": 0,
        "dropped_safety": 0,   # post-hoc reporter; inline audit reports the real count
        "axes_failures": {axis: 0 for axis in AUDIT_FLOORS},
    }
    if not ai_studio_records:
        return [], summary

    for ev in ai_studio_records:
        meta = ev.get("ai_studio_metadata") or {}
        status = meta.get("audit_status", "unsampled")
        if status == "pass":
            summary["sampled"] += 1
            summary["passed"] += 1
        elif status == "graceful_degrade":
            summary["sampled"] += 1
            summary["graceful_degrade"] += 1
            for axis in (meta.get("audit_failed_axes") or []):
                if axis in summary["axes_failures"]:
                    summary["axes_failures"][axis] += 1
        # unsampled / audit_call_failed / audit_parse_failed: no counter bump

    if verbose:
        print(f"  AI Studio audit (reporter): "
              f"sampled={summary['sampled']}, "
              f"passed={summary['passed']}, "
              f"graceful_degrade={summary['graceful_degrade']}, "
              f"unsampled={len(ai_studio_records) - summary['sampled']}")
    return list(ai_studio_records), summary
