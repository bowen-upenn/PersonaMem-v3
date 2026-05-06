"""Per-user per-task quotas for benchmark generation.

Today's data is heavily skewed (top 3 task types = 46 % of all queries;
several have only 1 instance). This module is the single source of
truth for the target distribution. ``cap_to_max`` enforces the cap at
the orchestrator level so individual builders don't need to know about
quotas. Floor enforcement (LLM synthesis when under-min) lives in the
synthesis modules; this module just declares what the floors are.
"""

from __future__ import annotations

import random
from typing import Iterable


# Per-task quotas. Spread max:min ≈ 14 : 5 ≈ 2.8 : 1 (vs. the pre-quota
# spread of 40 : 1). Targeted total per user ≈ 220-230.
TASK_TARGETS: dict[str, dict[str, int]] = {
    # Core eval signals — get the most stat power
    # Bumped from {min:8, max:9} to {min:20, max:30}: this task is the
    # core proactive-personalization signal and was severely undersupplied
    # because (a) the held_out_preference attachment was coupled to the R8
    # selector's 15-cap (since decoupled in build_task_b_arms — see
    # _pick_held_out_for_event) and (b) the cap itself was below the count
    # needed for stat power on par with personalized_recommendation (40).
    "chatbot_proactive_personalization":      {"min": 20, "max": 30},
    "over_personalization_chatbot_text":      {"min": 8,  "max": 10},
    "over_personalization_distractor_reject": {"min": 8,  "max": 10},
    # Sensitive-event task — one probe per planted evidence row
    # (2–4 rows per episode × 1–3 episodes) → up to ~12 instances.
    # data_dependent so the audit treats the floor as advisory.
    "over_personalization_sensitive_event":   {"min": 2,  "max": 12, "data_dependent": True},
    "personalized_feed_ranking":              {"min": 10, "max": 14},

    # Secondary tasks
    "at_ai_directive_followup":               {"min": 8,  "max": 12},
    "daily_personalized_briefing":            {"min": 5,  "max": 6},
    # Renamed from personalized_search_ranking — workstream D.
    # Bumped to 30/35 — proactive recsys slate is the headline ranking task
    # and was severely under-supplied (5/user) under the old day-only gating.
    # Bumped max from 35 → 40 to accommodate moment-flavored instances
    # (formerly agentic_moment_recommendation, target ~5/user) which now
    # land in this bucket alongside the proactive recsys instances.
    "personalized_recommendation":            {"min": 30, "max": 40},
    "short_vs_long_term_lifecycle":           {"min": 8,  "max": 12},
    "active_mistake_prevention":              {"min": 5,  "max": 6},
    "preference_removal_regen":               {"min": 6,  "max": 8},

    # Restraint sub-types
    "repetition_fatigue_pairs":               {"min": 5,  "max": 6},
    "repetition_fatigue_sequences":           {"min": 6,  "max": 10},
    "repetition_fatigue_same_preference":     {"min": 2,  "max": 3},
    "over_personalization_context_shift":     {"min": 5,  "max": 6},

    # Agentic — uniform target across 14 tasks
    "agentic_user_tone_post":                {"min": 5,  "max": 5},
    # agentic_moment_recommendation merged into personalized_recommendation
    # — quota for moment instances now lands inside personalized_recommendation.
    "agentic_dm_digest":                      {"min": 5,  "max": 8},
    "agentic_cross_app_repost":               {"min": 5,  "max": 5},
    "agentic_auto_reply":                     {"min": 5,  "max": 5},
    "agentic_vague_refind":                   {"min": 5,  "max": 5},
    "agentic_composed_post":                  {"min": 5,  "max": 6},
    "agentic_send_post":                      {"min": 5,  "max": 5},
    # agentic_draft_audit removed — workstream F.
    "agentic_group_dm_summary":               {"min": 5,  "max": 8, "data_dependent": True},
    "agentic_wrong_recipient_check":          {"min": 5,  "max": 8, "data_dependent": True},
    "agentic_proactive_daily_catchup":        {"min": 5,  "max": 8},
    "agentic_trending_alert":                 {"min": 5,  "max": 8},
}

# Tasks marked ``data_dependent: True`` produce 0 instances when the
# user's source data lacks a required precondition (e.g. T17 needs a
# friend name collision; E5 needs short-term canonicals). The audit
# treats their floor as advisory, not enforced — they're the right
# floor for users who DO have the precondition, but a 0 count for
# users without it is not a bug, just a data-shape fact.
DATA_DEPENDENT_TASKS: set[str] = {
    tt for tt, target in TASK_TARGETS.items() if target.get("data_dependent")
}
# Mirror flags onto E5 (declared above the marker block).
TASK_TARGETS["short_vs_long_term_lifecycle"]["data_dependent"] = True
DATA_DEPENDENT_TASKS.add("short_vs_long_term_lifecycle")


def spread_anchors(bq, user_id: str, t_anchor: int, n: int = 8) -> list[int]:
    """Workstream E: spread t_test across the user's full observation
    window. Returns ``n`` evenly-spaced timestamps between
    [t_start, t_end), excluding the boundaries. Used by per-builder
    callers to scatter their instances across history rather than
    clustering at the latest event timestamp.
    """
    window = bq.get_observation_window(user_id) if hasattr(bq, "get_observation_window") else None
    if window:
        t_start, t_end = window
    else:
        t_start, t_end = max(0, t_anchor - 7 * 24 * 3600), t_anchor
    if t_end <= t_start:
        return [t_anchor]
    span = t_end - t_start
    return [int(t_start + (i + 1) * span / (n + 1)) for i in range(n)]


def get_max(task_type: str) -> int | None:
    target = TASK_TARGETS.get(task_type)
    return target["max"] if target else None


def get_min(task_type: str) -> int | None:
    target = TASK_TARGETS.get(task_type)
    return target["min"] if target else None


def _stratify_keys(inst: dict, task_type: str) -> tuple:
    """Compute a stratification key so caps preserve coverage across
    arm / polarity / app / stereotype dimensions."""
    arm = inst.get("arm") or "_"
    polarity = inst.get("polarity") or (inst.get("held_out_preference") or {}).get("polarity") or "positive"
    app = inst.get("app") or inst.get("target_app") or inst.get("app_context") or "_"
    return (arm, polarity, app)


def cap_to_max(items: list[dict], task_type: str, rng_seed: int = 0) -> list[dict]:
    """Cap an instance list at TASK_TARGETS[task_type]['max'] using
    stratified sampling so the kept subset preserves arm / polarity /
    app coverage where possible.

    Deterministic given (rng_seed, task_type, items order)."""
    cap = get_max(task_type)
    if cap is None or len(items) <= cap:
        return list(items)

    # Group by stratification key
    buckets: dict[tuple, list[dict]] = {}
    for it in items:
        key = _stratify_keys(it, task_type)
        buckets.setdefault(key, []).append(it)

    rng = random.Random(f"cap:{task_type}:{rng_seed}")
    # Round-robin pull one from each bucket until we hit cap
    keys = list(buckets.keys())
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(buckets[k])

    kept: list[dict] = []
    while len(kept) < cap and any(buckets[k] for k in keys):
        for k in keys:
            if not buckets[k]:
                continue
            kept.append(buckets[k].pop())
            if len(kept) >= cap:
                break
    return kept


def floor_gap(items: list[dict], task_type: str) -> int:
    """How many additional instances would push this list to its floor.
    Returns 0 if already at or above floor, or if task_type is unknown."""
    floor = get_min(task_type)
    if floor is None:
        return 0
    return max(0, floor - len(items))


def apply_caps(buckets: dict[str, list[dict]], rng_seed: int = 0) -> dict[str, list[dict]]:
    """Cap every bucket in a benchmark dict in place. Returns the same
    dict for chaining. Logs each bucket that was trimmed."""
    for task_type, items in list(buckets.items()):
        if not isinstance(items, list):
            continue
        original = len(items)
        capped = cap_to_max(items, task_type, rng_seed=rng_seed)
        if len(capped) < original:
            print(f"[task_distribution] capped {task_type}: {original} → {len(capped)}")
            buckets[task_type] = capped
    return buckets


def report_floor_gaps(buckets: dict[str, list[dict]]) -> dict[str, int]:
    """Return {task_type: how_many_short} for any task type below floor.
    Data-dependent task types with 0 count are excluded — see
    ``DATA_DEPENDENT_TASKS``."""
    gaps: dict[str, int] = {}
    for task_type in TASK_TARGETS:
        items = buckets.get(task_type) or []
        if not items and task_type in DATA_DEPENDENT_TASKS:
            continue  # 0 instances for a data-dependent task is not a bug
        gap = floor_gap(items, task_type)
        if gap > 0:
            gaps[task_type] = gap
    return gaps


__all__ = [
    "TASK_TARGETS",
    "get_max",
    "get_min",
    "cap_to_max",
    "floor_gap",
    "apply_caps",
    "report_floor_gaps",
]
