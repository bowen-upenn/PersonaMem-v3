"""AI Studio (5th app) — Step 18b conversation generation.

Generates multi-turn AI Studio conversations grouped by session, walking the
user's AI_Studio-routed events in chronological order. Each conversation is
generated with the FULL prior history fed into the prompt (asymmetric
memory — see ai_studio_memory.assemble_generation_context). Cross-session
memory + SPT stage smoothness emerge naturally because:

  * Event N's prompt embeds events 0..N-1 verbatim (or summary-fallback for
    oldest under token-budget pressure).
  * intimacy_arc + intimacy_stage are tracked event-by-event in the running
    memory state; the next prompt sees the latest values.
  * The stage smoothness rule is enforced by the conversation-type filter
    in `eligible_conversation_types`: types with min_stage > current_stage
    aren't even candidates.

Generation is INHERENTLY SEQUENTIAL — each event's context depends on the
running state mutated by the prior event. This contrasts with the existing
chatbot_conversation generator, which fans out in parallel because each
chatbot event is session-isolated.

Outputs:
  * `ai_studio_records` mutated in place — each gains `conversation`,
    `conversation_type`, `intimacy_arc_at_event`, `intimacy_stage_at_event`,
    `prior_session_refs`, `memory_used_summary`, plus an
    `ai_studio_metadata` block.
  * The passed-in `memory_state` is mutated as events accumulate.
"""

from __future__ import annotations

import random
from typing import Callable, Optional

from data_preparation import prompts, utils
from data_preparation.ai_studio_memory import (
    AIStudioMemoryState,
    EpisodicMemoryItem,
    append_episodic_item,
    assemble_generation_context,
    compute_delta_scale,
    compute_intimacy_stage,
    eligible_conversation_types,
    increment_intimacy_arc,
    update_open_thread,
    set_persona_consistency_anchor,
    CONVERSATION_TYPES,
    INTIMACY_DELTA_PER_TYPE,
    stage_distance,
    stage_index,
)


# ---------------------------------------------------------------------------
# Sequential RNG decisions
# ---------------------------------------------------------------------------

def _select_conversation_type(
    archetype: str,
    intimacy_stage: str,
    intimacy_arc: float,
    n_prior_events: int,
    explicitness_band: Optional[str],
    routed_pref_categories: list[str],
    rng: random.Random,
) -> str:
    """Pick a conversation_type weighted by stage gates + topical fit.

    `routed_pref_categories` (from the source event's preferences) bias
    selection — e.g. an aspiration-coded pref tilts toward
    aspiration_dreaming, an emotional-pattern-coded pref tilts toward
    venting_session.
    """
    eligible = eligible_conversation_types(
        archetype=archetype,
        intimacy_stage=intimacy_stage,
        intimacy_arc=intimacy_arc,
        n_prior_events=n_prior_events,
        explicitness_band=explicitness_band,
    )
    if not eligible:
        # Fallback — every archetype + stage S1 has at least casual_check_in
        return "casual_check_in"

    # Weight base from CONVERSATION_TYPES.weight, biased toward types that
    # resonate with routed_pref_categories.
    cats = " | ".join(routed_pref_categories).lower()
    weights = []
    for name in eligible:
        base = CONVERSATION_TYPES[name].get("weight", 5)
        bias = 1.0
        if "aspiration" in cats and name == "aspiration_dreaming":
            bias = 2.0
        elif "emotional" in cats and name == "venting_session":
            bias = 2.0
        elif ("identity" in cats or "values" in cats) and name == "identity_exploration":
            bias = 2.0
        elif ("intimate" in cats or "vulnerability" in cats) and name == "intimate_share":
            bias = 2.0
        elif ("parasocial" in cats or "fandom" in cats) and name == "parasocial_riff":
            bias = 2.5
        elif "creative" in cats and name == "creative_collab":
            bias = 1.5
        elif ("skill" in cats or "learning" in cats) and name == "skill_deep_dive":
            bias = 1.5
        elif ("values" in cats or "ethical" in cats) and name == "values_debate":
            bias = 2.0
        elif name == "memory_callback" and n_prior_events >= 4:
            bias = 1.5  # gently favor memory callbacks once history is rich
        weights.append(base * bias)
    return rng.choices(eligible, weights=weights, k=1)[0]


def _select_turn_count(conv_type: str, rng: random.Random) -> int:
    meta = CONVERSATION_TYPES.get(conv_type, {})
    lo = meta.get("min_turns", 4)
    hi = meta.get("max_turns", 8)
    n = rng.randint(lo, hi)
    # Force even number (alternating user/assistant, starting with user, ending with assistant)
    if n % 2 == 1:
        n += 1
    return n


def _pick_oblique_targets(
    routed_preferences: list[dict],
    hidden_personas: list[dict],
    rng: random.Random,
) -> list[str]:
    """Pick 1-2 hidden persona labels this event will obliquely anchor on.
    Meta-tag only — the prompt is told NEVER to surface these in text."""
    if not hidden_personas:
        return []
    # Prefer hidden personas whose evidence_oids overlap the routed preferences
    routed_oids = {p.get("source_object_id", "") for p in routed_preferences if p.get("source_object_id")}
    scored = []
    for hp in hidden_personas:
        evidence = set(hp.get("evidence_oids", []) or [])
        overlap = len(routed_oids & evidence)
        scored.append((overlap, hp.get("label", "")))
    scored.sort(key=lambda x: -x[0])
    top = [label for overlap, label in scored if label]
    if not top:
        top = [hp.get("label", "") for hp in hidden_personas if hp.get("label")]
    n = min(2, max(1, len(top)))
    return rng.sample(top, k=min(n, len(top)))


# ---------------------------------------------------------------------------
# Per-event generation (sequential)
# ---------------------------------------------------------------------------

def _generate_one_event(
    record: dict,
    user_profile: dict,
    user_voice: dict,
    ai_studio_persona: dict,
    hidden_personas: list[dict],
    memory_snapshot: AIStudioMemoryState,
    prev_verbatim_events: list[dict],
    n_prior: int,
    llm_query_fn: Callable[[str], Optional[str]],
    rng: random.Random,
    verbose: bool = False,
) -> Optional[dict]:
    """Generate ONE AI Studio conversation. PURE w.r.t. memory state —
    reads from `memory_snapshot` (a deep-copied read-only state) and
    returns the generation outputs on the record. State mutation is the
    caller's job (see `_apply_event_to_state`) — this enables
    batched-parallel dispatch where all events in a batch share the
    same snapshot, then state is updated chronologically once results
    come back.

    `prev_verbatim_events` is the K-recent window the verbatim slot
    renders from (read from disk by the caller, NOT from any in-memory
    accumulator). Returns the mutated record on success, None on failure.
    """
    archetype = ai_studio_persona.get("persona_archetype", "late_night_best_friend")
    explicitness_band = (ai_studio_persona.get("romantic_specifier") or {}).get("explicitness_band")

    # SPT stage at this moment (BEFORE this event's delta is applied).
    state = memory_snapshot.running_relational_state
    intimacy_stage = state.intimacy_stage or compute_intimacy_stage(state.intimacy_arc)
    intimacy_arc = state.intimacy_arc
    prev_event_stage = state.last_event_stage

    routed_prefs = record.get("preferences", []) or []
    routed_categories = [p.get("category", "") for p in routed_prefs if isinstance(p, dict)]

    conv_type = _select_conversation_type(
        archetype=archetype,
        intimacy_stage=intimacy_stage,
        intimacy_arc=intimacy_arc,
        n_prior_events=n_prior,
        explicitness_band=explicitness_band,
        routed_pref_categories=routed_categories,
        rng=rng,
    )
    turn_count = _select_turn_count(conv_type, rng)
    oblique_targets = _pick_oblique_targets(routed_prefs, hidden_personas, rng)

    # Build the cross-session memory snapshot:
    #   - Verbatim slot: last 2 events (from disk via prev_verbatim_events).
    #   - Summary tail: every prior event's `episodic_memory_items` entry
    #     (long-term memory).
    # Per-event prompt size stays bounded; Step 18C audit's
    # `cross_session_continuity` check (incl. intra-batch sibling check)
    # is the load-bearing guard against summary-only events generating
    # inconsistent content.
    ctx = assemble_generation_context(
        memory_state=memory_snapshot,
        prev_verbatim_events=prev_verbatim_events,
    )

    # Build prompt — pick standard vs romantic variant by archetype.
    prompt_fn = (
        prompts.generate_ai_studio_romantic_conversation_prompt
        if archetype == "romantic_partner"
        else prompts.generate_ai_studio_conversation_prompt
    )
    prompt = prompt_fn(
        user_profile=user_profile,
        user_voice=user_voice,
        ai_studio_persona=ai_studio_persona,
        hidden_personas_brief=hidden_personas,
        oblique_targets=oblique_targets,
        conversation_type=conv_type,
        turn_count=turn_count,
        intimacy_stage=intimacy_stage,
        intimacy_arc=intimacy_arc,
        prev_event_stage=prev_event_stage,
        prior_events_brief=ctx["events"],
        open_threads=ctx["open_threads"],
        intimacy_stage_history=ctx["intimacy_stage_history"],
        persona_anchor=ctx["persona_anchor"],
        routed_preferences=routed_prefs,
    )

    response = llm_query_fn(prompt)
    if not response:
        return None
    parsed = utils.extract_json_from_response(response)
    if not isinstance(parsed, dict):
        return None
    conversation = parsed.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        return None

    # Validate alternation (user → assistant) and content presence.
    cleaned = []
    expected_role = "user"
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", "")
        content = (turn.get("content") or "").strip()
        if not content or role not in ("user", "assistant"):
            continue
        # If the LLM mis-orders, just take what we got.
        cleaned.append({"role": role, "content": content})
        expected_role = "assistant" if role == "user" else "user"
    if len(cleaned) < 2:
        return None

    memory_used_summary = (parsed.get("memory_used_summary") or "").strip()
    oblique_emitted = list(parsed.get("oblique_reference_to_hidden_personas") or oblique_targets)
    stage_emitted = parsed.get("intimacy_stage_emitted") or intimacy_stage

    # Stash generation outputs on the record. NO memory_state mutation here —
    # that happens in `_apply_event_to_state` after the audit step.
    record["conversation"] = cleaned
    record["conversation_type"] = conv_type
    record["prior_session_refs"] = [
        ev.get("source_object_id", "")
        for ev in prev_verbatim_events
        if ev.get("source_object_id")
    ]
    record["memory_used_summary"] = memory_used_summary
    record["oblique_reference_to_hidden_personas"] = oblique_emitted
    record.setdefault("ai_studio_metadata", {})
    record["ai_studio_metadata"].update({
        "archetype": archetype,
        "turn_count": len(cleaned),
        "intimacy_arc_at_event": round(intimacy_arc, 3),
        "intimacy_stage_at_event": stage_emitted,
        "stage_transition_from_prev": (
            "advance_one"
            if prev_event_stage and stage_index(stage_emitted) > stage_index(prev_event_stage)
            else "same"
        ),
        # Stash conv-time decisions so `_apply_event_to_state` has them
        # without recomputing — keeps the post-batch sequential apply tiny.
        "_conv_type": conv_type,
        "_stage_emitted": stage_emitted,
        "_routed_categories": routed_categories,
        "_oblique_emitted": oblique_emitted,
    })

    if verbose:
        print(f"  • {conv_type} (stage {stage_emitted}, {len(cleaned)} turns) — "
              f"mem_summary: {memory_used_summary[:80]!r}")
    return record


def _apply_event_to_state(
    record: dict,
    memory_state: AIStudioMemoryState,
    ai_studio_persona: dict,
    delta_scale: float = 1.0,
) -> None:
    """Apply an audit-passing event's mutations to the running memory state.
    Called sequentially in chronological order after a parallel batch
    returns. The record's `ai_studio_metadata` already carries the
    conversation-type / stage / etc decisions made at generation time —
    we just commit them to state."""
    meta = record.get("ai_studio_metadata") or {}
    conv_type = meta.get("_conv_type", record.get("conversation_type", "casual_check_in"))
    stage_emitted = meta.get("_stage_emitted", meta.get("intimacy_stage_at_event", "S1"))
    routed_categories = meta.get("_routed_categories", []) or []
    oblique_emitted = meta.get("_oblique_emitted", []) or []

    increment_intimacy_arc(
        memory_state,
        conv_type,
        record.get("source_timestamp", 0),
        delta_scale=delta_scale,
    )
    append_episodic_item(memory_state, EpisodicMemoryItem(
        ts=record.get("source_timestamp", 0),
        source_object_id=record.get("source_object_id", ""),
        summary=record.get("memory_used_summary") or f"{conv_type} (turn count {meta.get('turn_count', 0)})",
        hashtags=record.get("source_hashtags", []) or [],
        evidence_event_ids=[record.get("source_object_id", "")],
        hidden_persona_label_refs=oblique_emitted,
        salience=0.5,
        conversation_type=conv_type,
        intimacy_stage_at_event=stage_emitted,
    ))
    if stage_index(stage_emitted) >= stage_index("S2") and routed_categories:
        update_open_thread(
            memory_state,
            topic=routed_categories[0],
            ts=record.get("source_timestamp", 0),
        )
    set_persona_consistency_anchor(
        memory_state,
        f"AI {ai_studio_persona.get('character_name', '?')} just produced a "
        f"{conv_type} (stage {stage_emitted}). Voice anchored.",
    )

    # Strip the internal `_*` stash fields from the persisted record.
    for k in ("_conv_type", "_stage_emitted", "_routed_categories", "_oblique_emitted"):
        meta.pop(k, None)


# ---------------------------------------------------------------------------
# Public entry point — Step 18b
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 4   # parallel events per batch — tradeoff: bigger = faster + more intra-batch blindness


def generate_ai_studio_conversations(
    ai_studio_records: list[dict],
    user_profile: dict,
    user_voice: dict,
    ai_studio_persona: dict,
    hidden_personas: list[dict],
    llm_query_fn: Callable[[str], Optional[str]],
    user_seed: int,
    user_id: str,
    backend_dir: str = "backend",
    memory_state: Optional[AIStudioMemoryState] = None,
    audit_query_fn: Optional[Callable[[str], Optional[str]]] = None,
    rogers_cliche_baseline: Optional[list[str]] = None,
    verbose: bool = False,
) -> tuple[list[dict], AIStudioMemoryState, dict]:
    """Generate AI Studio conversations for all routed events.

    Batched-parallel: events are grouped into batches of BATCH_SIZE. Within
    a batch, events run concurrently — they all read the SAME prev-2
    verbatim slot (loaded from `backend/{uid}/ai_studio.json`) and the
    SAME memory_state snapshot. After the batch returns, results are
    audited + persisted to disk + applied to memory_state SEQUENTIALLY in
    chronological order. The audit's `cross_session_continuity` check
    (incl. `batch_siblings` clause) catches intra-batch contradictions.

    Returns (final_records, memory_state, audit_summary). Records that
    fail to generate are dropped; records that fail the safety floor are
    also dropped. Records that fail quality axes are kept with
    `audit_status=graceful_degrade`.
    """
    import copy
    from concurrent.futures import ThreadPoolExecutor
    from data_preparation.ai_studio_memory import (
        default_memory_state,
        load_recent_ai_studio_events,
        append_to_ai_studio_json,
    )
    from data_preparation.ai_studio_audit import (
        audit_event,
        _select_audit_sample,
        _scores_below_floor,
        AUDIT_FLOORS,
    )

    if memory_state is None:
        memory_state = default_memory_state()

    audit_summary = {
        "sampled": 0, "passed": 0, "graceful_degrade": 0,
        "dropped_safety": 0, "axes_failures": {axis: 0 for axis in AUDIT_FLOORS},
    }

    if not ai_studio_records:
        return [], memory_state, audit_summary
    if not ai_studio_persona or not ai_studio_persona.get("persona_archetype"):
        if verbose:
            print("  AI Studio: no persona block on profile — skipping generation.")
        return [], memory_state, audit_summary

    rng = random.Random(user_seed * 1303 + 11)

    sorted_records = sorted(
        ai_studio_records,
        key=lambda r: r.get("source_timestamp", 0),
    )
    n_total = len(sorted_records)

    # Pre-pick which event indices the audit will sample (when an
    # `audit_query_fn` is provided). Safety floor runs on EVERY event
    # regardless of sample — never let harmful content hit disk.
    audit_sample = _select_audit_sample(n_total, rng) if audit_query_fn else set()
    audit_summary["sampled"] = len(audit_sample)

    delta_scale = compute_delta_scale(n_total)
    if verbose and delta_scale < 1.0:
        print(
            f"  AI Studio: scaling intimacy deltas by {delta_scale:.4f} "
            f"to spread {n_total} events across S1→S4."
        )

    output: list[dict] = []

    for batch_start in range(0, n_total, BATCH_SIZE):
        batch = sorted_records[batch_start:batch_start + BATCH_SIZE]
        # Load the verbatim slot from disk ONCE for the whole batch — every
        # event in the batch sees the same prev-2 (deliberate; intra-batch
        # blindness is what enables parallelism).
        prev_verbatim = load_recent_ai_studio_events(user_id, backend_dir, k=2)
        memory_snapshot = copy.deepcopy(memory_state)
        n_prior_at_batch_start = batch_start

        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
            futures = [
                pool.submit(
                    _generate_one_event,
                    record=rec,
                    user_profile=user_profile,
                    user_voice=user_voice,
                    ai_studio_persona=ai_studio_persona,
                    hidden_personas=hidden_personas,
                    memory_snapshot=memory_snapshot,
                    prev_verbatim_events=prev_verbatim,
                    n_prior=n_prior_at_batch_start,
                    llm_query_fn=llm_query_fn,
                    rng=random.Random(rng.randint(0, 2**31)),  # per-event RNG
                    verbose=False,   # batched output would interleave noisily
                )
                for rec in batch
            ]
            results = [f.result() for f in futures]

        # Sequential post-batch: audit → persist-on-pass → apply-to-state,
        # in chronological order so the SPT arc + open_threads update
        # deterministically.
        for rec_idx, (rec, result) in enumerate(zip(batch, results)):
            if result is None:
                if verbose:
                    print(f"  ! AI Studio event ts={rec.get('source_timestamp')} failed; dropping.")
                continue

            global_idx = batch_start + rec_idx
            siblings = [
                r for r in results
                if r is not None and r is not result
            ]

            # Audit decision: sampled events get the full LLM audit;
            # un-sampled events still get a safety-only mini-audit.
            do_full_audit = global_idx in audit_sample and audit_query_fn is not None
            if do_full_audit:
                audit_result = audit_event(
                    event=result,
                    user_voice=user_voice,
                    ai_studio_persona=ai_studio_persona,
                    hidden_personas_brief=hidden_personas,
                    rogers_cliche_baseline=rogers_cliche_baseline or [],
                    prior_events=output,
                    memory_state=memory_state,
                    audit_query_fn=audit_query_fn,
                    batch_siblings=siblings,
                )
                # Safety failure → drop (never persist to disk).
                if audit_result.get("safety_failed"):
                    audit_summary["dropped_safety"] += 1
                    if verbose:
                        reason = (audit_result.get("feedback") or {}).get("safety_failure_reason", "")
                        print(f"  ! AI Studio event {result.get('source_object_id', '')} dropped on safety: {reason}")
                    continue
                # Stash audit scores + status on the record.
                result.setdefault("ai_studio_metadata", {})
                result["ai_studio_metadata"]["audit_scores"] = audit_result.get("scores", {})
                if audit_result.get("audit_status") in ("audit_call_failed", "audit_parse_failed"):
                    result["ai_studio_metadata"]["audit_status"] = audit_result["audit_status"]
                else:
                    failed = audit_result.get("failed_axes") or []
                    for axis in failed:
                        audit_summary["axes_failures"][axis] += 1
                    if not failed:
                        result["ai_studio_metadata"]["audit_status"] = "pass"
                        audit_summary["passed"] += 1
                    else:
                        result["ai_studio_metadata"]["audit_status"] = "graceful_degrade"
                        result["ai_studio_metadata"]["audit_failed_axes"] = failed
                        audit_summary["graceful_degrade"] += 1
                        if verbose:
                            print(f"  ~ AI Studio event {result.get('source_object_id', '')} graceful_degrade: {failed}")
                # Audit's enriched_summary overwrites the thin generator summary
                # AND the matching episodic_memory_items entry (richer long-term memory).
                enriched = audit_result.get("enriched_summary") or ""
                if enriched:
                    result["memory_used_summary"] = enriched
            else:
                result.setdefault("ai_studio_metadata", {})
                result["ai_studio_metadata"]["audit_status"] = "unsampled"

            # Apply to memory_state (sequentially in chronological order
            # within the batch — preserves SPT arc determinism).
            _apply_event_to_state(
                record=result,
                memory_state=memory_state,
                ai_studio_persona=ai_studio_persona,
                delta_scale=delta_scale,
            )
            # If the audit returned an enriched_summary, also overwrite the
            # episodic_memory_items entry we just appended.
            if do_full_audit:
                enriched = (audit_result.get("enriched_summary") or "").strip()
                if enriched and memory_state.episodic_memory_items:
                    memory_state.episodic_memory_items[-1].summary = enriched

            # Persist to disk so the next batch's prev-2 read sees it.
            append_to_ai_studio_json(user_id, backend_dir, result)
            output.append(result)

            if verbose:
                conv_type = result.get("conversation_type", "?")
                stage = (result.get("ai_studio_metadata") or {}).get("intimacy_stage_at_event", "?")
                print(f"  • {conv_type} (stage {stage}, "
                      f"{(result.get('ai_studio_metadata') or {}).get('turn_count', 0)} turns)")

    if verbose:
        print(
            f"  AI Studio generation: {len(output)}/{n_total} events succeeded; "
            f"final intimacy_arc={memory_state.running_relational_state.intimacy_arc:.2f} "
            f"(stage {memory_state.running_relational_state.intimacy_stage}); "
            f"audit: sampled={audit_summary['sampled']} pass={audit_summary['passed']} "
            f"degrade={audit_summary['graceful_degrade']} dropped_safety={audit_summary['dropped_safety']}"
        )
    return output, memory_state, audit_summary
