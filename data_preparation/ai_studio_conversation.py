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
    compute_intimacy_stage,
    eligible_conversation_types,
    increment_intimacy_arc,
    mark_events_as_permanently_demoted,
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
    memory_state: AIStudioMemoryState,
    prior_records: list[dict],
    llm_query_fn: Callable[[str], Optional[str]],
    rng: random.Random,
    verbose: bool = False,
) -> Optional[dict]:
    """Generate ONE AI Studio conversation. Mutates `record` in place and
    returns it on success, None on failure (caller drops the record).
    """
    archetype = ai_studio_persona.get("persona_archetype", "late_night_best_friend")
    explicitness_band = (ai_studio_persona.get("romantic_specifier") or {}).get("explicitness_band")
    n_prior = len(prior_records)

    # SPT stage at this moment (BEFORE this event's delta is applied).
    state = memory_state.running_relational_state
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

    # Build the cross-session memory snapshot that the prompt needs.
    # token_budget=32000 is generous enough that demotion rarely fires for
    # typical users (~30-60 events × ~250 tokens). When it does fire, the
    # newly-demoted event ids returned below get persisted into
    # memory_state.running_relational_state.permanently_demoted_event_ids
    # so future prompts render them as summary in a STABLE way — that's
    # the prompt-cache invariant that lets Azure / Anthropic cache the
    # constants + memory prefix across consecutive events.
    ctx = assemble_generation_context(
        memory_state=memory_state,
        all_prior_events=prior_records,
        token_budget=32000,
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

    # Mutate the record with generation outputs.
    record["conversation"] = cleaned
    record["conversation_type"] = conv_type
    record["prior_session_refs"] = [
        ev.get("source_object_id", "")
        for ev in prior_records
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
    })

    # Update memory state (called BEFORE next event's generation).
    # Persist newly-demoted prior-event ids so future prompts render them
    # as summary in a STABLE way — preserves the prompt-cache prefix.
    mark_events_as_permanently_demoted(
        memory_state,
        ctx.get("newly_demoted_event_ids", []),
    )
    increment_intimacy_arc(memory_state, conv_type, record.get("source_timestamp", 0))
    # Append a thin summary now (the audit pass may overwrite with a richer one later).
    append_episodic_item(memory_state, EpisodicMemoryItem(
        ts=record.get("source_timestamp", 0),
        source_object_id=record.get("source_object_id", ""),
        summary=memory_used_summary or f"{conv_type} (turn count {len(cleaned)})",
        hashtags=record.get("source_hashtags", []) or [],
        evidence_event_ids=[record.get("source_object_id", "")],
        hidden_persona_label_refs=oblique_emitted,
        salience=0.5,
        conversation_type=conv_type,
        intimacy_stage_at_event=stage_emitted,
    ))
    # Heuristic: events at S2+ open a thread on the dominant routed category.
    if stage_index(stage_emitted) >= stage_index("S2") and routed_categories:
        update_open_thread(
            memory_state,
            topic=routed_categories[0],
            ts=record.get("source_timestamp", 0),
        )
    # Persona anchor: keep a rolling 1-line summary of recent AI persona usage.
    set_persona_consistency_anchor(
        memory_state,
        f"AI {ai_studio_persona.get('character_name', '?')} just produced a "
        f"{conv_type} (stage {stage_emitted}). Voice anchored.",
    )

    if verbose:
        print(f"  • {conv_type} (stage {stage_emitted}, {len(cleaned)} turns) — "
              f"mem_summary: {memory_used_summary[:80]!r}")
    return record


# ---------------------------------------------------------------------------
# Public entry point — Step 18b
# ---------------------------------------------------------------------------

def generate_ai_studio_conversations(
    ai_studio_records: list[dict],
    user_profile: dict,
    user_voice: dict,
    ai_studio_persona: dict,
    hidden_personas: list[dict],
    llm_query_fn: Callable[[str], Optional[str]],
    user_seed: int,
    memory_state: Optional[AIStudioMemoryState] = None,
    verbose: bool = False,
) -> tuple[list[dict], AIStudioMemoryState]:
    """Generate AI Studio conversations for all routed events.

    Sequential: each event's prompt embeds the FULL prior history (asymmetric
    memory). Walks events in chronological order; mutates each record in
    place and updates `memory_state`.

    Returns (final_records, memory_state). Records that fail to generate
    are dropped from the output.
    """
    from data_preparation.ai_studio_memory import default_memory_state

    if memory_state is None:
        memory_state = default_memory_state()

    if not ai_studio_records:
        return [], memory_state
    if not ai_studio_persona or not ai_studio_persona.get("persona_archetype"):
        if verbose:
            print("  AI Studio: no persona block on profile — skipping generation.")
        return [], memory_state

    rng = random.Random(user_seed * 1303 + 11)

    # Sort events chronologically — generation is sequential.
    sorted_records = sorted(
        ai_studio_records,
        key=lambda r: r.get("source_timestamp", 0),
    )

    output: list[dict] = []
    for rec in sorted_records:
        result = _generate_one_event(
            record=rec,
            user_profile=user_profile,
            user_voice=user_voice,
            ai_studio_persona=ai_studio_persona,
            hidden_personas=hidden_personas,
            memory_state=memory_state,
            prior_records=output,   # everything generated so far
            llm_query_fn=llm_query_fn,
            rng=rng,
            verbose=verbose,
        )
        if result is None:
            if verbose:
                print(f"  ! AI Studio event ts={rec.get('source_timestamp')} failed; dropping.")
            continue
        output.append(result)

    if verbose:
        print(
            f"  AI Studio generation: {len(output)}/{len(sorted_records)} events succeeded; "
            f"final intimacy_arc={memory_state.running_relational_state.intimacy_arc:.2f} "
            f"(stage {memory_state.running_relational_state.intimacy_stage})"
        )
    return output, memory_state
