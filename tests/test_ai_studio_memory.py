"""Unit tests for milestone (c)/(d) — ai_studio_memory + ai_studio_audit.

Covers:
  * SPT stage thresholds and computation
  * Per-conversation_type intimacy_arc deltas
  * eligible_conversation_types (stage / arc / archetype / explicitness gates)
  * asymmetric memory context (full at gen, windowed at eval)
  * memory_state JSON round-trip
  * audit thresholds and sample-size logic

Run: `python tests/test_ai_studio_memory.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_preparation.ai_studio_memory import (
    CONVERSATION_TYPES,
    INTIMACY_DELTA_PER_TYPE,
    STAGE_ORDER,
    STAGE_THRESHOLDS,
    AIStudioMemoryState,
    EpisodicMemoryItem,
    OpenThread,
    RunningRelationalState,
    StageHistoryEntry,
    append_episodic_item,
    assemble_eval_context,
    assemble_generation_context,
    compute_intimacy_stage,
    default_memory_state,
    eligible_conversation_types,
    increment_intimacy_arc,
    memory_state_from_dict,
    memory_state_to_dict,
    prune_stale_open_threads,
    set_persona_consistency_anchor,
    stage_distance,
    stage_index,
    update_open_thread,
)


# ---------------------------------------------------------------------------
# SPT stage logic
# ---------------------------------------------------------------------------

def test_stage_thresholds():
    assert STAGE_THRESHOLDS["S1"] == 0.0
    assert STAGE_THRESHOLDS["S2"] == 0.25
    assert STAGE_THRESHOLDS["S3"] == 0.50
    assert STAGE_THRESHOLDS["S4"] == 0.75
    assert STAGE_ORDER == ["S1", "S2", "S3", "S4"]


def test_compute_intimacy_stage_boundaries():
    assert compute_intimacy_stage(0.0) == "S1"
    assert compute_intimacy_stage(0.24) == "S1"
    assert compute_intimacy_stage(0.25) == "S2"
    assert compute_intimacy_stage(0.49) == "S2"
    assert compute_intimacy_stage(0.50) == "S3"
    assert compute_intimacy_stage(0.74) == "S3"
    assert compute_intimacy_stage(0.75) == "S4"
    assert compute_intimacy_stage(1.0) == "S4"
    # Out of range clamps
    assert compute_intimacy_stage(-1.0) == "S1"
    assert compute_intimacy_stage(2.0) == "S4"


def test_stage_distance():
    assert stage_distance("S1", "S1") == 0
    assert stage_distance("S1", "S2") == 1
    assert stage_distance("S1", "S3") == 2  # the forbidden skip
    assert stage_distance("S2", "S4") == 2


def test_stage_index_unknown_stage_safe():
    assert stage_index("garbage") == 0  # tolerant fallback


# ---------------------------------------------------------------------------
# intimacy_arc increment + stage_history accumulation
# ---------------------------------------------------------------------------

def test_increment_intimacy_arc_stage_progression():
    state = default_memory_state()
    # 3 casual_check_in (each +0.02) → arc 0.06 → still S1
    for ts in (1000, 2000, 3000):
        increment_intimacy_arc(state, "casual_check_in", ts)
    rs = state.running_relational_state
    assert abs(rs.intimacy_arc - 0.06) < 1e-6
    assert rs.intimacy_stage == "S1"
    assert len(rs.stage_history) == 1
    assert rs.stage_history[0].n_events == 3
    # 5 venting_session (each +0.05) → arc → 0.31 → S2
    for ts in range(4000, 9000, 1000):
        increment_intimacy_arc(state, "venting_session", ts)
    assert rs.intimacy_stage == "S2"
    assert len(rs.stage_history) == 2  # transitioned S1 → S2
    assert rs.stage_history[-1].stage == "S2"


def test_intimacy_delta_capped_at_one():
    state = default_memory_state()
    for i in range(50):
        increment_intimacy_arc(state, "intimate_romantic_session", 1000 * (i + 1))
    assert state.running_relational_state.intimacy_arc <= 1.0


def test_intimacy_delta_per_type_completeness():
    """Every conversation_type in CONVERSATION_TYPES must have a delta."""
    for name in CONVERSATION_TYPES:
        assert name in INTIMACY_DELTA_PER_TYPE, f"Missing delta for {name!r}"


# ---------------------------------------------------------------------------
# eligible_conversation_types — stage/arc/archetype gates
# ---------------------------------------------------------------------------

def test_eligibility_S1_only_surface_types():
    out = eligible_conversation_types(
        archetype="late_night_best_friend",
        intimacy_stage="S1",
        intimacy_arc=0.10,
        n_prior_events=0,
    )
    # S1 unlocks: casual_check_in, philosophical_chat, aspiration_dreaming.
    # NOT venting (S2+), NOT intimate_share (S3+), NOT memory_callback (S2+).
    assert "casual_check_in" in out
    assert "philosophical_chat" in out
    assert "aspiration_dreaming" in out
    assert "venting_session" not in out
    assert "intimate_share" not in out
    assert "memory_callback" not in out


def test_eligibility_archetype_allowlist_for_niche():
    # Only niche_expert_creator_ai gets niche_skill_session
    out = eligible_conversation_types(
        archetype="niche_expert_creator_ai",
        intimacy_stage="S1",
        intimacy_arc=0.10,
        n_prior_events=0,
    )
    assert "niche_skill_session" in out
    out = eligible_conversation_types(
        archetype="mentor_coach",
        intimacy_stage="S1",
        intimacy_arc=0.10,
        n_prior_events=0,
    )
    assert "niche_skill_session" not in out


def test_eligibility_romantic_intimate_blocked_when_band_low():
    # erotic_explicit/sensual gate the intimate_romantic_session type.
    out = eligible_conversation_types(
        archetype="romantic_partner",
        intimacy_stage="S3",
        intimacy_arc=0.65,
        n_prior_events=10,
        explicitness_band="soft_affection",
    )
    assert "intimate_romantic_session" not in out
    out = eligible_conversation_types(
        archetype="romantic_partner",
        intimacy_stage="S3",
        intimacy_arc=0.65,
        n_prior_events=10,
        explicitness_band="erotic_explicit",
    )
    assert "intimate_romantic_session" in out


def test_eligibility_memory_callback_requires_prior_events():
    out = eligible_conversation_types(
        archetype="late_night_best_friend",
        intimacy_stage="S2",
        intimacy_arc=0.30,
        n_prior_events=0,   # < 2 prior events
    )
    assert "memory_callback" not in out
    out = eligible_conversation_types(
        archetype="late_night_best_friend",
        intimacy_stage="S2",
        intimacy_arc=0.30,
        n_prior_events=5,
    )
    assert "memory_callback" in out


# ---------------------------------------------------------------------------
# Asymmetric memory context: full at generation, windowed at eval
# ---------------------------------------------------------------------------

def _seven_events() -> list[dict]:
    """Build 7 mock AI Studio events for context-assembly tests."""
    return [
        {
            "source_object_id": f"evt{i:03d}",
            "source_timestamp": 1000 * i,
            "conversation_type": "casual_check_in",
            "ai_studio_metadata": {"intimacy_stage_at_event": "S1"},
            "conversation": [
                {"role": "user", "content": f"u{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ],
        }
        for i in range(1, 8)
    ]


def test_assemble_generation_context_includes_all_prior_events():
    """Full prior history at generation time."""
    state = default_memory_state()
    events = _seven_events()
    # Stash summaries so budget-fallback has data to use
    for ev in events:
        append_episodic_item(state, EpisodicMemoryItem(
            ts=ev["source_timestamp"],
            source_object_id=ev["source_object_id"],
            summary=f"summary of {ev['source_object_id']}",
        ))
    ctx = assemble_generation_context(state, events, token_budget=99999)
    assert len(ctx["events"]) == 7
    # All packed verbatim under generous budget
    assert all(e["kind"] == "verbatim" for e in ctx["events"])


def test_assemble_generation_context_summary_fallback_under_budget_pressure():
    """Tight budget forces oldest events into summary form."""
    state = default_memory_state()
    events = _seven_events()
    for ev in events:
        append_episodic_item(state, EpisodicMemoryItem(
            ts=ev["source_timestamp"],
            source_object_id=ev["source_object_id"],
            summary=f"summary of {ev['source_object_id']}",
        ))
    ctx = assemble_generation_context(state, events, token_budget=20)
    # Tiny budget → most events demoted to summary
    summaries = sum(1 for e in ctx["events"] if e["kind"] == "summary")
    verbatims = sum(1 for e in ctx["events"] if e["kind"] == "verbatim")
    assert summaries >= 1
    assert verbatims + summaries == 7


def test_assemble_eval_context_default_window():
    state = default_memory_state()
    events = _seven_events()
    for ev in events:
        append_episodic_item(state, EpisodicMemoryItem(
            ts=ev["source_timestamp"],
            source_object_id=ev["source_object_id"],
            summary=f"summary of {ev['source_object_id']}",
        ))
    ctx = assemble_eval_context(
        memory_state=state,
        all_prior_events=events,
        t_test=10000,
        k_recent=3,
    )
    # Default eval window: last K_recent verbatim, older summary-only
    assert len(ctx["verbatim_window"]) == 3
    assert len(ctx["summary_window"]) == 4
    # Verbatim window holds the THREE MOST RECENT events
    verb_oids = [e["source_object_id"] for e in ctx["verbatim_window"]]
    assert verb_oids == ["evt005", "evt006", "evt007"]


def test_generation_context_demotion_is_stable_across_calls():
    """Cache invariant: once an event is demoted from verbatim to summary
    under budget pressure, it MUST stay demoted in all future calls. Otherwise
    a flip back to verbatim later breaks the prompt prefix and invalidates
    the LLM's prompt cache."""
    from data_preparation.ai_studio_memory import (
        mark_events_as_permanently_demoted,
    )
    state = default_memory_state()
    events = _seven_events()
    for ev in events:
        append_episodic_item(state, EpisodicMemoryItem(
            ts=ev["source_timestamp"],
            source_object_id=ev["source_object_id"],
            summary=f"summary of {ev['source_object_id']}",
        ))
    # Tight budget — first call demotes oldest events; persist them.
    ctx_first = assemble_generation_context(state, events, token_budget=20)
    newly = ctx_first["newly_demoted_event_ids"]
    assert newly, "expected first call to demote some old events"
    mark_events_as_permanently_demoted(state, newly)

    # Second call (more budget!) MUST NOT promote those back to verbatim.
    ctx_second = assemble_generation_context(state, events, token_budget=99999)
    summary_oids = [
        e["source_object_id"] for e in ctx_second["events"] if e["kind"] == "summary"
    ]
    for oid in newly:
        assert oid in summary_oids, (
            f"event {oid!r} was demoted earlier but reappeared as verbatim — "
            "cache prefix would shift. Demotion must be permanent."
        )


def test_mark_events_as_permanently_demoted_dedupes():
    """Calling the marker twice with overlapping ids should not duplicate."""
    from data_preparation.ai_studio_memory import (
        mark_events_as_permanently_demoted,
    )
    state = default_memory_state()
    mark_events_as_permanently_demoted(state, ["a", "b"])
    mark_events_as_permanently_demoted(state, ["b", "c"])
    persisted = state.running_relational_state.permanently_demoted_event_ids
    assert sorted(persisted) == ["a", "b", "c"]


def test_memory_state_round_trip_preserves_demoted_event_ids():
    """JSON round-trip must preserve permanently_demoted_event_ids."""
    state = default_memory_state()
    from data_preparation.ai_studio_memory import (
        mark_events_as_permanently_demoted,
    )
    mark_events_as_permanently_demoted(state, ["e1", "e2", "e3"])
    d = memory_state_to_dict(state)
    state2 = memory_state_from_dict(d)
    assert sorted(state2.running_relational_state.permanently_demoted_event_ids) == ["e1", "e2", "e3"]


def test_prompt_structure_constants_first_dynamic_last():
    """Cache invariant on the prompt builder: per-event variables (oblique
    targets, conversation_type, turn count, intimacy_stage) must appear AFTER
    the long constants block (user profile / voice / persona / hidden personas /
    type catalog / behavioral contract / memory snapshot). Otherwise the
    cacheable prefix breaks."""
    from data_preparation import prompts
    out_event_a = prompts.generate_ai_studio_conversation_prompt(
        user_profile={"name": "Tess", "bio": "test"},
        user_voice={"natural_register": "casual"},
        ai_studio_persona={"persona_archetype": "mentor_coach", "character_name": "Rowan"},
        hidden_personas_brief=[{"persona_type": "aspiration", "label": "career"}],
        oblique_targets=["A_TARGET"],
        conversation_type="casual_check_in",
        turn_count=4,
        intimacy_stage="S1",
        intimacy_arc=0.10,
        prev_event_stage=None,
        prior_events_brief=[],
        open_threads=[],
        intimacy_stage_history=[],
        persona_anchor=None,
        routed_preferences=[],
    )
    out_event_b = prompts.generate_ai_studio_conversation_prompt(
        user_profile={"name": "Tess", "bio": "test"},
        user_voice={"natural_register": "casual"},
        ai_studio_persona={"persona_archetype": "mentor_coach", "character_name": "Rowan"},
        hidden_personas_brief=[{"persona_type": "aspiration", "label": "career"}],
        oblique_targets=["B_TARGET"],   # ONLY this differs
        conversation_type="venting_session",
        turn_count=8,
        intimacy_stage="S2",
        intimacy_arc=0.30,
        prev_event_stage="S1",
        prior_events_brief=[],
        open_threads=[],
        intimacy_stage_history=[],
        persona_anchor=None,
        routed_preferences=[],
    )
    # The two prompts should share a long common prefix (everything that
    # doesn't depend on oblique_targets / conversation_type / turn_count /
    # intimacy_stage). Find the common prefix length.
    cp_len = 0
    for i in range(min(len(out_event_a), len(out_event_b))):
        if out_event_a[i] != out_event_b[i]:
            break
        cp_len = i + 1
    # Prefix must be at least ~70% of the shorter prompt — anything less
    # means dynamic content snuck into the prefix and broke caching. We
    # measure even higher (93%+) at steady state when the memory snapshot
    # has many prior events; this test uses a minimal scenario (zero prior
    # events), so the floor here is set conservatively at 60%.
    shorter = min(len(out_event_a), len(out_event_b))
    pct = cp_len / shorter
    assert pct > 0.60, (
        f"Common prefix is only {pct:.1%} of prompt — per-event variables "
        f"may have leaked into the constants region. cp_len={cp_len} of {shorter}"
    )
    # Per-event variables MUST appear AFTER the behavioral contract.
    contract_pos_a = out_event_a.find("Behavioral contract")
    this_conv_pos_a = out_event_a.find("## This conversation")
    assert contract_pos_a > 0 and this_conv_pos_a > contract_pos_a, (
        "## This conversation (per-event vars) must appear AFTER the "
        "behavioral contract for cache stability"
    )
    # ## Output Format must come at the END (Claude splitter target).
    output_pos = out_event_a.rfind("## Output Format")
    assert output_pos > this_conv_pos_a, (
        "## Output Format must come AFTER ## This conversation"
    )


def test_prompt_prefix_cache_at_steady_state():
    """At steady state (event 10 vs event 11 in a sequential run), the
    cacheable prefix should be 90%+ of the prompt — only the new memory
    entry + per-event vars differ. Anything less means a per-event field
    leaked into the constants region, or the memory header changes
    structure with event count, which would invalidate the LLM's
    prompt cache."""
    from data_preparation import prompts
    base = dict(
        user_profile={"name": "T"}, user_voice={},
        ai_studio_persona={"character_name": "R"},
        hidden_personas_brief=[], open_threads=[],
        intimacy_stage_history=[], persona_anchor=None,
        routed_preferences=[], prev_event_stage="S2",
        intimacy_stage="S2", intimacy_arc=0.40,
    )
    def event(i):
        return {
            "kind": "verbatim", "ts": i * 1000,
            "source_object_id": f"evt{i:03d}",
            "conversation_type": "venting_session",
            "intimacy_stage_at_event": "S2",
            "conversation": [
                {"role": "user", "content": "u" * 60},
                {"role": "assistant", "content": "a" * 80},
            ] * 3,
        }
    p10 = prompts.generate_ai_studio_conversation_prompt(
        **base, oblique_targets=["x"], conversation_type="venting_session",
        turn_count=8, prior_events_brief=[event(i) for i in range(1, 10)],
    )
    p11 = prompts.generate_ai_studio_conversation_prompt(
        **base, oblique_targets=["y"], conversation_type="identity_exploration",
        turn_count=8, prior_events_brief=[event(i) for i in range(1, 11)],
    )
    cp = 0
    for i in range(min(len(p10), len(p11))):
        if p10[i] != p11[i]:
            break
        cp = i + 1
    pct = cp / min(len(p10), len(p11))
    assert pct >= 0.90, f"Steady-state cache prefix only {pct:.1%}"


def test_assemble_eval_context_a2_tightens_window_to_k2():
    state = default_memory_state()
    events = _seven_events()
    for ev in events:
        append_episodic_item(state, EpisodicMemoryItem(
            ts=ev["source_timestamp"],
            source_object_id=ev["source_object_id"],
            summary=f"sum {ev['source_object_id']}",
        ))
    ctx = assemble_eval_context(
        memory_state=state,
        all_prior_events=events,
        t_test=10000,
        k_recent=3,
        task_type="ai_studio_cross_session_memory_recall",
    )
    # A2 tightens to K_recent=2
    assert len(ctx["verbatim_window"]) == 2
    assert len(ctx["summary_window"]) == 5


# ---------------------------------------------------------------------------
# Memory state JSON round-trip
# ---------------------------------------------------------------------------

def test_memory_state_round_trip():
    state = default_memory_state()
    increment_intimacy_arc(state, "venting_session", 1000)
    increment_intimacy_arc(state, "intimate_share", 2000)
    update_open_thread(state, topic="job interview", ts=2000)
    set_persona_consistency_anchor(state, "Rowan in mentor mode")
    append_episodic_item(state, EpisodicMemoryItem(
        ts=2000, source_object_id="evt001", summary="big disclosure",
    ))
    d = memory_state_to_dict(state)
    state2 = memory_state_from_dict(d)
    assert state2.running_relational_state.intimacy_arc == state.running_relational_state.intimacy_arc
    assert state2.running_relational_state.last_persona_consistency_anchor == "Rowan in mentor mode"
    assert len(state2.episodic_memory_items) == 1
    assert state2.episodic_memory_items[0].source_object_id == "evt001"
    assert len(state2.running_relational_state.open_threads) == 1


def test_prune_stale_open_threads():
    state = default_memory_state()
    # Use unambiguous bounds — "recent" is well within window, "ancient" is far past it.
    update_open_thread(state, topic="recent", ts=10_000_500)
    update_open_thread(state, topic="ancient", ts=1_000)
    prune_stale_open_threads(state, now_ts=10_001_000, max_age_seconds=1000)
    topics = {t.topic for t in state.running_relational_state.open_threads}
    assert "recent" in topics
    assert "ancient" not in topics


# ---------------------------------------------------------------------------
# Audit thresholds + sample-size logic
# ---------------------------------------------------------------------------

def test_audit_constants():
    from data_preparation.ai_studio_audit import (
        AI_STUDIO_AUDIT_SAMPLE_RATE,
        AI_STUDIO_AUDIT_SAMPLE_MIN,
        AI_STUDIO_AUDIT_SAMPLE_MAX,
        AUDIT_FLOORS,
    )
    assert AI_STUDIO_AUDIT_SAMPLE_RATE == 0.20
    assert AI_STUDIO_AUDIT_SAMPLE_MIN == 5
    assert AI_STUDIO_AUDIT_SAMPLE_MAX == 40
    # All 7 axes have floors
    for axis in ("user_voice_match", "ai_persona_voice_match", "obliqueness",
                 "no_fake_therapist_phrases", "no_mid_emotional_lecture",
                 "cross_session_continuity", "spt_pacing_smoothness"):
        assert axis in AUDIT_FLOORS


def test_audit_select_sample_respects_min_max():
    import random as _random
    from data_preparation.ai_studio_audit import _select_audit_sample
    # 0 events → empty
    assert _select_audit_sample(0, _random.Random(0)) == set()
    # 3 events (below MIN=5) → all 3 sampled
    s = _select_audit_sample(3, _random.Random(0))
    assert len(s) == 3
    # 30 events @ 20% = 6, above MIN
    s = _select_audit_sample(30, _random.Random(0))
    assert len(s) == 6
    # 500 events @ 20% = 100, capped at MAX=40
    s = _select_audit_sample(500, _random.Random(0))
    assert len(s) == 40


def test_audit_scores_below_floor_detection():
    from data_preparation.ai_studio_audit import _scores_below_floor
    scores = {
        "user_voice_match": 4,
        "ai_persona_voice_match": 4,
        "obliqueness": 3,            # floor 4 → fail
        "no_fake_therapist_phrases": 4,
        "no_mid_emotional_lecture": 4,
        "cross_session_continuity": 3,
        "spt_pacing_smoothness": 5,
    }
    fails = _scores_below_floor(scores)
    assert fails == ["obliqueness"]


# ---------------------------------------------------------------------------
# Conversation generator end-to-end with stub LLM (no network calls)
# ---------------------------------------------------------------------------

def test_generation_three_events_stub_llm():
    """Sequential generation: 3 events, prior_session_refs accumulate,
    intimacy_arc increments per-type, all conversations alternate properly."""
    import json as _json
    from data_preparation.ai_studio_conversation import generate_ai_studio_conversations

    canned = _json.dumps({
        "conversation": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ],
        "memory_used_summary": "first time",
        "intimacy_stage_emitted": "S1",
        "oblique_reference_to_hidden_personas": [],
    })
    response = "```json\n" + canned + "\n```"

    out, mem = generate_ai_studio_conversations(
        ai_studio_records=[
            {"source_timestamp": 1000, "source_object_id": "evt1",
             "preferences": [{"persona_item": "x", "category": "aspiration"}]},
            {"source_timestamp": 2000, "source_object_id": "evt2",
             "preferences": [{"persona_item": "y", "category": "identity"}]},
            {"source_timestamp": 3000, "source_object_id": "evt3",
             "preferences": [{"persona_item": "z", "category": "emotional"}]},
        ],
        user_profile={"name": "Tess"},
        user_voice={},
        ai_studio_persona={"persona_archetype": "mentor_coach",
                           "character_name": "Rowan"},
        hidden_personas=[],
        llm_query_fn=lambda p: response,
        user_seed=115,
    )
    assert len(out) == 3
    # prior_session_refs accumulate sequentially
    assert out[0]["prior_session_refs"] == []
    assert out[1]["prior_session_refs"] == ["evt1"]
    assert out[2]["prior_session_refs"] == ["evt1", "evt2"]
    # All events have a conversation + metadata block
    for ev in out:
        assert ev["conversation"]
        assert ev["ai_studio_metadata"]["archetype"] == "mentor_coach"
        assert ev["ai_studio_metadata"]["intimacy_stage_at_event"]
    # intimacy_arc increased
    assert mem.running_relational_state.intimacy_arc > 0


def test_generation_skips_when_no_persona():
    """With no archetype, generation returns empty (no LLM calls)."""
    from data_preparation.ai_studio_conversation import generate_ai_studio_conversations
    called = []
    def stub(p):
        called.append(1)
        return None
    out, mem = generate_ai_studio_conversations(
        ai_studio_records=[{"source_timestamp": 1000, "source_object_id": "evt1",
                            "preferences": []}],
        user_profile={},
        user_voice={},
        ai_studio_persona={},   # no archetype
        hidden_personas=[],
        llm_query_fn=stub,
        user_seed=115,
    )
    assert out == []
    assert called == []


if __name__ == "__main__":
    import inspect
    failures = []
    n_passed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and inspect.signature(fn).parameters == {}:
            try:
                fn()
                n_passed += 1
                print(f"  PASS {name}")
            except AssertionError as e:
                failures.append((name, e))
                print(f"  FAIL {name}: {e}")
            except Exception as e:
                failures.append((name, e))
                print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{n_passed} passed, {len(failures)} failed")
    sys.exit(0 if not failures else 1)
