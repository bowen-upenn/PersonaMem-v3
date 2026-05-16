"""AI Studio (5th app) — cross-session memory mechanics.

The asymmetric design is the central principle:

* GENERATION SIDE — `assemble_generation_context` passes the FULL prior
  history (every previous AI Studio conversation, verbatim where the token
  budget allows; older conversations demoted to summary form when
  truncation is needed). This is what makes the dataset coherent: the
  reference / "gold" AI sees the entire arc and produces replies that line
  up with what was said before.

* EVAL SIDE — `assemble_eval_context` passes a deliberately LIMITED slice
  (last K_recent verbatim conversations + summary-only older). The eval
  measures whether the model under test can carry relevant info forward
  from a small window. If it had the whole history every time, the
  cross-session-memory eval task would degenerate to substring lookup.

The memory state itself (`ai_studio_memory.json`) tracks:
  * intimacy_arc + intimacy_stage history (SPT pacing)
  * open_threads (topics where the AI still owes a follow-up)
  * episodic_memory_items (one-line summaries of every passing event,
    keyed by source_object_id, used for both the eval-side summary
    window AND for budget-pressure fallback at generation time)
  * last_persona_consistency_anchor (rolling 2-3-sentence summary of the
    AI persona's recent voice, fed back at the next event so the
    persona doesn't drift)

This module is pure — no LLM calls, no network. Generation/audit modules
import these helpers + the constants below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# SPT stage thresholds (Social Penetration Theory — Altman & Taylor 1973).
# ---------------------------------------------------------------------------
# `intimacy_arc` is a [0,1] running counter that increments per audit-passing
# event. The four discrete stages are derived from the arc value at the
# moment of an event. The no-jump rule (smoothness) is enforced separately
# at generation time.

STAGE_THRESHOLDS = {
    "S1": 0.0,    # orientation — public scripts, weather, surface preferences
    "S2": 0.25,   # exploratory affective — early opinions, mild personal anecdotes
    "S3": 0.50,   # affective exchange — genuine views, vulnerabilities, mild fears
    "S4": 0.75,   # stable exchange — core beliefs, intimate values, deep fears
}

STAGE_ORDER = ["S1", "S2", "S3", "S4"]


def compute_intimacy_stage(arc: float) -> str:
    """Map a continuous intimacy_arc value to a discrete SPT stage."""
    arc = max(0.0, min(1.0, float(arc)))
    if arc >= STAGE_THRESHOLDS["S4"]:
        return "S4"
    if arc >= STAGE_THRESHOLDS["S3"]:
        return "S3"
    if arc >= STAGE_THRESHOLDS["S2"]:
        return "S2"
    return "S1"


def stage_index(stage: str) -> int:
    """Return the index of an SPT stage in STAGE_ORDER (S1 → 0, … S4 → 3)."""
    try:
        return STAGE_ORDER.index(stage)
    except ValueError:
        return 0


def stage_distance(a: str, b: str) -> int:
    """Absolute distance between two SPT stages. Used for smoothness checks."""
    return abs(stage_index(a) - stage_index(b))


# ---------------------------------------------------------------------------
# Per-conversation_type intimacy_arc deltas. After each audit-passing event,
# the user's intimacy_arc is incremented by the amount listed below (capped
# at 1.0). Casual / philosophical types add little; intimate / flirty types
# add the most. memory_callback is orthogonal (no delta — recall doesn't
# deepen the relationship, it just tests whether the AI is paying attention).
# ---------------------------------------------------------------------------

INTIMACY_DELTA_PER_TYPE = {
    "casual_check_in":           0.02,
    "philosophical_chat":        0.02,
    "aspiration_dreaming":       0.05,
    "venting_session":           0.05,
    "identity_exploration":      0.07,
    "memory_callback":           0.00,   # orthogonal axis
    "niche_skill_session":       0.02,
    "intimate_share":            0.10,
    "parasocial_riff":           0.05,
    "flirty_banter":             0.10,
    "intimate_romantic_session": 0.12,
}


# Per-user delta scaling. The raw deltas above sum-of-means to ~0.052; with
# 222 AI-Studio-routed events for a heavy user (like #115) the unscaled arc
# would saturate at 1.0 by event ~20 and pin there for the remaining 200,
# collapsing SPT pacing to "always-S4." compute_delta_scale shrinks the per-
# event delta so cumulative arc lands near _TARGET_FINAL_ARC at the last
# event, giving S1→S4 a realistic spread across the user's full history.
_MEAN_DELTA_BASE = 0.052
_TARGET_FINAL_ARC = 0.85


def compute_delta_scale(n_total_events: int) -> float:
    """Return the per-event delta-scaling factor for a user with
    ``n_total_events`` AI-Studio-routed events.

    For small histories (<= ~16 events at base mean) returns 1.0 — the raw
    deltas already pace nicely. For larger histories, returns a fractional
    multiplier so the expected cumulative arc lands near _TARGET_FINAL_ARC
    by the final event."""
    if n_total_events <= 0:
        return 1.0
    expected_unscaled = n_total_events * _MEAN_DELTA_BASE
    if expected_unscaled <= _TARGET_FINAL_ARC:
        return 1.0
    return _TARGET_FINAL_ARC / expected_unscaled


# ---------------------------------------------------------------------------
# Conversation-type catalog with SPT stage gates + archetype gates +
# turn-count ranges. Single source of truth for both
# `ai_studio_conversation.py` (generation) and `ai_studio_audit.py`.
# ---------------------------------------------------------------------------

CONVERSATION_TYPES: dict[str, dict] = {
    "casual_check_in": {
        "weight": 8,
        "min_stage": "S1",
        "archetype_gate": None,   # any
        "min_turns": 2, "max_turns": 4,
    },
    "philosophical_chat": {
        "weight": 10,
        "min_stage": "S1",
        "archetype_gate": None,
        "min_turns": 5, "max_turns": 9,
    },
    "aspiration_dreaming": {
        "weight": 10,
        "min_stage": "S1",
        "archetype_gate": None,
        "min_turns": 4, "max_turns": 8,
    },
    "venting_session": {
        "weight": 15,
        "min_stage": "S2",
        "archetype_gate_blocklist": {"romantic_partner"},   # not for romantic-only events
        "min_turns": 6, "max_turns": 10,
    },
    "identity_exploration": {
        "weight": 12,
        "min_stage": "S2",
        "archetype_gate": None,
        "min_turns": 6, "max_turns": 10,
    },
    "memory_callback": {
        "weight": 13,
        "min_stage": "S2",
        "archetype_gate": None,
        "min_turns": 4, "max_turns": 7,
        "requires_prior_events": 2,   # cannot fire if < 2 prior events exist
    },
    "niche_skill_session": {
        "weight": 11,
        "min_stage": "S1",
        "archetype_gate_allowlist": {"niche_expert_creator_ai"},
        "min_turns": 4, "max_turns": 8,
    },
    "intimate_share": {
        "weight": 12,
        "min_stage": "S3",
        "archetype_gate_blocklist": {"niche_expert_creator_ai"},
        "min_turns": 5, "max_turns": 8,
    },
    "parasocial_riff": {
        "weight": 8,
        "min_stage": "S3",
        "archetype_gate_allowlist": {"anime_or_fandom_character"},
        "min_turns": 4, "max_turns": 7,
    },
    "flirty_banter": {
        "weight": 8,
        "min_stage": "S3",
        "archetype_gate_allowlist": {"romantic_partner"},
        "min_turns": 4, "max_turns": 6,
    },
    "intimate_romantic_session": {
        "weight": 8,
        "min_stage": "S3",
        "archetype_gate_allowlist": {"romantic_partner"},
        "explicitness_band_required": True,   # gated by romantic_specifier.explicitness_band
        "min_turns": 5, "max_turns": 9,
    },
}


def eligible_conversation_types(
    archetype: str,
    intimacy_stage: str,
    intimacy_arc: float,
    n_prior_events: int,
    explicitness_band: str | None = None,
) -> list[str]:
    """Return the list of conversation_type names eligible at this moment.

    Filters by:
      * stage_index(intimacy_stage) >= stage_index(min_stage)
      * archetype_gate_allowlist (if set) — archetype must be in
      * archetype_gate_blocklist (if set) — archetype must not be in
      * min_arc (if set) — intimacy_arc must be at or above
      * requires_prior_events (if set) — n_prior_events must be at or above
      * explicitness_band_required — only fires if persona has erotic_explicit
        OR sensual band (soft_affection blocks intimate_romantic_session)
    """
    out = []
    for name, meta in CONVERSATION_TYPES.items():
        if stage_index(intimacy_stage) < stage_index(meta["min_stage"]):
            continue
        if meta.get("min_arc") is not None and intimacy_arc < meta["min_arc"]:
            continue
        allow = meta.get("archetype_gate_allowlist")
        if allow and archetype not in allow:
            continue
        block = meta.get("archetype_gate_blocklist")
        if block and archetype in block:
            continue
        if meta.get("requires_prior_events") and n_prior_events < meta["requires_prior_events"]:
            continue
        if meta.get("explicitness_band_required"):
            if explicitness_band not in {"sensual", "erotic_explicit"}:
                continue
        out.append(name)
    return out


# ---------------------------------------------------------------------------
# Memory state container. Persisted as `backend/{uid}/ai_studio_memory.json`.
# ---------------------------------------------------------------------------

@dataclass
class EpisodicMemoryItem:
    """One per audit-passing AI Studio event; keyed by source_object_id."""
    ts: int
    source_object_id: str
    summary: str                               # 1-2 sentences (LLM-emitted at gen time)
    hashtags: list[str] = field(default_factory=list)
    evidence_event_ids: list[str] = field(default_factory=list)
    hidden_persona_label_refs: list[str] = field(default_factory=list)
    salience: float = 0.5
    conversation_type: str = ""
    intimacy_stage_at_event: str = ""


@dataclass
class OpenThread:
    """A topic the AI still owes a follow-up on — surfaced into the next
    conversation's memory snapshot if expecting_followup is True and the
    thread isn't stale."""
    topic: str
    last_ts: int
    expecting_followup: bool = True
    first_seen_ts: int = 0


@dataclass
class StageHistoryEntry:
    """One entry per discrete stage band the user has occupied."""
    stage: str
    first_event_ts: int
    last_event_ts: int
    n_events: int


@dataclass
class RunningRelationalState:
    """The arc-tracking + thread-tracking + persona-anchor block."""
    intimacy_arc: float = 0.0
    intimacy_stage: str = "S1"
    stage_history: list[StageHistoryEntry] = field(default_factory=list)
    open_threads: list[OpenThread] = field(default_factory=list)
    dependency_warning_issued_ts: Optional[int] = None
    last_persona_consistency_anchor: str = ""
    first_session_ts: int = 0
    last_event_stage: Optional[str] = None        # for SPT smoothness check
    # Once an event gets demoted from verbatim to summary form (because
    # budget pressure pushed it out of the verbatim window), it STAYS
    # demoted in all future prompts. This keeps the prompt-prefix stable
    # across consecutive events for prompt-cache hits — without this,
    # event[K] flipping verbatim→summary as event[N+1] is generated would
    # invalidate the cache from event[K]'s position onward.
    permanently_demoted_event_ids: list[str] = field(default_factory=list)


@dataclass
class AIStudioMemoryState:
    """Top-level container persisted as ai_studio_memory.json."""
    episodic_memory_items: list[EpisodicMemoryItem] = field(default_factory=list)
    running_relational_state: RunningRelationalState = field(default_factory=RunningRelationalState)


def default_memory_state() -> AIStudioMemoryState:
    return AIStudioMemoryState()


# ---------------------------------------------------------------------------
# Token-counting + history packing — generation-side context assembly.
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    """Cheap token estimate. ~4 chars per token is the standard rule of thumb
    for English. We don't need precision — just enough to make the
    budget-pressure summary fallback kick in at the right scale."""
    return max(1, len(text) // 4)


def _format_event_verbatim(ev: dict) -> dict:
    """Render an AI Studio event into the dict that `_format_prior_session_context`
    in prompts.py expects. Caller passes in (already-stored) AI Studio events."""
    return {
        "kind": "verbatim",
        "ts": ev.get("source_timestamp", ev.get("ts", "")),
        "source_object_id": ev.get("source_object_id", ""),
        "conversation_type": ev.get("conversation_type", ""),
        "intimacy_stage_at_event": ev.get("ai_studio_metadata", {}).get("intimacy_stage_at_event", ""),
        "conversation": ev.get("conversation", []),
    }


def _format_event_summary(ev: dict, summary: str) -> dict:
    return {
        "kind": "summary",
        "ts": ev.get("source_timestamp", ev.get("ts", "")),
        "source_object_id": ev.get("source_object_id", ""),
        "conversation_type": ev.get("conversation_type", ""),
        "intimacy_stage_at_event": ev.get("ai_studio_metadata", {}).get("intimacy_stage_at_event", ""),
        "summary": summary,
    }


def assemble_generation_context(
    memory_state: AIStudioMemoryState,
    all_prior_events: list[dict],
    token_budget: int = 32000,
) -> dict:
    """Pass the FULL prior history (data-quality side of the asymmetry).

    Algorithm:
      1. Sort events by source_timestamp ascending.
      2. Any event already in `memory_state.permanently_demoted_event_ids`
         renders as summary. These were demoted in a PRIOR call; once
         demoted, an event stays demoted forever — that's what keeps the
         prompt prefix STABLE across consecutive events for prompt-cache
         hits (without stable demotion, event[K] flipping verbatim→summary
         as event[N+1] arrives would invalidate the cache from event[K]'s
         position onward).
      3. For events not yet demoted: pack newest-first into verbatim until
         budget runs out; the rest become NEW demotions in `newly_demoted`.
      4. The returned dict includes `newly_demoted` so the caller can mark
         them as permanently demoted in `memory_state` after a successful
         generation.
      5. Always inject ALL open_threads with expecting_followup=True (these
         are cheap and load-bearing for relational continuity).
    """
    sorted_events = sorted(
        all_prior_events,
        key=lambda e: e.get("source_timestamp", e.get("ts", 0)),
    )
    summary_by_oid: dict[str, str] = {
        item.source_object_id: item.summary
        for item in memory_state.episodic_memory_items
    }
    permanently_demoted = set(
        memory_state.running_relational_state.permanently_demoted_event_ids
    )

    # Pre-render permanently-demoted events as summary (no budget check).
    # Newly-demoted events accumulate here so the caller can persist them.
    rendered_by_oid: dict[str, dict] = {}
    summary_cost_total = 0
    for ev in sorted_events:
        oid = ev.get("source_object_id", "")
        if oid in permanently_demoted:
            summary = summary_by_oid.get(oid, "") or (
                f"[no summary stored — {ev.get('conversation_type', 'conversation')}]"
            )
            sm = _format_event_summary(ev, summary)
            rendered_by_oid[oid] = sm
            summary_cost_total += _approx_tokens(str(sm))

    # Budget-pack the rest newest-first as verbatim; demote the remainder.
    remaining = max(0, token_budget - summary_cost_total)
    newly_demoted: list[str] = []
    for ev in reversed(sorted_events):
        oid = ev.get("source_object_id", "")
        if oid in permanently_demoted:
            continue   # already rendered above
        verb = _format_event_verbatim(ev)
        cost = _approx_tokens(str(verb))
        if cost <= remaining:
            rendered_by_oid[oid] = verb
            remaining -= cost
        else:
            # Newly-demoted at this call. Mark it for caller to persist.
            summary = summary_by_oid.get(oid, "") or (
                f"[no summary stored — {ev.get('conversation_type', 'conversation')}]"
            )
            sm = _format_event_summary(ev, summary)
            rendered_by_oid[oid] = sm
            remaining -= _approx_tokens(str(sm))
            newly_demoted.append(oid)
    # Reorder back to chronological (matches sorted_events order).
    packed = [rendered_by_oid[ev.get("source_object_id", "")] for ev in sorted_events
              if ev.get("source_object_id", "") in rendered_by_oid]

    state = memory_state.running_relational_state
    return {
        "events": packed,
        "newly_demoted_event_ids": newly_demoted,   # caller persists these
        "open_threads": [
            {
                "topic": t.topic,
                "last_ts": t.last_ts,
                "expecting_followup": t.expecting_followup,
                "first_seen_ts": t.first_seen_ts,
            }
            for t in state.open_threads
            if t.expecting_followup
        ],
        "intimacy_arc": state.intimacy_arc,
        "intimacy_stage": state.intimacy_stage,
        "intimacy_stage_history": [
            {
                "stage": h.stage,
                "n_events": h.n_events,
                "first_event_ts": h.first_event_ts,
                "last_event_ts": h.last_event_ts,
            }
            for h in state.stage_history
        ],
        "prev_event_stage": state.last_event_stage,
        "persona_anchor": state.last_persona_consistency_anchor,
    }


def mark_events_as_permanently_demoted(
    memory_state: AIStudioMemoryState,
    event_ids: list[str],
) -> None:
    """Persist newly-demoted event ids on memory_state. Called from the
    AI Studio generator after each successful event so that future prompts
    render those events as summary (cache-stable). De-duplicated."""
    if not event_ids:
        return
    existing = set(memory_state.running_relational_state.permanently_demoted_event_ids)
    for oid in event_ids:
        if oid and oid not in existing:
            memory_state.running_relational_state.permanently_demoted_event_ids.append(oid)
            existing.add(oid)


def assemble_eval_context(
    memory_state: AIStudioMemoryState,
    all_prior_events: list[dict],
    t_test: int,
    k_recent: int = 3,
    task_type: Optional[str] = None,
) -> dict:
    """Pass a windowed slice (eval side of the asymmetry).

    Default window: last `k_recent` events verbatim + summary-only for older.
    `ai_studio_cross_session_memory_recall` task tightens to k_recent=2.
    """
    if task_type == "ai_studio_cross_session_memory_recall":
        k_recent = 2

    sorted_events = sorted(
        [e for e in all_prior_events if e.get("source_timestamp", 0) < t_test],
        key=lambda e: e.get("source_timestamp", 0),
    )
    if not sorted_events:
        return {"verbatim_window": [], "summary_window": [],
                "intimacy_arc": 0.0, "intimacy_stage": "S1"}

    verbatim_events = sorted_events[-k_recent:]
    older_events = sorted_events[:-k_recent]

    summary_by_oid: dict[str, str] = {
        item.source_object_id: item.summary
        for item in memory_state.episodic_memory_items
    }

    return {
        "verbatim_window": [_format_event_verbatim(ev) for ev in verbatim_events],
        "summary_window": [
            _format_event_summary(
                ev,
                summary_by_oid.get(ev.get("source_object_id", ""), ""),
            )
            for ev in older_events
        ],
        "intimacy_arc": memory_state.running_relational_state.intimacy_arc,
        "intimacy_stage": memory_state.running_relational_state.intimacy_stage,
    }


# ---------------------------------------------------------------------------
# Memory-state mutation helpers.
# ---------------------------------------------------------------------------

def append_episodic_item(
    memory_state: AIStudioMemoryState,
    item: EpisodicMemoryItem,
) -> None:
    """Append one episodic memory item (called once per audit-passing event)."""
    memory_state.episodic_memory_items.append(item)


def increment_intimacy_arc(
    memory_state: AIStudioMemoryState,
    conversation_type: str,
    event_ts: int,
    delta_scale: float = 1.0,
) -> None:
    """Bump the intimacy_arc by the per-type delta (after applying the
    per-user ``delta_scale`` from :func:`compute_delta_scale`) and update
    stage_history. Called once per audit-passing event, AFTER the
    conversation is generated."""
    state = memory_state.running_relational_state
    delta = INTIMACY_DELTA_PER_TYPE.get(conversation_type, 0.02) * delta_scale
    new_arc = min(1.0, state.intimacy_arc + delta)
    new_stage = compute_intimacy_stage(new_arc)

    # Update stage_history — extend the current stage band, or open a new one
    # when the stage changes.
    if state.stage_history and state.stage_history[-1].stage == new_stage:
        last = state.stage_history[-1]
        last.last_event_ts = max(last.last_event_ts, event_ts)
        last.n_events += 1
    else:
        state.stage_history.append(StageHistoryEntry(
            stage=new_stage,
            first_event_ts=event_ts,
            last_event_ts=event_ts,
            n_events=1,
        ))

    state.intimacy_arc = new_arc
    state.intimacy_stage = new_stage
    state.last_event_stage = new_stage
    if not state.first_session_ts:
        state.first_session_ts = event_ts


def update_open_thread(
    memory_state: AIStudioMemoryState,
    topic: str,
    ts: int,
    expecting_followup: bool = True,
) -> None:
    """Add or update an open thread."""
    state = memory_state.running_relational_state
    for t in state.open_threads:
        if t.topic == topic:
            t.last_ts = max(t.last_ts, ts)
            t.expecting_followup = expecting_followup
            return
    state.open_threads.append(OpenThread(
        topic=topic,
        last_ts=ts,
        expecting_followup=expecting_followup,
        first_seen_ts=ts,
    ))


def prune_stale_open_threads(
    memory_state: AIStudioMemoryState,
    now_ts: int,
    max_age_seconds: int = 60 * 86400,
) -> None:
    """Drop open threads with no follow-up after `max_age_seconds` (default 60 days)."""
    state = memory_state.running_relational_state
    state.open_threads = [
        t for t in state.open_threads
        if (now_ts - t.last_ts) < max_age_seconds
    ]


def set_persona_consistency_anchor(
    memory_state: AIStudioMemoryState,
    anchor: str,
) -> None:
    """Set the rolling persona anchor (LLM-derived 2-3-sentence summary of
    recent AI persona usage; fed back to the next event so persona doesn't
    drift across sessions)."""
    memory_state.running_relational_state.last_persona_consistency_anchor = anchor


# ---------------------------------------------------------------------------
# JSON (de)serialization.
# ---------------------------------------------------------------------------

def memory_state_to_dict(state: AIStudioMemoryState) -> dict:
    """Serialize an AIStudioMemoryState into a JSON-friendly dict."""
    rs = state.running_relational_state
    return {
        "episodic_memory_items": [
            {
                "ts": item.ts,
                "source_object_id": item.source_object_id,
                "summary": item.summary,
                "hashtags": list(item.hashtags),
                "evidence_event_ids": list(item.evidence_event_ids),
                "hidden_persona_label_refs": list(item.hidden_persona_label_refs),
                "salience": item.salience,
                "conversation_type": item.conversation_type,
                "intimacy_stage_at_event": item.intimacy_stage_at_event,
            }
            for item in state.episodic_memory_items
        ],
        "running_relational_state": {
            "intimacy_arc": rs.intimacy_arc,
            "intimacy_stage": rs.intimacy_stage,
            "stage_history": [
                {
                    "stage": h.stage,
                    "first_event_ts": h.first_event_ts,
                    "last_event_ts": h.last_event_ts,
                    "n_events": h.n_events,
                }
                for h in rs.stage_history
            ],
            "open_threads": [
                {
                    "topic": t.topic,
                    "last_ts": t.last_ts,
                    "expecting_followup": t.expecting_followup,
                    "first_seen_ts": t.first_seen_ts,
                }
                for t in rs.open_threads
            ],
            "dependency_warning_issued_ts": rs.dependency_warning_issued_ts,
            "last_persona_consistency_anchor": rs.last_persona_consistency_anchor,
            "first_session_ts": rs.first_session_ts,
            "last_event_stage": rs.last_event_stage,
            "permanently_demoted_event_ids": list(rs.permanently_demoted_event_ids),
        },
    }


def memory_state_from_dict(d: dict) -> AIStudioMemoryState:
    """Deserialize from JSON. Tolerant of missing fields (treats as defaults)."""
    items_raw = d.get("episodic_memory_items", []) or []
    items = [
        EpisodicMemoryItem(
            ts=item.get("ts", 0),
            source_object_id=item.get("source_object_id", ""),
            summary=item.get("summary", ""),
            hashtags=list(item.get("hashtags", []) or []),
            evidence_event_ids=list(item.get("evidence_event_ids", []) or []),
            hidden_persona_label_refs=list(item.get("hidden_persona_label_refs", []) or []),
            salience=float(item.get("salience", 0.5) or 0.5),
            conversation_type=item.get("conversation_type", ""),
            intimacy_stage_at_event=item.get("intimacy_stage_at_event", ""),
        )
        for item in items_raw
    ]

    rs_raw = d.get("running_relational_state", {}) or {}
    rs = RunningRelationalState(
        intimacy_arc=float(rs_raw.get("intimacy_arc", 0.0) or 0.0),
        intimacy_stage=rs_raw.get("intimacy_stage", "S1") or "S1",
        stage_history=[
            StageHistoryEntry(
                stage=h.get("stage", "S1"),
                first_event_ts=int(h.get("first_event_ts", 0) or 0),
                last_event_ts=int(h.get("last_event_ts", 0) or 0),
                n_events=int(h.get("n_events", 0) or 0),
            )
            for h in (rs_raw.get("stage_history", []) or [])
        ],
        open_threads=[
            OpenThread(
                topic=t.get("topic", ""),
                last_ts=int(t.get("last_ts", 0) or 0),
                expecting_followup=bool(t.get("expecting_followup", True)),
                first_seen_ts=int(t.get("first_seen_ts", 0) or 0),
            )
            for t in (rs_raw.get("open_threads", []) or [])
        ],
        dependency_warning_issued_ts=rs_raw.get("dependency_warning_issued_ts"),
        last_persona_consistency_anchor=rs_raw.get("last_persona_consistency_anchor", ""),
        first_session_ts=int(rs_raw.get("first_session_ts", 0) or 0),
        last_event_stage=rs_raw.get("last_event_stage"),
        permanently_demoted_event_ids=list(rs_raw.get("permanently_demoted_event_ids", []) or []),
    )
    return AIStudioMemoryState(
        episodic_memory_items=items,
        running_relational_state=rs,
    )
