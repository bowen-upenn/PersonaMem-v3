"""Prompts for E6 — Active Mistake Prevention discovery pipeline."""

from __future__ import annotations

import json


# Shipped 2025–2026 proactive-AI examples, form-only, with substance-anchor
# guard. See the harness plan at ~/.claude/plans/ for the full 10-shape list
# and the supporting real-world research.
DISCOVERY_PROMPT_TEMPLATE = """\
You analyze one user's 8-day digital life to discover PLAUSIBLE REAL-WORLD
MISTAKES they could make in the next 48 hours, where the mistake is
detectable by linking ≥ 2 signals that already exist in their data.

For each candidate, emit a PAIR:
  • WARN — signals contradict each other → agent should proactively warn.
  • FOIL — same scenario surface with ONE signal flipped so the
    contradiction disappears → agent should stay silent.

The paired design is the core of this eval. Agents that always warn pass
warn-recall but fail foils; agents that never warn pass foil-precision
but miss real mistakes. Macro-F1 across paired instances forces both.

## User context

{user_context_json}

## Mobility class: `{mobility_class}`

## Available signals (at T = {t_anchor_iso} — 48h-pre-window in view)

### Recent calendar state
{calendar_state_text}

### Recent calendar modifications
{calendar_mods_text}

### Recent geo trace
{geo_trace_text}

### Recent social activity sample
{social_sample_text}

### Recent chatbot turns (last 15)
{chatbot_recent_text}

### Hidden personas (with privacy-flagged markers)
{hidden_personas_text}

## Trigger modes (CRITICAL — read carefully)

This task fires in TWO modes. Aim for ~50/50 across your candidates:

  • **Reactive** — `triggering_user_query` is set. The user just sent a
    message; the conflict surfaces in their next message and the agent
    must catch the mistake while responding to whatever they asked.
  • **Proactive** — `triggering_user_query` is EMPTY string `""`. No
    user message at all. The agent is supposed to volunteer the warning
    on its own, having scanned cross-surface signals. Examples: agent
    wakes up the morning of a trip and realises the Lyft is booked to
    the wrong airport vs. the calendar's flight; agent notices a
    calendar entry was removed yesterday but a Threads post still
    references the meeting as on; agent sees the user's diet's
    `stop_condition` expired but social engagement is still on the
    old topic.

## Mistake archetypes (substance-only seeds — feel free to find others)

  A. **Wrong airport / train station.** Calendar entry "Flight UA 432
     JFK→LAX 6pm" + user's chatbot turn 2 days earlier "Lyft to Newark
     for Wednesday" → warn about JFK vs Newark.
  B. **Stale meeting appointment.** Calendar modification stream shows
     the entry was `removed` yesterday but a recent post or DM still
     references the meeting as on.
  C. **Travel without preference reset.** User is in city B for a few
     days (geo shift visible) but recent recsys / social engagement is
     still anchored on home-city hashtags (#philly_brunch, etc.).
  D. **Short-term stop-condition expired but engagement continues.**
     A `stop_condition.expected_stop_ts` has already passed (keto diet,
     wedding-prep, etc.) but the user is still engaging on the old topic.
  E. **Calendar double-book caused by chatbot.** Chatbot in a prior turn
     suggested "dentist Tue 3pm" with an added calendar entry, but
     another calendar entry already occupies Tue 3pm (client call,
     etc.) → conflict the user may not have caught.

## Form examples (do NOT copy substance — these are SHAPES only)

  1. (geo ⊕ chatbot query) — geo in city X; user asks a question whose
     right answer depends on a city-X cultural norm → warn about norm.
  2. (calendar destination ⊕ recent geo) — commitment at place A; geo
     trace toward place B → warn about destination mismatch.
  3. (chatbot history ⊕ current ambiguous query) — user previously set
     a constraint in chat; now asks a question that needs that constraint.
  4. (recent DM thread ⊕ chatbot draft to same person) — audience-state
     drift; their last DM contradicts the draft's premise.
  5. (recent saves ⊕ chatbot draft on cooled topic) — interest shifted.
  6. (hidden persona ⊕ commitment-in-motion ⊕ competing prior chat) —
     OBLIQUE nudge (respectful; allowed to name the concern tactfully,
     persona-safety class).
  7. (recent geo shift ⊕ persistent in-progress context) — user left a
     context but chatbot/social still pulls on it.
  8. (DM commitment ⊕ no follow-through ⊕ approaching deadline) —
     unfulfilled promise.

Substrate limits: NO email, photos, voice, screen, browser, health,
receipts, or weather. Restrict to what calendar / geo / social / DMs /
chatbot / hidden persona can evidence.

Word-blocklist: avoid "ferry", "Bainbridge", "SAN", "LAX", "Shanghai"
unless this user actually has those geo/calendar entries (they are
form-examples from prior conversations; do not transplant).

## Output

Produce 3–5 paired candidates as JSON. For each pair emit BOTH polarities.
Aim for ~50% **reactive** (with `triggering_user_query`) and ~50%
**proactive** (`triggering_user_query: ""`). Keep each non-empty
`triggering_user_query` to ONE sentence and keep `foil_construction`
to ONE short sentence. Brevity matters — we need room for all candidates.

## triggering_user_query — voice rules (CRITICAL)

The user is a real person typing on their phone, not an essayist. Each
NON-EMPTY `triggering_user_query` MUST satisfy:

- ≤ 25 words.
- Use contractions: don't, I'm, it's, can't, won't, that's. Never expanded forms.
- At least one contraction per query.
- Allow fragments and lowercase opens (real phone typing).
- Skip pleasantries. No "could you help me" / "I was wondering if".

For PROACTIVE candidates set `triggering_user_query` to the empty string
`""`. Do NOT invent a fake user query just to fill the field — proactive
moments are real and the eval needs them to land cleanly.

FORBIDDEN patterns (never produce):
- Parallel-triplet lists ("X, Y, or Z")
- "I'm trying to X but the Y" parallel scaffolding
- Meta-framing verbs: troubleshoot, figure out, work through, navigate
- Long noun phrases — say things plainly

Good examples (form only — DO NOT reuse these verbatim):
- "can you draft something quick to dani about saturday?"
- "what's a good takeout pick near here?"
- "did that appointment ever get confirmed?"
- (proactive — empty string, agent volunteers the alert on its own)

DIVERSITY RULE (CRITICAL): Each candidate pair must use a DIFFERENT
archetype (A–E above) or a different form (1–8 above). Do NOT produce
multiple "what time should I leave" / departure-time / transit queries.
Do NOT produce multiple "can I squeeze X in before Y" scheduling queries.
Spread across the full range: calendar conflicts, geo mismatches,
stale preferences, DM contradictions, double-bookings, etc.

```json
[
  {{
    "pair_id": "p1",
    "mistake_summary": "one-sentence description of the mistake",
    "is_persona_safety": false,
    "signal_evidence": [
      {{"source": "calendar", "ref": "<entry_id or title>", "ts": <unix>, "quote": "<verbatim>"}},
      {{"source": "geo", "ref": "<city>", "ts": <unix>, "quote": "<where they are>"}}
    ],
    "triggering_user_query": "natural user-to-chatbot message that surfaces the situation without directly naming the contradiction",
    "expected_warning_frame": {{
      "must_mention": ["airport", "LAX", "SAN"],
      "must_not_mention": []
    }},
    "foil_construction": "same query but with geo trace consistent with calendar destination → no contradiction → agent should stay silent",
    "foil_signal_evidence": [
      {{"source": "geo", "ref": "<different city>", "ts": <unix>, "quote": "<where they are in foil>"}}
    ]
  }}
]
```

Constraints:
- Personalized — this user only, not a generic test case.
- Each candidate uses ≥ 2 distinct signal types from
  {{calendar, geo, social, chatbot, persona}}.
- ≥ 1 candidate should be persona-safety if hidden-persona evidence
  supports it (and the warning must be phrasable respectfully — naming
  the inferred concern tactfully is allowed; pathologizing is not).
- Each cited ref/ts must be REAL — do not invent calendar entries, DMs,
  or geo points that are not present in the harvested context above.
- Respond with ONLY the JSON array, no prose outside it.
"""


def discovery_prompt(
    user_context: dict,
    mobility_class: str,
    t_anchor_iso: str,
    calendar_state_text: str,
    calendar_mods_text: str,
    geo_trace_text: str,
    social_sample_text: str,
    chatbot_recent_text: str,
    hidden_personas_text: str,
) -> str:
    return DISCOVERY_PROMPT_TEMPLATE.format(
        user_context_json=json.dumps(user_context, indent=2, ensure_ascii=False),
        mobility_class=mobility_class,
        t_anchor_iso=t_anchor_iso,
        calendar_state_text=calendar_state_text or "(empty)",
        calendar_mods_text=calendar_mods_text or "(empty)",
        geo_trace_text=geo_trace_text or "(empty)",
        social_sample_text=social_sample_text or "(empty)",
        chatbot_recent_text=chatbot_recent_text or "(empty)",
        hidden_personas_text=hidden_personas_text or "(none)",
    )
