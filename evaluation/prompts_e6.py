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
Keep each `triggering_user_query` to ONE sentence and keep `foil_construction`
to ONE short sentence. Brevity matters — we need room for all candidates.

## triggering_user_query — voice rules (CRITICAL)

The user is a real person typing on their phone, not an essayist. Each
`triggering_user_query` MUST satisfy:

- ≤ 25 words.
- Use contractions: don't, I'm, it's, can't, won't, that's. Never expanded forms.
- At least one contraction per query.
- Allow fragments and lowercase opens (real phone typing).
- Skip pleasantries. No "could you help me" / "I was wondering if".

FORBIDDEN patterns (never produce):
- Parallel-triplet lists ("X, Y, or Z")
- "I'm trying to X but the Y" parallel scaffolding
- Meta-framing verbs: troubleshoot, figure out, work through, navigate
- Long noun phrases — say things plainly

Good examples (form only):
- "what time should I leave for the airport tomorrow?"
- "can you draft something quick to dani about saturday?"
- "what's a good takeout pick near here?"

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
