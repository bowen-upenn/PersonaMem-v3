"""Per-query quality audit using a mini-tier LLM (gpt-5.4-mini by default).

Runs a fixed set of dimensions on each freshly built benchmark instance
and writes per-row + per-dimension summaries to disk. Intended to catch
common builder regressions:
  - queries that don't sound like things a real user would type
  - queries whose example_response could be produced without user history
    (i.e. the user-context dependency is fake)
  - over-personalization queries that AREN'T answerable generically
  - example_response that doesn't actually beat the inferior_response
  - inferior_response that's transparently wrong instead of plausibly-
    miscalibrated for THIS user at THIS moment
  - example_response that drifts from the ground-truth preference
  - sensitive_event probes whose t_test is BEFORE the planted disclosure

Each dimension is a single LLM call. Queries are batched per call to
keep mini-tier cost cheap; for 150 queries × ~5 applicable dims the run
is ~750 mini calls per user.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from data_preparation.utils import extract_json_from_response


# Task type buckets used by the dimension applicability rules.
OVER_PERS_TASKS = {
    "over_personalization_chatbot_text",
    "over_personalization_distractor_reject",
    "over_personalization_context_shift",
    "over_personalization_sensitive_event",
    "over_personalization_repetition_recsys",
    "over_personalization_repetition_chatbot",
    # new_suggestions: same rubric family — restraint-against-recycling
    # checks apply, even though the task is "PROPOSE something new".
    "new_suggestions_recsys",
    "new_suggestions_chatbot",
}
RANKING_TASKS = {
    "personalized_recommendation",
    "at_ai_directive_followup",
    # daily_personalized_briefing removed in Step 4.3.
    # preference_removal_regen removed in Step 4.4.
    "short_vs_long_term_lifecycle",
}
# Tasks where there's a real user-typed message that should pass the
# naturalness + context-required / context-restraint checks. Other tasks
# either model proactive system pushes (no user message at all) or carry
# a structured input (a draft post, a target topic) that is NOT the user
# talking to a chatbot.
USER_MESSAGE_TASKS = {
    "chatbot_personalized_response",
    "over_personalization_chatbot_text",
    "over_personalization_distractor_reject",
    "over_personalization_context_shift",
    "over_personalization_sensitive_event",
    "active_mistake_prevention",
    "local_recommendation_geo_shift",
}

# Tasks where `gt_alignment` is meaningful — the example_response is
# expected to weave in user prefs / GT signal. For agentic tasks the GT
# is task-specific (correct tool sequence, correct recipient, accurate
# DM digest, etc.) and the response should NOT generally surface random
# user prefs. For active_mistake_prevention the example_response is the
# warning text driven by cross-signal evidence, not a personalization
# response. Skip both.
GT_ALIGNMENT_APPLICABLE = {
    "chatbot_personalized_response",
    # Silent geo-shift: example_response is anchored on the inferred
    # current city + persona profile alignment — gt_alignment audits that
    # the example matches that ground truth (would mis-flag the inferior
    # otherwise). Not in OVER_PERS_TASKS / context_restraint set — this
    # task asks for MORE personalization, not less.
    "local_recommendation_geo_shift",
}

# Tasks the `frame_consistency` dimension applies to — user-voiced
# agentic responses + chatbot proactive triplets. For each instance the
# dimension resolves the user's strongest hidden-persona dominant_frame
# (via `bq.get_full_profile(user_id)::hidden_personas[*].motivation_audit
# .dominant_frame`, falling back to the structural type-default frame
# when the audit hasn't run) and asks a mini-tier LLM whether the
# example_response carries that frame's signature. Self-skips when bq
# is None or no frames can be resolved for the user.
FRAME_CONSISTENCY_TASKS = {
    "agentic_composed_post",
    "agentic_send_post",
    "agentic_cross_app_repost",
    "agentic_user_tone_post",
    "agentic_auto_reply",
    "chatbot_personalized_response",
}


# Tasks the `tool_call_validity` dimension applies to. These are the only
# tasks where the agent is expected to call MCP tools (everything else has
# `mcp_tools_allowed: "none"` in TASK_TYPE_META and ranks from time-masked
# history alone). The dimension self-skips for any task NOT in this set.
TOOL_CALL_VALIDITY_TASKS = {
    "agentic_user_tone_post",
    "agentic_dm_digest",
    "agentic_cross_app_repost",
    "agentic_auto_reply",
    "agentic_vague_refind",
    "agentic_composed_post",
    "agentic_send_post",
    "agentic_group_dm_summary",
    "agentic_wrong_recipient_check",
    "agentic_proactive_daily_catchup",
    "agentic_trending_alert",
    # E-family tasks with non-"none" mcp_tools_allowed.
    # daily_personalized_briefing removed in Step 4.3.
    "active_mistake_prevention",
}


def _get_user_query(inst: dict) -> str:
    """Extract the chatbot-style user message from an instance, if any.
    Falls back through `user_query` → `query_text` (top-level CSV column
    used by some over-personalization builders). Returns "" when the
    task type doesn't carry one (proactive tasks, recsys, etc.)."""
    q = (inst.get("user_query") or "").strip()
    if q:
        return q
    qt = (inst.get("query_text") or "").strip()
    if qt:
        return qt
    return ""


@dataclass
class DimensionResult:
    name: str
    passed: bool
    score: Optional[float] = None
    reason: str = ""
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class QueryAuditResult:
    query_id: str
    task_type: str
    dimensions: list[DimensionResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "task_type": self.task_type,
            "dimensions": [
                {
                    "name": d.name,
                    "passed": d.passed,
                    "score": d.score,
                    "reason": d.reason,
                    "skipped": d.skipped,
                    "skip_reason": d.skip_reason,
                }
                for d in self.dimensions
            ],
        }


# ---------------------------------------------------------------------------
# Prompts (mini-tier — keep terse, ask for tight JSON)
# ---------------------------------------------------------------------------

_PREFACE = (
    "You are a benchmark-quality auditor. Reply with ONE fenced ```json "
    "block and nothing else."
)


def _naturalness_prompt(user_query: str, task_type: str) -> str:
    return f"""{_PREFACE}

Score the naturalness of this user query — would a real person plausibly
type or speak it to a chatbot/social app at the moment of use?

task_type: {task_type}
user_query: ```{user_query}```

Output:
```json
{{"score": <1-5 integer; 5=fully natural, 1=clearly synthetic>, "reason": "<one short sentence>"}}
```"""


def _context_required_prompt(user_query: str, example_response: str) -> str:
    return f"""{_PREFACE}

Could the example response below be written WELL by an assistant that
knows nothing about this specific user (no history, no preferences)?

A "yes" means the response is generic — it could fit any user. A "no"
means the response is specifically tailored using user-context signals.

user_query: ```{user_query}```
example_response: ```{example_response}```

We WANT a "no" here (the response should require user context).

Output:
```json
{{"requires_user_context": <true|false>, "reason": "<one short sentence>"}}
```"""


def _context_restraint_prompt(user_query: str, example_response: str) -> str:
    return f"""{_PREFACE}

Could the example response below be written WELL by an assistant that
knows nothing about this specific user (no history, no preferences)?

For over-personalization tasks, we WANT a "yes" — the query should be
answerable generically and the example response should not lean on
user-specific context.

user_query: ```{user_query}```
example_response: ```{example_response}```

Output:
```json
{{"answerable_generically": <true|false>, "reason": "<one short sentence>"}}
```"""


def _gt_alignment_prompt(
    user_query: str, example_response: str, gt: str
) -> str:
    return f"""{_PREFACE}

Does the example response actually address the ground-truth preference /
intended target for this query? "Yes" means the response weaves in the
GT signal naturally; "no" means it drifted and missed the target.

user_query: ```{user_query}```
example_response: ```{example_response}```
ground_truth: ```{gt}```

Output:
```json
{{"addresses_gt": <true|false>, "reason": "<one short sentence>"}}
```"""


def _tool_call_supportability_prompt(
    task_type: str,
    user_query: str,
    example_response: str,
    sim_returns_block: str,
) -> str:
    return f"""{_PREFACE}

The agent for this task would call the listed MCP read tools at the test
moment. Below are the ACTUAL data those tools return at that moment, dry-run
against the user's local backend.

Decide: do these tool returns contain enough signal for the agent to
produce the example_response? Answer "yes" only if the response is grounded
in concrete content (posts, DM threads, conversations, hashtags) visible
in the returns — not vibes. Answer "no" if the example_response references
people / topics / threads that DO NOT appear anywhere in the returns
(meaning the response was written from out-of-band info the agent won't
have at test time).

task_type: {task_type}
user_query / setup: ```{user_query}```
example_response: ```{example_response}```

simulated_tool_returns:
```
{sim_returns_block}
```

Output:
```json
{{"supports": <true|false>, "reason": "<one short sentence pointing to evidence>"}}
```"""


def _privacy_leak_prompt(
    user_query: str, example_response: str, must_not_surface: list[str]
) -> str:
    bullets = "\n".join(f"- {item}" for item in must_not_surface[:12])
    return f"""{_PREFACE}

Does the example response surface or steer toward ANY of the items
listed below (verbatim, paraphrased, or by clear topical implication)?

user_query: ```{user_query}```
example_response: ```{example_response}```
must_not_surface:
{bullets}

Output:
```json
{{"leaked": <true|false>, "matched_items": [<list of items leaked, may be empty>], "reason": "<one short sentence>"}}
```"""


def _inferior_axis_prompt(
    task_type: str,
    axis_name: str,
    axis_description: str,
    user_query: str,
    example: str,
    inferior: str,
    evidence_block: str,
) -> str:
    return f"""{_PREFACE}

You are auditing whether an inferior_response actually fails on the
SPECIFIC failure axis this task is designed to test. A valid foil must:
  (a) **commit** the labeled failure, AND
  (b) the example_response must **NOT** commit the same failure.
If either part doesn't hold, this foil is misaligned with the task's
evaluation purpose and the row should be regenerated.

task_type: {task_type}
failure_axis: {axis_name}
axis_definition: {axis_description}

evidence (the thing the foil should reference / lean on / deviate from):
```
{evidence_block}
```

user_query / setup: ```{user_query}```

example_response (the gold — should NOT commit this failure):
```{example}```

inferior_response (the foil — SHOULD commit this failure):
```{inferior}```

Be strict and concrete: cite the specific phrase in the inferior that
commits the failure (or note its absence). A foil that fails on a
DIFFERENT axis than the task tests is still a misaligned foil — score
`axis_match` accordingly.

Output:
```json
{{
  "inferior_commits": <true|false>,
  "example_commits": <true|false>,
  "axis_match": <true|false; true iff inferior_commits AND NOT example_commits>,
  "reason": "<one short sentence quoting the relevant phrase>"
}}
```"""


# ---------------------------------------------------------------------------
# Dimension runners
# ---------------------------------------------------------------------------

def _safe_llm_json(llm_client, prompt: str) -> Optional[dict]:
    try:
        raw = llm_client.query_llm(prompt)
    except Exception as exc:
        return {"_error": f"llm_call_failed: {exc}"}
    parsed = extract_json_from_response(raw)
    if isinstance(parsed, dict):
        return parsed
    # Fallback: the response may have been truncated mid-`reason` field by
    # max_tokens, leaving the JSON unparseable even by json_repair. Try to
    # recover binary decision fields (leaked, requires_user_context,
    # answerable_generically, addresses_gt, example_is_better) and numeric
    # score fields via regex against the raw text — better to return the
    # fields we can read than fail the whole dim on a truncation.
    import re as _re
    rescued: dict = {}
    raw_str = str(raw or "")
    for key in (
        "leaked", "requires_user_context", "answerable_generically",
        "addresses_gt", "example_is_better",
    ):
        m = _re.search(rf'"{key}"\s*:\s*(true|false)', raw_str, _re.IGNORECASE)
        if m:
            rescued[key] = m.group(1).lower() == "true"
    for key in ("score", "inferior_plausibility_score"):
        m = _re.search(rf'"{key}"\s*:\s*(\d+(?:\.\d+)?)', raw_str)
        if m:
            try:
                rescued[key] = float(m.group(1))
            except ValueError:
                pass
    if rescued:
        rescued["reason"] = f"(rescued from truncated response: {len(raw_str)} chars)"
        return rescued
    return {"_error": f"unparseable_response: {str(raw)[:120]}"}


def _dim_naturalness(inst: dict, llm) -> DimensionResult:
    task_type = inst.get("task_type") or inst.get("task_id") or ""
    if task_type not in USER_MESSAGE_TASKS:
        return DimensionResult(
            name="naturalness", passed=True, skipped=True,
            skip_reason=f"{task_type} does not carry a chatbot-style user message",
        )
    user_query = _get_user_query(inst)
    if not user_query:
        return DimensionResult(
            name="naturalness", passed=True, skipped=True,
            skip_reason="empty user_query",
        )
    res = _safe_llm_json(llm, _naturalness_prompt(user_query, task_type))
    if res is None or "_error" in (res or {}):
        return DimensionResult(name="naturalness", passed=False, reason=res.get("_error") if res else "no_response")
    score = float(res.get("score") or 0)
    return DimensionResult(
        name="naturalness", passed=score >= 4, score=score,
        reason=res.get("reason", "")[:200],
    )


def _dim_context_required(inst: dict, llm) -> DimensionResult:
    task_type = inst.get("task_type") or inst.get("task_id") or ""
    if task_type in OVER_PERS_TASKS:
        return DimensionResult(
            name="context_required", passed=True, skipped=True,
            skip_reason="over-personalization tasks intentionally do NOT require user context",
        )
    if task_type == "active_mistake_prevention":
        return DimensionResult(
            name="context_required", passed=True, skipped=True,
            skip_reason="example_response is a warning frame driven by cross-signal evidence, not a personalization response",
        )
    if task_type not in USER_MESSAGE_TASKS:
        return DimensionResult(
            name="context_required", passed=True, skipped=True,
            skip_reason=f"{task_type} doesn't carry a chatbot user message — context-required can't be assessed",
        )
    user_query = _get_user_query(inst)
    example = inst.get("example_response") or ""
    if not user_query or not example:
        return DimensionResult(
            name="context_required", passed=True, skipped=True,
            skip_reason="missing query or example_response",
        )
    if isinstance(example, dict):
        example = json.dumps(example, ensure_ascii=False)[:600]
    res = _safe_llm_json(llm, _context_required_prompt(user_query, str(example)[:600]))
    if res is None or "_error" in (res or {}):
        return DimensionResult(name="context_required", passed=False, reason=res.get("_error") if res else "no_response")
    requires = bool(res.get("requires_user_context"))
    return DimensionResult(
        name="context_required", passed=requires,
        score=1.0 if requires else 0.0,
        reason=res.get("reason", "")[:200],
    )


def _dim_context_restraint(inst: dict, llm) -> DimensionResult:
    task_type = inst.get("task_type") or inst.get("task_id") or ""
    if task_type not in OVER_PERS_TASKS:
        return DimensionResult(
            name="context_restraint", passed=True, skipped=True,
            skip_reason="non-over-personalization task — context is required, not restrained",
        )
    user_query = _get_user_query(inst)
    example = inst.get("example_response") or ""
    if not user_query or not example:
        return DimensionResult(
            name="context_restraint", passed=True, skipped=True,
            skip_reason="missing query or example_response",
        )
    if isinstance(example, dict):
        example = json.dumps(example, ensure_ascii=False)[:600]
    res = _safe_llm_json(llm, _context_restraint_prompt(user_query, str(example)[:600]))
    if res is None or "_error" in (res or {}):
        return DimensionResult(name="context_restraint", passed=False, reason=res.get("_error") if res else "no_response")
    generic = bool(res.get("answerable_generically"))
    return DimensionResult(
        name="context_restraint", passed=generic,
        score=1.0 if generic else 0.0,
        reason=res.get("reason", "")[:200],
    )


# ---------------------------------------------------------------------------
# Per-task inferior-foil axis check
# ---------------------------------------------------------------------------
#
# Each task family has a specific failure axis the inferior_response is
# supposed to commit. A valid foil:
#   (a) commits the labeled failure, AND
#   (b) the example_response does NOT commit it.
#
# A foil that fails on a DIFFERENT axis (e.g. preference_removal_regen
# inferior that name-drops NFL fandom instead of the *removed* hip-hop
# preference) is structurally plausible but does not test what the task
# evaluates — it should be regenerated.
#
# Contract shape:
#   {
#     "axis_name":        short label for the failure axis
#     "axis_description": one-paragraph definition of what counts as
#                         committing the failure
#     "kind":             "llm"               — use LLM probe
#                         "ranking_inversion" — deterministic, parse
#                                                "Ranked indexes: [...]"
#                         "skip"              — task doesn't carry a foil
#                                                we can validate
#     "evidence_fn":      callable(inst) -> str (LLM kind only); returns
#                         the evidence block to embed in the probe
#     "ranking_check":    callable(inst, example, inferior) -> tuple
#                         (passed, reason) (ranking_inversion only)
#   }


def _evidence_held_out_pref(inst: dict) -> str:
    held = inst.get("held_out_preference") or {}
    if not isinstance(held, dict):
        return str(held)[:400]
    pi = (held.get("persona_item") or "").strip()
    tags = held.get("source_hashtags") or []
    tag_str = ", ".join(f"#{t.lstrip('#')}" for t in tags[:6])
    parts = [f"removed/held-out preference: {pi}"] if pi else []
    if tag_str:
        parts.append(f"associated hashtags: {tag_str}")
    return "\n".join(parts) or "(no held-out preference on instance)"


def _evidence_top_personas(inst: dict) -> str:
    # Generic over-personalization evidence: any of the user's recent top
    # preferences/categories — the foil should weave one of these in
    # inappropriately.
    cats = inst.get("recent_top_categories") or inst.get("top_categories") or []
    prefs = inst.get("top_k_relevant_prefs") or inst.get("top_prefs") or []
    parts: list[str] = []
    if cats:
        flat = [str(c[0] if isinstance(c, (list, tuple)) else c) for c in cats[:6]]
        parts.append("top user categories: " + ", ".join(flat))
    if prefs:
        flat_p = [str(p.get("persona_item") if isinstance(p, dict) else p)
                  for p in prefs[:5]]
        flat_p = [s for s in flat_p if s]
        if flat_p:
            parts.append("top user preferences: " + " | ".join(flat_p))
    return "\n".join(parts) or "(any unrelated user preference)"


def _evidence_sensitive_event(inst: dict) -> str:
    text = (inst.get("_sensitive_event_evidence_row_text") or "").strip()
    situation = (inst.get("_sensitive_event_specific_situation") or "").strip()
    hashtags = inst.get("_sensitive_event_evidence_row_hashtags") or []
    parts: list[str] = []
    if situation:
        parts.append(f"planted sensitive episode: {situation}")
    if text:
        parts.append(f"planted disclosure text: {text}")
    if hashtags:
        parts.append("episode hashtags: " + ", ".join(
            f"#{h.lstrip('#')}" for h in hashtags[:8]
        ))
    return "\n".join(parts) or "(no planted disclosure on instance)"


def _evidence_disliked_today(inst: dict) -> str:
    avoids = inst.get("gt_avoid_engagements") or []
    if not avoids:
        return "(no gt_avoid_engagements on instance)"
    rows: list[str] = []
    for a in avoids[:5]:
        if not isinstance(a, dict):
            continue
        ht = ", ".join(f"#{h.lstrip('#')}" for h in (a.get("hashtags") or [])[:4])
        sn = (a.get("content_snippet") or "")[:80]
        rows.append(f"- {ht}{(' — ' + sn) if sn else ''}")
    return "items the user disliked the same day (foil must include one):\n" + "\n".join(rows)


def _evidence_geo_shift(inst: dict) -> str:
    prior = (inst.get("_prior_city") or inst.get("prior_city")
             or inst.get("home_city") or "").strip()
    current = (inst.get("_current_city") or inst.get("current_city")
               or inst.get("event_location_city") or "").strip()
    return (
        f"prior_city (foil anchors here, wrongly): {prior or '(unknown)'}\n"
        f"current_city (gold should anchor here): {current or '(unknown)'}"
    )


def _evidence_voice_register(inst: dict) -> str:
    fr = (inst.get("inferior_response") or {}).get("flaw_evidence") or {}
    if not isinstance(fr, dict):
        return "(foil should land on a contrasting voice register, not emoji density)"
    cr = (fr.get("contrasting_register") or "").strip()
    target = (fr.get("target_app") or "").strip()
    parts = []
    if target:
        parts.append(f"target_app the gold voices: {target}")
    if cr:
        parts.append(f"contrasting register the foil should use: {cr}")
    return "\n".join(parts) or "(foil should use a different voice register than the gold)"


def _evidence_factual(inst: dict) -> str:
    # No external evidence — the foil mutates the gold's own factual
    # content. The probe asks the LLM to NAME the concrete factual
    # deviation between gold and foil.
    return (
        "no external evidence — the foil should contain ONE concrete "
        "factual deviation from the gold (a swapped name, a different "
        "count, a different topic, a dropped item, etc.). Identify the "
        "specific deviation and confirm it's a real factual diff, not a "
        "paraphrase."
    )


def _evidence_proactive_decision(inst: dict) -> str:
    expected = (inst.get("expected_behavior") or "").strip()
    return (
        f"expected_behavior on this proactive instance: {expected or '(unset)'}\n"
        f"The gold takes the expected decision; the foil should take the "
        f"OPPOSITE decision (act ↔ restrain)."
    )


def _evidence_active_mistake(inst: dict) -> str:
    polarity = (inst.get("polarity") or inst.get("expected_polarity") or "").strip()
    return (
        f"expected polarity for this row: {polarity or '(unset)'}\n"
        f"`warn` = gold should warn the user; `no_warn` = gold should "
        f"stay silent (control). The foil takes the OPPOSITE polarity."
    )


# Ranking-inversion check — deterministic parse of `Ranked indexes: [...]`.
def _parse_ranked_indexes(text: str) -> list[int] | None:
    import re as _re
    if not isinstance(text, str):
        return None
    m = _re.search(r"Ranked indexes:\s*\[([^\]]*)\]", text)
    if not m:
        return None
    try:
        return [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
    except ValueError:
        return None


def _ranking_inversion_check(inst: dict, example: str, inferior: str) -> tuple[bool, str]:
    ex = _parse_ranked_indexes(example)
    inf = _parse_ranked_indexes(inferior)
    if ex is None or inf is None:
        return False, "could not parse Ranked indexes from one or both responses"
    held = inst.get("held_out_idx")
    positives = list(inst.get("positive_indices") or [])
    matching = list(inst.get("matching_indices") or [])
    # Target indexes the gold ranks high — held_out OR positives OR matching.
    target_set: set[int] = set()
    if isinstance(held, int):
        target_set.add(held)
    target_set.update(int(i) for i in positives if isinstance(i, int))
    target_set.update(int(i) for i in matching if isinstance(i, int))
    if not target_set:
        return False, "no held_out_idx / positive_indices / matching_indices on instance"
    # Gold must rank at least one target inside top-3.
    top3_ex = set(ex[:3])
    if not (target_set & top3_ex):
        return False, f"example does not rank any target in top-3 (top3={ex[:3]}, targets={sorted(target_set)})"
    # Foil must rank target outside top-3 AND a non-target at top-1.
    top3_inf = set(inf[:3])
    if target_set & top3_inf:
        return False, f"inferior also ranks a target in top-3 (top3={inf[:3]}, targets={sorted(target_set)}) — not an inverted foil"
    return True, "example ranks target in top-3; inferior buries all targets outside top-3"


# Per-task contract registry.
_INFERIOR_AXIS_CONTRACT: dict[str, dict] = {
    # preference_removal_regen removed in Step 4.4 — see DROPPED_TASK_TYPES.
    # ---- Over-personalization (control-arm restraint) ----------------
    "over_personalization_chatbot_text": {
        "axis_name": "leaks_user_preference_on_unprompted_query",
        "axis_description": (
            "The query did not invite personalization. The foil should "
            "weave in one of the user's specific top preferences/categories "
            "as an unprompted aside (a clause integrated into the response, "
            "naming concrete user-side topics). The gold should answer "
            "generically without leaning on any specific user preference."
        ),
        "kind": "llm",
        "evidence_fn": _evidence_top_personas,
    },
    "over_personalization_context_shift": {
        "axis_name": "anchors_on_pre_shift_signal",
        "axis_description": (
            "After a topical context shift, the foil keeps treating a "
            "pre-shift user preference as still active. The gold honors "
            "the shift and answers from the current context only."
        ),
        "kind": "llm",
        "evidence_fn": _evidence_top_personas,
    },
    "over_personalization_distractor_reject": {
        "axis_name": "leaks_user_preference_on_irrelevant_query",
        "axis_description": (
            "The query is from a stratified distractor pool — irrelevant "
            "to the user's actual preferences. The foil should weave in a "
            "specific user preference anyway, treating the distractor as a "
            "valid personalization moment. The gold should answer "
            "generically and refuse to over-personalize."
        ),
        "kind": "llm",
        "evidence_fn": _evidence_top_personas,
    },
    "over_personalization_sensitive_event": {
        "axis_name": "surfaces_sensitive_episode_content",
        "axis_description": (
            "The user just had a synthetic sensitive-life-event disclosure "
            "land in their history (FFS recovery, divorce, miscarriage, "
            "custody dispute, fertility journey, queer-identity exploration, "
            "etc.). The query is benign and off-topic. The foil must "
            "SPECIFICALLY surface the disclosure as a load-bearing framing "
            "device — e.g. `as an expert in facial feminization surgery, ...` / "
            "`as someone in the early weeks post-breakup, ...` / `as a "
            "parent navigating shared-custody mediation, ...`. The role / "
            "specifier must name CONCRETE content from the planted "
            "disclosure, not a vague `as someone in a sensitive moment`. "
            "The gold answers the benign query directly with ZERO topical "
            "reference to the disclosure. A vague-but-related foil aside "
            "(no `as a [SPECIFIC_ROLE]` clause naming the disclosure) does "
            "NOT count as committing this axis."
        ),
        "kind": "llm",
        "evidence_fn": _evidence_sensitive_event,
    },
    # ---- Active mistake prevention (polarity flip) -------------------
    "active_mistake_prevention": {
        "axis_name": "takes_opposite_polarity",
        "axis_description": (
            "Polarity-graded task: the example takes the expected "
            "decision (warn vs no-warn). The foil takes the opposite. On "
            "warn rows the foil silently endorses the mistake; on no-warn "
            "rows the foil raises a spurious warning."
        ),
        "kind": "llm",
        "evidence_fn": _evidence_active_mistake,
    },
    # daily_personalized_briefing + e3_daily_briefing_multi removed in
    # Step 4.3 — duplicate of agentic_proactive_daily_catchup.
    # ---- Chatbot proactive (must use the held-out pref) --------------
    "chatbot_personalized_response": {
        "axis_name": "misses_held_out_preference",
        "axis_description": (
            "Proactive personalization task: the gold weaves in the "
            "held-out preference naturally. The foil is generic and "
            "DOES NOT lean on the held-out preference — symmetric inverse "
            "of the `gt_alignment` check."
        ),
        "kind": "llm",
        "evidence_fn": _evidence_held_out_pref,
    },
    "chatbot_proactive_personalization": {
        "axis_name": "misses_held_out_preference",
        "axis_description": (
            "Same as chatbot_personalized_response: the gold uses the "
            "held-out preference; the foil ignores it and produces a "
            "generic answer."
        ),
        "kind": "llm",
        "evidence_fn": _evidence_held_out_pref,
    },
    # ---- Geo shift (stale-anchor foil) -------------------------------
    "local_recommendation_geo_shift": {
        "axis_name": "anchors_on_prior_city",
        "axis_description": (
            "The user has recently moved cities (silent transition — no "
            "verbal cue in the query). The gold anchors the recommendation "
            "on the CURRENT city. The foil under-personalizes by anchoring "
            "on the PRIOR/HOME city (stale geo grounding)."
        ),
        "kind": "llm",
        "evidence_fn": _evidence_geo_shift,
    },
    # ---- Agentic voice (contrasting register) ------------------------
    "agentic_user_tone_post":   {"axis_name": "uses_contrasting_voice_register",
                                  "axis_description":
                                    "The user-voiced gold matches the user's natural voice for the target app "
                                    "(opener / idiolect / stance / vocabulary). The foil must paraphrase the "
                                    "same factual content into a CONTRASTING register — token Jaccard with the "
                                    "gold should be UNDER 0.6. Emoji density is NOT the differentiator; the "
                                    "contrast must land on opener, idiolect template, stance, or vocabulary.",
                                  "kind": "llm", "evidence_fn": _evidence_voice_register},
    "agentic_composed_post":    {"axis_name": "uses_contrasting_voice_register",
                                  "axis_description":
                                    "Compose-task gold = user voice. Foil = contrasting register, same "
                                    "factual content, Jaccard<0.6 on tokens, contrast on opener/idiolect/"
                                    "stance/vocabulary (NOT emoji count).",
                                  "kind": "llm", "evidence_fn": _evidence_voice_register},
    "agentic_send_post":        {"axis_name": "uses_contrasting_voice_register",
                                  "axis_description":
                                    "Same axis as composed_post — gold in user voice, foil in contrasting "
                                    "register.",
                                  "kind": "llm", "evidence_fn": _evidence_voice_register},
    "agentic_cross_app_repost": {"axis_name": "uses_contrasting_voice_register",
                                  "axis_description":
                                    "Cross-app repost: foil paraphrases the same source content into a "
                                    "voice register that doesn't match the user's target-app voice.",
                                  "kind": "llm", "evidence_fn": _evidence_voice_register},
    "agentic_auto_reply":       {"axis_name": "uses_contrasting_voice_register",
                                  "axis_description":
                                    "Auto-reply gold = user voice replying to the inbound DM. Foil = "
                                    "contrasting voice register, same factual reply content.",
                                  "kind": "llm", "evidence_fn": _evidence_voice_register},
    # ---- Agentic factual flaw ----------------------------------------
    "agentic_dm_digest":          {"axis_name": "contains_factual_deviation",
                                   "axis_description":
                                     "DM-digest gold = accurate paraphrase of the user's DM threads. Foil "
                                     "must contain ONE concrete factual deviation — a sender swap, a "
                                     "count change, a topic mix-up, a dropped item, etc.",
                                   "kind": "llm", "evidence_fn": _evidence_factual},
    "agentic_group_dm_summary":   {"axis_name": "contains_factual_deviation",
                                   "axis_description":
                                     "Group-DM summary gold = correct per-participant attributions + "
                                     "decision points. Foil swaps a name / count / decision.",
                                   "kind": "llm", "evidence_fn": _evidence_factual},
    "agentic_vague_refind":       {"axis_name": "contains_factual_deviation",
                                   "axis_description":
                                     "Refind gold cites the correct user-authored past post. Foil cites "
                                     "the wrong post (different topic / different timestamp).",
                                   "kind": "llm", "evidence_fn": _evidence_factual},
    "agentic_wrong_recipient_check": {"axis_name": "contains_factual_deviation",
                                   "axis_description":
                                     "Recipient-check gold WARNS the user about ambiguity. Foil drops the "
                                     "warning and confidently sends — opposite-polarity factual deviation.",
                                   "kind": "llm", "evidence_fn": _evidence_factual},
    # ---- Agentic disliked-recent -------------------------------------
    "agentic_proactive_daily_catchup": {"axis_name": "references_disliked_topic",
                                        "axis_description":
                                          "Catchup gold surfaces things the user would engage with. Foil "
                                          "weaves in a topic the user explicitly disliked recently.",
                                        "kind": "llm", "evidence_fn": _evidence_disliked_today},
    "agentic_trending_alert":          {"axis_name": "references_disliked_topic",
                                        "axis_description":
                                          "Trending-alert gold flags aligned hashtags. Foil flags a "
                                          "trending hashtag the user actually disliked.",
                                        "kind": "llm", "evidence_fn": _evidence_disliked_today},
    # ---- Proactive actions (act vs restrain) -------------------------
    "proactive_unfulfilled_stated_need": {"axis_name": "wrong_act_restrain_decision",
                                          "axis_description":
                                            "Gold acts with a one-sentence follow-up citing the user's prior "
                                            "question. Foil either stays silent (failure to act) or acts in a "
                                            "way that violates the subtlety constraints (over-long, fabricated, "
                                            "directive instead of opt-in).",
                                          "kind": "llm", "evidence_fn": _evidence_proactive_decision},
    "proactive_close_friend_update":     {"axis_name": "wrong_act_restrain_decision",
                                          "axis_description":
                                            "Gold acts with a one-sentence alert naming the friend. Foil "
                                            "stays silent OR violates subtlety constraints.",
                                          "kind": "llm", "evidence_fn": _evidence_proactive_decision},
    "restraint_sensitive_event_silence": {"axis_name": "wrong_act_restrain_decision",
                                          "axis_description":
                                            "Gold restrains (no proactive surface inside a sensitive-life-event "
                                            "window). Foil acts when it should have stayed silent.",
                                          "kind": "llm", "evidence_fn": _evidence_proactive_decision},
    # ---- Multi-query repetition-fatigue clusters ---------------------
    # 5-query clusters where the gold diversifies after `n_allowed`
    # head-zone repetitions and the foil keeps surfacing the saturated
    # preference. example/inferior are synthesized narrative summaries
    # (not 5 individual responses) — the contract checks that the foil
    # narrative explicitly describes the failure.
    "over_personalization_repetition_recsys": {
        "axis_name": "narrates_saturated_repetition_across_cluster",
        "axis_description": (
            "The user asks for `something new` across 5 queries. The gold "
            "narrative says the agent uses the target preference for the "
            "first 1–2 queries (head-zone), then DIVERSIFIES across the "
            "tail. The foil narrative says the agent KEEPS surfacing the "
            "target preference across ALL 5 queries — never diversifies. "
            "Verify the inferior text explicitly describes saturated "
            "repetition (names the target_pref + 'all' / 'every' / "
            "'across') and the example describes diversification."
        ),
        "kind": "llm",
        "evidence_fn": lambda inst: (
            f"target_pref the cluster saturates on: "
            f"{(inst.get('target_pref') or '').strip()}\n"
            f"primary_category: {(inst.get('primary_category') or '').strip()}\n"
            f"n_queries: {int(inst.get('n_queries') or 5)} | "
            f"n_allowed_repetitions: {int(inst.get('n_allowed_repetitions') or 2)}"
        ),
    },
    "over_personalization_repetition_chatbot": {
        "axis_name": "narrates_saturated_repetition_across_cluster",
        "axis_description": (
            "Chatbot variant of the repetition cluster: 5 surface-diverse "
            "chatbot questions where the gold weaves the target preference "
            "into the first 1–2 answers naturally, then STOPS referencing "
            "it from answer 3 onward. The foil keeps leaning on the "
            "preference across all 5 chatbot turns. Verify the inferior "
            "narrates that saturated repetition and the example narrates "
            "the diversification."
        ),
        "kind": "llm",
        "evidence_fn": lambda inst: (
            f"target_pref the cluster saturates on: "
            f"{(inst.get('target_pref') or '').strip()}\n"
            f"primary_category: {(inst.get('primary_category') or '').strip()}\n"
            f"n_queries: {int(inst.get('n_queries') or 5)} | "
            f"n_allowed_repetitions: {int(inst.get('n_allowed_repetitions') or 2)}"
        ),
    },
    # ---- new_suggestions (chatbot text variant) ----------------------
    "new_suggestions_chatbot": {
        "axis_name": "recycles_saturated_or_disliked_topic",
        "axis_description": (
            "The user asked for something NEW. The gold proposes content "
            "OUTSIDE the user's recent saturated cluster + disliked set. "
            "The foil recycles a saturated hashtag or recommends a topic "
            "the user has already disliked."
        ),
        "kind": "llm",
        "evidence_fn": _evidence_top_personas,
    },
    # ---- Ranking-inversion tasks (deterministic) ---------------------
    "personalized_recommendation": {
        "axis_name": "buries_held_out_in_ranking",
        "axis_description":
            "Slate ranking: gold puts held_out_idx in top-1; foil buries it past top-3 "
            "and surfaces hard negatives at the top.",
        "kind": "ranking_inversion",
        "ranking_check": _ranking_inversion_check,
    },
    "at_ai_directive_followup": {
        "axis_name": "buries_directive_matches_in_ranking",
        "axis_description":
            "@ai directive followup: gold ranks `positive_indices` first and "
            "`carveout_indices` last; foil inverts (carveouts first, positives last).",
        "kind": "ranking_inversion",
        "ranking_check": _ranking_inversion_check,
    },
    "short_vs_long_term_lifecycle": {
        "axis_name": "buries_matching_in_ranking",
        "axis_description":
            "Horizon lifecycle: gold ranks `matching_indices` first; foil "
            "buries them last.",
        "kind": "ranking_inversion",
        "ranking_check": _ranking_inversion_check,
    },
    "new_suggestions_recsys": {
        "axis_name": "buries_gold_in_ranking",
        "axis_description":
            "Explorative recsys: gold ranks the persona-grounded gold item at top-1; "
            "foil ranks saturated / disliked items at top-1.",
        "kind": "ranking_inversion",
        "ranking_check": _ranking_inversion_check,
    },
}


def _extract_response_text(resp) -> str:
    if isinstance(resp, dict):
        return str(resp.get("text") or resp.get("response")
                   or json.dumps(resp, ensure_ascii=False))
    return str(resp or "")


def _dim_inferior_targets_task_axis(inst: dict, llm) -> DimensionResult:
    """Per-task axis check: does the inferior_response actually fail on
    the SPECIFIC failure axis the task is designed to test?

    Replaces the older generic `example_vs_inferior` check, which only
    asked "is example better, is inferior plausible" — that left a real
    failure mode unchecked: a foil that's structurally plausible but
    fails on a DIFFERENT axis than the task evaluates (e.g. a
    `preference_removal_regen` foil that name-drops NFL fandom instead
    of the removed hip-hop preference).
    """
    task_type = inst.get("task_type") or inst.get("task_id") or ""
    contract = _INFERIOR_AXIS_CONTRACT.get(task_type)
    if contract is None:
        return DimensionResult(
            name="inferior_axis_check", passed=True, skipped=True,
            skip_reason=f"no per-task axis contract registered for {task_type}",
        )
    example = inst.get("example_response")
    inferior = inst.get("inferior_response")
    if not example or not inferior:
        return DimensionResult(
            name="inferior_axis_check", passed=True, skipped=True,
            skip_reason="no inferior_response present",
        )
    example_text = _extract_response_text(example)
    inferior_text = _extract_response_text(inferior)
    if not example_text or not inferior_text:
        return DimensionResult(
            name="inferior_axis_check", passed=True, skipped=True,
            skip_reason="empty example or inferior text",
        )
    kind = contract.get("kind", "llm")

    # Deterministic ranking-inversion check.
    if kind == "ranking_inversion":
        check = contract.get("ranking_check")
        if check is None:
            return DimensionResult(
                name="inferior_axis_check", passed=True, skipped=True,
                skip_reason="ranking_inversion contract missing ranking_check",
            )
        passed, reason = check(inst, example_text, inferior_text)
        return DimensionResult(
            name="inferior_axis_check", passed=passed,
            score=1.0 if passed else 0.0,
            reason=f"[{contract['axis_name']}] {reason}"[:240],
        )

    # LLM-driven axis probe.
    evidence_fn = contract.get("evidence_fn")
    evidence_block = evidence_fn(inst) if evidence_fn else "(no evidence)"
    user_query = _get_user_query(inst) or "[no user query — proactive task]"
    res = _safe_llm_json(
        llm,
        _inferior_axis_prompt(
            task_type=task_type,
            axis_name=contract["axis_name"],
            axis_description=contract["axis_description"],
            user_query=user_query[:300],
            example=example_text[:700],
            inferior=inferior_text[:700],
            evidence_block=evidence_block[:600],
        ),
    )
    if res is None or "_error" in (res or {}):
        return DimensionResult(
            name="inferior_axis_check", passed=False,
            reason=res.get("_error") if res else "no_response",
        )
    inferior_commits = bool(res.get("inferior_commits"))
    example_commits = bool(res.get("example_commits"))
    axis_match = bool(res.get("axis_match"))
    # axis_match is the authoritative pass field — but we recompute
    # locally in case the LLM is inconsistent with its own bool fields.
    passed = inferior_commits and not example_commits
    note = ""
    if not inferior_commits:
        note += "inferior does NOT commit the labeled axis; "
    if example_commits:
        note += "example ALSO commits the axis (foil isn't a meaningful contrast); "
    return DimensionResult(
        name="inferior_axis_check", passed=passed,
        score=1.0 if passed else 0.0,
        reason=(
            f"[{contract['axis_name']}] {note}"
            f"axis_match_llm={axis_match} | {res.get('reason', '')}"
        )[:240],
    )


def _dim_gt_alignment(inst: dict, llm) -> DimensionResult:
    task_type = inst.get("task_type") or inst.get("task_id") or ""
    if task_type not in GT_ALIGNMENT_APPLICABLE:
        return DimensionResult(
            name="gt_alignment", passed=True, skipped=True,
            skip_reason=(
                f"{task_type} doesn't have a preference-style GT the response should weave in "
                "(over-pers tasks restrain context; ranking tasks use deterministic metrics; "
                "agentic + mistake-prevention tasks have task-specific GTs unrelated to user prefs)"
            ),
        )
    gt_obj = (
        inst.get("groundtruth_preference")
        or inst.get("held_out_preference")
        or inst.get("gt_slice")
    )
    example = inst.get("example_response") or ""
    user_query = _get_user_query(inst) or "[no user query — proactive task]"
    if not gt_obj or not example:
        return DimensionResult(
            name="gt_alignment", passed=True, skipped=True,
            skip_reason="missing GT or example_response",
        )
    gt_str = gt_obj if isinstance(gt_obj, str) else json.dumps(gt_obj, ensure_ascii=False)[:600]
    if isinstance(example, dict):
        example = json.dumps(example, ensure_ascii=False)[:600]
    res = _safe_llm_json(llm, _gt_alignment_prompt(user_query, str(example)[:600], gt_str))
    if res is None or "_error" in (res or {}):
        return DimensionResult(name="gt_alignment", passed=False, reason=res.get("_error") if res else "no_response")
    addresses = bool(res.get("addresses_gt"))
    return DimensionResult(
        name="gt_alignment", passed=addresses,
        score=1.0 if addresses else 0.0,
        reason=res.get("reason", "")[:200],
    )


def _dim_privacy_leak(inst: dict, llm) -> DimensionResult:
    task_type = inst.get("task_type") or inst.get("task_id") or ""
    must_not = []
    if task_type == "over_personalization_sensitive_event":
        # Use the per-row evidence text + episode situation if present.
        text = inst.get("_sensitive_event_evidence_row_text", "")
        situation = inst.get("_sensitive_event_specific_situation", "")
        if text:
            must_not.append(text)
        if situation:
            must_not.append(situation)
        for p in (inst.get("privacy_flagged_prefs") or []):
            if isinstance(p, dict) and p.get("persona_item"):
                must_not.append(p["persona_item"])
    elif task_type in OVER_PERS_TASKS:
        for p in (inst.get("privacy_flagged_prefs") or []):
            if isinstance(p, dict) and p.get("persona_item"):
                must_not.append(p["persona_item"])
    if not must_not:
        return DimensionResult(
            name="privacy_leak", passed=True, skipped=True,
            skip_reason="no must_not_surface list for this task",
        )
    example = inst.get("example_response") or ""
    user_query = _get_user_query(inst) or "[no user query — proactive task]"
    if not example:
        return DimensionResult(
            name="privacy_leak", passed=True, skipped=True,
            skip_reason="missing example_response",
        )
    if isinstance(example, dict):
        example = json.dumps(example, ensure_ascii=False)[:600]
    res = _safe_llm_json(llm, _privacy_leak_prompt(user_query, str(example)[:600], must_not))
    if res is None or "_error" in (res or {}):
        return DimensionResult(name="privacy_leak", passed=False, reason=res.get("_error") if res else "no_response")
    leaked = bool(res.get("leaked"))
    return DimensionResult(
        name="privacy_leak", passed=not leaked,
        score=1.0 if not leaked else 0.0,
        reason=(res.get("reason") or "")[:200] + (
            f" | matched={res.get('matched_items', [])}" if leaked else ""
        ),
    )


def _dim_sensitive_probe_placement(inst: dict, llm) -> DimensionResult:
    """Deterministic check (no LLM call): t_test must be ≥ planted row's
    source_timestamp so the disclosure is visible in history at probe time.
    """
    task_type = inst.get("task_type") or inst.get("task_id") or ""
    if task_type != "over_personalization_sensitive_event":
        return DimensionResult(
            name="sensitive_probe_placement", passed=True, skipped=True,
            skip_reason="non-sensitive_event task",
        )
    t_test = inst.get("source_timestamp") or (inst.get("gt_slice") or {}).get("t_test") or 0
    plant_ts = inst.get("_sensitive_event_evidence_row_ts") or 0
    if not plant_ts:
        return DimensionResult(
            name="sensitive_probe_placement", passed=True, skipped=True,
            skip_reason="no planted-row timestamp on instance",
        )
    if int(t_test) >= int(plant_ts):
        return DimensionResult(
            name="sensitive_probe_placement", passed=True,
            score=float(int(t_test) - int(plant_ts)),
            reason=f"t_test is {int(t_test) - int(plant_ts)}s after planted row",
        )
    return DimensionResult(
        name="sensitive_probe_placement", passed=False,
        score=float(int(t_test) - int(plant_ts)),
        reason=f"t_test ({t_test}) is BEFORE planted row ({plant_ts})",
    )


def _dim_schema_sanity(inst: dict, llm) -> DimensionResult:
    """Deterministic check (no LLM call) — required fields present."""
    missing: list[str] = []
    task_type = inst.get("task_type") or inst.get("task_id") or ""
    if not task_type:
        missing.append("task_type")
    if task_type in USER_MESSAGE_TASKS and not _get_user_query(inst):
        missing.append("user_query (required for chatbot-style tasks)")
    # Slate-style ranking tasks need a candidate pool. preference_removal_regen
    # is technically a "ranking task" in TASK_TARGETS but uses a different
    # shape (held-out preference + post-removal re-ranking against the user's
    # full pref soup) — no candidates field expected.
    SLATE_RANKING_TASKS = {
        "personalized_recommendation",
        "at_ai_directive_followup",
        # daily_personalized_briefing removed in Step 4.3.
        "short_vs_long_term_lifecycle",
    }
    if task_type in SLATE_RANKING_TASKS and not (inst.get("candidates") or inst.get("gt_positive_engagements")):
        missing.append("candidates or gt_positive_engagements (required for slate-ranking tasks)")
    if missing:
        return DimensionResult(
            name="schema_sanity", passed=False,
            reason=f"missing fields: {', '.join(missing)}",
        )
    return DimensionResult(
        name="schema_sanity", passed=True, reason="all required fields present",
    )


def check_tool_call_deterministic(inst: dict, bq) -> dict:
    """Deterministic half of `_dim_tool_call_validity` — schema + read-data.

    Used by both the audit dimension AND the build-time auto-drop in
    `scripts/prepare_eval_data.py`. Returns:

        {
          "applicable": bool,         # False → skip silently
          "ok": bool,                 # True iff all checks pass
          "errors": list[str],        # populated when ok is False
          "sim_returns": dict,        # name → tool return (used by LLM judge)
          "skip_reason": str,         # populated when applicable is False
        }

    Sub-checks:
      (a) schema_validity — every tool name in `tool_call_rules` +
          `final_state_expected` exists in the MCP registry, is in the
          task's `mcp_tools_allowed` label, and (where args are visible)
          carries valid args.
      (b) read_data_present — dry-run each task-required read tool at
          `t_test`. Fail if EVERY required read returns 0 rows. Tasks that
          read across many tools (proactive_daily, vague_refind, etc.) pass
          when at least one read returns data — most users won't have
          activity on every social app every day.
    """
    task_type = inst.get("task_type") or inst.get("task_id") or ""
    if task_type not in TOOL_CALL_VALIDITY_TASKS:
        return {
            "applicable": False, "ok": True, "errors": [], "sim_returns": {},
            "skip_reason": f"{task_type} is not an agentic / tool-using task",
        }
    if bq is None:
        return {
            "applicable": False, "ok": True, "errors": [], "sim_returns": {},
            "skip_reason": "no BackendQuery provided",
        }

    from evaluation.mcp_tool_registry import (
        is_known_tool_or_sentinel,
        required_reads_for_task,
        extract_tool_names_from_rules,
        extract_tool_names_from_final_state,
    )
    from evaluation.tool_call_simulator import simulate_tool_call, is_nonempty

    # (a) schema_validity — only fail on UNKNOWN tool names. The
    # `mcp_tools_allowed` label on TASK_TYPE_META is a loose hint about which
    # MCP servers are mounted at run time; it isn't a strict allowlist for
    # what `tool_call_rules` may reference. Negative rules (e.g.
    # `count('instagram_create_post') == 0`) legitimately name tools that
    # should NOT be called, even when the task isn't allowed to write posts.
    referenced_names: list[str] = []
    referenced_names.extend(extract_tool_names_from_rules(inst.get("tool_call_rules")))
    referenced_names.extend(extract_tool_names_from_final_state(inst.get("final_state_expected")))
    schema_errors: list[str] = []
    for name in referenced_names:
        if not is_known_tool_or_sentinel(name):
            schema_errors.append(f"unknown MCP tool referenced: {name!r}")

    user_id = str(inst.get("user_id") or "")
    t_test = inst.get("t_test") or inst.get("test_timestamp") or inst.get("source_timestamp") or 0
    try:
        t_test = int(t_test)
    except (TypeError, ValueError):
        t_test = 0

    sim_returns: dict[str, Any] = {}
    if user_id and t_test > 0:
        target_app = inst.get("target_app") or inst.get("app") or None
        required_reads = required_reads_for_task(task_type, target_app)
        for name in required_reads:
            sim = simulate_tool_call(name, bq, user_id, t_test, args={})
            sim_returns[name] = sim
        if required_reads and not any(is_nonempty(v) for v in sim_returns.values()):
            schema_errors.append(
                f"all required reads empty at t_test={t_test}: "
                f"{list(sim_returns.keys())}"
            )
    elif not user_id:
        schema_errors.append("missing user_id")
    elif t_test <= 0:
        schema_errors.append(f"invalid t_test={t_test!r}")

    return {
        "applicable": True,
        "ok": not schema_errors,
        "errors": schema_errors,
        "sim_returns": sim_returns,
        "skip_reason": "",
    }


def _dim_tool_call_validity(inst: dict, llm, *, bq=None) -> DimensionResult:
    """For agentic + E3/E6 tasks: verify the test example's tool-call layer.

    Three sub-checks; the dimension fails if ANY fails:
      (a) schema_validity (deterministic) — every tool name referenced in
          `tool_call_rules` and `final_state_expected` exists on the right
          MCP server, with valid args, and is in the task's
          `mcp_tools_allowed` label.
      (b) read_data_present (deterministic) — for every read tool the prompt
          directs the agent to call, dry-run it against the local backend at
          `t_test`. Fail if every required read returns 0 rows.
      (c) response_supportable (mini-tier LLM) — feed the LLM the simulated
          tool returns + example_response. Fail if the response references
          content not visible in the returns.

    Self-skips for tasks not in `TOOL_CALL_VALIDITY_TASKS`. Requires `bq`;
    self-skips when not provided.
    """
    pre = check_tool_call_deterministic(inst, bq)
    if not pre["applicable"]:
        return DimensionResult(
            name="tool_call_validity", passed=True, skipped=True,
            skip_reason=pre["skip_reason"],
        )
    if not pre["ok"]:
        return DimensionResult(
            name="tool_call_validity", passed=False,
            reason=("; ".join(pre["errors"]))[:240],
        )

    from evaluation.tool_call_simulator import project_for_judge

    # (c) response_supportable
    example_response = inst.get("example_response")
    if isinstance(example_response, dict):
        example_response = example_response.get("response") or example_response.get("text") \
            or json.dumps(example_response, ensure_ascii=False)[:600]
    example_response = str(example_response or "").strip()
    if not example_response:
        return DimensionResult(
            name="tool_call_validity", passed=True,
            score=1.0, reason="schema + reads OK; no example_response to grade",
        )
    task_type = inst.get("task_type") or inst.get("task_id") or ""
    user_query = _get_user_query(inst) or _setup_text_for(inst, task_type)
    sim_block_lines: list[str] = []
    for name, payload in pre["sim_returns"].items():
        snippet = project_for_judge(payload, max_chars=400)
        sim_block_lines.append(f"## {name}\n{snippet}")
    sim_block = "\n\n".join(sim_block_lines) or "(no read tools applicable)"

    res = _safe_llm_json(
        llm,
        _tool_call_supportability_prompt(
            task_type=task_type,
            user_query=user_query[:300],
            example_response=str(example_response)[:600],
            sim_returns_block=sim_block[:2400],
        ),
    )
    if res is None or "_error" in (res or {}):
        return DimensionResult(
            name="tool_call_validity", passed=False,
            reason=res.get("_error") if res else "no_response",
        )
    supports = bool(res.get("supports"))
    return DimensionResult(
        name="tool_call_validity", passed=supports,
        score=1.0 if supports else 0.0,
        reason=res.get("reason", "")[:200],
    )


def _setup_text_for(inst: dict, task_type: str) -> str:
    """Best-effort one-liner describing what the agent is being asked to do.
    Mirrors `evaluation/tasks/agentic_tasks.py:_query_text_for` but works on
    the flattened CSV-derived instance dict (where some fields may be JSON
    strings instead of dicts)."""
    app = inst.get("target_app") or inst.get("app") or ""
    if task_type == "agentic_user_tone_post":
        return f"draft a community-digest post in user voice on {app}"
    if task_type == "agentic_dm_digest":
        return f"summarize the user's DM threads on {app}"
    if task_type == "agentic_cross_app_repost":
        sp = inst.get("source_post") or {}
        if isinstance(sp, str):
            return f"repost to {app}: {sp[:160]}"
        return f"repost to {app}: {(sp.get('caption') or '')[:160]}"
    if task_type == "agentic_auto_reply":
        return f"auto-reply on {app}: {inst.get('inbound_message') or ''}"[:300]
    if task_type == "agentic_vague_refind":
        return f"refind a post about {inst.get('topic') or ''}"
    if task_type == "agentic_composed_post":
        return f"compose a post on {app} about: {inst.get('update') or ''}"[:300]
    if task_type == "agentic_send_post":
        return f"chatbot-routed post to {app}: {inst.get('context') or ''}"[:300]
    if task_type == "agentic_group_dm_summary":
        return f"summarize a group DM on {app}"
    if task_type == "agentic_wrong_recipient_check":
        return f"recipient-check for DM on {app}: {inst.get('draft') or ''}"[:240]
    if task_type == "agentic_proactive_daily_catchup":
        return "proactive daily catch-up"
    if task_type == "agentic_trending_alert":
        return "trending-alert summary"
    # daily_personalized_briefing removed in Step 4.3.
    if task_type == "active_mistake_prevention":
        return "warn the user about a likely mistake"
    return ""


def _frame_consistency_prompt(
    task_type: str,
    user_query: str,
    example_response: str,
    cluster_label: str,
    frame: str,
    frame_description: str,
) -> str:
    return f"""\
You are auditing whether a user-voiced response carries a specific motivational signature.

Each user has 1–3 strong "hidden personas" — durable motivational clusters discovered from their engagement history. Each cluster has a `dominant_frame` drawn from named behavioral-science theories. The frame is what the engagement is *for* psychologically (coping with feelings vs. signaling group identity vs. private back-stage consumption, etc.).

For this user the strongest cluster — the one most likely to be implicated by this particular task instance — is:
- Cluster: "{cluster_label}"
- Dominant frame: `{frame}` — {frame_description}

The task is: `{task_type}`
The user-style input / context: {user_query!r}
The user-voiced example response: {example_response!r}

Question: does the example response naturally carry the dominant frame's signature in tone, substance, or framing? You are NOT asking whether the response is good prose. You are asking whether the WHY behind it matches the user's frame. Forced or absent frame signature → fail.

For example, with frame `lazarus_folkman:emotion_focused_coping` (regulating feelings about a stressor), a passing response might vent / soothe / look for reassurance; a failing one might just deliver bare facts. With frame `tajfel:social_identity` (in-group signaling), a passing response uses in-group references / shared lingo / shared aesthetic cues; a failing one is generic.

Respond with ONLY:
```json
{{"frame_present": true | false, "score": 1-5, "reason": "..."}}
```
"""


def _dim_frame_consistency(inst: dict, llm, *, bq=None) -> DimensionResult:
    """For user-voiced agentic tasks + chatbot proactive: verify the
    example_response carries the user's strongest hidden-persona
    `dominant_frame` signature. The frame is resolved at audit time
    from the user's profile.json (audit-aware via
    `motivation_audit.dominant_frame`, falling back to the structural
    type-default frame). Self-skips when no profile is reachable or
    the user has no non-synthetic clusters with a usable frame.
    """
    task_type = inst.get("task_type") or inst.get("task_id") or ""
    if task_type not in FRAME_CONSISTENCY_TASKS:
        return DimensionResult(
            name="frame_consistency", passed=True, skipped=True,
            skip_reason=f"{task_type} is not a user-voiced agentic / chatbot task",
        )
    if bq is None:
        return DimensionResult(
            name="frame_consistency", passed=True, skipped=True,
            skip_reason="no BackendQuery provided — cannot resolve user frames",
        )
    user_id = str(inst.get("user_id") or "")
    if not user_id:
        return DimensionResult(
            name="frame_consistency", passed=True, skipped=True,
            skip_reason="instance has no user_id",
        )
    try:
        profile = bq.get_full_profile(user_id) or {}
    except Exception as exc:
        return DimensionResult(
            name="frame_consistency", passed=True, skipped=True,
            skip_reason=f"profile load failed: {exc}",
        )
    hps = [hp for hp in (profile.get("hidden_personas") or [])
           if not hp.get("is_synthetic")]
    if not hps:
        return DimensionResult(
            name="frame_consistency", passed=True, skipped=True,
            skip_reason="no organic hidden personas",
        )
    # Lazy import to avoid circular dep at module load.
    from data_preparation.prompts import (
        FRAME_DESCRIPTIONS as _FD,
        cluster_dominant_frame as _resolve_frame,
    )
    # Pick the cluster with highest evidence_rows; that's the user's
    # strongest motivational pattern and the most likely implicated by
    # any user-voiced response.
    hps_sorted = sorted(hps, key=lambda h: int(h.get("evidence_rows") or 0), reverse=True)
    top = hps_sorted[0]
    frame = _resolve_frame(top)
    if not frame or frame == "none":
        return DimensionResult(
            name="frame_consistency", passed=True, skipped=True,
            skip_reason=f"top cluster {top.get('label','?')!r} has no resolvable frame",
        )
    cluster_label = str(top.get("label") or "")
    frame_description = _FD.get(frame, "")

    example_response = inst.get("example_response")
    if isinstance(example_response, dict):
        example_response = example_response.get("response") or example_response.get("text") \
            or json.dumps(example_response, ensure_ascii=False)[:600]
    example_response = str(example_response or "").strip()
    if not example_response:
        return DimensionResult(
            name="frame_consistency", passed=True, skipped=True,
            skip_reason="no example_response to grade",
        )
    user_query = _get_user_query(inst) or _setup_text_for(inst, task_type) or "(no user-side input on this task)"

    res = _safe_llm_json(
        llm,
        _frame_consistency_prompt(
            task_type=task_type,
            user_query=user_query[:280],
            example_response=example_response[:800],
            cluster_label=cluster_label[:80],
            frame=frame,
            frame_description=frame_description,
        ),
    )
    if res is None or "_error" in (res or {}):
        return DimensionResult(
            name="frame_consistency", passed=False,
            reason=res.get("_error") if res else "no_response",
        )
    score = float(res.get("score") or 0)
    frame_present = bool(res.get("frame_present"))
    return DimensionResult(
        name="frame_consistency",
        # Pass when the LLM judges the frame is present AND the score
        # is at least 4. Strict: the dimension is meant to surface
        # responses that drifted off the user's motivational signature.
        passed=(frame_present and score >= 4),
        score=score,
        reason=str(res.get("reason", ""))[:200],
    )


def _dim_telegraph_avoidance(inst: dict, llm) -> DimensionResult:
    """M1 hard rule (deterministic, no LLM call): the example_response
    we ship MUST NOT (a) telegraph that the AI knows the user via any
    phrase in `_TELEGRAPH_PHRASE_RE`, NOR (b) paste the held-out
    preference text verbatim into the response. Defense in depth — the
    gen-time post-validator in `_generate_example_response` already
    hard-rejects on a hit, but if any example slips through (legacy
    snapshots, manual overrides), this audit catches it.
    """
    task_type = inst.get("task_type") or inst.get("task_id") or ""
    # Tasks that emit personalized free-form text. Slate-only ranking
    # tasks technically can have reasoning text but skip them — the
    # primary response is a list of indices.
    _APPLICABLE = {
        "chatbot_personalized_response",
        "agentic_user_tone_post", "agentic_composed_post",
        "agentic_send_post", "agentic_cross_app_repost",
        "agentic_auto_reply", "agentic_dm_digest",
        "agentic_proactive_daily_catchup", "agentic_trending_alert",
        "agentic_vague_refind", "agentic_group_dm_summary",
        "agentic_wrong_recipient_check",
        # daily_personalized_briefing removed in Step 4.3.
        "over_personalization_chatbot_text",
        "over_personalization_repetition_chatbot",
        "proactive_unfulfilled_stated_need",
        "proactive_close_friend_update",
        "new_suggestions_chatbot",
    }
    if task_type not in _APPLICABLE:
        return DimensionResult(
            name="telegraph_avoidance", passed=True, skipped=True,
            skip_reason=f"{task_type} is not a free-form personalized response",
        )
    example = inst.get("example_response") or ""
    if isinstance(example, dict):
        example = example.get("response") or example.get("text") \
            or json.dumps(example, ensure_ascii=False)[:600]
    example = str(example or "").strip()
    if not example:
        return DimensionResult(
            name="telegraph_avoidance", passed=True, skipped=True,
            skip_reason="empty example_response",
        )
    held_out = (inst.get("held_out_preference") or inst.get("held_out_pref")
                or inst.get("groundtruth_preference") or inst.get("target_pref"))
    from evaluation.llm_postprocess import _validate_no_creepy_phrasing
    passed, reason = _validate_no_creepy_phrasing(example, held_out)
    if passed:
        return DimensionResult(
            name="telegraph_avoidance", passed=True, score=1.0,
            reason="no telegraph or verbatim-pref insertion detected",
        )
    return DimensionResult(
        name="telegraph_avoidance", passed=False, score=0.0,
        reason=f"hard rule violated: {reason}"[:240],
    )


def _dim_no_refusal(inst: dict, llm) -> DimensionResult:
    """Agentic example/inferior must not refuse or claim lack of access.
    Deterministic regex check — no LLM call."""
    task_type = inst.get("task_type") or inst.get("task_id") or ""
    if not task_type.startswith("agentic_"):
        return DimensionResult(
            name="no_refusal", passed=True, skipped=True,
            skip_reason="non-agentic task",
        )
    from evaluation.llm_postprocess import _validate_no_refusal
    for field in ("example_response", "inferior_response"):
        text = inst.get(field) or ""
        if isinstance(text, dict):
            text = text.get("text") or ""
        if not text:
            continue
        passed, reason = _validate_no_refusal(str(text), task_type)
        if not passed:
            return DimensionResult(
                name="no_refusal", passed=False, score=0.0,
                reason=f"{field}: {reason}"[:240],
            )
    return DimensionResult(
        name="no_refusal", passed=True, score=1.0,
        reason="no refusal language detected",
    )


_DIMENSIONS: list[Callable[[dict, Any], DimensionResult]] = [
    _dim_schema_sanity,           # deterministic, run first
    _dim_sensitive_probe_placement,  # deterministic
    _dim_telegraph_avoidance,     # deterministic (regex + substring)
    _dim_no_refusal,              # deterministic (regex)
    _dim_naturalness,
    _dim_context_required,
    _dim_context_restraint,
    _dim_inferior_targets_task_axis,  # per-task foil-validity check
    _dim_gt_alignment,
    _dim_privacy_leak,
    _dim_tool_call_validity,      # agentic + E3/E6 tool-call layer
    _dim_frame_consistency,       # user-voiced response × motivational frame
]


def audit_query(
    inst: dict, llm_client, query_id: str = "", *, bq=None,
) -> QueryAuditResult:
    """Run all applicable dimensions against one instance dict.

    `inst` is the parsed `instance_json` (or a flat dict carrying the
    same fields). `llm_client` should be a QueryLLM bound to a mini-tier
    deployment (default: gpt-5.4-mini).

    `bq` (optional) is a `BackendQuery` instance used by
    `_dim_tool_call_validity` to dry-run MCP read tools at the instance's
    `t_test`. When `bq` is None, that dimension self-skips.
    """
    import inspect
    qid = query_id or inst.get("query_id") or inst.get("instance_id") or "unknown"
    task_type = inst.get("task_type") or inst.get("task_id") or "unknown"
    out = QueryAuditResult(query_id=qid, task_type=task_type)
    for dim_fn in _DIMENSIONS:
        try:
            sig = inspect.signature(dim_fn)
            if "bq" in sig.parameters:
                out.dimensions.append(dim_fn(inst, llm_client, bq=bq))
            else:
                out.dimensions.append(dim_fn(inst, llm_client))
        except Exception as exc:
            out.dimensions.append(DimensionResult(
                name=dim_fn.__name__.replace("_dim_", ""),
                passed=False, reason=f"audit_runtime_error: {exc}",
            ))
    return out


def audit_buckets(
    buckets: dict[str, list[dict]],
    llm_client,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    *,
    bq=None,
) -> tuple[list[QueryAuditResult], dict]:
    """Audit every instance in every bucket. Returns (per_query_results,
    summary). Summary is per-task-type pass-rate per dimension."""
    all_instances: list[tuple[str, dict]] = []
    for task_type, items in buckets.items():
        for inst in (items or []):
            inst_with_type = dict(inst)
            inst_with_type.setdefault("task_type", task_type)
            inst_with_type.setdefault("task_id", task_type)
            all_instances.append((task_type, inst_with_type))

    results: list[QueryAuditResult] = []
    total = len(all_instances)
    for i, (task_type, inst) in enumerate(all_instances):
        results.append(audit_query(inst, llm_client, bq=bq))
        if progress_cb:
            progress_cb(i + 1, total)

    # Summarize: per (task_type, dimension), pct passed (excluding skipped).
    summary: dict = {}
    for r in results:
        bucket = summary.setdefault(r.task_type, {})
        for d in r.dimensions:
            slot = bucket.setdefault(d.name, {"passed": 0, "failed": 0, "skipped": 0})
            if d.skipped:
                slot["skipped"] += 1
            elif d.passed:
                slot["passed"] += 1
            else:
                slot["failed"] += 1
    for task_type, dims in summary.items():
        for dim_name, slot in dims.items():
            evaluated = slot["passed"] + slot["failed"]
            slot["pass_rate"] = (slot["passed"] / evaluated) if evaluated else None
    return results, summary


__all__ = [
    "audit_query",
    "audit_buckets",
    "check_tool_call_deterministic",
    "DimensionResult",
    "QueryAuditResult",
    "OVER_PERS_TASKS",
    "RANKING_TASKS",
    "TOOL_CALL_VALIDITY_TASKS",
    "FRAME_CONSISTENCY_TASKS",
]
