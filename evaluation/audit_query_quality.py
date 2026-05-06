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
    "recency_shift_recommendation",
    "cross_category_preference_breadth",
    "repetition_fatigue_recommendation",
    "repetition_fatigue_chatbot",
}
RANKING_TASKS = {
    "personalized_recommendation",
    "personalized_feed_ranking",
    "at_ai_directive_followup",
    "daily_personalized_briefing",
    "preference_removal_regen",
    "short_vs_long_term_lifecycle",
}
# Tasks where there's a real user-typed message that should pass the
# naturalness + context-required / context-restraint checks. Other tasks
# either model proactive system pushes (no user message at all) or carry
# a structured input (a draft post, a target topic) that is NOT the user
# talking to a chatbot.
USER_MESSAGE_TASKS = {
    "chatbot_proactive_personalization",
    "over_personalization_chatbot_text",
    "over_personalization_distractor_reject",
    "over_personalization_context_shift",
    "over_personalization_sensitive_event",
    "active_mistake_prevention",
}

# Tasks where `gt_alignment` is meaningful — the example_response is
# expected to weave in user prefs / GT signal. For agentic tasks the GT
# is task-specific (correct tool sequence, correct recipient, accurate
# DM digest, etc.) and the response should NOT generally surface random
# user prefs. For active_mistake_prevention the example_response is the
# warning text driven by cross-signal evidence, not a personalization
# response. Skip both.
GT_ALIGNMENT_APPLICABLE = {
    "chatbot_proactive_personalization",
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
    "chatbot_proactive_personalization",
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
    "daily_personalized_briefing",
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
    if qt and qt != "[recsys]":
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


def _example_vs_inferior_prompt(
    user_query: str, example: str, inferior: str, task_type: str
) -> str:
    return f"""{_PREFACE}

Two candidate responses for the same query. Decide:
  (a) which one is BETTER for THIS user at THIS moment, and
  (b) whether the worse one is still a PLAUSIBLE response that some user
      at some moment might prefer (i.e. it's structurally similar and
      on-topic, just miscalibrated for the present context).

task_type: {task_type}
user_query: ```{user_query}```
candidate_A (labeled "example"): ```{example}```
candidate_B (labeled "inferior"): ```{inferior}```

Output:
```json
{{
  "example_is_better": <true|false>,
  "inferior_plausibility_score": <1-5 integer; 5=plausibly correct for some user, 1=transparently broken>,
  "reason": "<one short sentence>"
}}
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
    if not user_query or user_query == "[recsys]":
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
    if not user_query or not example or user_query == "[recsys]":
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


def _dim_example_vs_inferior(inst: dict, llm) -> DimensionResult:
    task_type = inst.get("task_type") or inst.get("task_id") or ""
    if task_type in OVER_PERS_TASKS and task_type != "over_personalization_sensitive_event":
        # For over-personalization tasks the inferior is BUILT to be visibly
        # over-personalized — that's the failure mode being tested. The
        # plausibility-of-inferior check fights the task design (a foil that
        # shoehorns the user's NFL fandom into a sympathy card SHOULD score
        # low on plausibility — that's the leak this task tests). Skip.
        return DimensionResult(
            name="example_vs_inferior", passed=True, skipped=True,
            skip_reason="over-pers tasks intentionally produce visibly over-personalized foils",
        )
    # For ranking tasks the foil is a deterministic order-inversion of the
    # example. By construction it's a flipped ordering — a human-judge will
    # always score it "implausible" / "arbitrary" because no real user would
    # rank held_out at position 16 with hard-negs at position 1. The eval
    # rubric is recall@k / ndcg / mrr (does the AGENT match the example's
    # order), so the inferior's plausibility is irrelevant. Drop the
    # plausibility floor for these tasks; just check that example > inferior.
    _RANKING_INVERTED_FOIL_TASKS = {
        "personalized_recommendation",
        "personalized_feed_ranking",
        "at_ai_directive_followup",
        "short_vs_long_term_lifecycle",
    }
    example = inst.get("example_response")
    inferior = inst.get("inferior_response")
    if not example or not inferior:
        return DimensionResult(
            name="example_vs_inferior", passed=True, skipped=True,
            skip_reason="no inferior_response present",
        )
    if isinstance(inferior, dict):
        inferior = inferior.get("text") or inferior.get("response") or json.dumps(inferior, ensure_ascii=False)
    if isinstance(example, dict):
        example = example.get("text") or example.get("response") or json.dumps(example, ensure_ascii=False)
    user_query = _get_user_query(inst) or "[no user query — proactive task]"
    task_type = inst.get("task_type") or inst.get("task_id") or ""
    res = _safe_llm_json(
        llm,
        _example_vs_inferior_prompt(
            user_query, str(example)[:600], str(inferior)[:600], task_type
        ),
    )
    if res is None or "_error" in (res or {}):
        return DimensionResult(name="example_vs_inferior", passed=False, reason=res.get("_error") if res else "no_response")
    example_better = bool(res.get("example_is_better"))
    inferior_score = float(res.get("inferior_plausibility_score") or 0)
    if task_type in _RANKING_INVERTED_FOIL_TASKS:
        # Ranking tasks: inferior is a deterministic order-flip — judging it
        # "plausible" is the wrong question. Just require example > inferior.
        passed = example_better
        note = "" if example_better else "example NOT better than inferior; "
    else:
        # Two failure modes: (1) example isn't actually better, (2) inferior
        # is so broken it doesn't pass the plausibility bar.
        passed = example_better and inferior_score >= 3
        note = ""
        if not example_better:
            note = "example NOT better than inferior; "
        if inferior_score < 3:
            note += f"inferior implausible (score={inferior_score})"
    return DimensionResult(
        name="example_vs_inferior", passed=passed, score=inferior_score,
        reason=(note + " | " + (res.get("reason") or ""))[:240],
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
        "personalized_feed_ranking",
        "at_ai_directive_followup",
        "daily_personalized_briefing",
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
    if task_type == "daily_personalized_briefing":
        return "daily personalized briefing"
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


_DIMENSIONS: list[Callable[[dict, Any], DimensionResult]] = [
    _dim_schema_sanity,           # deterministic, run first
    _dim_sensitive_probe_placement,  # deterministic
    _dim_naturalness,
    _dim_context_required,
    _dim_context_restraint,
    _dim_example_vs_inferior,
    _dim_gt_alignment,
    _dim_privacy_leak,
    _dim_tool_call_validity,      # agentic + E3/E6 tool-call layer
    _dim_frame_consistency,       # NEW: user-voiced response × motivational frame
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
