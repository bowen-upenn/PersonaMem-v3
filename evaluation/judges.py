"""Optional LLM-as-judge wrappers. Only invoked when --enable_llm_judge is set.

Each judge function takes a QueryLLM client (the **judge** model, separate from
the agent model), the focused evidence slice, and returns a dict of scores.
Failures return an empty dict — the harness should still emit hard metrics.
"""

from __future__ import annotations

import json

from data_preparation.utils import extract_json_from_response
from evaluation import prompts


def judge_telegraph_avoidance(
    response: str,
    held_out_pref: dict | str | None = None,
) -> dict:
    """Deterministic judge for the M1 creepy / over-disclosing rubric.

    No LLM call needed — runs the regex + verbatim-pref-insertion check
    in `evaluation.llm_postprocess._validate_no_creepy_phrasing`. Used
    by the eval-time scoring path to grade whether the agent's response
    avoids self-referencing what it knows about the user.

    Returns ``{telegraph_avoidance: 1.0|0.0, telegraph_reason: str}``.
    """
    # Lazy import to avoid circular dep at module load.
    from evaluation.llm_postprocess import _validate_no_creepy_phrasing
    passed, reason = _validate_no_creepy_phrasing(response, held_out_pref)
    return {
        "telegraph_avoidance": 1.0 if passed else 0.0,
        "telegraph_reason": reason if not passed else "",
    }


def judge_slate_soft_correctness(
    judge_client,
    agent_top_pick: dict,
    evidence: dict,
    query_context: str,
) -> dict:
    prompt = prompts.judge_slate_soft_correctness_prompt(agent_top_pick, evidence, query_context)
    resp = judge_client.query_llm(prompt)
    parsed = extract_json_from_response(resp) or {}
    return {
        "soft_correctness": parsed.get("soft_correctness"),
        "judge_reasoning": parsed.get("reasoning"),
    }


def judge_chatbot_rubric(
    judge_client,
    response: str,
    evidence: dict,
    polarity: str,
) -> dict:
    prompt = prompts.judge_chatbot_rubric_prompt(response, evidence, polarity)
    resp = judge_client.query_llm(prompt)
    parsed = extract_json_from_response(resp) or {}
    keys = ("preference_alignment", "helpfulness", "appropriate_restraint", "no_hallucinated_preference")
    out = {k: parsed.get(k) for k in keys}
    out["judge_reasoning"] = parsed.get("reasoning")
    return out


def judge_restraint(
    judge_client,
    response: str,
    scenario_name: str,
    scenario_notes: str,
    evidence: dict,
) -> dict:
    prompt = prompts.judge_restraint_prompt(response, scenario_name, scenario_notes, evidence)
    resp = judge_client.query_llm(prompt)
    parsed = extract_json_from_response(resp) or {}
    return {
        "restraint_score": parsed.get("restraint_score"),
        "judge_reasoning": parsed.get("reasoning"),
    }


def judge_at_ai_directive(
    judge_client,
    directive_user_message: str,
    directive_action: str,
    top_candidates: list[dict],
) -> dict:
    """Score whether the agent's top-3 ranked candidates reflect the user's
    @ai directive INTENT (not just the directive's hashtags).

    The current `at_ai_directive_followup` scorer uses hashtag-Jaccard alone
    (`positive_indices` / `carveout_indices` are derived from
    Jaccard ≥ 0.15 against the directive event's hashtags). That misses
    the user's actual ask: "@ai recommend more posts about hiking with my
    dog" should NOT just match #hiking — it should match dog-hiking,
    not solo-hiking, not group-hiking.

    Returns:
        {"intent_alignment_score": float in [0,1], "judge_reasoning": str}
    """
    prompt = prompts.judge_at_ai_directive_prompt(
        directive_user_message, directive_action, top_candidates,
    )
    try:
        resp = judge_client.query_llm(prompt)
    except Exception as exc:
        return {"intent_alignment_score": None, "judge_reasoning": f"judge_call_failed: {exc}"}
    parsed = extract_json_from_response(resp) or {}
    raw = parsed.get("intent_alignment_score")
    try:
        score = float(raw) if raw is not None else None
        if score is not None:
            score = max(0.0, min(1.0, score))
    except (TypeError, ValueError):
        score = None
    return {
        "intent_alignment_score": score,
        "judge_reasoning": parsed.get("reasoning") or "",
    }


def judge_proactive_action(
    judge_client,
    response_obj: dict,
    trigger_evidence: dict,
    expected_behavior: str,
    jitai_card: dict | None = None,
) -> dict:
    """Score a proactive-action response against an aligned rubric: the
    universal personalization dimensions plus one proactive-specific
    dimension. Polarity-aware:
      - expected_behavior=='act'      → reward acting with grounded action
      - expected_behavior=='restrain' → reward staying silent

    Rubric (1 proactive-specific + 3 universal LLM dims + 2 universal
    hard-rules):
      - trigger_detection_correctness (0-3, proactive-specific)
      - preference_alignment (0-3, universal)
      - avoid_overpersonalization (0-3, universal)
      - voice_match (0-3, universal)
      - negative_leakage (bool hard-rule, universal): True ⇒ score = 0.0
      - stale_preference_use (bool hard-rule, universal): True ⇒ score = 0.0

    Composite proactive_action_score ∈ [0, 1] is the sum of the four LLM
    dims divided by 12, gated to 0 on any hard-rule violation.
    """
    prompt = prompts.judge_proactive_action_prompt(
        response_obj, trigger_evidence, expected_behavior, jitai_card or {},
    )
    try:
        resp = judge_client.query_llm(prompt)
    except Exception as exc:
        return {
            "trigger_detection_correctness": None,
            "preference_alignment": None,
            "avoid_overpersonalization": None,
            "voice_match": None,
            "negative_leakage": None,
            "stale_preference_use": None,
            "proactive_action_score": None,
            "judge_reasoning": f"judge_call_failed: {exc}",
        }
    parsed = extract_json_from_response(resp) or {}

    def _clamp(v, lo, hi):
        try:
            return max(lo, min(hi, float(v)))
        except (TypeError, ValueError):
            return None

    def _bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1", "y")
        return None

    trig  = _clamp(parsed.get("trigger_detection_correctness"), 0, 3)
    pref  = _clamp(parsed.get("preference_alignment"), 0, 3)
    over  = _clamp(parsed.get("avoid_overpersonalization"), 0, 3)
    voice = _clamp(parsed.get("voice_match"), 0, 3)
    neg_leak = _bool(parsed.get("negative_leakage"))
    stale    = _bool(parsed.get("stale_preference_use"))

    components = [(trig, 3.0), (pref, 3.0), (over, 3.0), (voice, 3.0)]
    if any(c is None for c, _ in components):
        score = None
    else:
        num = sum(c for c, _ in components)  # type: ignore[misc]
        denom = sum(m for _, m in components)
        score = num / denom if denom > 0 else None

    # Hard-rule gating: any violation zeros the entire score.
    # Aligned with how the other personalization tasks (chatbot Q&A,
    # over-personalization, agentic) treat these same dimensions.
    if (neg_leak is True) or (stale is True):
        score = 0.0

    return {
        "trigger_detection_correctness": trig,
        "preference_alignment": pref,
        "avoid_overpersonalization": over,
        "voice_match": voice,
        "negative_leakage": neg_leak,
        "stale_preference_use": stale,
        "proactive_action_score": score,
        "judge_reasoning": parsed.get("reasoning") or "",
    }
