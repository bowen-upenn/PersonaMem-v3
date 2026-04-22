"""Optional LLM-as-judge wrappers. Only invoked when --enable_llm_judge is set.

Each judge function takes a QueryLLM client (the **judge** model, separate from
the agent model), the focused evidence slice, and returns a dict of scores.
Failures return an empty dict — the harness should still emit hard metrics.
"""

from __future__ import annotations

import json

from data_preparation.utils import extract_json_from_response
from evaluation import prompts


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
