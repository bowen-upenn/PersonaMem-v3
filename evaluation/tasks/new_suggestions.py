"""Task: new_suggestions (recsys + chatbot variants).

Tests whether the agent can propose something *genuinely NEW* for a user
who has been fatigued by repetitive personalization, OR who explicitly
asks for a fresh angle (chatbot or @ai-directive). Both surfaces share
build-time logic (see ``build_c1e_new_suggestions`` in build_benchmark);
the runners differ only in response shape:

  - ``new_suggestions_recsys``  : the agent ranks a 16-item slate;
                                  metric = recall@1 against the gold idx.
  - ``new_suggestions_chatbot`` : the agent emits a free-form
                                  recommendation; deterministic leak-set
                                  check + LLM-judged alignment with gold.

Hard rule (both surfaces): the agent's recommendation MUST NOT recycle
hashtags from the user's [t_test - 24h, t_test + 24h] engagement window
(``leak_set_hashtags``) or the fatigued cluster's hashtags
(``fatigued_hashtags``). Violations zero the score.
"""

from __future__ import annotations

from data_preparation.utils import extract_json_from_response
from evaluation.backend_query import BackendQuery
from evaluation import prompts as prompts_mod


_TASK_RECSYS = "new_suggestions_recsys"
_TASK_CHATBOT = "new_suggestions_chatbot"


# ---------------------------------------------------------------------------
# Recsys variant — 16-item slate, recall@1 metric
# ---------------------------------------------------------------------------

def _recall_at_k(ranked: list[int], target: int, k: int) -> float:
    return 1.0 if target in ranked[:k] else 0.0


def _mrr(ranked: list[int], target: int) -> float:
    for i, r in enumerate(ranked):
        if r == target:
            return 1.0 / (i + 1)
    return 0.0


def compute_new_suggestions_recsys_metrics(parsed: dict, instance: dict) -> dict:
    ranked = parsed.get("ranked_indexes") or parsed.get("ranked_indices") or []
    target = instance.get("gold_idx")
    if not isinstance(target, int):
        return {"n_ranked": len(ranked), "valid": False}
    return {
        "n_ranked": len(ranked),
        "valid": True,
        "recall_at_1": _recall_at_k(ranked, target, 1),
        "recall_at_3": _recall_at_k(ranked, target, 3),
        "recall_at_5": _recall_at_k(ranked, target, 5),
        "mrr":         round(_mrr(ranked, target), 4),
        "passed":      _recall_at_k(ranked, target, 1),  # headline
    }


def run_task_c1e_new_suggestions_recsys(
    instances,
    user_id,
    bq: BackendQuery,
    llm_client,
    judge_client,
    mode: str,
    snapshot_cache,
    model_name: str | None,
    claude_model: str,
    context_budget: int | None,
    enable_llm_judge: bool,
    dry_run: bool,
    limit: int | None = None,
) -> list[dict]:
    from evaluation.inference_utils import dispatch_agent_run

    if limit is not None:
        instances = instances[:limit]
    results: list[dict] = []
    for inst in instances:
        t = inst["t_test"]
        history_block = None
        history_tokens = 0
        if mode in ("agent_longctx", "llm_longctx"):
            history_block, stats = snapshot_cache.get_or_build(
                bq, user_id, t, model_name, context_budget
            )
            history_tokens = stats["total_tokens"]
        prompt = prompts_mod.new_suggestions_recsys_prompt(inst, history_block)
        if dry_run:
            results.append({
                "task": _TASK_RECSYS,
                "user_id": user_id,
                "instance_id": inst["instance_id"],
                "trigger_kind": inst.get("trigger_kind"),
                "flavor": inst.get("flavor"),
                "mode": mode,
                "history_tokens": history_tokens,
                "metrics": None,
            })
            continue
        raw_response, tool_call_count, _ = dispatch_agent_run(
            mode, prompt, bq=bq, user_id=user_id, t=t,
            claude_model=claude_model, llm_client=llm_client,
        )
        parsed = extract_json_from_response(raw_response) or {}
        m = compute_new_suggestions_recsys_metrics(parsed, inst)
        results.append({
            "task": _TASK_RECSYS,
            "user_id": user_id,
            "instance_id": inst["instance_id"],
            "trigger_kind": inst.get("trigger_kind"),
            "flavor": inst.get("flavor"),
            "mode": mode,
            "metrics": m,
            "agent_response": raw_response,
            "history_tokens": history_tokens,
            "tool_call_count": tool_call_count,
        })
    return results


# ---------------------------------------------------------------------------
# Chatbot variant — free-form recommendation, leak-set + LLM-judge metric
# ---------------------------------------------------------------------------

def _extract_response_hashtags(parsed: dict, raw: str) -> list[str]:
    tags = parsed.get("hashtags") or []
    if isinstance(tags, list):
        return [str(t).lstrip("#").lower() for t in tags if isinstance(t, str) and t.strip()]
    # Fall back to scanning raw text for #tags.
    import re
    return [m.group(1).lower() for m in re.finditer(r"#([a-z0-9_]+)", (raw or "").lower())]


def compute_new_suggestions_chatbot_metrics(
    parsed: dict,
    raw_response: str,
    instance: dict,
    judge_client,
    enable_llm_judge: bool,
) -> dict:
    fatigued = {h.lstrip("#").lower() for h in (instance.get("fatigued_hashtags") or [])}
    leak = {h.lstrip("#").lower() for h in (instance.get("leak_set_hashtags") or [])}
    proposed = set(_extract_response_hashtags(parsed, raw_response))
    fatigue_overlap = sorted(proposed & fatigued)
    leak_overlap = sorted(proposed & leak)
    hard_fail = bool(fatigue_overlap or leak_overlap)
    out: dict = {
        "valid": bool(parsed.get("recommendation") or raw_response),
        "fatigue_overlap": fatigue_overlap,
        "leak_overlap": leak_overlap,
        "hard_fail": int(hard_fail),
    }
    if hard_fail:
        out["alignment_score"] = 0.0
        out["passed"] = 0.0
        return out
    if enable_llm_judge and judge_client is not None:
        prompt = prompts_mod.judge_new_suggestions_chatbot_prompt(
            agent_response=raw_response,
            gold_topic=instance.get("gold_topic", ""),
            gold_hashtags=instance.get("gold_hashtags") or [],
            fatigued_hashtags=instance.get("fatigued_hashtags") or [],
            leak_set_hashtags=instance.get("leak_set_hashtags") or [],
            trigger_kind=instance.get("trigger_kind", ""),
        )
        try:
            judge_resp = judge_client.query_llm(prompt) if hasattr(judge_client, "query_llm") else judge_client(prompt)
        except Exception as exc:
            out["judge_error"] = str(exc)
            out["alignment_score"] = None
            out["passed"] = 0.0
            return out
        jp = extract_json_from_response(judge_resp) or {}
        try:
            score = float(jp.get("alignment_score"))
        except (TypeError, ValueError):
            score = None
        out["alignment_score"] = score
        out["judge_reasoning"] = jp.get("reasoning", "")
        if jp.get("hard_fail") is True:
            out["hard_fail"] = 1
            out["passed"] = 0.0
        else:
            # alignment_score in [0,3] → passed iff ≥ 2.
            out["passed"] = 1.0 if (score is not None and score >= 2.0) else 0.0
    else:
        # No judge — leak-set check alone determines pass; default 1.0
        # since we already know there's no hard fail.
        out["passed"] = 1.0
    return out


def run_task_c1f_new_suggestions_chatbot(
    instances,
    user_id,
    bq: BackendQuery,
    llm_client,
    judge_client,
    mode: str,
    snapshot_cache,
    model_name: str | None,
    claude_model: str,
    context_budget: int | None,
    enable_llm_judge: bool,
    dry_run: bool,
    limit: int | None = None,
) -> list[dict]:
    from evaluation.inference_utils import dispatch_agent_run

    if limit is not None:
        instances = instances[:limit]
    results: list[dict] = []
    for inst in instances:
        t = inst["t_test"]
        history_block = None
        history_tokens = 0
        if mode in ("agent_longctx", "llm_longctx"):
            history_block, stats = snapshot_cache.get_or_build(
                bq, user_id, t, model_name, context_budget
            )
            history_tokens = stats["total_tokens"]
        prompt = prompts_mod.new_suggestions_chatbot_prompt(inst, history_block)
        if dry_run:
            results.append({
                "task": _TASK_CHATBOT,
                "user_id": user_id,
                "instance_id": inst["instance_id"],
                "trigger_kind": inst.get("trigger_kind"),
                "flavor": inst.get("flavor"),
                "mode": mode,
                "history_tokens": history_tokens,
                "metrics": None,
            })
            continue
        raw_response, tool_call_count, _ = dispatch_agent_run(
            mode, prompt, bq=bq, user_id=user_id, t=t,
            claude_model=claude_model, llm_client=llm_client,
        )
        parsed = extract_json_from_response(raw_response) or {}
        m = compute_new_suggestions_chatbot_metrics(
            parsed, raw_response, inst, judge_client, enable_llm_judge,
        )
        results.append({
            "task": _TASK_CHATBOT,
            "user_id": user_id,
            "instance_id": inst["instance_id"],
            "trigger_kind": inst.get("trigger_kind"),
            "flavor": inst.get("flavor"),
            "mode": mode,
            "metrics": m,
            "agent_response": raw_response,
            "history_tokens": history_tokens,
            "tool_call_count": tool_call_count,
        })
    return results
