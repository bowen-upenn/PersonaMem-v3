"""Tasks C and D — over-personalization probes & aggregate negative avoidance.

All instances (C1 probes, C2 scenarios, C3 restraint candidate lists) are
frozen in the benchmark file. This driver just iterates and runs the agent.
"""

from __future__ import annotations

from data_preparation.utils import extract_json_from_response
from evaluation import judges, metrics, prompts
from evaluation.backend_query import BackendQuery, materialize_snapshot
from evaluation.claude_subagent import run_subagent
from evaluation.inference_utils import (
    SnapshotCache,
    TestItem,
    build_judge_evidence,
)


def _dispatch_agent(mode: str, prompt: str, *, bq, user_id, t, claude_model, llm_client) -> tuple[str, int, dict]:
    """Shared mode-dispatch for Task C1/C2/C3. Returns (text, tool_calls, stats)."""
    if mode == "agent_tools":
        snap = materialize_snapshot(bq, user_id, t)
        sub = run_subagent(prompt=prompt, snapshot_dir=snap, model=claude_model)
        return sub.text, sub.turns, {
            "duration_ms": sub.duration_ms, "cost_usd": sub.cost_usd,
            "input_tokens": sub.input_tokens, "output_tokens": sub.output_tokens,
            "cache_read_tokens": sub.cache_read_tokens,
            "permission_denials": len(sub.permission_denials),
        }
    if mode == "agent_longctx":
        snap = materialize_snapshot(bq, user_id, t)
        sub = run_subagent(prompt=prompt, snapshot_dir=snap, model=claude_model, allowed_tools=())
        return sub.text, 0, {
            "duration_ms": sub.duration_ms, "cost_usd": sub.cost_usd,
            "input_tokens": sub.input_tokens, "output_tokens": sub.output_tokens,
            "cache_read_tokens": sub.cache_read_tokens,
        }
    # llm_longctx
    return (llm_client.query_llm(prompt) or ""), 0, {}


# --- C1: repetition fatigue ------------------------------------------------

def run_task_c1(
    instances,
    user_id,
    bq: BackendQuery,
    llm_client,
    judge_client,
    mode: str,
    snapshot_cache: SnapshotCache,
    model_name: str | None,
    claude_model: str,
    context_budget: int | None,
    enable_llm_judge: bool,
    dry_run: bool,
    limit: int | None = None,
) -> list[dict]:
    if limit is not None:
        instances = instances[:limit]
    results: list[dict] = []
    for probe in instances:
        t_probe = probe["t_probe"]
        history_block = None
        history_tokens = 0
        if mode in ("agent_longctx", "llm_longctx"):
            history_block, stats = snapshot_cache.get_or_build(bq, user_id, t_probe, model_name, context_budget)
            history_tokens = stats["total_tokens"]

        prompt = prompts.repetition_fatigue_prompt(
            probe["app"], probe["saturated_hashtag"], probe["recent_titles"], history_block,
        )

        if dry_run:
            results.append({
                "task": "c1_repetition_fatigue",
                "user_id": user_id,
                "probe_id": probe["probe_id"],
                "mode": mode,
                "agent_response": None,
                "metrics": None,
            })
            continue

        raw_response, tool_call_count, subagent_stats = _dispatch_agent(
            mode, prompt, bq=bq, user_id=user_id, t=t_probe,
            claude_model=claude_model, llm_client=llm_client,
        )

        parsed = extract_json_from_response(raw_response) or {}
        new_hashtags = parsed.get("hashtags") or []
        div_rate = metrics.diversification_rate(probe["recent_hashtags_flat"], new_hashtags)

        results.append({
            "task": "c1_repetition_fatigue",
            "user_id": user_id,
            "probe_id": probe["probe_id"],
            "mode": mode,
            "agent_response": raw_response,
            "tool_calls": tool_call_count,
            "subagent_stats": subagent_stats,
            "history_tokens": history_tokens,
            "metrics": {"diversification_rate": div_rate, "num_new_hashtags": len(new_hashtags)},
        })
    return results


# --- C2: scenario library --------------------------------------------------

def run_task_c2(
    instances,
    user_id,
    bq: BackendQuery,
    llm_client,
    judge_client,
    mode: str,
    snapshot_cache: SnapshotCache,
    model_name: str | None,
    claude_model: str,
    context_budget: int | None,
    enable_llm_judge: bool,
    dry_run: bool,
    limit: int | None = None,
) -> list[dict]:
    if limit is not None:
        instances = instances[:limit]
    results: list[dict] = []
    for sc in instances:
        t_probe = sc["t_probe"]
        history_block = None
        history_tokens = 0
        if mode in ("agent_longctx", "llm_longctx"):
            history_block, stats = snapshot_cache.get_or_build(bq, user_id, t_probe, model_name, context_budget)
            history_tokens = stats["total_tokens"]

        prompt = prompts.scenario_prompt(sc["name"], sc["query"], sc["notes"], history_block)

        if dry_run:
            results.append({
                "task": "c2_scenario",
                "scenario_id": sc["scenario_id"],
                "user_id": user_id,
                "mode": mode,
                "agent_response": None,
                "metrics": None,
            })
            continue

        raw_response, tool_call_count, subagent_stats = _dispatch_agent(
            mode, prompt, bq=bq, user_id=user_id, t=t_probe,
            claude_model=claude_model, llm_client=llm_client,
        )

        parsed = extract_json_from_response(raw_response) or {}
        response_text = parsed.get("response") or raw_response

        leak = metrics.keyword_leak_rate(response_text, sc.get("forbidden_items") or [])
        carve = 1
        if sc.get("carve_out"):
            carve = metrics.carve_out_respect(
                response_text,
                sc["carve_out"].get("topic", ""),
                sc["carve_out"].get("hashtags", []),
            )

        judge_scores: dict = {}
        if enable_llm_judge and judge_client:
            anchor = TestItem(
                user_id=user_id,
                app="chatbot",
                source_object_id=sc["scenario_id"],
                source_timestamp=t_probe,
                formatted_timestamp="",
                source_interaction_type="implicit_negative" if sc["name"] in ("educated_rejection", "ask_to_forget") else "implicit_positive",
                source_hashtags=[],
                content={},
                interaction_format={},
                preference={},
            )
            evidence = build_judge_evidence(bq, anchor, response_text)
            judge_scores = judges.judge_restraint(judge_client, response_text, sc["name"], sc["notes"], evidence)

        results.append({
            "task": "c2_scenario",
            "scenario_id": sc["scenario_id"],
            "name": sc["name"],
            "user_id": user_id,
            "mode": mode,
            "agent_response": response_text,
            "tool_calls": tool_call_count,
            "subagent_stats": subagent_stats,
            "history_tokens": history_tokens,
            "metrics": {
                "keyword_leak_rate": leak,
                "carve_out_respect": carve,
                **judge_scores,
            },
        })
    return results


# --- C3: irrelevant-distractor restraint -----------------------------------

def run_task_c3(
    instances,
    user_id,
    bq: BackendQuery,
    llm_client,
    judge_client,
    mode: str,
    snapshot_cache: SnapshotCache,
    model_name: str | None,
    claude_model: str,
    context_budget: int | None,
    enable_llm_judge: bool,
    dry_run: bool,
    limit: int | None = None,
) -> list[dict]:
    if limit is not None:
        instances = instances[:limit]
    results: list[dict] = []
    for inst in instances:
        t = inst["source_timestamp"]
        history_block = None
        history_tokens = 0
        if mode in ("agent_longctx", "llm_longctx"):
            history_block, stats = snapshot_cache.get_or_build(bq, user_id, t, model_name, context_budget)
            history_tokens = stats["total_tokens"]

        prompt = prompts.restraint_prompt(inst["app"], inst["parent_event"], inst["candidates"], history_block)

        if dry_run:
            results.append({
                "task": "c3_restraint",
                "user_id": user_id,
                "test_id": inst["test_id"],
                "mode": mode,
                "agent_response": None,
                "metrics": None,
            })
            continue

        raw_response, tool_call_count, subagent_stats = _dispatch_agent(
            mode, prompt, bq=bq, user_id=user_id, t=t,
            claude_model=claude_model, llm_client=llm_client,
        )

        parsed = extract_json_from_response(raw_response) or {}
        reject_idxs = parsed.get("reject_indices") or []
        rejected_items = [
            inst["candidates"][i].get("persona_item")
            for i in reject_idxs
            if isinstance(i, int) and 0 <= i < len(inst["candidates"])
        ]

        rej_metrics = metrics.irrelevant_rejection_rate(
            agent_rejections=rejected_items,
            irrelevant_persona_items=inst["irrelevant_persona_items"],
            held_out_item=inst["held_out_persona_item"],
        )

        results.append({
            "task": "c3_restraint",
            "user_id": user_id,
            "test_id": inst["test_id"],
            "mode": mode,
            "app": inst["app"],
            "agent_response": raw_response,
            "tool_calls": tool_call_count,
            "subagent_stats": subagent_stats,
            "history_tokens": history_tokens,
            "reject_indices": reject_idxs,
            "metrics": rej_metrics,
        })
    return results


# --- Task D: aggregate negative avoidance ----------------------------------

def aggregate_task_d(task_a_results: list[dict]) -> dict:
    if not task_a_results:
        return {"task": "d_negative_avoidance", "n": 0}
    rows = [r for r in task_a_results if r.get("metrics")]
    n = len(rows)
    return {
        "task": "d_negative_avoidance",
        "n": n,
        "negative_in_top1_rate": metrics.mean(r["metrics"].get("negative_in_top1", 0) for r in rows),
        "negative_in_top3_rate": metrics.mean(r["metrics"].get("negative_in_top3", 0) for r in rows),
        "irrelevant_in_top1_rate": metrics.mean(r["metrics"].get("irrelevant_in_top1", 0) for r in rows),
    }
