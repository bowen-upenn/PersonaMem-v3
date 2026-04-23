"""Task A — Cross-app slate ranking (runs against frozen benchmark instances).

Slate composition, shuffle order, and held-out position are fixed in the
benchmark JSON; this driver is purely deterministic given the model output.
"""

from __future__ import annotations

from data_preparation.utils import extract_json_from_response
from evaluation import judges, metrics, prompts
from evaluation.backend_query import BackendQuery, materialize_snapshot
from evaluation.claude_subagent import run_subagent
from evaluation.inference_utils import SnapshotCache, dispatch_agent_run


def compute_ranking_metrics(ranked: list[int], instance: dict) -> dict:
    held_out_idx = instance["held_out_idx"]
    origins = instance["origin_by_idx"]
    slate = instance["slate"]
    target = {held_out_idx}
    k = len(slate)
    return {
        "recall@1": metrics.recall_at_k(ranked, target, 1),
        "recall@3": metrics.recall_at_k(ranked, target, 3),
        "recall@5": metrics.recall_at_k(ranked, target, 5),
        "hit@1": metrics.hit_at_k(ranked, target, 1),
        "hit@3": metrics.hit_at_k(ranked, target, 3),
        "mrr": metrics.mrr(ranked, target),
        "ndcg@k": metrics.ndcg_at_k([1.0 if i == held_out_idx else 0.0 for i in ranked], k),
        "ild_topk": metrics.intra_list_diversity([slate[i].get("hashtags", []) for i in ranked[:k]]),
        "negative_in_top1": int(origins[ranked[0]] == "negative") if ranked else 0,
        "negative_in_top3": int(any(origins[i] == "negative" for i in ranked[:3])),
        "irrelevant_in_top1": int(origins[ranked[0]] == "irrelevant") if ranked else 0,
    }


def run_task_a(
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
        app = inst["app"]

        # Strip origin labels before presenting the slate to the agent.
        slate_for_agent = [
            {k: v for k, v in c.items() if k != "_origin"}
            for c in inst["slate"]
        ]

        history_block = None
        history_tokens = 0
        if mode in ("agent_longctx", "llm_longctx"):
            history_block, stats = snapshot_cache.get_or_build(bq, user_id, t, model_name, context_budget)
            history_tokens = stats["total_tokens"]

        query_item = {"hashtags": inst["query_hashtags"]}
        prompt = prompts.slate_ranking_prompt(app, query_item, slate_for_agent, history_block)

        if dry_run:
            results.append({
                "task": "slate_ranking",
                "user_id": user_id,
                "test_id": inst["test_id"],
                "mode": mode,
                "slate_size": len(slate_for_agent),
                "held_out_idx": inst["held_out_idx"],
                "history_tokens": history_tokens,
                "agent_response": None,
                "metrics": None,
            })
            continue

        raw_response, tool_call_count, subagent_stats = dispatch_agent_run(
            mode, prompt, bq=bq, user_id=user_id, t=t,
            claude_model=claude_model, llm_client=llm_client,
        )

        parsed = extract_json_from_response(raw_response) or {}
        ranked = parsed.get("ranked_indices") or []
        if not isinstance(ranked, list) or sorted(set(ranked)) != list(range(len(slate_for_agent))):
            ranked = list(range(len(slate_for_agent)))
        ranking_metrics = compute_ranking_metrics(ranked, inst)

        judge_scores: dict = {}
        if enable_llm_judge and judge_client and ranked and inst["origin_by_idx"][ranked[0]] != "held_out":
            # Build judge evidence from a lightweight TestItem-like synthetic anchor.
            from evaluation.inference_utils import TestItem, build_judge_evidence
            anchor = TestItem(
                user_id=user_id,
                app=app,
                source_object_id=inst["test_id"],
                source_timestamp=t,
                formatted_timestamp=inst.get("formatted_timestamp", ""),
                source_interaction_type="implicit_positive",
                source_hashtags=inst["query_hashtags"],
                content={},
                interaction_format={},
                preference={},
            )
            top_pick = {k: slate_for_agent[ranked[0]].get(k) for k in ("title", "caption", "hashtags", "content_type")}
            evidence = build_judge_evidence(bq, anchor, raw_response)
            judge_scores = judges.judge_slate_soft_correctness(
                judge_client,
                agent_top_pick=top_pick,
                evidence=evidence,
                query_context=f"Ranking slate on {app} at {inst.get('formatted_timestamp', '')}.",
            )

        results.append({
            "task": "slate_ranking",
            "user_id": user_id,
            "test_id": inst["test_id"],
            "mode": mode,
            "app": app,
            "source_timestamp": t,
            "history_tokens": history_tokens,
            "tool_calls": tool_call_count,
            "subagent_stats": subagent_stats,
            "slate_origins": inst["origin_by_idx"],
            "held_out_idx": inst["held_out_idx"],
            "ranked_indices": ranked,
            "agent_response": raw_response,
            "metrics": {**ranking_metrics, **judge_scores},
        })
    return results
