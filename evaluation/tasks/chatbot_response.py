"""Task B — Chatbot personalized response (runs against frozen benchmark instances).

User query, prior conversation, polarity, and same-day GT slice are frozen in
the benchmark. Distance-from-evidence is computed per-run because it depends
on the agent's effective context window, which varies with mode.
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


def _extract_query_and_prior(test: TestItem) -> tuple[str, list[dict]]:
    """Split the full conversation at the test turn. Used by `build_benchmark`."""
    convo = test.conversation or []
    user_msg = (test.interaction_format or {}).get("user_message") or ""
    if convo:
        for i, m in enumerate(convo):
            if m.get("role") == "user" and m.get("content", "").strip() == user_msg.strip():
                return user_msg, convo[:i]
        for i in range(len(convo) - 1, -1, -1):
            if convo[i].get("role") == "user":
                return convo[i].get("content", "") or user_msg, convo[:i]
    return user_msg, []


def _distance_from_evidence_tokens(bq: BackendQuery, user_id: str, t_test: int, source_hashtags: list[str]) -> int:
    support_ts = -1
    test_hashtags = {h.lower() for h in source_hashtags}
    for app in ("instagram", "facebook", "threads", "chatbot"):
        for e in bq.get_events(user_id=user_id, app=app, since_timestamp=t_test):
            if any(h.lower() in test_hashtags for h in e.get("source_hashtags", [])):
                if e.get("source_timestamp", 0) > support_ts:
                    support_ts = e["source_timestamp"]
    if support_ts < 0:
        return -1
    elapsed_events = 0
    for app in ("instagram", "facebook", "threads", "chatbot"):
        for e in bq.get_events(user_id=user_id, app=app, since_timestamp=t_test):
            if e.get("source_timestamp", 0) > support_ts:
                elapsed_events += 1
    return elapsed_events * 200


def run_task_b(
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
        user_query = inst["user_query"]
        prior = inst.get("prior_conversation") or []
        action = inst.get("action", "")

        history_block = None
        history_tokens = 0
        if mode in ("agent_longctx", "llm_longctx"):
            history_block, stats = snapshot_cache.get_or_build(bq, user_id, t, model_name, context_budget)
            history_tokens = stats["total_tokens"]

        prompt = prompts.chatbot_response_prompt(user_query, prior, history_block)

        if dry_run:
            results.append({
                "task": "chatbot_response",
                "user_id": user_id,
                "test_id": inst["test_id"],
                "mode": mode,
                "history_tokens": history_tokens,
                "prior_turns": len(prior),
                "user_query_len": len(user_query),
                "agent_response": None,
                "metrics": None,
            })
            continue

        tool_call_count = 0
        subagent_stats: dict = {}
        if mode == "agent_tools":
            snap = materialize_snapshot(bq, user_id, t)
            sub = run_subagent(prompt=prompt, snapshot_dir=snap, model=claude_model)
            raw_response = sub.text
            tool_call_count = sub.turns
            subagent_stats = {"duration_ms": sub.duration_ms, "cost_usd": sub.cost_usd,
                              "input_tokens": sub.input_tokens, "output_tokens": sub.output_tokens,
                              "cache_read_tokens": sub.cache_read_tokens,
                              "permission_denials": len(sub.permission_denials)}
        elif mode == "agent_longctx":
            snap = materialize_snapshot(bq, user_id, t)
            sub = run_subagent(prompt=prompt, snapshot_dir=snap, model=claude_model, allowed_tools=())
            raw_response = sub.text
            subagent_stats = {"duration_ms": sub.duration_ms, "cost_usd": sub.cost_usd,
                              "input_tokens": sub.input_tokens, "output_tokens": sub.output_tokens,
                              "cache_read_tokens": sub.cache_read_tokens}
        else:
            raw_response = llm_client.query_llm(prompt) or ""

        parsed = extract_json_from_response(raw_response) or {}
        response_text = parsed.get("response") or raw_response

        # Score against the FROZEN TARGET/AVOID slice from the benchmark file.
        gt = inst["gt_slice"]
        slice_metrics = metrics.score_response_against_slice(response_text, gt["target"], gt["avoid"])

        carve_out_respect = 1
        if action == "asked_not_to_personalize":
            msg = inst.get("user_query", "")
            carve_out_respect = metrics.carve_out_respect(response_text, msg, inst.get("source_hashtags", []))

        dist_tokens = _distance_from_evidence_tokens(bq, user_id, t, inst.get("source_hashtags", []))
        dist_bin = metrics.distance_bin(dist_tokens)

        judge_scores: dict = {}
        if enable_llm_judge and judge_client:
            anchor = TestItem(
                user_id=user_id,
                app="chatbot",
                source_object_id=inst["test_id"],
                source_timestamp=t,
                formatted_timestamp=inst.get("formatted_timestamp", ""),
                source_interaction_type="implicit_positive" if inst.get("polarity") == "positive" else "implicit_negative",
                source_hashtags=inst.get("source_hashtags", []),
                content={},
                interaction_format={"action": action},
                preference=inst.get("held_out_preference") or {},
            )
            evidence = build_judge_evidence(bq, anchor, response_text)
            polarity = inst.get("polarity", "positive")
            if action == "asked_not_to_personalize":
                polarity = "negative"
            judge_scores = judges.judge_chatbot_rubric(judge_client, response_text, evidence, polarity)

        results.append({
            "task": "chatbot_response",
            "user_id": user_id,
            "test_id": inst["test_id"],
            "mode": mode,
            "app": "chatbot",
            "source_timestamp": t,
            "polarity": inst.get("polarity"),
            "action": action,
            "history_tokens": history_tokens,
            "tool_calls": tool_call_count,
            "subagent_stats": subagent_stats,
            "distance_tokens": dist_tokens,
            "distance_bin": dist_bin,
            "agent_response": response_text,
            "metrics": {
                **slice_metrics,
                "carve_out_respect": carve_out_respect,
                **judge_scores,
            },
        })
    return results
