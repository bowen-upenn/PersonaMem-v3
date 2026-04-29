"""Task B — Chatbot personalized response (runs against frozen benchmark instances).

User query, prior conversation, polarity, and same-day GT slice are frozen in
the benchmark. Distance-from-evidence is computed per-run because it depends
on the agent's effective context window, which varies with mode.
"""

from __future__ import annotations

from data_preparation.utils import extract_json_from_response
from evaluation import judges, metrics, prompts
from evaluation import personalization_rubric as pr
from evaluation.backend_query import BackendQuery, materialize_snapshot
from evaluation.claude_subagent import run_subagent
from evaluation.inference_utils import (
    SnapshotCache,
    TestItem,
    build_judge_evidence,
    dispatch_agent_run,
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
        arm = inst.get("arm", "proactive")  # v2 arm tag; v1 instances default to proactive

        history_block = None
        history_tokens = 0
        if mode in ("agent_longctx", "llm_longctx"):
            history_block, stats = snapshot_cache.get_or_build(bq, user_id, t, model_name, context_budget)
            history_tokens = stats["total_tokens"]

        # Arm-aware prompt: control queries get the "don't personalize" framing.
        if arm == "control":
            prompt = prompts.chatbot_control_prompt(user_query, prior, history_block)
        else:
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

        raw_response, tool_call_count, subagent_stats = dispatch_agent_run(
            mode, prompt, bq=bq, user_id=user_id, t=t,
            claude_model=claude_model, llm_client=llm_client,
        )

        parsed = extract_json_from_response(raw_response) or {}
        response_text = parsed.get("response") or raw_response

        # Task B v2 metric bundle depends on arm.
        gt = inst.get("gt_slice") or {"target": [], "avoid": []}
        slice_metrics = metrics.score_response_against_slice(response_text, gt["target"], gt["avoid"])

        top_k_prefs = inst.get("top_k_relevant_prefs") or []
        privacy_flagged = inst.get("privacy_flagged_prefs") or []
        post_test = inst.get("post_test_window") or {}

        # Arm scoring (Phase I + J extensions):
        #   - control / adversarial / distractor_reject / stale: restraint —
        #     measure personalization_leak_rate against a do-not-surface pool.
        #   - proactive / contradiction: surface preferences — measure
        #     top_k alignment.
        # `stale` (J.5) puts the now-expired pref into a synthesized pool of one
        # ([_stale_pref]) so the leak rate fires only when the agent surfaces
        # that specific stale item.
        if arm in ("control", "adversarial", "distractor_reject", "stale"):
            if arm == "distractor_reject":
                leak_pool = privacy_flagged
            elif arm == "stale":
                stale_pi = inst.get("_stale_pref") or ""
                leak_pool = [{"persona_item": stale_pi}] if stale_pi else top_k_prefs
            else:
                leak_pool = top_k_prefs
            ctrl_leak = metrics.personalization_leak_rate(response_text, leak_pool)
            arm_metrics = {
                "personalization_leak_rate": ctrl_leak["leak_rate"],
                "personalization_leaks": ctrl_leak["leaked"],
            }
        else:
            # Proactive / contradiction: reward surfacing top-K relevant prefs.
            align = metrics.top_k_preference_alignment(response_text, top_k_prefs)
            arm_metrics = {
                "top_k_alignment_rate": align["alignment_rate"],
                "top_k_matched": align["matched"],
            }

        # Privacy hard constraint applies on both arms.
        priv = metrics.privacy_leak_rate(response_text, privacy_flagged)
        priv_metrics = {
            "privacy_leak_rate": priv["leak_rate"],
            "privacy_leak_hard_fail": priv["hard_fail"],
        }

        # Source B — behavioral hit/miss, only for proactive arm instances
        # that have a populated post-test window.
        behavioral: dict = {}
        if arm == "proactive" and post_test.get("post_test_positives"):
            bh = metrics.behavioral_hit_miss(
                response_text,
                post_test["post_test_positives"],
                post_test.get("post_test_negatives") or [],
            )
            behavioral = {
                "behavioral_hit_rate": bh["hit_rate"],
                "behavioral_miss_rate": bh["miss_rate"],
                "behavioral_false_hit_rate": bh["false_hit_rate"],
            }

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
            # All restraint-style arms get negative polarity for the judge
            # (agent under test should NOT personalize). `stale` (J.5) is also
            # restraint — agent must NOT use the expired pref.
            if action == "asked_not_to_personalize" or arm in (
                "control", "adversarial", "distractor_reject", "stale"
            ):
                polarity = "negative"
            judge_scores = judges.judge_chatbot_rubric(judge_client, response_text, evidence, polarity)

        # Universal personalization rubric (hard dims always; judge dims gated on enable_llm_judge).
        task_id = f"chatbot_response_{arm}"
        ground_truth = pr.build_source_a(
            bq, user_id, t,
            query_text=user_query,
            query_hashtags=inst.get("source_hashtags", []),
        )
        pers_rubric = pr.score(
            task_id=task_id,
            agent_output=response_text,
            ground_truth=ground_truth,
            source_b=inst.get("post_test_window"),
            judge_client=(judge_client if enable_llm_judge else None),
        )

        from evaluation.inference_utils import merge_token_metrics
        result_metrics = {
            **slice_metrics,
            **arm_metrics,
            **priv_metrics,
            **behavioral,
            "carve_out_respect": carve_out_respect,
            **judge_scores,
            **{f"pr_{k}": v for k, v in pers_rubric.items() if isinstance(v, (int, float, str))},
        }
        merge_token_metrics(result_metrics, prompt=prompt, response=raw_response or "",
                            stats=subagent_stats, model=model_name)
        results.append({
            "task": "chatbot_response",
            "arm": arm,
            "user_id": user_id,
            "test_id": inst["test_id"],
            "mode": mode,
            "app": "chatbot",
            "source_timestamp": t,
            "polarity": inst.get("polarity"),
            "action": action,
            "blind_check_score": inst.get("blind_check_score"),
            "history_tokens": history_tokens,
            "tool_calls": tool_call_count,
            "subagent_stats": subagent_stats,
            "distance_tokens": dist_tokens,
            "distance_bin": dist_bin,
            "agent_response": response_text,
            "metrics": result_metrics,
        })
    return results
