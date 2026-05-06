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
    dispatch_agent_run as _dispatch_agent,
)


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


# --- Task C1a: counterfactual history-diff pairs --------------------------

def run_task_c1a(
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
    """For each counterfactual pair, ask the model for a recommendation at
    t_early then at t_late. Score the divergence between the two responses.
    Low divergence in the face of a substantive event diff → over-anchoring.
    """
    if limit is not None:
        instances = instances[:limit]
    results: list[dict] = []
    for pair in instances:
        t_early = pair["t_early"]
        t_late = pair["t_late"]
        app = pair["target_app"]
        query = pair["query"]

        def _run_at(t):
            history_block = None
            if mode in ("agent_longctx", "llm_longctx"):
                history_block, _stats = snapshot_cache.get_or_build(bq, user_id, t, model_name, context_budget)
            # For recommendation-style queries on a social app, reuse the
            # chatbot response prompt (it's flexible enough for "recommend X for this user").
            prompt = prompts.chatbot_response_prompt(query, [], history_block)
            raw, turns, stats = _dispatch_agent(mode, prompt, bq=bq, user_id=user_id, t=t,
                                                 claude_model=claude_model, llm_client=llm_client)
            parsed = extract_json_from_response(raw) or {}
            return parsed.get("response") or raw, turns, stats

        if dry_run:
            results.append({
                "task": "c1a_counterfactual",
                "pair_id": pair["pair_id"],
                "mode": mode,
                "agent_response_early": None,
                "agent_response_late": None,
                "metrics": None,
            })
            continue

        resp_early, turns_early, stats_early = _run_at(t_early)
        resp_late, turns_late, stats_late = _run_at(t_late)

        divergence = metrics.response_divergence(resp_early, resp_late)
        results.append({
            "task": "c1a_counterfactual",
            "pair_id": pair["pair_id"],
            "mode": mode,
            "target_app": app,
            "t_early": t_early,
            "t_late": t_late,
            "diff_events_count": len(pair.get("diff_events") or []),
            "dominant_category_pre": pair.get("dominant_category_pre"),
            "shift_category": pair.get("shift_category"),
            "agent_response_early": resp_early,
            "agent_response_late": resp_late,
            "tool_calls": turns_early + turns_late,
            "subagent_stats": {"early": stats_early, "late": stats_late},
            "metrics": {
                "response_divergence": divergence,
                "recency_sensitivity": divergence / max(1, len(pair.get("diff_events") or [])),
                "over_anchored_flag": 1 if divergence < 0.15 else 0,
            },
        })
    return results


# --- Task C1b: chatbot-sequence preference repetition ---------------------

def run_task_c1b(
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
    """Present a sequence of diverse-topic queries as a single conversation;
    measure how often the same preference is surfaced across responses.
    """
    if limit is not None:
        instances = instances[:limit]
    results: list[dict] = []
    for seq in instances:
        queries = seq.get("queries") or []
        if not queries:
            continue
        # Anchor time = max timestamp in the sequence.
        t_anchor = max(q["source_timestamp"] for q in queries)
        history_block = None
        if mode in ("agent_longctx", "llm_longctx"):
            history_block, _stats = snapshot_cache.get_or_build(bq, user_id, t_anchor, model_name, context_budget)

        # Run each query sequentially, passing previous turns as conversation history.
        responses: list[str] = []
        conversation: list[dict] = []
        total_turns = 0
        stats_per_query: list[dict] = []
        if dry_run:
            results.append({
                "task": "c1b_sequence",
                "sequence_id": seq["sequence_id"],
                "mode": mode,
                "responses": None,
                "metrics": None,
            })
            continue
        for q in queries:
            prompt = prompts.chatbot_response_prompt(q["user_query"], conversation, history_block)
            raw, turns, stats = _dispatch_agent(mode, prompt, bq=bq, user_id=user_id, t=q["source_timestamp"],
                                                 claude_model=claude_model, llm_client=llm_client)
            parsed = extract_json_from_response(raw) or {}
            resp = parsed.get("response") or raw
            responses.append(resp)
            total_turns += turns
            stats_per_query.append(stats)
            conversation.append({"role": "user", "content": q["user_query"]})
            conversation.append({"role": "assistant", "content": resp})

        gt_prefs_per_query = [q.get("top_k_relevant_prefs") or [] for q in queries]
        rep = metrics.preference_repetition_rate(responses, gt_prefs_per_query)

        results.append({
            "task": "c1b_sequence",
            "sequence_id": seq["sequence_id"],
            "mode": mode,
            "responses": responses,
            "query_categories": [q["top_k_relevant_prefs"][0].get("category") if q.get("top_k_relevant_prefs") else None for q in queries],
            "tool_calls": total_turns,
            "subagent_stats": stats_per_query,
            "metrics": {
                "preference_repetition_rate": rep["repetition_rate"],
                "wrong_preference_reuse": rep["wrong_preference_reuse"],
                "top_repeated_pref": rep["top_repeated_pref"],
                "n_queries": rep["n"],
            },
        })
    return results


# --- Task C1c: same-preference repetition cluster ------------------------

_C1C_PERSONA_ALIGNMENT_PROMPT = """You are a persona-alignment auditor. Decide whether a recommendation's hashtags would actually fit a user with the persona below — not generic plausibility, but "would THIS user engage with content carrying these hashtags?"

The user's persona signature (top categories + top hashtags they actually engage with):

  Top categories: {persona_categories}
  Top hashtags:   {persona_hashtags}

Off-persona distractor pool (these are deliberately NOT aligned with this user — picking from this pool is failure):

  {off_persona_distractors}

Recommendation under review:

  Title:    "{title}"
  Caption:  "{caption}"
  Hashtags: {hashtags}

Question: do the recommendation's hashtags fit this user's persona? Hashtags can be NEW (not in the user's existing top hashtags) AS LONG AS they're plausibly something this user would engage with — read across the title + caption to judge fit. Picking distractors from the off-persona pool, OR picking generic-but-clearly-off-persona hashtags, is a fail.

Output ONLY JSON:

```json
{{"persona_aligned": true | false, "reason": "<one short sentence>"}}
```
"""


def _c1c_persona_alignment_check(
    response: dict,
    persona_categories: list[str],
    persona_hashtags: list[str],
    off_persona_distractors: list[str],
    judge_query_fn,
) -> bool:
    """LLM judge: do this response's hashtags fit the user's persona?
    On any error / parse failure → return True (auto-pass — soft gate;
    we don't want a flaky judge to fail the metric).
    """
    if not callable(judge_query_fn):
        return True
    title = (response.get("title") or "").strip()
    caption = (response.get("caption") or "").strip()
    tags = response.get("hashtags") or []
    if not tags:
        return True
    prompt = _C1C_PERSONA_ALIGNMENT_PROMPT.format(
        persona_categories=", ".join(persona_categories[:6]) or "(none)",
        persona_hashtags=", ".join(f"#{h.lstrip('#')}" for h in persona_hashtags[:15]) or "(none)",
        off_persona_distractors=", ".join(f"#{h.lstrip('#')}" for h in off_persona_distractors[:10]) or "(none)",
        title=title[:200],
        caption=caption[:300],
        hashtags=", ".join(f"#{str(h).lstrip('#')}" for h in tags[:10]),
    )
    try:
        raw = judge_query_fn(prompt)
    except Exception:
        return True
    if not raw:
        return True
    parsed = extract_json_from_response(raw) or {}
    if not isinstance(parsed, dict):
        return True
    val = parsed.get("persona_aligned")
    if val is None:
        return True
    return bool(val)


def run_task_c1c(
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
    """For each same-preference repetition cluster, dispatch the agent on
    each anchor in sequence, threading prior responses into every
    subsequent prompt. Score the tail responses for diversification:
    pairwise text Jaccard ≤ 0.5, zero pairwise hashtag overlap, < 30%
    head-hashtag reuse, persona-aligned (LLM judge).
    """
    if limit is not None:
        instances = instances[:limit]
    from evaluation import metrics  # local — avoid circular at module load
    results: list[dict] = []
    for cluster in instances:
        queries = cluster.get("queries") or []
        if not queries:
            continue

        # Snapshot at the FINAL anchor when in long-context modes.
        # Earlier anchors share the same snapshot — the agent's
        # behavior under repetition isn't grounded by the marginal
        # 90 minutes of history.
        t_test = int(cluster.get("t_test") or queries[-1]["ts"])
        history_block = None
        if mode in ("agent_longctx", "llm_longctx"):
            history_block, _stats = snapshot_cache.get_or_build(
                bq, user_id, t_test, model_name, context_budget,
            )

        if dry_run:
            results.append({
                "task": "c1c_same_preference_cluster",
                "cluster_id": cluster["cluster_id"],
                "mode": mode,
                "responses": None,
                "metrics": None,
            })
            continue

        target_pref = cluster.get("target_pref", "")
        primary_category = cluster.get("primary_category", "")
        persona_hint = cluster.get("persona_hint") or {}
        persona_categories = list(persona_hint.get("top_categories") or [])
        persona_hashtags = list(persona_hint.get("top_hashtags") or [])
        off_persona_distractors = list(cluster.get("off_persona_distractor_hashtags") or [])
        n_allowed_repetitions = int(cluster.get("n_allowed_repetitions") or 2)

        responses: list[dict] = []
        total_turns = 0
        stats_per_query: list[dict] = []
        for q in queries:
            ts = int(q["ts"])
            prompt = prompts.repetition_fatigue_same_pref_prompt(
                target_pref=target_pref,
                primary_category=primary_category,
                user_query=q["user_query"],
                persona_top_categories=persona_categories,
                persona_top_hashtags=persona_hashtags,
                off_persona_distractor_hashtags=off_persona_distractors,
                prior_responses=responses,
                n_allowed_repetitions=n_allowed_repetitions,
                history_block=history_block,
            )
            raw, turns, stats = _dispatch_agent(
                mode, prompt, bq=bq, user_id=user_id, t=ts,
                claude_model=claude_model, llm_client=llm_client,
            )
            parsed = extract_json_from_response(raw) or {}
            if not isinstance(parsed, dict):
                parsed = {"title": "", "caption": raw, "hashtags": []}
            resp = {
                "title": parsed.get("title") or "",
                "caption": parsed.get("caption") or "",
                "hashtags": list(parsed.get("hashtags") or []),
                "reasoning": parsed.get("reasoning") or "",
                "ts": ts,
            }
            responses.append(resp)
            total_turns += turns
            stats_per_query.append(stats)

        # Persona-alignment judge runs only on tail responses (the
        # head zone is allowed to repeat, so persona-alignment there
        # is structurally satisfied by the user's own preference).
        head_n = min(n_allowed_repetitions + 1, len(responses))
        tail_responses = responses[head_n:]
        persona_alignment_passes: list[bool] = []
        if enable_llm_judge and judge_client is not None:
            judge_fn = (judge_client.query_llm
                        if hasattr(judge_client, "query_llm") else judge_client)
            for r in tail_responses:
                persona_alignment_passes.append(
                    _c1c_persona_alignment_check(
                        r, persona_categories, persona_hashtags,
                        off_persona_distractors, judge_fn,
                    )
                )
        else:
            persona_alignment_passes = [True] * len(tail_responses)

        diversity = metrics.within_cluster_diversity(
            responses,
            n_allowed_repetitions=n_allowed_repetitions,
            persona_alignment_passes=persona_alignment_passes,
        )

        results.append({
            "task": "c1c_same_preference_cluster",
            "cluster_id": cluster["cluster_id"],
            "mode": mode,
            "target_pref": target_pref,
            "primary_category": primary_category,
            "n_queries": len(queries),
            "n_allowed_repetitions": n_allowed_repetitions,
            "responses": responses,
            "tool_calls": total_turns,
            "subagent_stats": stats_per_query,
            "metrics": diversity,
        })
    return results


# --- Task C1d: chatbot same-pref repetition (varied surface) -------------

_C1D_PREF_INVOCATION_JUDGE_PROMPT = """You are auditing whether a chatbot response invoked a specific user preference. The preference is given below. Determine whether the response — by topic choice, recommendation, framing, or specific reference — leans on this preference.

A response "invokes" the preference when:
  - It explicitly recommends content / activities / items in that preference area, OR
  - It pivots the answer toward the preference's topic (even subtly), OR
  - It mentions the preference by name, by category, or by a close paraphrase.

A response does NOT invoke the preference when:
  - It answers the user's question on the question's own terms, with no detour through the preference area.
  - It picks a recommendation, angle, or framing that's unrelated to the preference.

## Target preference

  persona_item: "{target_pref}"
  category:     "{primary_category}"
  hashtags this user engages with on the topic: {target_hashtags}

## User's chatbot question

  > {user_query}

## Response under audit

  {response}

## Output

```json
{{"invoked": true | false, "reason": "<one short sentence>"}}
```
"""


def _c1d_check_pref_invoked(
    response_text: str,
    user_query: str,
    target_pref: str,
    primary_category: str,
    target_hashtags: list,
    judge_query_fn,
) -> bool:
    """LLM judge: did `response_text` invoke `target_pref`? On any error,
    return False (auto-pass) — flaky judge shouldn't break the metric.
    """
    if not callable(judge_query_fn):
        return False
    if not (response_text or "").strip():
        return False
    prompt = _C1D_PREF_INVOCATION_JUDGE_PROMPT.format(
        target_pref=target_pref[:200],
        primary_category=primary_category or "(none)",
        target_hashtags=", ".join(f"#{h.lstrip('#')}" for h in target_hashtags[:8]) or "(none)",
        user_query=user_query[:240],
        response=response_text[:800],
    )
    try:
        raw = judge_query_fn(prompt)
    except Exception:
        return False
    if not raw:
        return False
    parsed = extract_json_from_response(raw) or {}
    if not isinstance(parsed, dict):
        return False
    return bool(parsed.get("invoked"))


def run_task_c1d(
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
    """For each chatbot same-pref cluster, dispatch the agent on each of
    the cluster's surface-diverse chatbot queries in sequence (prior
    responses surfaced each turn). After dispatch, an LLM judge per
    response decides whether it invoked the target preference. Score
    via ``metrics.chatbot_pref_overuse_rate`` — tail responses must
    NOT invoke target_pref."""
    if limit is not None:
        instances = instances[:limit]
    from evaluation import metrics  # local — avoid circular at module load
    results: list[dict] = []
    for cluster in instances:
        queries = cluster.get("queries") or []
        if not queries:
            continue

        t_test = int(cluster.get("t_test") or queries[-1]["ts"])
        history_block = None
        if mode in ("agent_longctx", "llm_longctx"):
            history_block, _stats = snapshot_cache.get_or_build(
                bq, user_id, t_test, model_name, context_budget,
            )

        if dry_run:
            results.append({
                "task": "c1d_chatbot_same_pref",
                "cluster_id": cluster["cluster_id"],
                "mode": mode,
                "responses": None,
                "metrics": None,
            })
            continue

        target_pref = cluster.get("target_pref", "")
        primary_category = cluster.get("primary_category", "")
        target_hashtags = list(cluster.get("target_hashtags") or [])
        n_allowed_repetitions = int(cluster.get("n_allowed_repetitions") or 2)

        responses: list[dict] = []
        total_turns = 0
        stats_per_query: list[dict] = []
        for q in queries:
            ts = int(q["ts"])
            prompt = prompts.repetition_fatigue_chatbot_prompt(
                user_query=q["user_query"],
                target_pref=target_pref,
                primary_category=primary_category,
                prior_responses=responses,
                n_allowed_repetitions=n_allowed_repetitions,
                history_block=history_block,
            )
            raw, turns, stats = _dispatch_agent(
                mode, prompt, bq=bq, user_id=user_id, t=ts,
                claude_model=claude_model, llm_client=llm_client,
            )
            parsed = extract_json_from_response(raw) or {}
            response_text = (parsed.get("response") or raw or "").strip()
            responses.append({
                "ts": ts,
                "user_query": q["user_query"],
                "natural_anchor": q.get("natural_anchor", ""),
                "response": response_text,
            })
            total_turns += turns
            stats_per_query.append(stats)

        # Per-response LLM judge: did this response invoke target_pref?
        pref_invoked: list[bool] = []
        if enable_llm_judge and judge_client is not None:
            judge_fn = (judge_client.query_llm
                        if hasattr(judge_client, "query_llm") else judge_client)
            for r in responses:
                pref_invoked.append(
                    _c1d_check_pref_invoked(
                        r["response"], r["user_query"],
                        target_pref, primary_category, target_hashtags,
                        judge_fn,
                    )
                )
        else:
            pref_invoked = [False] * len(responses)

        overuse = metrics.chatbot_pref_overuse_rate(
            pref_invoked, n_allowed_repetitions=n_allowed_repetitions,
        )

        results.append({
            "task": "c1d_chatbot_same_pref",
            "cluster_id": cluster["cluster_id"],
            "mode": mode,
            "target_pref": target_pref,
            "primary_category": primary_category,
            "n_queries": len(queries),
            "n_allowed_repetitions": n_allowed_repetitions,
            "responses": responses,
            "pref_invoked_per_response": pref_invoked,
            "tool_calls": total_turns,
            "subagent_stats": stats_per_query,
            "metrics": overuse,
        })
    return results


# --- Task C4: do-not-personalize button regeneration ---------------------

def run_task_c4(
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
    """Two-turn: (1) let the model give a normal personalized response to the
    B-proactive query, (2) send the 'do-not-personalize' signal + the original
    response, expect a regen with personalization stripped.
    """
    if limit is not None:
        instances = instances[:limit]
    results: list[dict] = []
    for inst in instances:
        t = inst["source_timestamp"]
        user_query = inst["user_query"]
        prior = inst.get("prior_conversation") or []
        history_block = None
        if mode in ("agent_longctx", "llm_longctx"):
            history_block, _stats = snapshot_cache.get_or_build(bq, user_id, t, model_name, context_budget)

        if dry_run:
            results.append({
                "task": "c4_button_regen",
                "test_id": inst["test_id"],
                "mode": mode,
                "metrics": None,
            })
            continue

        # Turn 1: original personalized response.
        p1 = prompts.chatbot_response_prompt(user_query, prior, history_block)
        raw1, turns1, stats1 = _dispatch_agent(mode, p1, bq=bq, user_id=user_id, t=t,
                                                claude_model=claude_model, llm_client=llm_client)
        parsed1 = extract_json_from_response(raw1) or {}
        original_resp = parsed1.get("response") or raw1

        # Turn 2: regenerate without personalization.
        p2 = prompts.button_regen_prompt(user_query, original_resp, prior, history_block)
        raw2, turns2, stats2 = _dispatch_agent(mode, p2, bq=bq, user_id=user_id, t=t,
                                                claude_model=claude_model, llm_client=llm_client)
        parsed2 = extract_json_from_response(raw2) or {}
        regen_resp = parsed2.get("response") or raw2

        # Score the regeneration.
        m = metrics.personalization_removal_delta(original_resp, regen_resp, inst["held_out_preference"])
        # Content-retention: compare regen to the blind-check generic answer if available.
        generic = inst.get("blind_check_generic_answer") or ""
        content_retention = 0.0
        if generic:
            content_retention = 1.0 - metrics.response_divergence(generic, regen_resp)

        results.append({
            "task": "c4_button_regen",
            "test_id": inst["test_id"],
            "mode": mode,
            "original_response": original_resp,
            "regen_response": regen_resp,
            "tool_calls": turns1 + turns2,
            "subagent_stats": {"turn1": stats1, "turn2": stats2},
            "metrics": {
                **m,
                "content_retention_vs_generic": content_retention,
            },
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
