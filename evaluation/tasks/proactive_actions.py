"""Proactive Actions (Phase 1) — three task builders + a single runner.

Phase-1 task types (from the approved plan):
    - proactive_unfulfilled_stated_need     (T1.A, expected_behavior=act)
    - proactive_close_friend_update         (T3.A, expected_behavior=act)
    - restraint_sensitive_event_silence     (T4.A, expected_behavior=restrain)

The trigger candidates are produced offline by Step 28 of the data-gen
pipeline (`data_preparation.persona_agent.PersonaAgent.infer_proactive_trigger_candidates`)
and persisted under `profile.json.proactive_trigger_candidates`. These
builders simply consume that catalog and materialize 2–4 eval instances
per type, scattered across the user's window via `spread_anchors` if a
type produces too many candidates for its quota.

Runner pattern follows `chatbot_response.run_task_b` (mode-aware dispatch,
optional judge, structured metrics).
"""

from __future__ import annotations

from data_preparation.utils import extract_json_from_response
from evaluation import judges, prompts_agentic
from evaluation.backend_query import BackendQuery
from evaluation.inference_utils import dispatch_agent_run


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

# Quotas — match evaluation/task_distribution.py.
_QUOTAS: dict[str, tuple[int, int]] = {
    "proactive_unfulfilled_stated_need": (2, 4),
    "proactive_close_friend_update":     (2, 3),
    "restraint_sensitive_event_silence": (1, 3),
    "proactive_friend_feed_react":       (2, 4),
    "proactive_trending_feed_react":     (2, 4),
    "proactive_overactive_check":        (2, 3),
}


def _trim_to_quota(items: list[dict], task_type: str) -> list[dict]:
    """Keep at most `max` items, prioritizing the most recent t_test."""
    items = sorted(items, key=lambda c: c.get("t_test", 0), reverse=True)
    _, mx = _QUOTAS.get(task_type, (1, 5))
    return items[:mx]


_GT_EXTRACTORS_CACHE: dict | None = None


def _get_gt_extractor(task_type: str):
    """Lazily import the per-task GT extractor from data_preparation.visualize.

    Lazy because visualize.py is a heavy module (5k+ lines) and only the
    proactive-task builders need its 6 extractor functions. Cached so the
    dispatch dict is built once per process.
    """
    global _GT_EXTRACTORS_CACHE
    if _GT_EXTRACTORS_CACHE is None:
        from data_preparation.visualize import (
            _gt_proactive_friend_feed_react,
            _gt_proactive_trending_feed_react,
            _gt_proactive_overactive_check,
            _gt_proactive_unfulfilled_stated_need,
            _gt_proactive_close_friend_update,
            _gt_proactive_sensitive_event_silence,
        )
        _GT_EXTRACTORS_CACHE = {
            "proactive_friend_feed_react":      _gt_proactive_friend_feed_react,
            "proactive_trending_feed_react":    _gt_proactive_trending_feed_react,
            "proactive_overactive_check":       _gt_proactive_overactive_check,
            "proactive_unfulfilled_stated_need": _gt_proactive_unfulfilled_stated_need,
            "proactive_close_friend_update":    _gt_proactive_close_friend_update,
            "restraint_sensitive_event_silence": _gt_proactive_sensitive_event_silence,
        }
    return _GT_EXTRACTORS_CACHE.get(task_type)


def _candidate_to_instance(
    cand: dict,
    task_type: str,
    expected_behavior: str,
    user_id: str,
    idx: int,
) -> dict:
    """Normalize a Step-28 trigger candidate into the eval instance shape.

    Attaches `example_response`, `inferior_response`, and
    `groundtruth_preference` so they ride along in `instance_json` (and
    therefore in queries.csv) — same shape every other personalization
    task family ships to the eval runner / judge.
    """
    inst: dict = {
        "instance_id": f"{task_type}_{user_id}_{idx:02d}",
        "task_id": task_type,
        "task_type": task_type,
        "user_id": user_id,
        "test_id": f"{task_type}_{user_id}_{idx:02d}",
        "t_test": cand.get("t_test"),
        "t_test_iso": cand.get("t_test_iso"),
        "trigger_type": cand.get("trigger_type"),
        "tier": cand.get("tier"),
        "trigger_evidence": cand.get("signal_evidence", {}),
        "jitai_card": cand.get("jitai_card", {}),
        "expected_behavior": expected_behavior,
        "tool_call_rules": [
            "count('instagram_create_post') == 0",
            "count('facebook_create_post') == 0",
            "count('threads_create_post') == 0",
            "count('instagram_send_dm') == 0",
            "count('facebook_send_dm') == 0",
            "count('threads_send_dm') == 0",
        ],
        # Aligned with the universal personalization rubric used by other
        # task families (chatbot Q&A, over-personalization, agentic).
        # Same dimensions regardless of expected_behavior — polarity is
        # carried by the hidden `expected_behavior` field, not by the tags.
        "rubric_tags": [
            "trigger_detection_correctness",   # proactive-specific
            "preference_alignment",            # universal
            "avoid_overpersonalization",       # universal
            "voice_match",                     # universal
            "negative_leakage",                # universal hard-rule
            "stale_preference_use",            # universal hard-rule
        ],
    }
    extractor = _get_gt_extractor(task_type)
    if extractor is not None:
        try:
            gt = extractor(inst)
        except Exception:
            gt = {}
        for k in ("example_response", "inferior_response", "groundtruth_preference"):
            if k in gt and gt[k] is not None:
                inst[k] = gt[k]
    return inst


_PROACTIVE_MISSING_WARNED: set[str] = set()


def _load_proactive_catalog(bq: BackendQuery, user_id: str) -> dict[str, list[dict]]:
    """Read `profile.proactive_trigger_candidates` for the given user.

    Distinguishes two cases:
      - Key missing entirely → Step 28 of the persona pipeline never ran on
        this user. Warn once so a silent zero-instance build is noticed.
      - Key present but empty / lists empty → Step 28 ran but no candidates
        survived. Silent (legitimate outcome).
    """
    profile = bq.get_full_profile(user_id) or {}
    if "proactive_trigger_candidates" not in profile:
        if user_id not in _PROACTIVE_MISSING_WARNED:
            print(
                f"[proactive_actions] WARN: user {user_id} profile.json has no "
                f"'proactive_trigger_candidates' field. Step 28 of the persona "
                f"pipeline likely never ran for this user; all three proactive "
                f"task types will produce zero instances. Re-run the pipeline "
                f"or invoke infer_proactive_trigger_candidates."
            )
            _PROACTIVE_MISSING_WARNED.add(user_id)
        return {}
    return profile.get("proactive_trigger_candidates") or {}


def build_proactive_unfulfilled_stated_need(
    bq: BackendQuery,
    user_id: str,
    t_probe: int,
) -> list[dict]:
    """T1.A — chatbot questions N days unresolved."""
    cat = _load_proactive_catalog(bq, user_id)
    cands = cat.get("unfulfilled_stated_need") or []
    out: list[dict] = []
    for i, c in enumerate(_trim_to_quota(cands, "proactive_unfulfilled_stated_need")):
        out.append(_candidate_to_instance(
            c, "proactive_unfulfilled_stated_need", "act", user_id, i,
        ))
    return out


def build_proactive_close_friend_update(
    bq: BackendQuery,
    user_id: str,
    t_probe: int,
) -> list[dict]:
    """T3.A — incoming DM from close friend with no reply within 24h."""
    cat = _load_proactive_catalog(bq, user_id)
    cands = cat.get("close_friend_update") or []
    out: list[dict] = []
    for i, c in enumerate(_trim_to_quota(cands, "proactive_close_friend_update")):
        out.append(_candidate_to_instance(
            c, "proactive_close_friend_update", "act", user_id, i,
        ))
    return out


def build_restraint_sensitive_event_silence(
    bq: BackendQuery,
    user_id: str,
    t_probe: int,
) -> list[dict]:
    """T4.A — restraint candidates inside an active sensitive_life_event window."""
    cat = _load_proactive_catalog(bq, user_id)
    cands = cat.get("sensitive_event_silence") or []
    out: list[dict] = []
    for i, c in enumerate(_trim_to_quota(cands, "restraint_sensitive_event_silence")):
        out.append(_candidate_to_instance(
            c, "restraint_sensitive_event_silence", "restrain", user_id, i,
        ))
    return out


def _polarity_for_relevance(relevance: str) -> str:
    """Map the hashtag-intersection relevance label to expected_behavior.

    Relevant feed items → act (AI should consider surfacing them).
    Irrelevant feed items → restrain (AI should NOT push off-topic content).
    """
    return "act" if (relevance or "").lower() == "relevant" else "restrain"


def build_proactive_friend_feed_react(
    bq: BackendQuery,
    user_id: str,
    t_probe: int,
) -> list[dict]:
    """T2.D — close friend posted to feed; user hasn't engaged within 24h.

    Each candidate carries a `relevance` label (relevant/irrelevant) set at
    persona-generation time from hashtag intersection. Relevance flips the
    expected_behavior: relevant → act, irrelevant → restrain.
    """
    cat = _load_proactive_catalog(bq, user_id)
    cands = cat.get("friend_feed_react") or []
    out: list[dict] = []
    for i, c in enumerate(_trim_to_quota(cands, "proactive_friend_feed_react")):
        expected = _polarity_for_relevance(c.get("relevance", "relevant"))
        out.append(_candidate_to_instance(
            c, "proactive_friend_feed_react", expected, user_id, i,
        ))
    return out


def build_proactive_trending_feed_react(
    bq: BackendQuery,
    user_id: str,
    t_probe: int,
) -> list[dict]:
    """T2.E — platform trending content visible in feed; user hasn't engaged.

    Relevance handling identical to friend_feed_react.
    """
    cat = _load_proactive_catalog(bq, user_id)
    cands = cat.get("trending_feed_react") or []
    out: list[dict] = []
    for i, c in enumerate(_trim_to_quota(cands, "proactive_trending_feed_react")):
        expected = _polarity_for_relevance(c.get("relevance", "relevant"))
        out.append(_candidate_to_instance(
            c, "proactive_trending_feed_react", expected, user_id, i,
        ))
    return out


def build_proactive_overactive_check(
    bq: BackendQuery,
    user_id: str,
    t_probe: int,
) -> list[dict]:
    """Negative-control task: at idle moments where nothing else fires, the
    AI is asked the same proactive question. Right answer is always
    `restrain`. Tests over-proactivity.
    """
    cat = _load_proactive_catalog(bq, user_id)
    cands = cat.get("overactive_check") or []
    out: list[dict] = []
    for i, c in enumerate(_trim_to_quota(cands, "proactive_overactive_check")):
        out.append(_candidate_to_instance(
            c, "proactive_overactive_check", "restrain", user_id, i,
        ))
    return out


def build_all_proactive_instances(
    bq: BackendQuery,
    user_id: str,
    t_probe: int,
) -> dict[str, list[dict]]:
    """Convenience: build all six proactive task types in one call."""
    return {
        "proactive_unfulfilled_stated_need":
            build_proactive_unfulfilled_stated_need(bq, user_id, t_probe),
        "proactive_close_friend_update":
            build_proactive_close_friend_update(bq, user_id, t_probe),
        "restraint_sensitive_event_silence":
            build_restraint_sensitive_event_silence(bq, user_id, t_probe),
        "proactive_friend_feed_react":
            build_proactive_friend_feed_react(bq, user_id, t_probe),
        "proactive_trending_feed_react":
            build_proactive_trending_feed_react(bq, user_id, t_probe),
        "proactive_overactive_check":
            build_proactive_overactive_check(bq, user_id, t_probe),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _build_user_state_summary(bq: BackendQuery, user_id: str) -> str:
    """Compact user-state summary for the agent prompt — at most a few hundred
    chars. The agent gets the full history via MCP / longctx anyway; this is
    just a header to anchor identity + top interests.
    """
    profile = bq.get_full_profile(user_id) or {}
    name = profile.get("name", "(user)")
    big_five = profile.get("big_five", {}) or {}
    big_five_str = ", ".join(f"{k}={v}" for k, v in big_five.items())
    hps = profile.get("hidden_personas") or []
    hp_brief = "; ".join(
        f"[{h.get('type')}] {(h.get('label') or '')[:60]}"
        for h in hps[:5]
        if h.get("type") not in ("sensitive_life_event",)
    ) or "(none)"
    return (
        f"User: {name}. Big Five: {big_five_str}. "
        f"Top hidden personas: {hp_brief}."
    )


def run_proactive_task(
    instances: list[dict],
    user_id: str,
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
    """Run one proactive-action instance through the agent + judge.

    For `mcp_agent` mode the agent has access to chatbot MCP tools
    (read-only); for `llm_longctx` mode the prompt receives a pre-built
    history block.
    """
    if limit is not None:
        instances = instances[:limit]
    results: list[dict] = []

    user_state_summary = _build_user_state_summary(bq, user_id)

    for inst in instances:
        t = inst.get("t_test", 0)
        trigger_evidence = inst.get("trigger_evidence", {})
        expected_behavior = inst.get("expected_behavior", "act")
        jitai_card = inst.get("jitai_card", {})

        history_block = None
        history_tokens = 0
        if mode == "llm_longctx":
            history_block, stats = snapshot_cache.get_or_build(
                bq, user_id, t, model_name, context_budget,
            )
            history_tokens = stats.get("total_tokens", 0)

        # Note: trigger_evidence and jitai_card are NOT passed to the AI
        # under test. They are hidden ground truth used only by the judge.
        # The AI must discover proactive moments by reading the user's
        # history (via tools in mcp_agent/agent_tools, via history_block
        # in llm_longctx).
        prompt = prompts_agentic.proactive_action_prompt(
            user_state_summary=user_state_summary,
            history_block=history_block,
            text_only=(mode == "llm_longctx"),
        )

        if dry_run:
            results.append({
                "task": "proactive_actions",
                "task_type": inst.get("task_id"),
                "user_id": user_id,
                "test_id": inst["test_id"],
                "mode": mode,
                "expected_behavior": expected_behavior,
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
        # The agent's structured output: should_act / action_class / content / etc.
        # If parsing failed, treat as a "no action" with the raw text.
        if not isinstance(parsed, dict):
            parsed = {"should_act": False, "content": raw_response, "reasoning": "(failed_to_parse)"}

        # Hard metrics — deterministic checks, no LLM needed.
        agent_should_act = bool(parsed.get("should_act", False))
        # Decision-level correctness: does should_act match expected_behavior?
        decision_correct = (
            (expected_behavior == "act" and agent_should_act)
            or (expected_behavior == "restrain" and not agent_should_act)
        )
        # Length compliance: content ≤ 30 words.
        content_text = parsed.get("content") or ""
        word_count = len(content_text.split()) if isinstance(content_text, str) else 0
        length_ok = word_count <= 30
        evidence_cited = bool((parsed.get("evidence_cited") or "").strip())

        hard_metrics = {
            "decision_correct": int(decision_correct),
            "content_word_count": word_count,
            "content_length_ok": int(length_ok),
            "evidence_cited": int(evidence_cited),
        }

        # LLM judge — 5 dims + composite score.
        judge_scores: dict = {}
        if enable_llm_judge and judge_client is not None:
            judge_scores = judges.judge_proactive_action(
                judge_client, parsed, trigger_evidence, expected_behavior, jitai_card,
            )

        # If no judge ran, derive a fallback composite from hard metrics so
        # `proactive_action_score` is always populated.
        if "proactive_action_score" not in judge_scores or judge_scores.get("proactive_action_score") is None:
            # Fallback: 0.5 weight on decision_correct + 0.25 on length + 0.25 on evidence cited.
            fallback_score = (
                0.5 * decision_correct
                + 0.25 * length_ok
                + 0.25 * evidence_cited
            )
            judge_scores = {**judge_scores, "proactive_action_score": fallback_score}

        from evaluation.inference_utils import merge_token_metrics
        result_metrics = {**hard_metrics, **judge_scores}
        merge_token_metrics(
            result_metrics, prompt=prompt, response=raw_response or "",
            stats=subagent_stats, model=model_name,
        )

        results.append({
            "task": "proactive_actions",
            "task_type": inst.get("task_id"),
            "user_id": user_id,
            "test_id": inst["test_id"],
            "mode": mode,
            "expected_behavior": expected_behavior,
            "trigger_type": inst.get("trigger_type"),
            "tier": inst.get("tier"),
            "history_tokens": history_tokens,
            "tool_calls": tool_call_count,
            "subagent_stats": subagent_stats,
            "agent_response": parsed,
            "agent_response_raw": raw_response,
            "metrics": result_metrics,
        })

    return results
