"""Drivers for agentic tasks T6-T19.

Each task has two symmetric pieces:
- `build_{task_id}(bq, user_id, ...)` → a list of frozen instances (called
  from build_benchmark.py).
- `run_{task_id}(instances, user_id, bq, ...)` → list of result rows.

Every task:
1. Iterates its frozen instances.
2. Builds a prompt via `prompts_agentic.{task_id}`.
3. Dispatches the agent via `inference_utils.dispatch_agent_run` (mode-agnostic).
4. Parses the response JSON.
5. Scores with:
   - Task-specific hard metrics (tool-call regex, final-state-diff, content rules).
   - Universal personalization rubric (`personalization_rubric.score`).
6. Returns a result row.

Tasks are kept short — most are ~50 LOC each — because the heavy lifting
(GT builders, rubric, dispatch) is all shared.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from data_preparation.utils import extract_json_from_response
from evaluation import metrics as metrics_mod
from evaluation import personalization_rubric as pr
from evaluation import prompts_agentic
from evaluation.backend_query import APPS, BackendQuery
from evaluation.inference_utils import SnapshotCache, dispatch_agent_run


SOCIAL_APPS = ("instagram", "facebook", "threads")


# ---------------------------------------------------------------------------
# Workstream G: per-task over-personalization triggers
#
# Mirror the chatbot proactive / overpersonalization split for every agentic
# task. The `proactive` arm uses the existing trigger (real DM, real
# life-update, real source post). The `overpersonalization` arm swaps in
# a generic trigger that does NOT call for personalization (out-of-office
# message, factual announcement, generic refind topic) — the agent must
# resist dragging in user preferences.
#
# Triggers are static text per task; no LLM synthesis. Each entry is a
# dict that overlays onto the proactive instance — only the listed keys
# are replaced; everything else (target_app, t_test, thread_id, …) is
# kept so the tool_call sequence remains valid.
# ---------------------------------------------------------------------------

_OVERPERSONALIZATION_OVERLAYS: dict[str, list[dict]] = {
    "agentic_user_voice_post": [
        # T6: generic announcement that doesn't call for community-flavored personalization.
        {"_trigger_kind": "out_of_office",
         "_trigger_note": "Generic out-of-office post — no personal interests should leak in."},
    ],
    "agentic_moment_recommendation": [
        # T7: a moment that doesn't naturally call for personalized content.
        {"moment": "morning standup", "_trigger_kind": "neutral_moment"},
        {"moment": "boarding a flight",  "_trigger_kind": "neutral_moment"},
    ],
    "agentic_dm_digest": [
        # T8: same trigger structure (digest), but the audit checks the agent
        # doesn't insert irrelevant personal commentary about senders.
        {"_trigger_kind": "neutral_digest",
         "_trigger_note": "Pure digest — no opinion on senders' inferred interests."},
    ],
    "agentic_cross_app_repost": [
        # T9: a non-personal source post (a generic news headline) — the
        # repost should not weave in the user's hashtags.
        {"source_post": {
            "caption": "Reminder: clocks change Sunday morning. Set your alarms accordingly.",
            "hashtags": [],
        }, "_trigger_kind": "neutral_repost"},
    ],
    "agentic_auto_reply": [
        # T10: a generic factual question — no preference signal, just a
        # short factual reply.
        {"inbound_message": "what time are we meeting?",
         "_trigger_kind": "factual_reply"},
        {"inbound_message": "did you get the email I sent?",
         "_trigger_kind": "factual_reply"},
    ],
    "agentic_vague_refind": [
        # T11: a topic the user has no engagement with — the agent should
        # say so honestly rather than fabricating.
        {"topic": "OPEC oil futures",      "_trigger_kind": "no_history_topic"},
        {"topic": "obscure UK panel show", "_trigger_kind": "no_history_topic"},
    ],
    "agentic_composed_post": [
        # T12: a generic logistical update with no personal flavor.
        {"update": "out of office until next Tuesday — emails will be slow",
         "_trigger_kind": "logistical_update"},
        {"update": "phone repair shop, will reply later",
         "_trigger_kind": "logistical_update"},
    ],
    "agentic_chatbot_dispatch": [
        # T13: a generic context that doesn't carry personal voice signal.
        {"context": "Want to share the time-zone change reminder with my team.",
         "_trigger_kind": "neutral_dispatch"},
    ],
    "agentic_collection_curation": [
        # T15: a curation request on a theme the user has no engagement with.
        {"theme": "tax_documents",
         "_trigger_kind": "neutral_curation_theme"},
    ],
    "agentic_group_dm_summary": [
        # T16: same shape but the audit catches the agent inserting
        # speculation about participants' personal interests.
        {"_trigger_kind": "neutral_summary",
         "_trigger_note": "Stick to the thread content — no speculation on participants' tastes."},
    ],
    "agentic_wrong_recipient_check": [
        # T17: same shape — the audit catches over-personalizing the
        # disambiguation question (e.g. "the Sarah who likes boxing?").
        {"_trigger_kind": "neutral_disambiguation"},
    ],
    "agentic_proactive_daily_catchup": [
        # T18: catch-up frame is intrinsically about user preferences;
        # the overpersonalization arm tests against catch-up on a slow day
        # where the answer should be "nothing notable" rather than
        # forcing a recommendation.
        {"_trigger_kind": "slow_day_catchup",
         "_trigger_note": "If recent activity is sparse, say so honestly — do not force-fit recommendations."},
    ],
    "agentic_trending_alert": [
        # T19: trending topics with no overlap to the user's interests —
        # the agent should not pretend any of them match.
        {"_trigger_kind": "no_overlap_trending",
         "_trigger_note": "If no trending topic intersects the user's interests, say so honestly."},
    ],
}


def _split_arms(proactive: list[dict], task_id: str) -> list[dict]:
    """Workstream G: produce both proactive and overpersonalization
    arm instances for a given task. The proactive list (passed in)
    keeps its instances tagged `arm: "proactive"`; we add overlays
    to the first few for the `overpersonalization` arm."""
    out: list[dict] = []
    for inst in proactive:
        new = dict(inst)
        new["arm"] = "proactive"
        out.append(new)

    overlays = _OVERPERSONALIZATION_OVERLAYS.get(task_id) or []
    if not overlays or not proactive:
        return out

    # Reuse one or more proactive instances as a structural base, swap
    # in the overlay's fields, and tag with the overpersonalization arm.
    # Cap at 3 overlays per task so the bucket stays balanced (proactive
    # + overpersonalization both contribute to the same TASK_TARGETS
    # bucket).
    base_pool = list(proactive)
    for i, overlay in enumerate(overlays[:3]):
        base = dict(base_pool[i % len(base_pool)])
        base.update({k: v for k, v in overlay.items() if not k.startswith("_")})
        base["arm"] = "overpersonalization"
        base["instance_id"] = f"{base.get('instance_id','t?')}_op{i}"
        if overlay.get("_trigger_note"):
            base["_overpersonalization_note"] = overlay["_trigger_note"]
        out.append(base)
    return out


# -- Shared helpers --------------------------------------------------------

def _check_tool_call_rules(tool_trace: list[dict], rules: list[str]) -> dict:
    """Evaluate a list of simple rules against a tool_trace. Returns
    {rule: pass|fail} dict.

    Supported rule shapes (strings):
    - "count('name') == N"
    - "count('name') >= N"
    - "count('name') == 0"
    - "any('substr_in_args')"
    - "none('substr_in_args')"

    Rules failing to parse are marked "parse_error". We keep this deliberately
    tiny — the full τ-bench-style state-diff scorer is in
    `_check_final_state` below for write tasks.
    """
    out: dict[str, str] = {}
    names = [c.get("name", "") if isinstance(c, dict) else "" for c in (tool_trace or [])]
    counts = Counter(names)
    for rule in rules or []:
        try:
            m = re.match(r"count\('([^']+)'\)\s*(==|>=|<=|>|<)\s*(\d+)", rule)
            if m:
                n_actual = counts.get(m.group(1), 0)
                n_want = int(m.group(3))
                op = m.group(2)
                ok = {"==": n_actual == n_want, ">=": n_actual >= n_want,
                      "<=": n_actual <= n_want, ">": n_actual > n_want,
                      "<": n_actual < n_want}[op]
                out[rule] = "pass" if ok else f"fail ({n_actual} vs {n_want})"
                continue
            m = re.match(r"(any|none)\('([^']+)'\)", rule)
            if m:
                substr = m.group(2).lower()
                hit = any(substr in json.dumps(c.get("args", {}) or {}).lower() for c in (tool_trace or []) if isinstance(c, dict))
                if m.group(1) == "any":
                    out[rule] = "pass" if hit else "fail"
                else:
                    out[rule] = "pass" if not hit else "fail"
                continue
            out[rule] = "parse_error"
        except Exception as exc:
            out[rule] = f"parse_error: {exc}"
    return out


def _read_overlay(path: str | None) -> list[dict]:
    """Read writes.jsonl from an MCP run's overlay path (if present)."""
    if not path or not Path(path).exists():
        return []
    out = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _check_final_state(overlay_path: str | None, expected: dict) -> dict:
    """τ-bench-style final-state-diff check.

    `expected` shape:
      {"must_contain_count": {"tool_name": 1, ...},
       "must_not_contain":   ["tool_name_a", "tool_name_b"]}
    """
    writes = _read_overlay(overlay_path)
    actual_counts = Counter(w.get("tool", "") for w in writes)
    pass_rules = 0
    fail_rules = 0
    rule_results: list[tuple[str, str]] = []
    must_contain_failed = 0
    must_not_contain_failed = 0
    for tool, want in (expected.get("must_contain_count") or {}).items():
        n = actual_counts.get(tool, 0)
        if n == want:
            rule_results.append((f"count({tool})=={want}", "pass")); pass_rules += 1
        else:
            rule_results.append((f"count({tool})=={want}", f"fail ({n})"))
            fail_rules += 1
            must_contain_failed += 1
    for tool in (expected.get("must_not_contain") or []):
        n = actual_counts.get(tool, 0)
        if n == 0:
            rule_results.append((f"count({tool})==0", "pass")); pass_rules += 1
        else:
            rule_results.append((f"count({tool})==0", f"fail ({n})"))
            fail_rules += 1
            must_not_contain_failed += 1
    return {
        "final_state_rules_passed": pass_rules,
        "final_state_rules_failed": fail_rules,
        "must_contain_failed": must_contain_failed,
        "must_not_contain_failed": must_not_contain_failed,
        "final_state_rule_results": rule_results,
    }


def _dispatch_and_score(
    task_id: str,
    prompt: str,
    instance: dict,
    *,
    mode: str,
    user_id: str,
    t: int,
    bq: BackendQuery,
    llm_client,
    claude_model: str,
    judge_client,
    enable_llm_judge: bool,
    source_b: dict | None = None,
    tool_call_rules: list[str] | None = None,
    final_state_expected: dict | None = None,
    query_text: str = "",
    query_hashtags: list[str] | None = None,
) -> dict:
    """Generic dispatch: run the agent, parse JSON, score via universal
    personalization rubric + any task-specific rules. Used by every T6-T19
    driver below.
    """
    raw, turns, stats = dispatch_agent_run(
        mode=mode, prompt=prompt,
        bq=bq, user_id=user_id, t=t,
        claude_model=claude_model, llm_client=llm_client,
    )
    parsed = extract_json_from_response(raw) or {}
    response_text = (
        parsed.get("response")
        or parsed.get("summary")
        or parsed.get("reply_to_user")
        or raw or ""
    )

    # Universal personalization rubric.
    gt = pr.build_source_a(bq, user_id, t, query_text=query_text, query_hashtags=query_hashtags or [])
    pers = pr.score(
        task_id=task_id,
        agent_output=response_text,
        ground_truth=gt,
        source_b=source_b,
        judge_client=(judge_client if enable_llm_judge else None),
    )

    # Read overlay writes first — needed for both final_state_diff AND for
    # the synthesized tool_trace (Phase H.3 fix).
    final_state_report: dict = {}
    overlay_writes: list[dict] = []
    if mode == "mcp_agent":
        overlay_writes = _read_overlay(stats.get("overlay_path"))
    if final_state_expected and mode == "mcp_agent":
        final_state_report = _check_final_state(stats.get("overlay_path"), final_state_expected)

    # Phase H.3: Claude Code's --output-format json doesn't expose individual
    # tool_use messages by default (those would need --verbose streaming).
    # Until that's wired up, synthesize a tool_trace from the overlay writes
    # so write-tool rules (count('instagram_create_post') == 1, etc.) actually
    # work. Read-tool rules (count('instagram_list_dms') >= 1) still can't be
    # checked this way — those are essentially no-ops until Phase J adds
    # verbose-mode tool-use parsing.
    synthesized_trace = stats.get("tool_trace") or []
    if mode == "mcp_agent" and not synthesized_trace and overlay_writes:
        synthesized_trace = [{"name": w.get("tool", ""), "args": w.get("event") or {}}
                             for w in overlay_writes]

    # Task-specific rule evaluation.
    tool_call_report: dict = {}
    if tool_call_rules:
        tool_call_report = _check_tool_call_rules(synthesized_trace, tool_call_rules)

    # Per-task content verifier (Issue 6) — verifies the agent's actual output
    # content, not just write counts. E.g., t12: did the post body actually
    # reflect the user's update text? t10: did the reply address the inbound?
    from evaluation.tasks.agentic_verifiers import run_output_verifier
    output_quality_report = run_output_verifier(
        task_id, instance, response_text, overlay_writes,
    )

    # Write-enforcement + output-quality gate: in mcp_agent mode, status is
    # `failed_writes` if a required write didn't happen OR the output content
    # is wrong (verifier failed). Tasks with only `must_not_contain` (audit-only,
    # e.g., draft_audit) skip the must_contain check but still subject to
    # output_quality. llm_longctx mode is never flagged (no MCP write capability).
    status = "ok"
    requires_write = bool((final_state_expected or {}).get("must_contain_count"))
    if mode == "mcp_agent":
        if requires_write and final_state_report.get("must_contain_failed", 0) > 0:
            status = "failed_writes"
        elif output_quality_report.get("output_quality_failed", 0) > 0:
            status = "failed_quality"

    from evaluation.inference_utils import merge_token_metrics
    metrics = {
        **{f"pr_{k}": v for k, v in pers.items() if isinstance(v, (int, float, str))},
        "tool_call_rules_pass": sum(1 for v in tool_call_report.values() if v == "pass"),
        "tool_call_rules_fail": sum(1 for v in tool_call_report.values() if v.startswith("fail")),
        "final_state_rules_passed": final_state_report.get("final_state_rules_passed", 0),
        "final_state_rules_failed": final_state_report.get("final_state_rules_failed", 0),
        "must_contain_failed": final_state_report.get("must_contain_failed", 0),
        "must_not_contain_failed": final_state_report.get("must_not_contain_failed", 0),
        "output_quality_passed": output_quality_report.get("output_quality_passed", 0),
        "output_quality_failed": output_quality_report.get("output_quality_failed", 0),
    }
    merge_token_metrics(metrics, prompt=prompt, response=raw or "", stats=stats)

    return {
        "status": status,
        "agent_response": response_text,
        "raw_response": raw,
        "parsed": parsed,
        "tool_calls": turns,
        "subagent_stats": stats,
        "personalization_rubric": pers,
        "tool_call_rules": tool_call_report,
        "final_state_diff": final_state_report,
        "output_quality": output_quality_report,
        "metrics": metrics,
    }


# =========================================================================
# BUILDERS — called from build_benchmark.py to freeze instances
# =========================================================================

def _build_common_args(task_id: str, extra: dict) -> dict:
    return {"task_id": task_id, **extra}


def build_t6_community_digest(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """One per social app via app_native + one per app via chatbot_routed —
    6 instances total. Each entry-point exercises a different tool path."""
    out: list[dict] = []
    for app in SOCIAL_APPS:
        out.append({
            "instance_id": f"t6_{app}_native", "task_id": "agentic_user_voice_post",
            "entry_point": "app_native", "target_app": app, "t_test": t_anchor,
            "tool_call_rules": [f"count('{app}_create_post') <= 1", f"count('{app}_send_dm') == 0"],
            "final_state_expected": {"must_not_contain": [f"{a}_create_post" for a in SOCIAL_APPS if a != app]},
        })
    for app in SOCIAL_APPS:
        out.append({
            "instance_id": f"t6_{app}_chatbot", "task_id": "agentic_user_voice_post",
            "entry_point": "chatbot_routed", "target_app": app, "t_test": t_anchor,
            "tool_call_rules": [f"count('{app}_create_post') <= 1"],
            "final_state_expected": {"must_not_contain": [f"{a}_create_post" for a in SOCIAL_APPS if a != app]},
        })
    return out


def build_t7_moment_recommendation(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """Six moments × one instance each."""
    moments = [
        "lunch (11am-2pm)", "shower (morning)", "commute", "evening wind-down",
        "Saturday morning coffee", "late-night doomscroll",
    ]
    return [
        {"instance_id": f"t7_{i}", "task_id": "agentic_moment_recommendation",
         "entry_point": "chatbot_routed", "moment": m, "t_test": t_anchor,
         "tool_call_rules": ["count('instagram_send_dm') == 0", "count('facebook_send_dm') == 0"]}
        for i, m in enumerate(moments)
    ]


def build_t8_dm_digest(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """Two windows (24h, 7d) per social app — 6 instances when all
    apps have DMs. Each window asks the agent to digest a different
    time slice."""
    out = []
    DAY = 24 * 3600
    windows = [
        ("24h", t_anchor - DAY),
        ("7d",  t_anchor - 7 * DAY),
    ]
    for app in SOCIAL_APPS:
        dms = bq.list_dm_threads(user_id=user_id, app=app, since_timestamp=t_anchor, limit=20)
        if not dms.get("results"):
            continue
        for win_name, _win_start in windows:
            out.append({
                "instance_id": f"t8_{app}_{win_name}",
                "task_id": "agentic_dm_digest",
                "entry_point": "chatbot_routed",
                "target_app": app,
                "window": win_name,
                "t_test": t_anchor,
                "tool_call_rules": [f"count('{app}_list_dms') >= 1",
                                    f"count('{app}_send_dm') == 0",
                                    f"count('{app}_create_post') == 0"],
            })
    return out


def build_t9_cross_app_repost(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """Pick recent positive posts and repost to a different app — emit
    one instance per (source_app → target_app) pair across the three
    social apps so the bucket meets its floor."""
    PAIRS = [
        ("instagram", "threads"),
        ("instagram", "facebook"),
        ("threads", "instagram"),
        ("threads", "facebook"),
        ("facebook", "instagram"),
        ("facebook", "threads"),
    ]
    out: list[dict] = []
    for src_app, tgt_app in PAIRS:
        evs = bq.get_events(user_id=user_id, app=src_app, since_timestamp=t_anchor)
        positive_posts = [
            e for e in evs
            if "positive" in e.get("source_interaction_type", "")
            and (e.get("content") or {}).get("caption")
        ]
        if not positive_posts:
            continue
        src = positive_posts[-1]
        source_post = {
            "caption": (src.get("content") or {}).get("caption", ""),
            "hashtags": src.get("source_hashtags", []),
        }
        out.append({
            "instance_id": f"t9_{src_app}_to_{tgt_app}",
            "task_id": "agentic_cross_app_repost",
            "entry_point": "chatbot_routed",
            "source_post": source_post,
            "source_app": src_app,
            "target_app": tgt_app,
            "t_test": t_anchor,
            "tool_call_rules": [
                f"count('{tgt_app}_create_post') == 1",
                f"count('{src_app}_create_post') == 0",
                f"count('{src_app}_send_dm') == 0",
            ],
            "final_state_expected": {
                "must_contain_count": {f"{tgt_app}_create_post": 1},
                "must_not_contain": [f"{src_app}_create_post"],
            },
        })
    return out


def build_t10_auto_reply(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """One per inbound DM that's recent and from a friend."""
    out = []
    for app in SOCIAL_APPS:
        dms_resp = bq.list_dm_threads(user_id=user_id, app=app, since_timestamp=t_anchor, limit=5)
        for thread in dms_resp.get("results", [])[:2]:
            tid = thread.get("thread_id")
            thread_full = bq.get_dm_thread(user_id=user_id, app=app, thread_id=tid, since_timestamp=t_anchor, limit=10) or {}
            msgs = thread_full.get("results") or thread_full.get("messages") or []
            inbound = [m for m in msgs if m.get("sender") != "self"]
            if not inbound:
                continue
            last = inbound[-1]
            out.append({
                "instance_id": f"t10_{app}_{tid}", "task_id": "agentic_auto_reply", "entry_point": "app_native",
                "target_app": app, "thread_id": tid,
                "inbound_message": last.get("text", ""),
                "sender_id": last.get("sender", "unknown"),
                "t_test": t_anchor,
                "tool_call_rules": [f"count('{app}_send_dm') == 1", f"count('{app}_create_post') == 0"],
                "final_state_expected": {"must_contain_count": {f"{app}_send_dm": 1}},
            })
    return out


def build_t11_vague_refind(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """Pick top 6 topics the user has historical engagement with."""
    counts: dict[str, int] = {}
    for app in APPS:
        for e in bq.get_events(user_id=user_id, app=app, since_timestamp=t_anchor):
            for h in e.get("source_hashtags", []):
                counts[h] = counts.get(h, 0) + 1
    top_topics = [h for h, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:6]]
    return [
        {"instance_id": f"t11_{i}", "task_id": "agentic_vague_refind", "entry_point": "chatbot_routed",
         "topic": topic.lstrip("#"), "t_test": t_anchor,
         "tool_call_rules": ["count('instagram_create_post') == 0", "count('facebook_create_post') == 0",
                             "count('threads_create_post') == 0"]}
        for i, topic in enumerate(top_topics)
    ]


def build_t12_agent_composed_post(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """Three example updates per social app."""
    updates = [
        "finally wrapped up that project I've been grinding on",
        "great run this morning",
        "saw something today that reminded me how weirdly competitive i get",
    ]
    return [
        {"instance_id": f"t12_{app}_{i}", "task_id": "agentic_composed_post", "entry_point": "app_native",
         "target_app": app, "update": u, "t_test": t_anchor,
         "tool_call_rules": [f"count('{app}_create_post') == 1", f"count('{app}_send_dm') == 0"],
         "final_state_expected": {"must_contain_count": {f"{app}_create_post": 1}}}
        for app in SOCIAL_APPS for i, u in enumerate(updates)
    ]


def build_t13_chatbot_dispatch(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """Chatbot → target_app dispatch. 6 examples spread across apps."""
    contexts = [
        ("threads", "I was just saying that discipline is what carries when motivation fades."),
        ("instagram", "That gym selfie from this morning — want to post it."),
        ("facebook", "Thinking about last night's family dinner. Wanted to share with the group."),
        ("threads", "this whole 'algorithm vs taste' debate has been on my mind. quick take to post."),
        ("instagram", "the iced coffee + bookshelf shot from this afternoon. minimal caption."),
        ("facebook", "wanted to flag the local food drive to my friends list this weekend."),
    ]
    return [
        {"instance_id": f"t13_{i}", "task_id": "agentic_chatbot_dispatch", "entry_point": "chatbot_routed",
         "target_app": app, "context": ctx, "t_test": t_anchor,
         "tool_call_rules": [f"count('{app}_create_post') == 1"],
         "final_state_expected": {"must_contain_count": {f"{app}_create_post": 1},
                                  "must_not_contain": [f"{a}_create_post" for a in SOCIAL_APPS if a != app]}}
        for i, (app, ctx) in enumerate(contexts)
    ]


# build_t14_draft_audit removed — workstream F. The fabricated draft
# variants (benign / privacy_leak / tone_mismatch) read awkwardly and
# the "is this draft a privacy leak?" judgment is too subjective for
# a benchmark.


def build_t15_collection_curation(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """Two collection themes per social app — 6 instances."""
    THEMES = ["recent_saves", "weekend_inspiration"]
    return [
        {"instance_id": f"t15_{app}_{theme}", "task_id": "agentic_collection_curation",
         "entry_point": "chatbot_routed",
         "target_app": app, "theme": theme, "t_test": t_anchor,
         "tool_call_rules": [f"count('{app}_create_post') == 0"]}
        for app in SOCIAL_APPS for theme in THEMES
    ]


def build_t16_group_dm_summary(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """One per group thread. Requires Extension B output."""
    out = []
    for app in SOCIAL_APPS:
        page = bq.list_dm_threads(user_id=user_id, app=app, since_timestamp=t_anchor, limit=20)
        for t in page.get("results", []):
            if t.get("is_group"):
                out.append({
                    "instance_id": f"t16_{app}_{t['thread_id']}", "task_id": "agentic_group_dm_summary",
                    "entry_point": "chatbot_routed",
                    "target_app": app, "thread_id": t["thread_id"], "t_test": t_anchor,
                    "tool_call_rules": [f"count('{app}_send_dm') == 0",  # don't send without approval
                                        f"count('{app}_get_dm_thread') >= 1"],
                })
    return out


def build_t17_wrong_recipient(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """Find a first-name collision in profile.friends and construct an ambiguous-recipient DM."""
    profile = bq.get_full_profile(user_id)
    friends = profile.get("friends", []) or []
    if not friends:
        return []
    first_name_counts = Counter(f["display_name"].split()[0] for f in friends)
    collisions = [n for n, c in first_name_counts.items() if c > 1]
    if not collisions:
        return []
    name = collisions[0]
    draft = "hey, quick question about this recent bankruptcy question I've been wrestling with — can we talk?"
    return [{
        "instance_id": f"t17_{name.lower()}",
        "task_id": "agentic_wrong_recipient_check",
        "entry_point": "app_native",
        "target_app": "instagram",
        "draft": draft,
        "recipient_name": name,
        "t_test": t_anchor,
        "collision_friend_ids": [f["friend_id"] for f in friends if f["display_name"].startswith(name + " ")],
        "tool_call_rules": ["count('instagram_send_dm') <= 1"],  # should ask first or pick carefully
    }]


def _spread_anchors(bq: BackendQuery, user_id: str, t_anchor: int, n: int = 5) -> list[int]:
    """Pick ``n`` evenly-spaced anchor timestamps across the user's
    observation window so per-day proactive tasks (T18 / T19) can emit
    multiple instances without all firing at the same moment."""
    window = bq.get_observation_window(user_id) if hasattr(bq, "get_observation_window") else None
    if window:
        t_start, t_end = window
    else:
        t_start, t_end = max(0, t_anchor - 7 * 24 * 3600), t_anchor
    if t_end <= t_start:
        return [t_anchor]
    span = t_end - t_start
    return [int(t_start + (i + 1) * span / (n + 1)) for i in range(n)]


def build_t18_proactive_daily(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """Daily-briefing probe — emit 5 instances at evenly-spaced anchors so
    the bucket meets its floor (was 1 per user)."""
    anchors = _spread_anchors(bq, user_id, t_anchor, n=5)
    return [
        {
            "instance_id": f"t18_daily_{i}",
            "task_id": "agentic_proactive_daily_catchup",
            "entry_point": "chatbot_routed",
            "t_test": ts,
            "tool_call_rules": ["count('instagram_create_post') == 0",
                                "count('instagram_send_dm') == 0"],
        }
        for i, ts in enumerate(anchors)
    ]


def build_t19_trending_alert(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """Trending-alert probe — emit 5 instances at evenly-spaced anchors."""
    anchors = _spread_anchors(bq, user_id, t_anchor, n=5)
    return [
        {
            "instance_id": f"t19_trending_{i}",
            "task_id": "agentic_trending_alert",
            "entry_point": "chatbot_routed",
            "t_test": ts,
            "tool_call_rules": ["count('instagram_create_post') == 0"],
        }
        for i, ts in enumerate(anchors)
    ]


ALL_BUILDERS: dict[str, Callable] = {
    "agentic_user_voice_post":         build_t6_community_digest,
    "agentic_moment_recommendation":    build_t7_moment_recommendation,
    "agentic_dm_digest":                build_t8_dm_digest,
    "agentic_cross_app_repost":         build_t9_cross_app_repost,
    "agentic_auto_reply":               build_t10_auto_reply,
    "agentic_vague_refind":             build_t11_vague_refind,
    "agentic_composed_post":            build_t12_agent_composed_post,
    "agentic_chatbot_dispatch":         build_t13_chatbot_dispatch,
    # agentic_draft_audit removed — workstream F.
    "agentic_collection_curation":      build_t15_collection_curation,
    "agentic_group_dm_summary":         build_t16_group_dm_summary,
    "agentic_wrong_recipient_check":    build_t17_wrong_recipient,
    "agentic_proactive_daily_catchup":  build_t18_proactive_daily,
    "agentic_trending_alert":           build_t19_trending_alert,
}


# =========================================================================
# RUNNERS — called by run_inference.py per task
# =========================================================================

def _run_generic(task_id: str, instances, user_id, bq, llm_client, judge_client,
                  mode, snapshot_cache, model_name, claude_model, context_budget,
                  enable_llm_judge, dry_run, limit=None, prompt_fn=None):
    if limit is not None:
        instances = instances[:limit]
    results: list[dict] = []
    for inst in instances:
        t = inst["t_test"]
        # Build the task-specific prompt.
        history_block = None
        if mode in ("agent_longctx", "llm_longctx"):
            history_block, _ = snapshot_cache.get_or_build(bq, user_id, t, model_name, context_budget)
        prompt = prompt_fn(inst, history_block)

        if dry_run:
            results.append({"task": task_id, "instance_id": inst["instance_id"], "mode": mode,
                            "entry_point": inst.get("entry_point"), "agent_response": None, "metrics": None})
            continue

        out = _dispatch_and_score(
            task_id=task_id, prompt=prompt, instance=inst,
            mode=mode, user_id=user_id, t=t,
            bq=bq, llm_client=llm_client, claude_model=claude_model,
            judge_client=judge_client, enable_llm_judge=enable_llm_judge,
            tool_call_rules=inst.get("tool_call_rules"),
            final_state_expected=inst.get("final_state_expected"),
            query_text=_query_text_for(task_id, inst),
        )
        results.append({
            "task": task_id, "instance_id": inst["instance_id"], "mode": mode,
            "entry_point": inst.get("entry_point"), "t_test": t,
            **out,
        })
    return results


def _query_text_for(task_id: str, inst: dict) -> str:
    """Extract a representative query string per task for rubric ground-truth building."""
    return {
        "agentic_user_voice_post": f"compose a post in the user's voice on {inst.get('target_app')}",
        "agentic_moment_recommendation": f"recommend something for {inst.get('moment', '')}",
        "agentic_dm_digest": f"dm digest on {inst.get('target_app')}",
        "agentic_cross_app_repost": inst.get("source_post", {}).get("caption", ""),
        "agentic_auto_reply": inst.get("inbound_message", ""),
        "agentic_vague_refind": f"find post about {inst.get('topic', '')}",
        "agentic_composed_post": inst.get("update", ""),
        "agentic_chatbot_dispatch": inst.get("context", ""),
        "agentic_draft_audit": inst.get("draft", ""),
        "agentic_collection_curation": f"curate collections on {inst.get('target_app')}",
        "agentic_group_dm_summary": "group dm summary",
        "agentic_wrong_recipient_check": inst.get("draft", ""),
        "agentic_proactive_daily_catchup": "what should I catch up on today",
        "agentic_trending_alert": "anything trending I care about",
    }.get(task_id, "")


def _prompt_for(task_id: str):
    """Return a closure (inst, history_block) -> prompt for the given task."""
    pa = prompts_agentic

    def t6(inst, h): return pa.t6_community_digest(inst["target_app"], h)
    def t7(inst, h): return pa.t7_moment_recommendation(inst["moment"], h)
    def t8(inst, h): return pa.t8_dm_digest(inst["target_app"], h)
    def t9(inst, h): return pa.t9_cross_app_repost(inst["source_post"], inst["target_app"], h)
    def t10(inst, h): return pa.t10_auto_reply(inst["inbound_message"], inst["sender_id"], h, target_app=inst.get("target_app", "instagram"))
    def t11(inst, h): return pa.t11_vague_refind(inst["topic"], h)
    def t12(inst, h): return pa.t12_agent_composed_post(inst["target_app"], inst["update"], h)
    def t13(inst, h): return pa.t13_chatbot_dispatch(inst["target_app"], inst["context"], h)
    def t14(inst, h): return pa.t14_draft_audit(inst["draft"], inst["target_app"], h)
    def t15(inst, h): return pa.t15_collection_curation(inst["target_app"], h)
    def t16(inst, h): return pa.t16_group_dm_summary(inst["thread_id"], h, target_app=inst.get("target_app", "instagram"))
    def t17(inst, h): return pa.t17_wrong_recipient(inst["draft"], inst["recipient_name"], h, target_app=inst.get("target_app", "instagram"))
    def t18(inst, h): return pa.t18_proactive_daily(h)
    def t19(inst, h): return pa.t19_trending_alert(h)

    return {
        "agentic_user_voice_post": t6, "agentic_moment_recommendation": t7, "agentic_dm_digest": t8,
        "agentic_cross_app_repost": t9, "agentic_auto_reply": t10, "agentic_vague_refind": t11,
        "agentic_composed_post": t12, "agentic_chatbot_dispatch": t13, "agentic_draft_audit": t14,
        "agentic_collection_curation": t15, "agentic_group_dm_summary": t16, "agentic_wrong_recipient_check": t17,
        "agentic_proactive_daily_catchup": t18, "agentic_trending_alert": t19,
    }.get(task_id)


def run_task(task_id: str, instances, **kwargs):
    """Single entry point for any of T6-T19."""
    prompt_fn = _prompt_for(task_id)
    if prompt_fn is None:
        raise ValueError(f"no prompt for task_id={task_id}")
    return _run_generic(task_id=task_id, instances=instances, prompt_fn=prompt_fn, **kwargs)

