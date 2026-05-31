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
from evaluation import ground_truth_builders
from evaluation import metrics as metrics_mod
from evaluation import personalization_rubric as pr
from evaluation import prompts_agentic
from evaluation.backend_query import APPS, BackendQuery
from evaluation.inference_utils import SnapshotCache, dispatch_agent_run


SOCIAL_APPS = ("instagram", "facebook", "threads")


# Workstream G's per-task over-personalization arm split was REMOVED.
# Reason: the chatbot family already has dedicated `over_personalization_*`
# task types for restraint testing, and a separate `arm` field on agentic
# instances created identical example_responses across arms (the op-arm
# never produced a generic counterpart). Cleaner to leave agentic tasks
# single-arm; overpersonalization testing is covered by chatbot tasks.


# -- Persona-relevance filters --------------------------------------------
#
# Selection criteria for "this instance actually requires the user's persona
# to answer well." Without these filters we accumulate noise instances —
# e.g. agentic_auto_reply on a logistics-only DM ("yeah saturday works"),
# or agentic_dm_digest on a user with one boring thread — where memory-
# using and memory-blind agents grade identically. Those instances dilute
# the benchmark's signal.
#
# Build a single persona-topic index per user once per build pass, then
# every task-specific filter reads from it.


def _build_persona_topic_index(bq: BackendQuery, user_id: str, t_anchor: int) -> dict:
    """Snapshot the user's top categories + top hashtags + friends index
    at t_anchor. Returns a dict the per-task filters consume.

    The `top_hashtags` set unions two sources:
      (a) the top 30 by raw event count, AND
      (b) every hashtag listed in any hidden_persona's `evidence_hashtags`.
    Niche hashtags that anchor on a hidden persona (e.g. `#Nems` for an
    East-Coast-rap identity_anchor) wouldn't show up in the top-30 count
    bucket but ARE persona-relevant by construction — including them here
    fixes agentic_group_dm_summary / dm_digest false negatives where the
    thread's source_hashtags were niche-but-aligned.
    """
    cat_counts: Counter = Counter()
    hashtag_counts: Counter = Counter()
    for app in SOCIAL_APPS:
        for e in bq.get_events(user_id=user_id, app=app, since_timestamp=t_anchor):
            for pref in (e.get("preferences") or []):
                cat = (pref.get("category") or "").strip().lower()
                if cat:
                    cat_counts[cat] += 1
            for h in (e.get("source_hashtags") or []):
                hashtag_counts[h.lstrip("#").lower()] += 1
    profile = bq.get_full_profile(user_id) or {}
    friends_by_id = {f.get("friend_id"): f for f in (profile.get("friends") or [])}
    top_hashtags: set[str] = {h for h, _ in hashtag_counts.most_common(30)}
    # Union with hidden-persona evidence_hashtags.
    for hp in (profile.get("hidden_personas") or []):
        for h in (hp.get("evidence_hashtags") or []):
            tag = (h or "").lstrip("#").lower()
            if tag:
                top_hashtags.add(tag)
    return {
        "top_cats": {c for c, _ in cat_counts.most_common(8)},
        "top_hashtags": top_hashtags,
        "friends_by_id": friends_by_id,
    }


def _text_touches_persona(target, idx: dict) -> bool:
    """True if `target` is persona-relevant. Accepts either a string
    (legacy callers) OR a dict (DM thread / event).

    A dict-shaped target also checks `source_hashtags` for set-overlap
    with the user's top hashtags. This matters for forwarded-post DMs:
    a thread carrying an NFL clip IS NFL-relevant even when the message
    text is just "lol" or "saw this" — the user wouldn't restate the
    topic when forwarding, but the carried hashtags are unambiguous.

    Short category words (≤3 chars) are ignored to avoid spurious matches
    on common English words."""
    if isinstance(target, dict):
        # Surface 1: hashtag overlap from the carried event / forward.
        carried = {(h or "").lstrip("#").lower() for h in (target.get("source_hashtags") or [])}
        if carried & idx["top_hashtags"]:
            return True
        # Surface 2: text fields on the event / thread.
        text_fields = (
            target.get("last_message_preview"),
            target.get("text"),
            (target.get("content") or {}).get("caption") if isinstance(target.get("content"), dict) else None,
            (target.get("content") or {}).get("title") if isinstance(target.get("content"), dict) else None,
        )
        text = " ".join(t for t in text_fields if t)
    else:
        text = target or ""
    if not text:
        return False
    t = text.lower()
    for h in idx["top_hashtags"]:
        if h and h in t:
            return True
    for c in idx["top_cats"]:
        # Split categories ("comedy video content") into tokens so we
        # match on any meaningful word, not just the full phrase.
        for word in (c or "").split():
            if len(word) > 3 and word in t:
                return True
    return False


def _is_opinion_or_recommendation_request(text: str) -> bool:
    """Heuristic: inbound message is asking for the user's take, advice,
    or recommendation — i.e. their preferences shape the right reply."""
    if not text:
        return False
    t = text.lower()
    if "?" in t:
        return True
    triggers = (
        "recommend", "thoughts on", "what do you think", "your take",
        "should i", "any tips", "any ideas", "suggest", "advice",
        "into it", "you'd like", "you should", "you'll like",
    )
    return any(tr in t for tr in triggers)


def _friend_shares_persona(friend: dict | None, idx: dict) -> bool:
    """True if the friend's `shared_interests` overlap the user's top
    categories or hashtags — meaning the relationship/interest context
    materially shapes the right reply tone."""
    if not friend:
        return False
    sis = (friend.get("shared_interests") or [])
    if not sis:
        return False
    for si in sis:
        s = (si or "").lower().lstrip("#")
        if s in idx["top_hashtags"]:
            return True
        for c in idx["top_cats"]:
            for word in (c or "").split():
                if len(word) > 3 and word in s:
                    return True
    return False


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


# _score_moment_recommendation removed — moment instances now ride the
# slate-based personalized_recommendation path with deterministic
# recall@k / ndcg@k / mrr metrics. The old agentic scoring was specific to
# the MCP-feed-call flavor, which doesn't have a live backend in this repo.


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
    parsed = extract_json_from_response(raw)
    # `final_answer` is the llm_longctx (text-only) JSON key — checked first
    # so write-task content lands in response_text instead of falling through
    # to a stale `summary` field. Defensive `isinstance` because the
    # parser can return a list/scalar/None when the LLM doesn't emit a
    # JSON object — `.get(...)` would crash on those. Also coerce the
    # picked field to a string: T16 group-DM summaries sometimes come
    # back as `{"summary": {"alice": "...", "bob": "..."}}` (dict instead
    # of string), and downstream rubric scoring calls `re.findall(s)`
    # which raises TypeError on non-strings.
    if isinstance(parsed, dict):
        picked = (
            parsed.get("final_answer")
            or parsed.get("response")
            or parsed.get("summary")
            or parsed.get("reply_to_user")
            or raw or ""
        )
    else:
        picked = raw or ""
    if isinstance(picked, str):
        response_text = picked
    else:
        # Flatten dict/list to JSON text so downstream string ops work.
        response_text = json.dumps(picked, ensure_ascii=False)

    # Universal personalization rubric.
    gt = pr.build_source_a(bq, user_id, t, query_text=query_text, query_hashtags=query_hashtags or [])
    pers = pr.score(
        task_id=task_id,
        agent_output=response_text,
        ground_truth=gt,
        source_b=source_b,
        judge_client=(judge_client if enable_llm_judge else None),
    )

    # llm_longctx is graded final-answer-only: rubric on response_text, no
    # tool-call rules, no overlay readout, no output_quality verifier (those
    # all assume an MCP overlay that doesn't exist in this mode). Returning
    # early keeps the cross-mode comparison honest — see DESIGN.md.
    from evaluation.inference_utils import merge_token_metrics
    if mode == "llm_longctx":
        metrics = {
            **{f"pr_{k}": v for k, v in pers.items() if isinstance(v, (int, float, str))},
            "mode_grading": "final_answer_only",
        }
        merge_token_metrics(metrics, prompt=prompt, response=raw or "", stats=stats)
        return {
            "status": "ok",
            "agent_response": response_text,
            "raw_response": raw,
            "parsed": parsed,
            "tool_calls": 0,
            "subagent_stats": stats,
            "personalization_rubric": pers,
            "tool_call_rules": {},
            "final_state_diff": {},
            "output_quality": {},
            "metrics": metrics,
        }

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
    # output_quality.
    status = "ok"
    requires_write = bool((final_state_expected or {}).get("must_contain_count"))
    if mode == "mcp_agent":
        if requires_write and final_state_report.get("must_contain_failed", 0) > 0:
            status = "failed_writes"
        elif output_quality_report.get("output_quality_failed", 0) > 0:
            status = "failed_quality"

    # Per-task deterministic backstops live here when needed. The moment
    # backstop was removed when agentic_moment_recommendation was merged
    # into personalized_recommendation (slate-based ranking metrics are
    # the authority now — recall@k / ndcg@k / mrr).
    moment_report: dict = {}

    metrics = {
        **{f"pr_{k}": v for k, v in pers.items() if isinstance(v, (int, float, str))},
        "mode_grading": "full",
        "tool_call_rules_pass": sum(1 for v in tool_call_report.values() if v == "pass"),
        "tool_call_rules_fail": sum(1 for v in tool_call_report.values() if v.startswith("fail")),
        "final_state_rules_passed": final_state_report.get("final_state_rules_passed", 0),
        "final_state_rules_failed": final_state_report.get("final_state_rules_failed", 0),
        "must_contain_failed": final_state_report.get("must_contain_failed", 0),
        "must_not_contain_failed": final_state_report.get("must_not_contain_failed", 0),
        "output_quality_passed": output_quality_report.get("output_quality_passed", 0),
        "output_quality_failed": output_quality_report.get("output_quality_failed", 0),
        **moment_report,
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


_T6_QUERY_GEN_PROMPT = """Generate a natural, casual message from a user asking their AI to write a post about {topic} on {app}.

User context:
- Name: {name}
- Career: {career}

Constraints:
- 5-15 words, casual phone message
- Ask the AI to write/draft/post for them
- Mention the topic and platform naturally
- Sound like a real person texting

Return ONLY JSON:
```json
{{"query": "the message"}}
```"""


def build_t6_user_tone_post(bq: BackendQuery, user_id: str, t_anchor: int,
                             discovery_llm=None) -> list[dict]:
    """One per social app via app_native + one per app via chatbot_routed —
    6 instances total. Each entry-point exercises a different tool path.

    Pre-test floor: skip the whole task when the user has fewer than
    `USER_VOICED_SAMPLES_FLOOR` user-voiced samples (self-posts, DM
    user-side, chatbot user-turns) before t_anchor — without enough
    voice evidence, the AI under eval has nothing to mimic.
    """
    if not ground_truth_builders.has_enough_user_voiced_history(bq, user_id, t_anchor):
        return []

    profile = {}
    if discovery_llm is not None:
        try:
            import json as _json
            from pathlib import Path as _Path
            pp = _Path(getattr(bq, "base", "backend")) / user_id / "profile.json"
            if pp.exists():
                profile = _json.loads(pp.read_text())
        except Exception:
            pass

    def _gen_query(app: str, entry: str) -> str | None:
        if discovery_llm is None:
            return None
        top_hashtags = profile.get("top_hashtags") or []
        topic = ""
        for h, _cnt in top_hashtags:
            if h.lower() not in {"life", "love", "selfie", "photo", "vibes", "mood"}:
                topic = h.lower().lstrip("#")
                break
        if not topic:
            return None
        from data_preparation.utils import extract_json_from_response
        try:
            prompt = _T6_QUERY_GEN_PROMPT.format(
                topic=topic, app=app,
                name=profile.get("name", "the user"),
                career=profile.get("career", ""),
            )
            raw = discovery_llm.query_llm(prompt)
            parsed = extract_json_from_response(raw) or {}
            if isinstance(parsed, dict):
                q = (parsed.get("query") or "").strip()
                if q and 3 <= len(q.split()) <= 20:
                    return q
        except Exception:
            pass
        return None

    out: list[dict] = []
    for app in SOCIAL_APPS:
        inst = {
            "instance_id": f"t6_{app}_native", "task_id": "agentic_community_post",
            "entry_point": "app_native", "target_app": app, "t_test": t_anchor,
            "tool_call_rules": [f"count('{app}_create_post') <= 1", f"count('{app}_send_dm') == 0"],
            "final_state_expected": {"must_not_contain": [f"{a}_create_post" for a in SOCIAL_APPS if a != app]},
        }
        q = _gen_query(app, "app_native")
        if q:
            inst["user_query"] = q
        out.append(inst)
    for app in SOCIAL_APPS:
        inst = {
            "instance_id": f"t6_{app}_chatbot", "task_id": "agentic_community_post",
            "entry_point": "chatbot_routed", "target_app": app, "t_test": t_anchor,
            "tool_call_rules": [f"count('{app}_create_post') <= 1"],
            "final_state_expected": {"must_not_contain": [f"{a}_create_post" for a in SOCIAL_APPS if a != app]},
        }
        q = _gen_query(app, "chatbot_routed")
        if q:
            inst["user_query"] = q
        out.append(inst)
    return out


# --- T7 moment-recommendation: hour-window + slate constants ---------------
MOMENT_HOUR_WINDOWS_UTC: dict[str, tuple[int, int]] = {
    "lunch (11am-2pm)":        (11, 14),
    "shower (morning)":        (6,  9),
    "commute":                 (7, 10),
    "evening wind-down":       (20, 23),
    "Saturday morning coffee": (8, 11),
    "late-night doomscroll":   (23, 26),   # 23:00..02:00 wrap
}
MOMENT_DAY_FILTER: dict[str, set[int]] = {
    "Saturday morning coffee": {5},        # 0=Mon..6=Sun
}
N_RECOMMENDED_POSTS: int = 12              # hard floor
N_TARGET_RECOMMENDED_POSTS: int = 15
N_ALIGNED_FLOOR: int = 3


def _summarize_post_for_slate(e: dict, app: str) -> dict | None:
    """Compact projection of an engagement event into a slate candidate.
    Drops entries with empty title AND empty caption (would render blank).
    """
    content = e.get("content") or {}
    title = (content.get("title") or "").strip()
    caption = (content.get("caption") or "").strip()
    if not title and not caption:
        return None
    return {
        "source_object_id": str(e.get("source_object_id") or ""),
        "title": title,
        "caption": caption,
        "hashtags": list(e.get("source_hashtags") or [])[:6],
        "source_app": app,
        "source_timestamp": int(e.get("source_timestamp") or 0),
        "source_interaction_type": e.get("source_interaction_type") or "",
    }


def _hour_in_window(hour: int, window: tuple[int, int]) -> bool:
    """Hour window predicate with wraparound support (e.g. (23, 26) means
    23:00..02:00 — `hour >= 23 or hour < 2`)."""
    lo, hi = window
    if hi <= 24:
        return lo <= hour < hi
    return hour >= lo or hour < (hi - 24)


def _collect_moment_engagements(bq: BackendQuery, user_id: str, t_anchor: int,
                                 moment: str) -> dict:
    """Bucket the user's pre-t_anchor social engagements for one moment.

    Returns four lists, each sorted by ``abs(source_timestamp - t_anchor)``
    ascending (closest-to-test-moment first):
      - ``aligned_positive``       — in-moment-window positives
      - ``aligned_negative``       — in-moment-window negatives
      - ``neutral_positive_pool``  — out-of-window positives (safe filler)
      - ``all_negative_pool``      — any-time negatives (foil fallback)

    Widens the moment window by ±1h once if in-window positives are below
    the floor (`N_ALIGNED_FLOOR`).
    """
    import datetime as _dt
    window = MOMENT_HOUR_WINDOWS_UTC[moment]
    day_filter = MOMENT_DAY_FILTER.get(moment)

    def _bucketize(win: tuple[int, int]) -> dict:
        aligned_positive: list[dict] = []
        aligned_negative: list[dict] = []
        neutral_positive_pool: list[dict] = []
        all_negative_pool: list[dict] = []
        for app in SOCIAL_APPS:
            for e in bq.get_events(user_id=user_id, app=app, since_timestamp=t_anchor):
                ts = int(e.get("source_timestamp") or 0)
                if not ts:
                    continue
                proj = _summarize_post_for_slate(e, app)
                if proj is None:
                    continue
                prefs = e.get("preferences") or []
                if prefs:
                    first_pref = prefs[0]
                    proj["_held_out_persona_item"] = (first_pref.get("persona_item") or "").strip()
                    proj["_held_out_category"] = (first_pref.get("category") or "").strip()
                dt = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc)
                in_win = _hour_in_window(dt.hour, win)
                if day_filter is not None and dt.weekday() not in day_filter:
                    in_win = False
                itype = (proj["source_interaction_type"] or "").lower()
                is_pos = "positive" in itype
                is_neg = "negative" in itype
                if in_win and is_pos:
                    aligned_positive.append(proj)
                elif in_win and is_neg:
                    aligned_negative.append(proj)
                elif is_pos:
                    neutral_positive_pool.append(proj)
                if is_neg:
                    all_negative_pool.append(proj)
        for lst in (aligned_positive, aligned_negative,
                    neutral_positive_pool, all_negative_pool):
            lst.sort(key=lambda p: abs(p["source_timestamp"] - t_anchor))
        return {
            "aligned_positive": aligned_positive,
            "aligned_negative": aligned_negative,
            "neutral_positive_pool": neutral_positive_pool,
            "all_negative_pool": all_negative_pool,
        }

    buckets = _bucketize(window)
    if len(buckets["aligned_positive"]) < N_ALIGNED_FLOOR:
        widened = (max(0, window[0] - 1), window[1] + 1)
        buckets = _bucketize(widened)
    return buckets


_MOMENT_QUERY_TEMPLATES_LOW_FORMALITY = (
    "open the feeds, it's {moment_short}",
    "{moment_short} — what's worth opening?",
    "curate my socials for {moment_short}",
    "show me something for {moment_short}",
    "feed me, it's {moment_short}",
)
_MOMENT_QUERY_TEMPLATES_MID_FORMALITY = (
    "Open my social feeds for {moment_short}.",
    "What's worth seeing right now? It's {moment_short}.",
    "Curate my feeds for {moment_short}.",
    "Pull together a feed for {moment_short}.",
)
_MOMENT_QUERY_TEMPLATES_HIGH_FORMALITY = (
    "Could you open my social feeds for {moment_short}?",
    "Please curate my feeds for {moment_short}.",
    "What's worth seeing on my feeds right now? It's {moment_short}.",
)


_MOMENT_QUERY_GEN_PROMPT = """Generate a natural, casual message from a user asking their AI assistant to open/curate their social media feeds for {moment_short}.

User voice context:
- Formality: {formality_desc}
- Capitalization: {caps}

Constraints:
- 5-15 words, like a real phone message
- Mention the moment/time naturally
- Don't use "I know you..." or similar
- Match the user's formality level

Return ONLY JSON:
```json
{{"query": "the message"}}
```"""


def _voice_flavored_moment_query(moment: str, user_voice: dict, rng: random.Random,
                                 discovery_llm=None) -> str:
    """Build a moment-aware user query that reads like the user typing on
    their phone. Pulls phrasing register from `user_voice.formality_baseline`
    + `default_capitalization`, and may inline one of the user's catchphrase
    residue items deterministically per (moment, user) pair.

    Falls back to a neutral mid-formality template when user_voice is
    missing or malformed. Backward-compat: reads `idiolect.catchphrase_residue`
    from the new schema and falls back to legacy `personal_phrases`.
    """
    # Strip the parenthetical hour annotation so the query reads naturally
    # ("evening wind-down" not "evening wind-down (20-23)").
    moment_short = moment.split("(", 1)[0].strip() or moment
    formality = 0.3
    caps = ""
    phrases: list[str] = []
    if isinstance(user_voice, dict):
        try:
            formality = float(user_voice.get("formality_baseline", 0.3) or 0.3)
        except (TypeError, ValueError):
            formality = 0.3
        caps = str(user_voice.get("default_capitalization") or "")
        idio = user_voice.get("idiolect") or {}
        residue = (idio.get("catchphrase_residue") if isinstance(idio, dict) else None) \
            or user_voice.get("personal_phrases") or []
        phrases = [p for p in residue if isinstance(p, str) and p.strip()]

    # Try LLM generation first, fall back to templates.
    base = None
    if discovery_llm is not None:
        from data_preparation.utils import extract_json_from_response
        formality_desc = "casual" if formality < 0.35 else ("moderate" if formality < 0.65 else "formal")
        try:
            prompt = _MOMENT_QUERY_GEN_PROMPT.format(
                moment_short=moment_short,
                formality_desc=formality_desc,
                caps=caps or "mixed",
            )
            raw = discovery_llm.query_llm(prompt)
            parsed = extract_json_from_response(raw) or {}
            if isinstance(parsed, dict):
                q = (parsed.get("query") or "").strip()
                if q and 3 <= len(q.split()) <= 20:
                    base = q
        except Exception:
            pass

    if base is None:
        if formality < 0.35:
            templates = _MOMENT_QUERY_TEMPLATES_LOW_FORMALITY
        elif formality < 0.65:
            templates = _MOMENT_QUERY_TEMPLATES_MID_FORMALITY
        else:
            templates = _MOMENT_QUERY_TEMPLATES_HIGH_FORMALITY
        base = rng.choice(templates).format(moment_short=moment_short)

    # Optionally weave in a personal phrase — only for low/mid formality, and
    # only ~50% of the time so it doesn't get repetitive across moments. The
    # phrase reads as a soft opener, not a forced injection. Lowercase the
    # first letter of the base when a phrase is prepended so the comma
    # doesn't introduce a sentence break ("Love this, curate ..." reads
    # naturally; "Love this, Curate ..." doesn't).
    if phrases and formality < 0.65 and rng.random() < 0.5:
        phrase = rng.choice(phrases).rstrip(".!? ")
        if base and base[0].isupper():
            base = base[0].lower() + base[1:]
        base = f"{phrase}, {base}"

    if caps == "all_lowercase":
        base = base.lower()
    elif caps == "sentence_case":
        # Capitalize first letter only if the template / phrase didn't
        # already start with a capital (preserve lowercase-only voices).
        if base and base[0].islower() and formality >= 0.35:
            base = base[0].upper() + base[1:]
    return base


def build_t7_moment_recommendation(bq: BackendQuery, user_id: str, t_anchor: int,
                                    discovery_llm=None) -> list[dict]:
    """Build moment-aware ``personalized_recommendation``-shaped instances.

    Previously emitted ``agentic_moment_recommendation`` instances that
    required ``mcp__{app}_get_feed`` MCP tools to actually run. Those tools
    don't have a live database backend in this repo, making the agentic
    path impractical to test. This builder now emits slate-based ranking
    instances with the SAME shape as
    ``personalized_recommendation`` (16-item slate, ``held_out_idx``,
    ``hard_negative_idxs``) but with a moment-flavored ``query_text`` —
    voiced in the user's own register — instead of the empty query_text
    used by the proactive recsys flavor.

    Slate construction per moment that survives the `N_RECOMMENDED_POSTS`
    pool floor:
      - held-out (target rank 1): the in-window positive closest to the
        anchor in time.
      - hard negatives: up to `_T7_HARD_NEGATIVES` from the moment's negative
        pool (in-window first, then any-time negatives).
      - fillers: drawn from the neutral positive pool until the slate
        reaches `_T7_SLATE_SIZE` items.

    Scoring uses the deterministic ranking metrics from
    ``personalized_recommendation`` (recall@k / ndcg@k / mrr / hit@k) — no
    LLM judge, no tool calls.
    """
    from evaluation.tasks.personalized_recommendation import SLATE_SIZE, N_HARD_NEGATIVES

    rng = random.Random(hash(user_id) % (2**31))
    profile = {}
    try:
        profile = bq._load_profile(user_id) if hasattr(bq, "_load_profile") else {}
    except Exception:
        profile = {}
    user_voice = (profile or {}).get("user_voice") or {}

    out: list[dict] = []
    for i, moment in enumerate(MOMENT_HOUR_WINDOWS_UTC):
        buckets = _collect_moment_engagements(bq, user_id, t_anchor, moment)
        aligned_pos = list(buckets["aligned_positive"])
        neutral_pool = list(buckets["neutral_positive_pool"])
        if len(aligned_pos) + len(neutral_pool) < N_RECOMMENDED_POSTS:
            continue
        if not aligned_pos:
            # Need at least one in-window positive to serve as the held-out
            # ranking target — otherwise the moment signal can't be tested.
            continue

        # Held-out: aligned_pos[0] is already sorted by closeness to anchor.
        held_out = aligned_pos[0]
        held_id = held_out.get("source_object_id") or ""

        def _row_title(p: dict) -> str:
            c = p.get("content") or {}
            return (c.get("title") or c.get("caption") or "").strip().lower()

        held_title = _row_title(held_out)
        seen_titles: set[str] = {held_title} if held_title else set()

        # Hard negatives: in-window negatives first (most moment-relevant for
        # ranking), then any-time negatives. When explicit negatives are
        # sparse (many users have ~none), BACKFILL from neutral_pool — these
        # are persona-liked posts that are OFF-moment, so they are the natural
        # "hard negatives" for moment-ranking (the agent must rank the
        # moment-aligned held-out ABOVE them using the moment signal). This
        # both makes the ranking discriminative and ensures the GT renders a
        # "Hard negatives:" section. Title-de-dup so the held-out's title is
        # never duplicated in the slate (audit 2026-05-31).
        seen_neg_ids: set[str] = set()
        hard_negs: list[dict] = []
        for src in (buckets["aligned_negative"], buckets["all_negative_pool"],
                    neutral_pool):
            for p in src:
                oid = p.get("source_object_id") or ""
                t = _row_title(p)
                if not oid or oid in seen_neg_ids or oid == held_id:
                    continue
                if t and t in seen_titles:
                    continue
                seen_neg_ids.add(oid)
                if t:
                    seen_titles.add(t)
                hard_negs.append(p)
                if len(hard_negs) >= N_HARD_NEGATIVES:
                    break
            if len(hard_negs) >= N_HARD_NEGATIVES:
                break

        # Fillers: leftover neutral pool, then leftover aligned_pos beyond the
        # held-out — title-unique vs held-out + hard_negs + each other.
        slate_target = SLATE_SIZE - 1 - len(hard_negs)
        fillers: list[dict] = []
        seen_filler_ids: set[str] = {held_id} | seen_neg_ids
        for p in neutral_pool + aligned_pos[1:]:
            oid = p.get("source_object_id") or ""
            t = _row_title(p)
            if not oid or oid in seen_filler_ids:
                continue
            if t and t in seen_titles:
                continue
            seen_filler_ids.add(oid)
            if t:
                seen_titles.add(t)
            fillers.append(p)
            if len(fillers) >= slate_target:
                break
        if len(fillers) + 1 + len(hard_negs) < N_RECOMMENDED_POSTS:
            # Slate would be smaller than the floor — drop this moment
            # rather than emit a degenerate ranking instance.
            continue

        # Assemble + shuffle so the held-out doesn't always sit at idx=0.
        slate_rows = [held_out] + hard_negs + fillers
        order = list(range(len(slate_rows)))
        rng.shuffle(order)
        slate = [slate_rows[j] for j in order]
        held_out_idx = order.index(0)
        hard_negative_idxs = [order.index(j + 1) for j in range(len(hard_negs))]

        query_text = _voice_flavored_moment_query(moment, user_voice, rng,
                                                    discovery_llm=discovery_llm)

        out.append({
            "instance_id": f"recsys_moment_{i}",
            "task_id": "personalized_recommendation",
            "entry_point": "chatbot_routed",
            "t_test": t_anchor,
            "candidates": slate,
            "held_out_idx": held_out_idx,
            "hard_negative_idxs": hard_negative_idxs,
            "query_text": query_text,
            # Moment metadata kept for traceability / per-moment slicing of
            # results downstream. Not used by the prompt or metrics.
            "moment": moment,
            "moment_window_utc": list(MOMENT_HOUR_WINDOWS_UTC[moment]),
            "moment_day_filter": sorted(MOMENT_DAY_FILTER.get(moment, set())),
            # day_label / anchor_hour_utc are required by
            # personalized_recommendation_prompt's history-block header;
            # synthesize from t_anchor so the prompt renders cleanly.
            "day_label": _moment_day_label(t_anchor),
            "anchor_hour_utc": _moment_anchor_hour_utc(t_anchor),
        })
    return out


def _moment_day_label(t_anchor: int) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(int(t_anchor or 0), tz=_dt.timezone.utc).strftime("%Y-%m-%d")


def _moment_anchor_hour_utc(t_anchor: int) -> int:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(int(t_anchor or 0), tz=_dt.timezone.utc).hour


def build_t8_dm_digest(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """Two windows (24h, 7d) per social app. Persona-relevance filter:
    skip an (app, window) pair unless there are ≥3 threads in scope AND
    ≥1 thread previews a persona-relevant topic. With <3 threads any
    digest is trivially "list all"; with 0 persona-relevant threads
    personalization can't shape the priority order."""
    idx = _build_persona_topic_index(bq, user_id, t_anchor)
    out = []
    DAY = 24 * 3600
    windows = [
        ("24h", t_anchor - DAY),
        ("7d",  t_anchor - 7 * DAY),
    ]
    for app in SOCIAL_APPS:
        dms = bq.list_dm_threads(user_id=user_id, app=app, since_timestamp=t_anchor, limit=20)
        threads = dms.get("results") or []
        if len(threads) < 3:
            continue
        n_relevant = sum(
            1 for t in threads
            if _text_touches_persona(t, idx)
        )
        if n_relevant < 1:
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
                "persona_relevance": {
                    "n_threads": len(threads),
                    "n_persona_relevant_threads": n_relevant,
                },
            })
    return out


def build_t9_cross_app_repost(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """Pick a positive post the user actually engaged with on the source
    app and repost it to a target app. Persona-relevance filter: the
    source post's hashtags must intersect the user's top-20 hashtags so
    voice + hashtag adaptation is a meaningful signal. Generic posts the
    user happened to like are skipped.

    Caption-length filter: prefer source posts with substantive captions
    (≥ 80 chars) so the rewritten example/inferior have enough material
    to differ on voice. Fall back to the longest available caption when
    no candidate clears the bar — the user reported one-liner sources
    making example/inferior trivially short and indistinguishable.

    Pre-test floor: voice-mimic task — skip when the user has fewer than
    `USER_VOICED_SAMPLES_FLOOR` user-voiced samples before t_anchor.
    """
    if not ground_truth_builders.has_enough_user_voiced_history(bq, user_id, t_anchor):
        return []
    MIN_CAPTION_CHARS = 80
    idx = _build_persona_topic_index(bq, user_id, t_anchor)
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
        # Prefer the most recent positive post that overlaps the user's
        # top-20 hashtags; fall back to nothing if none qualify.
        candidates = [
            e for e in evs
            if "positive" in e.get("source_interaction_type", "")
            and (e.get("content") or {}).get("caption")
            and any(
                (h or "").lstrip("#").lower() in idx["top_hashtags"]
                for h in (e.get("source_hashtags") or [])
            )
        ]
        if not candidates:
            continue
        # Prefer substantive captions; fall back to the longest available.
        long_enough = [
            e for e in candidates
            if len((e.get("content") or {}).get("caption", "") or "") >= MIN_CAPTION_CHARS
        ]
        if long_enough:
            src = long_enough[-1]
        else:
            src = max(
                candidates,
                key=lambda e: len((e.get("content") or {}).get("caption", "") or ""),
            )
        source_post = {
            "caption": (src.get("content") or {}).get("caption", ""),
            "hashtags": src.get("source_hashtags", []),
            # source_object_id propagates so _build_agentic_tool_call can
            # populate `{src_app}_get_post.args.post_id` with a real id —
            # the previous empty stub left the tool call obviously fake.
            "source_object_id": src.get("source_object_id", ""),
            "source_timestamp": src.get("source_timestamp", 0),
        }
        overlap = sorted({
            (h or "").lstrip("#").lower()
            for h in (src.get("source_hashtags") or [])
            if (h or "").lstrip("#").lower() in idx["top_hashtags"]
        })
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
            "persona_relevance": {"hashtag_overlap": overlap},
        })
    return out


def build_t10_auto_reply(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """One per inbound DM where the user's persona materially shapes the
    right reply. Voice-mimic task — short-circuits when history lacks
    `USER_VOICED_SAMPLES_FLOOR` user-voiced samples before t_anchor.

    We look across more threads than we keep, then filter to instances
    passing at least one persona-relevance check:
      - inbound message overlaps user's top hashtags / categories
      - sender is a friend whose shared_interests overlap user's persona
      - inbound asks an opinion / recommendation question
    Generic logistics replies ("yeah saturday works") are dropped — both
    memory-using and memory-blind agents handle those equally."""
    if not ground_truth_builders.has_enough_user_voiced_history(bq, user_id, t_anchor):
        return []
    idx = _build_persona_topic_index(bq, user_id, t_anchor)
    out = []
    per_app_cap = 2
    for app in SOCIAL_APPS:
        dms_resp = bq.list_dm_threads(user_id=user_id, app=app, since_timestamp=t_anchor, limit=20)
        kept_in_app = 0
        for thread in dms_resp.get("results", []):
            if kept_in_app >= per_app_cap:
                break
            tid = thread.get("thread_id")
            thread_full = bq.get_dm_thread(user_id=user_id, app=app, thread_id=tid, since_timestamp=t_anchor, limit=10) or {}
            msgs = thread_full.get("results") or thread_full.get("messages") or []
            # Pick the most recent inbound message that actually has body
            # text. Real DM threads sometimes end with a presence/reaction
            # stub whose `text` is empty — those produce hollow `[incoming
            # DM from X]` queries with nothing for the agent to respond to.
            inbound = [m for m in msgs
                       if m.get("sender") != "self"
                       and (m.get("text") or "").strip()]
            if not inbound:
                continue
            last = inbound[-1]
            text = last["text"]
            sender_id = last.get("sender") or "unknown"
            friend = idx["friends_by_id"].get(sender_id)

            topic_hit = _text_touches_persona(text, idx)
            relationship_hit = _friend_shares_persona(friend, idx)
            question_hit = _is_opinion_or_recommendation_request(text)
            if not (topic_hit or relationship_hit or question_hit):
                continue  # generic logistics — persona doesn't shape the answer

            out.append({
                "instance_id": f"t10_{app}_{tid}", "task_id": "agentic_auto_reply", "entry_point": "app_native",
                "target_app": app, "thread_id": tid,
                "inbound_message": text,
                "sender_id": sender_id,
                "t_test": t_anchor,
                "tool_call_rules": [f"count('{app}_send_dm') == 1", f"count('{app}_create_post') == 0"],
                "final_state_expected": {"must_contain_count": {f"{app}_send_dm": 1}},
                "persona_relevance": {
                    "topic_match": topic_hit,
                    "relationship_match": relationship_hit,
                    "opinion_request": question_hit,
                },
            })
            kept_in_app += 1
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
    """Three example updates per social app. Voice-mimic task —
    short-circuits when history lacks ``USER_VOICED_SAMPLES_FLOOR``
    user-voiced samples before t_anchor."""
    if not ground_truth_builders.has_enough_user_voiced_history(bq, user_id, t_anchor):
        return []
    updates = [
        ("instagram", "finally wrapped up that project I've been grinding on"),
        ("facebook",  "great run this morning"),
        ("threads",   "saw something today that reminded me how weirdly competitive i get"),
    ]
    return [
        {"instance_id": f"t12_{app}_{i}", "task_id": "agentic_send_post", "entry_point": "app_native",
         "target_app": app, "update": u, "t_test": t_anchor,
         "tool_call_rules": [f"count('{app}_create_post') == 1", f"count('{app}_send_dm') == 0"],
         "final_state_expected": {"must_contain_count": {f"{app}_create_post": 1}}}
        for i, (app, u) in enumerate(updates)
    ]


def build_t13_send_post(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """Chatbot → target_app dispatch. 6 examples spread across apps.

    Each `context` is a multi-sentence first-person narration the user
    might dictate to a chatbot before asking it to post on their behalf.
    Long-enough so the agent has substantive material to compose with;
    short examples produced one-line posts where the example_response and
    inferior_response barely differed.

    Voice-mimic task — short-circuits when history lacks
    ``USER_VOICED_SAMPLES_FLOOR`` user-voiced samples before t_anchor.
    """
    if not ground_truth_builders.has_enough_user_voiced_history(bq, user_id, t_anchor):
        return []
    contexts = [
        ("threads",
         "I was just saying that discipline is what carries when motivation fades — "
         "that 5am-when-you-don't-feel-like-it streak is what actually moves the "
         "needle, not the highlight reels everyone's posting. Want a quick threads "
         "take on that, in my voice, no listicle vibes."),
        ("instagram",
         "Pulled off a clean session at the gym this morning — felt strong, hit the "
         "lift I've been chasing for weeks, then got a decent shot in the mirror "
         "after. Wanting to post the photo with a caption that's hype but not "
         "preachy, just a real moment."),
    ]
    return [
        {"instance_id": f"t13_{i}", "task_id": "agentic_send_post", "entry_point": "chatbot_routed",
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


def build_t16_group_dm_summary(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """One per group thread, filtered to threads where summarization is
    non-trivial: the thread either references a topic in the user's
    persona OR contains an open question / decision signal. Pure
    logistics threads ("what time again?") get the same summary
    regardless of memory and pollute the benchmark."""
    idx = _build_persona_topic_index(bq, user_id, t_anchor)
    out = []
    DECISION_SIGNALS = ("?", "decide", "decision", "settle", "vote",
                         "thoughts", "should we", "what about", "let me know")
    for app in SOCIAL_APPS:
        page = bq.list_dm_threads(user_id=user_id, app=app, since_timestamp=t_anchor, limit=20)
        for t in page.get("results", []):
            if not t.get("is_group"):
                continue
            tid = t["thread_id"]
            thread_full = bq.get_dm_thread(user_id=user_id, app=app, thread_id=tid, since_timestamp=t_anchor, limit=20) or {}
            msgs = thread_full.get("results") or thread_full.get("messages") or []
            if not msgs:
                continue
            joined = " ".join((m.get("text") or "") for m in msgs).lower()
            # Persona check: hashtag overlap from the carried forward
            # (via thread_full) OR keyword hit in the joined message text.
            has_persona = _text_touches_persona(thread_full, idx) or _text_touches_persona(joined, idx)
            has_decision = any(sig in joined for sig in DECISION_SIGNALS)
            if not (has_persona or has_decision):
                continue
            out.append({
                "instance_id": f"t16_{app}_{tid}", "task_id": "agentic_group_dm_summary",
                "entry_point": "chatbot_routed",
                "target_app": app, "thread_id": tid, "t_test": t_anchor,
                "tool_call_rules": [f"count('{app}_send_dm') == 0",  # don't send without approval
                                    f"count('{app}_get_dm_thread') >= 1"],
                "persona_relevance": {
                    "topic_match": has_persona,
                    "has_decision_signal": has_decision,
                },
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
    """Workstream E: thin wrapper around task_distribution.spread_anchors
    so existing agentic callers continue to work."""
    from evaluation.task_distribution import spread_anchors
    return spread_anchors(bq, user_id, t_anchor, n=n)


def build_t18_proactive_daily(bq: BackendQuery, user_id: str, t_anchor: int) -> list[dict]:
    """Daily-briefing probe — emit instances at evenly-spaced anchors,
    keeping only those where the past 24h of activity spans ≥2 of the
    user's top categories. With <2 categories present the priority order
    is trivial (one obvious item or none), and personalization adds
    nothing over a generic recency-sorted list."""
    idx = _build_persona_topic_index(bq, user_id, t_anchor)
    anchors = _spread_anchors(bq, user_id, t_anchor, n=5)
    DAY = 24 * 3600
    out: list[dict] = []
    for i, ts in enumerate(anchors):
        cats_seen: set[str] = set()
        for app in SOCIAL_APPS:
            for e in bq.get_events(user_id=user_id, app=app, since_timestamp=ts):
                ets = int(e.get("source_timestamp") or 0)
                if ets < ts - DAY or ets >= ts:
                    continue
                for pref in (e.get("preferences") or []):
                    c = (pref.get("category") or "").strip().lower()
                    if c in idx["top_cats"]:
                        cats_seen.add(c)
        if len(cats_seen) < 2:
            continue
        out.append({
            "instance_id": f"t18_daily_{i}",
            "task_id": "agentic_proactive_daily_catchup",
            "entry_point": "chatbot_routed",
            "t_test": ts,
            "tool_call_rules": ["count('instagram_create_post') == 0",
                                "count('instagram_send_dm') == 0"],
            "persona_relevance": {"top_cats_in_window": sorted(cats_seen)},
        })
    return out


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


def _build_send_post_merged(bq, user_id, t_anchor, **kwargs):
    """Merged builder: T12 (seed→post) + T13 (narration→post)."""
    out = build_t12_agent_composed_post(bq, user_id, t_anchor)
    out += build_t13_send_post(bq, user_id, t_anchor)
    return out


ALL_BUILDERS: dict[str, Callable] = {
    "agentic_community_post":           build_t6_user_tone_post,
    "agentic_send_post":                _build_send_post_merged,
    # agentic_moment_recommendation was merged into personalized_recommendation
    # (slate-based ranking) — build_t7_moment_recommendation is now called
    # directly from build_benchmark.py and its output appended to e4_instances.
    "agentic_dm_digest":                build_t8_dm_digest,
    "agentic_cross_app_repost":         build_t9_cross_app_repost,
    "agentic_auto_reply":               build_t10_auto_reply,
    "agentic_vague_refind":             build_t11_vague_refind,
    # agentic_draft_audit removed — workstream F.
    "agentic_group_dm_summary":         build_t16_group_dm_summary,
    # agentic_wrong_recipient_check retired (task_distribution.py:108). The
    # builder (build_t17_wrong_recipient) + runner code paths are kept, but it
    # must NOT be in ALL_BUILDERS or it leaks ~1 untargeted row per user into
    # test.json (caught in the 2026-05-30 validation regen of user 105).
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
        # Always seed the prompt with a per-task ground-truth slice so the
        # model has real user data to ground its response in (instead of
        # refusing with "I can't access your DMs"). In mcp_agent mode the
        # agent may still call MCP tools for additional reads if needed.
        # In llm_longctx mode the prompt switches to final-answer-only:
        # write tasks ask for the actual content as JSON instead of telling
        # the model to call non-existent MCP tools.
        gt_block = ground_truth_builders.build_for_task(task_id, bq, user_id, t, inst)
        history_block = None
        if mode == "llm_longctx":
            history_block, _ = snapshot_cache.get_or_build(bq, user_id, t, model_name, context_budget)
        allow_extra = (mode == "mcp_agent")
        text_only = (mode == "llm_longctx")
        prompt = prompt_fn(inst, history_block,
                            ground_truth_block=gt_block or None,
                            allow_extra_tools=allow_extra,
                            text_only=text_only)

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
    if task_id == "agentic_community_post":
        return (inst.get("user_query")
                or f"compose a post in the user's voice on {inst.get('target_app')}")
    if task_id == "agentic_send_post":
        return (inst.get("context")
                or inst.get("update")
                or f"compose a post in the user's voice on {inst.get('target_app')}")
    return {
        "agentic_dm_digest": f"dm digest on {inst.get('target_app')}",
        "agentic_cross_app_repost": inst.get("source_post", {}).get("caption", ""),
        "agentic_auto_reply": inst.get("inbound_message", ""),
        "agentic_vague_refind": f"find post about {inst.get('topic', '')}",
        "agentic_draft_audit": inst.get("draft", ""),
        "agentic_group_dm_summary": "group dm summary",
        "agentic_wrong_recipient_check": inst.get("draft", ""),
        "agentic_proactive_daily_catchup": "what should I catch up on today",
        "agentic_trending_alert": "anything trending I care about",
    }.get(task_id, "")


def _prompt_for(task_id: str):
    """Return a closure (inst, history_block, **kwargs) -> prompt for the given task.

    kwargs forwarded to each prompt template:
      - ground_truth_block: focused per-task slice from ground_truth_builders
      - allow_extra_tools:  True in mcp_agent mode (lets the directive note
        that supplementary mcp__* read calls are permitted)
    """
    pa = prompts_agentic

    def t6(inst, h, **kw): return pa.t6_user_tone_post(inst["target_app"], h, **kw)
    def t_send_post(inst, h, **kw):
        body = inst.get("context") or inst.get("update") or ""
        return pa.t12_agent_composed_post(inst["target_app"], body, h, **kw)
    def t8(inst, h, **kw): return pa.t8_dm_digest(inst["target_app"], h, **kw)
    def t9(inst, h, **kw): return pa.t9_cross_app_repost(inst["source_post"], inst["target_app"], h, source_app=inst.get("source_app"), **kw)
    def t10(inst, h, **kw): return pa.t10_auto_reply(inst["inbound_message"], inst["sender_id"], h, target_app=inst.get("target_app", "instagram"), **kw)
    def t11(inst, h, **kw): return pa.t11_vague_refind(inst["topic"], h, **kw)
    def t14(inst, h, **kw): return pa.t14_draft_audit(inst["draft"], inst["target_app"], h)
    def t16(inst, h, **kw): return pa.t16_group_dm_summary(inst["thread_id"], h, target_app=inst.get("target_app", "instagram"), **kw)
    def t17(inst, h, **kw): return pa.t17_wrong_recipient(inst["draft"], inst["recipient_name"], h, target_app=inst.get("target_app", "instagram"), **kw)
    def t18(inst, h, **kw): return pa.t18_proactive_daily(h, **kw)
    def t19(inst, h, **kw): return pa.t19_trending_alert(h, **kw)

    return {
        "agentic_community_post": t6, "agentic_send_post": t_send_post,
        "agentic_dm_digest": t8,
        "agentic_cross_app_repost": t9, "agentic_auto_reply": t10, "agentic_vague_refind": t11,
        "agentic_draft_audit": t14,
        "agentic_group_dm_summary": t16, "agentic_wrong_recipient_check": t17,
        "agentic_proactive_daily_catchup": t18, "agentic_trending_alert": t19,
    }.get(task_id)


def run_task(task_id: str, instances, **kwargs):
    """Single entry point for any of T6-T19."""
    prompt_fn = _prompt_for(task_id)
    if prompt_fn is None:
        raise ValueError(f"no prompt for task_id={task_id}")
    return _run_generic(task_id=task_id, instances=instances, prompt_fn=prompt_fn, **kwargs)

