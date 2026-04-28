"""
Generates a standalone HTML persona visualization for a user.

Reads backend/{user_id}/profile.json plus the four per-app JSON files
(instagram.json, facebook.json, threads.json, chatbot.json).

Supports both the new interaction-event format (nested preferences per
event) and the legacy flat format (one record per preference).

Design: minimalist, Apple/Anthropic-inspired aesthetic.
No external dependencies — pure HTML/CSS/JS.
"""

from __future__ import annotations

import csv
import os
import json
from datetime import datetime, timezone

from data_preparation import utils


APPS = ["Instagram", "Facebook", "Threads", "Chatbot"]


# ---------------------------------------------------------------------------
# Test-sample annotation (Phase A4)
# ---------------------------------------------------------------------------

# Per-task ground-truth extractor — given the parsed instance_json from
# benchmark/{uid}/queries.csv, return a {ground_truth, rubric_tags} dict.
# Default extractor returns the task_id only with empty GT.
# ---------------------------------------------------------------------------
# Per-task GROUND-TRUTH extractor — returns a rich dict the JS template
# renders as multiple sections on the test card. Keys (all optional):
#   ground_truth        : str   short headline blurb
#   candidates          : list[(idx, title, origin)]  for ranking tasks
#   held_out_pref       : str   the persona-item text the agent should align to
#   target_prefs        : list[str]  preferences the agent SHOULD surface
#   privacy_flagged     : list[str]  preferences the agent must NOT surface
#   tool_call_rules     : list[str]  agentic write/read constraints
#   final_state_expected: dict  {must_contain_count, must_not_contain}
#   warn_frame          : {must_mention, must_not_mention, polarity}
#   signal_evidence     : list  active_mistake_prevention cross-signal trace
#   rubric_tags         : list[str]
# ---------------------------------------------------------------------------

def _gt_default(inst: dict) -> dict:
    return {"ground_truth": "", "rubric_tags": []}


def _truncate(s, n=120):
    s = s if isinstance(s, str) else str(s or "")
    return s[: n - 1] + "…" if len(s) > n else s


def _gt_personalized_feed_ranking(inst: dict) -> dict:
    held = inst.get("held_out_idx")
    slate = inst.get("slate") or []
    origins = inst.get("origin_by_idx") or []
    title = ""
    if isinstance(held, int) and 0 <= held < len(slate):
        title = slate[held].get("title") or slate[held].get("caption") or ""
    cands = []
    for i, c in enumerate(slate):
        origin = origins[i] if i < len(origins) else "?"
        cands.append({
            "idx": i,
            "title": _truncate(c.get("title") or c.get("caption") or "", 90),
            "hashtags": c.get("hashtags") or [],
            "origin": origin,
            "is_held_out": (i == held),
        })
    return {
        "ground_truth": f"held-out (rank-1 target): {_truncate(title, 100)}",
        "candidates": cands,
        "rubric_tags": ["preference_alignment", "behavioral_hit", "ndcg_graded@5"],
    }


def _gt_chatbot_proactive(inst: dict) -> dict:
    held = inst.get("held_out_preference") or {}
    held_pi = held.get("persona_item") or ""
    gt_slice = inst.get("gt_slice") or {}
    target = [p.get("persona_item") for p in (gt_slice.get("target") or []) if p.get("persona_item")]
    avoid = [p.get("persona_item") for p in (gt_slice.get("avoid") or []) if p.get("persona_item")]
    top_k = [p.get("persona_item") for p in (inst.get("top_k_relevant_prefs") or [])[:5] if p.get("persona_item")]
    privacy = [p.get("persona_item") for p in (inst.get("privacy_flagged_prefs") or [])[:5] if p.get("persona_item")]
    return {
        "ground_truth": _truncate(held_pi or (target[0] if target else "(no held-out preference; rubric checks restraint + privacy)"), 200),
        "held_out_pref": held_pi,
        "target_prefs": target[:6],
        "privacy_flagged": privacy,
        "top_k_relevant": top_k,
        "rubric_tags": ["preference_alignment", "over_personalization", "privacy_leak", "carve_out_respect"],
    }


def _gt_chatbot_restraint(inst: dict) -> dict:
    """Restraint-arm test: agent must NOT volunteer personalization for a
    generic question. We only show the prefs the agent COULD pull in
    (top-k context) so a reader sees what the model is being tempted by;
    we do NOT label any preferences as 'target' (there is none — the
    correct behavior is silence on personal preferences)."""
    top_k = [p.get("persona_item") for p in (inst.get("top_k_relevant_prefs") or [])[:5] if p.get("persona_item")]
    return {
        "ground_truth": "Restraint test — generic question; agent must NOT volunteer personalization.",
        "tempted_by": top_k,
        "rubric_tags": ["restraint", "personalization_leak_rate (lower=better)", "privacy_leak", "carve_out_respect"],
    }


def _gt_at_ai_directive(inst: dict) -> dict:
    cands = inst.get("candidates") or []
    pos = set(inst.get("positive_indices") or [])
    carve = set(inst.get("carveout_indices") or [])
    cand_list = []
    for i, c in enumerate(cands):
        cand_list.append({
            "idx": i,
            "title": _truncate(c.get("title") or c.get("caption") or "", 90),
            "hashtags": c.get("hashtags") or [],
            "origin": "match" if i in pos else ("carve_out" if i in carve else "filler"),
            "is_held_out": (i in pos),
        })
    return {
        "ground_truth": f"directive: {inst.get('directive_action', '')} on hashtags {inst.get('directive_hashtags', [])}; positive_indices={sorted(pos)}; carveout_indices={sorted(carve)}",
        "candidates": cand_list,
        "rubric_tags": ["preference_alignment", "stale_preference_use", "directive_respect"],
    }


def _gt_active_mistake_prevention(inst: dict) -> dict:
    ef = inst.get("expected_warning_frame") or {}
    sigs = (inst.get("cross_signal_signals") or {}).get("signal_evidence") or []
    return {
        "ground_truth": f"[{inst.get('polarity', '?')}] {_truncate(inst.get('mistake_summary', ''), 200)}",
        "warn_frame": {
            "polarity": inst.get("polarity", ""),
            "must_mention": ef.get("must_mention") or [],
            "must_not_mention": ef.get("must_not_mention") or [],
        },
        "signal_evidence": [
            {
                "source": s.get("source", ""),
                "ts": s.get("ts", ""),
                "ref": s.get("ref", ""),
                "quote": _truncate(s.get("quote", ""), 140),
            } for s in sigs
        ][:6],
        "rubric_tags": ["mistake_prevention_recall", "false_alarm_emission", "cross_signal_attribution",
                        "actionable_specificity", "warning_respectfulness"],
    }


def _gt_irrelevant_query_restraint(inst: dict) -> dict:
    cands = inst.get("candidates") or []
    origins = inst.get("origin_by_idx") or []
    held_text = inst.get("held_out_persona_item") or ""
    irrels = inst.get("irrelevant_persona_items") or []
    cand_list = [{
        "idx": i,
        "title": _truncate(c.get("persona_item") or c.get("title") or str(c), 100),
        "origin": origins[i] if i < len(origins) else "?",
        "is_held_out": (origins[i] == "held_out") if i < len(origins) else False,
    } for i, c in enumerate(cands)]
    return {
        "ground_truth": f"On app={inst.get('app', '')}: held-out persona item = {_truncate(held_text, 120)}",
        "candidates": cand_list,
        "irrelevant_persona_items": [_truncate(s, 100) for s in irrels[:4]],
        "rubric_tags": ["irrelevant_rejection_precision", "irrelevant_rejection_recall", "privacy_leak", "over_personalization"],
    }


def _gt_preference_removal_regen(inst: dict) -> dict:
    held = inst.get("held_out_preference") or {}
    return {
        "ground_truth": f"removed preference to test if agent stops using it: {_truncate(held.get('persona_item', ''), 160)}",
        "held_out_pref": held.get("persona_item", ""),
        "top_k_relevant": [p.get("persona_item") for p in (inst.get("top_k_relevant_prefs") or [])[:5] if p.get("persona_item")],
        "rubric_tags": ["over_personalization", "removal_success", "regen_identical_fail (lower=better)"],
    }


def _gt_repetition_fatigue_pairs(inst: dict) -> dict:
    return {
        "ground_truth": f"pair_id={inst.get('pair_id', '')} on {inst.get('target_app', '')}; tests recency_sensitivity to category={inst.get('shift_category', '')}",
        "extra_meta": {
            "dominant_category_pre": inst.get("dominant_category_pre"),
            "shift_category": inst.get("shift_category"),
            "t_early": inst.get("t_early"),
            "t_late": inst.get("t_late"),
        },
        "rubric_tags": ["over_personalization", "response_divergence", "recency_sensitivity"],
    }


def _gt_repetition_fatigue_sequences(inst: dict) -> dict:
    queries = inst.get("queries") or []
    return {
        "ground_truth": f"sequence_id={inst.get('sequence_id', '')} — {len(queries)} successive queries; agent must reduce repetition over the sequence",
        "extra_meta": {"n_queries": len(queries)},
        "rubric_tags": ["over_personalization", "preference_repetition_rate (lower=better)", "wrong_preference_reuse"],
    }


def _gt_context_shift_scenarios(inst: dict) -> dict:
    return {
        "ground_truth": f"scenario={inst.get('name', inst.get('scenario_id', ''))} — {_truncate(inst.get('notes', ''), 160)}",
        "carve_out": _truncate(inst.get("carve_out", ""), 200),
        "forbidden_items": [_truncate(s, 100) for s in (inst.get("forbidden_items") or [])[:4]],
        "rubric_tags": ["restraint", "avoid_leak", "privacy_leak", "over_personalization", "relationship_aware"],
    }


def _gt_daily_personalized_briefing(inst: dict) -> dict:
    return {
        "ground_truth": f"day {inst.get('day_index', '?')}: {inst.get('day_label', '')}; agent must surface relevant preferences without staleness",
        "rubric_tags": ["preference_alignment", "temporal_boundedness", "stale_preference_use"],
    }


def _gt_personalized_search_ranking(inst: dict) -> dict:
    return {
        "ground_truth": f"day {inst.get('day_index', '?')}: {inst.get('day_label', '')}; recent prefs: {_truncate(json.dumps(inst.get('recent_pref_summary', '')), 200)}",
        "rubric_tags": ["preference_alignment", "over_personalization"],
    }


def _gt_short_vs_long_term_lifecycle(inst: dict) -> dict:
    return {
        "ground_truth": f"horizon={inst.get('horizon_type', '?')}; tests when short-term preferences should fade",
        "rubric_tags": ["preference_alignment", "stale_preference_use", "temporal_boundedness"],
    }


def _gt_agentic(inst: dict) -> dict:
    """Generic agentic GT — surfaces tool_call_rules + final_state_expected
    + the natural query-shaped fields per task variant."""
    bits: list[str] = []
    target = inst.get("target_app") or ""
    if target:
        bits.append(f"target_app={target}")
    for k in ("update", "context", "draft", "topic", "moment", "thread_id", "recipient_name", "inbound_message"):
        if inst.get(k):
            bits.append(f"{k}={_truncate(str(inst[k]), 100)}")
    if inst.get("source_post"):
        sp = inst["source_post"]
        bits.append(f"source_post.caption={_truncate(sp.get('caption', ''), 100)}")
    return {
        "ground_truth": " | ".join(bits) if bits else "(agentic task; see tool rules + final state)",
        "tool_call_rules": inst.get("tool_call_rules") or [],
        "final_state_expected": inst.get("final_state_expected") or {},
        "rubric_tags": ["tool_call_rules", "final_state_diff", "output_quality", "voice_match", "preference_alignment"],
    }


TEST_GT_EXTRACTORS = {
    "personalized_feed_ranking":           _gt_personalized_feed_ranking,
    "slate_ranking":                       _gt_personalized_feed_ranking,  # v1 alias
    "chatbot_proactive_personalization":   _gt_chatbot_proactive,
    "chatbot_response_proactive":          _gt_chatbot_proactive,           # v1 alias
    "chatbot_restraint_control":           _gt_chatbot_restraint,
    "chatbot_response_control":            _gt_chatbot_restraint,           # v1 alias
    "at_ai_directive_followup":            _gt_at_ai_directive,
    "e2_at_ai_followup":                   _gt_at_ai_directive,             # v1 alias
    "active_mistake_prevention":           _gt_active_mistake_prevention,
    "e6_active_mistake_prevention":        _gt_active_mistake_prevention,   # v1 alias
    "irrelevant_query_restraint":          _gt_irrelevant_query_restraint,
    "preference_removal_regen":            _gt_preference_removal_regen,
    "repetition_fatigue_pairs":            _gt_repetition_fatigue_pairs,
    "repetition_fatigue_sequences":        _gt_repetition_fatigue_sequences,
    "context_shift_scenarios":             _gt_context_shift_scenarios,
    "daily_personalized_briefing":         _gt_daily_personalized_briefing,
    "personalized_search_ranking":         _gt_personalized_search_ranking,
    "short_vs_long_term_lifecycle":        _gt_short_vs_long_term_lifecycle,
    # All agentic_* tasks share the generic agentic extractor
    "agentic_community_digest":            _gt_agentic,
    "agentic_moment_recommendation":       _gt_agentic,
    "agentic_dm_digest":                   _gt_agentic,
    "agentic_cross_app_repost":            _gt_agentic,
    "agentic_auto_reply":                  _gt_agentic,
    "agentic_vague_refind":                _gt_agentic,
    "agentic_composed_post":               _gt_agentic,
    "agentic_chatbot_dispatch":            _gt_agentic,
    "agentic_draft_audit":                 _gt_agentic,
    "agentic_collection_curation":         _gt_agentic,
    "agentic_group_dm_summary":            _gt_agentic,
    "agentic_wrong_recipient_check":       _gt_agentic,
    "agentic_proactive_daily_catchup":     _gt_agentic,
    "agentic_trending_alert":              _gt_agentic,
}


def _gt_agentic_default(inst: dict) -> dict:
    """Fallback for unknown agentic tasks — defers to the generic agentic extractor."""
    return _gt_agentic(inst)


# ---------------------------------------------------------------------------
# Per-task USER-QUERY extractor — what the test card SHOWS as the "user's
# message at this time and place." Some tasks carry a natural user message
# (chatbot, agentic_auto_reply, e6); for ranking-style tasks we synthesize a
# task-shaped intent ("what should I be shown next on Instagram?") so the
# card has something readable.
# ---------------------------------------------------------------------------

def _q_default(inst: dict) -> str:
    return inst.get("user_query") or inst.get("user_message") or inst.get("query_text") or ""


def _q_personalized_feed_ranking(inst: dict) -> str:
    app = inst.get("app") or "this app"
    return f"[ranking task] What should I be shown next on {app}?"


def _q_chatbot(inst: dict) -> str:
    return inst.get("user_query") or inst.get("user_message") or "[chatbot turn]"


def _q_at_ai_directive(inst: dict) -> str:
    msg = inst.get("directive_user_message") or ""
    action = inst.get("directive_action") or ""
    return f"@ai {action}: {msg}" if msg else f"@ai {action}"


def _q_active_mistake_prevention(inst: dict) -> str:
    return inst.get("user_query") or inst.get("triggering_user_query") or "[mistake-prevention probe]"


def _q_agentic_community_digest(inst: dict) -> str:
    return f"[agentic] post a community digest on {inst.get('target_app', '')}"


def _q_agentic_moment_recommendation(inst: dict) -> str:
    return f"[agentic] recommend something for {inst.get('moment', '')}"


def _q_agentic_dm_digest(inst: dict) -> str:
    return f"[agentic] summarize my recent DMs on {inst.get('target_app', '')}"


def _q_agentic_cross_app_repost(inst: dict) -> str:
    src = inst.get("source_post") or {}
    cap = (src.get("caption") or "")[:120]
    return f"[agentic] repost this to {inst.get('target_app', '')}: {cap}"


def _q_agentic_auto_reply(inst: dict) -> str:
    sender = inst.get("sender_id") or "friend"
    msg = inst.get("inbound_message") or ""
    return f"[incoming DM from {sender}] {msg}"


def _q_agentic_vague_refind(inst: dict) -> str:
    return f"find that post I saw about {inst.get('topic', '')}"


def _q_agentic_composed_post(inst: dict) -> str:
    return f"[agentic] post on {inst.get('target_app', '')}: {inst.get('update', '')}"


def _q_agentic_chatbot_dispatch(inst: dict) -> str:
    return inst.get("context") or "[agentic dispatch]"


def _q_agentic_draft_audit(inst: dict) -> str:
    draft = (inst.get("draft") or "")[:160]
    return f"[agentic] audit this draft for {inst.get('target_app', '')}: {draft}"


def _q_agentic_collection_curation(inst: dict) -> str:
    return f"[agentic] curate collections on {inst.get('target_app', '')}"


def _q_agentic_group_dm_summary(inst: dict) -> str:
    return f"[agentic] summarize the group thread on {inst.get('target_app', '')}"


def _q_agentic_wrong_recipient_check(inst: dict) -> str:
    return f"[agentic] DM to {inst.get('recipient_name', '?')}: {(inst.get('draft') or '')[:120]}"


def _q_agentic_proactive_daily_catchup(inst: dict) -> str:
    return "what should I catch up on today?"


def _q_agentic_trending_alert(inst: dict) -> str:
    return "anything trending I care about right now?"


def _q_daily_personalized_briefing(inst: dict) -> str:
    return "[daily briefing] give me a personalized morning brief"


def _q_personalized_search_ranking(inst: dict) -> str:
    return f"[search] {inst.get('query_text') or inst.get('user_query') or inst.get('query', '')}"


def _q_short_vs_long_term_lifecycle(inst: dict) -> str:
    return "[lifecycle ranking] short-term vs long-term preference test"


TEST_QUERY_EXTRACTORS = {
    "personalized_feed_ranking":           _q_personalized_feed_ranking,
    "slate_ranking":                       _q_personalized_feed_ranking,
    "chatbot_proactive_personalization":   _q_chatbot,
    "chatbot_response_proactive":          _q_chatbot,
    "chatbot_restraint_control":           _q_chatbot,
    "chatbot_response_control":            _q_chatbot,
    "at_ai_directive_followup":            _q_at_ai_directive,
    "e2_at_ai_followup":                   _q_at_ai_directive,
    "active_mistake_prevention":           _q_active_mistake_prevention,
    "e6_active_mistake_prevention":        _q_active_mistake_prevention,
    "agentic_community_digest":            _q_agentic_community_digest,
    "agentic_moment_recommendation":       _q_agentic_moment_recommendation,
    "agentic_dm_digest":                   _q_agentic_dm_digest,
    "agentic_cross_app_repost":            _q_agentic_cross_app_repost,
    "agentic_auto_reply":                  _q_agentic_auto_reply,
    "agentic_vague_refind":                _q_agentic_vague_refind,
    "agentic_composed_post":               _q_agentic_composed_post,
    "agentic_chatbot_dispatch":            _q_agentic_chatbot_dispatch,
    "agentic_draft_audit":                 _q_agentic_draft_audit,
    "agentic_collection_curation":         _q_agentic_collection_curation,
    "agentic_group_dm_summary":            _q_agentic_group_dm_summary,
    "agentic_wrong_recipient_check":       _q_agentic_wrong_recipient_check,
    "agentic_proactive_daily_catchup":     _q_agentic_proactive_daily_catchup,
    "agentic_trending_alert":              _q_agentic_trending_alert,
    "daily_personalized_briefing":         _q_daily_personalized_briefing,
    "personalized_search_ranking":         _q_personalized_search_ranking,
    "short_vs_long_term_lifecycle":        _q_short_vs_long_term_lifecycle,
}


def _load_test_samples(uid: str, benchmark_dir: str = "benchmark") -> list[dict]:
    """Walk benchmark/{uid}/queries.csv → list of test-sample dicts.

    Each test sample is rendered as a STANDALONE timeline card at its own
    timestamp (sorted alongside regular events + calendar mods), with a
    distinct background color. Geo location is computed JS-side by walking
    backwards through events to find the nearest preceding event_location.

    Per-sample fields:
      ts (int)         — the moment the user is notionally asking
      ts_iso (str)     — formatted timestamp
      task_type        — e.g. "personalized_feed_ranking"
      task_family      — e.g. "agentic"
      query_id         — e.g. "115:0042:e6_115_p1_warn"
      query_text       — what the user (or the agent's prompt) effectively says
      ground_truth     — short blurb describing the expected answer
      rubric_tags      — list[str] of which rubric dimensions apply
    """
    qcsv = os.path.join(benchmark_dir, str(uid), "queries.csv")
    out: list[dict] = []
    if not os.path.exists(qcsv):
        return out
    csv.field_size_limit(10_000_000)
    with open(qcsv, "r", encoding="utf-8") as f:
        first = f.readline()
        if not first.startswith("#"):
            f.seek(0)
        for r in csv.DictReader(f):
            try:
                inst = json.loads(r.get("instance_json") or "{}")
            except Exception:
                inst = {}
            task_type = r.get("task_type", "")
            task_family = r.get("task_family", "")
            gt_extractor = (
                TEST_GT_EXTRACTORS.get(task_type)
                or (_gt_agentic_default if task_family == "agentic" else _gt_default)
            )
            q_extractor = TEST_QUERY_EXTRACTORS.get(task_type, _q_default)
            try:
                gt = gt_extractor(inst)
            except Exception as exc:
                gt = {"ground_truth": f"(extractor crashed: {type(exc).__name__})", "rubric_tags": []}
            try:
                q_text = q_extractor(inst) or ""
            except Exception:
                q_text = ""
            try:
                ts_int = int(r.get("ts") or 0)
            except Exception:
                ts_int = 0
            sample = {
                "ts": ts_int,
                "ts_iso": r.get("ts_iso", ""),
                "task_type": task_type,
                "task_family": task_family,
                "query_id": r.get("query_id", ""),
                "query_text": q_text,
                "ground_truth": gt.get("ground_truth", ""),
                "rubric_tags": gt.get("rubric_tags") or (r.get("rubric_tags", "").split(";") if r.get("rubric_tags") else []),
            }
            # Pass through optional rich fields when present — JS template
            # renders each one as its own labeled section on the test card.
            for k in ("candidates", "held_out_pref", "target_prefs", "privacy_flagged",
                     "top_k_relevant", "tempted_by", "tool_call_rules", "final_state_expected",
                     "warn_frame", "signal_evidence", "irrelevant_persona_items",
                     "carve_out", "forbidden_items", "extra_meta"):
                if k in gt:
                    sample[k] = gt[k]
            out.append(sample)
    return out


def _load_app_events(user_dir: str) -> tuple[list[dict], list[dict]]:
    """Load per-app JSON files and return (events, flat_prefs).

    ``events`` is the interaction-event list (new format). Each event
    has event-level fields + a ``preferences`` list.

    ``flat_prefs`` is the flattened preference list (for backwards-compat
    counts and profile serialization).

    Both lists are sorted by source_timestamp ascending.
    """
    all_events: list[dict] = []
    flat_prefs: list[dict] = []

    for app in APPS:
        path = os.path.join(user_dir, app.lower() + ".json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            if "preferences" in entry:
                # New interaction-event format
                entry["_app"] = app
                all_events.append(entry)
                for pref in entry["preferences"]:
                    flat = dict(pref)
                    flat["assigned_app"] = app
                    flat["source_object_id"] = entry.get("source_object_id", "")
                    flat["source_timestamp"] = entry.get("source_timestamp", 0)
                    flat["formatted_timestamp"] = entry.get("formatted_timestamp", "")
                    flat["source_hashtags"] = entry.get("source_hashtags", [])
                    flat["source_interaction_type"] = entry.get("source_interaction_type", "")
                    flat["interaction_format"] = entry.get("interaction_format", {})
                    flat_prefs.append(flat)
            else:
                # Legacy flat format — wrap as single-pref event
                entry.setdefault("assigned_app", app)
                event = {
                    "source_object_id": entry.get("source_object_id", ""),
                    "source_timestamp": entry.get("source_timestamp", 0),
                    "formatted_timestamp": entry.get("formatted_timestamp", ""),
                    "source_hashtags": entry.get("source_hashtags", []),
                    "source_interaction_type": entry.get("source_interaction_type", ""),
                    "interaction_format": entry.get("interaction_format", {}),
                    "_app": app,
                    "preferences": [{
                        "persona_item": entry.get("persona_item", ""),
                        "category": entry.get("category", ""),
                        "confidence_score_init": entry.get("confidence_score_init", 0),
                        "confidence_cross_referenced": entry.get("confidence_cross_referenced", 0),
                        "stereotype_mark": entry.get("stereotype_mark", "neutral"),
                        "split": entry.get("split", ""),
                        "update_history": entry.get("update_history", []),
                        "over_personalization_irrelevant": entry.get("over_personalization_irrelevant", []),
                    }],
                    "conversation": entry.get("conversation"),
                    "conversation_type": entry.get("conversation_type"),
                    "ask_to_forget": entry.get("ask_to_forget", False),
                }
                all_events.append(event)
                flat_prefs.append(entry)

    all_events.sort(key=lambda e: (int(e.get("source_timestamp") or 0), e.get("source_object_id", "")))
    flat_prefs.sort(key=lambda r: (int(r.get("source_timestamp") or 0), r.get("persona_item", "")))
    return all_events, flat_prefs


# Keep legacy loader for any external callers
def _load_app_prefs(user_dir: str) -> list[dict]:
    """Load per-app JSONs and return a flat list of preferences (legacy compat)."""
    _, flat = _load_app_events(user_dir)
    return flat


def _load_profile(user_dir: str) -> dict | None:
    path = os.path.join(user_dir, "profile.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_persona_html(user_id: str, backend_dir: str = "backend") -> str:
    """Read backend/{user_id}/ JSON files and produce a self-contained HTML file."""
    user_dir = os.path.join(backend_dir, str(user_id))

    profile = _load_profile(user_dir)
    events, flat_prefs = _load_app_events(user_dir)

    # Load calendar modification stream (Step 11c output). Optional — older
    # backends may not have it.
    calendar_mods: list[dict] = []
    calendar_path = os.path.join(user_dir, "calendar.json")
    if os.path.exists(calendar_path):
        try:
            with open(calendar_path, "r", encoding="utf-8") as f:
                cal_doc = json.load(f)
            if isinstance(cal_doc, dict):
                mods = cal_doc.get("modifications", [])
                if isinstance(mods, list):
                    calendar_mods = mods
        except (ValueError, OSError):
            pass

    now_str = datetime.now(tz=timezone.utc).strftime("%B %d, %Y at %H:%M UTC")

    # Serialize events + calendar mods for JS
    events_json = json.dumps(events)
    profile_json = json.dumps(profile) if profile else "null"
    calendar_json = json.dumps(calendar_mods)

    # Test-sample annotation: load benchmark/{uid}/queries.csv (when present).
    # Each test sample becomes a standalone timeline card at its own ts +
    # nearest preceding event's geo location, with a distinct background color.
    test_samples = _load_test_samples(user_id)
    test_samples_json = json.dumps(test_samples)

    # Counts
    n_events = len(events)
    n_prefs = len(flat_prefs)
    n_unique = len(set(r.get("persona_item", "") for r in flat_prefs))
    n_stereo = sum(1 for r in flat_prefs if r.get("stereotype_mark") == "stereotypical")
    n_anti = sum(1 for r in flat_prefs if r.get("stereotype_mark") == "anti-stereotypical")
    # Pref-instance test counts (one per supporting event)
    # R8: no more test/train split in data-gen output. Count short-term
    # horizons instead — those are the actionable eval-facing signal.
    n_short_term_instances = sum(1 for r in flat_prefs if r.get("time_horizon") == "short_term")
    n_ad_events = sum(1 for e in events if e.get("is_ad"))
    short_term_canonicals = {r.get("persona_item", "") for r in flat_prefs if r.get("time_horizon") == "short_term"}
    short_term_canonicals.discard("")
    n_short_term_canonicals = len(short_term_canonicals)
    per_app_counts = {}
    for app in APPS:
        per_app_counts[app] = sum(1 for e in events if e.get("_app") == app)

    # Event counts split by source_interaction_type
    _TYPES = ("explicit_positive", "explicit_negative", "implicit_positive", "implicit_negative")
    event_type_counts = {t: 0 for t in _TYPES}
    for e in events:
        t = e.get("source_interaction_type", "")
        if t in event_type_counts:
            event_type_counts[t] += 1

    # Canonical-preference counts split by their dominant interaction type.
    # For each unique persona_item, classify by priority:
    #   explicit_negative > explicit_positive > implicit_positive > implicit_negative.
    # (In practice surviving negatives are all promoted to explicit_negative,
    # so the implicit_negative canonical count will usually be 0.)
    pref_types_by_canonical: dict[str, set[str]] = {}
    for r in flat_prefs:
        pi = r.get("persona_item", "")
        if not pi:
            continue
        pref_types_by_canonical.setdefault(pi, set()).add(r.get("source_interaction_type", ""))
    canonical_type_counts = {t: 0 for t in _TYPES}
    for types in pref_types_by_canonical.values():
        if "explicit_negative" in types:
            canonical_type_counts["explicit_negative"] += 1
        elif "explicit_positive" in types:
            canonical_type_counts["explicit_positive"] += 1
        elif "implicit_positive" in types:
            canonical_type_counts["implicit_positive"] += 1
        elif "implicit_negative" in types:
            canonical_type_counts["implicit_negative"] += 1

    # Time period: earliest → latest event's formatted timestamps.
    event_ts = [int(e.get("source_timestamp") or 0) for e in events if e.get("source_timestamp")]
    if event_ts:
        first_ts, last_ts = min(event_ts), max(event_ts)
        first_fmt = utils.unix_to_formatted(first_ts) if hasattr(utils, "unix_to_formatted") else datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime("%H:%M, %m/%d/%Y")
        last_fmt = utils.unix_to_formatted(last_ts) if hasattr(utils, "unix_to_formatted") else datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime("%H:%M, %m/%d/%Y")
        span_days = (last_ts - first_ts) / 86400.0
        time_period = f"{first_fmt} → {last_fmt} ({span_days:.1f} days)"
    else:
        time_period = "—"

    # Number of distinct preference categories
    n_categories = len({r.get("category", "") for r in flat_prefs if r.get("category")})

    # Unique geo locations across all events (ordered by frequency desc).
    location_counts: dict[tuple[str, str, str], int] = {}
    for e in events:
        loc = e.get("event_location") or {}
        city = (loc.get("city") or "").strip()
        if not city:
            continue
        key = (city, (loc.get("region") or "").strip(), (loc.get("country") or "").strip())
        location_counts[key] = location_counts.get(key, 0) + 1
    if location_counts:
        location_parts = []
        for (city, region, country), cnt in sorted(
            location_counts.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            label = city
            if region:
                label += f", {region}"
            if country and country not in ("USA", "US"):
                label += f", {country}"
            location_parts.append(f"<span>{label} ({cnt})</span>")
        locations_html = "".join(location_parts)
    else:
        locations_html = '<span>—</span>'

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Persona — User {user_id}</title>
<style>
  :root {{
    --bg: #F7F7F5;
    --bg-card: #FFFFFF;
    --text: #1D1D1F;
    --text-secondary: #86868B;
    --text-tertiary: #AEAEB2;
    --border: #E5E5EA;
    --radius: 10px;
    --shadow: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-hover: 0 2px 8px rgba(0,0,0,0.07);
    --font: "Inter", -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: var(--font); background: var(--bg); color: var(--text); line-height: 1.6; -webkit-font-smoothing: antialiased; }}
  .container {{ max-width: 860px; margin: 0 auto; padding: 56px 24px; }}

  .header {{ margin-bottom: 40px; }}
  .header h1 {{ font-size: 28px; font-weight: 600; letter-spacing: -0.4px; margin-bottom: 6px; color: var(--text); }}
  .header .meta {{ color: var(--text-secondary); font-size: 13px; display: flex; flex-wrap: wrap; gap: 6px 18px; }}

  .profile-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 22px; margin-bottom: 40px; box-shadow: var(--shadow); }}
  .profile-card h2 {{ font-size: 18px; font-weight: 600; margin-bottom: 10px; letter-spacing: -0.2px; }}
  .profile-card .bio {{ font-size: 14px; line-height: 1.65; margin-bottom: 14px; color: var(--text); }}
  .profile-card .details {{ font-size: 12px; color: var(--text-secondary); }}
  .profile-card .details span {{ margin-right: 14px; }}
  .profile-card .big-five {{ display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }}
  .profile-card .b5-item {{ font-size: 11px; padding: 3px 10px; border-radius: 20px; background: #F2F2F7; color: var(--text-secondary); }}
  .profile-card .mbti {{ margin-top: 10px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}

  .section {{ margin-bottom: 40px; }}
  .section-title {{ font-size: 16px; font-weight: 600; letter-spacing: -0.2px; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border); color: var(--text); }}

  .event-grid {{ display: flex; flex-direction: column; gap: 12px; }}
  .event-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 20px; box-shadow: var(--shadow); transition: box-shadow 0.15s ease; border-left: 3px solid var(--border); }}
  .event-card:hover {{ box-shadow: var(--shadow-hover); }}
  .event-card.app-Instagram {{ border-left-color: #C13584; }}
  .event-card.app-Facebook {{ border-left-color: #4A6FA5; }}
  .event-card.app-Threads {{ border-left-color: #636366; }}
  .event-card.app-Chatbot {{ border-left-color: #C8956C; }}
  .event-card.implicit-negative {{ background: #F0F0F0; border-left-color: #B0B0B0; opacity: 0.65; filter: grayscale(100%); }}
  .event-card.implicit-negative .event-meta {{ color: #999; }}
  .event-card.implicit-negative .event-header {{ border-bottom-color: #E0E0E0; }}
  .event-card.implicit-negative .hashtags {{ color: #888; }}
  .event-card.implicit-negative .badge {{ background: #E0E0E0 !important; color: #777 !important; }}
  .event-card.implicit-negative .pref-item {{ background: #E8E8E8; border-color: #D0D0D0; }}
  .event-card.implicit-negative .pref-item .item-text {{ color: #666; }}
  .event-card.implicit-negative .conf-inline {{ color: #999; }}

  .event-header {{ margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid #F2F2F7; }}
  .event-header .event-meta {{ font-size: 11px; color: var(--text-secondary); margin-bottom: 4px; }}
  .event-header .hashtags {{ font-size: 12px; color: var(--text); margin-top: 4px; line-height: 1.5; }}

  .pref-list {{ display: flex; flex-direction: column; gap: 8px; }}
  .content-block + .pref-list {{ margin-top: 14px; }}
  .pref-list-label {{ font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: #6B7280; margin-bottom: 2px; }}
  .pref-item {{ padding: 10px 14px; border-radius: 8px; background: #FAFAFA; border: 1px solid #F2F2F7; }}
  .pref-item .item-text {{ font-size: 13px; font-weight: 500; line-height: 1.45; color: var(--text); margin-bottom: 4px; }}
  .pref-item .pref-meta {{ font-size: 10px; color: var(--text-secondary); }}

  .update-history {{ margin-top: 6px; padding-left: 10px; border-left: 2px solid #E8E8ED; }}
  .update-entry {{ font-size: 10px; color: var(--text-secondary); margin-bottom: 2px; }}
  .update-entry .ut-type {{ font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; }}
  .ut-reinforced {{ color: #2D6A4F; }}
  .ut-deepened {{ color: #1D4ED8; }}
  .ut-branched {{ color: #7C3AED; }}
  .ut-shifted {{ color: #B45309; }}
  .ut-intensified {{ color: #047857; }}
  .ut-contradicted {{ color: #B04050; }}
  .stance-res {{ font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; padding: 1px 6px; border-radius: 3px; margin-left: 4px; }}
  .stance-res.stance-passed {{ background: #FEE2E2; color: #7F1D1D; }}
  .stance-res.stance-ambivalence {{ background: #FFEDD5; color: #9A3412; }}
  .stance-res.stance-suppressed {{ background: #F3F4F6; color: #6B7280; text-decoration: line-through; }}
  .ut-ambivalent {{ color: #9A3412; }}
  .ut-faded {{ color: var(--text-tertiary); }}
  .ut-expanded {{ color: #1D4ED8; }}

  .conf-inline {{ font-size: 10px; color: var(--text-tertiary); font-variant-numeric: tabular-nums; }}
  .conf-inline span {{ margin-right: 10px; }}

  .badge {{ display: inline-block; font-size: 10px; font-weight: 500; padding: 2px 8px; border-radius: 4px; margin-right: 3px; letter-spacing: 0.1px; }}
  .badge.category {{ background: #F2F2F7; color: #636366; }}
  .badge.similar {{ background: #F2F2F7; color: #48854A; }}
  .badge.contradictory {{ background: #F2F2F7; color: #B04050; }}
  .badge.none {{ display: none; }}
  .badge.stereotypical {{ background: #FFF8E1; color: #8B6914; }}
  .badge.anti-stereotypical {{ background: #EEF2FF; color: #4A5DA8; }}
  .badge.train {{ background: #F2F2F7; color: var(--text-secondary); }}
  .badge.distractor {{ background: #FEF2F2; color: #9B2C2C; }}
  /* Standalone test-sample card — distinct gold-amber background so it stands
     out from regular events without needing inline annotations. */
  .event-card.test-sample-card {{
    background: #FFFBEB !important;
    border-left: 3px solid #d4af37 !important;
  }}
  .event-card.test-sample-card .event-header .event-meta code {{
    background: #FFF; padding: 1px 6px; border-radius: 3px; font-size: 11px;
    color: #7B5C00;
  }}
  .test-sample-query {{
    font-size: 14px; line-height: 1.45; color: var(--text);
    padding: 12px 14px; background: #fff; border-radius: 5px; margin: 8px 0;
    border: 1px solid #FBE9A1;
  }}
  .test-sample-meta {{
    font-size: 11px; color: var(--text-secondary); padding: 4px 6px;
  }}
  /* Rich-info sections inside a test card */
  .ts-section {{
    margin: 8px 0 4px 0; padding: 6px 10px; background: rgba(255,255,255,0.7);
    border-radius: 4px; border: 1px solid rgba(212,175,55,0.25);
  }}
  .ts-section-warn {{ background: #FEF2F2; border-color: #FCA5A5; }}
  .ts-section.ts-rubric-bar {{ background: #FFF8E1; }}
  .ts-label {{ font-weight: 600; font-size: 11px; color: #7B5C00; text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 4px; }}
  .ts-section-warn .ts-label {{ color: #B91C1C; }}
  .ts-sublabel {{ font-size: 10px; font-weight: 500; color: var(--text-secondary); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.3px; }}
  .ts-body {{ font-size: 12px; color: var(--text); line-height: 1.45; }}
  .ts-body.ts-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; white-space: pre-wrap; }}
  .ts-list {{ margin: 4px 0 0 0; padding-left: 18px; font-size: 12px; line-height: 1.5; }}
  .ts-list.ts-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }}
  .ts-list li {{ margin: 3px 0; }}
  .ts-origin {{ display: inline-block; font-size: 9px; padding: 1px 5px; border-radius: 3px; background: #E5E7EB; color: #374151; margin: 0 2px; text-transform: uppercase; letter-spacing: 0.3px; }}
  .ts-origin-held_out {{ background: #D4AF37; color: #fff; }}
  .ts-origin-future_positive {{ background: #BFDBFE; color: #1E40AF; }}
  .ts-origin-past_positive {{ background: #BBF7D0; color: #166534; }}
  .ts-origin-negative {{ background: #FECACA; color: #7F1D1D; }}
  .ts-origin-irrelevant, .ts-origin-random, .ts-origin-filler, .ts-origin-filler_lowsim {{ background: #F3F4F6; color: #6B7280; }}
  .ts-origin-match {{ background: #D4AF37; color: #fff; }}
  .ts-origin-carve_out {{ background: #FECACA; color: #7F1D1D; }}
  .ts-target {{ font-size: 10px; color: #B45309; font-weight: 700; }}
  .badge.platform {{ font-weight: 600; font-size: 11px; padding: 2px 10px; }}
  .badge.platform.p-Instagram {{ background: #C13584; color: #fff; }}
  .badge.platform.p-Facebook {{ background: #4A6FA5; color: #fff; }}
  .badge.platform.p-Threads {{ background: #8E8E93; color: #fff; }}
  .badge.platform.p-Chatbot {{ background: #C8956C; color: #fff; }}
  .badge.action {{ background: #E8E8ED; color: #48484A; font-weight: 500; }}
  .badge.hidden-persona {{ background: #EDE9FE; color: #6D28D9; font-weight: 500; }}
  .badge.short-term {{ background: #EFE1FF; color: #7C3AED; font-weight: 600; }}
  .stop-condition {{ margin-top: 4px; font-size: 10px; color: #7C3AED; opacity: 0.85; font-style: italic; }}
  .stop-condition .sc-type {{ text-transform: uppercase; font-weight: 700; letter-spacing: 0.4px; margin-right: 6px; }}
  .badge.sponsored {{ background: #FFF7ED; color: #9A3412; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; border: 1px solid #FED7AA; }}
  .event-location {{ font-size: 11px; color: var(--text-tertiary); }}
  .calendar-card {{ background: #F0FDF4; border: 1px solid #BBF7D0; border-left: 4px solid #16A34A; border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; font-size: 12px; color: #14532D; box-shadow: 0 1px 2px rgba(22,163,74,0.08); }}
  .calendar-card .cal-head {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-weight: 600; }}
  .calendar-card .cal-action {{ font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; padding: 2px 8px; border-radius: 4px; color: #fff; background: #16A34A; font-weight: 700; }}
  .calendar-card .cal-action.removed {{ background: #DC2626; }}
  .calendar-card .cal-action.updated {{ background: #CA8A04; }}
  .calendar-card .cal-meta {{ font-size: 11px; opacity: 0.75; margin-top: 2px; }}
  .ad-meta {{ margin-top: 6px; padding: 6px 10px; background: #FFF7ED; border: 1px solid #FED7AA; border-radius: 6px; font-size: 11px; color: #7C2D12; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
  .ad-meta .ad-sponsor {{ font-weight: 600; }}
  .ad-meta .ad-cta {{ background: #9A3412; color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; }}
  .event-card.is-ad .content-block {{ border-color: #FED7AA; background: #FFFBF5; }}

  .hidden-section {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 22px; margin-bottom: 40px; box-shadow: var(--shadow); }}
  .hidden-section h2 {{ font-size: 18px; font-weight: 600; margin-bottom: 14px; letter-spacing: -0.2px; }}
  .hidden-summary {{ font-size: 13px; line-height: 1.7; color: var(--text); margin-bottom: 16px; padding: 12px 16px; background: #FAFAFA; border-radius: 8px; border-left: 3px solid #6D28D9; }}
  .hp-card {{ padding: 12px 16px; margin-bottom: 10px; border-radius: 8px; background: #FAFAFA; border: 1px solid #F2F2F7; }}
  .hp-card .hp-label {{ font-size: 14px; font-weight: 600; color: var(--text); margin-bottom: 2px; }}
  .hp-card .hp-type {{ font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; color: #6D28D9; margin-bottom: 4px; }}
  .hp-card .hp-desc {{ font-size: 12px; color: var(--text); line-height: 1.6; margin-bottom: 6px; }}
  .hp-card .hp-meta {{ font-size: 10px; color: var(--text-secondary); }}
  .hp-card .hp-meta span {{ margin-right: 12px; }}
  .hp-card .hp-tags {{ font-size: 11px; color: var(--text-secondary); margin-top: 4px; }}
  .hp-card .hp-motivation {{ font-size: 11px; color: #6D28D9; margin-top: 4px; font-style: italic; }}
  .badge.interaction-type {{ font-weight: 600; padding: 2px 10px; }}
  .badge.interaction-type.explicit_positive {{ background: #D1FAE5; color: #065F46; }}
  .badge.interaction-type.implicit_positive {{ background: #EDF5E1; color: #3F6212; }}
  .badge.interaction-type.explicit_negative {{ background: #FEE2E2; color: #991B1B; }}
  .badge.interaction-type.implicit_negative {{ background: #FEF3C7; color: #92400E; }}

  .user-message {{ margin-top: 8px; padding: 8px 12px; background: #F2F2F7; border-left: 2px solid var(--text-tertiary); border-radius: 4px; font-size: 12px; color: var(--text); font-style: italic; }}

  /* Synthetic content (step 13b) rendering */
  .content-block {{ margin-top: 10px; padding: 12px 14px; background: #FAFAFC; border: 1px solid #ECECF1; border-radius: 8px; }}
  .content-block .c-type {{ font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: #6B7280; margin-bottom: 6px; }}
  .content-block .c-type.c-text {{ color: #4A5DA8; }}
  .content-block .c-type.c-image {{ color: #9B3068; }}
  .content-block .c-type.c-short_video {{ color: #7C3AED; }}
  .content-block .c-title {{ font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 4px; }}
  .content-block .c-caption {{ font-size: 12px; color: var(--text); margin-bottom: 6px; line-height: 1.5; }}
  .content-block .c-desc {{ font-size: 12px; color: var(--text-secondary); line-height: 1.55; margin-bottom: 8px; font-style: italic; }}
  .content-block .c-text-body {{ font-size: 13px; color: var(--text); line-height: 1.65; white-space: pre-wrap; }}
  .content-block details {{ margin-top: 6px; }}
  .content-block details summary {{ font-size: 11px; color: var(--text-secondary); cursor: pointer; padding: 2px 0; user-select: none; }}
  .content-block details summary:hover {{ color: var(--text); }}
  .content-block details[open] summary {{ color: var(--text); }}
  .content-block .c-parts, .content-block .c-frames {{ margin-top: 6px; padding-left: 2px; }}
  .content-block .c-part, .content-block .c-frame {{ font-size: 11px; color: var(--text); padding: 4px 8px; margin-bottom: 2px; background: #F2F2F7; border-radius: 4px; line-height: 1.45; }}
  .content-block .c-part .region, .content-block .c-frame .ts {{ font-weight: 600; color: #636366; margin-right: 6px; font-variant-numeric: tabular-nums; }}
  .content-block .c-transcript {{ font-size: 11px; color: var(--text); padding: 6px 10px; margin-top: 6px; background: #F2F2F7; border-radius: 4px; line-height: 1.5; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; }}
  .content-block .c-meta {{ margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px; }}
  .content-block .c-meta-chip {{ font-size: 10px; padding: 2px 8px; border-radius: 4px; background: #ECECF1; color: #636366; font-variant-numeric: tabular-nums; }}
  .event-card.implicit-negative .content-block {{ background: #ECECEC; border-color: #D8D8D8; }}

  .chat-thread {{ margin-top: 8px; display: flex; flex-direction: column; gap: 6px; }}
  .chat-bubble {{ max-width: 85%; padding: 10px 14px; border-radius: 14px; font-size: 12px; line-height: 1.6; word-wrap: break-word; }}
  .chat-bubble.user-bubble {{ align-self: flex-end; background: #1B72E8; color: #fff; border-bottom-right-radius: 4px; }}
  .chat-bubble.assistant-bubble {{ align-self: flex-start; background: #E4E6EB; color: var(--text); border-bottom-left-radius: 4px; }}
  .chat-role {{ font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }}
  .chat-bubble.user-bubble .chat-role {{ color: rgba(255,255,255,0.55); }}
  .chat-bubble.assistant-bubble .chat-role {{ color: var(--text-tertiary); }}
  .chat-conv-label {{ font-size: 10px; color: var(--text-tertiary); margin-top: 6px; margin-bottom: 2px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.3px; }}

  .empty {{ text-align: center; padding: 40px; color: var(--text-secondary); font-size: 13px; }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>User {user_id}</h1>
    <div class="meta">
      <span>{n_events} events</span>
      <span>{n_prefs} pref instances ({n_short_term_instances} short-term)</span>
      <span>{n_unique} canonicals ({n_short_term_canonicals} short-term)</span>
      <span>{n_ad_events} ad events</span>
      <span>{n_categories} categories</span>
      <span>{n_stereo} stereo</span>
      <span>{n_anti} anti-stereo</span>
    </div>
    <div class="meta" style="margin-top: 4px;">
      <span title="Events by source_interaction_type">Events:</span>
      <span>expl+ {event_type_counts["explicit_positive"]}</span>
      <span>expl− {event_type_counts["explicit_negative"]}</span>
      <span>impl+ {event_type_counts["implicit_positive"]}</span>
      <span>impl− {event_type_counts["implicit_negative"]}</span>
    </div>
    <div class="meta" style="margin-top: 4px;">
      <span title="Canonical preferences by dominant supporting interaction type">Canonicals:</span>
      <span>expl+ {canonical_type_counts["explicit_positive"]}</span>
      <span>expl− {canonical_type_counts["explicit_negative"]}</span>
      <span>impl+ {canonical_type_counts["implicit_positive"]}</span>
      <span>impl− {canonical_type_counts["implicit_negative"]}</span>
    </div>
    <div class="meta" style="margin-top: 4px;">
      <span>Period: {time_period}</span>
      <span>IG: {per_app_counts.get("Instagram", 0)}</span>
      <span>FB: {per_app_counts.get("Facebook", 0)}</span>
      <span>TH: {per_app_counts.get("Threads", 0)}</span>
      <span>AI: {per_app_counts.get("Chatbot", 0)}</span>
      <span>Generated {now_str}</span>
    </div>
    <div class="meta" style="margin-top: 4px;">
      <span title="Geo locations across all events">Locations:</span>
      {locations_html}
    </div>
  </div>

  <div id="profile-section"></div>
  <div id="hidden-personas-section"></div>

  <div class="section">
    <div class="section-title">Interaction Events (earliest &rarr; latest)</div>
    <div id="timeline-section"></div>
  </div>

</div>

<script>
const eventsData = {events_json};
const profileData = {profile_json};
const calendarMods = {calendar_json};
// Test-sample annotation (Phase A4 v2): each test sample becomes a standalone
// timeline card at its own ts + nearest-preceding event's location, with a
// distinct background color. No annotations are merged into regular event cards.
const testSamples = {test_samples_json};

// Label -> motivation lookup for hidden persona badge tooltips.
const hpMotivation = {{}};
if (profileData && profileData.hidden_personas) {{
  profileData.hidden_personas.forEach(hp => {{
    if (hp && hp.label) {{
      hpMotivation[hp.label] = hp.inferred_motivation || hp.description || '';
    }}
  }});
}}

// -- Profile card --
const ps = document.getElementById('profile-section');
if (profileData) {{
  const b5 = profileData.big_five || {{}};
  const b5Html = Object.entries(b5).map(([k,v]) => `<span class="b5-item">${{k}}: ${{v}}</span>`).join('');

  // MBTI block — reuse the Big Five chip style. One chip per dimension,
  // formatted as "axis: dominant letter". The MBTI type itself is just the
  // concatenation of these four letters, so we don't repeat it as a chip.
  let mbtiHtml = '';
  const mbti = profileData.mbti;
  if (mbti && mbti.dimensions) {{
    const dimOrder = ['E_I', 'S_N', 'T_F', 'J_P'];
    const dimChips = dimOrder.map(key => {{
      const d = mbti.dimensions[key];
      if (!d) return '';
      const [letterA, letterB] = key.split('_');
      const pA = Number(d[letterA] || 0);
      const pB = Number(d[letterB] || 0);
      const dominant = pA >= pB ? letterA : letterB;
      const pct = Math.round((pA >= pB ? pA : pB) * 100);
      const axis = `${{letterA}}/${{letterB}}`;
      const reason = (d.reason || '').replace(/"/g, '&quot;');
      return `<span class="b5-item" title="${{reason}}">${{axis}}: ${{dominant}} ${{pct}}%</span>`;
    }}).join('');
    if (dimChips) {{
      mbtiHtml = `<div class="mbti">${{dimChips}}</div>`;
    }}
  }}

  ps.innerHTML = `
    <div class="profile-card">
      <h2>${{profileData.name || ''}}</h2>
      <div class="bio">${{profileData.bio || ''}}</div>
      <div class="details">
        <span>${{profileData.gender || ''}}</span>
        <span>${{profileData.race_ethnicity || ''}}</span>
        <span>${{profileData.career || ''}}</span>
        <span>${{profileData.education || ''}}</span>
      </div>
      <div class="big-five">${{b5Html}}</div>
      ${{mbtiHtml}}
    </div>
  `;
}}

// -- Hidden Personas section --
const hps = document.getElementById('hidden-personas-section');
if (profileData && profileData.hidden_personas && profileData.hidden_personas.length > 0) {{
  let html = '<div class="hidden-section"><h2>Hidden Personas</h2>';

  // Summary paragraph
  if (profileData.hidden_persona_summary) {{
    html += `<div class="hidden-summary">${{profileData.hidden_persona_summary}}</div>`;
  }}

  // Individual hidden persona cards
  profileData.hidden_personas.forEach(hp => {{
    const tags = (hp.evidence_hashtags || []).join('  ');
    const ib = hp.interaction_breakdown || {{}};
    const ibStr = Object.entries(ib).map(([k,v]) => `${{k.replace(/_/g,' ')}}: ${{v}}`).join(' · ');
    const appDist = hp.app_distribution || {{}};
    const appStr = Object.entries(appDist).map(([k,v]) => `${{k}}: ${{v}}`).join(' · ');
    html += `
      <div class="hp-card">
        <div class="hp-type">${{hp.type || ''}}</div>
        <div class="hp-label">${{hp.label || ''}}</div>
        <div class="hp-desc">${{hp.description || ''}}</div>
        <div class="hp-meta">
          <span>${{hp.evidence_rows || 0}} rows (${{((hp.evidence_row_fraction || 0) * 100).toFixed(1)}}%)</span>
          <span>privacy: ${{((hp.privacy_ratio || 0) * 100).toFixed(0)}}%</span>
          <span>${{hp.temporal_spread_days || 0}} days</span>
          ${{appStr ? `<span>${{appStr}}</span>` : ''}}
        </div>
        <div class="hp-tags">${{tags}}</div>
        ${{hp.inferred_motivation ? `<div class="hp-motivation">"${{hp.inferred_motivation}}"</div>` : ''}}
      </div>
    `;
  }});

  html += '</div>';
  hps.innerHTML = html;
}}

// -- Render synthetic content (step 13b) --
// Produces an HTML block describing what the user saw on screen: the text,
// image, or short video. Returns empty string when the event has no content
// (Chatbot events and implicit_negative stubs).
function escapeHtml(s) {{
  if (s === null || s === undefined) return '';
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}}

function renderContentMeta(meta) {{
  if (!meta || typeof meta !== 'object') return '';
  const chips = Object.entries(meta)
    .filter(([k, v]) => v !== null && v !== undefined && v !== '')
    .map(([k, v]) => {{
      const val = typeof v === 'object' ? JSON.stringify(v) : String(v);
      return `<span class="c-meta-chip"><b>${{escapeHtml(k)}}</b> ${{escapeHtml(val)}}</span>`;
    }})
    .join('');
  return chips ? `<div class="c-meta">${{chips}}</div>` : '';
}}

function renderAdMeta(content) {{
  if (!content || typeof content !== 'object') return '';
  const md = content.ad_metadata;
  if (!md || typeof md !== 'object') return '';
  const sponsor = md.sponsor_name ? `<span class="ad-sponsor">${{escapeHtml(md.sponsor_name)}}</span>` : '';
  const cat = md.ad_category ? `<span>${{escapeHtml(md.ad_category.replace(/_/g, ' '))}}</span>` : '';
  const cta = md.cta_label ? `<span class="ad-cta">${{escapeHtml(md.cta_label)}}</span>` : '';
  const dest = md.cta_destination_kind ? `<span style="opacity:0.7">${{escapeHtml(md.cta_destination_kind.replace(/_/g, ' '))}}</span>` : '';
  return `<div class="ad-meta">${{sponsor}}${{cat}}${{cta}}${{dest}}</div>`;
}}

function renderContent(ev) {{
  const ctype = ev.content_type;
  const content = ev.content;
  if (!ctype || !content || typeof content !== 'object') return '';

  const typeLabel = ctype.replace(/_/g, ' ');
  const header = `<div class="c-type c-${{ctype}}">${{typeLabel}}</div>`;
  const adMetaHtml = renderAdMeta(content);

  if (ctype === 'text') {{
    return `<div class="content-block">${{header}}<div class="c-text-body">${{escapeHtml(content.text || '')}}</div>${{adMetaHtml}}</div>`;
  }}

  if (ctype === 'image') {{
    const caption = content.caption ? `<div class="c-caption">${{escapeHtml(content.caption)}}</div>` : '';
    const desc = content.overall_description ? `<div class="c-desc">${{escapeHtml(content.overall_description)}}</div>` : '';
    const parts = (content.parts || []).map(p =>
      `<div class="c-part"><span class="region">${{escapeHtml(p.region || '')}}</span>${{escapeHtml(p.description || '')}}</div>`
    ).join('');
    const partsBlock = parts
      ? `<details><summary>parts (${{content.parts.length}})</summary><div class="c-parts">${{parts}}</div></details>`
      : '';
    const metaBlock = renderContentMeta(content.metadata);
    const metaWrapped = metaBlock
      ? `<details><summary>metadata</summary>${{metaBlock}}</details>`
      : '';
    return `<div class="content-block">${{header}}${{caption}}${{desc}}${{partsBlock}}${{metaWrapped}}${{adMetaHtml}}</div>`;
  }}

  if (ctype === 'short_video') {{
    const title = content.title ? `<div class="c-title">${{escapeHtml(content.title)}}</div>` : '';
    const caption = content.caption ? `<div class="c-caption">${{escapeHtml(content.caption)}}</div>` : '';
    const desc = content.overall_description ? `<div class="c-desc">${{escapeHtml(content.overall_description)}}</div>` : '';
    const frames = (content.key_frames || []).map(f => {{
      const ts = typeof f.timestamp_s === 'number' ? f.timestamp_s.toFixed(1) + 's' : String(f.timestamp_s || '');
      return `<div class="c-frame"><span class="ts">${{escapeHtml(ts)}}</span>${{escapeHtml(f.description || '')}}</div>`;
    }}).join('');
    const framesBlock = frames
      ? `<details open><summary>key frames (${{content.key_frames.length}})</summary><div class="c-frames">${{frames}}</div></details>`
      : '';
    const transcript = content.audio_transcript
      ? `<details><summary>audio transcript</summary><div class="c-transcript">${{escapeHtml(content.audio_transcript)}}</div></details>`
      : '';
    const metaBlock = renderContentMeta(content.metadata);
    const metaWrapped = metaBlock
      ? `<details><summary>metadata</summary>${{metaBlock}}</details>`
      : '';
    return `<div class="content-block">${{header}}${{title}}${{caption}}${{desc}}${{framesBlock}}${{transcript}}${{metaWrapped}}${{adMetaHtml}}</div>`;
  }}

  return '';
}}

// -- Render update history --
function renderUpdateHistory(history, asOfTs) {{
  if (!history || !history.length) return '';
  // Filter to entries with timestamp <= asOfTs (the event being rendered).
  // The pref's full update_history reflects GLOBAL cross-ref resolutions
  // computed across the whole persona; at any given event we should only
  // surface history that existed up to that moment in time.
  let visible = history;
  if (typeof asOfTs === 'number' && asOfTs > 0) {{
    visible = history.filter(h => {{
      const ht = h.timestamp || h.ts || 0;
      return !ht || ht <= asOfTs;
    }});
  }}
  if (!visible.length) return '';
  const entries = visible.map(h => {{
    const cls = 'ut-' + (h.update_type || 'expanded');
    let text = `<span class="ut-type ${{cls}}">${{h.update_type}}</span>`;
    if (h.preference) text += ` ${{h.preference}}`;
    if (h.description) text += ` — ${{h.description}}`;
    if (h.formatted_timestamp) text += ` <span style="opacity:0.6">(${{h.formatted_timestamp}})</span>`;
    if (h.source_app) text += ` <span class="badge platform p-${{h.source_app}}" style="font-size:9px;padding:1px 6px;">${{h.source_app}}</span>`;
    if (h.total_occurrences) text += ` <span style="opacity:0.6">[occ ${{h.occurrence}}/${{h.total_occurrences}}]</span>`;
    if (h.resolution) {{
      const resCls = h.resolution === 'stance_shift_with_precedent' ? 'stance-passed'
                   : h.resolution === 'concurrent_ambivalence'     ? 'stance-ambivalence'
                   : h.resolution === 'different_granularity'      ? 'stance-ambivalence'
                   : 'stance-suppressed';
      text += ` <span class="stance-res ${{resCls}}">${{h.resolution.replace(/_/g, ' ')}}</span>`;
      if (typeof h.prior_corroboration_count === 'number') {{
        text += ` <span style="opacity:0.6">prior ${{h.prior_corroboration_count}}/${{h.required_precedent}}</span>`;
      }}
    }}
    return `<div class="update-entry">${{text}}</div>`;
  }}).join('');
  return `<div class="update-history">${{entries}}</div>`;
}}

// -- Calendar modification card renderer --
function renderCalendarMod(mod) {{
  const action = mod.action || '';
  const actionCls = `cal-action ${{action}}`;
  const actionLabel = action.toUpperCase();
  let title = '';
  let locChip = '';
  let extraLine = '';
  if (action === 'added' && mod.entry) {{
    const e = mod.entry;
    title = escapeHtml(e.title || '(untitled)');
    if (e.location && e.location.city) {{
      locChip = ` 📍 ${{escapeHtml(e.location.city)}}`;
    }}
    const start = e.start_ts ? new Date(e.start_ts * 1000).toISOString().slice(0, 16).replace('T', ' ') : '';
    const typeLbl = e.type ? `<span style="opacity:0.7">[${{escapeHtml(e.type)}}]</span>` : '';
    extraLine = `<div class="cal-meta">scheduled for ${{start}} ${{typeLbl}} ${{e.is_preference_driven ? '· preference-linked' : '· unrelated'}}</div>`;
  }} else if (action === 'updated') {{
    title = `update to ${{escapeHtml(mod.entry_id || '?')}}`;
    const diff = mod.diff || {{}};
    const fields = Object.keys(diff).join(', ');
    extraLine = `<div class="cal-meta">changed: ${{escapeHtml(fields)}}</div>`;
  }} else if (action === 'removed') {{
    title = `removed ${{escapeHtml(mod.entry_id || '?')}}`;
    if (mod.removal_reason) {{
      extraLine = `<div class="cal-meta">${{escapeHtml(mod.removal_reason)}}</div>`;
    }}
  }}
  const ts = mod.formatted_timestamp ? escapeHtml(mod.formatted_timestamp) : '';
  return `
    <div class="calendar-card">
      <div class="cal-head">
        <span class="${{actionCls}}">📅 Calendar ${{actionLabel}}</span>
        <span>${{title}}${{locChip}}</span>
      </div>
      <div style="opacity:0.6;font-size:11px;">${{ts}}</div>
      ${{extraLine}}
    </div>
  `;
}}

// -- Chronological timeline of interaction events + calendar modifications +
//    test-sample cards (Phase A4 v2). Test samples sit at their own ts in
//    the same timeline, with a distinct background color, no annotations on
//    regular event cards. --
const timeline = document.getElementById('timeline-section');

if (eventsData.length === 0) {{
  timeline.innerHTML = '<div class="empty">No interaction events available.</div>';
}} else {{
  const grid = document.createElement('div');
  grid.className = 'event-grid';

  // Pre-sort regular events by ts so test-sample location-lookup is O(log n).
  const sortedEvents = eventsData.slice().sort((a, b) => (a.source_timestamp || 0) - (b.source_timestamp || 0));
  // Find nearest preceding event's location for a given ts.
  function _locationAtTs(ts) {{
    let lo = 0, hi = sortedEvents.length - 1, best = null;
    while (lo <= hi) {{
      const mid = (lo + hi) >> 1;
      const evTs = sortedEvents[mid].source_timestamp || 0;
      if (evTs <= ts) {{ best = sortedEvents[mid]; lo = mid + 1; }}
      else hi = mid - 1;
    }}
    return (best && best.event_location) ? best.event_location : null;
  }}

  // Build merged timeline: events + calendar mods + test samples, sorted by ts.
  const timelineItems = [];
  eventsData.forEach((ev, i) => timelineItems.push({{ kind: 'event', ts: ev.source_timestamp || 0, data: ev, eventIdx: i }}));
  (calendarMods || []).forEach(mod => timelineItems.push({{ kind: 'cal', ts: mod.ts || 0, data: mod }}));
  (testSamples || []).forEach(t => timelineItems.push({{ kind: 'test', ts: t.ts || 0, data: t, location: _locationAtTs(t.ts || 0) }}));
  timelineItems.sort((a, b) => (a.ts || 0) - (b.ts || 0));

  timelineItems.forEach(item => {{
    if (item.kind === 'cal') {{
      const div = document.createElement('div');
      div.innerHTML = renderCalendarMod(item.data);
      grid.appendChild(div.firstElementChild);
      return;
    }}
    if (item.kind === 'test') {{
      // Standalone test-sample card — distinct background, same shape as
      // a regular event so the timeline reads naturally. Renders every
      // ground-truth field the extractor populated.
      const t = item.data;
      const loc = item.location;
      let locText = '';
      if (loc && typeof loc === 'object') {{
        const parts = [loc.city, loc.region].filter(x => x).map(escapeHtml);
        if (parts.length > 0) locText = `<span class="event-location">📍 ${{parts.join(', ')}}</span>`;
      }}
      // Match the regular event time format ("HH:MM, MM/DD/YYYY") instead
      // of the ISO string the queries.csv carries.
      let tsDisplay = '';
      if (t.ts) {{
        const d = new Date(t.ts * 1000);
        const pad = n => (n < 10 ? '0' : '') + n;
        tsDisplay = `${{pad(d.getUTCHours())}}:${{pad(d.getUTCMinutes())}}, ${{pad(d.getUTCMonth()+1)}}/${{pad(d.getUTCDate())}}/${{d.getUTCFullYear()}}`;
      }}

      // Build rich-info sections — only render keys the extractor populated.
      let sections = '';
      if (t.ground_truth) {{
        sections += `<div class="ts-section"><div class="ts-label">Expected (ground truth)</div><div class="ts-body">${{escapeHtml(t.ground_truth)}}</div></div>`;
      }}
      if (Array.isArray(t.candidates) && t.candidates.length > 0) {{
        const items = t.candidates.map(c => {{
          const tag = `<span class="ts-origin ts-origin-${{escapeHtml(c.origin || '')}}">${{escapeHtml(c.origin || '')}}</span>`;
          const star = c.is_held_out ? ' <span class="ts-target">★ target</span>' : '';
          const tags = (c.hashtags || []).slice(0, 4).map(escapeHtml).join(' ');
          return `<li><code>idx=${{c.idx}}</code> ${{tag}}${{star}} ${{escapeHtml(c.title || '')}}${{tags ? ` <small>${{tags}}</small>` : ''}}</li>`;
        }}).join('');
        sections += `<div class="ts-section"><div class="ts-label">Candidate pool (${{t.candidates.length}} items)</div><ul class="ts-list">${{items}}</ul></div>`;
      }}
      if (t.held_out_pref) {{
        sections += `<div class="ts-section"><div class="ts-label">Held-out preference</div><div class="ts-body">${{escapeHtml(t.held_out_pref)}}</div></div>`;
      }}
      if (Array.isArray(t.target_prefs) && t.target_prefs.length > 0) {{
        sections += `<div class="ts-section"><div class="ts-label">Target preferences agent SHOULD surface</div><ul class="ts-list">${{t.target_prefs.map(p => `<li>${{escapeHtml(p)}}</li>`).join('')}}</ul></div>`;
      }}
      if (Array.isArray(t.privacy_flagged) && t.privacy_flagged.length > 0) {{
        sections += `<div class="ts-section ts-section-warn"><div class="ts-label">Privacy-flagged (must NOT surface)</div><ul class="ts-list">${{t.privacy_flagged.map(p => `<li>${{escapeHtml(p)}}</li>`).join('')}}</ul></div>`;
      }}
      if (Array.isArray(t.top_k_relevant) && t.top_k_relevant.length > 0) {{
        sections += `<div class="ts-section"><div class="ts-label">Top-k relevant prefs (context)</div><ul class="ts-list">${{t.top_k_relevant.map(p => `<li>${{escapeHtml(p)}}</li>`).join('')}}</ul></div>`;
      }}
      if (Array.isArray(t.tempted_by) && t.tempted_by.length > 0) {{
        sections += `<div class="ts-section"><div class="ts-label">Top-k prefs the agent could pull in (BUT must NOT for restraint test)</div><ul class="ts-list">${{t.tempted_by.map(p => `<li>${{escapeHtml(p)}}</li>`).join('')}}</ul></div>`;
      }}
      if (Array.isArray(t.tool_call_rules) && t.tool_call_rules.length > 0) {{
        sections += `<div class="ts-section"><div class="ts-label">Tool-call rules</div><ul class="ts-list ts-mono">${{t.tool_call_rules.map(r => `<li><code>${{escapeHtml(r)}}</code></li>`).join('')}}</ul></div>`;
      }}
      if (t.final_state_expected && Object.keys(t.final_state_expected).length > 0) {{
        sections += `<div class="ts-section"><div class="ts-label">Final-state expected (writes.jsonl diff)</div><div class="ts-body ts-mono">${{escapeHtml(JSON.stringify(t.final_state_expected, null, 2))}}</div></div>`;
      }}
      if (t.warn_frame) {{
        const wf = t.warn_frame;
        const mm = (wf.must_mention || []).map(escapeHtml).map(s => `<li>${{s}}</li>`).join('');
        const mn = (wf.must_not_mention || []).map(escapeHtml).map(s => `<li>${{s}}</li>`).join('');
        sections += `<div class="ts-section ts-section-warn"><div class="ts-label">Expected warning frame [polarity=${{escapeHtml(wf.polarity || '')}}]</div>` +
                    (mm ? `<div class="ts-sublabel">must_mention</div><ul class="ts-list">${{mm}}</ul>` : '') +
                    (mn ? `<div class="ts-sublabel">must_not_mention</div><ul class="ts-list">${{mn}}</ul>` : '') +
                    `</div>`;
      }}
      if (Array.isArray(t.signal_evidence) && t.signal_evidence.length > 0) {{
        const items = t.signal_evidence.map(s => `<li><code>${{escapeHtml(s.source || '')}}</code> @${{escapeHtml(String(s.ts || ''))}} <small>${{escapeHtml(s.ref || '')}}</small><br>${{escapeHtml(s.quote || '')}}</li>`).join('');
        sections += `<div class="ts-section"><div class="ts-label">Cross-signal evidence (mistake reasoning)</div><ul class="ts-list">${{items}}</ul></div>`;
      }}
      if (Array.isArray(t.irrelevant_persona_items) && t.irrelevant_persona_items.length > 0) {{
        sections += `<div class="ts-section"><div class="ts-label">Irrelevant prefs (distractors agent must reject)</div><ul class="ts-list">${{t.irrelevant_persona_items.map(p => `<li>${{escapeHtml(p)}}</li>`).join('')}}</ul></div>`;
      }}
      if (t.carve_out) {{
        sections += `<div class="ts-section"><div class="ts-label">Carve-out (context shift)</div><div class="ts-body">${{escapeHtml(t.carve_out)}}</div></div>`;
      }}
      if (Array.isArray(t.forbidden_items) && t.forbidden_items.length > 0) {{
        sections += `<div class="ts-section ts-section-warn"><div class="ts-label">Forbidden items</div><ul class="ts-list">${{t.forbidden_items.map(p => `<li>${{escapeHtml(p)}}</li>`).join('')}}</ul></div>`;
      }}
      if (t.extra_meta && Object.keys(t.extra_meta).length > 0) {{
        sections += `<div class="ts-section"><div class="ts-label">Meta</div><div class="ts-body ts-mono">${{escapeHtml(JSON.stringify(t.extra_meta, null, 2))}}</div></div>`;
      }}
      const tags = (t.rubric_tags || []).filter(Boolean).join(', ');
      if (tags) {{
        sections += `<div class="ts-section ts-rubric-bar"><div class="ts-label">Rubric dimensions</div><div class="ts-body">${{escapeHtml(tags)}}</div></div>`;
      }}

      const card = document.createElement('div');
      card.className = 'event-card test-sample-card';
      card.innerHTML = `
        <div class="event-header">
          <div class="event-meta">
            <span style="font-weight:600;color:#7B5C00;">Test sample</span> &middot;
            ${{escapeHtml(tsDisplay)}} &middot;
            ${{locText}}${{locText ? ' &middot; ' : ''}}
            <code>${{escapeHtml(t.task_type || '')}}</code>
          </div>
        </div>
        <div class="ts-label" style="margin-top:6px;">User Query</div>
        <div class="test-sample-query">${{escapeHtml(t.query_text || '')}}</div>
        ${{sections}}
      `;
      grid.appendChild(card);
      return;
    }}
    const ev = item.data;
    const idx = item.eventIdx;
    const app = ev._app || 'Instagram';
    const fmt = ev.interaction_format || {{}};
    const prefs = ev.preferences || [];
    const hashtags = ev.source_hashtags || [];
    const itype = ev.source_interaction_type || '';
    const isImplicitNeg = itype === 'implicit_negative';

    const isAd = !!ev.is_ad;
    const card = document.createElement('div');
    card.className = `event-card app-${{app}}${{isImplicitNeg ? ' implicit-negative' : ''}}${{isAd ? ' is-ad' : ''}}`;

    // Location string
    let locText = '';
    if (ev.event_location && typeof ev.event_location === 'object') {{
      const loc = ev.event_location;
      const parts = [loc.city, loc.region].filter(x => x).map(escapeHtml);
      if (parts.length > 0) {{
        locText = `<span class="event-location">📍 ${{parts.join(', ')}}</span>`;
      }}
    }}

    // Event header (test annotations now live on standalone test cards,
    // not on regular event cards — keeps regular events uncluttered.)
    let headerHtml = `
      <div class="event-header">
        <div class="event-meta">
          <span style="font-weight:600;color:var(--text);">Event #${{idx+1}}</span> &middot;
          ${{ev.formatted_timestamp || ''}} &middot;
          ${{locText}}${{locText ? ' &middot; ' : ''}}
          ${{prefs.length}} preference${{prefs.length !== 1 ? 's' : ''}}
        </div>
        <div>
          <span class="badge platform p-${{app}}">${{app}}</span>
          <span class="badge interaction-type ${{itype}}">${{itype.replace(/_/g, ' ')}}</span>
          ${{fmt.action_label ? `<span class="badge action">${{fmt.action_label}}</span>` : ''}}
          ${{isAd ? `<span class="badge sponsored">Ads</span>` : ''}}
        </div>
        ${{hashtags.length ? `<div class="hashtags">${{hashtags.join('  ')}}</div>` : ''}}
      </div>
    `;

    // Preferences list
    let prefsHtml = '<div class="pref-list">';
    if (prefs.length > 0) {{
      prefsHtml += '<div class="pref-list-label">Preferences</div>';
    }}
    prefs.forEach(p => {{
      let badges = `<span class="badge category">${{p.category || ''}}</span>`;
      if (p.time_horizon === 'short_term') badges += `<span class="badge short-term">short-term</span>`;
      if (p.stereotype_mark && p.stereotype_mark !== 'neutral') badges += `<span class="badge ${{p.stereotype_mark}}">${{p.stereotype_mark}}</span>`;
      if (p.hidden_persona_labels && p.hidden_persona_labels.length > 0) {{
        p.hidden_persona_labels.forEach(lbl => {{
          const motiv = (hpMotivation[lbl] || '').replace(/"/g, '&quot;');
          const titleAttr = motiv ? ` title="${{motiv}}"` : '';
          badges += `<span class="badge hidden-persona"${{titleAttr}}>${{lbl}}</span>`;
        }});
      }}

      // R8: split / over_personalization_irrelevant are no longer emitted
      // by data-gen (eval picks its own test moments from the full history).

      const historyHtml = renderUpdateHistory(p.update_history, ev.source_timestamp);

      let stopConditionLine = '';
      if (p.time_horizon === 'short_term' && p.stop_condition && typeof p.stop_condition === 'object') {{
        const sc = p.stop_condition;
        const typeLabel = sc.type ? `<span class="sc-type">${{escapeHtml(sc.type)}}</span>` : '';
        const desc = sc.description ? escapeHtml(sc.description) : '';
        const stopTs = sc.expected_stop_ts ? new Date(sc.expected_stop_ts * 1000).toISOString().slice(0, 16).replace('T', ' ') : '';
        const stopSuffix = stopTs ? ` <span style="opacity:0.7">(stops ~${{stopTs}})</span>` : '';
        if (desc || typeLabel) {{
          stopConditionLine = `<div class="stop-condition">${{typeLabel}}${{desc}}${{stopSuffix}}</div>`;
        }}
      }}

      prefsHtml += `
        <div class="pref-item">
          <div class="item-text">${{p.persona_item || ''}}</div>
          <div class="conf-inline"><span>init ${{(p.confidence_score_init || 0).toFixed(2)}}</span><span>xref ${{(p.confidence_cross_referenced || 0).toFixed(1)}}</span></div>
          <div class="pref-meta">${{badges}}</div>
          ${{stopConditionLine}}
          ${{historyHtml}}
        </div>
      `;
    }});
    prefsHtml += '</div>';

    // Chatbot conversation
    let convHtml = '';
    if (ev.conversation && ev.conversation.length > 0) {{
      let convLabel = ev.conversation_type ? `<div class="chat-conv-label">${{ev.conversation_type.replace(/_/g, ' ')}}${{ev.ask_to_forget ? ' &middot; ask-to-forget' : ''}}</div>` : '';
      let bubbles = ev.conversation.map(t => {{
        const cls = t.role === 'user' ? 'user-bubble' : 'assistant-bubble';
        const label = t.role === 'user' ? 'You' : 'AI';
        return `<div class="chat-bubble ${{cls}}"><div class="chat-role">${{label}}</div>${{t.content}}</div>`;
      }}).join('');
      convHtml = `${{convLabel}}<div class="chat-thread">${{bubbles}}</div>`;
    }} else if (fmt.user_message) {{
      convHtml = `<div class="user-message">${{fmt.user_message}}</div>`;
    }}

    const contentHtml = renderContent(ev);
    card.innerHTML = headerHtml + contentHtml + prefsHtml + convHtml;
    grid.appendChild(card);
  }});

  timeline.appendChild(grid);
}}
</script>
</body>
</html>"""

    output_path = os.path.join(user_dir, "persona.html")
    os.makedirs(user_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"{utils.Colors.OKGREEN}Visualization saved to {output_path}{utils.Colors.ENDC}")
    return output_path
