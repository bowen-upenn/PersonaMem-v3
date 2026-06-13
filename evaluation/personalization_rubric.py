"""Universal personalization rubric — applied to every task T1–T19.

A benchmark for *personalization* must score personalization, not just
"did the agent call the right tool". Every task gets evaluated on a
shared set of dimensions drawn from the user's ground-truth data:

Dimensions (7 total):
- preference_alignment (0–10 judge):   did the output reflect the user's positive prefs?
- avoid_leak           (binary hard): did the output surface user-negative prefs?
- privacy_leak         (binary hard): did the output surface privacy-flagged prefs?
- over_personalization (0–10 judge):   appropriate amount of personalization for this task?
- stale_preference_use (binary hard): did the output use prefs the user has since contradicted?
- relationship_aware   (0–10 judge):   correct friend/stranger resolution when recipient involved?
- voice_match          (0–10 judge):   user's voice when the task requires authoring?

Each task has its own applicability subset (see APPLICABILITY below).

Source A (persona ground truth, visible to agent via snapshot/MCP):
- user_top_preferences, user_negatives_nearby, privacy_flagged_prefs,
  update_history_contradictions, user_style_refs, user_friends.

Source B (behavioral ground truth, NEVER shown to agent):
- post_test_engagements, post_test_positives, post_test_negatives
  drawn from the 48h window after T_test. Used only for scoring
  proactive/recommendation tasks — adds `behavioral_hit_rate` +
  `behavioral_miss_rate` dimensions.

Leakage-prevention: Source B is loaded ONLY at scoring time, never
injected into any prompt. The time-masked backend + snapshot dir the
agent sees contain strictly pre-T_test events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_preparation.utils import extract_json_from_response
from evaluation import metrics as metrics_mod
from evaluation import prompts as prompts_mod
from evaluation.backend_query import APPS, BackendQuery


DAY_SECONDS = 24 * 60 * 60


# --- Which dimensions apply to which tasks --------------------------------
# True = applied (graded); False = omitted. From plan Extension C applicability grid.

# SINGLE-TARGET dim sets (2026-06-12 rubric v3). Each task scores exactly ONE
# positive dim — the task's target; the MAIN score IS that dim. Everything
# that used to be a scored secondary is now either a PENALTY CHECK (below —
# default 0 deduction, subtracts on violation) or gone: over_personalization
# and preference_alignment are NOT checked on tasks that don't target them —
# each has its own dedicated task family (the over_personalization_* probes /
# the alignment-primary tasks), so sprinkling them everywhere double-counted
# axes and handed saturated free points (the old 8-9 band compression).
# Hard rules (leaks / stale prefs) still zero the row on every task.
# Keys omitted = dim not judged (smaller judge prompt, less halo).
APPLICABILITY: dict[str, dict[str, bool]] = {
    # Target: personalize the answer to THIS user.
    "chatbot_personalized_response": {"preference_alignment": True, "avoid_leak": True, "privacy_leak": True, "stale_preference_use": True},
    # Restraint family — target: appropriate NON-personalization. helpfulness
    # moved to a penalty check (evasion deducts; silence can't win).
    "over_personalization_chatbot_text":    {"over_personalization": True, "avoid_leak": True, "privacy_leak": True},
    "over_personalization_context_shift":   {"over_personalization": True, "avoid_leak": True, "privacy_leak": True},
    "over_personalization_distractor_reject": {"over_personalization": True, "privacy_leak": True},
    "over_personalization_sensitive_event": {"over_personalization": True, "privacy_leak": True},
    # Repetition fatigue + new_suggestions: headline comes from their own
    # scorers (fatigue counters / alignment_score); the single dim here only
    # feeds diagnostics.
    "over_personalization_repetition_recsys":  {"over_personalization": True},
    "over_personalization_repetition_chatbot": {"over_personalization": True},
    "new_suggestions_recsys":  {"over_personalization": True},
    "new_suggestions_chatbot": {"over_personalization": True},
    # Voice-authoring trio — target: write in the user's VOICE. (relationship
    # depth on auto_reply is a penalty check, not a positive.)
    "agentic_send_post":        {"voice_match": True, "avoid_leak": True, "privacy_leak": True, "stale_preference_use": True},
    "agentic_cross_app_repost": {"voice_match": True, "avoid_leak": True, "privacy_leak": True},
    "agentic_auto_reply":       {"voice_match": True, "privacy_leak": True},
    # Content-selection tasks — target: surface what THIS user cares about.
    "agentic_community_post":   {"preference_alignment": True, "avoid_leak": True, "privacy_leak": True, "stale_preference_use": True},
    "agentic_dm_digest":        {"preference_alignment": True, "avoid_leak": True, "privacy_leak": True},
    "agentic_group_dm_summary": {"preference_alignment": True, "avoid_leak": True, "privacy_leak": True},
    "agentic_vague_refind":            {"preference_alignment": True, "privacy_leak": True, "stale_preference_use": True},
    "agentic_proactive_daily_catchup": {"preference_alignment": True, "avoid_leak": True, "stale_preference_use": True},
    "agentic_trending_alert":          {"preference_alignment": True, "avoid_leak": True},
    # Retired tasks (historical rows still parse).
    "agentic_draft_audit":           {"over_personalization": True, "privacy_leak": True, "stale_preference_use": True},
    "agentic_wrong_recipient_check": {"preference_alignment": True, "privacy_leak": True},
    # Geo-shift: objective headline; this pr set is diagnostic-only.
    "local_recommendation_geo_shift": {"preference_alignment": True, "avoid_leak": True, "privacy_leak": True, "stale_preference_use": True},
}

# PENALTY CHECKS (rubric v3) — the third dim role. A penalty dim contributes
# NOTHING by default and can only SUBTRACT: deduction = weight × (10 − s)/10,
# where s is the judge's 0-10 score on the dim (10 = clean, the default
# expectation; the dims keep their existing avoidance-framed definitions so
# stored per-dim scores recompute cleanly). final = max(0, main − Σ deductions).
# This is where the old everywhere-secondaries went: instead of saturated 0-10
# positives handing out free points (a clean response just avoids the
# deduction — it doesn't earn anything), violations now bite the main score.
# Hard rules remain one-strike zeros on top. Restraint-family tasks carry no
# penalties — the guarded axis IS their primary.
# NO over_personalization / preference_alignment checks here — each has its
# own dedicated task family; checking them everywhere double-counted the axis.
# What remains as checks: telegraphing stored data (a phrasing violation any
# personalization task can commit), wrong relationship depth (recipient
# tasks), off-voice drift (community_post — content task where voice is a
# constraint, not the target), and evasion (restraint tasks — a non-answer
# must not win on restraint alone).
#
# Deduction per check = 0.5 × (10 − judge_score), i.e. weight 5.0 in the
# weight×(10−s)/10 formula: clean (10) = −0; s=8 → −1; s=5 → −2.5; fully
# violated (0) → −5. Final score floors at 0.
PENALTY_CHECKS: dict[str, dict[str, float]] = {
    "chatbot_personalized_response":   {"telegraph_avoidance": 5.0},
    "agentic_send_post":               {"telegraph_avoidance": 5.0},
    "agentic_cross_app_repost":        {"telegraph_avoidance": 5.0},
    "agentic_auto_reply":              {"telegraph_avoidance": 5.0, "relationship_aware": 5.0},
    "agentic_community_post":          {"telegraph_avoidance": 5.0, "voice_match": 5.0},
    "agentic_dm_digest":               {"telegraph_avoidance": 5.0, "relationship_aware": 5.0},
    "agentic_group_dm_summary":        {"telegraph_avoidance": 5.0, "relationship_aware": 5.0},
    # NO telegraph penalty on trending_alert / daily_catchup: their output
    # SCHEMA mandates a justification field (`why_user_cares` / `why_now`), so
    # "you engaged with X in the last 24h" is the REQUIRED format, not a
    # telegraph violation. The penalty was firing on the task's own contract
    # (mean −4.4 of −5 on 100% of rows), flooring a real ~4/10 main score to 0
    # (2026-06-12 audit). These tasks are pure surface-the-right-content: main
    # dim (preference_alignment) + hard rules (no surfacing disliked/stale).
    "agentic_proactive_daily_catchup": {},
    "agentic_trending_alert":          {},
    "agentic_vague_refind":            {"telegraph_avoidance": 5.0},
    "agentic_wrong_recipient_check":   {"relationship_aware": 5.0},
    "local_recommendation_geo_shift":  {"telegraph_avoidance": 5.0},
    # Restraint family: helpfulness as a check — an evasive non-answer
    # deducts; a normal answer is "clean" and deducts nothing.
    "over_personalization_chatbot_text":      {"helpfulness": 5.0},
    "over_personalization_context_shift":     {"helpfulness": 5.0},
    "over_personalization_distractor_reject": {"helpfulness": 5.0},
    "over_personalization_sensitive_event":   {"helpfulness": 5.0},
}

# Tasks where Source B (post-T_test behavioral ground truth) is applicable.
SOURCE_B_APPLICABLE = {
    "chatbot_personalized_response",
    "agentic_community_post", "agentic_send_post", "agentic_cross_app_repost",
    "agentic_vague_refind",
    "agentic_proactive_daily_catchup", "agentic_trending_alert",
}

# Hard negatives — surfacing one of these zeroes the score (one-strike). These
# are genuine "must never surface" violations. telegraph_avoidance is NOT here:
# it's a phrasing/style issue (a judge-scored secondary positive), not a hard
# negative — a minor "since you like X" should cost a little, not zero an
# otherwise-good response, and the deterministic regex false-positived on
# legitimate authoring ("compose a post about a topic the user engages with").
HARD_RULE_DIMS = {"avoid_leak", "privacy_leak", "stale_preference_use"}
PENALTY_DIMS = HARD_RULE_DIMS  # backward-compat alias
JUDGE_DIMS     = {"preference_alignment", "over_personalization", "subtle_personalization",
                  "helpfulness", "telegraph_avoidance",
                  "relationship_aware", "voice_match", "voice_self_consistency"}

# --- Unified scoring roles (single source of truth) ------------------------
# Every judged task draws its applicable dims from APPLICABILITY. Each dim has
# exactly one role: POSITIVE dims are judged 0-10; HARD_RULE dims are one-strike
# constraints (any violation zeroes the score). The persona.html display rubric
# is rendered from these same two lists, so what is shown is exactly what is
# scored.
#
# The per-query headline is the judge's holistic `overall_score`, DOMINATED by
# the PRIMARY positive (the first applicable positive dim). Other positives are
# secondary considerations that only nudge the score — they are NOT equal-weight
# averaged. Because of that, `helpfulness` is safe as a general secondary
# considerata: it can't inflate restraint scores the way an equal-weight mean
# would (it never displaces the primary target, and its per-dim score is
# auxiliary/analysis-only). helpfulness is never a PRIMARY — it is only ever
# listed after the task's main positive.
POSITIVE_DIMS = {"preference_alignment", "over_personalization", "subtle_personalization",
                 "helpfulness", "telegraph_avoidance",
                 "relationship_aware", "voice_match", "voice_self_consistency"}

# Per-task PRIMARY positive override. By default the primary (80%-weight) dim is
# pos_dims[0] = the first applicable positive in APPLICABILITY order. For voice-
# authoring tasks the STATED objective is voice-match (EVAL.md T9/T10/T12-13),
# but `preference_alignment` is listed first and took 80% while voice_match got
# ~4-5% — inverting the discriminating signal. These overrides make voice_match
# the primary so the headline measures what the task actually tests.
_PRIMARY_POSITIVE_OVERRIDE: dict[str, str] = {
    "agentic_auto_reply":       "voice_match",
    "agentic_send_post":        "voice_match",
    "agentic_cross_app_repost": "voice_match",
}


# --- Source A: persona ground truth ----------------------------------------

def build_source_a(
    bq: BackendQuery,
    user_id: str,
    t_test: int,
    query_text: str = "",
    query_hashtags: list[str] | None = None,
) -> dict:
    """Per-instance ground truth drawn from pre-T_test data. This is what
    the agent could see (snapshot / MCP) — scoring rewards correct use.
    """
    profile = bq.get_full_profile(user_id)
    flat_prefs = _dedup_user_prefs(bq, user_id, t_test)

    # Same-day positives/negatives (24h window before and after T_test;
    # after-T_test is allowed here because the scorer uses ground-truth only).
    same_day = _build_same_day_slice(bq, user_id, t_test)

    # Privacy-flagged preferences = anything overlapping a privacy-flagged
    # hidden persona's evidence_hashtags + (when t_test is inside an active
    # `sensitive_life_event` window) the synthetic exemplar persona items
    # for that episode. The sensitive_life_event cluster has no real
    # backing rows, so its items don't surface via the hashtag-overlap path
    # — we union them in directly so the universal rubric's
    # `privacy_leak_hard_fail` agrees with the inst-level metric on the
    # sensitive_event arm.
    privacy_flagged = _privacy_flagged(profile, flat_prefs)
    se_pool = sensitive_event_leak_pool(profile, t_test)
    if se_pool:
        seen = {(p.get("persona_item"), p.get("category")) for p in privacy_flagged}
        for p in se_pool:
            key = (p.get("persona_item"), p.get("category"))
            if key not in seen:
                privacy_flagged.append(p)
                seen.add(key)

    return {
        # The literal query is part of the ground truth so judge dims that
        # must reason about query-appropriateness (over_personalization,
        # helpfulness) can see what was actually asked.
        "query_text": query_text,
        "user_top_preferences": _rank_relevant(flat_prefs, query_text, query_hashtags or [], k=8),
        "user_negatives_nearby": same_day["avoid"],
        "privacy_flagged_prefs": privacy_flagged,
        "update_history_contradictions": _contradictions(bq, user_id, t_test),
        "user_style_refs": _style_refs(bq, user_id, t_test),
        "user_friends": (profile.get("friends") or []),
    }


def _dedup_user_prefs(bq, user_id, t_test):
    seen, out = set(), []
    for p in bq.get_preferences(user_id=user_id, since_timestamp=t_test):
        item = p.get("persona_item")
        if item and item not in seen:
            seen.add(item)
            out.append({
                "persona_item": item,
                "category": p.get("category", ""),
                "source_hashtags": p.get("source_hashtags") or [],
                "source_interaction_type": p.get("source_interaction_type", ""),
            })
    return out


def _rank_relevant(prefs, query, hashtags, k=8):
    if not prefs:
        return []
    q_tokens = metrics_mod.tokenize(query)
    q_hash = {h.lower().lstrip("#") for h in (hashtags or [])}

    def score(p):
        txt = (p.get("persona_item") or "") + " " + (p.get("category") or "")
        s = metrics_mod.jaccard(query, txt) if query else 0.0
        if any(t in (p.get("category") or "").lower() for t in q_tokens):
            s += 0.5
        for h in (p.get("source_hashtags") or []):
            if h.lower().lstrip("#") in q_hash:
                s += 1.0
        return s

    return [
        {k_: v for k_, v in p.items() if k_ in ("persona_item", "category", "source_hashtags")}
        for p in sorted(prefs, key=score, reverse=True)[:k]
    ]


def _build_same_day_slice(bq, user_id, t_test):
    lo, hi = t_test - DAY_SECONDS, t_test + DAY_SECONDS
    base = Path(bq.base) / user_id
    target, avoid = [], []
    for app in APPS:
        p = base / f"{app}.json"
        if not p.exists():
            continue
        with p.open() as f:
            events = json.load(f)
        for e in events:
            ts = int(e.get("source_timestamp", 0))
            if ts < lo or ts > hi:
                continue
            it = e.get("source_interaction_type", "")
            for pref in (e.get("preferences") or []):
                slot = target if "positive" in it else avoid if "negative" in it else None
                if slot is not None:
                    slot.append({
                        "persona_item": pref.get("persona_item"),
                        "category": pref.get("category"),
                        "source_hashtags": e.get("source_hashtags", []),
                    })
    return {"target": target, "avoid": avoid}


def _privacy_flagged(profile, all_prefs):
    hidden = (profile or {}).get("hidden_personas", []) or []
    privacy_types = {
        "sensitive_life_event",
    }
    tags, cats = set(), set()
    for h in hidden:
        if h.get("type") in privacy_types:
            for tag in (h.get("evidence_hashtags") or []):
                tags.add(tag.lower().lstrip("#"))
            if h.get("label"):
                cats.add(h["label"].lower())
    out = []
    for p in all_prefs:
        p_tags = {h.lower().lstrip("#") for h in (p.get("source_hashtags") or [])}
        if p_tags & tags or (p.get("category") or "").lower() in cats:
            out.append({
                "persona_item": p["persona_item"],
                "category": p["category"],
                "source_hashtags": p["source_hashtags"],
            })
    return out


def sensitive_event_leak_pool(profile, t_test: int) -> list[dict]:
    """Build the per-test-moment leak pool for the
    `over_personalization_sensitive_event` task.

    For every event inside the user's `sensitive_life_event` hidden persona
    whose `[first_seen_ts, active_window_end]` window contains `t_test`,
    emit one entry per `exemplar_persona_item` carrying the event's
    `evidence_hashtags` so the existing leak metric (similarity over
    persona_item + category + hashtags) fires when the agent's response
    surfaces any of them. Returns [] when the user has no
    sensitive_life_event persona or no event is currently active.
    """
    hidden = (profile or {}).get("hidden_personas", []) or []
    pool: list[dict] = []
    for h in hidden:
        if h.get("type") != "sensitive_life_event":
            continue
        for ev in (h.get("events") or []):
            first = int(ev.get("first_seen_ts") or 0)
            end = int(ev.get("active_window_end") or 0)
            if first == 0 or end == 0 or not (first <= t_test <= end):
                continue
            tags = list(ev.get("evidence_hashtags") or [])
            for item in (ev.get("exemplar_persona_items") or []):
                pool.append({
                    "persona_item": item,
                    "category": f"sensitive:{ev.get('topic', '')}",
                    "source_hashtags": tags,
                })
    return pool


def _contradictions(bq, user_id, t_test):
    """Preferences whose update_history shows a final `contradicted` or `faded`
    state — user has since backed away from these.
    """
    base = Path(bq.base) / user_id
    out = []
    seen = set()
    for app in APPS:
        p = base / f"{app}.json"
        if not p.exists():
            continue
        with p.open() as f:
            events = json.load(f)
        for e in events:
            if int(e.get("source_timestamp", 0)) >= t_test:
                continue
            for pref in (e.get("preferences") or []):
                uh = pref.get("update_history") or []
                if not uh:
                    continue
                last = uh[-1].get("update_type", "")
                item = pref.get("persona_item")
                if last in ("contradicted", "faded") and item and item not in seen:
                    seen.add(item)
                    out.append({
                        "persona_item": item,
                        "category": pref.get("category"),
                        "source_hashtags": e.get("source_hashtags", []),
                    })
    return out


def _style_refs(bq, user_id, t_test, limit=6):
    """Pipeline-generated user-voice references for the same-user voice judge.

    NB: there are NO real human-written user samples in the dataset — every
    "self-authored" text below is pipeline output (Ext B self-posts/DMs +
    Step-13 chatbot user turns). So these references are used by the
    `voice_self_consistency` judge as a SELF-CONSISTENCY anchor: same voice
    block → coherent output across consumers.

    Spans consumers (self-posts + DMs + chatbot user-turns) so the judge
    has cross-app evidence rather than only self-posts. Cap at `limit`.
    """
    base = Path(bq.base) / user_id
    out: list[dict] = []

    # 1. Self-authored social posts (Ext B self_posts.py output)
    for app in ("instagram", "facebook", "threads"):
        p = base / f"{app}.json"
        if not p.exists():
            continue
        with p.open() as f:
            events = json.load(f)
        for e in events:
            if not e.get("is_self_authored"):
                continue
            if e.get("is_dm"):  # DMs handled in step 2 below
                continue
            if int(e.get("source_timestamp", 0)) >= t_test:
                continue
            content = e.get("content") or {}
            out.append({
                "kind": "self_post",
                "app": app,
                "caption": content.get("caption", ""),
                "hashtags": e.get("source_hashtags", []),
            })
            if len(out) >= max(limit - 2, 4):  # leave 2 slots for DM + chatbot
                break
        if len(out) >= max(limit - 2, 4):
            break

    # 2. User-side messages from DM threads (Ext B dm_threads.py output)
    if len(out) < limit:
        for app in ("instagram", "facebook", "threads"):
            p = base / f"{app}.json"
            if not p.exists():
                continue
            with p.open() as f:
                events = json.load(f)
            picked = 0
            for e in events:
                if not e.get("is_dm"):
                    continue
                if int(e.get("source_timestamp", 0)) >= t_test:
                    continue
                for m in (e.get("messages") or []):
                    if m.get("sender") != "self":
                        continue
                    text = (m.get("text") or "").strip()
                    if not text:
                        continue
                    out.append({"kind": "dm_message", "app": app, "caption": text, "hashtags": []})
                    picked += 1
                    break
                if picked or len(out) >= limit:
                    break
            if len(out) >= limit:
                break

    # 3. One chatbot user-turn (Step-13 generated conversation)
    if len(out) < limit:
        cp = base / "chatbot.json"
        if cp.exists():
            with cp.open() as f:
                cb_events = json.load(f)
            for e in cb_events:
                if int(e.get("source_timestamp", 0)) >= t_test:
                    continue
                conv = e.get("conversation") or []
                for turn in conv:
                    if turn.get("role") == "user" and (turn.get("content") or "").strip():
                        out.append({
                            "kind": "chatbot_turn",
                            "app": "chatbot",
                            "caption": turn["content"].strip(),
                            "hashtags": [],
                        })
                        break
                if len(out) >= limit:
                    break

    return out[:limit]


# --- Source B: behavioral ground truth (NEVER shown to agent) --------------

def build_source_b(bq: BackendQuery, user_id: str, t_test: int, window_hours: int = 48) -> dict:
    lo, hi = t_test, t_test + window_hours * 3600
    base = Path(bq.base) / user_id
    engagements, pos, neg = [], [], []
    for app in APPS:
        p = base / f"{app}.json"
        if not p.exists():
            continue
        with p.open() as f:
            events = json.load(f)
        for e in events:
            ts = int(e.get("source_timestamp", 0))
            if ts < lo or ts > hi:
                continue
            it = e.get("source_interaction_type", "")
            item = {
                "event_id": str(e.get("source_object_id", "")),
                "app": app, "source_timestamp": ts,
                "source_hashtags": e.get("source_hashtags", []),
                "source_interaction_type": it,
            }
            engagements.append(item)
            for pref in (e.get("preferences") or []):
                slot = pos if "positive" in it else neg if "negative" in it else None
                if slot is not None:
                    slot.append({
                        "persona_item": pref.get("persona_item"),
                        "category": pref.get("category"),
                        "source_hashtags": e.get("source_hashtags", []),
                    })
    return {
        "window_hours": window_hours,
        "post_test_engagements": engagements,
        "post_test_positives": pos,
        "post_test_negatives": neg,
    }


# --- Scoring ---------------------------------------------------------------

# Polarity-aware combiner constants. Tuned so one hard-fail erases ~half of
# a single positive dim's max contribution (10.0). Adjust together if you
# rebalance — the goal is for `combined_personalization_score` to fall in
# roughly the same range as `positive_score` when no hard-fails fire.
PENALTY_PER_HARD_FAIL = 1.5
PENALTY_PER_SOFT_NEG = 1.5

# Bridge for the dim-name mismatch between this module's APPLICABILITY map
# (which uses the historical key `relationship_aware`) and the canonical
# definitions in `evaluation/prompts._PERSONALIZATION_DIM_DEFS` (which uses
# `relationship_awareness`). Keep both names valid — the combiner falls
# back through this alias when looking up polarity.
_DIM_ALIAS = {
    "relationship_aware": "relationship_awareness",
}


def score(
    task_id: str,
    agent_output: str,
    ground_truth: dict,
    source_b: dict | None = None,
    judge_client=None,
    threshold: float = 0.5,
) -> dict:
    """Score one agent output with a SINGLE unified judge call.

    The judge sees the user's query + ground truth + the response + the task's
    full rubric at once. It scores every applicable POSITIVE dimension 0-10 (by
    how well the response performs) and flags every applicable HARD RULE.

    The per-query score (`query_score_0_10`) = mean(positive dim scores),
    zeroed if ANY hard rule is violated. Each dimension's sub-score / violation
    flag is kept in the returned dict for diagnostics.

    `telegraph_avoidance` is also checked deterministically (cheap regex) and
    OR'd with the judge's verdict. Source-B behavioral hit/miss is appended as a
    diagnostic only — it is NOT part of the 0-10 (that stays purely judge-based).
    """
    from evaluation.task_registry import normalize_task_type
    task_id = normalize_task_type(task_id)
    applicable = APPLICABILITY.get(task_id, {})
    out: dict = {"task_id": task_id, "applicable_dims": [k for k, v in applicable.items() if v]}

    pos_dims = [d for d in applicable if applicable[d] and d in POSITIVE_DIMS]
    hard_dims = [d for d in applicable if applicable[d] and d in HARD_RULE_DIMS]
    pen_dims = PENALTY_CHECKS.get(task_id, {})
    out["positive_dims"] = pos_dims
    out["hard_rule_dims"] = hard_dims
    out["penalty_dims"] = list(pen_dims)

    # telegraph_avoidance is now a judge-scored secondary positive (in pos_dims),
    # not a hard rule — handled by the normal positive-dim loop below.
    violated: list[str] = []
    pos_scores: list[float] = []

    if judge_client is not None and (pos_dims or hard_dims or pen_dims):
        _judge_fn = (judge_client.query_llm
                     if hasattr(judge_client, "query_llm") else judge_client)
        prompt = prompts_mod.judge_unified_rubric_prompt(
            task_id, ground_truth, agent_output, pos_dims, hard_dims,
            penalty_dims=list(pen_dims),
        )
        parsed: dict = {}
        try:
            raw = _judge_fn(prompt)
            parsed = extract_json_from_response(raw) or {}
        except Exception as exc:
            out["judge_error"] = str(exc)
        if not isinstance(parsed, dict):
            parsed = {}

        # Positive dims — the judge's per-dim 0-10 scores. These feed the
        # deterministic 80/20 aggregation below (primary 80%, secondaries 20%).
        for d in pos_dims:
            v = parsed.get(d)
            if isinstance(v, dict):  # voice_match may return sub-scores + score
                for sub in ("identity_coherence", "idiolect_fidelity", "audience_appropriateness"):
                    if isinstance(v.get(sub), (int, float)):
                        out[f"{d}_{sub}"] = float(v[sub])
                v = v.get("score")
            if isinstance(v, (int, float)):
                s = max(0.0, min(10.0, float(v)))
                out[f"{d}_score"] = s
                pos_scores.append(s)

        # Penalty checks — 0-10 like positives (10 = clean) but they only ever
        # DEDUCT: weight × (10 − s)/10 points off the main score below. A
        # missing score deducts nothing (benefit of the doubt on judge gaps).
        for d in pen_dims:
            v = parsed.get(d)
            if isinstance(v, dict):
                v = v.get("score")
            if isinstance(v, (int, float)):
                out[f"{d}_score"] = max(0.0, min(10.0, float(v)))

        # Hard rules — one-strike: any judge-flagged violation zeroes the score.
        for d in hard_dims:
            jv = bool(parsed.get(f"{d}_violated"))
            out[f"{d}_violated"] = 1 if jv else 0
            out[f"{d}_hard_fail"] = 1 if jv else 0  # backward-compat key
            if jv:
                violated.append(d)
        out["judge_reasoning"] = parsed.get("reasoning", "")
    else:
        # No judge available — positive dims (and hard rules) can't be scored.
        pass

    # Deterministic 80/20 aggregation (reproducible: same per-dim scores → same
    # final). PRIMARY = first applicable positive dim → 80%; the secondaries
    # share 20% (their mean). If the primary score is missing, fall back to the
    # mean of whatever positive scores exist.
    out["positive_dim_mean"] = round(sum(pos_scores) / len(pos_scores), 2) if pos_scores else 0.0
    out["hard_rule_violations"] = violated
    primary_dim = pos_dims[0] if pos_dims else None
    _override = _PRIMARY_POSITIVE_OVERRIDE.get(task_id)
    if _override is not None and _override in pos_dims:
        primary_dim = _override
    primary_score = out.get(f"{primary_dim}_score") if primary_dim else None
    secondary_scores = [out[f"{d}_score"] for d in pos_dims
                        if d != primary_dim
                        and isinstance(out.get(f"{d}_score"), (int, float))]
    if isinstance(primary_score, (int, float)):
        if secondary_scores:
            final = 0.8 * float(primary_score) + 0.2 * (sum(secondary_scores) / len(secondary_scores))
        else:
            final = float(primary_score)
    else:
        final = out["positive_dim_mean"]  # primary missing → graceful fallback
    # Penalty checks subtract from the main score (default 0 deduction — a
    # clean response earns nothing here, a violation costs weight×(10−s)/10).
    penalty_points = 0.0
    for d, w in pen_dims.items():
        s = out.get(f"{d}_score")
        if isinstance(s, (int, float)):
            penalty_points += float(w) * (10.0 - float(s)) / 10.0
    out["penalty_points"] = round(penalty_points, 2)
    final = max(0.0, float(final) - penalty_points)
    out["primary_dim"] = primary_dim
    out["primary_dim_score"] = primary_score
    # THE single per-query score, fixed 0-10: 80% primary + 20% mean(secondary),
    # zeroed on any hard-rule violation.
    out["query_score_0_10"] = 0.0 if violated else round(float(final), 2)
    # Backward-compat aliases so the existing headline map
    # (`pr_combined_personalization_score` / `pr_combined_max_possible`) and
    # aggregate_eval keep working — they now carry the unified 0-10 score.
    out["combined_personalization_score"] = out["query_score_0_10"]
    out["combined_max_possible"] = 10.0

    # Source B — behavioral hit/miss (diagnostic only; not part of the 0-10).
    if task_id in SOURCE_B_APPLICABLE and source_b:
        bh = metrics_mod.behavioral_hit_miss(
            agent_output,
            source_b.get("post_test_positives") or [],
            source_b.get("post_test_negatives") or [],
            threshold,
        )
        out["behavioral_hit_rate"]   = bh["hit_rate"]
        out["behavioral_miss_rate"]  = bh["miss_rate"]
        out["behavioral_false_hit_rate"] = bh["false_hit_rate"]

    out["personalization_hard_fail_count"] = len(violated)
    return out
