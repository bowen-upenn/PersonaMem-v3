"""Universal personalization rubric — applied to every task T1–T19.

A benchmark for *personalization* must score personalization, not just
"did the agent call the right tool". Every task gets evaluated on a
shared set of dimensions drawn from the user's ground-truth data:

Dimensions (7 total):
- preference_alignment (0–3 judge):   did the output reflect the user's positive prefs?
- avoid_leak           (binary hard): did the output surface user-negative prefs?
- privacy_leak         (binary hard): did the output surface privacy-flagged prefs?
- over_personalization (0–3 judge):   appropriate amount of personalization for this task?
- stale_preference_use (binary hard): did the output use prefs the user has since contradicted?
- relationship_aware   (0–3 judge):   correct friend/stranger resolution when recipient involved?
- voice_match          (0–3 judge):   user's voice when the task requires authoring?

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

APPLICABILITY: dict[str, dict[str, bool]] = {
    # Chatbot response (was: chatbot_response_proactive / _control)
    "chatbot_personalized_response": {"preference_alignment": True, "avoid_leak": True, "privacy_leak": True, "over_personalization": True, "subtle_personalization": True, "stale_preference_use": True, "relationship_aware": False, "voice_match": False, "telegraph_avoidance": True},
    "over_personalization_chatbot_text": {"preference_alignment": False, "avoid_leak": True, "privacy_leak": True, "over_personalization": True, "subtle_personalization": True, "stale_preference_use": False, "relationship_aware": False, "voice_match": False, "telegraph_avoidance": True},
    # Restraint family — repetition fatigue tested in two surface modes
    # (recsys-loop vs varied chatbot questions on same pref).
    "over_personalization_repetition_recsys":  {"preference_alignment": False, "avoid_leak": False, "privacy_leak": False, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": False, "relationship_aware": False, "voice_match": False, "telegraph_avoidance": True},
    "over_personalization_repetition_chatbot": {"preference_alignment": False, "avoid_leak": False, "privacy_leak": False, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": False, "relationship_aware": False, "voice_match": False, "telegraph_avoidance": True},
    # new_suggestions — explorative recommendation. Agent must avoid
    # over-personalization (no recycling fatigued topics) AND avoid
    # the M1 telegraph phrasings. No fixed preference to align against
    # — the gold IS a fresh topic, so preference_alignment is off.
    "new_suggestions_recsys":  {"preference_alignment": False, "avoid_leak": False, "privacy_leak": False, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": False, "relationship_aware": False, "voice_match": False, "telegraph_avoidance": True},
    "new_suggestions_chatbot": {"preference_alignment": False, "avoid_leak": False, "privacy_leak": False, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": False, "relationship_aware": False, "voice_match": False, "telegraph_avoidance": True},
    "over_personalization_context_shift": {"preference_alignment": False, "avoid_leak": True, "privacy_leak": True, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": False, "relationship_aware": True, "voice_match": False, "telegraph_avoidance": True},
    "over_personalization_distractor_reject": {"preference_alignment": False, "avoid_leak": False, "privacy_leak": True, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": False, "relationship_aware": False, "voice_match": False, "telegraph_avoidance": True},
    "over_personalization_sensitive_event": {"preference_alignment": False, "avoid_leak": False, "privacy_leak": True, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": False, "relationship_aware": False, "voice_match": False, "telegraph_avoidance": True},
    "preference_removal_regen":          {"preference_alignment": False, "avoid_leak": False, "privacy_leak": True, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": False, "relationship_aware": False, "voice_match": False, "telegraph_avoidance": True},
    # Agentic family (was: t6..t19)
    "agentic_user_tone_post":          {"preference_alignment": True, "avoid_leak": True, "privacy_leak": True, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": True, "relationship_aware": False, "voice_match": True, "voice_self_consistency": True, "telegraph_avoidance": True},
    # agentic_moment_recommendation merged into personalized_recommendation
    # (slate-based ranking, deterministic recall@k / ndcg@k / mrr metrics).
    "agentic_dm_digest":                 {"preference_alignment": True, "avoid_leak": True, "privacy_leak": True, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": False, "relationship_aware": True, "voice_match": False, "telegraph_avoidance": True},
    "agentic_cross_app_repost":          {"preference_alignment": True, "avoid_leak": True, "privacy_leak": True, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": False, "relationship_aware": False, "voice_match": True, "voice_self_consistency": True, "telegraph_avoidance": True},
    "agentic_auto_reply":                {"preference_alignment": True, "avoid_leak": False, "privacy_leak": True, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": False, "relationship_aware": True, "voice_match": True, "voice_self_consistency": True, "telegraph_avoidance": True},
    "agentic_vague_refind":              {"preference_alignment": True, "avoid_leak": False, "privacy_leak": True, "over_personalization": False, "subtle_personalization": False, "stale_preference_use": True, "relationship_aware": False, "voice_match": False, "telegraph_avoidance": True},
    "agentic_composed_post":             {"preference_alignment": True, "avoid_leak": True, "privacy_leak": True, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": True, "relationship_aware": False, "voice_match": True, "voice_self_consistency": True, "telegraph_avoidance": True},
    "agentic_send_post":                 {"preference_alignment": True, "avoid_leak": True, "privacy_leak": True, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": False, "relationship_aware": False, "voice_match": True, "voice_self_consistency": True, "telegraph_avoidance": True},
    "agentic_draft_audit":               {"preference_alignment": False, "avoid_leak": False, "privacy_leak": True, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": True, "relationship_aware": False, "voice_match": False, "telegraph_avoidance": True},
    "agentic_group_dm_summary":          {"preference_alignment": True, "avoid_leak": True, "privacy_leak": True, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": False, "relationship_aware": True, "voice_match": False, "telegraph_avoidance": True},
    "agentic_wrong_recipient_check":     {"preference_alignment": True, "avoid_leak": False, "privacy_leak": True, "over_personalization": False, "subtle_personalization": False, "stale_preference_use": False, "relationship_aware": True, "voice_match": False, "telegraph_avoidance": True},
    "agentic_proactive_daily_catchup":   {"preference_alignment": True, "avoid_leak": True, "privacy_leak": True, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": True, "relationship_aware": False, "voice_match": False, "telegraph_avoidance": True},
    "agentic_trending_alert":            {"preference_alignment": True, "avoid_leak": True, "privacy_leak": False, "over_personalization": True, "subtle_personalization": False, "stale_preference_use": False, "relationship_aware": False, "voice_match": False, "telegraph_avoidance": True},
    # Silent geo-shift local recommendation. The agent is supposed to
    # personalize MORE (use the latest geo signal + the user's persona
    # profile), so preference_alignment is the headline judge dimension.
    # Hard rules still apply (no surfacing of negative prefs / privacy
    # flags / contradicted prefs). voice_match is off — the agent is
    # answering as the assistant, not authoring in the user's voice.
    "local_recommendation_geo_shift":    {"preference_alignment": True, "avoid_leak": True, "privacy_leak": True, "over_personalization": False, "subtle_personalization": False, "stale_preference_use": True, "relationship_aware": False, "voice_match": False, "telegraph_avoidance": True},
}

# Tasks where Source B (post-T_test behavioral ground truth) is applicable.
SOURCE_B_APPLICABLE = {
    "chatbot_personalized_response",
    "agentic_user_tone_post", "agentic_cross_app_repost",
    "agentic_vague_refind",
    "agentic_proactive_daily_catchup", "agentic_trending_alert",
}

HARD_RULE_DIMS = {"avoid_leak", "privacy_leak", "stale_preference_use",
                  "telegraph_avoidance"}
JUDGE_DIMS     = {"preference_alignment", "over_personalization", "subtle_personalization",
                  "relationship_aware", "voice_match", "voice_self_consistency",
                  "telegraph_avoidance"}


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
        "intimate_interest", "covert_concern", "compensatory_need",
        "medical_aesthetic_concern", "sensitive_life_event",
    }
    tags, cats = set(), set()
    for h in hidden:
        if (h.get("privacy_ratio") or 0) > 0.7 or h.get("type") in privacy_types:
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
# a single positive dim's max contribution (3.0). Adjust together if you
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


def combine_dim_scores_with_polarity(out: dict, applicable: dict) -> None:
    """Aggregate per-dim scores into a polarity-aware combined score.

    Mutates `out` in place to add three keys:
      - `positive_score`: sum of `out[f"{dim}_score"]` over each applicable
        `+` dim that has a numeric score (0-3 each).
      - `negative_penalty`: sum over each applicable `-` dim of either
        `hard_fail * PENALTY_PER_HARD_FAIL` (binary dims) or
        `(3 - dim_score) / 3 * PENALTY_PER_SOFT_NEG` (gap-from-ideal for
        soft 0-3 negative dims).
      - `combined_personalization_score`: clamp(positive_score - negative_penalty, 0, max_possible).

    Polarity comes from `evaluation.prompts._PERSONALIZATION_DIM_DEFS` via
    `prompts.get_dim_polarity`. Unknown dims default to `+`.

    Caller note: when `judge_client=None`, no `+` dim ever has a score, so
    `combined_personalization_score` reflects only hard-fail penalties
    (clamped at 0). Document this in the run report.
    """
    from evaluation import prompts as _prompts_mod

    def _polarity(dim: str) -> str:
        # Try canonical name; fall back through alias bridge.
        pol = _prompts_mod.get_dim_polarity(dim)
        if pol == "+" and dim in _DIM_ALIAS:
            # Default-fallback in get_dim_polarity returns "+" for unknown
            # dims. Re-check with the canonical name from the alias map so
            # `relationship_aware` correctly resolves to `+`.
            pol = _prompts_mod.get_dim_polarity(_DIM_ALIAS[dim])
        return pol

    pos_dims_applicable: list[str] = []
    neg_dims_applicable: list[str] = []
    for dim, is_on in (applicable or {}).items():
        if not is_on:
            continue
        if _polarity(dim) == "+":
            pos_dims_applicable.append(dim)
        else:
            neg_dims_applicable.append(dim)

    # Positive contribution: judge-dim scores on the 0-3 scale.
    positive_score = 0.0
    n_positive_scored = 0
    for dim in pos_dims_applicable:
        v = out.get(f"{dim}_score")
        if isinstance(v, (int, float)):
            positive_score += float(v)
            n_positive_scored += 1

    # Negative penalty: hard-fail flag × constant for binary dims; soft-gap
    # × constant for 0-3 negative dims.
    negative_penalty = 0.0
    for dim in neg_dims_applicable:
        if dim in HARD_RULE_DIMS:
            hf = out.get(f"{dim}_hard_fail")
            if isinstance(hf, (int, float)) and hf:
                negative_penalty += PENALTY_PER_HARD_FAIL
        else:
            # Soft 0-3 negative dim — gap from ideal counts as penalty.
            v = out.get(f"{dim}_score")
            if isinstance(v, (int, float)):
                gap = max(0.0, 3.0 - float(v)) / 3.0
                negative_penalty += gap * PENALTY_PER_SOFT_NEG

    max_possible = 3.0 * len(pos_dims_applicable)
    combined = max(0.0, min(max_possible, positive_score - negative_penalty))

    out["positive_score"] = positive_score
    out["negative_penalty"] = negative_penalty
    out["combined_personalization_score"] = combined
    out["combined_max_possible"] = max_possible


def score(
    task_id: str,
    agent_output: str,
    ground_truth: dict,
    source_b: dict | None = None,
    judge_client=None,
    threshold: float = 0.5,
) -> dict:
    """Score an agent output on the applicable personalization dimensions.

    Hard dims run always (cheap, no LLM). Judge dims run only when
    `judge_client` is provided.

    Returns a flat dict of dimension_name → score (and hard_fail flags).
    """
    # Phase 0: support v1 task names that may flow in from older queries.csv
    # files via runner. Normalize to the v2 name before APPLICABILITY lookup.
    from evaluation.task_registry import normalize_task_type
    task_id = normalize_task_type(task_id)
    applicable = APPLICABILITY.get(task_id, {})
    out: dict = {"task_id": task_id, "applicable_dims": [k for k, v in applicable.items() if v]}

    # Hard-rule dims — cheap, no LLM.
    if applicable.get("avoid_leak"):
        r = metrics_mod.privacy_leak_rate(agent_output, ground_truth.get("user_negatives_nearby") or [], threshold)
        out["avoid_leak_rate"] = r["leak_rate"]
        out["avoid_leak_hard_fail"] = r["hard_fail"]
    if applicable.get("privacy_leak"):
        r = metrics_mod.privacy_leak_rate(agent_output, ground_truth.get("privacy_flagged_prefs") or [], threshold)
        out["privacy_leak_rate"] = r["leak_rate"]
        out["privacy_leak_hard_fail"] = r["hard_fail"]
    if applicable.get("stale_preference_use"):
        r = metrics_mod.privacy_leak_rate(agent_output, ground_truth.get("update_history_contradictions") or [], threshold)
        out["stale_preference_use_rate"] = r["leak_rate"]
        out["stale_preference_use_hard_fail"] = r["hard_fail"]
    if applicable.get("telegraph_avoidance"):
        # Deterministic — runs the regex + verbatim-pref-insertion check
        # in `evaluation.llm_postprocess._validate_no_creepy_phrasing`.
        # No LLM needed; treat hard_fail like privacy_leak.
        from evaluation.judges import judge_telegraph_avoidance as _jta
        held_out = (ground_truth.get("held_out_preference")
                    or ground_truth.get("groundtruth_preference")
                    or ground_truth.get("target_pref"))
        ja = _jta(agent_output, held_out)
        out["telegraph_avoidance_score"] = ja["telegraph_avoidance"]
        out["telegraph_avoidance_hard_fail"] = 0 if ja["telegraph_avoidance"] >= 1.0 else 1
        if ja.get("telegraph_reason"):
            out["telegraph_avoidance_reason"] = ja["telegraph_reason"]

    # Judge dims — skip if no judge available.
    if judge_client:
        for dim in (
            "preference_alignment", "over_personalization", "subtle_personalization",
            "relationship_aware", "voice_match", "voice_self_consistency",
        ):
            if not applicable.get(dim):
                continue
            prompt = prompts_mod.judge_personalization_dim_prompt(dim, ground_truth, agent_output, task_id)
            try:
                resp = judge_client(prompt) if callable(judge_client) else judge_client.query_llm(prompt)
                parsed = extract_json_from_response(resp) or {}
                if dim in HARD_RULE_DIMS:
                    out[f"{dim}_judge_fail"] = int(parsed.get("fail", 0) or 0)
                else:
                    score_val = parsed.get("score")
                    if isinstance(score_val, (int, float)):
                        out[f"{dim}_score"] = float(score_val)
                    # voice_match returns 3 sub-scores plus the mean — surface
                    # them all for diagnostic visibility (helps diagnose whether
                    # a low score is identity / idiolect / audience).
                    if dim == "voice_match":
                        for sub in ("identity_coherence", "idiolect_fidelity", "audience_appropriateness"):
                            sub_val = parsed.get(sub)
                            if isinstance(sub_val, (int, float)):
                                out[f"voice_match_{sub}"] = float(sub_val)
            except Exception as exc:
                out[f"{dim}_judge_error"] = str(exc)

    # Source B — behavioral hit/miss for applicable tasks.
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

    # Compose a summary "personalization_pass" flag: no hard-rule fails.
    hard_fails = [k for k in out if k.endswith("_hard_fail") and out[k]]
    out["personalization_hard_fail_count"] = len(hard_fails)

    # Polarity-aware aggregation: emit positive_score / negative_penalty /
    # combined_personalization_score. Pre-existing per-dim keys are kept
    # untouched for backward compatibility.
    combine_dim_scores_with_polarity(out, applicable)

    return out
