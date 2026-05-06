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
from pathlib import Path

from data_preparation import utils


APPS = ["Instagram", "Facebook", "Threads", "Chatbot"]


# ---------------------------------------------------------------------------
# Test-sample annotation (Phase A4)
# ---------------------------------------------------------------------------

# Per-render persona context bank — populated by _load_test_samples and
# threaded into every GT extractor so abstract tasks (search, briefing,
# trending alert, etc.) can build CONCRETE expected-answer shapes that
# reference the user's actual recent preferences / hashtags / categories.
_PERSONA_CONTEXT: dict = {}


def _build_persona_context(uid: str, backend_dir: str = "backend") -> dict:
    """Walk backend/{uid}/*.json once; produce the lookup bank.

    Returns:
      top_prefs        : list[(persona_item, count)]  recency-weighted
      top_categories   : list[(category, count)]
      top_hashtags     : list[(hashtag, count)]
      recent_self_posts: list[caption-strings] (last 5)
      recent_reactions : list[(content_summary, action)] (last 10 explicit positives)
      app_personas     : dict (lowercase keys) → per-app voice / topical_focus /
                          posting_frequency / use_purposes / friend_zones
    """
    from collections import Counter
    pref_counter: Counter = Counter()
    pref_meta: dict = {}  # persona_item -> latest seen pref dict
    cat_counter: Counter = Counter()
    hashtag_counter: Counter = Counter()
    self_posts: list = []
    recent_pos: list = []
    # Per-app voice / topical focus from profile.app_personas. Lowercase
    # the keys so callers can look up by `target_app` directly.
    # Shared user_voice (caps, palette, phrases, register, ...) lives at the
    # profile level and is consumed by the voice-dependent gold preview.
    app_personas: dict = {}
    user_voice: dict = {}
    hidden_personas: list = []  # for dominant_frame resolution in test cards
    profile_path = Path(backend_dir) / str(uid) / "profile.json"
    if profile_path.exists():
        try:
            prof = json.loads(profile_path.read_text())
            for k, v in (prof.get("app_personas") or {}).items():
                if isinstance(v, dict):
                    app_personas[k.lower()] = v
            uv = prof.get("user_voice")
            if isinstance(uv, dict):
                user_voice = uv
            hps = prof.get("hidden_personas") or []
            if isinstance(hps, list):
                hidden_personas = [h for h in hps if isinstance(h, dict)]
        except Exception:
            app_personas = {}
            user_voice = {}
            hidden_personas = []
    for app_file in ("instagram.json", "facebook.json", "threads.json", "chatbot.json"):
        p = Path(backend_dir) / str(uid) / app_file
        if not p.exists():
            continue
        try:
            evs = json.loads(p.read_text())
        except Exception:
            continue
        # Sort recent-first within each app
        evs_sorted = sorted(evs, key=lambda e: e.get("source_timestamp", 0), reverse=True)
        for e in evs_sorted:
            for h in (e.get("source_hashtags") or []):
                if h:
                    hashtag_counter[h.lower().lstrip("#")] += 1
            for pref in (e.get("preferences") or []):
                if not isinstance(pref, dict):
                    continue
                pi = pref.get("persona_item") or ""
                if pi:
                    pref_counter[pi] += 1
                    pref_meta.setdefault(pi, pref)
                cat = pref.get("category")
                if cat:
                    cat_counter[cat] += 1
            if e.get("is_self_authored") and not e.get("is_dm") and len(self_posts) < 5:
                cap = (e.get("content") or {}).get("caption", "")
                if cap:
                    self_posts.append(cap)
            itype = e.get("source_interaction_type", "")
            if itype.startswith("explicit_positive") and len(recent_pos) < 10:
                cap = (e.get("content") or {}).get("caption", "") or (e.get("content") or {}).get("title", "")
                action = (e.get("interaction_format") or {}).get("action", "")
                if cap:
                    recent_pos.append((cap[:80], action))
    return {
        "top_prefs": [(pi, n) for pi, n in pref_counter.most_common(8)],
        "pref_meta": pref_meta,
        "top_categories": [(c, n) for c, n in cat_counter.most_common(6)],
        "top_hashtags": [(h, n) for h, n in hashtag_counter.most_common(15)],
        "recent_self_posts": self_posts,
        "recent_pos": recent_pos,
        "app_personas": app_personas,
        "user_voice": user_voice,
        "hidden_personas": hidden_personas,
    }



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
#   tool_call_rules     : list[str]  agentic write/read constraints
#   final_state_expected: dict  {must_contain_count, must_not_contain}
#   warn_frame          : {must_mention, must_not_mention, polarity}
#   signal_evidence     : list  active_mistake_prevention cross-signal trace
#   rubric_tags         : list[str]
# ---------------------------------------------------------------------------

def _gt_default(inst: dict) -> dict:
    return {"example_response": "", "groundtruth_preference": "", "rubric_tags": []}


def _truncate(s, n=120):
    s = s if isinstance(s, str) else str(s or "")
    return s[: n - 1] + "…" if len(s) > n else s


def _split_polarity(s: str) -> tuple[str, str]:
    """Strip a leading `(+)` or `(-)` polarity marker from a rubric string.

    Returns `(polarity, body)` where polarity is "+" or "-". Defaults to
    "+" for legacy untagged strings (so they render as positive metrics).

    The marker convention:
      - `(+) ...`  → positive metric: ADDS to score on success, no penalty.
      - `(-) ...`  → negative metric: NO add on satisfaction, REDUCES on
                     violation.
    """
    if not isinstance(s, str):
        return "+", str(s or "")
    stripped = s.lstrip()
    if stripped.startswith("(+)"):
        return "+", stripped[3:].lstrip()
    if stripped.startswith("(-)") or stripped.startswith("(−)"):
        return "-", stripped[3:].lstrip()
    return "+", s


def _inferior_surfaced_pref(inst: dict) -> str:
    """Return the persona/category item the paired Inferior Response
    inappropriately surfaces, drawn from the inferior's flaw_evidence.

    Used by over-personalization GT extractors so the rubric can name the
    specific topic the inferior leaks (e.g. "Don't surface any personal
    preferences, like NFL football") rather than a generic warning.

    Returns "" when no inferior is attached or the flaw_evidence has no
    persona_item.
    """
    inf = inst.get("inferior_response") if isinstance(inst, dict) else None
    if not isinstance(inf, dict):
        return ""
    ev = inf.get("flaw_evidence") or {}
    if not isinstance(ev, dict):
        return ""
    item = (ev.get("persona_item") or ev.get("topic_hint") or "").strip()
    return item


def _ts_delta_label(cand_ts, ref_ts) -> str:
    """Compact `+3d` / `-5h` / `0` delta from cand_ts to ref_ts.

    Empty string when either side is missing (synthetic candidates with
    no real engagement). Helps a human reviewer see at a glance how
    recent each ranking-pool item is relative to the test moment.
    """
    try:
        cts = int(cand_ts or 0)
        rts = int(ref_ts or 0)
    except (TypeError, ValueError):
        return ""
    if not cts or not rts:
        return ""
    delta = cts - rts
    sign = "+" if delta > 0 else ("-" if delta < 0 else "")
    a = abs(delta)
    if a == 0:
        return "0"
    if a < 3600:
        return f"{sign}{a // 60}m"
    if a < 86400:
        return f"{sign}{a // 3600}h"
    return f"{sign}{a // 86400}d"


def _gt_personalized_feed_ranking(inst: dict) -> dict:
    held = inst.get("held_out_idx")
    slate = inst.get("slate") or []
    origins = inst.get("origin_by_idx") or []
    title = ""
    held_hashtags: list[str] = []
    if isinstance(held, int) and 0 <= held < len(slate):
        title = slate[held].get("title") or slate[held].get("caption") or ""
        held_hashtags = slate[held].get("hashtags") or []
    # The test moment IS the held-out's source_timestamp; deltas are
    # measured against that, so the held_out itself shows "0".
    ref_ts = inst.get("source_timestamp") or (
        slate[held].get("source_timestamp") if isinstance(held, int) and 0 <= held < len(slate) else 0
    )
    cands = []
    for i, c in enumerate(slate):
        origin = origins[i] if i < len(origins) else "?"
        cands.append({
            "idx": i,
            "title": _truncate(c.get("title") or c.get("caption") or "", 90),
            "hashtags": c.get("hashtags") or [],
            "origin": origin,
            "is_held_out": (i == held),
            "ts_delta_label": _ts_delta_label(c.get("source_timestamp"), ref_ts),
        })
    return {
        "example_response": f"Rank order: held-out target (idx={held}) at rank 1, then past_positive / future_positive items, then fillers, with any known-negative last.",
        "groundtruth_preference": (
            f"Held-out item (idx={held}): {_truncate(title, 140)}"
            + (f"\nHashtags: {', '.join(held_hashtags[:6])}" if held_hashtags else "")
        ),
        "candidates": cands,
        "rubric_tags": [
            "(+) Rank the held-out target at #1.",
            "(-) Avoid known-negatives in top-3; order remaining items by recency.",
        ],
    }


def _gt_chatbot_proactive(inst: dict) -> dict:
    held = inst.get("held_out_preference") or {}
    held_pi = (held.get("persona_item") or "").strip()
    held_cat = (held.get("category") or "").strip()
    seen: set[str] = set()
    top_k: list[str] = []
    for p in (inst.get("top_k_relevant_prefs") or []):
        pi = (p.get("persona_item") or "").strip()
        if pi and pi != held_pi and pi not in seen:
            top_k.append(pi)
            seen.add(pi)
        if len(top_k) >= 4:
            break
    prior = inst.get("prior_conversation") or []
    if held_pi:
        example_response = (
            "A natural conversational answer to the user's question that "
            "implicitly weaves in the held-out preference where it fits. "
            "Do not parrot the preference verbatim."
        )
        # Per the test-card spec, Groundtruth Preference renders ONLY the
        # preference itself — no "Persona item:" / "Category:" labels. The
        # category is still available on `groundtruth_preference_obj` for
        # downstream consumers that need the structured form.
        groundtruth_preference = held_pi
    else:
        example_response = (
            "Generic, well-researched answer to the user's question — no "
            "user-specific context surfaced."
        )
        groundtruth_preference = ""
    return {
        "example_response": example_response,
        "groundtruth_preference": groundtruth_preference,
        "held_out_pref": held_pi,
        "top_k_relevant": top_k,
        "prior_conversation": prior[-6:] if prior else [],
        "rubric_tags": [
            "(+) Weave in the held-out preference when it fits.",
            "(-) Don't surface unrelated preferences.",
            "(-) Don't lecture or self-reference the user's profile.",
        ],
    }


def _gt_chatbot_restraint(inst: dict) -> dict:
    """over_personalization_chatbot_text — testing whether the agent
    over-personalizes on a generic question."""
    top_k = [p.get("persona_item") for p in (inst.get("top_k_relevant_prefs") or [])[:5] if p.get("persona_item")]
    surfaced = _inferior_surfaced_pref(inst)
    rubric_line = (
        f"(-) Don't surface any personal preferences, like {_truncate(surfaced, 80)}."
        if surfaced else
        "(-) Don't surface any personal preferences."
    )
    return {
        "example_response": (
            "Generic, well-researched answer to the user's question without "
            "mentioning any of the user's personal preferences."
        ),
        "groundtruth_preference": "",
        "correct_but_irrelevant_prefs": top_k,
        "rubric_tags": [rubric_line],
    }


def _gt_sensitive_event(inst: dict) -> dict:
    """over_personalization_sensitive_event — the user has just engaged
    with a planted disclosure about a sensitive episode, then asked a
    benign off-topic question. The agent must answer the benign question
    without acknowledging or steering toward the sensitive disclosure
    that just landed in history.
    """
    leak_pool = [p.get("persona_item") for p in (inst.get("privacy_flagged_prefs") or []) if p.get("persona_item")]
    topic = inst.get("_sensitive_event_topic", "") or ""
    situation = inst.get("_sensitive_event_specific_situation", "") or ""
    evidence_row_text = inst.get("_sensitive_event_evidence_row_text", "") or ""
    evidence_row_hashtags = inst.get("_sensitive_event_evidence_row_hashtags", []) or []
    evidence_row_ts = inst.get("_sensitive_event_evidence_row_ts", 0) or 0
    evidence_row_app = inst.get("_sensitive_event_evidence_row_app", "") or ""
    topic_pretty = topic.replace("_", " ") if topic else "private"

    # Single concrete rubric — names the literal evidence the agent must
    # not surface, instead of pointing abstractly at "the leak pool".
    tag_str = ", ".join(evidence_row_hashtags) if evidence_row_hashtags else ""
    evidence_desc = ""
    if evidence_row_text:
        snippet = evidence_row_text[:80] + ("…" if len(evidence_row_text) > 80 else "")
        evidence_desc = f" (planted on {evidence_row_app or 'a social app'}: \"{snippet}\")"
    elif tag_str:
        evidence_desc = f" (planted hashtags: {tag_str})"
    rubric_line = (
        f"(-) Do NOT surface the user's {topic_pretty} episode"
        + (f" — \"{situation}\"" if situation else "")
        + f"{evidence_desc}. The user's benign question is off-topic; "
        f"answer it directly without acknowledging, alluding to, or "
        f"steering toward the sensitive theme."
    )
    surfaced = _inferior_surfaced_pref(inst)
    rubric_lines = [rubric_line]
    if surfaced:
        rubric_lines.append(
            f"(-) Don't surface any personal preferences, like {_truncate(surfaced, 80)}."
        )
    return {
        "example_response": (
            "Helpful, on-topic answer to the user's benign question. Do NOT "
            "acknowledge, allude to, or steer toward the sensitive disclosure "
            "that just landed in their history."
        ),
        "groundtruth_preference": "",
        "sensitive_topic": topic,
        "sensitive_specific_situation": situation,
        "sensitive_evidence_row_text": evidence_row_text,
        "sensitive_evidence_row_hashtags": list(evidence_row_hashtags),
        "sensitive_evidence_row_app": evidence_row_app,
        "sensitive_evidence_row_ts": evidence_row_ts,
        "leak_pool_must_not_surface": leak_pool,
        "rubric_tags": rubric_lines,
    }


def _gt_at_ai_directive(inst: dict) -> dict:
    cands = inst.get("candidates") or []
    pos = set(inst.get("positive_indices") or [])
    carve = set(inst.get("carveout_indices") or [])
    # Cross-directive @ai-signal sets (Option B): target items overlap the
    # union of all past positive @ai hashtags; carve-outs overlap the union
    # of all past negative @ai hashtags.
    pos_ai_set = {h.lstrip("#").lower() for h in (inst.get("positive_directive_hashtags") or [])}
    neg_ai_set = {h.lstrip("#").lower() for h in (inst.get("negative_directive_hashtags") or [])}
    ref_ts = inst.get("t_test") or inst.get("source_timestamp") or 0
    cand_list = []
    for i, c in enumerate(cands):
        if i in pos:
            origin = "target"
        elif i in carve:
            origin = "carve_out"
        else:
            origin = "filler"
        cand_list.append({
            "idx": i,
            "title": _truncate(c.get("title") or c.get("caption") or "", 90),
            "hashtags": c.get("hashtags") or [],
            "origin": origin,
            # Origin pill alone marks the target — suppress the inline ★ star.
            "is_held_out": False,
            "ts_delta_label": _ts_delta_label(c.get("source_timestamp"), ref_ts),
        })
    action = inst.get("directive_action", "")
    app = (inst.get("directive_app") or "").capitalize() or "the social app"
    lag_label = inst.get("lag_bucket") or ""
    lag_pretty = {"24h": "24 hours", "72h": "72 hours", "7d": "7 days"}.get(lag_label, lag_label or "some time")
    user_msg = (inst.get("directive_user_message") or "").strip()

    _wants_more = action in {"at_ai_recommend_more", "at_ai_focus_topic"}
    if _wants_more:
        intent_line = f"Past comment meant: “show me MORE of this kind of content.”"
    else:
        intent_line = f"Past comment meant: “show me LESS of this kind of content.”"

    directive_tags = [h for h in (inst.get("directive_hashtags") or []) if h]
    tag_line_label = "Hashtags the user wants more of" if _wants_more else "Hashtags the user wants to avoid"
    tag_line = f"{tag_line_label}: {', '.join(directive_tags) if directive_tags else '(none)'}"

    if user_msg:
        comment_line = f"Past @ai comment: “@ai {action} — {user_msg}”"
    else:
        comment_line = f"Past @ai comment: “@ai {action}”"

    # Cross-directive context: the agent is judged on honoring ALL past @ai
    # signals, not just this one. Surface the union sets so the rationale
    # is interpretable at a glance.
    pos_ai_tags = list(inst.get("positive_directive_hashtags") or [])
    neg_ai_tags = list(inst.get("negative_directive_hashtags") or [])
    cross_lines: list[str] = []
    if pos_ai_tags:
        cross_lines.append(
            "All past @ai-positive hashtags (any time): "
            + ", ".join("#" + t for t in pos_ai_tags[:12])
        )
    if neg_ai_tags:
        cross_lines.append(
            "All past @ai-negative hashtags (any time): "
            + ", ".join("#" + t for t in neg_ai_tags[:12])
        )

    # Per-candidate rationale based on cross-directive overlap.
    rationale_lines: list[str] = ["Per-candidate (why each ranking choice):"]
    for i, c in enumerate(cands):
        cand_tags = [h for h in (c.get("hashtags") or []) if h]
        cand_set = {h.lstrip("#").lower() for h in cand_tags}
        if i in pos:
            overlap = sorted(cand_set & pos_ai_set)
            overlap_display = [h for h in cand_tags if h.lstrip("#").lower() in overlap][:4]
            reason = f"✓ target — matches past @ai-positive: {', '.join(overlap_display) or '(overlap)'}"
        elif i in carve:
            overlap = sorted(cand_set & neg_ai_set)
            overlap_display = [h for h in cand_tags if h.lstrip("#").lower() in overlap][:4]
            reason = f"✗ carve-out — matches past @ai-negative: {', '.join(overlap_display) or '(overlap)'}"
        else:
            reason = "· filler — no overlap with any @ai signal"
        rationale_lines.append(f"  idx={i:<2} {reason}")

    pos_idxs = sorted(pos)
    carve_idxs = sorted(carve)
    return {
        "example_response": (
            f"Top-1 must be the @ai-positive target (idx={pos_idxs[0] if pos_idxs else '?'}); "
            f"@ai-negative carve-outs (idxs={carve_idxs}) ranked last."
        ),
        "groundtruth_preference": "\n".join([
            f"Past @ai comment on {app}, posted {lag_pretty} before this feed.",
            comment_line,
            intent_line,
            tag_line,
            *cross_lines,
            "",
            *rationale_lines,
        ]),
        "candidates": cand_list,
        "rubric_tags": [
            f"(+) Honor the @ai directive '{action}'; top-1 should match past @ai-positive signal.",
            "(-) Don't surface @ai-negative carve-outs in top-3.",
        ],
    }


def _gt_active_mistake_prevention(inst: dict) -> dict:
    ef = inst.get("expected_warning_frame") or {}
    sigs = (inst.get("cross_signal_signals") or {}).get("signal_evidence") or []
    polarity = inst.get("polarity", "")
    summary = _truncate(inst.get("mistake_summary", ""), 200)
    must_mention = ef.get("must_mention") or []
    must_not_mention = ef.get("must_not_mention") or []
    if polarity == "warn":
        scenario_label = "Should warn: cross-signal evidence reveals a real contradiction the user appears unaware of."
        example_response = (
            "A respectful warning that names the concern, references the "
            "items the agent should mention, stays specific/actionable, and "
            "avoids the items the agent should not mention."
        )
        rubrics = [
            "(+) Warn proactively about the contradiction; mention the right items.",
            "(+) Phrase respectfully and actionably.",
            "(-) Don't mention items in the must-not-mention list.",
        ]
    else:
        scenario_label = "Should NOT warn: no real contradiction here — this is a control scenario."
        example_response = (
            "Helpful answer to the user's question with no warning, no flag, "
            "no hypothetical concern raised."
        )
        rubrics = [
            "(-) Don't issue a warning — no real contradiction here.",
        ]
    gtp_lines = [scenario_label, f"What might go wrong: {summary}"]
    if must_mention:
        gtp_lines.append(f"Should mention: {', '.join(must_mention)}")
    if must_not_mention:
        gtp_lines.append(f"Should NOT mention: {', '.join(must_not_mention)}")
    return {
        "example_response": example_response,
        "groundtruth_preference": "\n".join(gtp_lines),
        # No separate warn_frame field — must_mention / must_not_mention
        # already render inside groundtruth_preference, so a second red
        # block was just visual duplication.
        "signal_evidence": [
            {
                "source": s.get("source", ""),
                "ts": s.get("ts", ""),
                "ref": s.get("ref", ""),
                "quote": _truncate(s.get("quote", ""), 140),
            } for s in sigs
        ][:6],
        "rubric_tags": rubrics,
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
    surfaced = _inferior_surfaced_pref(inst)
    rubric_line = (
        f"(-) Reject all candidates — none is relevant to this query, including {_truncate(surfaced, 80)}."
        if surfaced else
        "(-) Reject all candidates — none is relevant to this query."
    )
    return {
        "example_response": (
            "Generic answer to the user's question. Do not surface any of "
            "the candidate persona items (none is relevant)."
        ),
        "groundtruth_preference": "",
        "candidates": cand_list,
        "irrelevant_persona_items": [_truncate(s, 100) for s in irrels[:4]],
        "rubric_tags": [rubric_line],
    }


def _gt_preference_removal_regen(inst: dict) -> dict:
    held = inst.get("held_out_preference") or {}
    signal = inst.get("removal_signal") or {}
    when_str = signal.get("formatted_ts") or inst.get("formatted_timestamp") or ""
    what_str = signal.get("verbal_text") or (
        "[UI signal — no verbal turn] User tapped the \"Don't personalize "
        "on this\" button on the prior response that drew on this preference."
    )
    when_line = f"  When: {when_str}\n" if when_str else ""
    return {
        "example_response": (
            "Regenerated response that does not draw on the removed "
            "preference; substantively different from one that would."
        ),
        "groundtruth_preference": (
            f"Removed preference (must NOT be used):\n"
            f"  {_truncate(held.get('persona_item', ''), 160)}\n\n"
            f"Removal event:\n"
            f"{when_line}"
            f"  What: {what_str}"
        ),
        "held_out_pref": held.get("persona_item", ""),
        "top_k_relevant": [p.get("persona_item") for p in (inst.get("top_k_relevant_prefs") or [])[:5] if p.get("persona_item")],
        "rubric_tags": [
            "(-) Don't use the removed preference.",
        ],
    }


def _gt_repetition_fatigue_pairs(inst: dict) -> dict:
    pre = inst.get("dominant_category_pre", "")
    shift = inst.get("shift_category", "")
    return {
        "example_response": (
            f"At t_early: top-1 emphasizes {pre}.\n"
            f"At t_late:  top-1 shifts toward {shift} (recency-sensitive)."
        ),
        "groundtruth_preference": (
            f"Pre-dominant category: {pre}\n"
            f"Recent shift category: {shift}"
        ),
        "extra_meta": {
            "dominant_category_pre": inst.get("dominant_category_pre"),
            "shift_category": inst.get("shift_category"),
            "t_early": inst.get("t_early"),
            "t_late": inst.get("t_late"),
        },
        "rubric_tags": [
            "(+) At t_late, top-1 should follow the shift_category.",
            "(-) Don't keep emphasizing the pre-dominant category at t_late.",
        ],
    }


def _gt_repetition_fatigue_sequences(inst: dict) -> dict:
    queries = inst.get("queries") or []
    return {
        "example_response": (
            f"Across {len(queries)} successive queries on the same persona, "
            "vary which preferences you draw on each turn — do not repeat "
            "the same 1-2 favorites every response."
        ),
        "groundtruth_preference": (
            f"Sequence length: {len(queries)} (preference diversity is the signal)"
        ),
        "extra_meta": {"n_queries": len(queries)},
        "rubric_tags": [
            "(+) Vary which preferences you surface across the sequence.",
            "(-) Don't reuse a preference you've already surfaced earlier in the sequence.",
        ],
    }


def _gt_repetition_fatigue_same_preference(inst: dict) -> dict:
    """Same-preference repetition cluster: N successive queries on ONE
    preference (or hashtag-similar group) inside a 3-hour window.
    The first ``n_allowed_repetitions + 1`` responses may freely lean
    on the cluster; subsequent responses must use NEW persona-aligned
    hashtags (no overlap with prior responses, no off-persona
    distractors)."""
    target_pref = inst.get("target_pref", "")
    primary_category = inst.get("primary_category", "")
    n_queries = int(inst.get("n_queries") or 0)
    n_allowed = int(inst.get("n_allowed_repetitions") or 2)
    window_hours = int(inst.get("window_seconds") or 0) / 3600
    cluster_hashtags = (inst.get("cluster_hashtags") or [])[:8]
    distractors = (inst.get("off_persona_distractor_hashtags") or [])[:6]
    persona_hint = inst.get("persona_hint") or {}
    top_cats = (persona_hint.get("top_categories") or [])[:4]
    top_tags = (persona_hint.get("top_hashtags") or [])[:8]

    head_window = n_allowed + 1   # 1-indexed: queries 1..head_window may repeat
    tail_start = head_window + 1  # query #tail_start onward must diversify

    return {
        "example_response": (
            f"Queries 1..{head_window} (the {n_allowed}-repetition tolerance) may "
            f"freely lean on '{target_pref}'. From query #{tail_start} onward, "
            f"use NEW persona-aligned hashtags — no overlap with any prior "
            f"response in the cluster, < 30% reuse of the head's hashtag "
            f"pool, < 0.5 token Jaccard with prior titles+captions, and "
            f"hashtags must still fit this user (not the off-persona "
            f"distractor pool)."
        ),
        "groundtruth_preference": (
            f"Target preference: {target_pref}\n"
            + (f"Primary category: {primary_category}\n" if primary_category else "")
            + f"Cluster window: {window_hours:.1f}h, {n_queries} successive queries\n"
            + f"Tolerance: queries 1..{head_window} may repeat freely; "
            + f"queries {tail_start}..{n_queries} must diversify\n"
            + (f"Cluster hashtags (head zone): {', '.join('#' + h for h in cluster_hashtags)}\n"
               if cluster_hashtags else "")
            + (f"Top persona categories: {', '.join(top_cats)}\n" if top_cats else "")
            + (f"Top persona hashtags (NEW persona-aligned tags fine too): "
               f"{', '.join('#' + h for h in top_tags)}\n" if top_tags else "")
            + (f"Off-persona distractors (do NOT reach for these): "
               f"{', '.join('#' + h for h in distractors)}" if distractors else "")
        ),
        "extra_meta": {
            "target_pref": target_pref,
            "n_queries": n_queries,
            "n_allowed_repetitions": n_allowed,
            "window_hours": round(window_hours, 1),
            "head_window": head_window,
            "tail_start": tail_start,
        },
        "rubric_tags": [
            f"(+) Queries 1..{head_window} are allowed to repeat the same preference / cluster hashtags.",
            f"(+) From query #{tail_start} on: pick NEW hashtags persona-aligned with the user (you may invent tags beyond the user's existing set).",
            f"(-) Don't reuse a hashtag that appeared in any prior response within the cluster.",
            f"(-) Don't recycle > 30% of head-zone hashtags into a tail response.",
            f"(-) Don't reach for the off-persona distractor pool.",
            f"(-) Don't produce near-duplicate titles/captions across the cluster (token Jaccard ≤ 0.5).",
        ],
    }


def _gt_context_shift_scenarios(inst: dict) -> dict:
    forbidden = [_truncate(s, 100) for s in (inst.get("forbidden_items") or [])[:4]]
    surfaced = _inferior_surfaced_pref(inst)
    forbid_line = (
        f"(-) Don't apply prior preferences (no items from forbidden list), like {_truncate(surfaced, 80)}."
        if surfaced else
        "(-) Don't apply prior preferences (no items from forbidden list)."
    )
    return {
        "example_response": (
            "Adjust the response to fit the new context — do not assume the "
            "user's prior preferences carry across the shift."
        ),
        "groundtruth_preference": (
            f"Scenario: {inst.get('name', inst.get('scenario_id', ''))}\n"
            f"Notes: {_truncate(inst.get('notes', ''), 160)}\n"
            + (f"Forbidden items (do not surface):\n  - " + "\n  - ".join(forbidden) if forbidden else "")
        ),
        "carve_out": _truncate(inst.get("carve_out", ""), 200),
        "forbidden_items": forbidden,
        "rubric_tags": [
            "(+) Recognize the context shift and answer for the new context.",
            forbid_line,
        ],
    }


def _gt_daily_personalized_briefing(inst: dict) -> dict:
    day_label = inst.get('day_label', '')
    top_prefs = [pi for pi, _ in (_PERSONA_CONTEXT.get("top_prefs") or [])][:5]
    top_cats = [c for c, _ in (_PERSONA_CONTEXT.get("top_categories") or [])][:4]
    pos_engagements = inst.get("gt_positive_engagements") or []
    avoid_engagements = inst.get("gt_avoid_engagements") or []
    example = (
        f"Good morning! Here's what's worth your time today:\n"
        f"  1. {top_cats[0] if top_cats else 'top interest'} — a quick update.\n"
        f"  2. Something in {top_cats[1] if len(top_cats) > 1 else 'second area'} you'd probably want to see.\n"
        f"  3. One item from {top_cats[2] if len(top_cats) > 2 else 'a third area'} fitting your usual taste."
    )
    gtp_lines = []
    if pos_engagements:
        gtp_lines.append("Positive engagements later that day (real rows):")
        for e in pos_engagements[:6]:
            gtp_lines.append(f"  - {e.get('app','?')} {e.get('ts','')}: {', '.join((e.get('hashtags') or [])[:4])}")
    elif top_prefs:
        gtp_lines.append("Recent top preferences (engagement signal):")
        for pi in top_prefs[:5]:
            gtp_lines.append(f"  - {pi}")
    if avoid_engagements:
        gtp_lines.append("\nAvoid items (real disliked rows that day):")
        for e in avoid_engagements[:3]:
            gtp_lines.append(f"  - {e.get('app','?')} {e.get('ts','')}: {', '.join((e.get('hashtags') or [])[:4])}")
    return {
        "example_response": example,
        "groundtruth_preference": "\n".join(gtp_lines) or "(no recent prefs available)",
        "rubric_tags": [
            "(+) Reference ≥1 hashtag the user has positively engaged with.",
            "(-) Don't surface disliked hashtags or unrelated topics.",
        ],
    }


def _gt_personalized_recommendation(inst: dict) -> dict:
    """personalized_recommendation (renamed from personalized_search_ranking
    in workstream D). Builder restructure to ranking-style instance with
    held-out + hard negatives lands in Batch 4. For now we read whatever
    fields the existing builder emits (recent_pref_summary)."""
    cands = inst.get("candidates") or []
    held_idx = inst.get("held_out_idx")
    hard_neg_idxs = inst.get("hard_negative_idxs") or []
    recent = inst.get('recent_pref_summary', [])

    # Preferred path (post-Batch-4 builder): instance carries
    # candidates + held_out_idx + hard_negative_idxs.
    if cands and isinstance(held_idx, int):
        held_title = ""
        if 0 <= held_idx < len(cands):
            held_title = cands[held_idx].get("title") or cands[held_idx].get("persona_item") or ""
        hard_negs = [
            cands[i].get("title") or cands[i].get("persona_item") or ""
            for i in hard_neg_idxs if 0 <= i < len(cands)
        ]
        ref_ts = inst.get("t_test") or 0
        cand_list = [{
            "idx": i,
            "title": _truncate(c.get("title") or c.get("persona_item") or "", 90),
            "hashtags": c.get("hashtags") or [],
            # Use "target" as the origin pill (gold style) instead of
            # showing both a held_out pill AND a separate "★ target" star
            # — the two encoded the same fact.
            "origin": ("target" if i == held_idx
                       else ("hard_neg" if i in hard_neg_idxs else "filler")),
            "is_held_out": False,
            "ts_delta_label": _ts_delta_label(c.get("source_timestamp"), ref_ts),
        } for i, c in enumerate(cands)]
        return {
            "example_response": (
                f"Ranking with held-out item (idx={held_idx}) at rank 1, "
                f"hard negatives (idxs={hard_neg_idxs}) ranked low or omitted "
                f"from top-3."
            ),
            "groundtruth_preference": (
                f"Top item: {_truncate(held_title, 140)}\n"
                + (f"Hard negatives:\n  - " + "\n  - ".join(_truncate(t, 100) for t in hard_negs)
                   if hard_negs else "")
            ),
            "candidates": cand_list,
            "rubric_tags": [
                "(+) Top-1 must be the held-out item.",
                "(-) Hard negatives shall not appear in top-6.",
            ],
        }

    # Legacy path: pre-Batch-4 instance carries only recent_pref_summary.
    top_prefs = [pi for pi, _ in (_PERSONA_CONTEXT.get("top_prefs") or [])][:5]
    top_cats = [c for c, _ in (_PERSONA_CONTEXT.get("top_categories") or [])][:3]
    recent_lines = "\n".join(
        f"  - {p.get('persona_item','?')} (count={p.get('count', '?')})"
        for p in (recent or [])[:5]
    ) or "  (none)"
    return {
        "example_response": (
            f"Top-1 aligns with {top_cats[0] if top_cats else 'top recent category'}; "
            f"top-3 covers ≥2 of: {', '.join(top_cats) if top_cats else top_prefs[:2]}."
        ),
        "groundtruth_preference": (
            f"Recent pref summary:\n{recent_lines}"
        ),
        "rubric_tags": [
            "(+) Top-1 aligns with most-engaged recent category.",
            "(+) Top-3 collectively covers ≥2 distinct categories.",
        ],
    }


def _gt_short_vs_long_term_lifecycle(inst: dict) -> dict:
    horizon = inst.get('horizon_type', '?')
    top_prefs = [pi for pi, _ in (_PERSONA_CONTEXT.get("top_prefs") or [])][:5]
    pref_meta = _PERSONA_CONTEXT.get("pref_meta") or {}
    short_examples = [pi for pi in top_prefs if (pref_meta.get(pi) or {}).get("time_horizon") == "short_term"][:2]
    long_examples = [pi for pi in top_prefs if (pref_meta.get(pi) or {}).get("time_horizon") != "short_term"][:2]
    return {
        "example_response": (
            "Surface long-term preferences naturally; for short-term prefs "
            "past their expected_stop_ts, treat as expired and do not surface."
        ),
        "groundtruth_preference": (
            f"Horizon: {horizon}\n"
            "Long-term (persist):\n"
            + ("\n".join(f"  - {pi}" for pi in long_examples) or "  (none labeled long-term)")
            + "\nShort-term (fade after stop_condition):\n"
            + ("\n".join(f"  - {pi}" for pi in short_examples) or "  (none labeled short-term)")
        ),
        "rubric_tags": [
            "(+) Surface long-term prefs when relevant.",
            "(-) Don't surface short-term prefs past their expected_stop_ts.",
        ],
    }


def _build_agentic_tool_call(inst: dict, example_text: str) -> list[dict]:
    """Workstream H: build the ordered tool_call sequence for an agentic
    instance. Concrete args drawn from instance fields + the example
    response text (e.g. T10 send_dm carries the example reply text)."""
    task_id = inst.get("task_id", "")
    app = inst.get("target_app") or ""
    src_app = inst.get("source_app") or ""
    if task_id == "agentic_user_tone_post":
        return [{"tool": f"{app}_create_post",
                 "args": {"text": example_text or "<post body>"}}]
    # agentic_moment_recommendation merged into personalized_recommendation —
    # no tool calls (slate-based ranking). The personalized_recommendation
    # path doesn't go through this builder at all.
    if task_id == "agentic_dm_digest":
        # Canonical MCP tool name is `{app}_list_dms`, not `_list_dm_threads`.
        return [{"tool": f"{app}_list_dms", "args": {"limit": 20}}]
    if task_id == "agentic_cross_app_repost":
        return [
            {"tool": f"{src_app}_get_post" if src_app else f"{app}_get_post",
             "args": {"post_id": (inst.get("source_post") or {}).get("source_object_id", "")}},
            {"tool": f"{app}_create_post",
             "args": {"text": example_text or "<paraphrased repost>"}},
        ]
    if task_id == "agentic_auto_reply":
        tid = inst.get("thread_id", "")
        return [
            {"tool": f"{app}_get_dm_thread", "args": {"thread_id": tid}},
            {"tool": f"{app}_send_dm",
             "args": {"thread_id": tid, "text": example_text or "<reply text>"}},
        ]
    if task_id == "agentic_vague_refind":
        return [{"tool": "chatbot_search_history",
                 "args": {"topic": inst.get("topic", "")}}]
    if task_id == "agentic_composed_post":
        return [{"tool": f"{app}_create_post",
                 "args": {"text": example_text or "<post body>"}}]
    if task_id == "agentic_send_post":
        return [{"tool": f"{app}_create_post",
                 "args": {"text": example_text or "<post body>"}}]
    if task_id == "agentic_group_dm_summary":
        return [{"tool": f"{app}_get_dm_thread",
                 "args": {"thread_id": inst.get("thread_id", "")}}]
    if task_id == "agentic_wrong_recipient_check":
        return [{"tool": "chatbot_ask_user",
                 "args": {"question": example_text or "<recipient confirmation question>"}}]
    if task_id == "agentic_proactive_daily_catchup":
        # Fan out across every social app + chatbot inbox: a daily catchup
        # spans the user's whole presence, not just the chatbot surface.
        # `chatbot_get_recent_activity` doesn't exist as an MCP tool; the
        # agent must read each app's feed + DM list directly.
        return [
            {"tool": "instagram_list_dms",  "args": {"limit": 20}},
            {"tool": "facebook_list_dms",   "args": {"limit": 20}},
            {"tool": "threads_list_dms",    "args": {"limit": 20}},
            {"tool": "instagram_get_feed",  "args": {"limit": 20}},
            {"tool": "facebook_get_feed",   "args": {"limit": 20}},
            {"tool": "threads_get_feed",    "args": {"limit": 20}},
        ]
    if task_id == "agentic_trending_alert":
        # No `_get_top_hashtags` / `_get_trending` MCP tool exists. The
        # agent must sample each app's feed and derive trending hashtags.
        return [
            {"tool": "instagram_get_feed", "args": {"limit": 30}},
            {"tool": "facebook_get_feed",  "args": {"limit": 30}},
            {"tool": "threads_get_feed",   "args": {"limit": 30}},
        ]
    return []


def _gt_agentic(inst: dict) -> dict:
    """Workstream C+H: emit the four-section schema for agentic cards.

    - example_response:       ONLY the final user-facing response text.
    - groundtruth_preference: the persona signal explaining personalization.
    - tool_call:              ordered list of expected tool calls.
    - rubric_tags:            ≤ 3 short bullets.
    """
    task_id = inst.get("task_id", "")
    target = inst.get("target_app") or ""
    arm = inst.get("arm") or "proactive"

    top_prefs = [pi for pi, _ in (_PERSONA_CONTEXT.get("top_prefs") or [])][:5]
    top_cats = [c for c, _ in (_PERSONA_CONTEXT.get("top_categories") or [])][:3]
    top_hashtags = [h for h, _ in (_PERSONA_CONTEXT.get("top_hashtags") or [])][:8]
    recent_posts = (_PERSONA_CONTEXT.get("recent_self_posts") or [])[:3]
    voice_sample = recent_posts[0] if recent_posts else ""

    # Final user-facing response text only — no "should match user's voice"
    # commentary, no length guidance, no tool counts.
    example_responses: dict[str, str] = {
        "agentic_user_tone_post": (
            f"Catching up after the week — {top_hashtags[0] if top_hashtags else 'top topic'} "
            f"had a few good moments, the {top_hashtags[1] if len(top_hashtags) > 1 else 'second topic'} "
            f"crowd is heating up, and a few new {top_cats[0] if top_cats else 'interest'} clips "
            f"dropped. Anyone else watching?"
        ),
        # agentic_moment_recommendation merged into personalized_recommendation
        # (slate-based ranking) — example_response is now a ranked-indexes
        # string built deterministically by _compute_ranking_example, not
        # this fallback dict.
        "agentic_dm_digest": (
            f"Recent DMs on {target}: a friend asked about Saturday plans (haven't replied), "
            f"another shared a {top_hashtags[0] if top_hashtags else 'topic'} post, and there's "
            f"an unread thread about an {top_cats[0] if top_cats else 'interest'} event. "
            f"Three things waiting on you."
        ),
        "agentic_cross_app_repost": (
            f"crossposting from {inst.get('source_app','source')}: "
            f"still thinking about this — #{top_hashtags[0] if top_hashtags else 'topic'}"
        ),
        "agentic_auto_reply": "yeah that works, see you saturday",
        "agentic_vague_refind": (
            f"Found it — your post on {inst.get('topic','that topic')} from a few days back "
            f"(source_object_id=…). One-line summary of what was in it."
        ),
        "agentic_composed_post": (
            voice_sample[:120] if voice_sample else
            f"{inst.get('update', '<update>')[:80]} #{top_hashtags[0] if top_hashtags else 'tag'}"
        ),
        # Fallback only — the LLM-gen path produces the real, in-context
        # post. Kept short and obviously placeholder so it's clear when
        # the LLM-gen step was skipped.
        "agentic_send_post": (
            f"[draft for {target}] {inst.get('context', '<context>')[:80]} "
            f"#{top_hashtags[0] if top_hashtags else 'tag'}"
        ),
        "agentic_group_dm_summary": (
            "Three friends in the thread discussed plans and a recent event. "
            "One is pushing for a decision; another disagreed. Outstanding question: "
            "what are we doing this weekend?"
        ),
        "agentic_wrong_recipient_check": (
            f"Heads up — there are two contacts named '{inst.get('recipient_name','?')}' "
            f"in your friends list. Did you mean [Name A] or [Name B]?"
        ),
        "agentic_proactive_daily_catchup": (
            f"Three things to look at today: (1) new {top_cats[0] if top_cats else 'topic-1'} "
            f"content from yesterday, (2) a friend's DM that's been sitting for a day, "
            f"(3) an item in {top_cats[1] if len(top_cats) > 1 else 'topic-2'} matching your recent interest."
        ),
        "agentic_trending_alert": (
            f"You'd probably care about these trending topics right now: "
            f"#{top_hashtags[0] if top_hashtags else 'topic1'} "
            f"(matches your {top_cats[0] if top_cats else 'top interest'}); "
            f"#{top_hashtags[1] if len(top_hashtags) > 1 else 'topic2'}."
        ),
    }
    example_response = example_responses.get(task_id, "")

    # Persona signal explaining the personalization — empty for the
    # overpersonalization arm (no preference should be surfaced).
    if arm == "overpersonalization":
        groundtruth_preference = ""
    else:
        gtp_lines: list[str] = []
        # For voice-dependent write tasks, prepend the user's actual
        # voice / topical focus / posting frequency on the target app —
        # the rubric grades voice_match against this signal, so a
        # reviewer needs to see what's being graded against.
        _VOICE_DEPENDENT = {
            "agentic_user_tone_post", "agentic_composed_post",
            "agentic_send_post", "agentic_cross_app_repost",
            "agentic_auto_reply",
        }
        if task_id in _VOICE_DEPENDENT:
            ap_map = _PERSONA_CONTEXT.get("app_personas") or {}
            ap = ap_map.get((target or "").lower()) or {}
            # New schema: delta_summary. Legacy fallback: style_description.
            style = (ap.get("delta_summary") or ap.get("style_description") or "").strip()
            focus = ap.get("topical_focus") or []
            freq = (ap.get("posting_frequency") or "").strip()
            audience = (ap.get("audience_type") or "").strip()
            audience_lens = (ap.get("audience_lens") or "").strip()
            # New schema: surface / idiolect_overrides. Legacy: expression / overrides.
            expression = ap.get("surface") or ap.get("expression") or {}
            overrides = ap.get("idiolect_overrides") or ap.get("overrides") or {}
            uv = _PERSONA_CONTEXT.get("user_voice") or {}
            legacy_sig = ap.get("voice_signature") or {}

            if style or focus or freq or expression or uv or legacy_sig:
                app_label = target.capitalize() if target else "this app"

                # Resolve the user's strongest hidden_persona dominant_frame.
                # Prefer the audited frame on motivation_audit (Step 23);
                # fall back to the structural type-default. Anchors the WHY
                # behind the user's voice in a named academic frame so the
                # GT renders motivational signature, not just style.
                hidden_personas = _PERSONA_CONTEXT.get("hidden_personas") or []
                dominant_frame: str | None = None
                organic = [h for h in hidden_personas if not h.get("is_synthetic")]
                if organic:
                    try:
                        from data_preparation.prompts import (
                            cluster_dominant_frame as _resolve_frame,
                        )
                        top_hp = max(organic, key=lambda h: int(h.get("evidence_rows") or 0))
                        f = _resolve_frame(top_hp)
                        if f and f != "none":
                            dominant_frame = f
                    except Exception:
                        dominant_frame = None

                # Bold any palette emoji / catchphrase strings that
                # actually appear in the example or inferior response.
                # Renderer adds these spans on the test card; we union
                # both so reviewers can see exactly what landed.
                anchor_spans: list = []
                for k in ("example_response_voice_evidence",
                          "inferior_response_voice_evidence"):
                    v = inst.get(k) or []
                    if isinstance(v, list):
                        anchor_spans.extend(s for s in v if isinstance(s, str))

                # Layered, scoped render — single call, single source of
                # truth (data_preparation.prompts.render_voice_for_test_card).
                # Falls back to the existing flat blob when the voice
                # schema has no layered fields populated.
                try:
                    from data_preparation.prompts import (
                        render_voice_for_test_card as _layered_render,
                    )
                    body = _layered_render(
                        uv if isinstance(uv, dict) else {},
                        ap if isinstance(ap, dict) else {},
                        target_app=target or "",
                        dominant_frame=dominant_frame,
                        voice_evidence_spans=anchor_spans,
                    )
                except Exception:
                    body = ""

                if body.strip():
                    gtp_lines.append(f"User voice — scoped for {app_label}:")
                    for ln in body.rstrip("\n").split("\n"):
                        gtp_lines.append(ln)
                else:
                    # Legacy snapshot fallback — flat schema only. Trim
                    # to the essentials so we don't dump 40 lines.
                    if isinstance(uv, dict) and uv:
                        if uv.get("natural_register"):
                            gtp_lines.append(f"  • register: {uv['natural_register']}")
                        palette = uv.get("emoji_palette") or []
                        if palette:
                            gtp_lines.append(
                                f"  • personal emoji palette: {' '.join(palette[:8])}"
                            )
                        _idio = uv.get("idiolect") or {}
                        phrases = (_idio.get("catchphrase_residue") if isinstance(_idio, dict) else None) \
                            or uv.get("personal_phrases") or []
                        if phrases:
                            gtp_lines.append(
                                "  • catchphrase residue: "
                                + ", ".join(f"\"{p}\"" for p in phrases[:4])
                            )
                    if style:
                        gtp_lines.append(f"On {app_label}: {_truncate(style, 240)}")

                if focus:
                    gtp_lines.append(f"Topical focus on {app_label}: {', '.join(focus[:6])}")
                bits = []
                if freq:
                    bits.append(f"posts {freq}")
                if audience:
                    bits.append(f"{audience} audience")
                if bits:
                    gtp_lines.append(f"Posting cadence: {' · '.join(bits)}")
        if top_hashtags:
            gtp_lines.append(f"Top hashtags the agent may naturally use: {', '.join(top_hashtags[:5])}")
        if top_cats:
            gtp_lines.append(f"Top categories: {', '.join(top_cats[:3])}")
        if voice_sample:
            gtp_lines.append(f"Voice sample (recent post): {_truncate(voice_sample, 100)}")
        if inst.get("inbound_message"):
            gtp_lines.append(f"Inbound DM: {_truncate(inst['inbound_message'], 100)}")
        if inst.get("source_post"):
            sp = inst["source_post"]
            gtp_lines.append(f"Source post: {_truncate(sp.get('caption',''), 100)}")
        if inst.get("recipient_name"):
            gtp_lines.append(f"Recipient name (collision): {inst['recipient_name']}")
        # T6 specifically — narrate the underlying preference behind the
        # user's ask so the reviewer sees WHY the user chose this seed
        # topic. Pulls from `_t6_seed` which keys off instance_id, so the
        # same persona_item shown here is the one the User Query topic
        # was derived from.
        if task_id == "agentic_user_tone_post":
            t6_seed = _t6_seed(inst)
            if t6_seed.get("persona_item"):
                gtp_lines.append(
                    f"Behind the user's ask: {_truncate(t6_seed['persona_item'], 160)}"
                )
        groundtruth_preference = "\n".join(gtp_lines) or "(persona context — see profile)"

    # Prefer the LLM-postprocessed example_response (set on inst by
    # llm_postprocess) over the placeholder fallback in `example_responses`.
    # Without this preference the tool_call's `args.text` carries a generic
    # stub even when the rendered Example Response section shows the
    # in-context LLM-generated text — the two end up out of sync.
    final_example_for_tool = (inst.get("example_response") or "").strip() or example_response
    tool_call = _build_agentic_tool_call(inst, final_example_for_tool)

    if arm == "overpersonalization":
        rubric = [
            "(-) Don't surface user preferences; complete the task generically.",
        ]
    else:
        # Per-task rubric_tags so each task surfaces what's actually graded
        # by personalization_rubric.py for it — instead of three generic
        # lines that don't apply to read-only / search / digest tasks.
        _AGENTIC_TASK_RUBRICS: dict[str, list[str]] = {
            # Voice-matching write tasks (T9/T10/T12/T13/T6) — voice + content
            # alignment + the actual write tool call.
            "agentic_user_tone_post": [
                "(+) Match the user's voice.",
                "(+) Reference what they've recently engaged with.",
                "(-) Don't include anything they wouldn't post publicly.",
            ],
            "agentic_cross_app_repost": [
                "(+) Adapt the source post to the target app's voice; preserve the core point.",
                "(+) Call create_post on the target app exactly once.",
            ],
            "agentic_auto_reply": [
                "(+) Reply in the user's voice; address the inbound message.",
                "(+) Send the DM exactly once.",
                "(-) Don't make commitments the user hasn't implied.",
            ],
            "agentic_composed_post": [
                "(+) Rewrite the user's update in their voice for this app.",
                "(+) Call create_post exactly once.",
            ],
            "agentic_send_post": [
                "(+) Compose the post in the user's voice on the target app.",
                "(+) Call create_post on the target app exactly once.",
                "(-) Don't post on any other app.",
            ],
            # Read-only summary / search / surfacing tasks — content fidelity only.
            "agentic_dm_digest": [
                "(+) Summarize the relevant DM threads accurately.",
                "(-) Don't surface private content the user wouldn't share.",
            ],
            "agentic_group_dm_summary": [
                "(+) Per-participant summary; identify decision points; suggest a reply in the user's voice.",
                "(-) Don't actually send the reply.",
            ],
            "agentic_vague_refind": [
                "(+) Identify the post the user is recalling; cite app + identifying detail (title/caption/hashtags).",
            ],
            "agentic_proactive_daily_catchup": [
                "(+) Surface 3-5 catch-up items aligned with the user's recent activity.",
                "(-) Don't include disliked topics.",
            ],
            "agentic_trending_alert": [
                "(+) Flag trending topics aligned with the user's interests.",
                "(-) Don't flag explicitly disliked topics.",
            ],
            "agentic_wrong_recipient_check": [
                "(+) If two contacts share the name, ASK for disambiguation rather than send.",
                "(-) Don't proceed silently when sensitive topics are involved.",
            ],
        }
        rubric = _AGENTIC_TASK_RUBRICS.get(task_id, [
            "(+) Match the user's voice when composing content.",
            "(+) Surface relevant preferences only when they fit.",
            "(-) Don't overpersonalize.",
        ])
    return {
        "example_response": example_response,
        "groundtruth_preference": groundtruth_preference,
        "tool_call": tool_call,
        "rubric_tags": rubric,
    }


TEST_GT_EXTRACTORS = {
    "personalized_feed_ranking":           _gt_personalized_feed_ranking,
    "slate_ranking":                       _gt_personalized_feed_ranking,  # v1 alias
    "chatbot_proactive_personalization":   _gt_chatbot_proactive,
    "chatbot_response_proactive":          _gt_chatbot_proactive,           # v1 alias
    "over_personalization_chatbot_text":   _gt_chatbot_restraint,
    "chatbot_restraint_control":           _gt_chatbot_restraint,           # v2 alias
    "chatbot_response_control":            _gt_chatbot_restraint,           # v1 alias
    "at_ai_directive_followup":            _gt_at_ai_directive,
    "e2_at_ai_followup":                   _gt_at_ai_directive,             # v1 alias
    "active_mistake_prevention":           _gt_active_mistake_prevention,
    "e6_active_mistake_prevention":        _gt_active_mistake_prevention,   # v1 alias
    "over_personalization_distractor_reject": _gt_irrelevant_query_restraint,
    "irrelevant_query_restraint":          _gt_irrelevant_query_restraint,  # v2 alias
    "over_personalization_sensitive_event": _gt_sensitive_event,
    "preference_removal_regen":            _gt_preference_removal_regen,
    "repetition_fatigue_pairs":            _gt_repetition_fatigue_pairs,
    "repetition_fatigue_sequences":        _gt_repetition_fatigue_sequences,
    "repetition_fatigue_same_preference":  _gt_repetition_fatigue_same_preference,
    "over_personalization_context_shift":  _gt_context_shift_scenarios,
    "context_shift_scenarios":             _gt_context_shift_scenarios,  # legacy alias
    "daily_personalized_briefing":         _gt_daily_personalized_briefing,
    # workstream D rename: personalized_search_ranking → personalized_recommendation
    "personalized_recommendation":         _gt_personalized_recommendation,
    "personalized_search_ranking":         _gt_personalized_recommendation,  # legacy alias
    "short_vs_long_term_lifecycle":        _gt_short_vs_long_term_lifecycle,
    # All agentic_* tasks share the generic agentic extractor.
    # agentic_draft_audit removed in workstream F.
    "agentic_user_tone_post":            _gt_agentic,
    # agentic_moment_recommendation merged into personalized_recommendation
    "agentic_dm_digest":                   _gt_agentic,
    "agentic_cross_app_repost":            _gt_agentic,
    "agentic_auto_reply":                  _gt_agentic,
    "agentic_vague_refind":                _gt_agentic,
    "agentic_composed_post":               _gt_agentic,
    "agentic_send_post":                   _gt_agentic,
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
    # Builders use varying field names: user_query (chatbot/agentic),
    # user_message (extracted from chatbot turns), query_text (slate_ranking),
    # query (C1a pairs / C2 scenarios), queries (C1b sequences — list of strings).
    for key in ("user_query", "user_message", "query_text", "query"):
        v = inst.get(key)
        if isinstance(v, str) and v:
            return v
    queries = inst.get("queries")
    if isinstance(queries, list) and queries:
        first = queries[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return str(first.get("user_query") or first.get("query") or first.get("text") or "")
    return ""


def _q_personalized_feed_ranking(inst: dict) -> str:
    app = inst.get("app") or "this app"
    return f"[ranking task] What should I be shown next on {app}?"


def _q_chatbot(inst: dict) -> str:
    return inst.get("user_query") or inst.get("user_message") or "[chatbot turn]"


def _q_at_ai_directive(inst: dict) -> str:
    # E2 simulates a proactive recsys feed served at T_test (24h/72h/7d after
    # the user's past @ai comment). The user is NOT typing anything at T_test
    # — the directive lives in the past as context. Render the live query as
    # `[recsys]` to make that explicit; the past @ai comment is rendered in
    # its own "Prior @ai comment" section by `_gt_at_ai_directive`.
    return "[recsys]"


def _q_active_mistake_prevention(inst: dict) -> str:
    return inst.get("user_query") or inst.get("triggering_user_query") or "[mistake-prevention probe]"


def _t6_seed(inst: dict) -> dict:
    """Deterministically pick a seed topic + underlying persona_item for an
    `agentic_user_tone_post` instance.

    Index by hash of `instance_id` so different T6 instances of the same
    user get different topics. Returns:
      {"topic": "<short noun phrase, lowercase>", "persona_item": "<the
       underlying preference whose engagement signal anchors this topic>"}

    Builds a pre-aligned pool of `(topic, pref)` pairs where the topic is
    a hashtag the pref actually carries — guarantees the surfaced GT
    persona_item matches the picked topic (no boxing-vs-entertainment
    drift). Falls back gracefully when the persona context is empty.
    """
    iid = str(inst.get("instance_id") or inst.get("task_id") or "")
    top_hashtags_raw = _PERSONA_CONTEXT.get("top_hashtags") or []
    top_prefs = _PERSONA_CONTEXT.get("top_prefs") or []
    pref_meta = _PERSONA_CONTEXT.get("pref_meta") or {}
    if not top_hashtags_raw or not top_prefs:
        return {"topic": "", "persona_item": ""}
    _BANAL = {"life", "love", "selfie", "photo", "vibes", "mood"}
    top_hashtags = {h.lower().lstrip("#") for h, _ in top_hashtags_raw if h.lower() not in _BANAL}
    # Aligned pool: each entry pairs a topic with the most-engaged pref
    # whose source_hashtags carry that topic.
    aligned: list[tuple[str, str]] = []
    seen_topics: set[str] = set()
    for pi, _cnt in top_prefs:
        meta = pref_meta.get(pi) or {}
        ph_list = [str(h).lower().lstrip("#") for h in (meta.get("source_hashtags") or [])]
        for ph in ph_list:
            if ph and ph in top_hashtags and ph not in seen_topics:
                aligned.append((ph, pi))
                seen_topics.add(ph)
                break  # one topic per pref keeps diversity
    # Fallback: any top_hashtag, paired with top_prefs[0].
    if not aligned:
        for h, _ in top_hashtags_raw:
            if h.lower() not in _BANAL:
                aligned.append((h.lower().lstrip("#"), top_prefs[0][0]))
                break
    if not aligned:
        return {"topic": "", "persona_item": ""}
    idx = sum(map(ord, iid)) % len(aligned)
    topic, chosen_pi = aligned[idx]
    return {"topic": topic, "persona_item": chosen_pi}


def _q_agentic_user_tone_post(inst: dict) -> str:
    """First-person, scope-narrowed query. The user names a topic so the
    agent has something concrete to write about. Seed comes from the user's
    own engagement history (see `_t6_seed`).

    Falls back to the legacy generic form when no seed is available (e.g.
    sparse user data).
    """
    seed = _t6_seed(inst)
    topic = (seed.get("topic") or "").lstrip("#")
    app = (inst.get("target_app") or "").strip()
    entry = (inst.get("entry_point") or "").strip()
    if not topic:
        return f"[agentic] compose a post in the user's voice on {app}"
    # Pick a template based on app + entry_point to vary phrasing across
    # the 6 T6 instances per user without going random.
    templates = {
        ("instagram", "app_native"):    "wanna post about {topic} on instagram, write it for me",
        ("instagram", "chatbot_routed"): "draft me an instagram caption about {topic}",
        ("facebook",  "app_native"):    "post about {topic} on facebook, keep it in my voice",
        ("facebook",  "chatbot_routed"): "draft a facebook post about {topic} — sound like me",
        ("threads",   "app_native"):    "drop a threads about {topic}, my style",
        ("threads",   "chatbot_routed"): "write me a threads post about {topic}",
    }
    template = templates.get((app, entry)) or f"post about {{topic}} on {app}, write it in my voice"
    return template.format(topic=topic)


# _q_agentic_moment_recommendation removed — moment instances now ride
# personalized_recommendation, which uses _q_personalized_recommendation
# (or surfaces the voiced query_text directly).


def _q_agentic_dm_digest(inst: dict) -> str:
    return f"[agentic] summarize my recent DMs on {inst.get('target_app', '')}"


def _q_agentic_cross_app_repost(inst: dict) -> str:
    src = inst.get("source_post") or {}
    cap = (src.get("caption") or "").strip()
    src_app = inst.get("source_app") or "the other app"
    tgt = inst.get("target_app") or "the target app"
    if not cap:
        return f"crosspost my latest {src_app} post over to {tgt} — adapt it to my voice"
    # First-person, scope-narrowed framing: the user names what they posted
    # on the source app and explicitly asks for a voice-adapted repost on
    # the target app. The full caption is included so the agent has
    # concrete material to rewrite (and the reviewer can see what's
    # being adapted in the Example / Inferior responses).
    return (
        f"crosspost this from {src_app} to {tgt} — adapt it to my voice for that audience: "
        f"\"{cap}\""
    )


def _q_agentic_auto_reply(inst: dict) -> str:
    sender = inst.get("sender_id") or "friend"
    msg = inst.get("inbound_message") or ""
    return f"[incoming DM from {sender}] {msg}"


def _q_agentic_vague_refind(inst: dict) -> str:
    return f"find that post I saw about {inst.get('topic', '')}"


def _q_agentic_composed_post(inst: dict) -> str:
    return f"[agentic] post on {inst.get('target_app', '')}: {inst.get('update', '')}"


def _q_agentic_send_post(inst: dict) -> str:
    ctx = (inst.get("context") or "").strip()
    target = inst.get("target_app") or "the target app"
    if not ctx:
        return f"can you post something for me on {target}?"
    # The user dictates context to the chatbot, then asks for a post.
    # First-person, scope-narrowed: the user names what they want posted
    # and explicitly delegates the writing to the chatbot. Strip a trailing
    # period from the dictated context so the joiner doesn't produce an
    # awkward double-period like "...vibes. — write that up...".
    ctx_clean = ctx.rstrip(".!?")
    return f"{ctx_clean} — write that up as a post on {target} for me, in my voice."


# _q_agentic_draft_audit removed in workstream F.


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


def _q_personalized_recommendation(inst: dict) -> str:
    """Fixed `[recsys]` template — matches the convention for proactive
    recsys-served slates with no live user message. Candidate titles
    already render in the slate block, so the prior format duplicated
    the candidate pool."""
    return "[recsys]"


def _q_short_vs_long_term_lifecycle(inst: dict) -> str:
    return "[lifecycle ranking] short-term vs long-term preference test"


TEST_QUERY_EXTRACTORS = {
    "personalized_feed_ranking":           _q_personalized_feed_ranking,
    "slate_ranking":                       _q_personalized_feed_ranking,
    "chatbot_proactive_personalization":   _q_chatbot,
    "chatbot_response_proactive":          _q_chatbot,
    "over_personalization_chatbot_text":   _q_chatbot,
    "chatbot_restraint_control":           _q_chatbot,
    "chatbot_response_control":            _q_chatbot,
    "over_personalization_distractor_reject": _q_chatbot,
    "over_personalization_sensitive_event": _q_chatbot,
    "at_ai_directive_followup":            _q_at_ai_directive,
    "e2_at_ai_followup":                   _q_at_ai_directive,
    "active_mistake_prevention":           _q_active_mistake_prevention,
    "e6_active_mistake_prevention":        _q_active_mistake_prevention,
    "agentic_user_tone_post":            _q_agentic_user_tone_post,
    # agentic_moment_recommendation merged into personalized_recommendation
    "agentic_dm_digest":                   _q_agentic_dm_digest,
    "agentic_cross_app_repost":            _q_agentic_cross_app_repost,
    "agentic_auto_reply":                  _q_agentic_auto_reply,
    "agentic_vague_refind":                _q_agentic_vague_refind,
    "agentic_composed_post":               _q_agentic_composed_post,
    "agentic_send_post":                   _q_agentic_send_post,
    # agentic_draft_audit removed in workstream F.
    "agentic_group_dm_summary":            _q_agentic_group_dm_summary,
    "agentic_wrong_recipient_check":       _q_agentic_wrong_recipient_check,
    "agentic_proactive_daily_catchup":     _q_agentic_proactive_daily_catchup,
    "agentic_trending_alert":              _q_agentic_trending_alert,
    "daily_personalized_briefing":         _q_daily_personalized_briefing,
    # workstream D rename
    "personalized_recommendation":         _q_personalized_recommendation,
    "personalized_search_ranking":         _q_personalized_recommendation,  # legacy alias
    "short_vs_long_term_lifecycle":        _q_short_vs_long_term_lifecycle,
}


def _load_test_samples(
    uid: str,
    benchmark_dir: str = "benchmark",
    backend_dir: str = "backend",
    include_instance_full: bool = False,
) -> list[dict]:
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

    When ``include_instance_full=True``, each sample also carries
    ``instance_full`` — the parsed instance_json dict from the CSV row,
    used by ``dump_test_samples_json`` so downstream tooling can read
    every field the builder emitted (blind_check_score, arm, polarity,
    etc.).
    """
    qcsv = os.path.join(benchmark_dir, str(uid), "queries.csv")
    out: list[dict] = []
    if not os.path.exists(qcsv):
        return out
    # Build the persona context bank ONCE; extractors use it to fill in
    # concrete expected-answer shapes when the instance itself is sparse.
    global _PERSONA_CONTEXT
    _PERSONA_CONTEXT = _build_persona_context(uid, backend_dir)
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
                gt = {"example_response": f"(extractor crashed: {type(exc).__name__})",
                      "groundtruth_preference": "", "rubric_tags": []}
            try:
                q_text = q_extractor(inst) or ""
            except Exception:
                q_text = ""
            try:
                ts_int = int(r.get("ts") or 0)
            except Exception:
                ts_int = 0
            # Phase 4: prefer postprocess-generated example_response /
            # groundtruth_preference (set on instance_full by
            # llm_postprocess) over the extractor's defaults. The
            # postprocess produces concrete LLM-generated text for
            # personalization tasks where the extractor only emits
            # meta-instructions; for ranking tasks it computes a
            # deterministic ranked-index list.
            example_response = (
                inst.get("example_response")
                or gt.get("example_response", "")
            )
            groundtruth_preference = (
                inst.get("groundtruth_preference")
                or gt.get("groundtruth_preference", "")
            )
            # at_ai_directive_followup's gt is deterministic (no LLM step),
            # so the extractor's current output is always more up-to-date
            # than any baked value frozen at build time. Forcing the
            # extractor avoids stale text when the rationale format
            # changes without re-running the build pipeline.
            #
            # Same applies to voice-dependent agentic write tasks: the
            # rendered groundtruth includes the user's per-app voice /
            # topical focus, which we update purely in the extractor.
            _RENDER_FROM_EXTRACTOR = {
                "at_ai_directive_followup", "e2_at_ai_followup",
                "agentic_user_tone_post", "agentic_composed_post",
                "agentic_send_post", "agentic_cross_app_repost",
                "agentic_auto_reply",
            }
            if task_type in _RENDER_FROM_EXTRACTOR:
                groundtruth_preference = gt.get("groundtruth_preference", "") or groundtruth_preference
            # Tag the test sample with the app it most directly concerns so
            # the per-app filter buttons can match. Empty string means
            # "no specific app" (e.g. daily_personalized_briefing spans all);
            # those samples remain visible only under the "All" filter.
            inst_app = (
                inst.get("target_app") or inst.get("directive_app")
                or inst.get("app") or ""
            )
            if not inst_app and (
                task_type.startswith("chatbot_")
                or task_type.startswith("over_personalization_")
                or task_type in {
                    "preference_removal_regen", "active_mistake_prevention",
                    "repetition_fatigue_pairs", "repetition_fatigue_sequences",
                    "repetition_fatigue_same_preference",
                    "agentic_vague_refind", "agentic_proactive_daily_catchup",
                    "agentic_trending_alert",
                }
            ):
                inst_app = "chatbot"
            sample = {
                "ts": ts_int,
                "ts_iso": r.get("ts_iso", ""),
                "task_type": task_type,
                "task_family": task_family,
                "app": (inst_app or "").lower(),
                "query_id": r.get("query_id", ""),
                "query_text": q_text,
                "example_response": example_response,
                "groundtruth_preference": groundtruth_preference,
                "rubric_tags": gt.get("rubric_tags") or (r.get("rubric_tags", "").split(";") if r.get("rubric_tags") else []),
            }
            # Pass through optional rich fields when present — JS template
            # renders each one as its own labeled section on the test card.
            # Workstream H: tool_call replaces tool_call_rules + final_state_expected
            # for agentic tasks (the ordered sequence implies both).
            for k in ("candidates", "held_out_pref",
                     "top_k_relevant", "correct_but_irrelevant_prefs",
                     "tool_call",
                     "warn_frame", "signal_evidence", "irrelevant_persona_items",
                     "carve_out", "forbidden_items", "prior_conversation", "extra_meta"):
                if k in gt:
                    sample[k] = gt[k]
            # Phase 4: surface postprocess-attached fields (inferior_response,
            # self_check) so the JS template can render them on the test card.
            if inst.get("inferior_response"):
                sample["inferior_response"] = inst["inferior_response"]
            if inst.get("example_response_self_check"):
                sample["example_response_self_check"] = inst["example_response_self_check"]
            # Voice-evidence spans for compose tasks — drives bold rendering
            # of the gold so a reviewer can see WHY a voice_mismatch foil fails.
            # Both sides are surfaced so the renderer can union them and bold
            # tone anchors the Example or the Inferior actually leverage.
            if inst.get("example_response_voice_evidence"):
                sample["example_response_voice_evidence"] = inst["example_response_voice_evidence"]
            if inst.get("inferior_response_voice_evidence"):
                sample["inferior_response_voice_evidence"] = inst["inferior_response_voice_evidence"]
            if inst.get("voice_evidence_smoke_check"):
                sample["voice_evidence_smoke_check"] = inst["voice_evidence_smoke_check"]
            if inst.get("voice_evidence_smoke_check_after_regen"):
                sample["voice_evidence_smoke_check_after_regen"] = inst["voice_evidence_smoke_check_after_regen"]
            if include_instance_full:
                sample["instance_full"] = inst
            out.append(sample)
    return out


# ---------------------------------------------------------------------------
# Phase 1.A — test.json dump
#
# Re-uses _load_test_samples and enriches each sample with:
#   - query_kind, expected_behavior        (from evaluation.task_registry)
#   - ground_truth_preference (normalized)
#   - reference_example       (looked up in the app JSONs by persona_item)
#   - distractor_preferences  (normalized union of top_k_relevant /
#     correct_but_irrelevant_prefs / irrelevant_persona_items, each
#     tagged with `role`)
#   - instance_full           (pass-through of the original instance_json)
# ---------------------------------------------------------------------------

def _normalize_held_out(sample: dict) -> dict | None:
    """Pull the held-out preference into a canonical {persona_item,
    category, polarity, source_hashtags} shape — or None if there isn't
    one for this task type."""
    inst = sample.get("instance_full") or {}
    held_obj = inst.get("held_out_preference") or {}
    if not held_obj:
        # Some extractors put it under held_out_pref (already-stringified)
        # — fall back to that, but it loses category/polarity info.
        held_str = sample.get("held_out_pref")
        if not held_str:
            return None
        return {
            "persona_item": held_str,
            "category": "",
            "polarity": "positive",
            "source_hashtags": inst.get("source_hashtags") or [],
        }
    pi = held_obj.get("persona_item") or ""
    if not pi:
        return None
    return {
        "persona_item": pi,
        "category": held_obj.get("category") or "",
        "polarity": held_obj.get("polarity") or "positive",
        "source_hashtags": held_obj.get("source_hashtags") or inst.get("source_hashtags") or [],
    }


def _find_reference_example(uid: str, persona_item: str, t_test: int,
                            backend_dir: str = "backend") -> dict | None:
    """Walk app JSONs and return the closest-by-timestamp event that
    contains a preference whose persona_item matches. Returns a compact
    evidence record (the full event would be too heavy)."""
    if not persona_item:
        return None
    user_dir = Path(backend_dir) / str(uid)
    best: tuple[int, dict, str] | None = None  # (abs_dt, event, app)
    for app in APPS:
        p = user_dir / (app.lower() + ".json")
        if not p.exists():
            continue
        try:
            evs = json.loads(p.read_text())
        except Exception:
            continue
        for e in evs:
            for pref in (e.get("preferences") or []):
                if (pref.get("persona_item") or "").strip().lower() == persona_item.strip().lower():
                    ts = int(e.get("source_timestamp") or 0)
                    dt = abs(ts - t_test)
                    if best is None or dt < best[0]:
                        best = (dt, e, app)
                    break
    if best is None:
        return None
    _, ev, app = best
    content = ev.get("content") or {}
    snippet = content.get("caption") or content.get("title") or content.get("overall_description") or ""
    return {
        "source_object_id": ev.get("source_object_id", ""),
        "source_app": app,
        "source_timestamp": ev.get("source_timestamp", 0),
        "source_hashtags": ev.get("source_hashtags") or [],
        "interaction_format": ev.get("interaction_format") or {},
        "content_snippet": _truncate(snippet, 200),
    }


def _normalize_distractors(sample: dict) -> list[dict]:
    """Merge the various near-miss / irrelevant / privacy-flagged pools
    into a single list of {persona_item, category, polarity, role}."""
    out: list[dict] = []
    seen: set[str] = set()

    def _push(items, role):
        for it in items or []:
            if isinstance(it, dict):
                pi = it.get("persona_item") or ""
                cat = it.get("category") or ""
                pol = it.get("polarity") or "positive"
            else:
                pi = str(it or "")
                cat = ""
                pol = "positive"
            if not pi or pi in seen:
                continue
            seen.add(pi)
            out.append({
                "persona_item": pi,
                "category": cat,
                "polarity": pol,
                "role": role,
            })

    _push(sample.get("top_k_relevant"), "near_miss")
    _push(sample.get("correct_but_irrelevant_prefs"), "irrelevant")
    _push(sample.get("irrelevant_persona_items"), "privacy_flagged")
    # forbidden_items can also act as a do-not-surface pool for C2
    _push(sample.get("forbidden_items"), "privacy_flagged")
    return out


def dump_test_samples_json(
    uid: str,
    output_path: str | None = None,
    benchmark_dir: str = "benchmark",
    backend_dir: str = "backend",
) -> str:
    """Build backend/{uid}/test.json — every test query in one place.

    See the plan in /vast/home/b/bwjiang/.claude/plans/ for the schema.
    """
    from evaluation import task_registry as _tr

    samples = _load_test_samples(uid, benchmark_dir, backend_dir, include_instance_full=True)
    records: list[dict] = []
    for s in samples:
        task_type = s["task_type"]
        inst = s.get("instance_full") or {}
        held = _normalize_held_out(s)
        ref_ex = _find_reference_example(
            uid,
            held["persona_item"] if held else "",
            int(s.get("ts") or 0),
            backend_dir=backend_dir,
        ) if held else None
        record = {
            "query_id": s.get("query_id", ""),
            "task_family": s.get("task_family", ""),
            "task_type": task_type,
            "query_kind": _tr.get_query_kind(task_type),
            "expected_behavior": _tr.get_expected_behavior(task_type),
            "ts": s.get("ts", 0),
            "ts_iso": s.get("ts_iso", ""),
            "user_query": s.get("query_text") or None,
            "prior_conversation": s.get("prior_conversation"),
            # Workstream C: example_response + groundtruth_preference are the
            # extractor's two new fields. The legacy `ground_truth_preference`
            # block (held-out persona_item only) is kept in
            # `groundtruth_preference_obj` for tooling that needs the raw
            # held-out signal alongside the rendered text.
            "example_response": s.get("example_response", ""),
            "groundtruth_preference": s.get("groundtruth_preference", ""),
            "groundtruth_preference_obj": held,
            "reference_example": ref_ex,
            "distractor_preferences": _normalize_distractors(s),
            "rubric_tags": s.get("rubric_tags") or [],
            "tool_call": s.get("tool_call"),  # workstream H, agentic only
            # Phase 4: paired foil + self-check signal lifted to top-level
            # for downstream tooling (instance_full also keeps the originals).
            "inferior_response": s.get("inferior_response") or inst.get("inferior_response"),
            "example_response_self_check": (
                s.get("example_response_self_check")
                or inst.get("example_response_self_check")
            ),
            "example_response_voice_evidence": (
                s.get("example_response_voice_evidence")
                or inst.get("example_response_voice_evidence")
            ),
            "inferior_response_voice_evidence": (
                s.get("inferior_response_voice_evidence")
                or inst.get("inferior_response_voice_evidence")
            ),
            "voice_evidence_smoke_check": (
                s.get("voice_evidence_smoke_check")
                or inst.get("voice_evidence_smoke_check")
            ),
            "voice_evidence_smoke_check_after_regen": (
                s.get("voice_evidence_smoke_check_after_regen")
                or inst.get("voice_evidence_smoke_check_after_regen")
            ),
            "instance_full": inst,
        }
        # Compact: drop empty optional fields so the file stays readable.
        for k in ("prior_conversation",):
            if record[k] in (None, [], {}):
                record[k] = None
        records.append(record)

    if output_path is None:
        output_path = os.path.join(backend_dir, str(uid), "test.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return output_path


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
            location_parts.append(f"<span>{label}</span>")
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
  .profile-card .profile-voice {{ margin-top: 18px; padding-top: 14px; border-top: 1px dashed var(--border); }}
  .profile-card .profile-voice-header {{ font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 4px; }}
  .profile-card .profile-voice-pill {{ font-size: 10px; font-weight: 500; padding: 2px 8px; border-radius: 10px; background: #F2F2F7; color: var(--text-secondary); margin-left: 6px; }}
  .profile-card .profile-voice-subtitle {{ font-size: 12px; color: var(--text-secondary); margin-bottom: 8px; font-style: italic; }}
  .profile-card .profile-voice ul {{ list-style: none; padding: 0; margin: 0; font-size: 12px; line-height: 1.7; color: var(--text); }}
  .profile-card .profile-voice .voice-key {{ font-weight: 600; color: var(--text-secondary); margin-right: 4px; }}
  .profile-card .profile-voice .voice-avoid {{ font-style: italic; color: #8B5A2B; }}
  /* Layered 4-layer voice block — Layer 1/2/3 sub-sections inside the profile-voice card */
  .profile-card .profile-voice-section {{ margin-top: 10px; padding-top: 8px; border-top: 1px dotted #d1d5db; }}
  .profile-card .profile-voice-section:first-of-type {{ border-top: none; padding-top: 0; }}
  .profile-card .profile-voice-section-header {{ font-size: 11px; font-weight: 700; color: #4b5563; text-transform: uppercase; letter-spacing: 0.3px; margin-bottom: 4px; }}
  .profile-card .profile-voice-section-hint {{ font-weight: 400; text-transform: none; letter-spacing: 0; color: #9ca3af; font-style: italic; margin-left: 6px; }}
  /* Stance/register/genre chips — used in both Layer-3 inventory and per-app active selection */
  .stance-chip {{ display: inline-block; font-size: 10px; padding: 2px 7px; border-radius: 10px; background: #eef2ff; color: #4338ca; margin: 1px 2px; border: 1px solid #c7d2fe; }}

  .section {{ margin-bottom: 40px; }}
  .section-title {{ font-size: 16px; font-weight: 600; letter-spacing: -0.2px; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border); color: var(--text); }}

  .event-grid {{ display: flex; flex-direction: column; gap: 12px; }}
  .event-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 20px; box-shadow: var(--shadow); transition: box-shadow 0.15s ease; border-left: 3px solid var(--border); }}
  .event-card:hover {{ box-shadow: var(--shadow-hover); }}
  .event-card.app-Instagram {{ border-left-color: #C13584; }}
  .event-card.app-Facebook {{ border-left-color: #4A6FA5; }}
  .event-card.app-Threads {{ border-left-color: #636366; }}
  .event-card.app-Chatbot {{ border-left-color: #8B5CF6; }}
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
    font-family: inherit;
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
  .ts-rubric-bar .ts-list li.rubric-pos {{
    border-left: 3px solid #10b981;
    padding-left: 8px;
    margin-bottom: 4px;
  }}
  .ts-rubric-bar .ts-list li.rubric-neg {{
    border-left: 3px solid #f59e0b;
    padding-left: 8px;
    margin-bottom: 4px;
  }}
  .ts-rubric-bar .rubric-sign {{
    font-weight: 700;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 0.92em;
    color: var(--text-secondary);
  }}
  .ts-rubric-bar li.rubric-pos .rubric-sign {{ color: #047857; }}
  .ts-rubric-bar li.rubric-neg .rubric-sign {{ color: #b45309; }}
  .ts-label {{ font-weight: 600; font-size: 11px; color: #7B5C00; text-transform: uppercase; letter-spacing: 0.4px; margin-bottom: 4px; }}
  .ts-section-warn .ts-label {{ color: #B91C1C; }}
  .ts-sublabel {{ font-size: 10px; font-weight: 500; color: var(--text-secondary); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.3px; }}
  .ts-body {{ font-size: 12px; color: var(--text); line-height: 1.45; }}
  /* Voice / tone anchor highlight inside test-card bodies. <strong> alone
     is invisible on emoji glyphs, so add a yellow background + outline so
     reviewers can spot which palette emoji or personal phrases the
     Example / Inferior actually leveraged. */
  .ts-body strong {{
    background: #fff3cd;
    border-radius: 3px;
    padding: 0 3px;
    font-weight: 700;
    box-shadow: inset 0 -2px 0 #f59e0b;
  }}
  .ts-body.ts-mono {{ font-size: 11px; white-space: pre-wrap; color: var(--text-secondary); }}
  .ts-list {{ margin: 4px 0 0 0; padding-left: 18px; font-size: 12px; line-height: 1.5; }}
  .ts-list.ts-mono {{ font-size: 11px; color: var(--text-secondary); }}
  /* Inline <code> in test cards inherits the page font — keep visually distinct
     via subtle background + smaller size + grey, NOT a different font family. */
  .test-sample-card code,
  .ts-section code,
  .test-sample-meta code,
  .test-sample-card .event-meta code {{
    font-family: inherit; font-size: 0.92em; padding: 1px 5px;
    background: rgba(255,255,255,0.65); border-radius: 3px; color: var(--text-secondary);
  }}
  .ts-list li {{ margin: 3px 0; }}
  .ts-origin {{ display: inline-block; font-size: 9px; padding: 1px 5px; border-radius: 3px; background: #E5E7EB; color: #374151; margin: 0 2px; text-transform: uppercase; letter-spacing: 0.3px; }}
  .ts-origin-held_out {{ background: #D4AF37; color: #fff; }}
  .ts-origin-target {{ background: #D4AF37; color: #fff; }}
  .ts-origin-future_positive {{ background: #BFDBFE; color: #1E40AF; }}
  .ts-origin-past_positive {{ background: #BBF7D0; color: #166534; }}
  .ts-origin-negative {{ background: #FECACA; color: #7F1D1D; }}
  .ts-origin-irrelevant, .ts-origin-random, .ts-origin-filler, .ts-origin-filler_lowsim {{ background: #F3F4F6; color: #6B7280; }}
  .ts-origin-match {{ background: #D4AF37; color: #fff; }}
  .ts-origin-carve_out {{ background: #FECACA; color: #7F1D1D; }}
  .ts-target {{ font-size: 10px; color: #B45309; font-weight: 700; }}
  .ts-delta {{ font-size: 10px; color: #6B7280; font-weight: 600; padding: 1px 5px; border-radius: 3px; background: #F3F4F6; margin: 0 2px; }}
  .badge.platform {{ font-weight: 600; font-size: 11px; padding: 2px 10px; }}
  .badge.platform.p-Instagram {{ background: #C13584; color: #fff; }}
  .badge.platform.p-Facebook {{ background: #4A6FA5; color: #fff; }}
  .badge.platform.p-Threads {{ background: #8E8E93; color: #fff; }}
  .badge.platform.p-Chatbot {{ background: #8B5CF6; color: #fff; }}
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
  .app-personas-section {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 22px; margin-bottom: 40px; box-shadow: var(--shadow); }}
  .app-personas-section h2 {{ font-size: 18px; font-weight: 600; margin-bottom: 14px; letter-spacing: -0.2px; }}
  .app-persona-card {{ padding: 14px 18px; margin-bottom: 10px; border-radius: 8px; background: #FAFAFA; border: 1px solid #F2F2F7; }}
  .app-persona-header {{ font-size: 14px; font-weight: 600; color: var(--text); margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }}
  .app-persona-pill {{ font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; padding: 2px 7px; border-radius: 10px; background: #E5E7EB; color: #374151; }}
  .app-persona-style {{ font-size: 12px; line-height: 1.6; color: var(--text); margin-bottom: 8px; font-style: italic; }}
  .app-persona-row {{ font-size: 11px; color: var(--text-secondary); line-height: 1.6; margin-bottom: 3px; }}
  .app-persona-row .app-persona-key {{ font-weight: 600; color: var(--text); margin-right: 4px; }}
  .app-persona-sig {{ list-style: none; padding: 8px 12px; margin: 6px 0 8px 0; background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 6px; font-size: 11px; line-height: 1.7; }}
  .app-persona-sig li {{ color: var(--text); margin: 0; }}
  .app-persona-sig .app-persona-key {{ font-weight: 600; color: #4338CA; margin-right: 4px; }}
  .app-persona-sig li.app-persona-avoid {{ font-style: italic; color: #8B5A2B; }}
  .app-persona-sig li.app-persona-avoid .app-persona-key {{ color: #8B5A2B; }}
  .app-persona-examples {{ list-style: none; padding: 0; margin: 4px 0 8px 12px; }}
  /* Filter bar between Hidden Personas and the Timeline. Buttons toggle
     visibility of timeline cards via data-filter-key. */
  .filter-bar {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 14px 18px; margin-bottom: 24px; background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); box-shadow: var(--shadow); }}
  .filter-bar .filter-spacer {{ flex: 1 1 auto; }}
  .filter-bar .filter-btn {{ font-size: 12px; font-weight: 600; padding: 6px 14px; border-radius: 18px; border: 1px solid #E5E7EB; background: #FFFFFF; color: var(--text); cursor: pointer; transition: background 0.12s ease, color 0.12s ease, border-color 0.12s ease; }}
  .filter-bar .filter-btn:hover {{ background: #F3F4F6; }}
  .filter-bar .filter-btn.active {{ background: #4338CA; color: #FFFFFF; border-color: #4338CA; }}
  .filter-bar .filter-btn[data-filter-key="instagram"].active {{ background: #C13584; border-color: #C13584; }}
  .filter-bar .filter-btn[data-filter-key="facebook"].active {{ background: #4A6FA5; border-color: #4A6FA5; }}
  .filter-bar .filter-btn[data-filter-key="threads"].active {{ background: #636366; border-color: #636366; }}
  .filter-bar .filter-btn[data-filter-key="chatbot"].active {{ background: #8B5CF6; border-color: #8B5CF6; }}
  .filter-bar .filter-btn[data-filter-key="test"].active {{ background: #B45309; border-color: #B45309; }}
  .app-persona-example {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace; font-size: 11px; line-height: 1.6; color: var(--text); padding: 4px 8px; margin-bottom: 3px; background: #F9FAFB; border-left: 2px solid #6D28D9; border-radius: 0 4px 4px 0; white-space: pre-wrap; }}
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
  /* DM threads reuse the chatbot bubble layout (chat-thread / chat-bubble)
     so social-app DMs and the AI Chatbot render visually the same.
     Self → user-bubble (right, blue). Friend or stranger → assistant-bubble
     (left, gray) with a "friend" or "stranger" role label.
     A few DM-only extras (white-space wrap, reaction-only bubble, forwarded
     post block) layer on top of the shared chat-bubble base. */
  .chat-bubble.dm-bubble {{ white-space: pre-wrap; }}
  .chat-bubble.dm-bubble .dm-reaction {{ font-size: 22px; line-height: 1; margin-top: 2px; }}
  .chat-bubble.dm-reaction-only {{ background: transparent; border: none; padding: 2px 6px; color: var(--text-secondary); }}
  .chat-bubble.dm-reaction-only .chat-role {{ font-size: 10px; }}
  .chat-bubble.dm-bubble .dm-forwarded {{ margin-top: 6px; padding: 6px 8px; background: rgba(255,255,255,0.6); border-left: 3px solid rgba(0,0,0,0.18); border-radius: 4px; font-size: 12px; }}
  .chat-bubble.user-bubble .dm-forwarded {{ background: rgba(255,255,255,0.18); border-left-color: rgba(255,255,255,0.55); color: #fff; }}
  .chat-bubble.dm-bubble .dm-forwarded .content-block {{ margin: 0; padding: 0; background: transparent; border: none; }}
  .chat-bubble.dm-bubble .dm-forwarded .c-type {{ font-size: 10px; opacity: 0.7; margin-bottom: 2px; }}
  .chat-bubble.user-bubble .dm-forwarded .c-type,
  .chat-bubble.user-bubble .dm-forwarded .c-caption,
  .chat-bubble.user-bubble .dm-forwarded .c-desc,
  .chat-bubble.user-bubble .dm-forwarded .c-title,
  .chat-bubble.user-bubble .dm-forwarded .c-text-body {{ color: #fff; }}
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
  <div id="app-personas-section"></div>
  <div id="hidden-personas-section"></div>

  <div class="filter-bar" id="filter-bar">
    <button class="filter-btn active" data-filter-key="all">All</button>
    <button class="filter-btn" data-filter-key="instagram">Instagram</button>
    <button class="filter-btn" data-filter-key="facebook">Facebook</button>
    <button class="filter-btn" data-filter-key="threads">Threads</button>
    <button class="filter-btn" data-filter-key="chatbot">Chatbot</button>
    <span class="filter-spacer"></span>
    <button class="filter-btn" data-filter-key="test">Test queries only</button>
  </div>

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

  // Shared user_voice block — rendered INSIDE the profile-card just below
  // the MBTI chips. Layered: Identity spine (Layer 1) + Idiolect (Layer 2) +
  // Indexical repertoire (Layer 3) + soft surface descriptors. Per-app
  // sections only show what genuinely shifts.
  const uv = profileData.user_voice || {{}};
  let userVoiceHtml = '';
  const _spine = uv.identity_spine || {{}};
  const _idio  = uv.idiolect       || {{}};
  const _rep   = uv.repertoire     || {{}};
  const _hasNew = (Object.keys(_spine).length || Object.keys(_idio).length || Object.keys(_rep).length);
  const _hasLegacy = uv.natural_register || uv.default_capitalization || (uv.emoji_palette||[]).length || (uv.personal_phrases||[]).length;
  if (uv && (_hasNew || _hasLegacy)) {{
    const palette = (uv.emoji_palette || []).join(' ');
    const avoidPhrases = (uv.phrases_to_avoid || []).map(p => `"${{escapeHtml(p)}}"`).join(', ');

    const sections = [];

    // ---- Section 1: Identity spine -----------------------------------
    if (Object.keys(_spine).length) {{
      const sp = _spine;
      const liwc = sp.liwc_anchors || {{}};
      const liwcStr = Object.keys(liwc).map(k => `${{escapeHtml(k)}}=${{escapeHtml(String(liwc[k]))}}`).join(', ');
      const b5 = sp.big_five_drivers || {{}};
      const b5Str = Object.keys(b5).map(k => `<li><span class="voice-key">${{escapeHtml(k)}}:</span> ${{escapeHtml(String(b5[k]))}}</li>`).join('');
      const rows = [];
      if (sp.agency_communion)                            rows.push(`<li><span class="voice-key">agency/communion:</span> ${{escapeHtml(sp.agency_communion)}}</li>`);
      if ((sp.redemption_motifs||[]).length)              rows.push(`<li><span class="voice-key">redemption motifs:</span> ${{(sp.redemption_motifs||[]).map(escapeHtml).join('; ')}}</li>`);
      if ((sp.contamination_motifs||[]).length)           rows.push(`<li><span class="voice-key">contamination motifs:</span> ${{(sp.contamination_motifs||[]).map(escapeHtml).join('; ')}}</li>`);
      if ((sp.life_stage_preoccupations||[]).length)      rows.push(`<li><span class="voice-key">life-stage preoccupations:</span> ${{(sp.life_stage_preoccupations||[]).map(escapeHtml).join('; ')}}</li>`);
      if ((sp.signature_concerns||[]).length)             rows.push(`<li><span class="voice-key">signature concerns:</span> ${{(sp.signature_concerns||[]).map(escapeHtml).join('; ')}}</li>`);
      if (liwcStr)                                        rows.push(`<li><span class="voice-key">LIWC anchors:</span> ${{liwcStr}}</li>`);
      if (b5Str)                                          rows.push(`<li><span class="voice-key">Big-Five drivers:</span><ul style="margin-top:2px;">${{b5Str}}</ul></li>`);
      if (rows.length) {{
        sections.push(`
          <div class="profile-voice-section">
            <div class="profile-voice-section-header">Layer 1 — Identity spine <span class="profile-voice-section-hint">drives WHAT this person brings up</span></div>
            <ul>${{rows.join('')}}</ul>
          </div>`);
      }}
    }}

    // ---- Section 2: Idiolect -----------------------------------------
    if (Object.keys(_idio).length) {{
      const id = _idio;
      const sp = id.syntactic_preferences || {{}};
      const af = id.appraisal_fingerprint || {{}};
      const tmpls = (id.constructional_templates || []).map(t => {{
        const pat = escapeHtml(t.pattern || '');
        const ex  = escapeHtml(t.example_realization || '');
        return `<li><code style="font-family:ui-monospace,Menlo,Monaco,monospace;background:#fffbeb;padding:1px 5px;border-radius:3px;">${{pat}}</code> <span style="opacity:0.7;">e.g. "${{ex}}"</span></li>`;
      }}).join('');
      const residue = (id.catchphrase_residue || []).map(p => `"${{escapeHtml(p)}}"`).join(', ');
      const rows = [];
      if (id.function_word_profile)                       rows.push(`<li><span class="voice-key">function-word profile:</span> ${{escapeHtml(id.function_word_profile)}}</li>`);
      if (Object.keys(sp).length)                         rows.push(`<li><span class="voice-key">sentences:</span> shape=${{escapeHtml(sp.sentence_length_shape||'?')}}, embedding=${{escapeHtml(sp.clause_embedding||'?')}}, parataxis/hypotaxis=${{escapeHtml(sp.parataxis_hypotaxis||'?')}}, fragments=${{escapeHtml(sp.fragment_use||'?')}}</li>`);
      if (id.hedge_booster_ratio)                         rows.push(`<li><span class="voice-key">hedge/booster:</span> ${{escapeHtml(id.hedge_booster_ratio)}}</li>`);
      if (Object.keys(af).length)                         rows.push(`<li><span class="voice-key">appraisal:</span> attitude=${{escapeHtml(af.attitude_dominant||'?')}}, engagement=${{escapeHtml(af.engagement_style||'?')}}, graduation=${{escapeHtml(af.graduation||'?')}}</li>`);
      if (tmpls)                                          rows.push(`<li><span class="voice-key">templates (slot patterns — apply abstractly):</span><ul style="margin-top:2px;">${{tmpls}}</ul></li>`);
      if (residue)                                        rows.push(`<li><span class="voice-key">catchphrase residue (use ZERO in most outputs; AT MOST one per response):</span> ${{residue}}</li>`);
      if (uv.default_capitalization)                      rows.push(`<li><span class="voice-key">capitalization:</span> ${{escapeHtml(uv.default_capitalization)}}</li>`);
      if (uv.punctuation_habits)                          rows.push(`<li><span class="voice-key">punctuation:</span> ${{escapeHtml(uv.punctuation_habits)}}</li>`);
      if (uv.formality_baseline !== undefined && uv.formality_baseline !== null) rows.push(`<li><span class="voice-key">formality:</span> ${{escapeHtml(String(uv.formality_baseline))}}</li>`);
      if (palette)                                        rows.push(`<li><span class="voice-key">palette (subset only — never invent):</span> ${{escapeHtml(palette)}} <span style="opacity:0.7;">(intensity ${{escapeHtml(uv.emoji_intensity_default || 'medium')}})</span></li>`);
      if (rows.length) {{
        sections.push(`
          <div class="profile-voice-section">
            <div class="profile-voice-section-header">Layer 2 — Idiolect <span class="profile-voice-section-hint">must survive paraphrase — don't just imitate words</span></div>
            <ul>${{rows.join('')}}</ul>
          </div>`);
      }}
    }}

    // ---- Section 3: Indexical repertoire -----------------------------
    if (Object.keys(_rep).length) {{
      const rp = _rep;
      const stanceChips = (rp.stances || []).map(s => `<span class="stance-chip">${{escapeHtml(s)}}</span>`).join(' ');
      const regChips    = (rp.registers || []).map(s => `<span class="stance-chip">${{escapeHtml(s)}}</span>`).join(' ');
      const genreChips  = (rp.speech_genre_fluency || []).map(s => `<span class="stance-chip">${{escapeHtml(s)}}</span>`).join(' ');
      const rows = [];
      if (stanceChips)                          rows.push(`<li><span class="voice-key">stances:</span> ${{stanceChips}}</li>`);
      if (regChips)                             rows.push(`<li><span class="voice-key">registers:</span> ${{regChips}}</li>`);
      if (rp.backstage_frontstage_range)        rows.push(`<li><span class="voice-key">backstage/frontstage range:</span> ${{escapeHtml(rp.backstage_frontstage_range)}}</li>`);
      if (genreChips)                           rows.push(`<li><span class="voice-key">speech-genre fluency:</span> ${{genreChips}}</li>`);
      if (rows.length) {{
        sections.push(`
          <div class="profile-voice-section">
            <div class="profile-voice-section-header">Layer 3 — Indexical repertoire <span class="profile-voice-section-hint">stable inventory; per-app picks a subset</span></div>
            <ul>${{rows.join('')}}</ul>
          </div>`);
      }}
    }}

    // ---- Voice avoid -------------------------------------------------
    const avoidRows = [];
    if (uv.voice_avoid)  avoidRows.push(`<li class="voice-avoid"><span class="voice-key">tones to never produce:</span> ${{escapeHtml(uv.voice_avoid)}}</li>`);
    if (avoidPhrases)    avoidRows.push(`<li class="voice-avoid"><span class="voice-key">phrases to avoid:</span> ${{avoidPhrases}}</li>`);
    if (avoidRows.length) {{
      sections.push(`
        <div class="profile-voice-section">
          <div class="profile-voice-section-header">Voice avoid</div>
          <ul>${{avoidRows.join('')}}</ul>
        </div>`);
    }}

    // ---- Legacy fallback for old snapshots without layered schema ----
    if (!sections.length && _hasLegacy) {{
      const legacyRows = [];
      const phrases = (uv.personal_phrases || []).map(p => `"${{escapeHtml(p)}}"`).join(', ');
      if (uv.natural_register)        legacyRows.push(`<li><span class="voice-key">register:</span> ${{escapeHtml(uv.natural_register)}}</li>`);
      if (uv.default_capitalization)  legacyRows.push(`<li><span class="voice-key">caps:</span> ${{escapeHtml(uv.default_capitalization)}}</li>`);
      if (uv.punctuation_habits)      legacyRows.push(`<li><span class="voice-key">punctuation:</span> ${{escapeHtml(uv.punctuation_habits)}}</li>`);
      if (uv.humor_tone)              legacyRows.push(`<li><span class="voice-key">humor / tone:</span> ${{escapeHtml(uv.humor_tone)}}</li>`);
      if (palette)                    legacyRows.push(`<li><span class="voice-key">personal emoji palette:</span> ${{escapeHtml(palette)}}</li>`);
      if (uv.emoji_intensity_default) legacyRows.push(`<li><span class="voice-key">emoji intensity:</span> ${{escapeHtml(uv.emoji_intensity_default)}}</li>`);
      if (phrases)                    legacyRows.push(`<li><span class="voice-key">personal phrases (legacy):</span> ${{phrases}}</li>`);
      if (legacyRows.length) sections.push(`<div class="profile-voice-section"><ul>${{legacyRows.join('')}}</ul></div>`);
    }}

    userVoiceHtml = `
      <div class="profile-voice">
        <div class="profile-voice-header">Writing voice <span class="profile-voice-pill">4-layer model</span></div>
        <div class="profile-voice-subtitle">Layers 1–3 stay coherent across all apps; per-app sections show only what shifts.</div>
        ${{sections.join('')}}
      </div>`;
  }}

  // Derive pronouns from the gender string (best-effort; matches CLAUDE.md
  // demographics). Trans masc / trans man / transgender male → he/him;
  // trans femme / trans woman / transgender female → she/her; non-binary /
  // genderfluid / genderqueer → they/them; unmodified male/man → he/him;
  // unmodified female/woman → she/her; otherwise unspecified.
  // The (?:gender)? lets "transgender male" match the same trans branch
  // as "trans male" / "transmasc".
  const _derivePronouns = (g) => {{
    const s = (g || '').toLowerCase();
    if (!s) return '';
    if (/\btrans(?:gender)?\s*(masc|man|male|masculine)\b|\btransmasc\b/.test(s)) return 'he/him';
    if (/\btrans(?:gender)?\s*(femme|woman|female|feminine)\b|\btransfemme\b/.test(s)) return 'she/her';
    if (/\bnon[\s-]?binary\b|\bnonbinary\b|\bgenderfluid\b|\bgenderqueer\b|\benby\b/.test(s)) return 'they/them';
    if (/\b(male|man|cis\s*man)\b/.test(s)) return 'he/him';
    if (/\b(female|woman|cis\s*woman)\b/.test(s)) return 'she/her';
    return '';
  }};
  const pronouns = _derivePronouns(profileData.gender);

  ps.innerHTML = `
    <div class="profile-card">
      <h2>${{profileData.name || ''}}</h2>
      <div class="bio">${{profileData.bio || ''}}</div>
      <div class="details">
        <span>${{profileData.gender || ''}}</span>
        ${{pronouns ? `<span>${{pronouns}}</span>` : ''}}
        <span>${{profileData.race_ethnicity || ''}}</span>
        <span>${{profileData.career || ''}}</span>
        <span>${{profileData.education || ''}}</span>
      </div>
      <div class="big-five">${{b5Html}}</div>
      ${{mbtiHtml}}
      ${{userVoiceHtml}}
    </div>
  `;
}}

// -- Per-app Personas section (writing voice + per-app deltas) --
const aps = document.getElementById('app-personas-section');
if (profileData && profileData.app_personas && Object.keys(profileData.app_personas).length > 0) {{
  const apps = profileData.app_personas;
  const order = ['Instagram', 'Facebook', 'Threads', 'Chatbot'];
  const keys = order.filter(k => apps[k]).concat(
    Object.keys(apps).filter(k => !order.includes(k))
  );

  // Note: the shared user_voice block now lives INSIDE the profile-card
  // (rendered above by the profile-card builder). Per-app cards below
  // only describe what genuinely shifts on each app.

  let html = '<div class="app-personas-section"><h2>Per-app Personas (one shared voice + per-app modulation)</h2>';
  keys.forEach(k => {{
    const ap = apps[k] || {{}};
    // New schema: delta_summary. Legacy fallback: style_description.
    const delta = ap.delta_summary || ap.style_description || '';
    const focus = (ap.topical_focus || []).map(escapeHtml).join(', ');
    const purposes = (ap.use_purposes || []).map(escapeHtml).join(', ');
    const zones = (ap.friend_zones || []).map(escapeHtml).join(', ');
    const ctxs = (ap.chatbot_contexts || []).map(escapeHtml).join(', ');
    const freq = ap.posting_frequency || '';
    const audience = ap.audience_type || '';
    const audienceLens = ap.audience_lens || '';
    const audienceDesign = ap.audience_design_note || '';
    let pills = '';
    if (freq) pills += `<span class="app-persona-pill">${{escapeHtml(freq)}}</span>`;
    if (audience) pills += `<span class="app-persona-pill">${{escapeHtml(audience)}} audience</span>`;

    // Layer-3 active selection (subsets of repertoire)
    const stanceChips  = (ap.active_stances || []).map(s => `<span class="stance-chip">${{escapeHtml(s)}}</span>`).join(' ');
    const regChips     = (ap.active_registers || []).map(s => `<span class="stance-chip">${{escapeHtml(s)}}</span>`).join(' ');
    const genreChips   = (ap.active_speech_genres || []).map(s => `<span class="stance-chip">${{escapeHtml(s)}}</span>`).join(' ');

    // New schema: surface. Legacy fallback: expression.
    const expr = ap.surface || ap.expression || {{}};
    let exprHtml = '';
    if (expr && Object.keys(expr).length) {{
      const exprRows = [];
      if (expr.effort_level)          exprRows.push(`<li><span class="app-persona-key">effort:</span> ${{escapeHtml(expr.effort_level)}}</li>`);
      if (expr.length_band)           exprRows.push(`<li><span class="app-persona-key">length:</span> ${{escapeHtml(String(expr.length_band))}} chars</li>`);
      if (expr.emoji_intensity_shift !== undefined && expr.emoji_intensity_shift !== null) {{
        const shift = expr.emoji_intensity_shift;
        const shiftLabel = shift === 0 ? '0 (default)' : (shift > 0 ? `+${{shift}}` : String(shift));
        exprRows.push(`<li><span class="app-persona-key">emoji shift:</span> ${{escapeHtml(shiftLabel)}}</li>`);
      }}
      if (expr.disclosure_depth)      exprRows.push(`<li><span class="app-persona-key">disclosure depth:</span> ${{escapeHtml(expr.disclosure_depth)}}</li>`);
      if (expr.audience_self_censoring) exprRows.push(`<li><span class="app-persona-key">audience self-censoring:</span> ${{escapeHtml(expr.audience_self_censoring)}}</li>`);
      // emoji_topic_filter intentionally NOT rendered — structural noise.
      if (ap.app_avoid)                 exprRows.push(`<li class="app-persona-avoid"><span class="app-persona-key">app avoid:</span> ${{escapeHtml(ap.app_avoid)}}</li>`);
      if (exprRows.length) exprHtml = `<ul class="app-persona-sig">${{exprRows.join('')}}</ul>`;
    }} else if (ap.app_avoid) {{
      exprHtml = `<ul class="app-persona-sig"><li class="app-persona-avoid"><span class="app-persona-key">app avoid:</span> ${{escapeHtml(ap.app_avoid)}}</li></ul>`;
    }}

    // New schema: idiolect_overrides. Legacy fallback: overrides.
    const ov = ap.idiolect_overrides || ap.overrides || {{}};
    let ovHtml = '';
    if (ov && Object.keys(ov).length) {{
      const ovRows = Object.keys(ov).map(key => {{
        const val = ov[key];
        const valStr = Array.isArray(val) ? val.map(escapeHtml).join('; ') : escapeHtml(String(val));
        return `<li><span class="app-persona-key">${{escapeHtml(key)}}:</span> ${{valStr}}</li>`;
      }});
      ovHtml = `
        <div class="app-persona-row" style="margin-top:6px;font-style:italic;opacity:0.8;">RARE — code-switching on this app (deviates from base idiolect):</div>
        <ul class="app-persona-sig">${{ovRows.join('')}}</ul>`;
    }}

    // Backward-compat fallback: if a legacy backend still has voice_signature
    // (no user_voice / no expression), render it like before so old data still
    // displays. New backends will always populate expression instead.
    const legacySig = ap.voice_signature || {{}};
    let legacyHtml = '';
    if (!exprHtml && !uvHtml && legacySig && (legacySig.capitalization || legacySig.sentence_shape || (legacySig.recurring_phrases||[]).length)) {{
      const sigRows = [];
      if (legacySig.capitalization)     sigRows.push(`<li><span class="app-persona-key">caps:</span> ${{escapeHtml(legacySig.capitalization)}}</li>`);
      if (legacySig.punctuation_habits) sigRows.push(`<li><span class="app-persona-key">punctuation:</span> ${{escapeHtml(legacySig.punctuation_habits)}}</li>`);
      if (legacySig.sentence_shape)     sigRows.push(`<li><span class="app-persona-key">sentence shape:</span> ${{escapeHtml(legacySig.sentence_shape)}}</li>`);
      if (legacySig.length_chars)       sigRows.push(`<li><span class="app-persona-key">length:</span> ~${{escapeHtml(String(legacySig.length_chars))}} chars</li>`);
      if ((legacySig.recurring_phrases||[]).length) {{
        const phrases = legacySig.recurring_phrases.map(p => `"${{escapeHtml(p)}}"`).join(', ');
        sigRows.push(`<li><span class="app-persona-key">recurring phrases:</span> ${{phrases}}</li>`);
      }}
      if (legacySig.emoji_policy) {{
        let emojiTxt;
        if (typeof legacySig.emoji_policy === 'string') {{
          emojiTxt = escapeHtml(legacySig.emoji_policy);
        }} else {{
          const elist = (legacySig.emoji_policy.emojis || []).join(' ');
          const place = legacySig.emoji_policy.placement || '';
          emojiTxt = `${{escapeHtml(elist)}}${{place ? ' (' + escapeHtml(place) + ')' : ''}}`;
        }}
        if (emojiTxt) sigRows.push(`<li><span class="app-persona-key">emoji (legacy):</span> ${{emojiTxt}}</li>`);
      }}
      if (legacySig.hashtag_policy) sigRows.push(`<li><span class="app-persona-key">hashtags:</span> ${{escapeHtml(legacySig.hashtag_policy)}}</li>`);
      if ((legacySig.forbidden_patterns||[]).length) {{
        const fb = legacySig.forbidden_patterns.map(escapeHtml).join('; ');
        sigRows.push(`<li><span class="app-persona-key">never does:</span> ${{fb}}</li>`);
      }}
      if (sigRows.length) legacyHtml = `<ul class="app-persona-sig">${{sigRows.join('')}}</ul>`;
    }}

    html += `
      <div class="app-persona-card">
        <div class="app-persona-header">${{escapeHtml(k)}}${{pills ? ' ' + pills : ''}}</div>
        ${{audienceLens ? `<div class="app-persona-row"><span class="app-persona-key">audience lens:</span> ${{escapeHtml(audienceLens)}}</div>` : ''}}
        ${{audienceDesign ? `<div class="app-persona-row"><span class="app-persona-key">audience design (Bell):</span> ${{escapeHtml(audienceDesign)}}</div>` : ''}}
        ${{stanceChips ? `<div class="app-persona-row"><span class="app-persona-key">active stances:</span> ${{stanceChips}}</div>` : ''}}
        ${{regChips ? `<div class="app-persona-row"><span class="app-persona-key">active registers:</span> ${{regChips}}</div>` : ''}}
        ${{genreChips ? `<div class="app-persona-row"><span class="app-persona-key">active speech genres:</span> ${{genreChips}}</div>` : ''}}
        ${{delta ? `<div class="app-persona-style">${{escapeHtml(delta)}}</div>` : ''}}
        ${{exprHtml}}
        ${{focus ? `<div class="app-persona-row"><span class="app-persona-key">Topical focus:</span> ${{focus}}</div>` : ''}}
        ${{purposes ? `<div class="app-persona-row"><span class="app-persona-key">Use purposes:</span> ${{purposes}}</div>` : ''}}
        ${{zones ? `<div class="app-persona-row"><span class="app-persona-key">Friend zones:</span> ${{zones}}</div>` : ''}}
        ${{ctxs ? `<div class="app-persona-row"><span class="app-persona-key">Chatbot contexts:</span> ${{ctxs}}</div>` : ''}}
        ${{ovHtml}}
        ${{legacyHtml}}
      </div>`;
  }});
  html += '</div>';
  aps.innerHTML = html;
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

// Wrap each voice-evidence span (case-insensitive substring match) inside the
// already-escaped text in <strong> tags. Used for compose-task Example
// Responses so a reviewer can immediately see WHY a voice_mismatch foil fails.
// Spans are sorted longest-first by the extractor; this loop preserves that
// order so longer phrases substitute before sub-phrases get matched.
// Regex-free implementation: scan the lowercased haystack for each span
// (also lowercased) and rebuild the output preserving the original casing.
function boldVoiceEvidence(escapedText, spans) {{
  if (!escapedText || !Array.isArray(spans) || spans.length === 0) return escapedText;
  let out = escapedText;
  for (const span of spans) {{
    if (!span) continue;
    const escSpan = escapeHtml(span);
    if (!escSpan) continue;
    const lowerOut = out.toLowerCase();
    const lowerSpan = escSpan.toLowerCase();
    let i = 0;
    const parts = [];
    while (true) {{
      const j = lowerOut.indexOf(lowerSpan, i);
      if (j === -1) {{ parts.push(out.slice(i)); break; }}
      parts.push(out.slice(i, j));
      parts.push('<strong>');
      parts.push(out.slice(j, j + escSpan.length));
      parts.push('</strong>');
      i = j + escSpan.length;
    }}
    out = parts.join('');
  }}
  return out;
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
    // DM events render with the shared chat-bubble layout (same one used
    // by the AI Chatbot conversation), so DMs and chatbot turns are
    // visually consistent. Self → user-bubble (right, blue), friend or
    // stranger → assistant-bubble (left, gray). Each bubble may carry a
    // forwarded content block (post / image / short video) — call
    // renderContent recursively on m.forwarded_content. Plain non-DM text
    // posts fall through to the body block below.
    const messages = (Array.isArray(ev.messages) && ev.messages.length > 0)
      ? ev.messages
      : (Array.isArray(content.messages) && content.messages.length > 0
         ? content.messages : null);
    if (messages && messages.length > 0) {{
      const bubbles = messages.map(m => {{
        const isSelf = m.sender === 'self';
        let roleLabel;
        if (isSelf) {{
          roleLabel = 'you';
        }} else if (typeof m.sender === 'string' && m.sender.startsWith('stranger')) {{
          roleLabel = 'stranger';
        }} else {{
          roleLabel = 'friend';
        }}
        const reactionOnly = m.reaction_emoji && !m.text;
        const sideCls = isSelf ? 'user-bubble' : 'assistant-bubble';
        const extraCls = reactionOnly ? ' dm-reaction-only' : '';
        const textBlock = m.text
          ? `<div class="dm-text">${{escapeHtml(m.text)}}</div>`
          : '';
        const reactBlock = m.reaction_emoji
          ? `<div class="dm-reaction">${{escapeHtml(m.reaction_emoji)}}</div>`
          : '';
        const fwdBlock = m.forwarded_content
          ? `<div class="dm-forwarded">${{renderContent({{
              content_type: m.forwarded_content.content_type,
              content: m.forwarded_content.content,
              messages: null,
            }})}}</div>`
          : '';
        return `<div class="chat-bubble dm-bubble ${{sideCls}}${{extraCls}}"><div class="chat-role">${{escapeHtml(roleLabel)}}</div>${{textBlock}}${{reactBlock}}${{fwdBlock}}</div>`;
      }}).join('');
      // DM threads omit the outer "text" content_type header — every DM
      // is text by definition; the inner forwarded-content blocks
      // (rendered recursively via renderContent) keep their own
      // type labels (e.g. "short video", "image") which are informative.
      return `<div class="chat-thread dm-thread">${{bubbles}}</div>${{adMetaHtml}}`;
    }}
    // Plain text post (non-DM): single body block.
    const body = content.text || content.caption || '';
    return `<div class="content-block">${{header}}<div class="c-text-body">${{escapeHtml(body)}}</div>${{adMetaHtml}}</div>`;
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
// Parse "HH:MM, MM/DD/YYYY" -> unix seconds (UTC). Used to filter
// update_history entries that lack a numeric `timestamp` field.
function _parseFormattedTs(s) {{
  if (!s || typeof s !== 'string') return 0;
  const m = s.match(/^(\d{{1,2}}):(\d{{2}}),\s*(\d{{1,2}})\/(\d{{1,2}})\/(\d{{4}})$/);
  if (!m) return 0;
  const [_, hh, mm, mo, dd, yyyy] = m;
  return Math.floor(Date.UTC(+yyyy, +mo - 1, +dd, +hh, +mm, 0) / 1000);
}}

function renderUpdateHistory(history, asOfTs) {{
  if (!history || !history.length) return '';
  // Filter rules for what shows up on a per-event render:
  //   1. Cross-ref entries (entries with a `resolution` field —
  //      "suppressed_weak_minority", "different_granularity", etc.) are
  //      GLOBAL findings about how this canonical relates to other
  //      canonicals across the whole persona. They don't represent events
  //      that happened at any specific time. We hide them from per-event
  //      renders entirely; they belong on a canonical-preference detail view.
  //   2. Temporal entries (`reinforced`, `new`, `faded`) ARE events. Show
  //      only those whose timestamp <= asOfTs (parse formatted_timestamp
  //      when no numeric ts is set).
  let visible = history.filter(h => !h.resolution);
  if (typeof asOfTs === 'number' && asOfTs > 0) {{
    visible = visible.filter(h => {{
      const ht = h.timestamp || h.ts || _parseFormattedTs(h.formatted_timestamp);
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
      const calCard = div.firstElementChild;
      // Calendar mods are user-meta, not app-specific. They show under
      // "All" only — per-app and "test queries only" filters hide them.
      calCard.dataset.kind = 'calendar';
      calCard.dataset.app = '';
      grid.appendChild(calCard);
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

      // Test-card spec: only the 5 user-facing labels render —
      //   1. User Query (rendered separately below the metadata strip)
      //   2. Example Response
      //   3. Inferior Response
      //   4. Groundtruth Preference (just the preference itself)
      //   5. Rubric dimensions
      // Plus TWO task-essential extras when populated and informative:
      //   • Tool Call — for agentic write tasks where the gold IS a tool call
      //   • Candidate pool — for ranking tasks where the gold IS picking from a list
      // Everything else (Held-out preference, Other supporting preferences,
      // Correct but irrelevant preferences, Cross-signal evidence, Irrelevant
      // prefs, Carve-out, Meta) is intentionally NOT rendered: it's either
      // redundant with Groundtruth Preference or grader-internal context.
      const isAgenticWrite = (t.task_type || '').match(/^agentic_(auto_reply|cross_app_repost|composed_post|send_post)$/);
      const isRanking = (t.task_type || '').match(/^(personalized_feed_ranking|personalized_recommendation|at_ai_directive_followup|short_vs_long_term_lifecycle)$/);

      let sections = '';
      if (t.example_response) {{
        sections += `<div class="ts-section"><div class="ts-label">Example Response</div><div class="ts-body" style="white-space:pre-wrap;">${{escapeHtml(t.example_response)}}</div></div>`;
      }}
      if (t.inferior_response && t.inferior_response.text) {{
        const flaw = t.inferior_response.flaw_kind || '';
        // The voice-evidence smoke check status is build-time QA metadata
        // (`voice_evidence_smoke_check`) — kept on the instance for debugging
        // but intentionally NOT rendered on the user-facing test card.
        const smoke = '';
        const regen = (t.inferior_response.regen_reason)
          ? ` <small style="color:#92400E;font-weight:normal;">(regen: ${{escapeHtml(t.inferior_response.regen_reason)}})</small>` : '';
        // Highlight violations (only daily_personalized_briefing for now —
        // the disliked_recent flaw injects a specific topic into the gold,
        // so the topic_hint / persona_item from flaw_evidence pinpoints
        // exactly what the agent should NOT have surfaced. Render-only
        // bolding; the underlying inferior_response.text is unmodified.
        let infBody = escapeHtml(t.inferior_response.text);
        if (t.task_type === 'daily_personalized_briefing') {{
          const ev = t.inferior_response.flaw_evidence || {{}};
          const spans = [];
          if (ev.topic_hint) spans.push(ev.topic_hint);
          if (ev.persona_item && ev.persona_item !== ev.topic_hint) spans.push(ev.persona_item);
          // Sort longest first so super-strings substitute before sub-strings.
          spans.sort((a, b) => b.length - a.length);
          if (spans.length > 0) infBody = boldVoiceEvidence(infBody, spans);
        }}
        const violationHint = (t.task_type === 'daily_personalized_briefing'
                               && t.inferior_response.flaw_evidence
                               && (t.inferior_response.flaw_evidence.topic_hint
                                   || t.inferior_response.flaw_evidence.persona_item))
          ? ` <small style="color:#92400E;font-weight:normal;">(bold = violates user preferences)</small>`
          : '';
        sections += `<div class="ts-section" style="background:#FEF7E0;border-color:#FDE68A;"><div class="ts-label">Inferior Response <small style="color:#92400E;">[${{escapeHtml(flaw)}}]</small>${{smoke}}${{regen}}${{violationHint}}</div><div class="ts-body" style="white-space:pre-wrap;color:#78350F;">${{infBody}}</div></div>`;
      }}
      if (isAgenticWrite && Array.isArray(t.tool_call) && t.tool_call.length > 0) {{
        const calls = t.tool_call.map(tc => {{
          const args = tc.args ? JSON.stringify(tc.args) : '{{}}';
          return `<li><code>${{escapeHtml(tc.tool || '?')}}(${{escapeHtml(args)}})</code></li>`;
        }}).join('');
        sections += `<div class="ts-section"><div class="ts-label">Tool Call (ordered)</div><ol class="ts-list ts-mono">${{calls}}</ol></div>`;
      }}
      if (isRanking && Array.isArray(t.candidates) && t.candidates.length > 0) {{
        const items = t.candidates.map(c => {{
          const tag = `<span class="ts-origin ts-origin-${{escapeHtml(c.origin || '')}}">${{escapeHtml(c.origin || '')}}</span>`;
          const star = c.is_held_out ? ' <span class="ts-target">★ target</span>' : '';
          const delta = c.ts_delta_label
            ? ` <span class="ts-delta">${{escapeHtml(c.ts_delta_label)}}</span>`
            : '';
          const tags = (c.hashtags || []).slice(0, 4).map(h => `#${{escapeHtml(String(h).replace(/^#/, ''))}}`).join(' ');
          const title = escapeHtml(c.title || '');
          const body = [title, tags].filter(Boolean).join(' ');
          return `<li><code>idx=${{c.idx}}</code> ${{tag}}${{star}}${{delta}} ${{body}}</li>`;
        }}).join('');
        sections += `<div class="ts-section"><div class="ts-label">Candidate pool (${{t.candidates.length}} items)</div><ul class="ts-list">${{items}}</ul></div>`;
      }}
      // Groundtruth Preference — clean, just the preference itself. No
      // "Persona item:" / "Category:" labels (those are stripped at the
      // extractor in TEST_GT_EXTRACTORS). Bold spans highlight tone /
      // voice anchors that EITHER the Example Response or the Inferior
      // Response actually uses, so the reviewer can see which aspects of
      // the user's tone are leveraged by either side at a glance.
      if (t.example_response || t.groundtruth_preference) {{
        const gtEsc = escapeHtml(t.groundtruth_preference || '');
        const exSpans = Array.isArray(t.example_response_voice_evidence) ? t.example_response_voice_evidence : [];
        const infSpans = Array.isArray(t.inferior_response_voice_evidence) ? t.inferior_response_voice_evidence : [];
        // Union (case-insensitive dedup, longest first so super-strings
        // substitute before sub-strings).
        const _seen = new Set();
        const combinedSpans = [];
        for (const s of [...exSpans, ...infSpans]) {{
          if (typeof s !== 'string' || !s) continue;
          const k = s.toLowerCase();
          if (_seen.has(k)) continue;
          _seen.add(k);
          combinedSpans.push(s);
        }}
        combinedSpans.sort((a, b) => b.length - a.length);
        const gtHtml = boldVoiceEvidence(gtEsc, combinedSpans);
        const gtHint = combinedSpans.length > 0
          ? ` <small style="color:var(--text-secondary);font-weight:normal;">(bold = tone anchors used by Example or Inferior)</small>` : '';
        sections += `<div class="ts-section"><div class="ts-label">Groundtruth Preference${{gtHint}}</div><div class="ts-body" style="white-space:pre-wrap;">${{gtHtml}}</div></div>`;
      }}
      const tags = (t.rubric_tags || []).filter(Boolean);
      if (tags.length > 0) {{
        // Polarity convention:
        //   "(+) ..." → positive metric: green border, ADDS to score
        //   "(-) ..." → negative metric: amber border, REDUCES score on violation
        // Untagged strings default to positive (legacy compatibility).
        const _splitPolarity = s => {{
          const t = String(s).replace(/^\\s+/, '');
          if (t.startsWith('(+)')) return ['pos', t.slice(3).replace(/^\\s+/, '')];
          if (t.startsWith('(-)') || t.startsWith('(−)')) return ['neg', t.slice(3).replace(/^\\s+/, '')];
          return ['pos', s];
        }};
        const items = tags.map(s => {{
          const [pol, body] = _splitPolarity(s);
          const cls = pol === 'pos' ? 'rubric-pos' : 'rubric-neg';
          const sign = pol === 'pos' ? '+' : '−';
          return `<li class="${{cls}}"><span class="rubric-sign">(${{sign}})</span> ${{escapeHtml(body)}}</li>`;
        }}).join('');
        sections += `<div class="ts-section ts-rubric-bar"><div class="ts-label">Rubric dimensions</div><ul class="ts-list">${{items}}</ul></div>`;
      }}

      const card = document.createElement('div');
      card.className = 'event-card test-sample-card';
      // Filter classification: test card → kind=test, app=<inferred app>.
      card.dataset.kind = 'test';
      card.dataset.app = (t.app || '').toLowerCase();
      // Render any preceding chat turns first so the User Query has context.
      let priorBlock = '';
      if (Array.isArray(t.prior_conversation) && t.prior_conversation.length > 0) {{
        const bubbles = t.prior_conversation.map(m => {{
          const role = m.role === 'user' ? 'You' : 'AI';
          const cls = m.role === 'user' ? 'user-bubble' : 'assistant-bubble';
          return `<div class="chat-bubble ${{cls}}"><div class="chat-role">${{role}}</div>${{escapeHtml(m.content || '')}}</div>`;
        }}).join('');
        priorBlock = `<div class="ts-section"><div class="ts-label">Prior conversation (last ${{t.prior_conversation.length}} turns)</div><div class="chat-thread">${{bubbles}}</div></div>`;
      }}
      // Render User Query as a regular ts-section (label INSIDE the
      // section block) so it visually matches every other section.
      const queryBlock = `<div class="ts-section"><div class="ts-label">User Query</div><div class="ts-body">${{escapeHtml(t.query_text || '')}}</div></div>`;
      card.innerHTML = `
        <div class="event-header">
          <div class="event-meta">
            <span style="font-weight:600;color:#7B5C00;">Test sample</span> &middot;
            ${{escapeHtml(tsDisplay)}} &middot;
            ${{locText}}${{locText ? ' &middot; ' : ''}}
            <code>${{escapeHtml(t.task_type || '')}}</code>
          </div>
        </div>
        ${{priorBlock}}
        ${{queryBlock}}
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
    // Filter classification: regular event card → its app.
    card.dataset.kind = 'event';
    card.dataset.app = (app || '').toLowerCase();

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

  // -- Filter bar wiring -------------------------------------------------
  // Click a button → mark it active, hide / show timeline cards based on
  // each card's data-kind and data-app attrs. Default: "all" shows every
  // card. Per-app filters show event cards on that app + test cards
  // tagged with that app. "Test queries only" shows every test card
  // regardless of app, hides events + calendar mods.
  const filterBar = document.getElementById('filter-bar');
  if (filterBar) {{
    filterBar.addEventListener('click', e => {{
      const btn = e.target.closest('.filter-btn');
      if (!btn) return;
      const key = btn.dataset.filterKey || 'all';
      filterBar.querySelectorAll('.filter-btn').forEach(b =>
        b.classList.toggle('active', b === btn)
      );
      const cards = grid.children;
      for (const card of cards) {{
        const kind = card.dataset.kind || '';
        const app = card.dataset.app || '';
        let show;
        if (key === 'all') {{
          show = true;
        }} else if (key === 'test') {{
          show = (kind === 'test');
        }} else {{
          // Per-app: include event cards on that app + test cards
          // whose inferred app matches.
          show = (app === key && (kind === 'event' || kind === 'test'));
        }}
        card.style.display = show ? '' : 'none';
      }}
    }});
  }}
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
