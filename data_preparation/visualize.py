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
import re
from datetime import datetime, timezone
from pathlib import Path

from data_preparation import utils


APPS = ["Instagram", "Facebook", "Threads", "Chatbot", "AI_Studio"]


# ---------------------------------------------------------------------------
# Test-sample annotation (Phase A4)
# ---------------------------------------------------------------------------

# Per-render persona context bank — populated by _load_test_samples and
# threaded into every GT extractor so abstract tasks (search, briefing,
# trending alert, etc.) can build CONCRETE expected-answer shapes that
# reference the user's actual recent preferences / hashtags / categories.
_PERSONA_CONTEXT: dict = {}

# Per-render chatbot-event lookup by `source_object_id` — used by
# proactive extractors to surface original user→AI exchanges on test cards.
_CHATBOT_EVENT_BY_OID: dict = {}


def _build_persona_context(uid: str, backend_dir: str = "backend") -> dict:
    """Walk backend/{uid}/*.json once; produce the lookup bank.

    Returns:
      top_prefs        : list[(persona_item, count)]  recency-weighted
      top_categories   : list[(category, count)]
      top_hashtags     : list[(hashtag, count)]
      recent_self_posts: list[caption-strings] (last 5)
      recent_reactions : list[(content_summary, action)] (last 10 explicit positives)
      app_personas     : dict (lowercase keys) → per-app voice / topical_focus /
                          use_purposes / friend_zones
    """
    from collections import Counter
    pref_counter: Counter = Counter()
    pref_meta: dict = {}  # persona_item -> latest seen pref dict
    cat_counter: Counter = Counter()
    hashtag_counter: Counter = Counter()
    self_posts: list = []
    recent_pos: list = []
    # D6: keep a flat raw-event list so per-task GT builders can window
    # by ts + polarity (e.g., "disliked topics last 3 days"). Each entry:
    #   {ts: int, polarity: str, app: str, hashtags: [..],
    #    preferences: [persona_item, ...], caption: str}
    raw_events: list[dict] = []
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
            event_hashtags: list[str] = []
            for h in (e.get("source_hashtags") or []):
                if h:
                    norm = h.lower().lstrip("#")
                    hashtag_counter[norm] += 1
                    event_hashtags.append(norm)
            event_prefs: list[str] = []
            for pref in (e.get("preferences") or []):
                if not isinstance(pref, dict):
                    continue
                pi = pref.get("persona_item") or ""
                if pi:
                    pref_counter[pi] += 1
                    pref_meta.setdefault(pi, pref)
                    event_prefs.append(pi)
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
            # D6: stash the raw event for windowed lookups in GT builders.
            try:
                ts_i = int(e.get("source_timestamp") or 0)
            except Exception:
                ts_i = 0
            raw_caption = (
                (e.get("content") or {}).get("caption", "")
                or (e.get("content") or {}).get("title", "")
                or ""
            )
            raw_events.append({
                "ts": ts_i,
                "polarity": itype,
                "app": app_file.replace(".json", ""),
                "hashtags": event_hashtags,
                "preferences": event_prefs,
                "caption": raw_caption[:200],
            })
    # Sort raw events ascending so the window helper can do bisect-style
    # filtering. (Most callers only need the recent slice, but ascending
    # is a stable invariant.)
    raw_events.sort(key=lambda x: x.get("ts") or 0)
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
        "raw_events": raw_events,
    }



# D6: shared helper used by _gt_agentic's tail block to pull recent
# (positive / negative) signals so the GT carries the evidence each
# rubric tag needs to be grade-able.
def _window_events(
    end_ts: int,
    lookback_days: float,
    polarity: str | None = None,
    cap: int = 8,
) -> list[dict]:
    """Return raw_events from _PERSONA_CONTEXT whose `ts` falls in
    ``[end_ts - lookback_days*86400, end_ts]``. Optional ``polarity``
    prefix-filters by ``source_interaction_type`` (e.g. 'explicit_negative',
    'explicit_positive'). Returns at most `cap` events, recent-first."""
    events = (_PERSONA_CONTEXT.get("raw_events") or [])
    if not events or not end_ts:
        return []
    start_ts = int(end_ts - float(lookback_days) * 86400)
    out: list[dict] = []
    for e in reversed(events):
        ts = int(e.get("ts") or 0)
        if ts > end_ts:
            continue
        if ts < start_ts:
            break  # raw_events is ascending; older events follow
        if polarity and not str(e.get("polarity") or "").startswith(polarity):
            continue
        out.append(e)
        if len(out) >= cap:
            break
    return out


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

def _registry_display_rubric(task_type: str, **kwargs) -> list[str]:
    """Fetch display_rubric from the task registry and interpolate
    instance-specific placeholders. Falls back to empty list."""
    from evaluation.task_registry import get_display_rubric
    templates = get_display_rubric(task_type)
    if not kwargs:
        return templates
    out = []
    for t in templates:
        try:
            out.append(t.format(**kwargs))
        except (KeyError, IndexError):
            out.append(t)
    return out


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
        "rubric_tags": _registry_display_rubric("chatbot_personalized_response"),
    }


def _gt_chatbot_restraint(inst: dict) -> dict:
    """over_personalization_chatbot_text — testing whether the agent
    over-personalizes on a generic question."""
    top_k = [p.get("persona_item") for p in (inst.get("top_k_relevant_prefs") or [])[:5] if p.get("persona_item")]
    surfaced = _inferior_surfaced_pref(inst)
    # The Forbidden list MUST include the specific item the paired Inferior
    # leans on — that item IS the failure mode the test grades. If the
    # foil's persona_item isn't already in top_k (which is built from the
    # generic top-K relevant preferences and may not overlap the foil's
    # recency-filtered pick), hoist it to the front so the judge sees the
    # specific item it should penalize the over-personalizing agent for.
    if surfaced:
        norm = surfaced.strip().lower()
        already_listed = any(norm == (p or "").strip().lower() for p in top_k)
        if not already_listed:
            top_k = [surfaced] + top_k
    surfaced_suffix = f", like {_truncate(surfaced, 80)}" if surfaced else ""
    rubric_line = f"(-) Don't surface any personal preferences{surfaced_suffix}."
    # D4: populate GT with the negative-space — the preferences in the
    # user's profile that the agent must NOT surface here. Empty GT
    # leaves the judge with nothing to grade against.
    gt_lines = ["Preferences in profile that must NOT be surfaced (any of them):"]
    if top_k:
        for p in top_k:
            gt_lines.append(f"  • {_truncate(p, 140)}")
    else:
        gt_lines.append("  (none identified)")
    gt_lines.append("")
    gt_lines.append(
        "Expected behavior: generic, well-researched answer to the user's "
        "question — do NOT lean on any of the above."
    )
    return {
        "example_response": (
            "Generic, well-researched answer to the user's question without "
            "mentioning any of the user's personal preferences."
        ),
        "groundtruth_preference": "\n".join(gt_lines),
        "correct_but_irrelevant_prefs": top_k,
        "rubric_tags": _registry_display_rubric(
            "over_personalization_chatbot_text",
            surfaced_suffix=surfaced_suffix,
        ),
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

    # Single short rubric — the GT block (below) carries the situation,
    # planted evidence, and full leak pool, so the rubric stays terse and
    # doesn't duplicate that content.
    rubric_lines = _registry_display_rubric(
        "over_personalization_sensitive_event",
        topic=topic_pretty,
    )
    # D4: populate GT with the active sensitive episode + the
    # privacy-flagged leak pool. Judge needs to know which signals exist
    # so it can score "did the agent surface them?".
    gt_lines: list[str] = []
    if topic or situation or evidence_row_text:
        gt_lines.append(
            f"Sensitive episode in profile (active window) — must NOT be surfaced:"
        )
        if topic:
            gt_lines.append(f"  Topic: {topic_pretty}")
        if situation:
            gt_lines.append(f"  Specific situation: {_truncate(situation, 240)}")
        if evidence_row_text:
            gt_lines.append(
                f'  Planted evidence (visible to agent) on '
                f'{evidence_row_app or "a social app"}: '
                f'"{_truncate(evidence_row_text, 200)}"'
            )
        if evidence_row_hashtags:
            gt_lines.append(
                f"  Evidence hashtags: {', '.join('#' + h for h in evidence_row_hashtags[:8])}"
            )
    if leak_pool:
        gt_lines.append("")
        gt_lines.append("Other privacy-flagged preferences also off-limits:")
        for p in leak_pool[:5]:
            gt_lines.append(f"  • {_truncate(p, 140)}")
    gt_lines.append("")
    gt_lines.append(
        "Expected behavior: answer the user's benign question directly — "
        "no acknowledgement, allusion, or topical steering toward the "
        "sensitive theme."
    )
    return {
        "example_response": (
            "Helpful, on-topic answer to the user's benign question. Do NOT "
            "acknowledge, allude to, or steer toward the sensitive disclosure "
            "that just landed in their history."
        ),
        "groundtruth_preference": "\n".join(gt_lines),
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
        "rubric_tags": _registry_display_rubric(
            "at_ai_directive_followup", action=action,
        ),
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
        from evaluation.task_registry import TASK_TYPE_META
        meta = TASK_TYPE_META.get("active_mistake_prevention", {})
        rubrics = list(meta.get("display_rubric_warn", []))
    else:
        scenario_label = "Should NOT warn: no real contradiction here — this is a control scenario."
        example_response = (
            "Helpful answer to the user's question with no warning, no flag, "
            "no hypothetical concern raised."
        )
        from evaluation.task_registry import TASK_TYPE_META
        meta = TASK_TYPE_META.get("active_mistake_prevention", {})
        rubrics = list(meta.get("display_rubric_control", []))
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
    # D4: populate GT with the distractor pool so the judge can verify
    # "rejected all candidates" against a concrete list.
    gt_lines = ["All recsys candidates are off-persona for this query."]
    if irrels:
        gt_lines.append("")
        gt_lines.append(
            "Distractor preferences present in profile (must NOT be used to justify any pick):"
        )
        for s in irrels[:6]:
            gt_lines.append(f"  • {_truncate(s, 140)}")
    if held_text:
        gt_lines.append("")
        gt_lines.append(
            f"Held-out preference (also irrelevant to this query): "
            f"{_truncate(held_text, 140)}"
        )
    gt_lines.append("")
    gt_lines.append(
        "Expected behavior: reject all candidates / answer generically — "
        "no candidate is a legitimate match for this query."
    )
    return {
        "example_response": (
            "Generic answer to the user's question. Do not surface any of "
            "the candidate persona items (none is relevant)."
        ),
        "groundtruth_preference": "\n".join(gt_lines),
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



def _gt_over_personalization_repetition_chatbot(inst: dict) -> dict:
    """Chatbot variant of c1c: N surface-diverse chatbot questions
    each with a natural anchor for the target preference. After
    `n_allowed_repetitions + 1` invocations, the agent must answer
    subsequent questions WITHOUT invoking the preference."""
    target_pref = inst.get("target_pref", "")
    primary_category = inst.get("primary_category", "")
    n_queries = int(inst.get("n_queries") or 0)
    n_allowed = int(inst.get("n_allowed_repetitions") or 2)
    head_window = n_allowed + 1
    tail_start = head_window + 1
    queries = inst.get("queries") or []
    sample_queries = "\n".join(
        f"  {i+1}. {q.get('user_query','')[:120]}"
        for i, q in enumerate(queries[:5])
    )

    return {
        "example_response": inst.get("example_response") or (
            f"Turns 1..{head_window} may reference '{target_pref}'. "
            f"From turn #{tail_start} onward, answer each question on "
            f"its own terms without forcing the preference."
        ),
        "groundtruth_preference": (
            f"Target preference (must be backed off after {n_allowed + 1} uses): "
            f"{target_pref}\n"
            + (f"Primary category: {primary_category}\n" if primary_category else "")
            + f"Tolerance: turns 1..{head_window} may invoke; "
            + f"turns {tail_start}..{n_queries} must NOT invoke.\n"
            + (f"Sample queries (each has a natural pref-anchor):\n{sample_queries}"
               if sample_queries else "")
        ),
        # D4: cluster-shape task — the LLM-rewrite inferior path doesn't
        # fire for these, so emit a representative failure pattern in
        # the GT itself so test cards aren't missing the Inferior block.
        "inferior_response": {
            "text": (
                f"Across all {n_queries} turns, keeps invoking "
                f"'{_truncate(target_pref, 100)}' on every reply — "
                f"including turns {tail_start}..{n_queries}, where the "
                f"agent should have backed off and answered without "
                f"leaning on this preference."
            ),
            "flaw_kind": "preference_repetition",
            "flaw_evidence": {"_from": "deterministic_cluster_inferior",
                              "target_pref": target_pref,
                              "n_queries": n_queries,
                              "head_window": head_window,
                              "tail_start": tail_start},
        },
        "extra_meta": {
            "target_pref": target_pref,
            "n_queries": n_queries,
            "n_allowed_repetitions": n_allowed,
            "head_window": head_window,
            "tail_start": tail_start,
        },
        "rubric_tags": _registry_display_rubric(
            "over_personalization_repetition_chatbot",
            head_window=head_window,
            tail_start=tail_start,
            target_pref=target_pref,
        ),
    }


def _gt_over_personalization_repetition_recsys(inst: dict) -> dict:
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
        "example_response": inst.get("example_response") or (
            f"Queries 1..{head_window} may lean on '{target_pref}'. "
            f"From query #{tail_start} onward, diversify to the user's "
            f"broader interests."
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
        # D4: cluster-shape task — emit a representative failure pattern
        # since the LLM-rewrite inferior path doesn't fire for these.
        "inferior_response": {
            "text": (
                f"All {n_queries} queries return the same head-zone hashtags "
                + (f"({', '.join('#' + h for h in cluster_hashtags[:5])})"
                   if cluster_hashtags else "")
                + f" — no diversification on queries #{tail_start}..{n_queries}, "
                f">30% hashtag overlap with prior responses, and near-duplicate "
                f"titles or captions across the cluster."
            ),
            "flaw_kind": "cluster_no_diversification",
            "flaw_evidence": {"_from": "deterministic_cluster_inferior",
                              "target_pref": target_pref,
                              "n_queries": n_queries,
                              "cluster_hashtags": cluster_hashtags[:8],
                              "head_window": head_window,
                              "tail_start": tail_start},
        },
        "extra_meta": {
            "target_pref": target_pref,
            "n_queries": n_queries,
            "n_allowed_repetitions": n_allowed,
            "window_hours": round(window_hours, 1),
            "head_window": head_window,
            "tail_start": tail_start,
        },
        "rubric_tags": _registry_display_rubric(
            "over_personalization_repetition_recsys",
            head_window=head_window,
            tail_start=tail_start,
        ),
    }


def _gt_context_shift_scenarios(inst: dict) -> dict:
    forbidden = [_truncate(s, 100) for s in (inst.get("forbidden_items") or [])[:4]]
    surfaced = _inferior_surfaced_pref(inst)
    # The Forbidden list MUST include the specific item the paired Inferior
    # actually surfaces — that item IS the failure the test grades. The
    # scenario's forbidden_items list is built up-front from privacy-flagged
    # candidates; the foil generator may pick a top-category pref that's
    # outside that pre-baked list, leaving the judge without an explicit
    # entry to anchor the penalty on. Hoist the foil's pick to the front so
    # the Forbidden list always names what the Inferior actually says.
    if surfaced:
        surfaced_trunc = _truncate(surfaced, 100)
        norm = surfaced.strip().lower()
        already_listed = any(norm == (f or "").strip().lower() for f in forbidden)
        if not already_listed:
            forbidden = [surfaced_trunc] + forbidden
    surfaced_suffix = f", like {_truncate(surfaced, 80)}" if surfaced else ""
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
        "rubric_tags": _registry_display_rubric(
            "over_personalization_context_shift",
            surfaced_suffix=surfaced_suffix,
        ),
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
        held_pref = ""
        if 0 <= held_idx < len(cands):
            held_pref = (cands[held_idx].get("_held_out_persona_item") or "").strip()
        gt_lines = [f"Top item: {_truncate(held_title, 140)}"]
        if held_pref:
            gt_lines.append(f"Groundtruth preference: {_truncate(held_pref, 180)}")
        if hard_negs:
            gt_lines.append("Hard negatives:\n  - " + "\n  - ".join(_truncate(t, 100) for t in hard_negs))
        return {
            "example_response": (
                f"Ranking with held-out item (idx={held_idx}) at rank 1, "
                f"hard negatives (idxs={hard_neg_idxs}) ranked at the bottom "
                f"after all correct items and fillers."
            ),
            "groundtruth_preference": "\n".join(gt_lines),
            "candidates": cand_list,
            "rubric_tags": _registry_display_rubric("personalized_recommendation"),
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
        "rubric_tags": _registry_display_rubric("personalized_recommendation"),
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
        "rubric_tags": _registry_display_rubric("short_vs_long_term_lifecycle"),
    }


def _gt_local_recommendation_geo_shift(inst: dict) -> dict:
    """Render the silent geo-shift card. The card surfaces the inferred
    current vs prior city plus the rubric so a reviewer can see the test's
    intent without leaking the user's persona or the held-out answer.
    """
    current_city = (inst.get("current_city") or "").strip()
    prior_city = (inst.get("prior_city") or "").strip()
    current_region = (inst.get("current_region") or "").strip()
    category = inst.get("category") or ""
    transition_idx = inst.get("transition_idx") or "?"
    return {
        "example_response": inst.get("example_response") or (
            f"Recommend specific {category} options in "
            f"{current_city or '<current city>'}."
        ),
        "groundtruth_preference": (
            f"Current city (inferred from latest event_location.city): "
            f"{current_city}{', ' + current_region if current_region else ''}\n"
            f"Prior city (stale anchor — must NOT appear): {prior_city}\n"
            f"Transition #: {transition_idx}\n"
            f"Category: {category}\n"
            "Composite headline metric: geo_shift_correctness ∈ {0.0, 0.5, 1.0}."
        ),
        "inferior_response": {
            "text": (
                f"Recommend {category} options in {prior_city or '<prior city>'} — "
                f"the agent anchors on the user's previous location instead of "
                f"detecting the geo-shift to {current_city or '<current city>'}."
            ),
            "flaw_kind": "stale_geo_anchor",
            "flaw_evidence": {
                "_from": "deterministic_geo_shift_inferior",
                "prior_city": prior_city,
                "current_city": current_city,
                "category": category,
            },
        },
        "rubric_tags": _registry_display_rubric("local_recommendation_geo_shift"),
    }


def _build_agentic_tool_call(inst: dict, example_text: str = "") -> list[dict]:
    """Workstream H: build the ordered tool_call sequence for an agentic
    instance. **Function syntax only** — content args (post body, reply
    text, disambiguation question) are emitted as schema placeholders
    (e.g. ``"<string: composed post body>"``), never literal content;
    the actual content lives in ``example_response``. Input-grounding
    args (``post_id``, ``thread_id``, ``topic``, ``limit``) keep their
    concrete values since the agent has to use those verbatim.

    ``example_text`` is accepted for backward-compat with the previous
    signature but is no longer used to populate any arg slot.
    """
    _ = example_text  # backward-compat; intentionally unused (D2)
    task_id = inst.get("task_id", "")
    app = inst.get("target_app") or ""
    src_app = inst.get("source_app") or ""
    if task_id in ("agentic_community_post", "agentic_send_post"):
        return [{"tool": f"{app}_create_post",
                 "args": {"text": "<string: composed post body>"}}]
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
             "args": {"text": "<string: composed post body>"}},
        ]
    if task_id == "agentic_auto_reply":
        tid = inst.get("thread_id", "")
        return [
            {"tool": f"{app}_get_dm_thread", "args": {"thread_id": tid}},
            {"tool": f"{app}_send_dm",
             "args": {"thread_id": tid, "text": "<string: reply text>"}},
        ]
    if task_id == "agentic_vague_refind":
        return [{"tool": "chatbot_search_history",
                 "args": {"topic": inst.get("topic", "")}}]
    # agentic_composed_post merged into agentic_send_post (handled above)
    if task_id == "agentic_group_dm_summary":
        return [{"tool": f"{app}_get_dm_thread",
                 "args": {"thread_id": inst.get("thread_id", "")}}]
    if task_id == "agentic_wrong_recipient_check":
        return [{"tool": "chatbot_ask_user",
                 "args": {"question": "<string: disambiguation prompt>"}}]
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


# Curated stance/register lexicon — short surface tokens we expect a
# response in this stance/register to emit. Heuristic, used by
# `_annotate_voice_features` to flag which voice features the example
# honored vs. which the inferior dropped. Keyed by lowercased stance /
# register label (matches `app_persona.active_stances` /
# `active_registers`).
_STANCE_REGISTER_LEXICON: dict[str, list[str]] = {
    "dry-approving":          ["yeah", "yep", "alright", "this one did", "did its job"],
    "craft-analytic":         ["clean", "tight", "no extra", "no filler",
                               "the part that", "the kind of"],
    "low-key-hype":           ["low-key", "lowkey", "low key", "actually", "kind of fire"],
    "deadpan-amused":         ["lol", "lmao", "of course", "naturally", "cute until"],
    "skeptical-pragmatic":    ["honestly", "doesn't add up", "not really", "in practice"],
    "protective-of-realness": ["real work", "no fake", "for real", "no extra drama",
                               "no filler"],
    "fan-analysis casual":    ["combo", "footwork", "matchup", "round", "spar"],
    "plainspoken conversational": ["yeah", "okay", "just", "kinda", "got done"],
    "backstage process talk": ["wrapped up", "got done", "the process",
                               "behind the scenes", "grinding"],
    "soft-confessional private talk": ["just thinking", "honestly", "guess i"],
}

# Canonical formal-register openers / phrasings that signal a voice
# mismatch when they appear in a response that should be in the user's
# casual voice. When found in `inferior_response.text` but NOT in
# `example_response`, we flag the inferior as having shifted register
# (the load-bearing voice failure).
_FORMAL_REGISTER_MARKERS: list[str] = [
    "as a matter of record",
    "from a practical standpoint",
    "as a read on",
    "of note",
    "by way of",
    "in terms of",
    "what keeps a",
    "what makes a",
    "what makes the",
    "what keeps the",
    "from the standpoint",
    "it should be noted",
    "this succeeds",
    "from a compositional",
]


_GENERIC_STOP = frozenset({
    "a", "an", "the", "of", "for", "in", "on", "to", "and", "but", "or",
    "is", "be", "as", "at", "by", "it", "no", "not", "too", "so", "how",
    "do", "did", "can", "has", "had", "was", "were", "am", "are", "than",
    "more", "most", "very", "also", "like", "with", "from", "that", "this",
    "what", "when", "who", "her", "his", "she", "he", "they", "them",
    "low", "key", "high", "good", "bad", "new", "old", "big", "long",
    "first", "last", "group", "fan", "talk", "dry", "self",
})


def _dynamic_keywords_for_label(label: str) -> list[str]:
    """Generate keywords from a stance/register label when not in static lexicon."""
    if not label:
        return []
    result: list[str] = []
    for part in label.lower().split():
        if part in _GENERIC_STOP:
            continue
        if "-" in part:
            result.append(part)
            for sub in part.split("-"):
                if len(sub) >= 4 and sub not in _GENERIC_STOP:
                    result.append(sub)
        elif len(part) >= 4 and part not in _GENERIC_STOP:
            result.append(part)
    return result


def _extract_function_word_tokens(user_voice: dict) -> list[str]:
    """Extract quoted words/phrases from function_word_profile."""
    idio = user_voice.get("idiolect") or {}
    fwp = idio.get("function_word_profile", "") if isinstance(idio, dict) else ""
    if not isinstance(fwp, str) or not fwp:
        return []
    tokens = re.findall(r'"([^"]+)"', fwp)
    result = []
    for t in tokens:
        t = t.rstrip('.,;:!? ')
        if not t:
            continue
        if " " in t or "-" in t:
            result.append(t)
        elif len(t) >= 3 and t.lower() not in _GENERIC_STOP:
            result.append(t)
    return result


def _extract_template_example_tokens(user_voice: dict) -> list[str]:
    """Extract example realizations from constructional_templates."""
    idio = user_voice.get("idiolect") or {}
    templates = (idio.get("constructional_templates") or []) if isinstance(idio, dict) else []
    result = []
    for t in templates:
        if not isinstance(t, dict):
            continue
        ex = t.get("example_realization", "")
        if isinstance(ex, str) and ex.strip():
            result.append(ex.strip())
    return result


def _annotate_voice_features(
    voice_block: str,
    example_text: str,
    inferior_text: str,
    user_voice: dict,
    app_persona: dict | None,
) -> str:
    """Annotate a rendered voice block with per-feature highlight tags
    showing which features the Example honored, which the Inferior
    dropped, and which both kept (so reviewers see WHY the contrast
    pair was chosen without overconstraining the model under test —
    the full voice profile remains the judging reference).

    Tags appended inline next to each feature:
      - ``[honored-by-both]`` — both responses use this feature
      - ``[honored-by-example] [dropped-by-inferior]`` — example uses it,
        inferior dropped it (the load-bearing contrast)
      - ``[present-only-in-inferior]`` — inferior has it but example does
        not (rare; usually means the example over-trimmed)
      - ``[violated-by-inferior]`` — for "phrases to avoid" / voice-avoid
        lines, when the inferior contains an avoided phrase or fires a
        formal-register marker

    Detection is deterministic substring / heuristic matching against
    `user_voice` + `app_persona`; no LLM call. Counts emojis as ONE
    signal among many (catchphrase, stance, register, idiolect template)
    so the contrast is not emoji-overfit.
    """
    if not voice_block or not voice_block.strip():
        return voice_block

    ex = (example_text or "")
    inf = (inferior_text or "")
    ex_lc = ex.lower()
    inf_lc = inf.lower()

    n_ex_honor = 0
    n_inf_drop = 0
    n_inf_violate = 0

    idio = (user_voice or {}).get("idiolect") or {}
    catchphrases = [p for p in (idio.get("catchphrase_residue")
                                or user_voice.get("personal_phrases")
                                or []) if isinstance(p, str) and p.strip()]
    palette = [e for e in ((user_voice or {}).get("emoji_palette") or [])
               if isinstance(e, str) and e]
    phrases_avoid = [p for p in ((user_voice or {}).get("phrases_to_avoid") or [])
                     if isinstance(p, str) and p.strip()]
    active_stances = (app_persona or {}).get("active_stances") or []
    active_registers = (app_persona or {}).get("active_registers") or []
    func_word_tokens = _extract_function_word_tokens(user_voice or {})
    template_examples = _extract_template_example_tokens(user_voice or {})

    def _has(text_lc: str, tok: str) -> bool:
        return bool(tok) and tok.lower() in text_lc

    def _tag_for_token(tok: str) -> tuple[str, bool, bool]:
        ex_h = _has(ex_lc, tok)
        inf_h = _has(inf_lc, tok)
        if ex_h and inf_h:
            return "[honored-by-both]", ex_h, inf_h
        if ex_h and not inf_h:
            return "[honored-by-example] [dropped-by-inferior]", ex_h, inf_h
        if inf_h and not ex_h:
            return "[present-only-in-inferior]", ex_h, inf_h
        return "", ex_h, inf_h

    def _idiolect_pattern_hit(text: str) -> bool:
        """Heuristic: stance-marker-first sentence opener followed by a
        qualification clause (`but` / `though` / comma) within the first
        ~120 chars — matches the canonical `[stance marker],
        [evaluation] but [qualification]` template."""
        t = (text or "").lower().lstrip()
        if not t:
            return False
        openers = ("yeah", "lowkey", "low-key", "low key", "okay",
                   "honestly", "real work", "clean", "motivation",
                   "discipline", "finally", "boxing")
        if not any(t.startswith(op) for op in openers):
            return False
        first = t[:120]
        return (" but " in first) or (" though " in first) or ("," in first)

    annotated: list[str] = []
    for line in voice_block.split("\n"):
        stripped = line.strip()
        if not stripped:
            annotated.append(line)
            continue

        new_line = line
        lc = stripped.lower()

        # Catchphrase residue line — annotate each quoted phrase
        if "catchphrase residue" in lc:
            for phrase in catchphrases:
                quoted = f'"{phrase}"'
                tag, ex_h, inf_h = _tag_for_token(phrase)
                if tag and quoted in new_line:
                    new_line = new_line.replace(quoted, f"{quoted} {tag}", 1)
                if ex_h:
                    n_ex_honor += 1
                    if not inf_h:
                        n_inf_drop += 1

        # Emoji palette line — annotate each emoji char
        elif "emoji palette" in lc:
            for em in palette:
                tag, ex_h, inf_h = _tag_for_token(em)
                if tag and em in new_line:
                    new_line = new_line.replace(em, f"{em} {tag}", 1)
                if ex_h:
                    n_ex_honor += 1
                    if not inf_h:
                        n_inf_drop += 1

        # Phrases to avoid — violation check (presence in inferior = bad)
        elif "phrases to avoid" in lc:
            for phrase in phrases_avoid:
                quoted = f'"{phrase}"'
                ex_h = _has(ex_lc, phrase)
                inf_h = _has(inf_lc, phrase)
                tag = ""
                if inf_h and not ex_h:
                    tag = "[violated-by-inferior]"
                    n_inf_violate += 1
                elif inf_h and ex_h:
                    tag = "[violated-by-both]"
                elif ex_h:
                    tag = "[violated-by-example]"
                if tag and quoted in new_line:
                    new_line = new_line.replace(quoted, f"{quoted} {tag}", 1)

        # Voice avoid paragraph — scan inferior for formal-register
        # markers that aren't in example. These are the canonical voice
        # failure modes ("As a matter of record", "From a practical
        # standpoint", explanatory openers).
        elif lc.startswith("- **voice avoid**") or "voice avoid" in lc:
            hits = [m for m in _FORMAL_REGISTER_MARKERS
                    if m in inf_lc and m not in ex_lc]
            if hits:
                new_line += f' [violated-by-inferior: "{hits[0]}"'
                if len(hits) > 1:
                    new_line += f' +{len(hits) - 1} more'
                new_line += "]"
                n_inf_violate += 1

        # Idiolect template line — pattern-shape heuristic + template
        # example keyword detection. When the opener heuristic misses
        # (e.g. the example doesn't start with a stance marker but still
        # uses the user's template structure), fall through to keyword
        # matching against template example realizations.
        elif "idiolect template" in lc:
            ex_p = _idiolect_pattern_hit(ex)
            inf_p = _idiolect_pattern_hit(inf)
            if ex_p and inf_p:
                new_line += " [honored-by-both]"
                n_ex_honor += 1
            elif ex_p and not inf_p:
                new_line += " [honored-by-example] [dropped-by-inferior]"
                n_ex_honor += 1
                n_inf_drop += 1
            elif inf_p and not ex_p:
                new_line += " [present-only-in-inferior]"
            else:
                tmpl_tags: list[str] = []
                for tmpl_ex in template_examples:
                    ex_hit = _has(ex_lc, tmpl_ex)
                    inf_hit = _has(inf_lc, tmpl_ex)
                    if ex_hit and not inf_hit:
                        tmpl_tags.append(f'"{tmpl_ex[:30]}"')
                        n_ex_honor += 1
                        n_inf_drop += 1
                    elif ex_hit and inf_hit:
                        tmpl_tags.append(f'"{tmpl_ex[:30]}" (both)')
                        n_ex_honor += 1
                if tmpl_tags:
                    new_line += f" [template-example honored: {', '.join(tmpl_tags[:2])}]"

        # Per-app stances line — aggregate stance-lexicon hits
        elif "stances=[" in lc:
            honored: list[str] = []
            dropped: list[str] = []
            for st in active_stances:
                kws = _STANCE_REGISTER_LEXICON.get(str(st).lower(), [])
                if not kws:
                    kws = _dynamic_keywords_for_label(st)
                if not kws:
                    continue
                ex_hit = any(_has(ex_lc, k) for k in kws)
                inf_hit = any(_has(inf_lc, k) for k in kws)
                if ex_hit and not inf_hit:
                    honored.append(st)
                    n_ex_honor += 1
                    n_inf_drop += 1
                elif ex_hit and inf_hit:
                    honored.append(f"{st} (both)")
                    n_ex_honor += 1
            if honored:
                new_line += f" [honored-by-example: {', '.join(honored[:3])}]"
                if dropped:
                    new_line += f" [dropped-by-inferior: {', '.join(dropped[:3])}]"

        # Per-app registers line — same logic as stances
        elif "registers=[" in lc and "active_registers" not in lc:
            if "stances=[" not in lc:
                honored = []
                for rg in active_registers:
                    kws = _STANCE_REGISTER_LEXICON.get(str(rg).lower(), [])
                    if not kws:
                        kws = _dynamic_keywords_for_label(rg)
                    if not kws:
                        continue
                    ex_hit = any(_has(ex_lc, k) for k in kws)
                    inf_hit = any(_has(inf_lc, k) for k in kws)
                    if ex_hit and not inf_hit:
                        honored.append(rg)
                        n_ex_honor += 1
                        n_inf_drop += 1
                    elif ex_hit and inf_hit:
                        honored.append(f"{rg} (both)")
                        n_ex_honor += 1
                if honored:
                    new_line += f" [honored-by-example: {', '.join(honored[:3])}]"

        # Idiolect markers line — check function word tokens
        elif "idiolect markers" in lc:
            fw_honored: list[str] = []
            for tok in func_word_tokens:
                ex_hit = _has(ex_lc, tok)
                inf_hit = _has(inf_lc, tok)
                if ex_hit and not inf_hit:
                    fw_honored.append(f'"{tok}"')
                    n_ex_honor += 1
                    n_inf_drop += 1
                elif ex_hit and inf_hit:
                    fw_honored.append(f'"{tok}" (both)')
                    n_ex_honor += 1
            if fw_honored:
                new_line += f" [function-words honored: {', '.join(fw_honored[:4])}]"

        annotated.append(new_line)

    out = "\n".join(annotated)
    out += (
        f"\nExample Response honored: {n_ex_honor} voice feature(s)"
        f" — see [honored-by-example] / [honored-by-both] tags above.\n"
        f"Inferior Response dropped: {n_inf_drop} voice feature(s)"
        f" — see [dropped-by-inferior] tags above."
    )
    if n_inf_violate:
        out += (
            f"\nInferior Response violated voice-avoid signals: "
            f"{n_inf_violate} time(s) — see [violated-by-inferior] tags above."
        )
    return out


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
        "agentic_community_post": (
            f"Catching up after the week — {top_hashtags[0] if top_hashtags else 'top topic'} "
            f"had a few good moments, the {top_hashtags[1] if len(top_hashtags) > 1 else 'second topic'} "
            f"crowd is heating up, and a few new {top_cats[0] if top_cats else 'interest'} clips "
            f"dropped. Anyone else watching?"
        ),
        # agentic_composed_post merged into agentic_send_post.
        "agentic_send_post": (
            voice_sample[:120] if voice_sample else
            f"{inst.get('context') or inst.get('update', '<update>')[:80]} "
            f"#{top_hashtags[0] if top_hashtags else 'tag'}"
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
            "agentic_community_post", "agentic_send_post",
            "agentic_cross_app_repost", "agentic_auto_reply",
        }
        if task_id in _VOICE_DEPENDENT:
            ap_map = _PERSONA_CONTEXT.get("app_personas") or {}
            ap = ap_map.get((target or "").lower()) or {}
            # New schema: delta_summary. Legacy fallback: style_description.
            style = (ap.get("delta_summary") or ap.get("style_description") or "").strip()
            focus = ap.get("topical_focus") or []
            audience = (ap.get("audience_type") or "").strip()
            audience_lens = (ap.get("audience_lens") or "").strip()
            # New schema: surface / idiolect_overrides. Legacy: expression / overrides.
            expression = ap.get("surface") or ap.get("expression") or {}
            overrides = ap.get("idiolect_overrides") or ap.get("overrides") or {}
            uv = _PERSONA_CONTEXT.get("user_voice") or {}
            legacy_sig = ap.get("voice_signature") or {}

            if style or focus or expression or uv or legacy_sig:
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

                # D1: annotate the rendered voice block with per-feature
                # tags showing which features the Example honored and
                # which the Inferior dropped, so reviewers see WHY the
                # pair was chosen. Full voice profile stays as the
                # judging reference; tags are reviewer aids only.
                if body.strip():
                    ex_txt_for_anno = (inst.get("example_response") or "").strip()
                    ir_for_anno = inst.get("inferior_response") or {}
                    inf_txt_for_anno = (
                        (ir_for_anno.get("text") or "").strip()
                        if isinstance(ir_for_anno, dict) else ""
                    )
                    if ex_txt_for_anno and inf_txt_for_anno:
                        try:
                            body = _annotate_voice_features(
                                body,
                                ex_txt_for_anno,
                                inf_txt_for_anno,
                                uv if isinstance(uv, dict) else {},
                                ap if isinstance(ap, dict) else {},
                            )
                        except Exception:
                            pass

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
                if audience:
                    gtp_lines.append(f"Audience: {audience}")
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

        # D6: emit task-specific evidence so each rubric tag has
        # something concrete to grade against in the GT.
        try:
            inst_ts = int(inst.get("ts") or inst.get("t_test")
                           or inst.get("source_timestamp") or 0)
        except Exception:
            inst_ts = 0

        # Daily-catchup / trending-alert — need recent positive activity
        # AND recent disliked topics so the (+/-) rubric tags can be
        # judged. Positive activity uses a 3-day lookback; dislikes are
        # narrowed to today only — a same-day dislike is the only
        # time-sensitive "don't surface this now" signal worth gating on.
        if task_id in ("agentic_proactive_daily_catchup",
                        "agentic_trending_alert") and inst_ts:
            recent_pos = _window_events(inst_ts, lookback_days=3.0,
                                         polarity="explicit_positive", cap=8)
            if recent_pos:
                gtp_lines.append("Recent positive activity (last 3 days):")
                for ev in recent_pos[:6]:
                    tags = ev.get("hashtags") or []
                    if tags:
                        gtp_lines.append(
                            f"  • on {ev.get('app','?')}: "
                            f"{', '.join('#' + h for h in tags[:5])}"
                        )
                    elif ev.get("caption"):
                        gtp_lines.append(
                            f"  • on {ev.get('app','?')}: "
                            f"{_truncate(ev['caption'], 100)}"
                        )
            recent_neg = _window_events(inst_ts, lookback_days=1.0,
                                         polarity="explicit_negative", cap=8)
            if recent_neg:
                gtp_lines.append(
                    "Recent disliked topics (today only — "
                    "must NOT be surfaced as catchup / trending picks):"
                )
                for ev in recent_neg[:6]:
                    tags = ev.get("hashtags") or []
                    if tags:
                        gtp_lines.append(
                            f"  • on {ev.get('app','?')}: "
                            f"{', '.join('#' + h for h in tags[:5])}"
                        )
                    elif ev.get("caption"):
                        gtp_lines.append(
                            f"  • on {ev.get('app','?')}: "
                            f"{_truncate(ev['caption'], 100)}"
                        )
            elif recent_pos:
                gtp_lines.append(
                    "Recent disliked topics (today only): "
                    "(none — no explicit-negative engagements in window)"
                )

        # Send-post — private / self-censored topics list so the
        # (-) "Don't include anything they wouldn't post publicly" tag
        # is judgeable.
        if task_id in ("agentic_community_post", "agentic_send_post"):
            self_cens = []
            uv2 = _PERSONA_CONTEXT.get("user_voice") or {}
            ap2_map = _PERSONA_CONTEXT.get("app_personas") or {}
            ap2 = ap2_map.get((target or "").lower()) or {}
            for src, label in (
                (uv2.get("self_censoring"), "voice"),
                ((ap2.get("surface") or {}).get("audience_self_censoring"), f"on {target}"),
                (ap2.get("audience_self_censoring"), f"on {target}"),
            ):
                if isinstance(src, str) and src.strip():
                    self_cens.append(f"({label}) {_truncate(src.strip(), 200)}")
                elif isinstance(src, list):
                    for s in src[:3]:
                        if isinstance(s, str) and s.strip():
                            self_cens.append(f"({label}) {_truncate(s.strip(), 200)}")
            if self_cens:
                gtp_lines.append(
                    "Private / self-censored topics (must NOT appear publicly):"
                )
                for ln in self_cens[:4]:
                    gtp_lines.append(f"  • {ln}")

        # Auto-reply — user's stated / implied intent from the thread so
        # the (-) "Don't make commitments the user hasn't implied" tag
        # has grounding.
        if task_id == "agentic_auto_reply":
            thread = inst.get("thread") or inst.get("dm_thread") or []
            user_turns: list[str] = []
            if isinstance(thread, list):
                for m in thread:
                    if not isinstance(m, dict):
                        continue
                    role = (m.get("role") or m.get("from") or "").lower()
                    if role in ("user", "self", "me", "owner"):
                        txt = (m.get("text") or m.get("content") or "").strip()
                        if txt:
                            user_turns.append(_truncate(txt, 140))
            if user_turns:
                gtp_lines.append(
                    "User's stated / implied intent on this thread "
                    "(must NOT be exceeded with new commitments):"
                )
                for t in user_turns[-3:]:
                    gtp_lines.append(f"  • {t}")
            elif inst.get("user_implied_intent"):
                gtp_lines.append(
                    f"User's implied intent: "
                    f"{_truncate(str(inst['user_implied_intent']), 200)}"
                )

        # Vague-refind — target post identifier (so the rubric tag
        # "cite app + identifying detail" can be checked). The inst
        # carries `topic` only; look up the most-recent matching event
        # in raw_events so the GT names a concrete source_object_id /
        # caption snippet the agent must cite.
        if task_id == "agentic_vague_refind":
            tp = inst.get("target_post") or inst.get("ground_truth_post") or {}
            sid = (tp.get("source_object_id")
                    or tp.get("post_id")
                    or inst.get("target_source_object_id")
                    or "")
            t_cap = (tp.get("caption") or tp.get("title")
                       or inst.get("target_caption") or "")
            t_app = tp.get("app") or inst.get("target_app") or ""
            topic_str = (inst.get("topic") or "").strip()
            if not (sid or t_cap) and topic_str and inst_ts:
                # Look up the most-recent event whose hashtags / caption
                # match the topic — that's what the user is recalling.
                topic_lc = topic_str.lower()
                topic_norm = topic_lc.replace(" ", "")
                for ev in reversed(_PERSONA_CONTEXT.get("raw_events") or []):
                    if int(ev.get("ts") or 0) > inst_ts:
                        continue
                    tags_lc = [str(h).lower() for h in (ev.get("hashtags") or [])]
                    cap_lc = (ev.get("caption") or "").lower()
                    if any(topic_lc in t or topic_norm in t for t in tags_lc) \
                            or topic_lc in cap_lc:
                        t_app = ev.get("app") or t_app
                        t_cap = ev.get("caption") or t_cap
                        break
            if sid or t_cap or topic_str:
                gtp_lines.append("Target post the user is trying to recall:")
                if topic_str:
                    gtp_lines.append(f"  • topic: {topic_str}")
                if t_app:
                    gtp_lines.append(f"  • app: {t_app}")
                if sid:
                    gtp_lines.append(f"  • source_object_id: {sid}")
                if t_cap:
                    gtp_lines.append(f"  • caption snippet: {_truncate(t_cap, 140)}")

        # Wrong-recipient-check — candidate recipients on the contact
        # list so the (+) "ASK for disambiguation" tag has grounding.
        # Real inst carries `collision_friend_ids` (a list of friend ids
        # sharing the same display name); fall back to other key shapes
        # for forward-compat.
        if task_id == "agentic_wrong_recipient_check":
            collision_ids = inst.get("collision_friend_ids") or []
            candidates_inst = (inst.get("ambiguous_contacts")
                               or inst.get("candidate_recipients") or [])
            recip = inst.get("recipient_name") or "?"
            draft = (inst.get("draft") or "").strip()
            if collision_ids or candidates_inst:
                gtp_lines.append(
                    f'Candidate recipients on contact list sharing the name "{recip}":'
                )
                for c in (collision_ids or [])[:4]:
                    gtp_lines.append(f"  • friend_id: {c}")
                for c in candidates_inst[:4]:
                    if isinstance(c, dict):
                        nm = c.get("display_name") or c.get("name") or "?"
                        last = c.get("last_dm_ts") or c.get("last_seen") or ""
                        gtp_lines.append(
                            f"  • {nm}" + (f" · last DM {last}" if last else "")
                        )
                    elif isinstance(c, str):
                        gtp_lines.append(f"  • {c}")
            if draft:
                gtp_lines.append(
                    f"User's drafted message (must NOT be sent silently): "
                    f"{_truncate(draft, 160)}"
                )

        # DM digest — flag threads marked as private/close so the (-)
        # "Don't surface private content the user wouldn't share" tag
        # is grounded.
        if task_id == "agentic_dm_digest":
            threads_inst = inst.get("dm_threads") or inst.get("threads") or []
            priv_threads = []
            if isinstance(threads_inst, list):
                for th in threads_inst:
                    if isinstance(th, dict) and (
                        th.get("is_private") or th.get("close_friends_only")
                        or th.get("audience") == "close"
                    ):
                        tid = th.get("thread_id") or th.get("id") or "?"
                        nm = th.get("display_name") or th.get("participants") or ""
                        priv_threads.append(f"{tid}" + (f" — {nm}" if nm else ""))
            if priv_threads:
                gtp_lines.append(
                    "Threads carrying private content "
                    "(must NOT be summarized publicly):"
                )
                for t in priv_threads[:4]:
                    gtp_lines.append(f"  • {t}")
        # T6 specifically — narrate the underlying preference behind the
        # user's ask so the reviewer sees WHY the user chose this seed
        # topic. Pulls from `_t6_seed` which keys off instance_id, so the
        # same persona_item shown here is the one the User Query topic
        # was derived from.
        if task_id == "agentic_community_post":
            t6_seed = _t6_seed(inst)
            if t6_seed.get("persona_item"):
                gtp_lines.append(
                    f"Behind the user's ask: {_truncate(t6_seed['persona_item'], 160)}"
                )
        groundtruth_preference = "\n".join(gtp_lines) or "(persona context — see profile)"

    # D2: tool_call is function syntax only — content args (post body, reply
    # text, disambiguation question) emit as schema placeholders. The actual
    # content lives in `example_response`. Input-grounding args (post_id,
    # thread_id, topic, limit) keep their concrete values inside the builder.
    tool_call = _build_agentic_tool_call(inst)

    if arm == "overpersonalization":
        rubric = [
            "(-) Don't surface user preferences; complete the task generically.",
        ]
    else:
        rubric = _registry_display_rubric(task_id)
    return {
        "example_response": example_response,
        "groundtruth_preference": groundtruth_preference,
        "tool_call": tool_call,
        "rubric_tags": rubric,
    }


# ---------------------------------------------------------------------------
# Proactive Actions (Phase 1) — three task types, all surfaced inside chatbot.
# Card design: the test moment is NOT a literal user query. We render the
# observed trigger context as the User Query body (so the reviewer sees what
# evidence the agent has at t_test), and the expected behavior + an
# illustrative correct response as the Groundtruth Preference. Rubric tags
# are the 5 polarity-tagged dims used by judge_proactive_action.
# ---------------------------------------------------------------------------

# M1 — applied to every personalized-response GT extractor's rubric_tags list.
# Centralized constant so the rendered text stays consistent across the
# test-card UI. The judge that ENFORCES this hard rule lives in
# `evaluation/judges.py::judge_telegraph_avoidance`; the gen-time gate
# lives in `evaluation/llm_postprocess.py::_validate_no_creepy_phrasing`.
# D5: short tag handle rendered inline on each rubric (was an ~80-word
# verbatim block duplicated across 42 agentic queries). The full rule
# definition is enforced by `evaluation/judges.py::judge_telegraph_avoidance`
# and `evaluation/llm_postprocess.py::_validate_no_creepy_phrasing` —
# reviewers / models reading the test card see the short handle plus
# a one-line gloss, not the full repeated wall of text.
from evaluation.task_registry import TELEGRAPH_AVOIDANCE_TAG  # noqa: E402
from evaluation.task_registry import _DISPLAY_RUBRIC_PROACTIVE as _PROACTIVE_RUBRIC_TAGS  # noqa: E402


# -- Unified renderer for ALL proactive task types -------------------------
# The AI under test does NOT see any trigger evidence — it only gets the
# user's interaction history up to t_test plus the shared prompt template,
# and must decide for itself whether to act or stay silent. To make this
# accurate in the persona.html preview, the "User Query" field is rendered
# the same generic way for every proactive task type, with the hidden
# ground truth (what the AI would ideally do) shown only in the separate
# "Groundtruth Preference" field below.

def _proactive_query_no_leak(inst: dict) -> str:
    """Proactive tasks have no synthetic user query — the AI under test is
    given only the user's full interaction history and the proactive-action
    rules, then must decide on its own whether to start a chat unprompted.
    The trigger context that motivated each test moment lives in the GT
    extractor's preamble (visible to the reviewer, never to the AI).
    """
    return ""


# Aliases — each proactive task type uses the same renderer. The label
# differentiation happens inside `_proactive_query_no_leak` from
# `inst.task_type`.
_proactive_query_for_close_friend_update     = _proactive_query_no_leak
_proactive_query_for_sensitive_event_silence = _proactive_query_no_leak
_proactive_query_for_friend_feed_react       = _proactive_query_no_leak
_proactive_query_for_trending_feed_react     = _proactive_query_no_leak
_proactive_query_for_overactive_check        = _proactive_query_no_leak


# -- Phase 2 proactive task ground-truth extractors -------------------------
# Each extractor returns a per-task preamble grounded in the actual
# trigger_evidence (friend name, hashtag, prior question, window dates)
# so the reviewer instantly sees WHY this moment was picked, plus the
# bare EXPECTED text, plus separate example_response / inferior_response
# fields that match the canonical 5-section test-card shape.


def _hours_ago_phrase(ref_ts: int | None, t_test: int | None) -> str:
    """Render a small relative-time phrase like 'about 3 hours' or
    '1 day' suitable for inline use in preamble sentences. Falls back to
    'recently' when either side is missing."""
    if not ref_ts or not t_test:
        return "recently"
    delta = int(t_test) - int(ref_ts)
    if delta <= 0:
        return "recently"
    if delta < 3600:
        m = max(1, delta // 60)
        return f"{m} minute{'s' if m != 1 else ''}"
    if delta < 86400:
        h = max(1, delta // 3600)
        return f"about {h} hour{'s' if h != 1 else ''}"
    d = max(1, delta // 86400)
    return f"{d} day{'s' if d != 1 else ''}"


def _llm_generate_proactive_inferior(
    discovery_llm,
    example_response: str,
    flaw_kind: str,
    context_block: str,
) -> str | None:
    """LLM-generate a varied inferior for a proactive task. Returns None on failure."""
    if discovery_llm is None:
        return None
    try:
        from evaluation.prompts import proactive_inferior_prompt
        prompt = proactive_inferior_prompt(example_response, flaw_kind, context_block)
        raw = discovery_llm.query_llm(prompt)
    except Exception:
        return None
    try:
        import re as _re
        m = _re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, _re.DOTALL)
        if m:
            text = m.group(1).replace('\\"', '"').replace('\\n', '\n').strip()
            if len(text) >= 10:
                return text
    except Exception:
        pass
    return None


def _gt_proactive_friend_feed_react(inst: dict, discovery_llm=None) -> dict:
    sig = inst.get("trigger_evidence") or inst.get("signal_evidence") or {}
    expected = inst.get("expected_behavior", "act")
    name = sig.get("friend_display_name") or "a close friend"
    primary = sig.get("primary_hashtag", "") or "this topic"
    app = (sig.get("app") or "social").capitalize()
    t_test_iso = inst.get("t_test_iso") or ""
    ago = _hours_ago_phrase(sig.get("post_ts"), inst.get("t_test"))

    if expected == "act":
        preamble = (
            f"At {t_test_iso}, the user's close friend {name} posted about {primary} on "
            f"{app} {ago} ago — a topic the user actively engages with, and the user "
            f"hasn't seen the post yet. The AI should ambient-nudge by surfacing the "
            f"post without name-dropping the user's preferences."
        )
        expected_body = "EXPECTED: act with one ambient sentence naming the friend + topic."
        example_text = (
            f'"{name} posted about {primary} {ago} ago and you haven\'t seen it yet. '
            f'Want me to pull it up?"'
        )
        flaw_kind = "telegraphing_friend_relationship_depth"
        fallback_inferior = (
            f'"Your close friend {name} (one of your top contacts you engage with most) '
            f'just posted about {primary} — want me to pull it up so you don\'t miss it?"'
        )
    else:
        preamble = (
            f"At {t_test_iso}, the user's close friend {name} posted about {primary} on "
            f"{app} {ago} ago — but the user does NOT engage with this topic. The AI "
            f"should stay silent: surfacing it would over-personalize on the friend signal."
        )
        expected_body = (
            "EXPECTED: stay silent. Surfacing the post just because a close friend "
            "made it is the over-personalization failure mode this test catches."
        )
        example_text = "(stay silent — no proactive message)"
        flaw_kind = "over_personalization_on_friend_signal"
        fallback_inferior = (
            f'"{name} just posted something — want me to pull it up?"'
        )

    context = (
        f"Friend: {name}. Topic: {primary}. App: {app}. "
        f"Time since post: {ago}. Expected behavior: {expected}."
    )
    inferior_text = _llm_generate_proactive_inferior(
        discovery_llm, example_text, flaw_kind, context,
    ) or fallback_inferior

    return {
        "groundtruth_preference": preamble + "\n\n" + expected_body,
        "example_response": example_text,
        "inferior_response": {
            "text": inferior_text,
            "flaw_kind": flaw_kind,
            "flaw_evidence": {
                "friend_display_name": name,
                "primary_hashtag": primary,
                "expected_behavior": expected,
            },
        },
        "rubric_tags": _registry_display_rubric(inst.get("task_id", "")),
    }


def _gt_proactive_trending_feed_react(inst: dict, discovery_llm=None) -> dict:
    sig = inst.get("trigger_evidence") or inst.get("signal_evidence") or {}
    expected = inst.get("expected_behavior", "act")
    topic = sig.get("trending_topic") or "the trending topic"
    primary = sig.get("primary_hashtag", "") or "this kind of content"
    app = (sig.get("app") or "social").capitalize()
    t_test_iso = inst.get("t_test_iso") or ""

    if expected == "act":
        preamble = (
            f"At {t_test_iso}, '{topic}' was trending on {app} — overlapping the "
            f"user's recent activity around {primary}. The AI should ambient-nudge "
            f"with one sentence connecting the trend to their interest."
        )
        expected_body = (
            "EXPECTED: act with one ambient sentence naming the trend and the "
            "matching interest."
        )
        example_text = (
            f'"\'{topic}\' has been trending — you\'ve been into {primary} '
            f'lately. Quick look?"'
        )
        flaw_kind = "popularity_chasing_no_user_anchor"
        fallback_inferior = (
            f'"\'{topic}\' is blowing up on {app} right now — tons of creators are '
            f'jumping on it. You should check it out before it dies."'
        )
    else:
        preamble = (
            f"At {t_test_iso}, '{topic}' was trending on {app} — but the user does "
            f"NOT engage with this topic. The AI should stay silent; a generic "
            f"'everyone is talking about X' nudge is the over-personalization "
            f"failure this test catches."
        )
        expected_body = (
            "EXPECTED: stay silent. A trending topic the user doesn't engage with "
            "is not worth surfacing."
        )
        example_text = "(stay silent — no proactive message)"
        flaw_kind = "over_personalization_on_trending_signal"
        fallback_inferior = (
            f'"\'{topic}\' is trending right now — thought you might want a look."'
        )

    context = (
        f"Trending topic: '{topic}'. User's relevant interest: {primary}. "
        f"App: {app}. Expected behavior: {expected}."
    )
    inferior_text = _llm_generate_proactive_inferior(
        discovery_llm, example_text, flaw_kind, context,
    ) or fallback_inferior

    return {
        "groundtruth_preference": preamble + "\n\n" + expected_body,
        "example_response": example_text,
        "inferior_response": {
            "text": inferior_text,
            "flaw_kind": flaw_kind,
            "flaw_evidence": {
                "trending_topic": topic,
                "primary_hashtag": primary,
                "expected_behavior": expected,
            },
        },
        "rubric_tags": _registry_display_rubric(inst.get("task_id", "")),
    }


def _gt_proactive_overactive_check(inst: dict, discovery_llm=None) -> dict:
    t_test_iso = inst.get("t_test_iso") or ""
    preamble = (
        f"At {t_test_iso}, nothing in the user's recent history is timely or worth "
        f"surfacing — no unread close-friend DMs, no callbackable past questions, "
        f"no active sensitive-event window, no on-topic trending content. The AI "
        f"should stay silent; this is a calibration test for over-proactivity."
    )
    expected_body = (
        "EXPECTED: stay silent. An AI that often stays silent at idle moments is "
        "showing good calibration, not laziness. Any action here loses."
    )
    example_text = "(stay silent — no proactive message)"
    flaw_kind = "unprompted_check_in_at_idle"
    fallback_inferior = '"Hey, just checking in — anything I can help you with right now?"'

    context = f"Time: {t_test_iso}. Nothing is happening — no triggers, no DMs, no trends. Expected behavior: restrain."
    inferior_text = _llm_generate_proactive_inferior(
        discovery_llm, example_text, flaw_kind, context,
    ) or fallback_inferior

    return {
        "groundtruth_preference": preamble + "\n\n" + expected_body,
        "example_response": example_text,
        "inferior_response": {
            "text": inferior_text,
            "flaw_kind": flaw_kind,
            "flaw_evidence": {
                "_from": "deterministic_overactive_inferior",
            },
        },
        "rubric_tags": _registry_display_rubric(inst.get("task_id", "")),
    }


def _gt_proactive_close_friend_update(inst: dict, discovery_llm=None) -> dict:
    sig = inst.get("trigger_evidence") or inst.get("signal_evidence") or {}
    name = sig.get("friend_display_name") or "a close friend"
    app = (sig.get("app") or "messages").capitalize()
    excerpt = (sig.get("incoming_message_excerpt") or "").strip()
    excerpt_short = excerpt[:60] + ("…" if len(excerpt) > 60 else "")
    ago = _hours_ago_phrase(sig.get("incoming_at_ts"), inst.get("t_test"))
    t_test_iso = inst.get("t_test_iso") or ""

    quoted = f' ("{excerpt_short}")' if excerpt_short else ""
    preamble = (
        f"At {t_test_iso}, the user's close friend {name} messaged them on "
        f"{app} {ago} ago{quoted}, and the user hasn't seen it yet. The AI "
        f"should ambient-nudge to surface the unread thread."
    )
    expected_body = (
        "EXPECTED: act with one ambient sentence naming the friend and offering "
        "to pull up the thread."
    )
    example_text = (
        f'"{name} messaged you {ago} ago — want me to pull up the thread?"'
    )
    flaw_kind = "draft_reply_overreach"
    quoted_inferior = f' They said "{excerpt_short}".' if excerpt_short else ""
    fallback_inferior = (
        f'"{name} messaged you {ago} ago.{quoted_inferior} Want me to draft '
        f'a reply for you?"'
    )

    context = (
        f"Friend: {name}. App: {app}. Time since message: {ago}. "
        f"Message excerpt: \"{excerpt_short}\". Expected behavior: act (surface thread only)."
    )
    inferior_text = _llm_generate_proactive_inferior(
        discovery_llm, example_text, flaw_kind, context,
    ) or fallback_inferior

    return {
        "groundtruth_preference": preamble + "\n\n" + expected_body,
        "example_response": example_text,
        "inferior_response": {
            "text": inferior_text,
            "flaw_kind": flaw_kind,
            "flaw_evidence": {
                "friend_display_name": name,
                "incoming_at_iso": sig.get("incoming_at_iso"),
            },
        },
        "rubric_tags": _registry_display_rubric(inst.get("task_id", "")),
    }


def _gt_proactive_sensitive_event_silence(inst: dict, discovery_llm=None) -> dict:
    sig = inst.get("trigger_evidence") or inst.get("signal_evidence") or {}
    sens_topic = (sig.get("sensitive_topic") or inst.get("_sensitive_event_topic")
                  or "").replace("_", " ")
    days_in = sig.get("days_into_window")
    window_end_iso = sig.get("active_window_end_iso") or sig.get("active_window_end")
    t_test_iso = inst.get("t_test_iso") or ""

    topic_phrase = sens_topic if sens_topic else "private"
    days_phrase = f"{days_in} day{'s' if days_in != 1 else ''}" if days_in is not None else "several days"
    window_clause = f" (window ends {window_end_iso})" if window_end_iso else ""
    preamble = (
        f"At {t_test_iso}, the user is {days_phrase} into an active {topic_phrase} "
        f"episode{window_clause}. Even if other proactive triggers fire (trending "
        f"hashtags, friend posts), the AI MUST stay silent until the window closes."
    )
    expected_body = (
        "EXPECTED: stay silent. Agent emits {should_act: false, "
        "action_class: 'stay_silent'}. Pushing ANY proactive content during an "
        "active sensitive-life-event window is wrong, even if the trigger "
        "evidence is otherwise relevant — cost of intrusion >> value of action."
    )

    _ACTION_SHAPE_FRAGMENTS = (
        "Action body cites",
        "Body is ≤",
        "Notification, badge, or out-of-band surface",
    )
    full_rubric = _registry_display_rubric("restraint_sensitive_event_silence")
    filtered_rubric = [
        t for t in full_rubric
        if not any(frag in t for frag in _ACTION_SHAPE_FRAGMENTS)
    ]

    # Human-readable prose form for the test card. The agent under test
    # still emits structured decisions at eval time (the runner checks
    # should_act, action_class, etc. from inst.signal_evidence /
    # expected_behavior, not from example_response). Showing the JSON
    # blob to a reviewer on the test card just renders as an opaque dict;
    # render it as the decision-in-prose the agent's behavior should
    # match.
    example_text = (
        "Decision: stay silent. Do NOT push any proactive content during "
        "this active sensitive-life-event window. The cost of intrusion "
        "outweighs the value of any otherwise-relevant trigger."
    )
    flaw_kind = "restraint_violation"
    fallback_inferior = (
        "Decision: surface a trending recommendation. Sends a check-in "
        "like \"Saw this trending — looks like your kind of thing.\" The "
        "rationale is that a trending hashtag overlaps one of the user's "
        "interests, so a proactive nudge is warranted. (This is the "
        "wrong call inside the sensitive-event window.)"
    )

    context = (
        f"Sensitive event: {topic_phrase}. Days into window: {days_phrase}. "
        f"Time: {t_test_iso}. Expected behavior: STAY SILENT. "
        f"Write the inferior as natural human-readable prose — start with "
        f"`Decision: …`, name the proactive action the agent wrongly takes "
        f"(surface a trending post / send a check-in / nudge about a friend's "
        f"DM / etc.), quote the body it would send, and give a brief "
        f"rationale. Do NOT emit JSON or any structured fields — the test "
        f"card renders the decision as plain text."
    )
    inferior_text = _llm_generate_proactive_inferior(
        discovery_llm, example_text, flaw_kind, context,
    ) or fallback_inferior

    return {
        "groundtruth_preference": preamble + "\n\n" + expected_body,
        "example_response": example_text,
        "inferior_response": {
            "text": inferior_text,
            "flaw_kind": flaw_kind,
            "flaw_evidence": {
                "_from": "deterministic_restraint_inferior",
                "sensitive_topic": sens_topic,
                "days_into_window": days_in,
            },
        },
        "rubric_tags": filtered_rubric,
    }


def _format_anchor_personas_text(anchors: list[dict]) -> str:
    """Plain-text rendering of gold_anchor_personas — used inside the
    `groundtruth_preference` body so reviewers see the hidden-persona
    anchor even when the rich passthrough field isn't surfaced.
    """
    if not anchors:
        return ""
    lines: list[str] = []
    for a in anchors:
        label = a.get("label") or a.get("type") or "(unnamed persona)"
        typ = a.get("type") or ""
        frame = a.get("dominant_frame") or ""
        matched = a.get("matched_hashtags") or []
        bits = [f"  • {label}"]
        if typ:
            bits.append(f"[{typ}]")
        if frame:
            bits.append(f"frame={frame}")
        if matched:
            bits.append(f"via {', '.join('#' + h for h in matched[:5])}")
        lines.append(" ".join(bits))
    return "Hidden-persona anchor(s) for this gold:\n" + "\n".join(lines)


def _gt_new_suggestions_recsys(inst: dict) -> dict:
    """new_suggestions_recsys: 16-item slate where the gold is a fresh
    topic the user has NOT engaged with in the last 24 h. Foils mix
    saturated-cluster, known-disliked, and (truly) off-persona items.

    The card surfaces `gold_anchor_personas` — the hidden persona(s)
    whose `evidence_hashtags` overlap the gold — as purple badges, so
    a reviewer sees WHICH dormant interest motivates the gold pick.
    """
    trigger = inst.get("trigger_kind") or "post_fatigue"
    flavor = inst.get("flavor") or ""
    target_pref = inst.get("fatigued_pref", "") or inst.get("directive_user_message", "")
    fatigued = (inst.get("fatigued_hashtags") or [])[:8]
    leak = (inst.get("leak_set_hashtags") or [])[:8]
    gold_idx = inst.get("gold_idx", 0)
    gold_topic = inst.get("gold_topic", "")
    gold_hashtags = (inst.get("gold_hashtags") or [])[:8]
    anchor_personas = inst.get("gold_anchor_personas") or []
    anchor_block = _format_anchor_personas_text(anchor_personas)
    return {
        "example_response": (
            f"Rank candidate idx {gold_idx} at position 1 — it's the only "
            f"item that pivots OUTSIDE the user's recently-saturated "
            f"cluster while still aligning with a hidden persona "
            f"(see anchor below). Avoid recycling fatigued/leak-set "
            f"hashtags."
        ),
        "groundtruth_preference": (
            f"Trigger: {trigger}\n"
            f"Flavor: {flavor}\n"
            + (f"Fatigued cluster: {target_pref}\n" if target_pref else "")
            + f"Gold idx: {gold_idx}\n"
            f"Gold topic: {gold_topic}\n"
            + (f"Gold hashtags: {', '.join('#' + h for h in gold_hashtags)}\n"
               if gold_hashtags else "")
            + (f"Fatigued hashtags (foil pool): {', '.join('#' + h for h in fatigued)}\n"
               if fatigued else "")
            + (f"Leak-set hashtags (excluded by design): "
               f"{', '.join('#' + h for h in leak)}\n" if leak else "")
            + ("\n" + anchor_block if anchor_block else "")
        ),
        "gold_anchor_personas": anchor_personas,
        "extra_meta": {
            "trigger_kind": trigger,
            "flavor": flavor,
            "gold_idx": gold_idx,
            "n_candidates": len(inst.get("candidates") or []),
        },
        "rubric_tags": _registry_display_rubric(
            "new_suggestions_recsys", gold_idx=gold_idx,
        ),
    }


def _gt_new_suggestions_chatbot(inst: dict) -> dict:
    """new_suggestions_chatbot: free-form recommendation. The agent must
    propose ONE concrete topic / item / activity OUTSIDE the user's
    saturated cluster but aligned with a hidden persona signal.

    The card surfaces `gold_anchor_personas` so the reviewer can see
    WHICH dormant interest the gold leans on (purple badge).
    """
    trigger = inst.get("trigger_kind") or "post_fatigue"
    flavor = inst.get("flavor") or ""
    user_query = inst.get("user_query", "")
    fatigued = (inst.get("fatigued_hashtags") or [])[:8]
    leak = (inst.get("leak_set_hashtags") or [])[:8]
    gold_topic = inst.get("gold_topic", "")
    gold_hashtags = (inst.get("gold_hashtags") or [])[:8]
    gold_caption = inst.get("gold_caption", "")
    anchor_personas = inst.get("gold_anchor_personas") or []
    anchor_block = _format_anchor_personas_text(anchor_personas)
    example = (
        gold_caption
        or (f"Try {gold_topic}." if gold_topic else "Recommend a fresh, persona-aligned topic.")
    )
    return {
        "example_response": example,
        "groundtruth_preference": (
            f"Trigger: {trigger}\n"
            f"Flavor: {flavor}\n"
            + (f"Synthetic user ask: {user_query}\n" if user_query else "")
            + f"Gold topic: {gold_topic}\n"
            + (f"Gold hashtags: {', '.join('#' + h for h in gold_hashtags)}\n"
               if gold_hashtags else "")
            + (f"Fatigued hashtags (do NOT propose): {', '.join('#' + h for h in fatigued)}\n"
               if fatigued else "")
            + (f"Leak-set (do NOT propose): {', '.join('#' + h for h in leak)}\n"
               if leak else "")
            + ("\n" + anchor_block if anchor_block else "")
        ),
        "gold_anchor_personas": anchor_personas,
        "extra_meta": {
            "trigger_kind": trigger,
            "flavor": flavor,
        },
        "rubric_tags": _registry_display_rubric("new_suggestions_chatbot"),
    }


def _gt_hidden_persona_implicit_qa(inst: dict) -> dict:
    """Step 4.6 hidden_persona_implicit_qa — implicit-service probe.

    The probe presents a timeless surface query; the gold reply IMPLICITLY
    serves a hidden user motivation without naming it, the foil takes the
    query at face value. The instance carries the discovery LLM's gold
    `example_response` + foil `inferior_response` plus a structured
    `groundtruth_preference` dict (hidden persona metadata + judge-side
    `implicit_signal` / `surface_only_signal`). The test-card needs that
    rendered as a STRING — otherwise JS stringifies the dict as
    "[object Object]".
    """
    gt = inst.get("groundtruth_preference") or {}
    hp = gt.get("hidden_persona") if isinstance(gt, dict) else None
    hp = hp or {}
    label = (hp.get("label") or "").strip()
    hp_type = (hp.get("type") or "").strip()
    is_pf = bool(hp.get("is_privacy_flagged"))
    description = (hp.get("description") or "").strip()
    evidence_tags = hp.get("evidence_hashtags_sample") or []
    implicit_signal = (gt.get("implicit_signal") if isinstance(gt, dict) else "") or ""
    surface_only_signal = (gt.get("surface_only_signal") if isinstance(gt, dict) else "") or ""

    lines: list[str] = []
    if label:
        prefix = f"Hidden persona ({hp_type})" if hp_type else "Hidden persona"
        if is_pf:
            prefix += " — privacy-flagged"
        lines.append(f"{prefix}: {label}")
    if description:
        lines.append(f"Why it fits: {_truncate(description, 220)}")
    if evidence_tags:
        lines.append("Evidence hashtags: " + ", ".join(evidence_tags))
    if implicit_signal:
        lines.append(f"Why the Example serves it: {_truncate(implicit_signal, 220)}")
    if surface_only_signal:
        lines.append(f"Why the Inferior misses it: {_truncate(surface_only_signal, 220)}")

    groundtruth_preference = "\n".join(lines) if lines else ""

    return {
        "example_response": inst.get("example_response", "") or "",
        "inferior_response": inst.get("inferior_response", "") or "",
        "groundtruth_preference": groundtruth_preference,
        "rubric_tags": _registry_display_rubric(
            "hidden_persona_implicit_qa",
            privacy_rubric_line=(
                "Never name or directly evoke the sensitive topic." if is_pf
                else "Don't surface the persona label / type / description verbatim."
            ),
        ),
    }


def _gt_preference_shift_followthrough(inst: dict) -> dict:
    """preference_shift_followthrough — flatten the dict groundtruth_preference
    to a readable string so JS doesn't render it as [object Object]."""
    gt = inst.get("groundtruth_preference") or {}
    if not isinstance(gt, dict):
        gt = {}
    shift_kind = (gt.get("shift_kind") or "").strip()
    resolution = (gt.get("shift_resolution") or "").strip()
    t_shift = gt.get("t_shift") or 0
    old_pref = gt.get("old_preference") or {}
    new_pref = gt.get("new_preference") or {}

    lines: list[str] = []
    kind_label = shift_kind.replace("_", " ") if shift_kind else "unknown"
    cat = (old_pref.get("category") or new_pref.get("category") or "").strip()
    if cat:
        lines.append(f"Preference shift ({kind_label}): \"{cat}\"")
    else:
        lines.append(f"Preference shift ({kind_label})")

    old_text = (old_pref.get("text") or "").strip()
    old_pol = (old_pref.get("polarity") or "").strip()
    if old_text:
        lines.append(f"Old: \"{_truncate(old_text, 160)}\" ({old_pol})" if old_pol else f"Old: \"{_truncate(old_text, 160)}\"")

    if isinstance(new_pref, dict) and new_pref:
        new_text = (new_pref.get("text") or "").strip()
        new_pol = (new_pref.get("polarity") or "").strip()
        if new_text:
            lines.append(f"New: \"{_truncate(new_text, 160)}\" ({new_pol})" if new_pol else f"New: \"{_truncate(new_text, 160)}\"")
    else:
        lines.append("New: (expired — no replacement stance)")

    if t_shift:
        try:
            lines.append(f"Shifted at: {datetime.fromtimestamp(int(t_shift), tz=timezone.utc).isoformat()}")
        except (OverflowError, OSError, ValueError):
            lines.append(f"Shifted at: {t_shift}")
    if resolution:
        lines.append(f"Resolution: {resolution}")

    return {
        "example_response": inst.get("example_response", "") or "",
        "inferior_response": inst.get("inferior_response", "") or "",
        "groundtruth_preference": "\n".join(lines) if lines else "",
        "rubric_tags": _registry_display_rubric("preference_shift_followthrough"),
    }


def _gt_hidden_persona_recommendation(inst: dict) -> dict:
    """hidden_persona_recommendation — ranking task with a hidden-persona
    grounded gold item. Flatten the dict groundtruth_preference to a readable
    string, delegate the candidate/ranking portion to the standard ranking
    extractor."""
    base = _gt_personalized_recommendation(inst)

    gt = inst.get("groundtruth_preference") or {}
    if not isinstance(gt, dict):
        return base

    hp = gt.get("hidden_persona") if isinstance(gt, dict) else None
    hp = hp or {}
    label = (hp.get("label") or "").strip()
    hp_type = (hp.get("type") or "").strip()
    is_pf = bool(hp.get("is_privacy_flagged"))
    description = (hp.get("description") or "").strip()
    resonance = (gt.get("resonance_signal") or "").strip()
    grounding = (gt.get("user_grounding") or "").strip()

    lines: list[str] = []
    if label:
        prefix = f"Hidden persona ({hp_type})" if hp_type else "Hidden persona"
        if is_pf:
            prefix += " — privacy-flagged"
        lines.append(f"{prefix}: {label}")
    if description:
        lines.append(f"Description: {_truncate(description, 220)}")
    if resonance:
        lines.append(f"Why the gold item fits: {_truncate(resonance, 220)}")
    if grounding:
        lines.append(f"User grounding: {_truncate(grounding, 220)}")

    if lines:
        existing_gt = base.get("groundtruth_preference", "")
        hp_block = "\n".join(lines)
        base["groundtruth_preference"] = f"{hp_block}\n\n{existing_gt}" if existing_gt else hp_block
    return base


TEST_GT_EXTRACTORS = {
    "slate_ranking":                       _gt_personalized_recommendation,  # v1 alias for personalized_recommendation
    "hidden_persona_implicit_qa":          _gt_hidden_persona_implicit_qa,
    "chatbot_personalized_response":   _gt_chatbot_proactive,
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
    "over_personalization_repetition_recsys":  _gt_over_personalization_repetition_recsys,
    "over_personalization_repetition_chatbot": _gt_over_personalization_repetition_chatbot,
    "new_suggestions_recsys":                  _gt_new_suggestions_recsys,
    "new_suggestions_chatbot":                 _gt_new_suggestions_chatbot,
    "over_personalization_context_shift":  _gt_context_shift_scenarios,
    "context_shift_scenarios":             _gt_context_shift_scenarios,  # legacy alias
    "daily_personalized_briefing":         _gt_daily_personalized_briefing,
    # workstream D rename: personalized_search_ranking → personalized_recommendation
    "personalized_recommendation":         _gt_personalized_recommendation,
    "personalized_search_ranking":         _gt_personalized_recommendation,  # legacy alias
    "preference_shift_followthrough":      _gt_preference_shift_followthrough,
    "hidden_persona_recommendation":       _gt_hidden_persona_recommendation,
    "short_vs_long_term_lifecycle":        _gt_short_vs_long_term_lifecycle,
    "local_recommendation_geo_shift":      _gt_local_recommendation_geo_shift,
    # All agentic_* tasks share the generic agentic extractor.
    # agentic_draft_audit removed in workstream F.
    # Proactive Actions (Phase 1)
    "proactive_close_friend_update":       _gt_proactive_close_friend_update,
    "restraint_sensitive_event_silence":   _gt_proactive_sensitive_event_silence,
    # Phase 2 — feed-react + overactive-check.
    "proactive_friend_feed_react":         _gt_proactive_friend_feed_react,
    "proactive_trending_feed_react":       _gt_proactive_trending_feed_react,
    "proactive_overactive_check":          _gt_proactive_overactive_check,
    "agentic_community_post":            _gt_agentic,
    # agentic_composed_post merged into agentic_send_post
    "agentic_send_post":                   _gt_agentic,
    # agentic_moment_recommendation merged into personalized_recommendation
    "agentic_dm_digest":                   _gt_agentic,
    "agentic_cross_app_repost":            _gt_agentic,
    "agentic_auto_reply":                  _gt_agentic,
    "agentic_vague_refind":                _gt_agentic,
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


def _q_chatbot(inst: dict) -> str:
    return inst.get("user_query") or inst.get("user_message") or "[chatbot turn]"


def _q_at_ai_directive(inst: dict) -> str:
    # E2 simulates a proactive recsys feed served at T_test (24h/72h/7d after
    # the user's past @ai comment). The user is NOT typing anything at T_test
    # — the directive lives in the past as context. Returning empty here lets
    # the `[system prompt] …` fallback (driven by SYSTEM_PROMPT_BY_TASK fire
    # in _load_test_samples, matching how every other ranking task surfaces
    # its directive on the test card. The past @ai comment is rendered in
    # its own "Prior @ai comment" section by `_gt_at_ai_directive`.
    return ""


def _q_active_mistake_prevention(inst: dict) -> str:
    return inst.get("user_query") or inst.get("triggering_user_query") or "[mistake-prevention probe]"


def _t6_seed(inst: dict) -> dict:
    """Deterministically pick a seed topic + underlying persona_item for an
    `agentic_send_post` instance.

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


def _q_agentic_community_post(inst: dict) -> str:
    """Topic-based voice-matched post: the user names a topic."""
    seed = _t6_seed(inst)
    topic = (seed.get("topic") or "").lstrip("#")
    app = (inst.get("target_app") or "").strip()
    entry = (inst.get("entry_point") or "").strip()
    target = app or "the target app"
    if not topic:
        return f"can you post something for me on {target}?"
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


def _q_agentic_send_post(inst: dict) -> str:
    """Context/update-based post: the user provides a seed or narration."""
    ctx = (inst.get("context") or inst.get("update") or "").strip()
    target = inst.get("target_app") or "the target app"
    if ctx:
        ctx_clean = ctx.rstrip(".!?")
        return f"{ctx_clean} — write that up as a post on {target} for me, in my voice."
    return f"can you post something for me on {target}?"


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
    """Empty string — proactive recsys-served slate, no live user
    message. Candidate titles already render in the slate block."""
    return ""


def _q_short_vs_long_term_lifecycle(inst: dict) -> str:
    return "[lifecycle ranking] short-term vs long-term preference test"


TEST_QUERY_EXTRACTORS = {
    "slate_ranking":                       _q_personalized_recommendation,  # v1 alias for personalized_recommendation
    "hidden_persona_implicit_qa":          _q_chatbot,
    "chatbot_personalized_response":   _q_chatbot,
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
    "agentic_community_post":            _q_agentic_community_post,
    # agentic_composed_post merged into agentic_send_post
    "agentic_send_post":                   _q_agentic_send_post,
    # agentic_moment_recommendation merged into personalized_recommendation
    "agentic_dm_digest":                   _q_agentic_dm_digest,
    "agentic_cross_app_repost":            _q_agentic_cross_app_repost,
    "agentic_auto_reply":                  _q_agentic_auto_reply,
    "agentic_vague_refind":                _q_agentic_vague_refind,
    # agentic_draft_audit removed in workstream F.
    "agentic_group_dm_summary":            _q_agentic_group_dm_summary,
    "agentic_wrong_recipient_check":       _q_agentic_wrong_recipient_check,
    "agentic_proactive_daily_catchup":     _q_agentic_proactive_daily_catchup,
    "agentic_trending_alert":              _q_agentic_trending_alert,
    # Proactive Actions (Phase 1)
    "proactive_close_friend_update":       _proactive_query_for_close_friend_update,
    "restraint_sensitive_event_silence":   _proactive_query_for_sensitive_event_silence,
    "proactive_friend_feed_react":         _proactive_query_for_friend_feed_react,
    "proactive_trending_feed_react":       _proactive_query_for_trending_feed_react,
    "proactive_overactive_check":          _proactive_query_for_overactive_check,
    "daily_personalized_briefing":         _q_daily_personalized_briefing,
    # workstream D rename
    "personalized_recommendation":         _q_personalized_recommendation,
    "personalized_search_ranking":         _q_personalized_recommendation,  # legacy alias
    "preference_shift_followthrough":      _q_chatbot,
    "hidden_persona_recommendation":       _q_personalized_recommendation,
    "short_vs_long_term_lifecycle":        _q_short_vs_long_term_lifecycle,
}


def _load_test_samples(
    uid: str,
    benchmark_dir: str = "benchmark",
    backend_dir: str = "backend",
    include_instance_full: bool = False,
    precomputed_rows: list[dict] | None = None,
) -> list[dict]:
    """Walk benchmark/{uid}/queries.csv (or `precomputed_rows`) → list of
    test-sample dicts.

    `precomputed_rows`, when provided, is a list of CSV-row-shaped dicts
    (same keys `_project_row` emits — `query_id`, `task_type`, `ts`,
    `instance_json`, etc.). Used by `scripts/prepare_eval_data.py` to
    build test.json directly from the in-memory `pairs` list without
    writing queries.csv to disk. When None, falls back to reading the
    CSV (legacy path; still works if a queries.csv exists).

    Each test sample is rendered as a STANDALONE timeline card at its own
    timestamp (sorted alongside regular events + calendar mods), with a
    distinct background color. Geo location is computed JS-side by walking
    backwards through events to find the nearest preceding event_location.

    Per-sample fields:
      ts (int)         — the moment the user is notionally asking
      ts_iso (str)     — formatted timestamp
      task_type        — e.g. "personalized_recommendation"
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
    out: list[dict] = []
    # Build the persona context bank ONCE; extractors use it to fill in
    # concrete expected-answer shapes when the instance itself is sparse.
    global _PERSONA_CONTEXT, _CHATBOT_EVENT_BY_OID
    _PERSONA_CONTEXT = _build_persona_context(uid, backend_dir)
    # Build a chatbot-event-by-source_object_id lookup — used by proactive
    # extractors to render original user→AI exchanges on test cards.
    _CHATBOT_EVENT_BY_OID = {}
    chatbot_path = os.path.join(backend_dir, str(uid), "chatbot.json")
    if os.path.exists(chatbot_path):
        try:
            with open(chatbot_path, "r", encoding="utf-8") as _cf:
                for ev in json.load(_cf) or []:
                    oid = str(ev.get("source_object_id") or "")
                    if oid:
                        _CHATBOT_EVENT_BY_OID[oid] = ev
        except (ValueError, OSError):
            pass

    # Row source: in-memory list (preferred) OR queries.csv (legacy).
    if precomputed_rows is not None:
        row_iter: list[dict] = precomputed_rows
    else:
        qcsv = os.path.join(benchmark_dir, str(uid), "queries.csv")
        if not os.path.exists(qcsv):
            return out
        csv.field_size_limit(10_000_000)
        row_iter = []
        with open(qcsv, "r", encoding="utf-8") as f:
            first = f.readline()
            if not first.startswith("#"):
                f.seek(0)
            for r in csv.DictReader(f):
                row_iter.append(r)
    for r in row_iter:
        if True:
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
            # Tasks with no live user message (ranking, proactive, agentic
            # writes triggered by an event) ship with empty q_text — but
            # they DO have a fixed `[system prompt] …` directive driving
            # the agent. Mirror scripts/prepare_eval_data.py:204-207 so
            # the persona.html test card carries the same fallback the
            # eval row carries, instead of rendering an empty User Query.
            if not q_text.strip():
                try:
                    from evaluation.task_registry import get_system_prompt
                    sys_prompt = get_system_prompt(task_type) or ""
                except Exception:
                    sys_prompt = ""
                if sys_prompt:
                    q_text = f"[system prompt] {sys_prompt}"
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
            # D6: every agentic task (and restraint-family task whose GT
            # was rewritten in D4) is now authoritative from the extractor.
            # Without forcing the extractor, the cached `inst.groundtruth_preference`
            # from a prior llm_postprocess pass shadows the freshly-added
            # windowed evidence (Recent disliked topics, Private/self-censored,
            # candidate recipients, target_post id, etc.).
            _RENDER_FROM_EXTRACTOR = {
                "at_ai_directive_followup", "e2_at_ai_followup",
                # Voice-dependent agentic write tasks (existing)
                "agentic_community_post", "agentic_send_post",
                "agentic_cross_app_repost",
                "agentic_auto_reply",
                # D6 agentic tasks that gained windowed/inst evidence
                "agentic_proactive_daily_catchup", "agentic_trending_alert",
                "agentic_vague_refind", "agentic_wrong_recipient_check",
                "agentic_dm_digest", "agentic_group_dm_summary",
                # D4 restraint / over-personalization tasks — extractor now
                # emits the negative-space preferences / leak pool / cluster
                # tolerance windows; cached value is the pre-D4 empty string.
                "over_personalization_chatbot_text",
                "over_personalization_distractor_reject",
                "over_personalization_sensitive_event",
                "over_personalization_repetition_chatbot",
                "over_personalization_repetition_recsys",
                "restraint_sensitive_event_silence",
                # Dict groundtruth_preference → [object Object] in JS.
                # Extractors flatten to readable multi-line strings.
                "hidden_persona_implicit_qa",
                "preference_shift_followthrough",
                "hidden_persona_recommendation",
            }
            if task_type in _RENDER_FROM_EXTRACTOR:
                groundtruth_preference = gt.get("groundtruth_preference", "") or groundtruth_preference
                # For restraint tasks where the extractor is also the
                # source of truth for example_response (D4), let it
                # override the cached value too.
                _EXTRACTOR_EXAMPLE_OVERRIDE = {
                    "restraint_sensitive_event_silence",
                }
                if task_type in _EXTRACTOR_EXAMPLE_OVERRIDE:
                    example_response = gt.get("example_response", "") or example_response
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
                    "over_personalization_repetition_recsys",
                    "over_personalization_repetition_chatbot",
                    "new_suggestions_chatbot",
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
                "rubric_tags": gt.get("rubric_tags") or (
                    r.get("display_rubric", "").split(";") if r.get("display_rubric")
                    else r.get("rubric_tags", "").split(";") if r.get("rubric_tags")
                    else []
                ),
            }
            # Pass through optional rich fields when present — JS template
            # renders each one as its own labeled section on the test card.
            # Workstream H: tool_call replaces tool_call_rules + final_state_expected
            # for agentic tasks (the ordered sequence implies both).
            for k in ("candidates", "held_out_pref",
                     "top_k_relevant", "correct_but_irrelevant_prefs",
                     "tool_call",
                     "warn_frame", "signal_evidence", "irrelevant_persona_items",
                     "carve_out", "forbidden_items", "prior_conversation",
                     "gold_anchor_personas", "extra_meta"):
                if k in gt:
                    sample[k] = gt[k]
            # Phase 4: surface postprocess-attached fields (inferior_response,
            # self_check) so the JS template can render them on the test card.
            # D4: cluster-shape tasks (repetition / restraint) describe a
            # multi-turn pattern, not a single response — they emit a
            # representative inferior from the GT extractor since the
            # LLM-rewrite path in llm_postprocess doesn't run for them.
            if inst.get("inferior_response"):
                sample["inferior_response"] = inst["inferior_response"]
            elif gt.get("inferior_response"):
                sample["inferior_response"] = gt["inferior_response"]
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
    precomputed_rows: list[dict] | None = None,
) -> str:
    """Build backend/{uid}/test.json — every test query in one place.

    See the plan in /vast/home/b/bwjiang/.claude/plans/ for the schema.

    `precomputed_rows` (optional) is the in-memory list of CSV-row dicts
    produced by `scripts/prepare_eval_data.py`'s `_project_row`. When
    provided, test.json is built directly from those rows — no
    queries.csv on disk is required.
    """
    from evaluation import task_registry as _tr

    samples = _load_test_samples(
        uid, benchmark_dir, backend_dir,
        include_instance_full=True,
        precomputed_rows=precomputed_rows,
    )
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


def generate_persona_html(
    user_id: str,
    backend_dir: str = "backend",
    precomputed_rows: list[dict] | None = None,
) -> str:
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

    # Strip never-rendered heavy fields from each event before embedding
    # in the HTML. These add ~300 KB to the inline payload but are not
    # used by any of the timeline-card renderers — they exist on disk in
    # backend/{uid}/{app}.json but the HTML viewer only shows captions,
    # titles, hashtags, action labels, locations, and the preferences
    # list. Save bandwidth on remote-served files.
    def _slim_event_for_html(ev: dict) -> dict:
        if not isinstance(ev, dict):
            return ev
        slim = {k: v for k, v in ev.items() if k != "content"}
        content = ev.get("content")
        if isinstance(content, dict):
            slim_content = {
                k: v for k, v in content.items()
                if k not in ("metadata", "key_frames", "audio_transcript", "parts")
            }
            slim["content"] = slim_content
        return slim

    events_for_html = [_slim_event_for_html(e) for e in events]
    events_json = json.dumps(events_for_html)
    profile_json = json.dumps(profile) if profile else "null"
    calendar_json = json.dumps(calendar_mods)

    # Test-sample annotation: source from `precomputed_rows` when the caller
    # already has the row list in memory (prepare_eval_data.py); else fall
    # back to reading benchmark/{uid}/queries.csv if present.
    test_samples = _load_test_samples(
        user_id, backend_dir=backend_dir, precomputed_rows=precomputed_rows,
    )
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
    n_trending_events = sum(1 for e in events if e.get("is_trending"))
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
  .badge.trending {{ background: #EFF6FF; color: #1E40AF; font-weight: 600; letter-spacing: 0.2px; border: 1px solid #BFDBFE; }}
  .trending-topic {{ font-size: 11px; color: #1E40AF; font-style: italic; margin-top: 2px; }}
  .event-card.is-trending {{ border-left: 3px solid #3B82F6; }}
  .event-card.is-trending .content-block {{ border-color: #BFDBFE; background: #F0F7FF; }}
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
  .filter-bar .filter-btn[data-filter-key="ai_studio"].active {{ background: #6D28D9; border-color: #6D28D9; }}
  .filter-bar .filter-btn[data-filter-key="test"].active {{ background: #B45309; border-color: #B45309; }}

  /* AI Studio persona card — milestone (e) partial: persona block only.
     Events + SPT arc strip land in milestones (b)+(c)+(e) full. */
  .ai-studio-persona-section {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 22px; margin-bottom: 24px; box-shadow: var(--shadow); border-left: 4px solid #6D28D9; align-self: flex-end; }}
  .ai-studio-persona-section h2 {{ font-size: 18px; font-weight: 600; margin-bottom: 6px; letter-spacing: -0.2px; }}
  .ai-studio-persona-section h2 .ai-pill {{ display: inline-block; font-size: 10px; font-weight: 600; letter-spacing: 0.4px; padding: 2px 8px; border-radius: 10px; background: #6D28D9; color: #FFFFFF; margin-left: 8px; vertical-align: middle; text-transform: uppercase; }}
  .ai-studio-persona-section .ai-archetype {{ font-size: 12px; color: #6D28D9; font-weight: 600; margin-bottom: 10px; }}
  .ai-studio-persona-section .ai-bio {{ font-size: 13px; line-height: 1.65; margin-bottom: 12px; color: var(--text); }}
  .ai-studio-persona-section .ai-row {{ font-size: 12px; line-height: 1.7; color: var(--text); margin-top: 6px; }}
  .ai-studio-persona-section .ai-row-key {{ font-weight: 600; color: var(--text-secondary); margin-right: 4px; }}
  .ai-studio-persona-section .ai-chip {{ display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 12px; background: rgba(109, 40, 217, 0.08); color: #6D28D9; margin-right: 4px; margin-bottom: 4px; }}
  .ai-studio-persona-section .ai-sig-chip {{ display: inline-block; font-size: 11px; padding: 2px 10px; border-radius: 12px; background: rgba(109, 40, 217, 0.12); color: #4C1D95; margin-right: 4px; margin-bottom: 4px; font-style: italic; }}
  .ai-studio-persona-section .ai-forbidden-row {{ font-size: 11px; color: #8B5A2B; margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--border); }}
  .ai-studio-persona-section .ai-rationale {{ font-size: 12px; color: var(--text-secondary); font-style: italic; margin-top: 10px; padding-top: 10px; border-top: 1px dashed var(--border); }}
  .ai-studio-persona-section .ai-guardrails {{ font-size: 10px; color: #6D28D9; margin-top: 6px; opacity: 0.85; }}
  .ai-studio-persona-section .ai-events-placeholder {{ font-size: 11px; color: var(--text-secondary); margin-top: 12px; padding: 10px 14px; background: #F9FAFB; border: 1px dashed var(--border); border-radius: 8px; font-style: italic; }}
  .ai-studio-persona-section .ai-voice-block {{ margin-top: 18px; padding-top: 14px; border-top: 1px dashed var(--border); }}
  .ai-studio-persona-section .ai-voice-header {{ font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 4px; }}
  .ai-studio-persona-section .ai-voice-pill {{ font-size: 10px; font-weight: 500; padding: 2px 8px; border-radius: 10px; background: rgba(109, 40, 217, 0.10); color: #6D28D9; margin-left: 6px; }}
  .ai-studio-persona-section .ai-voice-subtitle {{ font-size: 12px; color: var(--text-secondary); margin-bottom: 10px; font-style: italic; }}
  .ai-studio-persona-section .ai-voice-section {{ margin-top: 10px; padding-top: 8px; border-top: 1px dotted #d1d5db; }}
  .ai-studio-persona-section .ai-voice-section:first-of-type {{ border-top: none; padding-top: 0; }}
  .ai-studio-persona-section .ai-voice-section-header {{ font-size: 11px; font-weight: 700; color: #4C1D95; text-transform: uppercase; letter-spacing: 0.3px; margin-bottom: 4px; }}
  .ai-studio-persona-section .ai-voice-section-hint {{ font-weight: 400; text-transform: none; letter-spacing: 0; color: #9ca3af; font-style: italic; margin-left: 6px; }}
  .ai-studio-persona-section .ai-voice-list {{ list-style: none; padding: 0; margin: 0; font-size: 12px; line-height: 1.7; color: var(--text); }}
  .ai-studio-persona-section .ai-voice-list ul {{ list-style: none; padding-left: 12px; margin: 4px 0 0; }}
  .ai-studio-persona-section .ai-voice-avoid {{ font-style: italic; color: #8B5A2B; }}

  /* AI Studio event-card chrome — chat-bubble label, stage badge, memory pill */
  .chat-conv-label.ai-conv-label {{ color: #6D28D9; }}
  .ai-stage-badge {{ display: inline-block; font-size: 9px; font-weight: 700; padding: 2px 7px; border-radius: 10px; color: #FFFFFF; margin-left: 8px; letter-spacing: 0.4px; vertical-align: middle; }}
  .ai-stage-badge.ai-stage-S1 {{ background: #C7D2FE; color: #312E81; }}
  .ai-stage-badge.ai-stage-S2 {{ background: #A78BFA; }}
  .ai-stage-badge.ai-stage-S3 {{ background: #6D28D9; }}
  .ai-stage-badge.ai-stage-S4 {{ background: #4C1D95; }}
  .ai-memory-link {{ display: inline-block; font-size: 10px; padding: 2px 7px; border-radius: 10px; background: rgba(109, 40, 217, 0.08); color: #6D28D9; margin-left: 8px; cursor: help; }}
  .ai-oblique-row {{ font-size: 11px; color: #6B7280; margin: 4px 0 2px; }}

  /* SPT arc strip — proportional 4-stage band showing how the user's
     conversations distribute across S1→S4 (Social Penetration Theory). */
  .ai-arc-strip {{ margin-top: 12px; padding: 10px 12px; background: linear-gradient(to right, rgba(109, 40, 217, 0.04), rgba(109, 40, 217, 0.10)); border-radius: 8px; border: 1px solid rgba(109, 40, 217, 0.12); }}
  .ai-arc-strip-header {{ display: flex; align-items: baseline; gap: 8px; margin-bottom: 6px; }}
  .ai-arc-strip-label {{ font-size: 10px; font-weight: 600; color: #4C1D95; text-transform: uppercase; letter-spacing: 0.4px; white-space: nowrap; }}
  .ai-arc-strip-sublabel {{ font-size: 11px; color: #6B7280; }}
  .ai-arc-band {{ display: flex; height: 18px; border-radius: 4px; overflow: hidden; }}
  .ai-arc-band-seg {{ display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 600; color: #FFFFFF; cursor: help; transition: opacity 0.12s ease; min-width: 0; padding: 0 4px; white-space: nowrap; overflow: hidden; }}
  .ai-arc-band-seg:hover {{ opacity: 0.8; }}
  .ai-arc-band-seg.seg-S1 {{ background: #C7D2FE; color: #312E81; }}
  .ai-arc-band-seg.seg-S2 {{ background: #A78BFA; }}
  .ai-arc-band-seg.seg-S3 {{ background: #6D28D9; }}
  .ai-arc-band-seg.seg-S4 {{ background: #4C1D95; }}
  .ai-arc-legend {{ display: flex; gap: 12px; margin-top: 8px; font-size: 10px; color: #6B7280; flex-wrap: wrap; }}
  .ai-arc-legend-item {{ display: inline-flex; align-items: center; gap: 4px; }}
  .ai-arc-legend-swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; }}
  .ai-arc-legend-swatch.sw-S1 {{ background: #C7D2FE; }}
  .ai-arc-legend-swatch.sw-S2 {{ background: #A78BFA; }}
  .ai-arc-legend-swatch.sw-S3 {{ background: #6D28D9; }}
  .ai-arc-legend-swatch.sw-S4 {{ background: #4C1D95; }}
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
      <span>{n_trending_events} trending events</span>
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
    <button class="filter-btn" data-filter-key="ai_studio">AI Studio</button>
    <span class="filter-spacer"></span>
    <button class="filter-btn" data-filter-key="test">Test queries only</button>
  </div>

  <!-- AI Studio persona card — visible only when "AI Studio" filter is selected.
       Populated from profileData.ai_studio_persona by JS below. Milestone (e)
       partial slice; full event timeline + SPT arc strip ship after (b)+(c). -->
  <div id="ai-studio-persona-section" style="display:none;"></div>

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
    const audience = ap.audience_type || '';
    const audienceLens = ap.audience_lens || '';
    const audienceDesign = ap.audience_design_note || '';
    let pills = '';
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
      const isRanking = (t.task_type || '').match(/^(personalized_recommendation|hidden_persona_recommendation|at_ai_directive_followup|short_vs_long_term_lifecycle)$/);

      let sections = '';
      if (t.example_response) {{
        sections += `<div class="ts-section"><div class="ts-label">Example Response</div><div class="ts-body" style="white-space:pre-wrap;">${{escapeHtml(t.example_response)}}</div></div>`;
      }}
      // Normalize inferior_response: some tasks (e.g. preference_shift_followthrough)
      // emit a plain string; the dict-shape path expects an object with text + flaw_kind keys.
      const _infRaw = t.inferior_response;
      const _infObj = (typeof _infRaw === 'string')
        ? {{text: _infRaw}}
        : (_infRaw && typeof _infRaw === 'object' ? _infRaw : null);
      if (_infObj && _infObj.text) {{
        const flaw = _infObj.flaw_kind || '';
        // The voice-evidence smoke check status is build-time QA metadata
        // (`voice_evidence_smoke_check`) — kept on the instance for debugging
        // but intentionally NOT rendered on the user-facing test card.
        const smoke = '';
        const regen = (_infObj.regen_reason)
          ? ` <small style="color:#92400E;font-weight:normal;">(regen: ${{escapeHtml(_infObj.regen_reason)}})</small>` : '';
        // Highlight violations (only daily_personalized_briefing for now —
        // the disliked_recent flaw injects a specific topic into the gold,
        // so the topic_hint / persona_item from flaw_evidence pinpoints
        // exactly what the agent should NOT have surfaced. Render-only
        // bolding; the underlying inferior_response.text is unmodified.
        let infBody = escapeHtml(_infObj.text);
        if (t.task_type === 'daily_personalized_briefing') {{
          const ev = _infObj.flaw_evidence || {{}};
          const spans = [];
          if (ev.topic_hint) spans.push(ev.topic_hint);
          if (ev.persona_item && ev.persona_item !== ev.topic_hint) spans.push(ev.persona_item);
          // Sort longest first so super-strings substitute before sub-strings.
          spans.sort((a, b) => b.length - a.length);
          if (spans.length > 0) infBody = boldVoiceEvidence(infBody, spans);
        }}
        const violationHint = (t.task_type === 'daily_personalized_briefing'
                               && _infObj.flaw_evidence
                               && (_infObj.flaw_evidence.topic_hint
                                   || _infObj.flaw_evidence.persona_item))
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
        // Hidden-persona anchor badges — for new_suggestions tasks, surface
        // the dominant_frame + label of every hidden persona whose
        // evidence_hashtags overlap the gold. Same purple badge style used
        // on profile preference labels (`.badge.hidden-persona`) so the
        // reader recognizes it as the same signal class.
        let anchorBadges = '';
        if (Array.isArray(t.gold_anchor_personas) && t.gold_anchor_personas.length > 0) {{
          const items = t.gold_anchor_personas.map(a => {{
            const label = escapeHtml(a.label || a.type || '(unnamed persona)');
            const typ = a.type ? `<small style="opacity:0.75;margin-left:4px;">${{escapeHtml(a.type)}}</small>` : '';
            const frame = a.dominant_frame ? ` · ${{escapeHtml(a.dominant_frame)}}` : '';
            const matched = (a.matched_hashtags || []).slice(0, 5)
              .map(h => '#' + h).join(' ');
            const titleAttr = matched ? ` title="matches: ${{escapeHtml(matched)}}"` : '';
            return `<span class="badge hidden-persona"${{titleAttr}}>${{label}}${{typ}}${{frame}}</span>`;
          }}).join(' ');
          anchorBadges = `<div style="margin-top:6px;"><small style="color:var(--text-secondary);">Hidden-persona anchor for the gold:</small><br/>${{items}}</div>`;
        }}
        sections += `<div class="ts-section"><div class="ts-label">Groundtruth Preference${{gtHint}}</div><div class="ts-body" style="white-space:pre-wrap;">${{gtHtml}}</div>${{anchorBadges}}</div>`;
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
      // The section is ALWAYS rendered, even when empty — for proactive
      // tasks the empty body itself is the signal that the AI receives
      // no synthetic user message and must decide unprompted.
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
    const isTrending = !!ev.is_trending;
    const card = document.createElement('div');
    card.className = `event-card app-${{app}}${{isImplicitNeg ? ' implicit-negative' : ''}}${{isAd ? ' is-ad' : ''}}${{isTrending ? ' is-trending' : ''}}`;
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
          ${{isTrending ? `<span class="badge trending">${{ev.trending_relevance === 'relevant' ? '📈 Trending · Relevant' : '📈 Trending'}}</span>` : ''}}
        </div>
        ${{isTrending && ev.trending_topic ? `<div class="trending-topic">Trend: ${{escapeHtml(ev.trending_topic)}}</div>` : ''}}
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

    // Chatbot / AI Studio conversation rendering
    let convHtml = '';
    const isAiStudio = (ev._app === 'AI_Studio');
    if (ev.conversation && ev.conversation.length > 0) {{
      // AI Studio events show the chosen AI character's name on assistant
      // bubbles (e.g. "Rowan" instead of generic "AI") and surface the
      // SPT stage badge + memory-link pills.
      const aiName = isAiStudio
        ? escapeHtml((profileData && profileData.ai_studio_persona && profileData.ai_studio_persona.character_name) || 'AI')
        : 'AI';
      const stageTooltips = {{
        S1: 'SPT stage S1 — orientation: surface scripts, casual preferences. What a stranger safely shares.',
        S2: 'SPT stage S2 — exploratory affective: early opinions, mild personal anecdotes. Still hedged.',
        S3: 'SPT stage S3 — affective exchange: genuine views, vulnerabilities, mild fears.',
        S4: 'SPT stage S4 — stable exchange: core beliefs, intimate values, deep fears. Reserved for trusted relationships.',
      }};
      const stage = (isAiStudio && ev.ai_studio_metadata) ? ev.ai_studio_metadata.intimacy_stage_at_event : '';
      const stageBadge = stage
        ? `<span class="ai-stage-badge ai-stage-${{stage}}" title="${{(stageTooltips[stage] || '').replace(/"/g, '&quot;')}}">${{stage}}</span>`
        : '';
      const memoryPills = (isAiStudio && ev.prior_session_refs && ev.prior_session_refs.length)
        ? (() => {{
            const n = ev.prior_session_refs.length;
            const summary = (ev.memory_used_summary || '').trim();
            const tipBase = `Cross-session memory: this conversation references ${{n}} earlier AI Studio session${{n === 1 ? '' : 's'}}.`;
            const tip = summary ? `${{tipBase}}\n\nRecall summary: ${{summary}}` : tipBase;
            return `<span class="ai-memory-link" title="${{tip.replace(/"/g, '&quot;')}}">↗ recalls ${{n}} earlier session${{n === 1 ? '' : 's'}}</span>`;
          }})()
        : '';
      const obliqueChips = (isAiStudio && ev.oblique_reference_to_hidden_personas && ev.oblique_reference_to_hidden_personas.length)
        ? '<div class="ai-oblique-row"><span class="ai-row-key">Oblique anchors (hidden personas, never named in text):</span> '
          + ev.oblique_reference_to_hidden_personas.map(t => `<span class="ai-chip">${{escapeHtml(t)}}</span>`).join('')
          + '</div>'
        : '';
      let convLabel = '';
      if (ev.conversation_type) {{
        const ctypeLabel = ev.conversation_type.replace(/_/g, ' ')
          + (ev.ask_to_forget ? ' &middot; ask-to-forget' : '');
        convLabel = `<div class="chat-conv-label ${{isAiStudio ? 'ai-conv-label' : ''}}">${{ctypeLabel}}${{stageBadge}}${{memoryPills}}</div>`;
      }} else if (stageBadge || memoryPills) {{
        convLabel = `<div class="chat-conv-label ${{isAiStudio ? 'ai-conv-label' : ''}}">${{stageBadge}}${{memoryPills}}</div>`;
      }}
      let bubbles = ev.conversation.map(t => {{
        const cls = t.role === 'user' ? 'user-bubble' : 'assistant-bubble';
        const label = t.role === 'user' ? 'You' : aiName;
        return `<div class="chat-bubble ${{cls}}"><div class="chat-role">${{label}}</div>${{t.content}}</div>`;
      }}).join('');
      convHtml = `${{convLabel}}${{obliqueChips}}<div class="chat-thread">${{bubbles}}</div>`;
    }} else if (fmt.user_message) {{
      convHtml = `<div class="user-message">${{fmt.user_message}}</div>`;
    }}

    const contentHtml = renderContent(ev);
    card.innerHTML = headerHtml + contentHtml + prefsHtml + convHtml;
    grid.appendChild(card);
  }});

  timeline.appendChild(grid);

  // -- AI Studio persona card builder (milestone (e) partial) ----------
  // Renders profileData.ai_studio_persona — the SAME 4-layer voice model
  // used for user_voice (identity_spine + idiolect + repertoire + soft
  // holdovers + negatives), but for a fictional AI character.
  // Event timeline + SPT arc strip land in milestones (b)+(c)+(e) full.
  const aiPersonaSection = document.getElementById('ai-studio-persona-section');
  function renderAiStudioPersona(asp) {{
    if (!asp || !asp.persona_archetype) return '';
    const archetype = escapeHtml(asp.persona_archetype || '');
    const name = escapeHtml(asp.character_name || '');
    const bio = escapeHtml(asp.backstory_brief || '');
    const stance = escapeHtml(asp.relational_stance || '');
    const style = escapeHtml(asp.communication_style || '');
    const addressTerms = (asp.address_terms || []).map(escapeHtml).join(', ');
    const niche = asp.niche_specifier ? escapeHtml(asp.niche_specifier) : '';
    const rs = asp.romantic_specifier || {{}};
    const rsAxes = ['gender_presentation', 'sexuality_orientation',
                    'aesthetic_vibe', 'body_role_coding',
                    'relational_dynamic', 'explicitness_band'];
    const rsParts = rsAxes
      .map(k => rs[k] ? `${{k.replace(/_/g, ' ')}}: <b>${{escapeHtml(rs[k])}}</b>` : null)
      .filter(Boolean).join(' &middot; ');
    const palette = (asp.emoji_palette || []).join(' ');
    const palIntensity = escapeHtml(asp.emoji_intensity_default || '');
    const forbiddenCount = (asp.forbidden_phrases || []).length;
    const sigs = (asp.signature_phrases || []).map(s =>
      `<span class="ai-sig-chip">"${{escapeHtml(s)}}"</span>`).join('');
    const topical = (asp.topical_strengths || []).map(t =>
      `<span class="ai-chip">${{escapeHtml(t)}}</span>`).join('');
    const guardrails = asp.generation_guardrails || {{}};
    const guardrailParts = [];
    if (guardrails.boundary_on_diagnosis === 'never_diagnose') guardrailParts.push('No diagnosis');
    if (guardrails.boundary_on_medication_advice) guardrailParts.push('Decline medication advice');
    if (guardrails.anti_sycophancy_pledge) guardrailParts.push('Push back when warranted');
    if (guardrails.honesty_when_asked_if_ai === 'answer_truthfully') guardrailParts.push('Honest when asked');
    const guardrailLine = guardrailParts.join(' · ');
    const rationale = escapeHtml(asp.fit_rationale || '');

    // ---- 4-layer voice sections (mirror user_voice rendering) -----
    const sections = [];

    // Layer 1 — Character Identity Spine
    const spine = asp.identity_spine || {{}};
    if (Object.keys(spine).length) {{
      const liwc = spine.liwc_anchors_inferred || {{}};
      const liwcStr = Object.keys(liwc).map(k => `${{escapeHtml(k)}}=${{escapeHtml(String(liwc[k]))}}`).join(', ');
      const b5 = spine.big_five_proxy || {{}};
      const b5Str = Object.keys(b5).map(k => `<li><span class="ai-row-key">${{escapeHtml(k)}}:</span> ${{escapeHtml(String(b5[k]))}}</li>`).join('');
      const rows = [];
      if (spine.agency_communion)                       rows.push(`<li><span class="ai-row-key">agency/communion:</span> ${{escapeHtml(spine.agency_communion)}}</li>`);
      if ((spine.redemption_motifs||[]).length)         rows.push(`<li><span class="ai-row-key">redemption motifs:</span> ${{(spine.redemption_motifs||[]).map(escapeHtml).join('; ')}}</li>`);
      if ((spine.contamination_motifs||[]).length)      rows.push(`<li><span class="ai-row-key">contamination motifs:</span> ${{(spine.contamination_motifs||[]).map(escapeHtml).join('; ')}}</li>`);
      if ((spine.life_stage_preoccupations||[]).length) rows.push(`<li><span class="ai-row-key">life-stage preoccupations:</span> ${{(spine.life_stage_preoccupations||[]).map(escapeHtml).join('; ')}}</li>`);
      if ((spine.signature_concerns||[]).length)        rows.push(`<li><span class="ai-row-key">signature concerns:</span> ${{(spine.signature_concerns||[]).map(escapeHtml).join('; ')}}</li>`);
      if (liwcStr)                                      rows.push(`<li><span class="ai-row-key">LIWC anchors (inferred):</span> ${{liwcStr}}</li>`);
      if (b5Str)                                        rows.push(`<li><span class="ai-row-key">Big-Five proxy:</span><ul style="margin-top:2px;">${{b5Str}}</ul></li>`);
      if (rows.length) {{
        sections.push(`
          <div class="ai-voice-section">
            <div class="ai-voice-section-header">Layer 1 — Character identity spine <span class="ai-voice-section-hint">drives WHAT this character brings up</span></div>
            <ul class="ai-voice-list">${{rows.join('')}}</ul>
          </div>`);
      }}
    }}

    // Layer 2 — Character Idiolect
    const idio = asp.idiolect || {{}};
    if (Object.keys(idio).length) {{
      const sp = idio.syntactic_preferences || {{}};
      const af = idio.appraisal_fingerprint || {{}};
      const tmpls = (idio.constructional_templates || []).map(t => {{
        const pat = escapeHtml(t.pattern || '');
        const ex  = escapeHtml(t.example_realization || '');
        return `<li><code style="font-family:ui-monospace,Menlo,Monaco,monospace;background:rgba(109,40,217,0.07);padding:1px 5px;border-radius:3px;color:#4C1D95;">${{pat}}</code> <span style="opacity:0.7;">e.g. "${{ex}}"</span></li>`;
      }}).join('');
      const residue = (idio.catchphrase_residue || []).map(p => `"${{escapeHtml(p)}}"`).join(', ');
      const rows = [];
      if (idio.function_word_profile)             rows.push(`<li><span class="ai-row-key">function-word profile:</span> ${{escapeHtml(idio.function_word_profile)}}</li>`);
      if (Object.keys(sp).length)                 rows.push(`<li><span class="ai-row-key">sentences:</span> shape=${{escapeHtml(sp.sentence_length_shape||'?')}}, embedding=${{escapeHtml(sp.clause_embedding||'?')}}, parataxis/hypotaxis=${{escapeHtml(sp.parataxis_hypotaxis||'?')}}, fragments=${{escapeHtml(sp.fragment_use||'?')}}</li>`);
      if (idio.hedge_booster_ratio)               rows.push(`<li><span class="ai-row-key">hedge/booster:</span> ${{escapeHtml(idio.hedge_booster_ratio)}}</li>`);
      if (Object.keys(af).length)                 rows.push(`<li><span class="ai-row-key">appraisal:</span> attitude=${{escapeHtml(af.attitude_dominant||'?')}}, engagement=${{escapeHtml(af.engagement_style||'?')}}, graduation=${{escapeHtml(af.graduation||'?')}}</li>`);
      if (tmpls)                                  rows.push(`<li><span class="ai-row-key">templates (slot patterns — apply abstractly):</span><ul style="margin-top:2px;">${{tmpls}}</ul></li>`);
      if (residue)                                rows.push(`<li><span class="ai-row-key">catchphrase residue (≤1 per response):</span> ${{residue}}</li>`);
      if (asp.default_capitalization)             rows.push(`<li><span class="ai-row-key">capitalization:</span> ${{escapeHtml(asp.default_capitalization)}}</li>`);
      if (asp.punctuation_habits)                 rows.push(`<li><span class="ai-row-key">punctuation:</span> ${{escapeHtml(asp.punctuation_habits)}}</li>`);
      if (asp.formality !== undefined && asp.formality !== null) rows.push(`<li><span class="ai-row-key">formality:</span> ${{escapeHtml(String(Number(asp.formality).toFixed(2)))}}</li>`);
      if (palette)                                rows.push(`<li><span class="ai-row-key">emoji palette:</span> ${{escapeHtml(palette)}} <span style="opacity:0.7;">(${{palIntensity}} intensity)</span></li>`);
      if (rows.length) {{
        sections.push(`
          <div class="ai-voice-section">
            <div class="ai-voice-section-header">Layer 2 — Character idiolect <span class="ai-voice-section-hint">must survive paraphrase — not just word imitation</span></div>
            <ul class="ai-voice-list">${{rows.join('')}}</ul>
          </div>`);
      }}
    }}

    // Layer 3 — Character Repertoire
    const rep = asp.repertoire || {{}};
    if (Object.keys(rep).length) {{
      const stanceChips = (rep.stances || []).map(s => `<span class="ai-chip">${{escapeHtml(s)}}</span>`).join(' ');
      const regChips    = (rep.registers || []).map(s => `<span class="ai-chip">${{escapeHtml(s)}}</span>`).join(' ');
      const genreChips  = (rep.speech_genre_fluency || []).map(s => `<span class="ai-chip">${{escapeHtml(s)}}</span>`).join(' ');
      const rows = [];
      if (stanceChips)                       rows.push(`<li><span class="ai-row-key">stances:</span> ${{stanceChips}}</li>`);
      if (regChips)                          rows.push(`<li><span class="ai-row-key">registers:</span> ${{regChips}}</li>`);
      if (rep.backstage_frontstage_range)    rows.push(`<li><span class="ai-row-key">backstage/frontstage range:</span> ${{escapeHtml(rep.backstage_frontstage_range)}}</li>`);
      if (genreChips)                        rows.push(`<li><span class="ai-row-key">speech-genre fluency:</span> ${{genreChips}}</li>`);
      if (rows.length) {{
        sections.push(`
          <div class="ai-voice-section">
            <div class="ai-voice-section-header">Layer 3 — Character repertoire <span class="ai-voice-section-hint">stable inventory of stances/registers/genres</span></div>
            <ul class="ai-voice-list">${{rows.join('')}}</ul>
          </div>`);
      }}
    }}

    // Voice avoid + forbidden_phrases
    const avoidRows = [];
    if (asp.voice_avoid)        avoidRows.push(`<li class="ai-voice-avoid"><span class="ai-row-key">tones to avoid:</span> ${{escapeHtml(asp.voice_avoid)}}</li>`);
    avoidRows.push(`<li class="ai-voice-avoid"><span class="ai-row-key">forbidden phrases (${{forbiddenCount}}):</span> includes the Rogers-cliché baseline + archetype-specific.</li>`);
    sections.push(`
      <div class="ai-voice-section">
        <div class="ai-voice-section-header">Voice avoid</div>
        <ul class="ai-voice-list">${{avoidRows.join('')}}</ul>
      </div>`);

    return `<div class="ai-studio-persona-section">
      <h2>${{name || 'AI Persona'}} <span class="ai-pill">AI &middot; ${{archetype}}</span></h2>
      <div class="ai-archetype">${{stance}}</div>
      ${{bio ? `<div class="ai-bio">${{bio}}</div>` : ''}}
      ${{style ? `<div class="ai-row"><span class="ai-row-key">Communication style:</span> ${{style}}</div>` : ''}}
      ${{sigs ? `<div class="ai-row"><span class="ai-row-key">Signature phrases (≤1 per conv):</span><br>${{sigs}}</div>` : ''}}
      ${{topical ? `<div class="ai-row"><span class="ai-row-key">Topical strengths:</span> ${{topical}}</div>` : ''}}
      ${{addressTerms ? `<div class="ai-row"><span class="ai-row-key">Address terms:</span> ${{addressTerms}}</div>` : ''}}
      ${{niche ? `<div class="ai-row"><span class="ai-row-key">Niche specifier:</span> <span class="ai-chip">${{niche}}</span></div>` : ''}}
      ${{rsParts ? `<div class="ai-row"><span class="ai-row-key">Romantic specifier:</span> ${{rsParts}}</div>` : ''}}
      <div class="ai-voice-block">
        <div class="ai-voice-header">Character voice <span class="ai-voice-pill">4-layer model</span></div>
        <div class="ai-voice-subtitle">Same structure as user_voice — built from the chosen archetype's character DNA, not from the user's raw data.</div>
        ${{sections.join('')}}
      </div>
      ${{rationale ? `<div class="ai-rationale">Fit rationale &mdash; ${{rationale}}</div>` : ''}}
      ${{guardrailLine ? `<div class="ai-guardrails">Guardrails &middot; ${{guardrailLine}}</div>` : ''}}
      <div id="ai-arc-strip-mount"></div>
    </div>`;
  }}

  // SPT arc strip — proportional 4-stage band. Each segment's width =
  // share of AI Studio conversations spent in that stage. SPT = Social
  // Penetration Theory (Altman & Taylor, 1973) — a model of how relational
  // depth progresses from surface scripts (S1) to intimate disclosure (S4).
  function renderAiArcStrip(events) {{
    const aiEvents = events
      .filter(e => e._app === 'AI_Studio' && e.ai_studio_metadata && e.ai_studio_metadata.intimacy_stage_at_event)
      .sort((a, b) => (a.source_timestamp || 0) - (b.source_timestamp || 0));
    if (!aiEvents.length) return '';
    const stageMeta = {{
      S1: {{ name: 'orientation',          blurb: 'surface scripts, casual preferences — what a stranger safely shares' }},
      S2: {{ name: 'exploratory affective', blurb: 'early opinions, mild personal anecdotes — still hedged'             }},
      S3: {{ name: 'affective exchange',    blurb: 'genuine views, vulnerabilities, mild fears'                          }},
      S4: {{ name: 'stable exchange',       blurb: 'core beliefs, intimate values, deep fears — reserved for trusted relationships' }},
    }};
    const stageOrder = ['S1', 'S2', 'S3', 'S4'];
    const counts = {{ S1: 0, S2: 0, S3: 0, S4: 0 }};
    aiEvents.forEach(e => {{
      const s = e.ai_studio_metadata.intimacy_stage_at_event || 'S1';
      if (counts[s] !== undefined) counts[s] += 1;
    }});
    const total = aiEvents.length;
    const segs = stageOrder
      .filter(s => counts[s] > 0)
      .map(s => {{
        const n = counts[s];
        const pct = (n * 100) / total;
        const meta = stageMeta[s];
        const tip = `${{s}} ${{meta.name}} — ${{meta.blurb}} · ${{n}} of ${{total}} conversations (${{pct.toFixed(1)}}%)`;
        const label = pct >= 6 ? s : '';
        return `<div class="ai-arc-band-seg seg-${{s}}" style="flex: ${{n}} 1 0;" title="${{tip.replace(/"/g, '&quot;')}}">${{label}}</div>`;
      }})
      .join('');
    const legend = stageOrder
      .map(s => `<span class="ai-arc-legend-item"><span class="ai-arc-legend-swatch sw-${{s}}"></span>${{s}} ${{stageMeta[s].name}}</span>`)
      .join('');
    return `<div class="ai-arc-strip">
      <div class="ai-arc-strip-header">
        <span class="ai-arc-strip-label">SPT arc</span>
        <span class="ai-arc-strip-sublabel">how the AI–user relationship deepens across ${{total}} conversations · Social Penetration Theory</span>
      </div>
      <div class="ai-arc-band">${{segs}}</div>
      <div class="ai-arc-legend">${{legend}}</div>
    </div>`;
  }}
  if (aiPersonaSection && profileData && profileData.ai_studio_persona && profileData.ai_studio_persona.persona_archetype) {{
    aiPersonaSection.innerHTML = renderAiStudioPersona(profileData.ai_studio_persona);
    // Mount the SPT arc strip after the card is rendered.
    const arcMount = document.getElementById('ai-arc-strip-mount');
    if (arcMount && eventsData) {{
      arcMount.innerHTML = renderAiArcStrip(eventsData);
    }}
  }}

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
      // AI Studio persona card visibility — show only on the AI Studio tab.
      if (aiPersonaSection) {{
        aiPersonaSection.style.display = (key === 'ai_studio' && profileData && profileData.ai_studio_persona && profileData.ai_studio_persona.persona_archetype) ? 'block' : 'none';
      }}
      const cards = grid.children;
      for (const card of cards) {{
        const kind = card.dataset.kind || '';
        const app = card.dataset.app || '';
        let show;
        if (key === 'all') {{
          show = true;
        }} else if (key === 'test') {{
          show = (kind === 'test');
        }} else if (key === 'ai_studio') {{
          // Milestone (b)+(c) will produce ai_studio events; today there
          // are none, so the timeline is empty under this filter — only
          // the persona card shows.
          show = (app === 'ai_studio' && (kind === 'event' || kind === 'test'));
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
