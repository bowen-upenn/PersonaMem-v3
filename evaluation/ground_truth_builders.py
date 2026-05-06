"""Per-task ground-truth slice builders for agentic eval prompts.

The agentic prompts (T6-T19, defined in prompts_agentic.py) historically
told the model to call MCP tools (mcp__instagram__list_dms etc.) to fetch
data. In llm_longctx and agent_longctx modes those tools don't exist, and
even in mcp_agent mode the model sometimes refuses upfront ("I can't access
your DMs"). These builders fetch the focused data each task needs from
backend/{user_id}/*.json and embed it in the prompt so the model has real
data to ground its response in.

For voice-matching tasks (T6/T9/T10/T12/T13/T16) the slice intentionally
includes profile.user_voice + app_personas[target_app] -- these tasks
measure voice-fidelity given context, not voice inference. Other agentic
tasks (T8/T11/T17/T18/T19) keep the Phase G firewall: events only, no
profile leak.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from evaluation.backend_query import APPS, BackendQuery


_DAY = 24 * 3600
_SOCIAL_APPS = ("instagram", "facebook", "threads")
_ALL_APPS = ("instagram", "facebook", "threads", "chatbot")

# Voice-mimic compose tasks (agentic_user_tone_post, agentic_composed_post,
# agentic_send_post, agentic_cross_app_repost, agentic_auto_reply) require
# at least this many user-voiced samples in history before t_test. Below
# the floor, the AI under evaluation has insufficient evidence of how the
# user writes — `profile.user_voice` is firewalled at test time, so the
# only voice signal it can read is what self-posts, DM-thread user-side
# messages, and chatbot user-turns actually contain. 5 is the empirical
# minimum that lets a model reasonably infer a recurring voice pattern.
USER_VOICED_SAMPLES_FLOOR = 5


def count_user_voiced_samples_before(
    bq: BackendQuery, user_id: str, t_test: int,
) -> int:
    """Count distinct user-voiced sample events visible to the AI under
    evaluation before ``t_test``. A "sample" is one of:

      (a) a self-authored post (``is_self_authored=True``, ``is_dm=False``).
      (b) a DM thread (``is_dm=True``) with at least one message where
          ``sender == "self"``.
      (c) a chatbot event with at least one ``role: "user"`` turn carrying
          non-empty content (the user's chat-turn text + any pasted
          drafts inside it).

    Counted across all 4 apps; one event = at most one sample. Used as
    the pre-test floor for voice-mimic agentic tasks (T6/T9/T10/T12/T13)
    via ``has_enough_user_voiced_history``.
    """
    n = 0
    for app in _ALL_APPS:
        try:
            events = bq.get_events(
                user_id, app, since_timestamp=t_test, include_dms=True,
            )
        except Exception:
            continue
        for e in events:
            if not isinstance(e, dict):
                continue
            # (a) self-authored social post
            if e.get("is_self_authored") and not e.get("is_dm"):
                n += 1
                continue
            # (b) DM thread with user-side outbound text
            if e.get("is_dm"):
                msgs = e.get("messages") or []
                has_user_msg = any(
                    isinstance(m, dict)
                    and (m.get("sender") or "") == "self"
                    and (m.get("text") or "").strip()
                    for m in msgs
                )
                if has_user_msg:
                    n += 1
                continue
            # (c) chatbot conversation with at least one user turn
            conv = e.get("conversation") or []
            if isinstance(conv, list) and any(
                isinstance(t, dict)
                and t.get("role") == "user"
                and (t.get("content") or "").strip()
                for t in conv
            ):
                n += 1
    return n


def has_enough_user_voiced_history(
    bq: BackendQuery, user_id: str, t_test: int,
    floor: int = USER_VOICED_SAMPLES_FLOOR,
) -> bool:
    """Convenience wrapper used by voice-mimic compose-task builders.
    Returns True iff the user has at least ``floor`` user-voiced samples
    before ``t_test``. Compose-task builders short-circuit (return
    ``[]``) when this returns False — the test card would be unanswerable
    by an agent that cannot see the user's voice ground truth."""
    return count_user_voiced_samples_before(bq, user_id, t_test) >= floor


# =========================================================================
# Public entry point
# =========================================================================

def build_for_task(task_id: str, bq: BackendQuery, user_id: str,
                   t_test: int, inst: dict) -> str:
    """Return a markdown ground-truth block for the task. Empty string if
    the task has no builder or no data applies."""
    fn = _DISPATCH.get(task_id)
    if fn is None:
        return ""
    try:
        return fn(bq, user_id, int(t_test), inst or {})
    except Exception as e:
        return f"## Ground-truth context (real data from the user)\n\n_(error building context: {type(e).__name__}: {e})_\n"


# =========================================================================
# Formatting helpers
# =========================================================================

def _fmt_ts(ts: int | float | None) -> str:
    if not ts:
        return "?"
    try:
        return _dt.datetime.fromtimestamp(int(ts), tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"


def _truncate(text: str, n: int) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "…"


def _capitalize_app(app: str) -> str:
    """Profile.app_personas keys use capitalized app names."""
    return (app or "").capitalize()


def _section_header(title: str) -> str:
    return f"\n### {title}\n"


def _wrap_block(body_sections: list[str]) -> str:
    if not body_sections:
        return ""
    return ("## Ground-truth context (real user data — base your response on this)\n"
            + "".join(body_sections))


def _format_voice_block(
    profile: dict,
    target_app: str | None,
    *,
    voice_evidence_spans: list | None = None,
) -> str:
    """Render a SCOPED voice slice for the GT preference card on
    voice-matching tasks (T6/T9/T10/T12/T13/T16).

    Delegates to ``data_preparation.prompts.render_voice_for_test_card``
    so eval-side and viz-side share one renderer. The scope rule:
      - Always: identity_spine.signature_concerns + idiolect markers +
        ONE constructional template + per-app surface block + per-app
        delta_summary.
      - If the user's profile carries a strongest hidden_persona with a
        resolvable dominant_frame, include it.
      - If `voice_evidence_spans` is provided, palette emoji and
        catchphrase residue strings that actually surface in the
        example/inferior pair are bolded inline.

    Skipped if the profile has no user_voice. Falls back to legacy flat
    rendering when the layered schema fields are absent (older
    snapshots).
    """
    voice = (profile or {}).get("user_voice") or {}
    if not voice:
        return ""
    cap_app = _capitalize_app(target_app) if target_app else ""
    ap = ((profile or {}).get("app_personas") or {}).get(cap_app) or {} if cap_app else {}

    # Resolve the user's strongest hidden_persona dominant_frame.
    # Prefer the audited frame on motivation_audit; fall back to the
    # structural type-default. None when no organic clusters exist.
    dominant_frame: str | None = None
    hps = [hp for hp in ((profile or {}).get("hidden_personas") or [])
           if not hp.get("is_synthetic")]
    if hps:
        from data_preparation.prompts import cluster_dominant_frame as _resolve_frame
        top = max(hps, key=lambda h: int(h.get("evidence_rows") or 0))
        f = _resolve_frame(top)
        if f and f != "none":
            dominant_frame = f

    from data_preparation.prompts import render_voice_for_test_card as _layered_render
    body = _layered_render(
        voice, ap,
        target_app=target_app or "",
        dominant_frame=dominant_frame,
        voice_evidence_spans=voice_evidence_spans or [],
    )
    if not body.strip():
        return ""
    return _section_header("User voice — scoped for this task") + body


def _format_self_posts(bq: BackendQuery, user_id: str, t_test: int,
                       target_app: str, lookback_days: int = 14,
                       max_posts: int = 8) -> str:
    """Render the user's recent self-authored posts on `target_app`."""
    cutoff = t_test - lookback_days * _DAY
    posts = []
    for e in bq.get_events(user_id=user_id, app=target_app, since_timestamp=t_test):
        if not e.get("is_self_authored") or e.get("is_dm"):
            continue
        if (e.get("source_timestamp") or 0) < cutoff:
            continue
        posts.append(e)
    posts = posts[-max_posts:]  # already sorted ascending
    if not posts:
        return _section_header(f"Recent self-authored posts on {target_app} (last {lookback_days}d)") + \
               f"_(no self-authored posts on {target_app} in the past {lookback_days} days)_\n"
    lines = [_section_header(f"Recent self-authored posts on {target_app} (last {lookback_days}d)")]
    for p in posts:
        ts = _fmt_ts(p.get("source_timestamp"))
        content = p.get("content") or {}
        caption = _truncate(content.get("caption") or content.get("title") or "", 360)
        action = ((p.get("interaction_format") or {}).get("action_label")) or "posted"
        tags = " ".join(p.get("source_hashtags") or [])
        lines.append(f"- [{ts}] _{action}_: \"{caption}\"")
        if tags:
            lines.append(f"  · {tags}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _format_dm_thread(thread: dict, max_msgs: int = 30) -> str:
    """Render one DM thread (the dict returned by BackendQuery.get_dm_thread)."""
    if not thread:
        return ""
    tid = thread.get("thread_id", "?")
    parts = thread.get("participants") or []
    is_group = thread.get("is_group", False)
    msgs = thread.get("results") or thread.get("messages") or []
    msgs = msgs[-max_msgs:]
    head = (f"**thread_id**: {tid} — "
            f"participants: {', '.join(parts) if parts else '?'} "
            f"(group={is_group}, msgs_shown={len(msgs)})\n")
    lines = [head]
    for m in msgs:
        ts = _fmt_ts(m.get("timestamp"))
        sender = m.get("sender") or "?"
        text = _truncate(m.get("text") or "", 280)
        lines.append(f"- [{ts}] {sender}: \"{text}\"")
        fwd = m.get("forwarded_content")
        if fwd:
            cap = _truncate(((fwd.get("content") or {}).get("caption")
                             or (fwd.get("content") or {}).get("title") or ""), 200)
            tags = " ".join(fwd.get("source_hashtags") or [])
            lines.append(f"  ↳ forwarded post ({fwd.get('source_app','?')}): "
                         f"\"{cap}\" {tags}".rstrip())
    return "\n".join(lines) + "\n"


def _format_friend(fr: dict) -> str:
    if not fr:
        return ""
    interests = ", ".join(fr.get("shared_interests") or [])
    return (f"- **{fr.get('friend_id','?')}** — {fr.get('display_name','?')} "
            f"(relationship: {fr.get('relationship_depth','?')}; "
            f"shared interests: {interests or '—'})")


# =========================================================================
# Per-task builders
# =========================================================================

def _t6_user_tone_post(bq: BackendQuery, user_id: str, t_test: int, inst: dict) -> str:
    target_app = inst.get("target_app") or "instagram"
    profile = bq.get_full_profile(user_id)
    sections: list[str] = []
    voice = _format_voice_block(profile, target_app)
    if voice:
        sections.append(voice)
    sections.append(_format_self_posts(bq, user_id, t_test, target_app, lookback_days=14, max_posts=6))
    # Recent positive engagement on target_app — what topics the user cared about
    recent_pos = []
    for e in bq.get_events(user_id=user_id, app=target_app, since_timestamp=t_test,
                            interaction_type="explicit_positive"):
        if (e.get("source_timestamp") or 0) < t_test - 7 * _DAY:
            continue
        recent_pos.append(e)
    recent_pos = recent_pos[-8:]
    if recent_pos:
        sections.append(_section_header(f"Recent explicit-positive engagement on {target_app} (last 7d)"))
        for e in recent_pos:
            ts = _fmt_ts(e.get("source_timestamp"))
            content = e.get("content") or {}
            cap = _truncate(content.get("caption") or content.get("title") or "", 200)
            tags = " ".join(e.get("source_hashtags") or [])
            sections.append(f"- [{ts}] \"{cap}\" {tags}".rstrip() + "\n")
    return _wrap_block(sections)


def _t8_dm_digest(bq: BackendQuery, user_id: str, t_test: int, inst: dict) -> str:
    target_app = inst.get("target_app") or "facebook"
    window = inst.get("window") or "24h"
    win_seconds = _DAY if window == "24h" else 7 * _DAY
    cutoff = t_test - win_seconds

    page = bq.list_dm_threads(user_id=user_id, app=target_app,
                               since_timestamp=t_test, limit=20)
    threads = page.get("results") or []
    in_window = [t for t in threads if (t.get("latest_ts") or 0) >= cutoff]
    sections: list[str] = []
    sections.append(_section_header(f"DM threads on {target_app} in past {window} "
                                     f"(at T={_fmt_ts(t_test)})"))
    if not in_window:
        sections.append(f"_(no DM threads on {target_app} with messages in the past {window})_\n")
        return _wrap_block(sections)
    for t in in_window:
        full = bq.get_dm_thread(user_id=user_id, app=target_app,
                                 thread_id=t["thread_id"],
                                 since_timestamp=t_test, limit=30)
        if full:
            sections.append(_format_dm_thread(full, max_msgs=15))
    return _wrap_block(sections)


def _t9_cross_app_repost(bq: BackendQuery, user_id: str, t_test: int, inst: dict) -> str:
    target_app = inst.get("target_app") or "instagram"
    profile = bq.get_full_profile(user_id)
    sections: list[str] = []
    voice = _format_voice_block(profile, target_app)
    if voice:
        sections.append(voice)
    sections.append(_format_self_posts(bq, user_id, t_test, target_app, lookback_days=21, max_posts=6))
    return _wrap_block(sections)


def _t10_auto_reply(bq: BackendQuery, user_id: str, t_test: int, inst: dict) -> str:
    target_app = inst.get("target_app") or "instagram"
    thread_id = inst.get("thread_id")
    sender_id = inst.get("sender_id")
    profile = bq.get_full_profile(user_id)
    sections: list[str] = []
    voice = _format_voice_block(profile, target_app)
    if voice:
        sections.append(voice)
    if thread_id:
        full = bq.get_dm_thread(user_id=user_id, app=target_app,
                                 thread_id=thread_id, since_timestamp=t_test, limit=50)
        if full:
            sections.append(_section_header(f"Full thread {thread_id} on {target_app}"))
            sections.append(_format_dm_thread(full, max_msgs=30))
    if sender_id:
        fr = bq.get_friend(user_id, sender_id)
        if fr:
            sections.append(_section_header(f"Sender {sender_id} (from friends graph)"))
            sections.append(_format_friend(fr) + "\n")
    return _wrap_block(sections)


def _t11_vague_refind(bq: BackendQuery, user_id: str, t_test: int, inst: dict) -> str:
    topic = (inst.get("topic") or "").lstrip("#").strip()
    if not topic:
        return ""
    sections: list[str] = []
    sections.append(_section_header(f"Events whose hashtags or content match \"{topic}\" (top 10 most recent)"))
    matches: list[tuple[str, dict]] = []
    for app in APPS:
        for e in bq.get_events(user_id=user_id, app=app, since_timestamp=t_test, hashtag=topic):
            matches.append((app, e))
    matches.sort(key=lambda kv: kv[1].get("source_timestamp", 0), reverse=True)
    matches = matches[:10]
    if not matches:
        sections.append(f"_(no events match topic \"{topic}\")_\n")
        return _wrap_block(sections)
    for app, e in matches:
        ts = _fmt_ts(e.get("source_timestamp"))
        content = e.get("content") or {}
        title = _truncate(content.get("title") or "", 100)
        cap = _truncate(content.get("caption") or content.get("overall_description") or "", 240)
        tags = " ".join(e.get("source_hashtags") or [])
        oid = e.get("source_object_id", "?")
        author = e.get("author_id") or "?"
        sections.append(f"- [{ts}] {app}/{oid} (author={author}): "
                         f"**{title}** \"{cap}\" {tags}".rstrip() + "\n")
    return _wrap_block(sections)


def _t12_composed_post(bq: BackendQuery, user_id: str, t_test: int, inst: dict) -> str:
    # Same context as T6/T9: voice + recent self-posts.
    target_app = inst.get("target_app") or "instagram"
    profile = bq.get_full_profile(user_id)
    sections: list[str] = []
    voice = _format_voice_block(profile, target_app)
    if voice:
        sections.append(voice)
    sections.append(_format_self_posts(bq, user_id, t_test, target_app, lookback_days=21, max_posts=6))
    return _wrap_block(sections)


def _t13_send_post(bq: BackendQuery, user_id: str, t_test: int, inst: dict) -> str:
    target_app = inst.get("target_app") or "instagram"
    profile = bq.get_full_profile(user_id)
    sections: list[str] = []
    voice = _format_voice_block(profile, target_app)
    if voice:
        sections.append(voice)
    sections.append(_format_self_posts(bq, user_id, t_test, target_app, lookback_days=21, max_posts=6))
    return _wrap_block(sections)


def _t16_group_dm_summary(bq: BackendQuery, user_id: str, t_test: int, inst: dict) -> str:
    target_app = inst.get("target_app") or "instagram"
    thread_id = inst.get("thread_id")
    profile = bq.get_full_profile(user_id)
    sections: list[str] = []
    voice = _format_voice_block(profile, target_app)
    if voice:
        sections.append(voice)
    if thread_id:
        full = bq.get_dm_thread(user_id=user_id, app=target_app,
                                 thread_id=thread_id, since_timestamp=t_test, limit=80)
        if full:
            sections.append(_section_header(f"Group DM thread {thread_id} on {target_app}"))
            sections.append(_format_dm_thread(full, max_msgs=40))
            # Also dump per-participant friend records so the model can label each.
            parts = full.get("participants") or []
            friend_records = [bq.get_friend(user_id, p) for p in parts if p and p != "self"]
            friend_records = [fr for fr in friend_records if fr]
            if friend_records:
                sections.append(_section_header("Participants (from friends graph)"))
                for fr in friend_records:
                    sections.append(_format_friend(fr) + "\n")
    return _wrap_block(sections)


def _t17_wrong_recipient(bq: BackendQuery, user_id: str, t_test: int, inst: dict) -> str:
    target_app = inst.get("target_app") or "instagram"
    recipient_name = (inst.get("recipient_name") or "").strip()
    if not recipient_name:
        return ""
    profile = bq.get_full_profile(user_id)
    friends = profile.get("friends") or []
    matches = [fr for fr in friends
               if (fr.get("display_name") or "").split()[0].lower() == recipient_name.lower()]
    sections: list[str] = []
    sections.append(_section_header(f"Friends matching first name \"{recipient_name}\""))
    if not matches:
        sections.append(f"_(no friends with first name \"{recipient_name}\" — should refuse or ask)_\n")
        return _wrap_block(sections)
    for fr in matches:
        sections.append(_format_friend(fr) + "\n")
    # Recent DM threads on target_app with each candidate (helps disambiguation).
    sections.append(_section_header(f"Recent DM threads on {target_app} with candidates "
                                     f"(may indicate which one user normally talks to)"))
    page = bq.list_dm_threads(user_id=user_id, app=target_app,
                               since_timestamp=t_test, limit=20)
    cand_ids = {fr.get("friend_id") for fr in matches}
    rendered = 0
    for t in page.get("results") or []:
        parts = set(t.get("participants") or [])
        if parts & cand_ids:
            full = bq.get_dm_thread(user_id=user_id, app=target_app,
                                     thread_id=t["thread_id"],
                                     since_timestamp=t_test, limit=8)
            if full:
                sections.append(_format_dm_thread(full, max_msgs=4))
                rendered += 1
                if rendered >= 4:
                    break
    if rendered == 0:
        sections.append("_(no recent DM threads with these candidates on this app)_\n")
    return _wrap_block(sections)


def _t18_proactive_daily(bq: BackendQuery, user_id: str, t_test: int, inst: dict) -> str:
    cutoff = t_test - _DAY
    sections: list[str] = []
    sections.append(_section_header(f"Activity in past 24h (T={_fmt_ts(t_test)})"))
    any_events = False
    for app in APPS:
        evs = [e for e in bq.get_events(user_id=user_id, app=app, since_timestamp=t_test)
               if (e.get("source_timestamp") or 0) >= cutoff]
        if not evs:
            continue
        any_events = True
        sections.append(f"\n#### {app} ({len(evs)} events)\n")
        for e in evs[-10:]:
            ts = _fmt_ts(e.get("source_timestamp"))
            content = e.get("content") or {}
            cap = _truncate(content.get("caption") or content.get("title") or "", 220)
            it = e.get("source_interaction_type") or "?"
            action = ((e.get("interaction_format") or {}).get("action_label")) or it
            tags = " ".join(e.get("source_hashtags") or [])
            sections.append(f"- [{ts}] _{action}_: \"{cap}\" {tags}".rstrip() + "\n")
    if not any_events:
        sections.append("_(no events in past 24h on any app)_\n")
    # Top hashtags (lifetime) so model knows which interests dominate.
    summary = bq.hashtag_summary(user_id=user_id, since_timestamp=t_test)[:10]
    if summary:
        sections.append(_section_header("Top hashtags (lifetime, by total engagement)"))
        for row in summary:
            sections.append(f"- {row.get('hashtag','?')}: "
                             f"+{row.get('positive', 0)} / -{row.get('negative', 0)}\n")
    return _wrap_block(sections)


def _t19_trending_alert(bq: BackendQuery, user_id: str, t_test: int, inst: dict) -> str:
    sections: list[str] = []
    trending = bq.get_trending(user_id) or []
    sections.append(_section_header("Trending hashtags right now"))
    if not trending:
        sections.append("_(no trending data available)_\n")
    else:
        for h in trending[:15]:
            tag = h.get("hashtag", "?")
            rank = h.get("rank", "?")
            aligned = h.get("user_aligned")
            note = " (aligned with this user's interests)" if aligned else ""
            sections.append(f"- #{rank}: {tag}{note}\n")
    summary = bq.hashtag_summary(user_id=user_id, since_timestamp=t_test)
    pos = [r for r in summary if (r.get("positive") or 0) > 0][:10]
    neg = [r for r in summary if (r.get("negative") or 0) > 0][:10]
    if pos:
        sections.append(_section_header("User's top positive hashtags"))
        for row in pos:
            sections.append(f"- {row['hashtag']}: +{row['positive']}\n")
    if neg:
        sections.append(_section_header("User's explicit-negative hashtags (do NOT alert on these)"))
        for row in neg:
            sections.append(f"- {row['hashtag']}: -{row['negative']}\n")
    return _wrap_block(sections)


_DISPATCH = {
    "agentic_user_tone_post":            _t6_user_tone_post,
    "agentic_dm_digest":                 _t8_dm_digest,
    "agentic_cross_app_repost":          _t9_cross_app_repost,
    "agentic_auto_reply":                _t10_auto_reply,
    "agentic_vague_refind":              _t11_vague_refind,
    "agentic_composed_post":             _t12_composed_post,
    "agentic_send_post":                 _t13_send_post,
    "agentic_group_dm_summary":          _t16_group_dm_summary,
    "agentic_wrong_recipient_check":     _t17_wrong_recipient,
    "agentic_proactive_daily_catchup":   _t18_proactive_daily,
    "agentic_trending_alert":            _t19_trending_alert,
}
