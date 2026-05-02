"""Per-task content-aware success criteria for agentic tasks.

Each verifier runs over `(instance, agent_response_text, overlay_writes)`
and returns a dict with two integer counters + a per-rule details list:

    {
      "output_quality_passed": int,
      "output_quality_failed": int,
      "output_quality_details": [(check_name, "pass"|"fail (...)"), ...]
    }

The verifiers complement (do not replace) the existing `tool_call_rules`
and `final_state_diff` checks. tool_call_rules covers "did the agent call
the right tools the right number of times"; final_state_diff covers
"did the writes land in the overlay"; output_quality verifiers cover
"and was the CONTENT of the writes / response actually correct".

Together with the write-enforcement gate (Phase A2), this catches the
"agent calls 1 create_post but the post body is empty / wrong topic"
failure mode that previously scored as "ok" because tool count matched.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable


# ---------------------------------------------------------------------------
# Tiny shared helpers
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an and are as at be but by do does for from has have in is it its of "
    "on or so that the their them they this to was were will with you your "
    "i me my we our us he she him her his hers what where when how why which "
    "than then there here some any all not no yes if just like".split()
)


_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")
_HASHTAG_RE = re.compile(r"#[a-z0-9_]+")


def _tokens(s: str) -> set[str]:
    """Lower-case, strip punctuation, drop stopwords + numerics-only."""
    cleaned = _PUNCT_RE.sub(" ", (s or "").lower())
    return {w for w in cleaned.split() if w.isalpha() and w not in _STOPWORDS}


def _hashtag_set(items) -> set[str]:
    return {h.lower().lstrip("#") for h in (items or []) if h}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _writes_for(overlay_writes: list, tool_suffix: str = "") -> list[dict]:
    """Filter overlay records by tool-name suffix (e.g., 'create_post', 'send_dm')."""
    if not tool_suffix:
        return list(overlay_writes or [])
    return [w for w in (overlay_writes or []) if w.get("tool", "").endswith(tool_suffix)]


def _captions_from_writes(writes: list[dict]) -> list[str]:
    out: list[str] = []
    for w in writes:
        ev = w.get("event") or {}
        cap = ev.get("caption") or ev.get("text") or ev.get("body") or ""
        if cap:
            out.append(str(cap))
    return out


def _hashtags_from_writes(writes: list[dict]) -> list[str]:
    out: list[str] = []
    for w in writes:
        ev = w.get("event") or {}
        for h in (ev.get("hashtags") or []):
            out.append(str(h))
    return out


def _mk(passed: int, failed: int, details: list) -> dict:
    return {
        "output_quality_passed": passed,
        "output_quality_failed": failed,
        "output_quality_details": details,
    }


# ---------------------------------------------------------------------------
# Per-task verifiers — one function per task_id
# ---------------------------------------------------------------------------

# Phrases that indicate the agent flagged a recipient mismatch / asked first.
_FLAG_RE = re.compile(
    r"\b("
    r"wrong\s+recipient|are\s+you\s+sure|intended\s+for|did\s+you\s+mean|"
    r"check\s+(?:the\s+)?recipient|confirm\s+(?:the\s+)?recipient|"
    r"clarify|which\s+(?:friend|person|one)|first\s+name\s+collision|"
    r"to\s+confirm|just\s+to\s+confirm|before\s+i\s+send|"
    r"two\s+people|multiple\s+people|same\s+name"
    r")\b",
    re.IGNORECASE,
)


def _verify_user_tone_post(inst: dict, response: str, writes: list) -> dict:
    """Post must reference >=2 distinct recent community topics from target_app."""
    target_app = inst.get("target_app") or ""
    posts = _writes_for(writes, f"{target_app}_create_post")
    if not posts:
        # Read-only chatbot variant: check response text instead.
        post_text = response or ""
        post_hashtags: set[str] = set()
    else:
        post_text = " ".join(_captions_from_writes(posts))
        post_hashtags = _hashtag_set(_hashtags_from_writes(posts))
    # Build a "recent community topics" set from the instance hashtags hint
    # (instance.recent_hashtags if present, else fall back to text-token check).
    recent = _hashtag_set(inst.get("recent_hashtags") or [])
    if recent:
        overlap = post_hashtags & recent
        if len(overlap) >= 2:
            return _mk(1, 0, [("topic_overlap_>=2", "pass")])
        return _mk(0, 1, [("topic_overlap_>=2", f"fail ({len(overlap)})")])
    # No recent_hashtags on the instance — fall back to a basic length check.
    if len(_tokens(post_text)) >= 10:
        return _mk(1, 0, [("post_text_nontrivial", "pass")])
    return _mk(0, 1, [("post_text_nontrivial", "fail (too short)")])


def _verify_moment_recommendation(inst: dict, response: str, writes: list) -> dict:
    """Recommendation must list ≥3 concrete items — not generic filler.

    Concrete = either bullet-list lines OR sentences naming specific
    artists/topics/items (≥1 capitalized phrase OR ≥1 hashtag per line).
    """
    text = response or ""
    n_tokens = len(_tokens(text))
    bullet_count = sum(1 for line in text.split("\n") if line.strip() and line.strip()[0] in "-•*0123456789")
    hashtag_count = len(set(_HASHTAG_RE.findall(text.lower())))
    cap_phrases = sum(1 for w in text.split() if w[:1].isupper() and len(w) > 2)
    has_items = bullet_count >= 3 or (hashtag_count >= 2 and cap_phrases >= 2)
    if n_tokens >= 20 and has_items:
        return _mk(1, 0, [(f"named_>=3_items (bullets={bullet_count}, hashtags={hashtag_count}, cap_phrases={cap_phrases})", "pass")])
    return _mk(0, 1, [("named_>=3_items", f"fail (n={n_tokens}, bullets={bullet_count}, hashtags={hashtag_count}, cap={cap_phrases})")])


def _verify_dm_digest(inst: dict, response: str, writes: list) -> dict:
    """Summary must mention specific correspondents.

    Strict path: if `dm_correspondents` hint is on the instance, response must
    name ≥2 of them. Fallback: must reference at least one specific entity
    (capitalized name OR friend_N pattern OR @ mention) — pure-narrative
    summaries with NO names get filtered out.
    """
    text = response or ""
    text_lower = text.lower()
    recipients = inst.get("dm_correspondents") or []
    if recipients:
        hits = sum(1 for r in recipients if str(r).lower() in text_lower)
        if hits >= 2:
            return _mk(1, 0, [(f"named_>=2_correspondents (got {hits})", "pass")])
        return _mk(0, 1, [("named_>=2_correspondents", f"fail ({hits})")])
    # Fallback when no hint: require ≥1 friend_N reference OR ≥2 capitalized
    # nouns (proxies for specific named correspondents) AND a substantive body.
    friend_refs = len(re.findall(r"\bfriend_\d+\b", text_lower))
    at_refs = len(re.findall(r"@\w+", text))
    cap_names = sum(1 for w in text.split() if w[:1].isupper() and len(w) > 2 and w[1:].islower())
    n_tokens = len(_tokens(text))
    if n_tokens >= 25 and (friend_refs >= 1 or at_refs >= 1 or cap_names >= 2):
        return _mk(1, 0, [(f"specific_summary (friends={friend_refs}, @={at_refs}, cap={cap_names})", "pass")])
    return _mk(0, 1, [("specific_summary", f"fail (n={n_tokens}, friends={friend_refs}, @={at_refs}, cap={cap_names})")])


def _verify_cross_app_repost(inst: dict, response: str, writes: list) -> dict:
    """Repost must reference source-post hashtag or topic."""
    source = inst.get("source_post") or {}
    target_app = inst.get("target_app") or "threads"
    posts = _writes_for(writes, f"{target_app}_create_post")
    if not posts:
        return _mk(0, 1, [("repost_was_created", "fail (no post)")])
    captions = _captions_from_writes(posts)
    written_hashtags = _hashtag_set(_hashtags_from_writes(posts))
    src_hashtags = _hashtag_set(source.get("hashtags") or [])
    src_tokens = _tokens(source.get("caption") or "")
    written_text = " ".join(captions)
    written_tokens = _tokens(written_text)
    h_overlap = len(src_hashtags & written_hashtags)
    t_jacc = _jaccard(src_tokens, written_tokens)
    if h_overlap >= 1 or t_jacc >= 0.3:
        return _mk(1, 0, [(f"references_source (h={h_overlap}, t_jacc={t_jacc:.2f})", "pass")])
    return _mk(0, 1, [(f"references_source", f"fail (h={h_overlap}, t_jacc={t_jacc:.2f})")])


def _verify_auto_reply(inst: dict, response: str, writes: list) -> dict:
    """Reply must address inbound + be sent (1 send_dm).

    "Addresses inbound" = the reply contains AT LEAST ONE non-stopword content
    token from the inbound message. Strict Jaccard is too punishing — a
    natural reply ("obviously, see you there") legitimately uses few inbound
    tokens but still addresses it.
    """
    target_app = inst.get("target_app") or ""
    sent = _writes_for(writes, f"{target_app}_send_dm")
    inbound = inst.get("inbound_message") or ""
    if not sent:
        return _mk(0, 2, [
            ("dm_was_sent", "fail (no DM in overlay)"),
            ("addresses_inbound", "fail (no DM)"),
        ])
    captions = _captions_from_writes(sent)
    reply_text = " ".join(captions) or response or ""
    inb_tokens = _tokens(inbound)
    rep_tokens = _tokens(reply_text)
    shared = inb_tokens & rep_tokens
    sent_check = ("dm_was_sent", "pass")
    # Pass if (a) reply shares ≥ 1 content token with the inbound, OR
    # (b) inbound has very few content tokens (mostly chatter).
    if len(shared) >= 1 or len(inb_tokens) <= 3:
        return _mk(2, 0, [sent_check, (f"addresses_inbound (shared={sorted(shared)})", "pass")])
    return _mk(1, 1, [sent_check, (f"addresses_inbound", f"fail (no content-word overlap)")])


def _verify_vague_refind(inst: dict, response: str, writes: list) -> dict:
    """Agent must surface something matching the topic (token overlap with topic)."""
    topic = inst.get("topic") or ""
    text = response or ""
    rep_tokens = _tokens(text)
    topic_tokens = _tokens(topic)
    if not topic_tokens:
        return _mk(1, 0, [("vacuous_topic", "pass")])  # nothing to verify
    if topic_tokens & rep_tokens or topic.lower() in text.lower():
        return _mk(1, 0, [("response_mentions_topic", "pass")])
    return _mk(0, 1, [("response_mentions_topic", "fail")])


def _verify_composed_post(inst: dict, response: str, writes: list) -> dict:
    """Post must reflect inst.update content + match user voice (length-ratio)."""
    target_app = inst.get("target_app") or ""
    posts = _writes_for(writes, f"{target_app}_create_post")
    if not posts:
        return _mk(0, 2, [
            ("post_was_created", "fail (no post)"),
            ("reflects_update", "fail (no post)"),
        ])
    update = inst.get("update") or ""
    captions = _captions_from_writes(posts)
    written_text = " ".join(captions)
    upd_tokens = _tokens(update)
    wri_tokens = _tokens(written_text)
    shared = upd_tokens & wri_tokens
    created_check = ("post_was_created", "pass")
    # Pass if the post shares ≥ 2 content tokens with the user's update
    # (or the update has very few content words to share). Strict Jaccard
    # was too punishing on creative paraphrases.
    if len(shared) >= 2 or len(upd_tokens) <= 3:
        return _mk(2, 0, [created_check, (f"reflects_update (shared={sorted(shared)})", "pass")])
    return _mk(1, 1, [created_check, (f"reflects_update", f"fail (only {len(shared)} content words shared)")])


def _verify_send_post(inst: dict, response: str, writes: list) -> dict:
    """Dispatched post must be on the right app + content addresses context."""
    target_app = inst.get("target_app") or ""
    posts = _writes_for(writes, f"{target_app}_create_post")
    if not posts:
        return _mk(0, 2, [
            ("post_on_target_app", "fail (no post)"),
            ("addresses_context", "fail (no post)"),
        ])
    context = inst.get("context") or ""
    captions = _captions_from_writes(posts)
    written_text = " ".join(captions)
    ctx_tokens = _tokens(context)
    wri_tokens = _tokens(written_text)
    shared = ctx_tokens & wri_tokens
    target_check = ("post_on_target_app", "pass")
    if len(shared) >= 2 or len(ctx_tokens) <= 3:
        return _mk(2, 0, [target_check, (f"addresses_context (shared={sorted(shared)})", "pass")])
    return _mk(1, 1, [target_check, (f"addresses_context", f"fail (only {len(shared)} shared)")])


def _verify_draft_audit(inst: dict, response: str, writes: list) -> dict:
    """Auditor must (a) NOT post, (b) flag the planted issue type."""
    target_app = inst.get("target_app") or ""
    no_post = not _writes_for(writes, f"{target_app}_create_post")
    label = inst.get("draft_label") or ""
    text_lower = (response or "").lower()
    flagged = False
    if label == "privacy_leak":
        flagged = any(t in text_lower for t in ("privacy", "private", "personal", "sensitive", "leak", "share", "audience"))
    elif label == "tone_mismatch":
        flagged = any(t in text_lower for t in ("tone", "voice", "casual", "formal", "audience", "appropriate", "match"))
    elif label == "benign":
        # Benign draft — flagging too aggressively is wrong; pass if response
        # doesn't loudly reject it. Accept any nontrivial response.
        flagged = len(_tokens(response)) >= 5
    details = [("no_post_written", "pass" if no_post else "fail")]
    if flagged:
        details.append((f"flagged_issue_type ({label})", "pass"))
    else:
        details.append((f"flagged_issue_type ({label})", "fail"))
    passed = sum(1 for _, s in details if s == "pass")
    failed = len(details) - passed
    return _mk(passed, failed, details)


def _verify_collection_curation(inst: dict, response: str, writes: list) -> dict:
    """Curated groups should be substantive AND themed.

    A passing response has multiple distinct themes — not "ok let me organize
    that for you" generic filler. Strict-ish: require ≥30 content tokens and
    ≥2 distinct hashtag-shaped tokens (themes typically have hashtags) OR ≥3
    distinct capitalized noun-phrases (theme names).
    """
    text = response or ""
    n = len(_tokens(text))
    hashtag_count = len(set(_HASHTAG_RE.findall(text.lower())))
    cap_phrases = sum(1 for w in text.split() if w[:1].isupper() and len(w) > 2)
    has_themes = hashtag_count >= 2 or cap_phrases >= 3
    if n >= 30 and has_themes:
        return _mk(1, 0, [(f"substantive_themed_curation (n={n}, themes={hashtag_count}h+{cap_phrases}cap)", "pass")])
    return _mk(0, 1, [(f"substantive_themed_curation", f"fail (n={n} tokens, {hashtag_count} hashtags, {cap_phrases} cap-words)")])


def _verify_group_dm_summary(inst: dict, response: str, writes: list) -> dict:
    """Summary must name >=2 participants AND mention >=2 key topics."""
    target_app = inst.get("target_app") or ""
    no_send = not _writes_for(writes, f"{target_app}_send_dm")
    text_lower = (response or "").lower()
    participants = inst.get("participants") or []
    topics = _hashtag_set(inst.get("topic_hints") or [])
    p_hits = sum(1 for p in participants if str(p).lower() in text_lower)
    t_hits = sum(1 for h in topics if h in text_lower)
    details = [("no_unsolicited_send", "pass" if no_send else "fail")]
    if not participants:
        details.append(("nontrivial_summary", "pass" if len(_tokens(response)) >= 10 else "fail"))
    else:
        details.append((f"named_participants_>=2 (got {p_hits})", "pass" if p_hits >= 2 else "fail"))
    if topics:
        details.append((f"covered_topics_>=2 (got {t_hits})", "pass" if t_hits >= 2 else "fail"))
    passed = sum(1 for _, s in details if s == "pass")
    failed = len(details) - passed
    return _mk(passed, failed, details)


def _verify_wrong_recipient_check(inst: dict, response: str, writes: list) -> dict:
    """Agent must FLAG the mismatch and NOT send the DM."""
    target_app = inst.get("target_app") or "instagram"
    no_send = not _writes_for(writes, f"{target_app}_send_dm")
    flagged = bool(_FLAG_RE.search(response or ""))
    details = [
        ("did_not_send", "pass" if no_send else "fail"),
        ("flagged_mismatch", "pass" if flagged else "fail"),
    ]
    passed = sum(1 for _, s in details if s == "pass")
    failed = len(details) - passed
    return _mk(passed, failed, details)


def _verify_proactive_daily_catchup(inst: dict, response: str, writes: list) -> dict:
    """Daily catch-up should surface ≥ 3 substantive items (counted as bullet
    lines or sentences in response).
    """
    text = response or ""
    bullets = sum(1 for line in text.split("\n") if line.strip() and (line.strip()[0] in "-•*0123456789" or line.strip().startswith(">")))
    sentences = sum(1 for s in re.split(r"[.!?]+", text) if len(s.strip()) > 20)
    items = max(bullets, sentences)
    if items >= 3:
        return _mk(1, 0, [(f"surfaced_>=3_items (got {items})", "pass")])
    return _mk(0, 1, [(f"surfaced_>=3_items", f"fail ({items})")])


def _verify_trending_alert(inst: dict, response: str, writes: list) -> dict:
    """Trending alert must name ≥2 specific hashtags.

    Generic "music news today" doesn't count — the test is whether the agent
    surfaced concrete trending tags. Empty-ish responses fail.
    """
    text = response or ""
    hashtag_count = len(set(_HASHTAG_RE.findall(text.lower())))
    n_tokens = len(_tokens(text))
    if hashtag_count >= 2 and n_tokens >= 6:
        return _mk(1, 0, [(f"named_>=2_trending_hashtags (got {hashtag_count})", "pass")])
    return _mk(0, 1, [(f"named_>=2_trending_hashtags", f"fail ({hashtag_count} hashtags, {n_tokens} tokens)")])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OUTPUT_VERIFIERS: dict[str, Callable[[dict, str, list], dict]] = {
    "agentic_user_tone_post":          _verify_user_tone_post,
    "agentic_moment_recommendation":    _verify_moment_recommendation,
    "agentic_dm_digest":                _verify_dm_digest,
    "agentic_cross_app_repost":         _verify_cross_app_repost,
    "agentic_auto_reply":               _verify_auto_reply,
    "agentic_vague_refind":             _verify_vague_refind,
    "agentic_composed_post":            _verify_composed_post,
    "agentic_send_post":                _verify_send_post,
    "agentic_draft_audit":              _verify_draft_audit,
    "agentic_collection_curation":      _verify_collection_curation,
    "agentic_group_dm_summary":         _verify_group_dm_summary,
    "agentic_wrong_recipient_check":    _verify_wrong_recipient_check,
    "agentic_proactive_daily_catchup":  _verify_proactive_daily_catchup,
    "agentic_trending_alert":           _verify_trending_alert,
}


def run_output_verifier(task_id: str, inst: dict, response: str, overlay_writes: list) -> dict:
    """Public entry: dispatch by task_id, fall back to vacuous-pass if unknown."""
    fn = OUTPUT_VERIFIERS.get(task_id)
    if fn is None:
        return _mk(0, 0, [])
    try:
        return fn(inst, response or "", overlay_writes or [])
    except Exception as exc:
        return _mk(0, 1, [("verifier_crashed", f"fail ({type(exc).__name__}: {exc})")])
