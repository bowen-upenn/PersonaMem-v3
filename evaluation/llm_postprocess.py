"""Workstreams I + J — post-build LLM passes.

Run after the benchmark dict is assembled in build_benchmark, BEFORE
queries.csv is written. Two passes:

I.  Self-consistency check on every saved EXAMPLE_RESPONSE — using
    gpt-5.4-mini, score the example response against the universal
    `avoid_overpersonalization` rubric. The gold reference shouldn't
    itself trip the rubric; if it does, log + flag (soft default).

J.  Paired INFERIOR_RESPONSE per personalization instance — generate
    a structurally-identical foil that introduces ONE subtle flaw
    (incorrect_personalization / disliked_recent / over_personalization).
    Substitution evidence is sourced from the user's real data — no
    hallucinated friend names or topic lists.

Both passes attach result fields onto the instance dicts in-place.
The dump_test_samples_json + audit pick them up automatically via
`instance_full`.

Cost: ~1 LLM call per personalization instance for self-check, ~1
for inferior generation. ≈ 200-400 calls per user. Caller decides
via `--skip_postprocess` / `--enable_inferior` flags.
"""

from __future__ import annotations

import json
import re
import random
from typing import Callable

from data_preparation.utils import extract_json_from_response


# Personalization-task instances are the ones that need both passes.
# (Restraint-arm instances also get self-checked — the rubric should
# fire on them only if they fail to restrain.)
_PERSONALIZATION_TASKS = {
    "chatbot_proactive_personalization",
    "over_personalization_chatbot_text",
    "over_personalization_distractor_reject",
    "over_personalization_sensitive_event",
    "over_personalization_context_shift",
    "preference_removal_regen",
    "personalized_feed_ranking",
    "personalized_recommendation",
    "at_ai_directive_followup",
    "daily_personalized_briefing",
    "short_vs_long_term_lifecycle",
    "active_mistake_prevention",
    "repetition_fatigue_pairs",
    "repetition_fatigue_sequences",
    "agentic_user_tone_post",
    # agentic_moment_recommendation merged into personalized_recommendation
    "agentic_dm_digest",
    "agentic_cross_app_repost",
    "agentic_auto_reply",
    "agentic_vague_refind",
    "agentic_composed_post",
    "agentic_send_post",
    "agentic_group_dm_summary",
    "agentic_wrong_recipient_check",
    "agentic_proactive_daily_catchup",
    "agentic_trending_alert",
}


# ---------------------------------------------------------------------------
# NEW pass — LLM-generate concrete example_response for tasks where the
# extractor returned meta-instruction text. The extractor in visualize.py
# emits things like "A natural conversational answer to the user's question
# that implicitly weaves in the held-out preference where it fits." — those
# are rubric guidance, not gold answers. This pass replaces them with
# concrete text grounded in the user's data.
#
# Ranking tasks (slate, recommendation, at_ai_directive, lifecycle) are
# handled deterministically (compute the actual ranked index list) — no
# LLM needed for those.
# ---------------------------------------------------------------------------

# Tasks whose example_response the extractor emits as concrete text
# (already real). These get SKIPPED by the LLM-gen pass.
_TASKS_ALREADY_CONCRETE: set[str] = set()
# Previously contained agentic_vague_refind / agentic_wrong_recipient_check.
# Both moved out: the templated stubs at visualize.py:747-770 emitted
# `(source_object_id=…)` / `[Name A] / [Name B]` placeholders that broke
# the eval. They now route through the LLM-gen path with task-specific
# grounding (`_task_grounding`) so the gold names a real post or real
# friend pair from the user's backend.

# Ranking tasks — example_response is a deterministic ranked index list,
# computed without LLM.
_RANKING_TASKS = {
    "personalized_feed_ranking",
    "personalized_recommendation",
    "at_ai_directive_followup",
    "short_vs_long_term_lifecycle",
}

# Deterministic-gold tasks whose output is a structured JSON list of real
# backend posts (NOT a `Ranked indexes: [...]` wrapper). Same dispatch
# pattern as `_RANKING_TASKS` — no LLM gold-gen, no LLM foil-gen, no
# self-check pass — but the gold/foil are computed by task-specific
# builders rather than the index-list helpers. Currently empty —
# agentic_moment_recommendation was the only entry and it merged into
# personalized_recommendation (which uses the index-list ranking helpers).
# Kept as an extension point for future deterministic JSON tasks.
_DETERMINISTIC_GOLD_TASKS: set[str] = set()

# Tasks where the gold is too short / structural for any kind of paired
# foil to make sense. Note: ranking tasks are NOT in this set anymore —
# they get a deterministic inferior via `_compute_ranking_inferior` (a
# different index order in the same `Ranked indexes: [...]` wrapper).
_TASKS_NO_FOIL = {
    "agentic_wrong_recipient_check",
    "agentic_vague_refind",
}


_EXAMPLE_GEN_PROMPT = """You are answering the user's request below. Reply naturally with the actual response — the words you would say.
{length_guidance}
{grounding_block}
User query:
\"\"\"{query}\"\"\"

Style rules — IMPORTANT:
  - Do NOT telegraph that you know the user. Stock phrases that "advertise"
    personalization are forbidden:
      • "as a fan of X"
      • "since you love X" / "since you like X"
      • "I know you're into X" / "I know you love X"
      • "given your interest in X"
      • "knowing how much you X"
      • "as someone who loves X"
      • "you'll appreciate this because X"
    The grounding context is for choosing WHAT to mention (which topic,
    which friend, which post), NOT for advertising that you have a profile.
    Just answer the query naturally — the personalization should be
    visible in the *content choice*, not in self-reference about the user.

Output ONE fenced ```json block:
```json
{{"text": "<the actual response>"}}
```"""


# Telegraph phrases the example_response must avoid — case-insensitive.
# When the LLM emits one of these the example is regenerated once; on a
# second hit we accept it (graceful degrade).
_TELEGRAPH_PHRASE_RE = re.compile(
    r"(as a fan of|since you (love|like|enjoy|prefer|are into)|"
    r"i know you('?re| are) (into|a fan of)|"
    r"i know you (love|like|enjoy)|"
    r"given your (interest|love|passion) (in|for)|"
    r"knowing how much you|"
    r"as someone who (loves|likes|enjoys|is into|is a fan of)|"
    r"you'?ll appreciate this because)",
    re.IGNORECASE,
)


_COMPOSE_TASKS = {
    "agentic_composed_post",
    "agentic_send_post",
    "agentic_cross_app_repost",
    "agentic_auto_reply",
}

# Tasks where the voice-evidence smoke test runs (overlap with _COMPOSE_TASKS
# today; kept separate so the set can grow without changing length-band logic).
_VOICE_EVIDENCE_TASKS = set(_COMPOSE_TASKS)


def _extract_voice_evidence_spans(text: str, user_voice: dict) -> list[str]:
    """Heuristically extract the substrings of `text` that carry the user's
    voice signal — personal_phrases (case-insensitive substring match) and
    palette emoji (exact char match). Used to bold the spans in the rendered
    Example Response so a reviewer can see WHY a voice_mismatch foil fails.

    Returns the matched substrings preserving the original casing as they
    appear in `text`. Longest matches first so renderer can substitute
    without nested-match collisions. Empty list when nothing matches.
    """
    if not isinstance(text, str) or not text:
        return []
    if not isinstance(user_voice, dict):
        return []
    spans: list[str] = []
    seen: set[str] = set()
    text_lower = text.lower()

    # 1. Personal phrases — case-insensitive substring lookup, preserve
    #    original casing in the matched span (use the slice of `text`,
    #    not the catalog entry).
    for phrase in (user_voice.get("personal_phrases") or []):
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        needle = phrase.lower()
        idx = text_lower.find(needle)
        while idx != -1:
            span = text[idx:idx + len(needle)]
            if span and span.lower() not in seen:
                spans.append(span)
                seen.add(span.lower())
            idx = text_lower.find(needle, idx + len(needle))

    # 2. Palette emoji — exact char match. Each match becomes its own span
    #    so the renderer can bold every occurrence.
    for emoji in (user_voice.get("emoji_palette") or []):
        if not isinstance(emoji, str) or not emoji:
            continue
        if emoji in text and emoji not in seen:
            spans.append(emoji)
            seen.add(emoji)

    spans.sort(key=lambda s: -len(s))
    return spans


_VOICE_EVIDENCE_VERIFY_PROMPT = """You are a quality-control reviewer for a benchmark of personalized AI responses. Two responses were written for the same user query. ONE was crafted to honor the user's writing voice (the "good" example); the OTHER intentionally drops voice anchors to model a `voice_mismatch` failure (the "bad" inferior).

Below is the user's writing voice signal. The bolded items in the gold response are the specific voice anchors that matter (personal phrases, palette emoji).

## User's writing voice signal
{voice_signal}

## Bolded voice anchors in the gold response
{bolded_anchors}

## Response A
{response_a}

## Response B
{response_b}

## Your task
Decide which response (A or B) better honors the user's writing voice. The "better" response is the one that uses the bolded anchors / voice signal naturally; the "worse" one drops them.

Respond with EXACTLY this JSON, no commentary:
```json
{{"better": "A" | "B", "confidence": 0.0-1.0, "reason": "1 short sentence"}}
```"""


def _verify_voice_evidence_distinguishability(
    voice_check_llm: Callable[[str], str],
    example: str,
    inferior: str,
    user_voice: dict,
    evidence_spans: list[str],
    rng: random.Random,
) -> dict:
    """Smoke-test that a mini-tier LLM, given the bolded voice signal,
    can correctly pick the gold over the foil. If it can't (or picks the
    foil), the example/inferior pair is too similar on the voice axis and
    should be regenerated.

    Returns {"passed": bool, "picked": "A"|"B"|None, "expected": "A"|"B",
             "reason": str, "raw": str}.
    "passed" is True iff the mini correctly picks the gold AND confidence
    is above a low-bar threshold (>= 0.55).
    """
    if not voice_check_llm or not example or not inferior:
        return {"passed": True, "picked": None, "expected": None,
                "reason": "skipped (missing inputs)", "raw": ""}
    if not evidence_spans:
        # No bolded anchors to verify against — skip rather than fail
        # (some compose-task golds legitimately don't surface voice
        # phrases / palette emoji).
        return {"passed": True, "picked": None, "expected": None,
                "reason": "skipped (no evidence spans)", "raw": ""}

    voice_lines: list[str] = []
    if isinstance(user_voice, dict):
        if user_voice.get("natural_register"):
            voice_lines.append(f"- register: {user_voice['natural_register']}")
        if user_voice.get("default_capitalization"):
            voice_lines.append(f"- capitalization: {user_voice['default_capitalization']}")
        phrases = user_voice.get("personal_phrases") or []
        if phrases:
            voice_lines.append(f"- personal phrases: {', '.join(phrases[:6])}")
        palette = user_voice.get("emoji_palette") or []
        if palette:
            voice_lines.append(f"- emoji palette: {' '.join(palette)}")
        if user_voice.get("voice_avoid"):
            voice_lines.append(f"- voice avoid: {user_voice['voice_avoid']}")
    voice_signal = "\n".join(voice_lines) or "(no voice signal available)"
    bolded_anchors = "\n".join(f"- **{s}**" for s in evidence_spans[:8])

    # Randomize A/B assignment so the LLM can't lock onto a positional bias.
    expected = rng.choice(["A", "B"])
    if expected == "A":
        response_a, response_b = example, inferior
    else:
        response_a, response_b = inferior, example

    prompt = _VOICE_EVIDENCE_VERIFY_PROMPT.format(
        voice_signal=voice_signal,
        bolded_anchors=bolded_anchors,
        response_a=response_a,
        response_b=response_b,
    )
    raw = ""
    try:
        raw = voice_check_llm(prompt) or ""
    except Exception as exc:
        return {"passed": True, "picked": None, "expected": expected,
                "reason": f"skipped (verifier raised: {exc})", "raw": ""}
    parsed = extract_json_from_response(raw) or {}
    picked = (parsed.get("better") or "").strip().upper()
    confidence = parsed.get("confidence") or 0.0
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    reason = (parsed.get("reason") or "").strip()
    if picked not in ("A", "B"):
        return {"passed": False, "picked": None, "expected": expected,
                "reason": "verifier returned no clear pick", "raw": raw[:300]}
    passed = (picked == expected) and (confidence >= 0.55)
    return {"passed": passed, "picked": picked, "expected": expected,
            "confidence": confidence, "reason": reason, "raw": raw[:300]}

# Per-app fallback bands matching the AppPersona.expression.length_band defaults
# in prompts.generate_app_personas_prompt. Used when the user-specific band is
# unavailable so compose-task golds still land on a real-post-length target.
_COMPOSE_DEFAULT_BANDS = {
    "instagram": (70, 150),
    "facebook":  (120, 220),
    "threads":   (45, 120),
    "chatbot":   (90, 190),
}


def _parse_length_band(raw: str | None) -> tuple[int, int] | None:
    """Parse 'lo-hi' / 'lo–hi' into (lo, hi) ints. Returns None if unparseable."""
    if not isinstance(raw, str):
        return None
    s = raw.strip().replace("–", "-").replace("—", "-")
    if "-" not in s:
        return None
    try:
        lo_str, hi_str = s.split("-", 1)
        lo = int(lo_str.strip())
        hi = int(hi_str.strip())
        if lo > 0 and hi >= lo:
            return (lo, hi)
    except (ValueError, AttributeError):
        return None
    return None


def _length_guidance(task_type: str, inst: dict | None = None,
                     app_persona: dict | None = None) -> str:
    if task_type == "chatbot_proactive_personalization":
        return "Length: 2–3 sentences."
    if task_type in ("over_personalization_chatbot_text",
                     "over_personalization_distractor_reject",
                     "over_personalization_sensitive_event"):
        return "Length: 1–3 sentences."
    if task_type == "daily_personalized_briefing":
        return "Length: 3–5 short bullet items."
    if task_type == "repetition_fatigue_pairs":
        return "Length: 2 short labelled lines."
    if task_type == "active_mistake_prevention":
        return "Length: 1–3 sentences."
    if task_type in _COMPOSE_TASKS:
        # Compose tasks emit real social-media posts (or DM auto-replies) —
        # caption-length, not chatbot one-liners. Use the user's per-app
        # length_band when available, falling back to per-app defaults.
        band: tuple[int, int] | None = None
        if isinstance(app_persona, dict):
            expr = app_persona.get("expression") or {}
            band = _parse_length_band(expr.get("length_band"))
        if band is None and isinstance(inst, dict):
            target_app = (inst.get("target_app") or "").lower()
            band = _COMPOSE_DEFAULT_BANDS.get(target_app)
        if band is None:
            band = (90, 200)  # generic fallback
        lo, hi = band
        return (
            f"Length: ~{lo}–{hi} characters (a real social-media post / "
            f"reply — full caption, not a one-liner). Use multiple short "
            f"sentences or a sentence + 1–3 hashtags as natural for this app."
        )
    return "Length: 1–4 sentences."


def _generate_example_response(llm: Callable[[str], str],
                               task_type: str, query: str,
                               grounding: str = "",
                               inst: dict | None = None,
                               app_persona: dict | None = None) -> str | None:
    if not llm or not query:
        return None
    grounding_block = (
        f"\nUse only the grounding facts below — do NOT invent posts, "
        f"friends, threads, or topics that aren't named here. Quote the "
        f"actual titles / IDs / names verbatim.\n"
        f"Grounding facts (real backend data):\n{grounding}\n"
        if grounding else ""
    )
    base_prompt = _EXAMPLE_GEN_PROMPT.format(
        query=query[:1500],
        length_guidance=_length_guidance(task_type, inst=inst, app_persona=app_persona),
        grounding_block=grounding_block,
    )
    text: str | None = None
    for attempt in range(2):
        prompt = base_prompt
        if attempt > 0 and text is not None:
            prompt = base_prompt + (
                "\n\nYour previous draft contained a telegraph phrase that "
                "advertises personalization (e.g., 'as a fan of', 'since you "
                "love', 'I know you're into'). Rewrite the response so the "
                "topic choice itself is the personalization signal — do not "
                "self-reference what you know about the user.\n"
                f"Previous draft (DO NOT REUSE): \"\"\"{text}\"\"\""
            )
        raw = llm(prompt)
        parsed = extract_json_from_response(raw) or {}
        candidate = parsed.get("text")
        if isinstance(candidate, str) and candidate.strip():
            text = candidate.strip()
            if not _TELEGRAPH_PHRASE_RE.search(text):
                return text
            # Telegraph phrase detected — retry once. If second attempt
            # also trips, accept it (graceful degrade).
        else:
            return None
    return text


def _format_iso_date(ts: int) -> str:
    """Compact YYYY-MM-DD from a unix timestamp; empty if unparseable."""
    if not ts:
        return ""
    import datetime as _dt
    try:
        return _dt.datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _task_grounding(inst: dict, task_id: str, bq, user_id: str) -> str:
    """Build a per-task grounding block for the gold-gen prompt.

    Tasks where the gold MUST reference real backend state (specific posts,
    friends, threads, topics) need concrete data so the LLM doesn't invent
    placeholders. Returns "" for tasks that don't need grounding (the
    clean-LLM-baseline default for personalization tasks where the rubric
    is the authority).
    """
    try:
        if task_id == "agentic_vague_refind":
            topic = (inst.get("topic") or "").strip()
            t_test = int(inst.get("t_test") or 0)
            if not topic or not t_test:
                return ""
            # Window: 3 days before t_test. The user vaguely remembers
            # "that post about X" — they mean something they actually saw
            # very recently, not from weeks ago.
            window_lo = t_test - 3 * 86400
            APP_DISPLAY = {
                "instagram": "Instagram",
                "facebook": "Facebook",
                "threads": "Threads",
                "chatbot": "the chatbot",
            }
            # Per-app loop so we know which app each event came from
            # (bq.get_events strips the app tag from the returned events).
            collected: list[tuple[int, str, dict]] = []
            for app in ("instagram", "facebook", "threads", "chatbot"):
                evs = bq.get_events(
                    user_id=user_id, app=app,
                    since_timestamp=t_test, hashtag=topic,
                ) or []
                for e in evs:
                    ts = int(e.get("source_timestamp") or 0)
                    if ts < window_lo:
                        continue
                    collected.append((ts, app, e))
            if not collected:
                return ""
            # 1-2 most recent on the topic, deduped by source_object_id.
            collected.sort(key=lambda x: x[0], reverse=True)
            seen_oids: set[str] = set()
            picks: list[tuple[int, str, dict]] = []
            for tup in collected:
                oid = str(tup[2].get("source_object_id") or "")
                if oid in seen_oids:
                    continue
                seen_oids.add(oid)
                picks.append(tup)
                if len(picks) >= 2:
                    break

            lines: list[str] = [
                f"User vaguely asked: \"find that post I saw about {topic}\".",
                "Real recent posts on this topic (within 3 days before now). "
                "Reference 1-2 of these in the gold response by NAMING the "
                "actual app and a short subject — never output a bracket "
                "ID like '[social 12345]', never output empty quotes \"\".",
                "",
            ]
            for ts, app, e in picks:
                content = e.get("content") or {}
                title = (content.get("title") or content.get("caption") or "").strip()
                hashtags = [h for h in (e.get("source_hashtags") or []) if h][:5]
                date = _format_iso_date(ts)
                # Build a short human-readable subject. Prefer real text;
                # fall back to a hashtag-derived noun phrase so the gold
                # never has to write `""`.
                subject = title[:120]
                if not subject and hashtags:
                    subject = "a post tagged " + ", ".join(
                        "#" + h.lstrip("#") for h in hashtags[:3]
                    )
                if not subject:
                    subject = f"a post about #{topic}"
                lines.append(
                    f"- On {APP_DISPLAY.get(app, app)} ({date}): {subject}"
                    + (
                        " — hashtags: " + " ".join(
                            "#" + h.lstrip("#") for h in hashtags
                        )
                        if hashtags else ""
                    )
                )
            return "\n".join(lines)

        if task_id == "agentic_wrong_recipient_check":
            collision_ids = list(inst.get("collision_friend_ids") or [])
            recipient_name = (inst.get("recipient_name") or "").strip()
            draft = (inst.get("draft") or "").strip()
            if not collision_ids:
                return ""
            prof = bq.get_full_profile(user_id) or {}
            friends_by_id = {f.get("friend_id"): f for f in (prof.get("friends") or [])}
            lines = [
                f"User typed a draft addressed to '{recipient_name}', but multiple "
                f"friends share that first name. Real candidates from profile.friends:"
            ]
            for fid in collision_ids:
                fr = friends_by_id.get(fid) or {}
                name = (fr.get("display_name") or fid).strip()
                rel = (fr.get("relationship_label") or fr.get("relationship") or "").strip()
                lines.append(
                    f"- {name} ({fid})"
                    + (f" — {rel}" if rel else "")
                )
            if draft:
                lines.append(f"Draft message: \"{draft[:140]}\"")
            return "\n".join(lines)

        if task_id == "agentic_proactive_daily_catchup":
            t_test = int(inst.get("t_test") or 0)
            if not t_test:
                return ""
            day_sec = 24 * 3600
            lines: list[str] = []
            # Unread / unanswered DMs across every social app + chatbot.
            for app in ("instagram", "facebook", "threads"):
                resp = bq.list_dm_threads(
                    user_id=user_id, app=app,
                    since_timestamp=t_test, limit=10,
                ) or {}
                threads = resp.get("results") or []
                for t in threads[:3]:
                    if (t.get("latest_ts") or 0) < (t_test - day_sec):
                        continue
                    snippet = (t.get("last_message_preview") or "").strip()
                    parts = ", ".join((t.get("participants") or [])[:3])
                    if snippet:
                        lines.append(
                            f"- DM on {app} (thread={t.get('thread_id')}, with {parts}): "
                            f"\"{snippet[:80]}\" ({_format_iso_date(t.get('latest_ts') or 0)})"
                        )
            # Recent feed engagements in the past 24h, across social apps.
            feed_lines: list[str] = []
            for app in ("instagram", "facebook", "threads"):
                events = bq.get_events(
                    user_id=user_id, app=app,
                    since_timestamp=t_test, limit=8,
                ) or []
                for e in events[-3:][::-1]:
                    ts = int(e.get("source_timestamp") or 0)
                    if ts < (t_test - day_sec):
                        continue
                    content = e.get("content") or {}
                    title = (content.get("title") or content.get("caption") or "").strip()
                    hashtags = (e.get("source_hashtags") or [])[:4]
                    feed_lines.append(
                        f"- {app} engagement: \"{title[:80]}\""
                        + (f" #{' #'.join(h.lstrip('#') for h in hashtags)}" if hashtags else "")
                        + f" ({_format_iso_date(ts)})"
                    )
            if feed_lines:
                lines.append("Recent feed activity (past 24h):")
                lines.extend(feed_lines[:6])
            return ("Catchup grounding (past 24h, real backend):\n" + "\n".join(lines)) if lines else ""

        if task_id == "agentic_trending_alert":
            t_test = int(inst.get("t_test") or 0)
            if not t_test:
                return ""
            day_sec = 24 * 3600
            from collections import Counter
            tag_counts: Counter = Counter()
            user_top_cats: set[str] = set()
            # Past-24h hashtag frequency across all social apps the user touched.
            for app in ("instagram", "facebook", "threads"):
                events = bq.get_events(
                    user_id=user_id, app=app,
                    since_timestamp=t_test,
                ) or []
                for e in events:
                    ts = int(e.get("source_timestamp") or 0)
                    if ts < (t_test - day_sec):
                        continue
                    for h in (e.get("source_hashtags") or []):
                        if h:
                            tag_counts[h.lstrip("#").lower()] += 1
                    for pref in (e.get("preferences") or []):
                        cat = (pref.get("category") or "").strip().lower()
                        if cat:
                            user_top_cats.add(cat)
            if not tag_counts:
                return ""
            top = tag_counts.most_common(8)
            lines = ["Trending hashtags in the user's window (past 24h):"]
            for tag, n in top:
                lines.append(f"- #{tag} ({n} engagements)")
            return "\n".join(lines)

        if task_id == "daily_personalized_briefing":
            # The gold is a 3–5 bullet "what should I catch up on today" briefing.
            # Without grounding the example LLM produces generic productivity
            # advice ("scan calendar, prioritize tasks") that doesn't lean on
            # any user prefs — and the disliked_recent foil then has nothing
            # persona-shaped to contrast against. Surface the user's actual
            # positive engagements for this t_test so the gold names real
            # topics (boxing, hip-hop, comedy, …) and the foil's swapped-in
            # disliked topic produces a visible-but-natural diff.
            pos = inst.get("gt_positive_engagements") or []
            avoid = inst.get("gt_avoid_engagements") or []
            if not pos:
                return ""
            # Pull a handful of distinct hashtags from the positive set,
            # preserving order so the briefing surfaces the user's freshest
            # interests first. Cap to 8 tags so the prompt stays compact.
            seen_tags: set[str] = set()
            tag_lines: list[str] = []
            for e in pos:
                for h in (e.get("hashtags") or []):
                    norm = (h or "").lstrip("#").lower()
                    if not norm or norm in seen_tags:
                        continue
                    seen_tags.add(norm)
                    tag_lines.append(f"#{h.lstrip('#')}")
                    if len(tag_lines) >= 8:
                        break
                if len(tag_lines) >= 8:
                    break
            avoid_tags: list[str] = []
            avoid_seen: set[str] = set()
            for e in avoid:
                for h in (e.get("hashtags") or []):
                    norm = (h or "").lstrip("#").lower()
                    if not norm or norm in avoid_seen or norm in seen_tags:
                        continue
                    avoid_seen.add(norm)
                    avoid_tags.append(f"#{h.lstrip('#')}")
                    if len(avoid_tags) >= 5:
                        break
                if len(avoid_tags) >= 5:
                    break
            lines = [
                "User's actual positive engagements for this briefing window "
                "(weave 2–3 of these into the bullets so the briefing is "
                "anchored to what the user is genuinely into right now — "
                "describe each in plain language, do NOT list raw hashtags):",
                "  " + ", ".join(tag_lines),
            ]
            if avoid_tags:
                lines.append(
                    "Topics the user explicitly disliked in the same window "
                    "(do NOT mention any of these — the foil will leak one):"
                )
                lines.append("  " + ", ".join(avoid_tags))
            return "\n".join(lines)

        if task_id in _VOICE_DEPENDENT_WRITE_TASKS:
            return _voice_grounding(inst, task_id, bq, user_id)
    except Exception:
        return ""
    return ""


# Voice-dependent agentic write tasks. Their gold is graded on `voice_match`,
# so the gold-gen LLM must see the user's actual voice for the target_app.
_VOICE_DEPENDENT_WRITE_TASKS = {
    "agentic_user_tone_post",
    "agentic_composed_post",
    "agentic_send_post",
    "agentic_cross_app_repost",
    "agentic_auto_reply",
}


def _voice_grounding(inst: dict, task_id: str, bq, user_id: str) -> str:
    """Build a voice-anchored grounding block for write tasks.

    Pulls the target_app's `style_description` / `topical_focus` /
    `posting_frequency` from `profile.app_personas`, the user's top
    hashtags on that app, and the 2-3 most recent self-posts. Each
    task adds its own specific input (the update / context / source
    post / inbound DM) so the gold has both voice anchor and the
    concrete content to respond to.
    """
    from collections import Counter

    target_app = (inst.get("target_app") or "instagram").lower()
    t_test = int(inst.get("t_test") or 0)
    horizon = t_test or 9999999999

    # 1. style_description / topical_focus / posting_frequency from app_personas
    # plus the user's shared writing voice (caps, palette, phrases, ...) — the
    # same person types this on every app, so the gold-gen LLM needs both
    # blocks to mimic the voice correctly.
    app_persona: dict = {}
    user_voice: dict = {}
    try:
        prof = bq._load_profile(user_id) if hasattr(bq, "_load_profile") else {}
        for k, v in (prof.get("app_personas") or {}).items():
            if isinstance(v, dict) and k.lower() == target_app:
                app_persona = v
                break
        uv = prof.get("user_voice") if isinstance(prof, dict) else None
        if isinstance(uv, dict):
            user_voice = uv
    except Exception:
        app_persona = {}
        user_voice = {}

    # 2. Top hashtags + recent self-posts on the target app.
    tag_counts: Counter = Counter()
    self_posts: list[str] = []
    try:
        for e in bq.get_events(
            user_id=user_id, app=target_app, since_timestamp=horizon,
        ) or []:
            for h in (e.get("source_hashtags") or []):
                if h:
                    tag_counts[h.lstrip("#").lower()] += 1
            if (e.get("source_interaction_type") or "") == "self_post":
                content = e.get("content") or {}
                text = (content.get("caption") or content.get("title") or "").strip()
                if text:
                    self_posts.append(text[:200])
    except Exception:
        pass

    lines: list[str] = []
    # 1a. Shared writing voice — the same person types this on every app.
    if isinstance(user_voice, dict) and user_voice:
        lines.append("User's shared writing voice (consistent across all apps):")
        if user_voice.get("natural_register"):
            lines.append(f"  register: {user_voice['natural_register']}")
        if user_voice.get("default_capitalization"):
            lines.append(f"  capitalization: {user_voice['default_capitalization']}")
        if user_voice.get("punctuation_habits"):
            lines.append(f"  punctuation: {user_voice['punctuation_habits']}")
        if user_voice.get("humor_tone"):
            lines.append(f"  humor / tone: {user_voice['humor_tone']}")
        palette = user_voice.get("emoji_palette") or []
        if palette:
            lines.append(
                f"  personal emoji palette (subset only — never invent new): "
                f"{' '.join(palette)} (intensity: {user_voice.get('emoji_intensity_default', 'medium')})"
            )
        phrases = user_voice.get("personal_phrases") or []
        if phrases:
            lines.append(
                "  personal phrases (cross-app — use occasionally as natural tics): "
                + ", ".join(f"\"{p}\"" for p in phrases[:6])
            )
        if user_voice.get("formality_baseline") is not None:
            lines.append(f"  formality baseline: {user_voice['formality_baseline']}")

    # 1b. Per-app expression — what shifts on the target app.
    style = (app_persona.get("style_description") or "").strip()
    expression = app_persona.get("expression") or {}
    overrides = app_persona.get("overrides") or {}
    audience_lens = (app_persona.get("audience_lens") or "").strip()
    if style or expression or audience_lens:
        lines.append(f"On {target_app} this voice modulates:")
        if audience_lens:
            lines.append(f"  audience lens: {audience_lens}")
        if style:
            lines.append(f"  style delta: {style[:400]}")
        if isinstance(expression, dict) and expression:
            if expression.get("effort_level"):
                lines.append(f"  effort: {expression['effort_level']}")
            if expression.get("length_band"):
                lines.append(f"  length: ~{expression['length_band']} chars")
            eis = expression.get("emoji_intensity_shift")
            if eis is not None:
                shift_label = "0 (default)" if eis == 0 else (f"+{eis}" if eis > 0 else str(eis))
                lines.append(f"  emoji intensity shift: {shift_label}")
            if expression.get("emoji_topic_filter"):
                lines.append(f"  which palette emoji surface here: {expression['emoji_topic_filter']}")
            if expression.get("audience_self_censoring"):
                lines.append(f"  audience self-censoring: {expression['audience_self_censoring']}")
        if isinstance(overrides, dict) and overrides:
            lines.append("  overrides (apply only these; other voice mechanics inherit from shared voice):")
            for ok, ov in overrides.items():
                ov_repr = "; ".join(ov) if isinstance(ov, list) else str(ov)
                lines.append(f"    {ok}: {ov_repr}")

    # 1c. Legacy backward-compat — if the backend predates the shared-voice
    # refactor, fall back to the old voice_signature block so old eval runs
    # still get a voice anchor.
    if not user_voice and not expression:
        sig = app_persona.get("voice_signature") or {}
        if isinstance(sig, dict) and sig:
            lines.append(f"Voice signature for {target_app} (legacy):")
            if sig.get("capitalization"):
                lines.append(f"  capitalization: {sig['capitalization']}")
            if sig.get("punctuation_habits"):
                lines.append(f"  punctuation: {sig['punctuation_habits']}")
            if sig.get("sentence_shape"):
                lines.append(f"  sentence shape: {sig['sentence_shape']}")
            if sig.get("length_chars"):
                lines.append(f"  length: ~{sig['length_chars']} chars")
            rec = sig.get("recurring_phrases") or []
            if rec:
                lines.append("  recurring phrases the user actually uses: "
                             + ", ".join(f"\"{p}\"" for p in rec[:5]))
            emoji = sig.get("emoji_policy")
            if isinstance(emoji, dict):
                elist = emoji.get("emojis") or []
                place = emoji.get("placement") or ""
                if elist:
                    lines.append(f"  emoji policy: uses {' '.join(elist)}"
                                 + (f", placed {place}" if place else ""))
            elif isinstance(emoji, str) and emoji.strip():
                lines.append(f"  emoji policy: {emoji}")
            if sig.get("hashtag_policy"):
                lines.append(f"  hashtag policy: {sig['hashtag_policy']}")
            forbid = sig.get("forbidden_patterns") or []
            if forbid:
                lines.append("  user NEVER does: " + "; ".join(forbid[:5]))

    topical = app_persona.get("topical_focus") or []
    if topical:
        lines.append(f"Topical focus on {target_app}: " + ", ".join(topical[:6]))
    freq = (app_persona.get("posting_frequency") or "").strip()
    if freq:
        lines.append(f"Posting frequency: {freq}")
    top_tags = [t for t, _ in tag_counts.most_common(8)]
    if top_tags:
        lines.append("Top hashtags: " + ", ".join("#" + t for t in top_tags))
    for sp in self_posts[-3:][::-1]:
        lines.append(f"- recent self-post: \"{sp}\"")
    # Anti-cliché floor — applies even if voice_signature is missing
    # (legacy backends without the structured fields). The LLM defaults
    # to a generic Instagram-caption register otherwise; this list bans
    # the worst offenders so the gold has to actually look at the
    # signature / recent self-posts to find a register.
    lines.append(
        "FORBIDDEN clichés (do NOT include any of these in the gold response): "
        "\"feeling proud, exhausted, and grateful\"; \"big things coming soon\"; "
        "trailing \"✨\" / \"💫\"; \"so blessed\"; \"forever grateful\"; "
        "\"this season of life\"; \"such a vibe\"; \"living for this\"; "
        "\"I'll never forget\"; emotion-summary tricolons (\"X, Y, and Z\" of feelings); "
        "any sentence that telegraphs personalization (\"as a fan of\", \"since you love\")."
    )
    lines.append(
        "Mimic the shared writing voice mechanically — apply the user's "
        "default_capitalization, occasional personal_phrases, punctuation_habits, "
        "and a topic-fit subset of the emoji palette (NEVER invent new emoji). "
        "The per-app expression block tells you how length / effort / emoji "
        "intensity shift on this specific app. The gold should sound like THIS "
        "user wrote it on this app, not a generic LLM."
    )

    # 3. Per-task specific input.
    if task_id == "agentic_composed_post":
        update = (inst.get("update") or "").strip()
        if update:
            lines.append(f"User's life-update brief to post: \"{update}\"")
    elif task_id == "agentic_send_post":
        ctx = (inst.get("context") or "").strip()
        if ctx:
            lines.append(f"Chat context to dispatch as a post: \"{ctx}\"")
    elif task_id == "agentic_cross_app_repost":
        sp = inst.get("source_post") or {}
        cap = (sp.get("caption") or sp.get("title") or "").strip()
        src_app = (inst.get("source_app") or sp.get("source_app") or "").strip()
        if cap:
            lines.append(
                f"Source post" + (f" from {src_app}" if src_app else "") + f": \"{cap[:240]}\""
            )
    elif task_id == "agentic_auto_reply":
        sender = (inst.get("sender_id") or "").strip()
        msg = (inst.get("inbound_message") or "").strip()
        if msg:
            lines.append(
                f"Inbound DM from {sender or 'a friend'}: \"{msg[:240]}\""
            )

    return "\n".join(lines)


def _compute_ranking_example(inst: dict, task_type: str) -> str:
    """Deterministic ranked-index 'example_response' for ranking tasks.
    Returns a compact list of ints with the held-out at rank 1, hard
    negatives last, fillers in between."""
    if task_type == "personalized_feed_ranking":
        slate = inst.get("slate") or []
        held = inst.get("held_out_idx")
        origins = inst.get("origin_by_idx") or []
        if not isinstance(held, int) or not slate:
            return ""
        # Order: held-out → past_positive → future_positive → others → negative last
        n = len(slate)
        priority = {"held_out": 0, "past_positive": 1, "future_positive": 2,
                    "filler": 3, "irrelevant": 3, "hard_neg": 4, "negative": 5}
        order = sorted(
            range(n),
            key=lambda i: (priority.get(origins[i] if i < len(origins) else "filler", 3),
                          0 if i == held else 1, i),
        )
        return f"Ranked indexes: {order}"
    if task_type == "personalized_recommendation":
        cands = inst.get("candidates") or []
        held = inst.get("held_out_idx")
        hard_negs = set(inst.get("hard_negative_idxs") or [])
        t_test = int(inst.get("t_test") or 0)
        n = len(cands)
        if not isinstance(held, int) or not cands:
            return ""
        # Held-out anchored at rank 1 (it's the metric's target). Remaining
        # fillers and hard_negs each sorted by |ts − t_test| asc, future-first
        # tie-break.
        def _key(i: int) -> tuple:
            ts = int((cands[i] or {}).get("source_timestamp") or 0)
            return (abs(ts - t_test), 0 if ts >= t_test else 1, i)
        fillers = sorted(
            (i for i in range(n) if i != held and i not in hard_negs),
            key=_key,
        )
        ranked_negs = sorted(hard_negs, key=_key)
        order = [held] + fillers + ranked_negs
        return f"Ranked indexes: {order}"
    if task_type == "at_ai_directive_followup":
        cands = inst.get("candidates") or []
        pos = list(inst.get("positive_indices") or [])
        carve = set(inst.get("carveout_indices") or [])
        n = len(cands)
        order = pos + [i for i in range(n) if i not in set(pos) and i not in carve]
        order += sorted(carve)
        return f"Ranked indexes: {order}"
    if task_type == "short_vs_long_term_lifecycle":
        cands = inst.get("candidates") or []
        matching = list(inst.get("matching_indices") or [])
        n = len(cands)
        order = matching + [i for i in range(n) if i not in set(matching)]
        return f"Ranked indexes: {order}"
    return ""


def _compute_ranking_inferior(inst: dict, task_type: str) -> str:
    """Deterministic 'inferior_response' for ranking tasks — same wrapper
    and length as the example, but the ordering inverts the personalization
    rule (hard negatives / disliked carve-outs surfaced first, the
    held-out / positive matches buried last).

    No LLM call. The example/inferior pair differ ONLY in index order, so
    a grader cannot win on surface features (length, tone, format)."""
    if task_type == "personalized_feed_ranking":
        slate = inst.get("slate") or []
        held = inst.get("held_out_idx")
        origins = inst.get("origin_by_idx") or []
        if not isinstance(held, int) or not slate:
            return ""
        n = len(slate)
        # Inverted: negatives first, held-out last.
        priority = {"negative": 0, "hard_neg": 1, "irrelevant": 2,
                    "filler": 2, "future_positive": 3, "past_positive": 4,
                    "held_out": 5}
        order = sorted(
            range(n),
            key=lambda i: (priority.get(origins[i] if i < len(origins) else "filler", 2),
                          1 if i == held else 0, i),
        )
        return f"Ranked indexes: {order}"
    if task_type == "personalized_recommendation":
        cands = inst.get("candidates") or []
        held = inst.get("held_out_idx")
        hard_negs = set(inst.get("hard_negative_idxs") or [])
        t_test = int(inst.get("t_test") or 0)
        n = len(cands)
        if not isinstance(held, int) or not cands:
            return ""
        # Inverted: hard negatives surfaced first (closest in time = most
        # confusable bad item), then fillers (same time-key), held-out buried
        # last.
        def _key(i: int) -> tuple:
            ts = int((cands[i] or {}).get("source_timestamp") or 0)
            return (abs(ts - t_test), 0 if ts >= t_test else 1, i)
        ranked_negs = sorted(hard_negs, key=_key)
        fillers = sorted(
            (i for i in range(n) if i != held and i not in hard_negs),
            key=_key,
        )
        order = ranked_negs + fillers + [held]
        return f"Ranked indexes: {order}"
    if task_type == "at_ai_directive_followup":
        cands = inst.get("candidates") or []
        pos = list(inst.get("positive_indices") or [])
        carve = set(inst.get("carveout_indices") or [])
        n = len(cands)
        # Inverted: carve-outs (the user told the AI to stop recommending
        # these) surfaced first, positive matches buried last.
        order = sorted(carve) + [i for i in range(n) if i not in set(pos) and i not in carve]
        order += pos
        return f"Ranked indexes: {order}"
    if task_type == "short_vs_long_term_lifecycle":
        cands = inst.get("candidates") or []
        matching = list(inst.get("matching_indices") or [])
        n = len(cands)
        # Inverted: non-matching first, matching buried last.
        order = [i for i in range(n) if i not in set(matching)] + matching
        return f"Ranked indexes: {order}"
    return ""


# Deterministic gold/foil for agentic_moment_recommendation removed when
# that task merged into personalized_recommendation. The new moment-flavored
# instances ride the same `_compute_ranking_example` / `_compute_ranking_inferior`
# path as the proactive recsys flavor — `Ranked indexes: [...]` wrapper, not
# the JSON-list shape this code used to emit.


# ---------------------------------------------------------------------------
# Workstream I — self-check prompt
# ---------------------------------------------------------------------------

_SELF_CHECK_PROMPT = """You are checking whether a candidate response actually \
ATTEMPTED the user's request — not whether it's high quality, just whether \
it's a real attempt as opposed to an echo or a non-answer.

The candidate response is the "gold reference" produced by sending only the \
user query to a clean LLM. It should at minimum (a) be different from the \
input where the request implies a transformation (tighten / rewrite / \
summarize / translate / shorten), and (b) actually engage with what was \
asked, not deflect.

Score 0 to 3:
  3 = clear, direct attempt at the request.
  2 = engaged with the request but partial / superficial.
  1 = barely engaged; very close to the input or a deflection.
  0 = echoes the input verbatim, or refuses without addressing the request.

Respond with ONE fenced ```json block:
```json
{{"score": <0..3>, "reason": "<one short sentence>"}}
```

User query:
\"\"\"
{query}
\"\"\"

Candidate response:
\"\"\"
{response}
\"\"\""""


def _run_self_check(llm: Callable[[str], str], task_type: str, query: str, response: str) -> dict:
    if not llm or not response:
        return {"score": 3, "passed": True, "reason": "(no llm available; defaulted to pass)"}
    raw = llm(_SELF_CHECK_PROMPT.format(
        query=(query or "")[:1500],
        response=response[:1500],
    ))
    parsed = extract_json_from_response(raw) or {}
    score = parsed.get("score")
    if not isinstance(score, (int, float)):
        return {"score": None, "passed": False, "reason": f"(parse failure: {raw[:80]!r})"}
    return {
        "score": int(score),
        "passed": int(score) >= 2,
        "reason": (parsed.get("reason") or "")[:160],
    }


# ---------------------------------------------------------------------------
# Workstream J — inferior_response generation
# ---------------------------------------------------------------------------

_FLAW_KINDS = ("incorrect_personalization", "disliked_recent", "over_personalization", "factual_error", "voice_mismatch")

# Personalization-flavored flaws (for chatbot / write-a-post tasks where
# the gold leans on persona signal).
_FLAW_KINDS_PERSONALIZATION = ("incorrect_personalization", "disliked_recent", "over_personalization")
# Factual flaws (for summary / lookup tasks where the gold is reporting
# real content — the natural failure mode is a wrong detail, not an
# off-topic persona aside).
_FLAW_KINDS_FACTUAL = ("factual_error",)
# Voice-only flaw (for write/compose agentic tasks where the rubric is
# voice_match + preference_alignment + avoid_overpersonalization). The
# natural failure mode is right content / wrong tone register — NOT a
# topic leak. The foil preserves the gold's content verbatim and only
# swaps the voice.
_FLAW_KINDS_VOICE = ("voice_mismatch",)

# Per-task allowlist. Tasks NOT listed fall back to the personalization set.
# Summarization / digest / lookup tasks need factual flaws; mechanically
# inserting "Follows mixed martial arts as a fan." into a DM digest reads
# absurdly because the gold doesn't have a preference reference to replace.
# Write/compose tasks (agentic_user_tone_post, agentic_composed_post,
# agentic_send_post, agentic_cross_app_repost, agentic_auto_reply) get
# voice_mismatch — the rubric grades voice_match, so the natural failure
# is wrong tone register, not a hashtag splice.
_TASK_FLAW_KINDS: dict[str, tuple[str, ...]] = {
    "agentic_dm_digest":               _FLAW_KINDS_FACTUAL,
    "agentic_group_dm_summary":        _FLAW_KINDS_FACTUAL,
    # Personalization-driven proactive tasks — gold picks content FOR
    # this user, natural failure is picking content for a different user
    # (a topic this user explicitly disliked recently). Was _FLAW_KINDS_FACTUAL
    # which produced friend_9↔friend_10 single-name swaps — too subtle and
    # structural rather than persona-divergent (the model would learn "use
    # the right ID" instead of "pick the right content"). Now pinned to
    # disliked_recent so the foil swaps in a freshly-disliked topic — the
    # diff is (1) visible to a human reader, (2) subtle + natural (the foil
    # is a plausible recommendation for some user just not for this one
    # at this moment), (3) not a structural / format difference.
    "agentic_proactive_daily_catchup": ("disliked_recent",),
    "agentic_trending_alert":          ("disliked_recent",),
    "agentic_vague_refind":            _FLAW_KINDS_FACTUAL,
    "agentic_user_tone_post":          _FLAW_KINDS_VOICE,
    "agentic_composed_post":           _FLAW_KINDS_VOICE,
    "agentic_send_post":               _FLAW_KINDS_VOICE,
    "agentic_cross_app_repost":        _FLAW_KINDS_VOICE,
    "agentic_auto_reply":              _FLAW_KINDS_VOICE,
    # daily_personalized_briefing's rubric explicitly grades
    # `negative_leakage` against `gt_avoid_engagements`. Pin the foil to
    # `disliked_recent` so the inferior response is the gold + ONE topic
    # the user actually disliked that day — the exact failure mode the
    # rubric measures. Generic incorrect_personalization rewrites tend to
    # produce near-identical paraphrases that don't visibly fail the rubric.
    "daily_personalized_briefing":     ("disliked_recent",),
}

_INFERIOR_PROMPT = """You are creating a paired *foil* response. The foil must be a \
plausible, helpful answer that a competent agent might give to a different \
user (or to this user in a different moment) — NOT obviously wrong, dumb, or \
low-quality. The flaw is targeting the WRONG personalization signal, not bad \
response quality.

Gold reference ({gold_length} chars):
\"\"\"
{response}
\"\"\"

User query the gold was answering:
\"\"\"{query}\"\"\"

Flaw kind: {flaw_kind}
{flaw_instruction}

Rules (universal):
  - The foil must be a fluent, grammatical message a real human would send.
    NEVER produce splice fragments like ", #NFL, #ZachWilson" or bare
    third-person fact-statements ("Enjoys X.", "Follows Y.").
  - Subtlety: the foil should be a wrong-but-PLAUSIBLE response, not obvious
    vandalism. A reader who didn't see the gold should still parse the foil
    as a normal, well-formed message that another user could reasonably want.
    Avoid phrasings like "you're interested in X", "you like Y" — the
    personalization slip should look like the agent reaching for a real
    preference at the wrong moment, not reciting a profile.
  - DO NOT prepend or append a separable clause to the gold. DO NOT keep the
    gold's exact opening words. The foil must be an INDEPENDENT rewrite, not
    a minimal edit of the gold. A reader who saw both side-by-side should
    find them visibly different in wording — NOT "the gold + a tail clause"
    and NOT "the gold with one word swapped".
  - Length: foil should land within ±20% of {gold_length} characters
    (≈ {gold_length_lo}–{gold_length_hi} chars). Per-flaw instructions may
    tighten this further.
  - Tone register: match the gold's overall register (casual / formal /
    bulleted / single-sentence / etc) — only the personalization signal
    differs.
  - Do NOT add disclaimers, parenthetical notes, or commentary about the
    change.

(The per-flaw instruction above governs WHAT to change. These universal rules
 govern HOW to keep the rewrite valid.)

Output ONE fenced ```json block:
```json
{{"text": "<the foil response — independently rewritten, plausible-but-wrong-for-this-moment>"}}
```"""


def _texts_too_similar(a: str, b: str) -> bool:
    """True only when the rewrite is byte-identical or a trivial whitespace
    variant. Kept for back-compat; the broader similarity check lives in
    `_validate_inferior`."""
    a = " ".join((a or "").split())
    b = " ".join((b or "").split())
    if not a or not b:
        return False
    return a == b


# Loosened from 0.15 → 0.20 to match the new per-family instructions.
_FOIL_LENGTH_TOLERANCE: float = 0.20


def _check_length_match(gold: str, foil: str,
                        tolerance: float = _FOIL_LENGTH_TOLERANCE) -> bool:
    """True if the foil's character count is within ±tolerance of the gold's.
    Without this, an LLM-generated foil often pads the gold with an extra
    clause — a grader can then distinguish gold/foil by length alone.
    """
    if not gold:
        return True
    return abs(len(foil) - len(gold)) / len(gold) <= tolerance


def _token_jaccard(a: str, b: str) -> float:
    """Token-level Jaccard over lowercase whitespace tokens. 0.0 if either
    is empty. Used to detect minimal-edit foils (Jaccard > 0.85 means the
    foil is mostly a word-swap of the gold)."""
    at = set((a or "").lower().split())
    bt = set((b or "").lower().split())
    if not at or not bt:
        return 0.0
    return len(at & bt) / len(at | bt)


_OPENING_TOKEN_OVERLAP = 5  # if first N tokens match exactly, reject


def _normalize_for_compare(s: str) -> str:
    """Collapse whitespace + lowercase for substring containment checks.
    Doesn't strip punctuation — preserves the contour of the sentence."""
    return " ".join((s or "").lower().split())


def _validate_inferior(example: str, inferior: str,
                       jaccard_max: float = 0.85,
                       jaccard_min: float = 0.05) -> tuple[bool, str]:
    """Reject foils that fail any of the structural similarity bounds.

    Returns (passed, reason). The reason string is fed back to the LLM on
    retry so it knows which constraint to fix.

    Rejects:
      - prefix overlap (one is a literal prefix of the other) → templated tail
      - substring containment (one wraps the other after normalization) →
        mid-string injection, e.g. EX kept verbatim with hashtags wedged in
      - same opening N tokens → minimal-edit foil that didn't rewrite the
        opening clause
      - token Jaccard > jaccard_max → minimal-edit foil
      - token Jaccard < jaccard_min → off-topic, doesn't address the same query
      - length-ratio > 0.5 → grader can win on length alone

    Voice-style flaws use a stricter jaccard_max (0.6) — the prompt already
    asks for paraphrase. Pass jaccard_max=0.6 from the call site."""
    if not example or not inferior:
        return False, "empty_text"
    a = example.strip()
    b = inferior.strip()
    if a == b:
        return False, "identical_text"
    if a and b and (a.startswith(b) or b.startswith(a)):
        return False, "prefix_overlap (one response is a literal prefix of the other)"
    # Normalized substring containment catches mid-string injection (e.g.
    # gold kept verbatim with #hashtags wedged in before the closing
    # punctuation). _normalize_for_compare collapses whitespace + casing
    # so trivial reformatting can't bypass the check.
    na, nb = _normalize_for_compare(a), _normalize_for_compare(b)
    # Only flag containment when the contained side is non-trivial (≥40 chars
    # or ≥7 tokens). Below that, contained-substring matches happen by
    # chance — e.g. a one-line foil and a one-line gold may share a
    # generic phrase like "Sure, here's what I'd say."
    if (na in nb or nb in na) and min(len(na), len(nb)) >= 40 \
            and len(min(na, nb, key=len).split()) >= 7:
        return False, (
            "substring_containment (one response wraps the other "
            "verbatim — looks like a mid-string injection rather than "
            "an independent rewrite)"
        )
    # Opening-N-tokens overlap: if the first N tokens match exactly, the
    # foil failed to rewrite the opening clause. _OPENING_TOKEN_OVERLAP=5
    # is small enough that a stylistic match (e.g., "Hi, I just wanted to")
    # passes naturally for short replies but catches "I'm so sorry for
    # your loss." kept verbatim.
    a_tokens = a.lower().split()
    b_tokens = b.lower().split()
    n_open = _OPENING_TOKEN_OVERLAP
    if (len(a_tokens) >= n_open and len(b_tokens) >= n_open
            and a_tokens[:n_open] == b_tokens[:n_open]):
        return False, (
            f"opening_overlap (first {n_open} tokens of foil match gold's "
            f"opening verbatim — rewrite the opening clause)"
        )
    j = _token_jaccard(a, b)
    if j > jaccard_max:
        return False, (
            f"too_similar (token Jaccard {j:.2f} > {jaccard_max:.2f}; "
            f"foil shares too many words with gold — paraphrase more)"
        )
    if j < jaccard_min:
        return False, (
            f"too_dissimilar (token Jaccard {j:.2f} < {jaccard_min:.2f}; "
            f"foil shares too few words with gold — likely off-topic)"
        )
    if abs(len(b) - len(a)) / max(len(a), 1) > 0.5:
        return False, (
            f"length_mismatch ({len(b)} chars vs gold {len(a)} chars; "
            f"too far apart — grader could win on length alone)"
        )
    return True, ""


def _flaw_instruction(flaw_kind: str, evidence: dict, task_id: str = "") -> str:
    if flaw_kind == "incorrect_personalization":
        pi = evidence.get("persona_item", "")
        return (
            f"INDEPENDENTLY write a NEW response to the user query above. The "
            f"foil should naturally weave in this persona item: \"{pi}\". The "
            f"persona item IS true for the user but is IRRELEVANT to the "
            f"current query — a competent agent answering this same query "
            f"for THIS user at THIS moment would not lean on it. The foil "
            f"agent has drifted off-topic.\n"
            f"  - Reference the persona item as fluent prose woven into a "
            f"clause (e.g. \"btw, did you catch the comedy clips this "
            f"weekend?\"), NOT a third-person fact-statement (\"Enjoys X.\", "
            f"\"Follows Y.\"), NOT parenthetical.\n"
            f"  - Do NOT echo the gold's opening words. Do NOT borrow the "
            f"gold's specific phrasing. The foil should read as if a "
            f"different agent wrote it — not as the gold with edits.\n"
            f"  - The foil should still be a coherent, helpful response that "
            f"another user might appreciate — it's wrong because of the "
            f"WHAT (the topic choice), not because the writing is bad."
        )
    if flaw_kind == "disliked_recent":
        pi = evidence.get("persona_item", "")
        topic_hint = (evidence.get("topic_hint") or pi or "").strip()
        content_snippet = (evidence.get("content_snippet") or "").strip()
        if task_id == "daily_personalized_briefing":
            grounding_lines = [
                f"  - Topic the user disliked later that same day: {topic_hint}",
            ]
            if content_snippet:
                grounding_lines.append(
                    f"  - Real content snippet from the disliked engagement "
                    f"(use as inspiration for the bullet's subject — do NOT "
                    f"quote verbatim): \"{content_snippet}\""
                )
            grounding = "\n".join(grounding_lines)
            return (
                f"The gold is a bulleted briefing. Produce a foil briefing "
                f"with EXACTLY THE SAME number of bullets as the gold. "
                f"REPLACE one of the gold's bullets (chosen at a plausible, "
                f"non-trivial position — NOT always the last one) with a "
                f"new bullet whose SUBJECT is the topic below — a topic the "
                f"user actually disliked elsewhere that same day, so a "
                f"competent agent would have left it off the briefing.\n"
                f"{grounding}\n"
                f"Bullet style — STRICT (the leak should be the topic "
                f"choice, not the wording):\n"
                f"  - Write the replacement bullet in the EXACT SAME prose "
                f"style as the other bullets. If the other bullets are full "
                f"sentences with no hashtags, the replacement is also a "
                f"full sentence with no hashtags.\n"
                f"  - DO NOT include any `#` characters, hashtag tokens, or "
                f"comma-separated tag lists (e.g., `#hiphop, #rapmusic, "
                f"#newyorkrap`). Describe the topic in plain natural "
                f"language the way a normal news/lifestyle bullet would.\n"
                f"  - DO NOT use meta-framing prefixes that mark the bullet "
                f"as different — phrases like `One quick cultural note:`, "
                f"`Cultural update:`, `Trending now:`, `Also worth flagging:`, "
                f"`On a separate note:`, `By the way:` are banned. Open the "
                f"replacement bullet with the same kind of opening the "
                f"other bullets use (e.g. a topic noun-phrase, a `Markets / "
                f"Tech / World / Sports:` style label, or whatever pattern "
                f"the gold establishes).\n"
                f"  - Match the OTHER bullets' length (±25%), tone, "
                f"punctuation, emoji density, and starting character.\n"
                f"  - The foil bullet must read like a normal briefing item "
                f"a careless agent would include — NOT a meta note that the "
                f"user disliked it, NOT a third-person fact-statement.\n"
                f"  - DO NOT keep the replaced bullet. DO NOT add a new "
                f"bullet on top of the gold's bullets. Total bullet count "
                f"must equal the gold's.\n"
                f"  - You MAY paraphrase the OTHER bullets slightly "
                f"(preserving meaning) so the foil doesn't share long "
                f"word-for-word stretches with the gold."
            )
        return (
            f"INDEPENDENTLY write a new response to the user query above "
            f"that mentions this topic naturally in passing: \"{topic_hint}\". The "
            f"user explicitly DISLIKED this in the last 48 hours — a "
            f"competent agent would not have brought it up. The mention "
            f"must read fluently (a clause woven into a sentence), NOT a "
            f"third-person fact-statement spliced in.\n"
            f"  - DO NOT include `#` characters or hashtag tokens — describe "
            f"the topic in plain natural language.\n"
            f"  - Do NOT echo the gold's opening words or specific phrasing.\n"
            f"  - The foil should still be a coherent, helpful response that "
            f"another user might appreciate — it's wrong because the topic "
            f"is fresh-sour, not because the writing is bad."
        )
    if flaw_kind == "over_personalization":
        pi = evidence.get("persona_item", "")
        return (
            f"INDEPENDENTLY write a new response to the user query above "
            f"that includes an unprompted aside about: \"{pi}\". This is "
            f"a top user category but has ZERO overlap with the current "
            f"query — a digression a competent agent correctly omitted.\n"
            f"  - The aside must be a grammatical clause integrated into "
            f"the prose, NOT a separate fact-statement, NOT a "
            f"prepended/appended fragment, NOT a tail tacked onto the gold.\n"
            f"  - Do NOT echo the gold's opening words. Do NOT borrow the "
            f"gold's specific phrasing. The foil should read as a "
            f"separately-authored response.\n"
            f"  - The foil should still be a coherent, helpful response — "
            f"it's wrong because of the unsolicited topical drift, not "
            f"because the writing is bad."
        )
    if flaw_kind == "factual_error":
        return (
            "Introduce ONE subtle factual error inside the foil's content — "
            "for example: change a sender name to a different plausible "
            "name, swap one topic for another, change a count (e.g. "
            "'three' → 'two'), drop one of the items the gold lists, or "
            "attribute a message to the wrong person.\n"
            "  - You may rewrite the response with somewhat different "
            "phrasing than the gold (so the foil isn't trivially "
            "distinguishable from the gold by surface features), but "
            "preserve the gold's overall structure and length.\n"
            "  - The error should look like a careless mistake a hurried "
            "summarizer would make, not an off-topic personalization leak.\n"
            "  - Do NOT introduce any persona reference. Do NOT add "
            "disclaimers about the error."
        )
    if flaw_kind == "voice_mismatch":
        contrasting = evidence.get("contrasting_register", "")
        target_app = evidence.get("target_app", "")
        ctx_line = (
            f"Target app for this post: {target_app}. The gold matches the "
            f"user's natural {target_app} voice."
        ) if target_app else (
            "The gold matches the user's natural voice for this surface."
        )
        return (
            f"{ctx_line}\n"
            f"PARAPHRASE the gold's factual content (the message, the "
            f"update, the named entities, the hashtags if any) into this "
            f"contrasting voice register: \"{contrasting}\". The foil "
            f"communicates the SAME factual gist, but in a voice that's "
            f"clearly off for this user.\n"
            f"  - PARAPHRASE — do NOT preserve word-for-word matches with "
            f"the gold. Use different vocabulary and different sentence "
            f"structures. Aim for token-level Jaccard with the gold UNDER "
            f"0.6 (i.e., share at most ~60% of distinct word tokens).\n"
            f"  - Examples of correctly contrasted voice pairs:\n"
            f"      • casual IG caption (\"just dropped, link in bio 👀\") ↔ "
            f"stiff PR (\"Pleased to announce the launch is now live; full "
            f"details available.\")\n"
            f"      • friendly DM (\"hey omg congrats!! so excited for u "
            f"🥳\") ↔ formal corporate (\"I would like to extend my "
            f"congratulations on this milestone.\")\n"
            f"  - Length within ±20% of the gold. Structure: do NOT begin "
            f"the foil with the same opening words as the gold.\n"
            f"  - Do NOT introduce new hashtags, new topics, or new "
            f"factual claims. The ONLY axis that varies is voice register.\n"
            f"  - Both gold and foil must read as plausible utterances "
            f"someone might actually send — the foil is just plausibly "
            f"sent by a different user/persona than this one."
        )
    return ""


def _pick_flaw_evidence(flaw_kind: str, inst: dict, persona_ctx: dict,
                        rng: random.Random, task_id: str = "") -> dict | None:
    """Source the substitution from the user's real data. Returns None
    if no eligible evidence exists for this flaw kind on this instance."""
    if flaw_kind == "incorrect_personalization":
        # Another canonical from the persona that has zero hashtag overlap
        # with this instance's source_hashtags.
        inst_hashtags = {h.lower().lstrip("#") for h in (inst.get("source_hashtags") or [])}
        prefs = persona_ctx.get("top_prefs") or []
        candidates = [pi for pi, _ in prefs if pi]
        rng.shuffle(candidates)
        for pi in candidates:
            return {"persona_item": pi}  # we don't track per-pref hashtags here
        return None
    if flaw_kind == "disliked_recent":
        # Prefer the instance's own `gt_avoid_engagements` when present
        # (daily_personalized_briefing carries the actual rows the user
        # disliked the SAME day, AFTER t_test — directly tied to the
        # rubric's negative_leakage check, and harder to game than
        # remembering yesterday's negatives). Fall back to the generic
        # 48h `recent_negatives` only for tasks that don't carry per-
        # instance avoid lists; for daily_personalized_briefing we
        # explicitly do NOT fall back, since recent past dislikes are
        # too easy to recall and make the foil trivially detectable.
        def _clean_topic_hint(hashtags: list, snippet: str) -> str:
            cleaned = [str(h or "").lstrip("#").replace("_", " ").strip()
                       for h in hashtags]
            cleaned = [c for c in cleaned if c]
            if cleaned:
                return " / ".join(cleaned[:3])
            return (snippet or "").strip()[:80]

        gt_avoid = inst.get("gt_avoid_engagements") or []
        if gt_avoid:
            evidence = rng.choice(gt_avoid)
            hashtags = list(evidence.get("hashtags") or [])[:3]
            snippet = (evidence.get("content_snippet") or "")[:140]
            topic_hint = _clean_topic_hint(hashtags, snippet)
            if not topic_hint:
                return None
            return {
                "persona_item":    topic_hint,
                "topic_hint":      topic_hint,
                "content_snippet": snippet,
                "source_timestamp": evidence.get("ts", 0),
                "source_app":       evidence.get("app", ""),
                "_from": "gt_avoid_engagements",
            }
        # daily_personalized_briefing: no forward-looking dislike → skip
        # the foil entirely rather than reaching for past dislikes.
        if task_id == "daily_personalized_briefing":
            return None
        recent_negs = persona_ctx.get("recent_negatives") or []
        if not recent_negs:
            return None
        evidence = rng.choice(recent_negs)
        hashtags = list(evidence.get("hashtags") or [])[:3]
        snippet = evidence.get("persona_item", "") or ""
        topic_hint = _clean_topic_hint(hashtags, snippet)
        if not topic_hint:
            return None
        return {
            "persona_item":    topic_hint,
            "topic_hint":      topic_hint,
            "content_snippet": snippet,
            "source_object_id": evidence.get("source_object_id", ""),
            "source_timestamp": evidence.get("ts", 0),
            "source_app":       evidence.get("app", ""),
        }
    if flaw_kind == "over_personalization":
        cats = persona_ctx.get("top_categories") or []
        if cats:
            return {"persona_item": cats[0][0]}
        return None
    if flaw_kind == "factual_error":
        # No external evidence needed — the LLM mutates the gold's own
        # content. Return a sentinel non-None dict so the caller proceeds.
        return {"persona_item": ""}
    if flaw_kind == "voice_mismatch":
        # Pick a CONTRASTING voice register: prefer another app's voice
        # from the user's own app_personas (cross-app voice swap), fall
        # back to a generic stiff/corporate register opposite to the
        # casual social-app default.
        target_app = (inst.get("target_app") or "").lower()
        app_personas = persona_ctx.get("app_personas") or {}
        # Build candidates: every app voice EXCEPT the target.
        candidates: list[tuple[str, str]] = []
        for app, ap in app_personas.items():
            if app == target_app:
                continue
            style = (ap.get("style_description") or "").strip()
            if style:
                # Truncate aggressively — the foil prompt only needs a
                # short one-liner pointing at the contrasting voice.
                candidates.append((app, style[:280]))
        if candidates:
            rng.shuffle(candidates)
            other_app, style = candidates[0]
            register = f"the user's {other_app} voice — {style}"
            return {
                "contrasting_register": register,
                "target_app": target_app,
                "_from": f"app_personas[{other_app}]",
            }
        # Fallback: stiff corporate register opposite the casual social default.
        return {
            "contrasting_register": (
                "stiff, formal, fully-capitalized corporate-announcement "
                "voice — long noun phrases, no emoji, no lowercase, no "
                "casual contractions, like a press release"
            ),
            "target_app": target_app,
            "_from": "fallback_corporate",
        }
    return None


def _generate_inferior(llm: Callable[[str], str], response: str,
                       flaw_kind: str, evidence: dict,
                       task_id: str = "",
                       user_query: str = "") -> str | None:
    """LLM-rewrite path for non-ranking foils.

    Ranking-task foils are deterministic — see `_compute_ranking_inferior`
    and the dispatch in the foil loop. This function is only invoked for
    Family 2/3/4 (list/digest, voice, freeform).
    """
    if not llm or not response or not evidence:
        return None
    gold_len = len(response)
    gold_len_lo = max(1, int(gold_len * (1 - _FOIL_LENGTH_TOLERANCE)))
    gold_len_hi = int(gold_len * (1 + _FOIL_LENGTH_TOLERANCE))
    # Voice-style foils intentionally paraphrase the same content into a
    # different register — the prompt asks for Jaccard < 0.6, and a heavy
    # paraphrase can legitimately drop near zero. Loosen the validator on
    # both ends for voice. Other flaws keep the standard topical-overlap
    # floor (0.20) so an off-topic LLM hallucination still gets caught.
    if flaw_kind == "voice_mismatch":
        jaccard_max, jaccard_min = 0.6, 0.0
    else:
        jaccard_max, jaccard_min = 0.85, 0.05
    prompt = _INFERIOR_PROMPT.format(
        response=response[:1500],
        query=(user_query or "(no user query available)")[:600],
        flaw_kind=flaw_kind,
        flaw_instruction=_flaw_instruction(flaw_kind, evidence, task_id),
        gold_length=gold_len,
        gold_length_lo=gold_len_lo,
        gold_length_hi=gold_len_hi,
    )
    last_text: str | None = None
    last_reason = ""
    for attempt in range(3):
        suffix = ""
        if attempt > 0 and last_text is not None:
            suffix = (
                f"\n\nYour previous foil was REJECTED. Reason: {last_reason}.\n"
                f"Previous foil ({len(last_text)} chars):\n"
                f"\"\"\"{last_text[:600]}\"\"\"\n"
                f"Try again. Produce a NEW foil that does NOT trip this "
                f"check. Keep length within {gold_len_lo}–{gold_len_hi} "
                f"chars and follow the per-flaw instructions."
            )
        raw = llm(prompt + suffix)
        parsed = extract_json_from_response(raw) or {}
        text = parsed.get("text") or None
        if not text:
            last_text, last_reason = "", "no text returned"
            continue
        passed, reason = _validate_inferior(response, text,
                                            jaccard_max=jaccard_max,
                                            jaccard_min=jaccard_min)
        if not passed:
            last_text, last_reason = text, reason
            continue
        return text
    # All 3 attempts failed → return None so the caller drops the foil.
    return None


def _synthesize_user_query(inst: dict, task_id: str) -> str:
    """Agentic builders don't store `user_query` on the instance — the
    visualizer synthesizes it at render time via TEST_QUERY_EXTRACTORS.
    Postprocess needs the same string so the LLM-gen prompt has a real
    {query}. Reuse the visualizer's dispatcher rather than re-implementing.
    """
    direct = inst.get("user_query") or inst.get("query")
    if isinstance(direct, str) and direct.strip():
        return direct
    try:
        from data_preparation.visualize import TEST_QUERY_EXTRACTORS
        fn = TEST_QUERY_EXTRACTORS.get(task_id)
        if fn:
            return fn(inst) or ""
    except Exception:
        pass
    return ""


def _build_persona_ctx(bq, user_id: str, t_test: int) -> dict:
    """Lightweight persona context for inferior generation: top prefs,
    top categories, real negative engagements in the last 48h, and the
    per-app voice registers (app_personas) for voice_mismatch foils."""
    from collections import Counter
    DAY = 24 * 3600
    pref_counts: Counter = Counter()
    cat_counts: Counter = Counter()
    recent_negs: list[dict] = []
    # app_personas — capitalized keys in profile.json: Instagram/Facebook/Threads/Chatbot.
    # Normalize to lowercase keys to match inst["target_app"].
    # user_voice — shared across all apps; needed by the voice-evidence smoke
    # test in postprocess_benchmark to extract bolded anchors from the gold.
    app_personas: dict[str, dict] = {}
    user_voice: dict = {}
    try:
        prof = bq._load_profile(user_id) if hasattr(bq, "_load_profile") else {}
        for k, v in (prof.get("app_personas") or {}).items():
            if isinstance(v, dict):
                app_personas[k.lower()] = v
        uv = prof.get("user_voice") if isinstance(prof, dict) else None
        if isinstance(uv, dict):
            user_voice = uv
    except Exception:
        app_personas = {}
        user_voice = {}
    for app in ("instagram", "facebook", "threads"):
        for e in bq._load_events(user_id, app):
            ts = int(e.get("source_timestamp") or 0)
            if ts >= t_test:
                continue
            for pref in (e.get("preferences") or []):
                pi = (pref.get("persona_item") or "").strip()
                cat = (pref.get("category") or "").strip()
                if pi:
                    pref_counts[pi] += 1
                if cat:
                    cat_counts[cat] += 1
            if (e.get("source_interaction_type") or "") == "explicit_negative" \
               and (t_test - 2 * DAY) <= ts < t_test:
                content = e.get("content") or {}
                recent_negs.append({
                    "ts": ts,
                    "app": app,
                    "source_object_id": e.get("source_object_id", ""),
                    "hashtags": e.get("source_hashtags") or [],
                    "persona_item": (content.get("caption") or content.get("title") or "")[:100],
                })
    return {
        "top_prefs": pref_counts.most_common(8),
        "top_categories": cat_counts.most_common(6),
        "recent_negatives": recent_negs,
        "app_personas": app_personas,
        "user_voice": user_voice,
    }


# ---------------------------------------------------------------------------
# Orchestrator entry point
# ---------------------------------------------------------------------------

def postprocess_benchmark(bm: dict, bq, user_id: str,
                          self_check_llm: Callable[[str], str] | None = None,
                          inferior_llm: Callable[[str], str] | None = None,
                          voice_check_llm: Callable[[str], str] | None = None,
                          rng_seed: int = 0,
                          verbose: bool = False) -> dict:
    """Run workstream I (self-check) + J (inferior_response) over every
    personalization instance in the assembled benchmark dict. Mutates
    instances in place; returns the same dict for chaining.

    `voice_check_llm` (mini-tier, e.g. gpt-5.4-mini) gates compose-task
    example/inferior pairs: after both are generated, the bolded voice
    evidence in the gold is extracted and the mini is asked to pick the
    better response. If it fails or picks the foil, the inferior is
    regenerated once. Falls back to `self_check_llm` when None.
    """
    rng = random.Random(rng_seed or hash(user_id) % (2**31))
    if voice_check_llm is None:
        voice_check_llm = self_check_llm
    n_example_llm_gen = 0
    n_example_ranking = 0
    n_self_check = 0
    n_self_check_failed = 0
    n_inferior_built = 0
    n_inferior_skipped = 0
    n_voice_check_passed = 0
    n_voice_check_failed = 0
    n_voice_check_regen = 0

    # Lazy-build persona ctx once per t_test so we don't re-scan per instance.
    _ctx_cache: dict[int, dict] = {}

    # Pull the visualize extractors so we can compute example_response for
    # each instance at build time.
    from data_preparation.visualize import (
        TEST_GT_EXTRACTORS, _gt_default, _gt_agentic, _build_persona_context,
    )
    # The extractors read a module-level _PERSONA_CONTEXT — populate it once.
    import data_preparation.visualize as _viz
    _viz._PERSONA_CONTEXT = _build_persona_context(user_id)

    for task_id, items in bm.items():
        if not isinstance(items, list) or task_id not in _PERSONALIZATION_TASKS:
            continue
        gt_extractor = TEST_GT_EXTRACTORS.get(task_id)
        if gt_extractor is None:
            # Agentic tasks fall back to the generic extractor.
            if task_id.startswith("agentic_"):
                gt_extractor = _gt_agentic
            else:
                gt_extractor = _gt_default
        for idx, inst in enumerate(items):
            try:
                gt_out = gt_extractor(inst) or {}
            except Exception:
                gt_out = {}
            example = (gt_out.get("example_response") or "").strip()
            groundtruth = gt_out.get("groundtruth_preference") or ""
            inst["example_response"] = example
            inst["groundtruth_preference"] = groundtruth
            if "tool_call" in gt_out:
                inst["tool_call"] = gt_out["tool_call"]

            # NEW pass: replace meta-instruction example_response with a
            # concrete one. The gold-gen LLM call receives ONLY the user
            # query — no persona, no task_guidance, no prior_conversation,
            # no enrichment. The Example Response represents what a clean
            # LLM would say given just this query. The eval rubric (not
            # the gold) is the authority on whether the agent's actual
            # response uses persona correctly.
            user_query = _synthesize_user_query(inst, task_id)
            if task_id in _RANKING_TASKS:
                ranked = _compute_ranking_example(inst, task_id)
                if ranked:
                    example = ranked
                    inst["example_response"] = example
                    n_example_ranking += 1
            elif task_id in _DETERMINISTIC_GOLD_TASKS:
                # Extension point for future deterministic-JSON gold tasks
                # (set is empty today — agentic_moment_recommendation merged
                # into personalized_recommendation, which uses the index-list
                # ranking path instead).
                pass
            elif task_id not in _TASKS_ALREADY_CONCRETE and self_check_llm is not None:
                grounding = _task_grounding(inst, task_id, bq, user_id)
                # For compose tasks, look up the target_app's AppPersona so
                # _length_guidance can use the user's per-app length_band
                # instead of the generic 1–4 sentence default.
                ap_for_len: dict | None = None
                if task_id in _COMPOSE_TASKS:
                    target_app = (inst.get("target_app") or "").lower()
                    try:
                        prof = bq._load_profile(user_id) if hasattr(bq, "_load_profile") else {}
                        for k, v in (prof.get("app_personas") or {}).items():
                            if isinstance(v, dict) and k.lower() == target_app:
                                ap_for_len = v
                                break
                    except Exception:
                        ap_for_len = None
                generated = _generate_example_response(
                    self_check_llm, task_id, user_query, grounding=grounding,
                    inst=inst, app_persona=ap_for_len,
                )
                if generated:
                    example = generated
                    inst["example_response"] = example
                    n_example_llm_gen += 1

            if not example:
                continue

            # Heuristic voice-evidence span extraction for compose tasks.
            # Runs independently of inferior generation so the renderer can
            # bold the gold's voice anchors even when no foil exists.
            if task_id in _VOICE_EVIDENCE_TASKS:
                t_test_v = int(inst.get("t_test") or 0)
                if t_test_v not in _ctx_cache:
                    _ctx_cache[t_test_v] = _build_persona_ctx(bq, user_id, t_test_v)
                _voice_ctx = _ctx_cache[t_test_v]
                _uv_block = (_voice_ctx or {}).get("user_voice") if isinstance(_voice_ctx, dict) else {}
                _spans = _extract_voice_evidence_spans(example, _uv_block or {})
                if _spans:
                    inst["example_response_voice_evidence"] = _spans

            # Workstream I: self-check. Skip for deterministic-gold tasks —
            # the gold is structured JSON, not natural-language prose, so
            # the prose-oriented self-check would always score low.
            if (self_check_llm is not None
                    and task_id not in _DETERMINISTIC_GOLD_TASKS):
                user_query = inst.get("user_query") or inst.get("query") or ""
                check = _run_self_check(self_check_llm, task_id, user_query, example)
                inst["example_response_self_check"] = check
                n_self_check += 1
                if not check.get("passed", True):
                    n_self_check_failed += 1

            # Workstream J: inferior_response. Skip for chatbot restraint-arm
            # instances (control/adversarial/stale) where the gold response
            # is intentionally generic — no inferior pair makes sense.
            arm = inst.get("arm") or "proactive"
            if (arm in ("proactive", "contradiction")
                    and task_id not in _TASKS_NO_FOIL):
                # Family 0 (deterministic-gold tasks) — extension point.
                # Set is empty today (agentic_moment_recommendation merged
                # into personalized_recommendation which uses the ranking
                # foil path below). Future deterministic-JSON tasks plug in
                # here by adding to _DETERMINISTIC_GOLD_TASKS and dispatching.
                if task_id in _DETERMINISTIC_GOLD_TASKS:
                    pass
                # Family 1 (ranking) — deterministic inverted ordering, no
                # LLM call. Same `Ranked indexes: [...]` wrapper as the
                # example, identical length, just a different (bad) order.
                elif task_id in _RANKING_TASKS:
                    inferior_text = _compute_ranking_inferior(inst, task_id)
                    if inferior_text:
                        inst["inferior_response"] = {
                            "text": inferior_text,
                            "flaw_kind": "ranking_inversion",
                            "flaw_evidence": {"_from": "deterministic_ranking_inversion"},
                        }
                        n_inferior_built += 1
                    else:
                        n_inferior_skipped += 1
                # Family 2/3/4 — LLM rewrite path with per-flaw instruction
                # + similarity validator + 3-attempt retry loop.
                elif inferior_llm is not None:
                    t_test = int(inst.get("t_test") or 0)
                    if t_test not in _ctx_cache:
                        _ctx_cache[t_test] = _build_persona_ctx(bq, user_id, t_test)
                    ctx = _ctx_cache[t_test]
                    allowed_flaws = _TASK_FLAW_KINDS.get(task_id, _FLAW_KINDS_PERSONALIZATION)
                    flaw_kind = allowed_flaws[(idx + len(items)) % len(allowed_flaws)]
                    evidence = _pick_flaw_evidence(flaw_kind, inst, ctx, rng, task_id)
                    if evidence is None:
                        for fk in allowed_flaws:
                            if fk == flaw_kind:
                                continue
                            evidence = _pick_flaw_evidence(fk, inst, ctx, rng, task_id)
                            if evidence is not None:
                                flaw_kind = fk
                                break
                    if evidence is None:
                        n_inferior_skipped += 1
                        continue
                    text = _generate_inferior(
                        inferior_llm, example, flaw_kind, evidence, task_id,
                        user_query=user_query,
                    )
                    if text:
                        inst["inferior_response"] = {
                            "text": text,
                            "flaw_kind": flaw_kind,
                            "flaw_evidence": evidence,
                        }
                        n_inferior_built += 1
                    else:
                        # Validator rejected all 3 attempts — drop the foil
                        # for this sample and tag for traceability. Eval
                        # harness handles missing inferior_response.
                        inst["inferior_drop_reason"] = "validator_failed_after_3_attempts"
                        n_inferior_skipped += 1

                # Workstream J': voice-evidence smoke test for compose tasks.
                # If a mini tier can't tell gold > foil given the bolded
                # voice anchors (already extracted above), the pair is too
                # similar on the voice axis; regen the inferior once then
                # accept whatever we have (no infinite loop).
                if (task_id in _VOICE_EVIDENCE_TASKS
                        and voice_check_llm is not None
                        and isinstance(inst.get("inferior_response"), dict)):
                    user_voice_block = ctx.get("user_voice") or {}
                    spans = inst.get("example_response_voice_evidence") or []
                    inferior_text = (inst["inferior_response"].get("text") or "").strip()
                    check = _verify_voice_evidence_distinguishability(
                        voice_check_llm, example, inferior_text,
                        user_voice_block, spans, rng,
                    )
                    inst["voice_evidence_smoke_check"] = check
                    if check.get("passed"):
                        n_voice_check_passed += 1
                    else:
                        n_voice_check_failed += 1
                        # One regen attempt: rebuild the inferior with the
                        # same flaw_kind + evidence; the gold stays put
                        # (the gold's bolded anchors are already accurate;
                        # what failed is the foil being too close on voice).
                        text2 = _generate_inferior(
                            inferior_llm, example, flaw_kind, evidence,
                            task_id, user_query=user_query,
                        )
                        if text2 and text2.strip() != inferior_text:
                            inst["inferior_response"]["text"] = text2
                            inst["inferior_response"]["regen_reason"] = "voice_evidence_smoke_failed"
                            n_voice_check_regen += 1
                            # Re-verify after regen; record the second check
                            # so the operator can see whether regen helped.
                            check2 = _verify_voice_evidence_distinguishability(
                                voice_check_llm, example, text2,
                                user_voice_block, spans, rng,
                            )
                            inst["voice_evidence_smoke_check_after_regen"] = check2

    if verbose:
        print(f"[llm_postprocess] example_llm_gen={n_example_llm_gen} "
              f"example_ranking={n_example_ranking} "
              f"self_check={n_self_check} self_check_failed={n_self_check_failed} "
              f"inferior_built={n_inferior_built} "
              f"inferior_skipped={n_inferior_skipped} "
              f"voice_check_passed={n_voice_check_passed} "
              f"voice_check_failed={n_voice_check_failed} "
              f"voice_check_regen={n_voice_check_regen}")
    bm["postprocess_stats"] = {
        "example_llm_gen": n_example_llm_gen,
        "example_ranking": n_example_ranking,
        "self_check_total": n_self_check,
        "self_check_failed": n_self_check_failed,
        "inferior_built": n_inferior_built,
        "inferior_skipped": n_inferior_skipped,
        "voice_check_passed": n_voice_check_passed,
        "voice_check_failed": n_voice_check_failed,
        "voice_check_regen": n_voice_check_regen,
    }
    return bm
