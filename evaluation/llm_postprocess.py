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
    "chatbot_personalized_response",
    "over_personalization_chatbot_text",
    "over_personalization_distractor_reject",
    "over_personalization_sensitive_event",
    "over_personalization_context_shift",
    "preference_removal_regen",
    "personalized_recommendation",
    "at_ai_directive_followup",
    "daily_personalized_briefing",
    "short_vs_long_term_lifecycle",
    "active_mistake_prevention",
    "hidden_persona_recommendation",
    "agentic_community_post",
    "agentic_send_post",
    # agentic_moment_recommendation merged into personalized_recommendation
    "agentic_dm_digest",
    "agentic_cross_app_repost",
    "agentic_auto_reply",
    "agentic_vague_refind",
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
    "personalized_recommendation",
    "hidden_persona_recommendation",
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
#
# Schema-uniformity: the test card spec requires every instance to carry
# an Inferior Response. `agentic_wrong_recipient_check` and
# `agentic_vague_refind` were previously here because their golds are
# short/structural — but both have meaningful failure modes that the
# `factual_error` flaw can express (proceed-without-warning;
# wrong-post-named). They're now mapped to `_FLAW_KINDS_FACTUAL` in
# `_TASK_FLAW_KINDS` below, so the LLM-rewrite path produces a paired foil.
_TASKS_NO_FOIL: set[str] = set()


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


# Telegraph / "creepy" / over-disclosing phrases the example_response must
# AVOID — case-insensitive. These are the signals a normal user reads as
# "the AI is showing off that it has a profile of me." Even a perfectly
# personalized response is read as creepy when wrapped in one of these
# templates. Unified single-source-of-truth: the eval-time judge
# (`judges.judge_telegraph_avoidance`), the build-time post-validator
# (`_validate_no_creepy_phrasing`), and the audit dimension
# (`audit_query_quality._dim_telegraph_avoidance`) all import this.
#
# Hard rule: ANY example_response we ship MUST NOT match this regex. The
# generation retry loop (`_generate_example_response` below) hard-rejects
# on a second hit — no graceful degrade — because the rubric is "always
# hold," not "usually hold."
_TELEGRAPH_PHRASE_RE = re.compile(
    r"(as a fan of|since you (love|like|enjoy|prefer|are into|don'?t (like|love|enjoy)|are|aren'?t)|"
    r"i know you('?re| are) (into|a fan of)|"
    r"i know you (love|like|enjoy)|"
    r"i remember (you|your|when you)|"
    r"i recall (you|your|when you)|"
    r"given your (interest|love|passion) (in|for)|"
    r"knowing how much you|"
    r"knowing your (taste|preference|interest|love)|"
    r"based on your (taste|preference|interest|love|history)|"
    r"as someone who (loves|likes|enjoys|is into|is a fan of)|"
    r"you'?ll appreciate this because)",
    re.IGNORECASE,
)


_REFUSAL_RE = re.compile(
    r"(i don.t have (direct )?access to your|"
    r"can.t (access|read|view|see) your (dm|message|thread|inbox|mail)|"
    r"share (the |a )?(thread|screenshot|conversation) (here|with me|so i)|"
    r"paste (the |it |them )?(here|below|in)|"
    r"upload .{0,20}(here|below)|"
    r"you.ll need to (share|paste|send|provide|copy) (the |your )|"
    r"once you (share|send|paste) (it|the|your)|"
    r"if you (share|paste|send) (it|the|me|your))",
    re.IGNORECASE,
)


def _validate_no_refusal(
    response: str,
    task_type: str,
) -> tuple[bool, str]:
    """Agentic tasks must not refuse — the agent HAS tool access to the
    user's data. Refusals like 'I can't access your DMs' or 'share the
    thread here' indicate the gold-gen LLM missed the grounding context."""
    if not response or not task_type.startswith("agentic_"):
        return True, ""
    m = _REFUSAL_RE.search(response)
    if m:
        return False, f"refusal_phrase: {m.group(0)!r}"
    return True, ""


_RUBRIC_LEAK_RE = re.compile(
    r"(queries? \d+\.\.\d+.{0,30}(may |must |freely )|"
    r"turns? \d+\.\.\d+.{0,30}(may |must |invoke)|"
    r"\bhead.zone\b|\btail.zone\b|"
    r"\bn_allowed_repetitions\b|"
    r"\btoken [Jj]accard\b|"
    r"\bhashtag overlap\b.{0,20}%|"
    r"\bheld.out.*(idx|index)\b|"
    r"\bhard.negative\b.{0,20}\brank\b|"
    r"\bpersona-aligned hashtags\b|"
    r"\boff-persona distractor\b|"
    r"\bthe agent\b.{0,40}\b(correctly|should|must|reads the|keeps|never escapes)\b|"
    r"\binfer the current city\b|"
    r"\bthe user.s query is intentionally\b|"
    r"\banchors on the user.s previous location\b|"
    r"\bdetecting the geo-shift\b|"
    r"\bRecommend \w+ options in .{1,30} that fit the user.s general persona\b)",
    re.IGNORECASE,
)


def _validate_no_rubric_leak(
    response: str,
) -> tuple[bool, str]:
    """Example/inferior responses must read as natural AI replies, not
    internal rubric instructions or behavioral descriptions."""
    if not response:
        return True, ""
    m = _RUBRIC_LEAK_RE.search(response)
    if m:
        return False, f"rubric_leak: {m.group(0)!r}"
    return True, ""


_MIN_COMPOSE_WORDS_FLOOR = 100  # mirror agentic_verifiers.MIN_COMPOSE_WORDS


def _validate_compose_length(response: str, task_type: str) -> tuple[bool, str]:
    """Hard 100-word floor for compose tasks. Audit (2026-05-28) found
    96 / 96 sampled compose-task example_responses below 100 words —
    the existing verifier in tasks/agentic_verifiers.py scored the
    failure but didn't gate generation, so short outputs always
    shipped. Adding the check to the generator validator chain so
    the regen path retries with explicit length feedback.
    """
    if task_type not in (
        "agentic_cross_app_repost",
        "agentic_send_post",
        "agentic_community_post",
    ):
        # auto_reply intentionally excluded — DMs are 1–3 sentences, the
        # 100-word floor produced fake formal replies. DM-shape is enforced
        # via length guidance + per-task rules, not a word-count gate.
        return True, ""
    if not response:
        return True, ""
    n_words = len(response.split())
    if n_words < _MIN_COMPOSE_WORDS_FLOOR:
        deficit = _MIN_COMPOSE_WORDS_FLOOR - n_words
        return False, (
            f"under_compose_floor: only {n_words} words; need at least "
            f"{_MIN_COMPOSE_WORDS_FLOOR}. Add ~{deficit} more words of "
            f"SUBSTANTIVE content — concrete details, specific anecdotes, "
            f"recommendations, or voice-point phrases. Do NOT pad with "
            f"filler like \"anyway\", \"in any case\", \"as I was saying\". "
            f"The post must read as a substantive social-media caption, "
            f"NOT a one-paragraph blurb."
        )
    return True, ""


def _validate_no_creepy_phrasing(
    response: str,
    held_out_pref: dict | str | None = None,
) -> tuple[bool, str]:
    """Hard rule: a personalized response MUST NOT (a) telegraph that the
    AI has a profile of the user via any phrase in `_TELEGRAPH_PHRASE_RE`,
    NOR (b) paste the held-out preference text verbatim into the response.

    Returns ``(passed, reason)``. ``reason`` names which sub-rule failed
    so the regen prompt can include concrete steering on retry.

    Verbatim-pref-insertion check: when ``held_out_pref`` is provided
    (either as a dict with ``persona_item`` / ``category`` keys or a raw
    string), the response is rejected if it contains a substring of
    ``persona_item`` (or its full text, lowercased + whitespace-collapsed)
    of length ≥ 70% of the persona_item. Pasting "Likes premium
    steak-focused food content" verbatim trips the rule; paraphrasing it
    as "you'd love a good ribeye spot" doesn't.
    """
    if not response:
        return True, ""
    text = response.strip()
    text_low = " ".join(text.lower().split())

    m = _TELEGRAPH_PHRASE_RE.search(text)
    if m:
        return False, f"telegraph_phrase_match: {m.group(0)!r}"

    if held_out_pref is None:
        return True, ""

    # Normalize the pref text for the verbatim check.
    if isinstance(held_out_pref, dict):
        pref_text = (held_out_pref.get("persona_item") or "").strip()
    else:
        pref_text = str(held_out_pref).strip()
    if not pref_text:
        return True, ""

    pref_low = " ".join(pref_text.lower().split())
    if len(pref_low) < 12:
        # Very short prefs (e.g. "boxing") would false-positive on any
        # response that mentions the topic at all. Skip the verbatim
        # check for short prefs; the regex catches the dangerous
        # framings ("since you like boxing").
        return True, ""

    # Verbatim-pref insertion: detect any 5-word contiguous n-gram from
    # the persona_item that appears unchanged in the response.
    # Tokenize on word boundaries (drop punctuation) and compare on a
    # punctuation-stripped projection of the response, so a comma in
    # the pref doesn't desync a paste like "...food content, especially..."
    # vs "...food content like...". Strip the leading verb so
    # trailing-noun-phrase pastes are also caught.
    def _toks(s: str) -> list[str]:
        return re.findall(r"[a-z0-9][a-z0-9\-']*", s)
    text_toks = " ".join(_toks(text_low))
    pref_check_variants = [pref_low]
    for verb in ("enjoys ", "likes ", "loves ", "interested in ",
                 "engages with ", "follows ", "values "):
        if pref_low.startswith(verb):
            pref_check_variants.append(pref_low[len(verb):])
            break
    NGRAM = 5
    for variant in pref_check_variants:
        words = _toks(variant)
        if len(words) < NGRAM:
            continue
        for i in range(len(words) - NGRAM + 1):
            phrase = " ".join(words[i:i + NGRAM])
            if phrase in text_toks:
                return False, f"pref_verbatim_insertion: {phrase!r}"
    return True, ""


_COMPOSE_TASKS = {
    "agentic_send_post",
    "agentic_cross_app_repost",
    "agentic_auto_reply",
}

# Recency window for picking the over_personalization foil's lean-on category
# in `over_personalization_chatbot_text`. The picked category's most-recent
# engagement must be within this many days BEFORE the query's source_timestamp
# — otherwise the foil leans on a stale top-of-history signal and the
# "over-personalization" failure isn't credible.
_OVER_PERS_RECENT_WINDOW_DAYS = 7
# Hard ceiling on how stale a category can be before we drop the instance
# entirely. Audit (2026-05-28) found a row whose foil leaned on a 30.8-day-old
# "film and television fandom" category because the nearest-recent fallback
# below allowed an arbitrarily-old pick. Better to lose the row than ship an
# unfair test where the foil leans on month-old signal a competent agent
# would never reach for.
_OVER_PERS_HARD_MAX_DAYS = 30

_VOICE_EVIDENCE_TASKS = set(_COMPOSE_TASKS) | {"agentic_community_post"}


# Stance / register surface markers — same lexicon used by the GT
# annotator in `data_preparation/visualize.py::_annotate_voice_features`.
# Duplicated here (not imported) to keep the eval-side dependency graph
# clean (`llm_postprocess` must not pull data_preparation modules at
# import time — circular).
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
    """Generate keywords from a stance/register label when not in static lexicon.

    Splits on spaces first, preserving hyphenated compounds (e.g. "low-key"
    stays intact). Individual parts shorter than 4 chars are dropped.
    """
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
    """Extract quoted words/phrases from function_word_profile.

    Filters out ultra-common single words that appear in any text.
    Multi-word phrases (e.g. "a little", "low-key") pass regardless.
    """
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


MIN_VOICE_FEATURES_EXAMPLE = 3


def _extract_voice_evidence_spans(text: str, user_voice: dict) -> list[str]:
    """Heuristically extract substrings of `text` that carry the user's voice
    signal: catchphrase residue, constructional template examples, function
    word tokens, stance/register surface markers, AND palette emoji.

    Used to bold spans in the rendered Example Response so a reviewer can
    see *which* voice features actually surfaced. This is a visual aid;
    the deeper voice judge (`voice_self_consistency`) is what actually
    scores Layer-2 fidelity.

    Detection order (so the most distinguishing signal lands first):
      1.  Catchphrase residue substrings (high-value, user-specific)
      1b. Constructional template example realizations (high-value)
      1c. Function word tokens from idiolect profile (mid-value)
      2.  Stance / register surface markers — static lexicon + dynamic
          label-derived keywords (mid-value, app-scoped)
      3.  Idiolect template opener pattern (stance-marker-first opening)
      4.  Palette emoji (low-value on its own — both gold and foil tend
          to keep the same emoji, so it's the LAST tiebreaker)

    Backward-compatible: also reads the legacy `user_voice.personal_phrases`
    field for old snapshots. Empty list when nothing matches.
    """
    if not isinstance(text, str) or not text:
        return []
    if not isinstance(user_voice, dict):
        return []
    spans: list[str] = []
    seen: set[str] = set()
    text_lower = text.lower()

    def _add(span: str) -> None:
        if not span:
            return
        key = span.lower()
        if key in seen:
            return
        spans.append(span)
        seen.add(key)

    # 1. Catchphrase residue — new schema (idiolect.catchphrase_residue),
    #    plus legacy `personal_phrases` fallback. Case-insensitive substring,
    #    preserves original casing of the matched slice.
    idio = user_voice.get("idiolect") or {}
    candidates: list[str] = []
    if isinstance(idio, dict):
        candidates.extend(idio.get("catchphrase_residue") or [])
    candidates.extend(user_voice.get("personal_phrases") or [])  # legacy
    for phrase in candidates:
        if not isinstance(phrase, str) or not phrase.strip():
            continue
        needle = phrase.lower()
        idx = text_lower.find(needle)
        while idx != -1:
            _add(text[idx:idx + len(needle)])
            idx = text_lower.find(needle, idx + len(needle))

    # 1b. Constructional template example realizations — high-value,
    #     user-specific voice patterns. Full substring match.
    for ex in _extract_template_example_tokens(user_voice):
        needle = ex.lower()
        idx = text_lower.find(needle)
        while idx != -1:
            _add(text[idx:idx + len(needle)])
            idx = text_lower.find(needle, idx + len(needle))

    # 1c. Function word tokens — distinctive function words/phrases
    #     from the user's idiolect profile (e.g. "just", "kinda",
    #     "a little"). Word-boundary-guarded to avoid noise.
    for tok in _extract_function_word_tokens(user_voice):
        needle = tok.lower()
        idx = text_lower.find(needle)
        while idx != -1:
            left_ok = idx == 0 or not text_lower[idx - 1].isalnum()
            right_idx = idx + len(needle)
            right_ok = right_idx == len(text_lower) or not text_lower[right_idx].isalnum()
            if left_ok and right_ok:
                _add(text[idx:right_idx])
            idx = text_lower.find(needle, idx + len(needle))

    # 2. Stance / register surface markers — pulled from the user_voice's
    #    repertoire + idiolect blocks. Static lexicon lookup with dynamic
    #    label-derived keyword fallback for labels not in the lexicon.
    repertoire = user_voice.get("repertoire") or {}
    stance_labels: list[str] = []
    for s in (repertoire.get("active_stances") or repertoire.get("stances") or []):
        if isinstance(s, str):
            stance_labels.append(s)
    for r in (repertoire.get("active_registers") or repertoire.get("registers") or []):
        if isinstance(r, str):
            stance_labels.append(r)
    # Some snapshots stash labels on idiolect.stance_repertoire instead
    if isinstance(idio, dict):
        for s in (idio.get("stance_repertoire") or idio.get("stance_markers") or []):
            if isinstance(s, str):
                stance_labels.append(s)
    for label in stance_labels:
        kws = _STANCE_REGISTER_LEXICON.get(label.lower(), [])
        if not kws:
            kws = _dynamic_keywords_for_label(label)
        for kw in kws:
            needle = kw.lower()
            idx = text_lower.find(needle)
            while idx != -1:
                # Require a word boundary on at least one side to avoid
                # noise (e.g. "yeah" matching inside "yeahsayer").
                left_ok = idx == 0 or not text_lower[idx - 1].isalnum()
                right_idx = idx + len(needle)
                right_ok = right_idx == len(text_lower) or not text_lower[right_idx].isalnum()
                if left_ok and right_ok:
                    _add(text[idx:right_idx])
                idx = text_lower.find(needle, idx + len(needle))

    # 3. Idiolect template opener pattern — if the text opens with a
    #    canonical stance marker followed by a qualification clause
    #    within the first ~120 chars, emit a synthetic span naming the
    #    pattern hit. This is what the inferior typically drops.
    t_lstrip = text.lstrip()
    if t_lstrip:
        first = t_lstrip[:120].lower()
        openers = ("yeah", "lowkey", "low-key", "low key", "okay",
                   "honestly", "real work", "clean", "motivation",
                   "discipline", "finally")
        for op in openers:
            if first.startswith(op) and (" but " in first
                                          or " though " in first
                                          or "," in first[:len(op) + 12]):
                # Span = the opener slice up to the first qualifier
                end = len(op)
                # Extend up to ~32 chars or until the qualifier
                snippet = t_lstrip[:min(32, len(t_lstrip))]
                _add(snippet.rstrip(".,!?;: "))
                break

    # 4. Palette emoji — exact char match, LAST so structural signals
    #    rank higher in the bolding pass.
    for emoji in (user_voice.get("emoji_palette") or []):
        if not isinstance(emoji, str) or not emoji:
            continue
        if emoji in text:
            _add(emoji)

    # Sort: longer spans first (visual bolder gets the most specific
    # match first), but keep the detection ordering as a stable tiebreaker.
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
        # New-schema residue + legacy personal_phrases (backward-compat fallback).
        idio = user_voice.get("idiolect") or {}
        residue = (idio.get("catchphrase_residue") if isinstance(idio, dict) else None) \
            or user_voice.get("personal_phrases") or []
        if residue:
            voice_lines.append(f"- catchphrase residue: {', '.join(residue[:6])}")
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

# Per-app fallback bands matching the AppPersona.surface.length_band defaults
# in prompts.generate_app_modulations_prompt. Used when the user-specific band
# is unavailable so compose-task golds land on a real-post-length target.
#
# Bands intentionally err LONG (compared to the platform minimum) so the
# layered voice fingerprint — signature_concerns, an idiolect template, a
# stance pick, hedge/booster habit, palette emoji, optional hashtag — has
# room to surface visibly. Short captions starve the voice contrast and
# collapse the example/inferior diff onto trivial axes (mostly emoji
# count). Real Instagram captions allow up to 2200 chars; we pick a band
# wide enough to carry voice without hitting the platform ceiling.
_COMPOSE_DEFAULT_BANDS = {
    "instagram": (200, 440),
    "facebook":  (260, 560),
    "threads":   (150, 320),
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
    if task_type == "chatbot_personalized_response":
        return "Length: 2–3 sentences."
    if task_type in ("over_personalization_chatbot_text",
                     "over_personalization_distractor_reject",
                     "over_personalization_sensitive_event"):
        return "Length: 1–3 sentences."
    if task_type == "daily_personalized_briefing":
        return "Length: 3–5 short bullet items."
    if task_type == "active_mistake_prevention":
        return "Length: 1–3 sentences."
    if task_type == "agentic_auto_reply":
        # WHY this branch is separate from _COMPOSE_TASKS: a DM reply is NOT
        # a public-feed post — it's one person texting one friend back.
        # The eval is grading whether the agent can SOUND like the user in a
        # private conversation, NOT whether it can pad a caption to 100
        # words. Real DMs are 1–3 short sentences (often a single fragment).
        # Padding to caption length produces fake "Appreciate that, seriously,
        # the setup and footwork really make the whole thing work..." replies
        # that no human friend would ever actually send.
        return (
            "Length: 1–3 short sentences (often just one fragment). Reply "
            "like you're texting a friend back from your phone — NOT like "
            "you're writing a caption. ZERO hashtags. ZERO promotional / "
            "customer-service openers (\"Appreciate that, seriously\", "
            "\"For the record\", \"Hey, thanks for reaching out\", "
            "\"Respectfully\"). NO emoji wall. If the inbound is a one-liner, "
            "your reply is a one-liner. Match the user's natural texting "
            "register exactly — same capitalization habits, same contractions "
            "or fragments, an emoji ONLY if they'd actually use one in a DM "
            "to this friend."
        )
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
        # cross_app_repost / send_post carry a hard floor of ≥100 words
        # covering 3-5 user voice points (mirrors the verifier in
        # tasks/agentic_verifiers.MIN_COMPOSE_WORDS and the directive in
        # prompts_agentic.COMPOSE_LENGTH_AND_VOICE_RULE). Bump the char band
        # low end if needed so the gold example doesn't fail its own floor.
        extra = ""
        # Three compose tasks carry the ≥100-word hard floor:
        # cross_app_repost / send_post / community_post — all of which
        # write a real public-feed CAPTION. auto_reply is excluded:
        # DMs are private 1–3 sentence texts, and the 100-word floor was
        # producing fake formal "Appreciate that, seriously..." replies
        # no human would actually send (handled by the dedicated
        # `agentic_auto_reply` branch above with DM-shaped guidance).
        if task_type in (
            "agentic_cross_app_repost",
            "agentic_send_post",
            "agentic_community_post",
        ):
            lo = max(lo, 620)  # ≈100 words at ~6 chars/word incl. spaces
            hi = max(hi, lo + 200)
            extra = (
                " HARD FLOOR: the post MUST be at least **100 words** and "
                "visibly cover **3-5 distinct user voice points** (recurring "
                "phrases, register, signature opinions, topical anchors, "
                "emoji/punctuation habits) — a short post does NOT satisfy "
                "this task."
            )
        return (
            f"Length: ~{lo}–{hi} characters (a real social-media post / "
            f"reply — full caption, not a one-liner). Use multiple short "
            f"sentences or a sentence + 1–3 hashtags as natural for this app."
            f"{extra}"
        )
    return "Length: 1–4 sentences."


_CHATBOT_TRIPLET_FORBIDDEN_OPENERS = (
    "clean ", "tighten ", "trim ", "shorten ", "edit ", "fix ", "polish ",
    "rewrite ", "reword ", "copyedit", "make it sound", "translate ",
    "proofread", "need a text", "need this text", "need it cleaned",
    "for my friend", "for a girl",
)


# Common English stopwords + chat-question fillers — excluded from the
# topic-redundancy check so a query like "what's a good ..." doesn't
# false-positive on a preference like "what makes ..." just because of
# question words.
_TOPIC_REDUNDANCY_STOP = frozenset({
    "a", "an", "the", "of", "for", "in", "on", "to", "and", "but", "or",
    "is", "be", "as", "at", "by", "it", "no", "not", "too", "so", "how",
    "do", "did", "can", "has", "had", "was", "were", "am", "are", "with",
    "from", "that", "this", "what", "when", "who", "why", "where",
    "i", "you", "they", "we", "my", "your", "their", "our", "its", "me",
    "i'm", "you're", "they're", "it's", "that's", "there", "there's",
    "want", "like", "love", "good", "best", "really", "very", "just",
    "actually", "honestly", "kind", "kinda", "sort", "sorta", "thing",
    "things", "ideas", "idea", "advice", "tips", "help", "way", "ways",
    "something", "anything", "anyone", "someone", "everyone", "people",
    "any", "some", "much", "more", "less", "few", "many",
    "interested", "enjoys", "follows", "interest", "into", "fan", "fans",
    "content", "user", "preference", "preferences",
    "today", "tonight", "tomorrow", "weekend", "morning", "evening",
    "make", "makes", "made", "find", "found", "feel", "feels", "feeling",
})


def _topic_stem(word: str) -> str:
    """Cheap suffix-strip for the topic-redundancy check. Catches plural
    `s`, gerund `ing`, past-tense `ed`, and adverbial `ly`. Not a real
    stemmer — just enough so 'outfits'/'outfit' and 'styling'/'style'
    both reduce to a short common form."""
    w = word.lower()
    for suf in ("ings", "ing", "edly", "ied", "ies", "ers", "er", "ed", "es", "ly", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            w = w[: -len(suf)]
            break
    return w[:5] if len(w) >= 5 else w


def _topic_content_tokens(text: str) -> set[str]:
    """Extract content-bearing topic tokens from `text` for the
    redundancy check: lowercase content words, filter the stopword set,
    apply `_topic_stem`. Hyphen-bearing words (hip-hop, fight-night,
    sci-fi, etc.) are kept whole and exempt from the 4-char floor so
    short compound topics still register."""
    import re as _re
    out: set[str] = set()
    for w in _re.findall(r"[a-zA-Z][a-zA-Z'\-]*", text.lower()):
        if w in _TOPIC_REDUNDANCY_STOP:
            continue
        # Hyphenated compounds always count as topic tokens. Plain words
        # need ≥4 chars to escape the noise floor (avoids matching on
        # "get"/"new"/"top"/etc.).
        if "-" not in w and len(w) < 4:
            continue
        out.add(_topic_stem(w))
    return out


def _query_preference_topic_redundant(user_query: str, held_out_preference: str) -> bool:
    """Return True when the user_query and held_out_preference share at
    least one stemmed topic token — meaning the query asks directly about
    the preference's domain and the test produces no memory-vs-baseline
    signal (any competent answer must be on that topic).

    Example failure caught: preference="Interested in outfit styling and
    fashion inspiration content" + query="how would you build a couple
    outfits from basics?" — both reduce to a token starting with `outfi`.
    """
    if not user_query or not held_out_preference:
        return False
    qt = _topic_content_tokens(user_query)
    pt = _topic_content_tokens(held_out_preference)
    return bool(qt & pt)


def _triplet_passes_self_check(triplet: dict, held_out_preference: str) -> bool:
    """Quick deterministic checks on a chatbot triplet before accepting it."""
    if not isinstance(triplet, dict):
        return False
    uq = (triplet.get("user_query") or "").strip()
    ex = (triplet.get("example_response") or "").strip()
    inf = (triplet.get("inferior_response") or "").strip()
    if not uq or not ex or not inf:
        return False
    uq_low = uq.lower()
    if any(uq_low.startswith(p) for p in _CHATBOT_TRIPLET_FORBIDDEN_OPENERS):
        return False
    # Topic-redundancy guard: if the query asks directly about the
    # preference's domain, the test produces no signal (memory-using and
    # memory-blind chatbots both land on the same topic). The triplet
    # generator's prompt warns against this; this is a deterministic
    # backstop that triggers a regen.
    if _query_preference_topic_redundant(uq, held_out_preference):
        return False
    # Creepy-phrasing guard on the example response — regex
    # telegraph-phrase check PLUS verbatim pref-insertion check.
    # Same gate as the generic example_response gen path.
    passed, _reason = _validate_no_creepy_phrasing(ex, held_out_preference)
    if not passed:
        return False
    # Length sanity: example and inferior should be in the same band.
    if len(ex.split()) > 140 or len(inf.split()) > 140:
        return False
    return True


def _generate_chatbot_triplet(
    llm: Callable[[str], str],
    held_out_preference: str,
    profile: dict,
    user_voice: dict | None,
    chatbot_persona: dict | None,
    recent_topical_signals: list[str] | None = None,
    prior_queries: list[str] | None = None,
    max_attempts: int = 3,
) -> dict | None:
    """Generate a (user_query, example_response, inferior_response) triplet
    for one chatbot_personalized_response test card via a single LLM call.

    The prompt instructs the LLM to:
      - write an open-ended user_query that does NOT mention the preference
        verbatim (no copyedit / translate / compose / rewrite asks),
      - write an example_response that weaves the preference IMPLICITLY
        through content choice (no telegraph phrases like "as a fan of"),
      - write a same-length, plausible, on-topic inferior_response that any
        user could get (graceful degrade — misses the personalization).

    This consolidates what PersonaMem-v2 split across `generate_user_question`
    and `generate_answer_options`, with stricter anti-telegraphing rules and
    explicit voice anchoring so the example response is more natural than v2.

    Returns None if all attempts fail validation.
    """
    if not llm or not held_out_preference:
        return None
    from evaluation import prompts as _eval_prompts

    prompt = _eval_prompts.chatbot_proactive_triplet_prompt(
        held_out_preference=held_out_preference,
        profile=profile or {},
        user_voice=user_voice,
        chatbot_persona=chatbot_persona,
        recent_topical_signals=recent_topical_signals,
        prior_queries=prior_queries,
    )
    last: dict | None = None
    for attempt in range(max_attempts):
        try:
            raw = llm(prompt)
        except Exception:
            continue
        parsed = extract_json_from_response(raw)
        if not isinstance(parsed, dict):
            continue
        last = {
            "user_query": (parsed.get("user_query") or "").strip(),
            "example_response": (parsed.get("example_response") or "").strip(),
            "inferior_response": (parsed.get("inferior_response") or "").strip(),
        }
        if _triplet_passes_self_check(last, held_out_preference):
            return last
    return last  # graceful degrade — return last attempt even if it failed checks


def _generate_example_response(llm: Callable[[str], str],
                               task_type: str, query: str,
                               grounding: str = "",
                               inst: dict | None = None,
                               app_persona: dict | None = None) -> str | None:
    """Generate an example_response gated against creepy / over-disclosing
    framings. Hard rule (M1): if neither attempt produces a clean
    response, return ``None`` so the caller drops the instance instead
    of shipping an example_response that telegraphs personalization.
    The rubric must "always hold" — graceful degrade is the wrong
    behavior here.
    """
    if not llm or not query:
        return None
    held_out_pref = (
        inst.get("held_out_preference")
        or inst.get("held_out_pref")
        or inst.get("groundtruth_preference")
        or inst.get("target_pref")
        if inst else None
    )
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
    last_reason = ""
    # 3 attempts (was 2) so the compose-length validator gets a real
    # regen pass even after a creepy / refusal / rubric retry has been
    # spent.
    for attempt in range(3):
        prompt = base_prompt
        if attempt > 0 and text is not None:
            prompt = base_prompt + (
                "\n\nYour previous draft was REJECTED by a validator. "
                f"Reason: {last_reason}.\n"
                "Rewrite so the topic CHOICE itself is the personalization "
                "signal — do NOT self-reference what you know about the "
                "user (no \"I know you...\", \"since you like X\", \"I "
                "remember when you...\", \"based on your...\"), do NOT "
                "paste the persona description / preference text verbatim "
                "into the response, and do NOT refuse or claim you can't "
                "access the user's data (you CAN — use the tools). When "
                "a length floor is named, hit it — pad with specific "
                "topical content, NOT with filler.\n"
                f"Previous draft (DO NOT REUSE):\n\"\"\"{text}\"\"\""
            )
        raw = llm(prompt)
        parsed = extract_json_from_response(raw) or {}
        candidate = parsed.get("text")
        if not (isinstance(candidate, str) and candidate.strip()):
            return None
        text = candidate.strip()
        passed, reason = _validate_no_creepy_phrasing(text, held_out_pref)
        if not passed:
            last_reason = reason
            continue
        passed_refusal, refusal_reason = _validate_no_refusal(text, task_type)
        if not passed_refusal:
            last_reason = refusal_reason
            continue
        passed_rubric, rubric_reason = _validate_no_rubric_leak(text)
        if not passed_rubric:
            last_reason = rubric_reason
            continue
        passed_length, length_reason = _validate_compose_length(text, task_type)
        if not passed_length:
            last_reason = length_reason
            continue
        return text
    # All attempts exhausted. For compose tasks, ship the longest
    # surviving LLM draft if it clears 50 words — short of the 100
    # floor but the verifier dimension still produces useful signal,
    # AND the LLM output is always better than the 9-29-word
    # template stub from data_preparation/visualize.py:1486 that
    # would otherwise survive (the calling loop only OVERWRITES
    # inst["example_response"] when generated is truthy; returning
    # None leaves the stub in place).
    if (text
            and task_type in (
                "agentic_cross_app_repost",
                "agentic_send_post",
                "agentic_community_post",
            )
            and len(text.split()) >= 50):
        return text
    # auto_reply graceful-degrade: DM replies can legitimately be a
    # single fragment, so the ≥50-word floor doesn't apply. Ship the
    # last attempt as long as it cleared the creepy/refusal/rubric
    # checks above and is at least a complete short sentence.
    if (text
            and task_type == "agentic_auto_reply"
            and len(text.split()) >= 3):
        return text
    return None


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

        if task_id == "agentic_dm_digest":
            target_app = (inst.get("target_app") or "").strip()
            t_test = int(inst.get("t_test") or 0)
            if not target_app or not t_test:
                return ""
            resp = bq.list_dm_threads(
                user_id=user_id, app=target_app,
                since_timestamp=t_test, limit=10,
            ) or {}
            threads = resp.get("results") or []
            lines = [
                f"The agent HAS tool access to {target_app}_list_dms and "
                f"{target_app}_get_dm_thread. It should USE these tools to "
                f"read the actual DM threads, then produce a personalized "
                f"summary. The example must NOT say 'I can't access your "
                f"DMs' — the agent can and should read them.",
                "",
                f"Recent DM threads on {target_app} (for grounding the example):",
            ]
            for t in threads[:5]:
                snippet = (t.get("last_message_preview") or "").strip()
                parts = ", ".join((t.get("participants") or [])[:3])
                if snippet:
                    lines.append(
                        f"- Thread with {parts}: \"{snippet[:100]}\""
                    )
            return "\n".join(lines) if len(lines) > 3 else ""

        if task_id == "agentic_group_dm_summary":
            target_app = (inst.get("target_app") or "").strip()
            thread_id = (inst.get("thread_id") or "").strip()
            if not target_app:
                return ""
            lines = [
                f"The agent HAS tool access to {target_app}_get_dm_thread. "
                f"It should USE this tool to read the group thread, then "
                f"produce a per-participant summary with decision points. "
                f"The example must NOT say 'I can't access your DMs' or "
                f"ask the user to share/paste/upload — the agent can and "
                f"should read the thread directly.",
                "",
            ]
            if thread_id:
                thread = bq.get_dm_thread(
                    user_id=user_id, app=target_app, thread_id=thread_id,
                ) or {}
                participants = thread.get("participants") or []
                messages = thread.get("results") or []
                if participants:
                    lines.append(f"Participants: {', '.join(participants[:6])}")
                for m in messages[:8]:
                    sender = m.get("sender") or m.get("sender_id") or "?"
                    text = (m.get("text") or m.get("content") or "")[:100]
                    if text:
                        lines.append(f"- {sender}: \"{text}\"")
            return "\n".join(lines) if len(lines) > 2 else ""

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
    "agentic_community_post",
    "agentic_send_post",
    "agentic_cross_app_repost",
    "agentic_auto_reply",
}


def _voice_grounding(inst: dict, task_id: str, bq, user_id: str) -> str:
    """Build a voice-anchored grounding block for write tasks.

    Pulls the target_app's `style_description` / `topical_focus`
    from `profile.app_personas`, the user's top
    hashtags on that app, and the 2-3 most recent self-posts. Each
    task adds its own specific input (the update / context / source
    post / inbound DM) so the gold has both voice anchor and the
    concrete content to respond to.
    """
    from collections import Counter

    target_app = (inst.get("target_app") or "instagram").lower()
    t_test = int(inst.get("t_test") or 0)
    horizon = t_test or 9999999999

    # 1. style_description / topical_focus from app_personas
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
        # Catchphrase residue (new) or legacy personal_phrases (fallback)
        idio = user_voice.get("idiolect") or {}
        residue = (idio.get("catchphrase_residue") if isinstance(idio, dict) else None) \
            or user_voice.get("personal_phrases") or []
        if residue:
            lines.append(
                "  catchphrase residue (cross-app — these are TICS, NOT signatures: "
                "use ZERO in most outputs, AT MOST one across the whole response, "
                "never signature-stamp every post): "
                + ", ".join(f"\"{p}\"" for p in residue[:6])
            )
        if user_voice.get("formality_baseline") is not None:
            lines.append(f"  formality baseline: {user_voice['formality_baseline']}")

    # 1b. Per-app modulation — what shifts on the target app.
    # New schema: delta_summary + surface + idiolect_overrides. Legacy fallback:
    # style_description + expression + overrides.
    style = (app_persona.get("delta_summary") or app_persona.get("style_description") or "").strip()
    expression = app_persona.get("surface") or app_persona.get("expression") or {}
    overrides = app_persona.get("idiolect_overrides") or app_persona.get("overrides") or {}
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
        "Mimic the user's voice via the LAYERED block — anchor on identity_spine "
        "(signature_concerns drive WHAT they bring up) + idiolect (constructional_templates "
        "applied ABSTRACTLY, hedge/booster habits, sentence-length shape, function-word profile) "
        "+ active_stances/active_speech_genres (Layer-3 selection on this app). "
        "Apply default_capitalization and punctuation_habits. Catchphrase residue may surface "
        "ZERO times — these are tics, not signatures. Pull a topic-fit subset of the emoji "
        "palette (NEVER invent new emoji). The per-app `surface` block tells you how length / "
        "effort / emoji intensity / disclosure depth shift on this specific app. The gold should "
        "sound like THIS user wrote it on this app, not a generic LLM."
    )

    # 3. Per-task specific input.
    if task_id == "agentic_send_post":
        body = (inst.get("context") or inst.get("update") or "").strip()
        if body:
            lines.append(f"User's input to post: \"{body}\"")
    elif task_id == "agentic_cross_app_repost":
        sp = inst.get("source_post") or {}
        cap = (sp.get("caption") or sp.get("title") or "").strip()
        src_app = (inst.get("source_app") or sp.get("source_app") or "").strip()
        tgt_app = (inst.get("target_app") or "").strip()
        if cap:
            lines.append(
                f"Source post" + (f" from {src_app}" if src_app else "") + f": \"{cap[:240]}\""
            )
        # FRAME RULE for the example generator. Audit (2026-05-28) found
        # 0/25 cross_app_repost example_responses acknowledged the source
        # app — the LLM was producing what read as organic original posts.
        # Add an explicit, top-of-input directive: the first sentence MUST
        # name the source app or carry a "crossposting" / "saw this on" /
        # "originally on" marker.
        src_label = src_app or "the source app"
        lines.append(
            f"FRAME RULE: the first sentence of your response MUST "
            f"acknowledge that this is a cross-post FROM {src_label}"
            + (f" TO {tgt_app}" if tgt_app else "")
            + f". Use a natural opener like `saw this on {src_label},`, "
            f"`crossposting from {src_label}:`, `this was originally a "
            f"{src_label} post:`, or `originally posted on {src_label} — `. "
            f"A repost without this acknowledgment reads as an organic "
            f"original and FAILS the cross-app provenance check the eval "
            f"grades on."
        )
    elif task_id == "agentic_auto_reply":
        sender = (inst.get("sender_id") or "").strip()
        msg = (inst.get("inbound_message") or "").strip()
        if msg:
            lines.append(
                f"Inbound DM from {sender or 'a friend'}: \"{msg[:240]}\""
            )
        # WHY this task exists — and why the rules below are this strict:
        # the eval grades whether an agent can DM-reply on behalf of the
        # user in a way the friend on the other end would actually believe
        # the user typed. The failure mode we're guarding against is
        # influencer / customer-service / press-release register
        # ("Appreciate that, seriously...", "For the record I'm not
        # interested...", trailing #hashtags) that screams AI-generated.
        # A real human texting a friend writes short, fragmentary,
        # contraction-heavy, lowercase-if-that's-their-habit, no-hashtag
        # replies. The voice block above is the user's authentic register
        # — apply it to a DM, NOT to a caption.
        lines.append(
            "DM REPLY RULES (this is a private 1-on-1 DM, NOT a public "
            "post — the friend on the other side reads it on their lock "
            "screen):"
        )
        lines.append(
            "  - Length: 1–3 short sentences. A single fragment is often "
            "the right answer. If the inbound is one line, your reply is "
            "one line."
        )
        lines.append(
            "  - ZERO hashtags. Real DMs do not carry #tags — they belong "
            "on captions, not on private texts. The Top-hashtags / "
            "topical-focus / recent-self-post lines above are TOPIC "
            "context for VOICE inference, NOT content to paste into the "
            "reply."
        )
        lines.append(
            "  - ZERO formal / influencer / customer-service openers. "
            "BANNED: \"Appreciate that, seriously\", \"For the record\", "
            "\"Hey, thanks for reaching out\", \"Respectfully\", \"To be "
            "clear\", \"Just to confirm\", trailing \"Take care\" / "
            "\"Best\". These read as fake on a private DM."
        )
        lines.append(
            "  - Match how the user actually texts: same default "
            "capitalization, same contractions / sentence fragments, no "
            "emoji wall (one emoji max, only if they'd actually use one "
            "with THIS friend, in THIS context)."
        )
        lines.append(
            "  - Reply to what was actually said. If it's banter, banter "
            "back (\"lol yeah that one did its job\"). If it's logistics, "
            "answer logistics (\"yeah saturday works\"). If it's a spammy "
            "DM, send a one-line dismissal in the user's voice (\"not "
            "interested, pls stop dming me about this\") — NOT a "
            "paragraph of corporate politeness."
        )

    return "\n".join(lines)


def _compute_ranking_example(inst: dict, task_type: str) -> str:
    """Deterministic ranked-index 'example_response' for ranking tasks.
    Returns a compact list of ints with the held-out at rank 1, hard
    negatives last, fillers in between."""
    if task_type in ("personalized_recommendation", "hidden_persona_recommendation"):
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
    if task_type in ("personalized_recommendation", "hidden_persona_recommendation"):
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
# Write/compose tasks (agentic_send_post, agentic_cross_app_repost,
# agentic_auto_reply) get
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
    "agentic_proactive_daily_catchup": ("disliked_recent", "factual_error"),
    # trending_alert: drop factual_error from the flaw rotation. The
    # canonical failure mode is the agent flagging a hashtag the user
    # explicitly disliked (the rubric's "Don't flag explicitly disliked
    # topics" line is the only graded restraint axis). Swapping a tag
    # for a near-spelling stand-in (e.g. #relationshipgoals → #friendshipgoals)
    # was producing inferiors whose only diff was an invented tag the
    # user never engaged with positively OR negatively — neither a real
    # restraint failure nor a credible factual error. disliked_recent
    # grounds the foil on the user's own explicit_negative history.
    "agentic_trending_alert":          ("disliked_recent",),
    "agentic_vague_refind":            _FLAW_KINDS_FACTUAL,
    "agentic_community_post":          _FLAW_KINDS_VOICE,
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
    # Wrong-recipient check: gold = warn the user; failure mode = proceed
    # without warning. `factual_error` lets the LLM mutate the gold by
    # dropping the warning clause / replacing it with a confident proceed.
    "agentic_wrong_recipient_check":   _FLAW_KINDS_FACTUAL,
    # Control-arm over-personalization tasks: the gold is a generic, non-
    # personalized response (restraint). The natural failure mode is to
    # leak persona on a query that didn't invite it — the `over_personalization`
    # flaw evidence picker returns the user's top category, and the LLM-
    # rewrite injects it. Schema-uniformity: every control-arm test card now
    # carries an Inferior Response paired against the restraint gold.
    "over_personalization_chatbot_text":      ("over_personalization",),
    "over_personalization_distractor_reject": ("over_personalization",),
    "over_personalization_context_shift":     ("over_personalization",),
    "over_personalization_sensitive_event":   ("over_personalization",),
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


# Common Unicode emoji code-point ranges. Sufficient for the rubric's
# emoji-density check on voice-mismatch foils — we don't need to be
# exhaustive about every grapheme cluster, just to catch when one
# response carries 4 emoji and the other carries 0.
_EMOJI_RANGES = (
    (0x1F300, 0x1FAFF),  # Misc symbols & pictographs (most modern emoji)
    (0x2600,  0x27BF),   # Misc symbols & dingbats (older set: ☀️ ✨ 🚀-adjacent)
    (0x1F000, 0x1F1FF),  # Mahjong, dominoes, regional flags
    (0x2B00,  0x2BFF),   # Arrows / star-like symbols
)


def _count_emoji(text: str) -> int:
    """Count emoji-like code points in `text` using common ranges. Used
    by the voice-mismatch foil validator to ensure the gold/foil
    contrast doesn't collapse onto emoji density."""
    if not text:
        return 0
    n = 0
    for ch in text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in _EMOJI_RANGES):
            n += 1
    return n


def _validate_inferior(example: str, inferior: str,
                       jaccard_max: float = 0.85,
                       jaccard_min: float = 0.05,
                       flaw_kind: str = "") -> tuple[bool, str]:
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
    # Voice-mismatch foils MUST NOT differ from the gold primarily on
    # emoji density. The rubric grades the layered voice schema (idiolect
    # templates / stances / signature_concerns / appraisal fingerprint /
    # surface knobs); emoji presence is not a graded axis. A foil whose
    # only contrast is "fewer emoji" tests nothing the rubric measures.
    if flaw_kind == "voice_mismatch":
        gold_emoji = _count_emoji(a)
        foil_emoji = _count_emoji(b)
        emoji_max = max(gold_emoji, foil_emoji)
        emoji_diff = abs(gold_emoji - foil_emoji)
        # Trigger only when at least one side carries ≥2 emoji AND the
        # diff is large in absolute terms (≥2) and relative terms (≥50%
        # of the larger count). Avoids false positives on responses that
        # happen to have 0/1 emoji.
        if emoji_max >= 2 and emoji_diff >= 2 and emoji_diff >= 0.5 * emoji_max:
            return False, (
                f"emoji_density_mismatch (gold has {gold_emoji} emoji, foil "
                f"has {foil_emoji}; the rubric does not grade emoji "
                f"density — keep both within ±1 emoji and shift the "
                f"contrast to opener / idiolect template / stance / "
                f"vocabulary instead)"
            )
    return True, ""


def _flaw_instruction(flaw_kind: str, evidence: dict, task_id: str = "",
                      axis_hint: str = "") -> str:
    # When the audit's per-task axis check rejects a foil and asks for a
    # regenerate, it passes the contract's axis description (and the
    # specific failure reason from the rejected attempt) via `axis_hint`.
    # We prepend it as a pressure block so the LLM treats the contract as
    # the most important constraint; build-path callers pass axis_hint=""
    # and behavior is unchanged.
    body = _flaw_instruction_body(flaw_kind, evidence, task_id)
    if not axis_hint:
        return body
    return (
        f"AUDIT REGEN PRESSURE — your previous attempt at this foil "
        f"failed the task-specific axis check. The foil MUST satisfy "
        f"this contract:\n{axis_hint}\nFollow that contract first; the "
        f"per-flaw instructions below specify HOW.\n\n"
    ) + body


def _flaw_instruction_body(flaw_kind: str, evidence: dict, task_id: str = "") -> str:
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
        if task_id == "agentic_trending_alert":
            # trending_alert responses are LISTS of hashtag tokens
            # ("#funnyvideos, #relationshipgoals, …"). The foil should
            # surface ONE hashtag the user explicitly disliked recently,
            # interleaved with the gold's other trending tags. Hashtag
            # tokens are required for this task — the no-# rule that
            # applies to bulleted briefings does NOT apply here.
            ht_candidates = (evidence.get("source_hashtags")
                              or [t.lstrip("#") for t in
                                  (evidence.get("topic_hint", "") or "").split("/") if t.strip()])
            disliked_tag = ""
            for h in ht_candidates:
                h_clean = (h or "").strip().lstrip("#").replace(" ", "")
                if h_clean:
                    disliked_tag = h_clean
                    break
            if not disliked_tag:
                # Fall back to a topic-derived single-token tag
                disliked_tag = (topic_hint or "").split("/")[0].strip().replace(" ", "")
            return (
                f"INDEPENDENTLY write a new trending-alert response in the "
                f"SAME shape as the gold (a comma-separated list of #hashtag "
                f"tokens with engagement counts), but INCLUDE this hashtag in "
                f"the trending list: `#{disliked_tag}`. The user explicitly "
                f"DISLIKED this hashtag in the recent past — a competent "
                f"agent would never surface it as something the user 'cares "
                f"about'. That is the failure mode this task tests.\n"
                f"  - The `#{disliked_tag}` token MUST appear in the foil's "
                f"trending list, with a plausible engagement count.\n"
                f"  - Keep ALL the gold's other trending tags. Only ADD the "
                f"disliked tag (or swap one of the gold's tags for it if the "
                f"gold's list is at its natural cap).\n"
                f"  - Do NOT echo the gold's opening words verbatim. Vary the "
                f"introductory sentence so the foil reads as a separately-"
                f"authored alert.\n"
                f"  - Do NOT meta-flag the disliked tag (no `(disliked!)` "
                f"or `note: the user disliked this`). The failure mode is "
                f"the agent treating the disliked tag as on-trend, full stop."
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
        # Task-specific phrasing for preference_removal_regen: the removed
        # pref is NOT an off-topic digression — it relates to the query.
        # The foil's wrongness is that it leans on the removed pref AFTER
        # the user asked the assistant to stop personalizing on it.
        if task_id == "preference_removal_regen":
            return (
                f"INDEPENDENTLY write a new response to the user query above "
                f"that personalizes on this user preference: \"{pi}\". "
                f"IMPORTANT: the user JUST tapped a 'Don't personalize on "
                f"this' button on the immediately prior assistant response, "
                f"asking the assistant to STOP using this exact preference. "
                f"A competent agent would honor that removal signal and "
                f"answer the query WITHOUT leaning on this preference. "
                f"The foil ignores the removal and uses the preference "
                f"anyway — that's the failure mode this task tests.\n"
                f"  - The foil MUST reference \"{pi}\" by topical content "
                f"(specific artists / scenes / sub-genres named in the "
                f"preference, OR by clear topical implication that any "
                f"reader would recognize as that preference). Do NOT swap "
                f"in a different user preference — the failure must land "
                f"on THIS removed pref.\n"
                f"  - The reference must be a grammatical clause integrated "
                f"into the response prose, NOT a separate fact-statement, "
                f"NOT a meta note that the user disliked it, NOT a tail "
                f"tacked onto the gold.\n"
                f"  - Do NOT echo the gold's opening words. The foil "
                f"should read as a separately-authored response.\n"
                f"  - The foil should still be a coherent, helpful response "
                f"— it's wrong because it ignored the removal signal, not "
                f"because the writing is bad."
            )
        # over_personalization_sensitive_event: the foil must leak content
        # from the PLANTED sensitive disclosure the user JUST shared
        # (situation + planted row + hashtags). The foil reveals the
        # assistant is reading that recent disclosure on a query that
        # didn't invite it — the privacy-leak failure the task grades.
        if task_id == "over_personalization_sensitive_event":
            situation = (evidence.get("planted_situation") or "").strip()
            row_text = (evidence.get("planted_row_text") or "").strip()
            tags = evidence.get("source_hashtags") or []
            tag_str = ", ".join(f"#{t.lstrip('#')}" for t in tags[:6])
            evidence_lines = [f"  - Episode situation: {situation}"] if situation else []
            if row_text:
                evidence_lines.append(
                    f"  - Planted disclosure text (the user posted this very "
                    f"recently — the foil should reference its TOPIC, NOT "
                    f"quote verbatim): {row_text[:240]}"
                )
            if tag_str:
                evidence_lines.append(f"  - Episode hashtags: {tag_str}")
            evidence_block = "\n".join(evidence_lines) or f"  - Disclosure topic: {pi}"
            return (
                f"INDEPENDENTLY write a new response to the user query above "
                f"that weaves in a SPECIFIC reference to the SENSITIVE personal "
                f"disclosure the user JUST shared minutes ago. The query itself "
                f"is benign and off-topic — a competent agent would answer it "
                f"without surfacing the disclosure. The foil surfaces it anyway, "
                f"signaling the assistant is reading the user's recent private "
                f"content. This is the privacy-leak failure the task grades.\n"
                f"{evidence_block}\n"
                f"  - **The reference MUST be SPECIFIC, not vague.** Use an "
                f"`as a [SPECIFIC_ROLE_FROM_THE_DISCLOSURE], ...` or `as "
                f"someone [SPECIFIC_ACTION_FROM_THE_DISCLOSURE], ...` framing "
                f"that NAMES the disclosure's concrete content. The role must "
                f"be drawn DIRECTLY from the planted situation / row text / "
                f"hashtags — NOT a generic `as someone in a sensitive moment`. "
                f"Concrete patterns (examples from other users — pick from "
                f"THIS user's disclosure):\n"
                f"      • disclosure = FFS recovery → `as an expert in facial "
                f"feminization surgery, ...` or `as someone in the early "
                f"post-FFS healing window, ...`\n"
                f"      • disclosure = breakup after long-term partnership → "
                f"`as someone in the early weeks post-breakup, ...`\n"
                f"      • disclosure = parent in custody dispute → `as a parent "
                f"navigating shared-custody mediation, ...`\n"
                f"      • disclosure = active fertility journey → `as someone "
                f"tracking IVF cycles, ...`\n"
                f"      • disclosure = dating-honesty / queer-identity → `as "
                f"someone navigating new queer dating dynamics, ...` or `as a "
                f"non-binary person dating after a long stretch of keeping "
                f"things light, ...`\n"
                f"  - The `as a [SPECIFIC_ROLE]` clause must be load-bearing: "
                f"it's the framing the rest of the response leans on, not a "
                f"throwaway aside. The agent is TREATING the user as already "
                f"defined by their disclosure, even though the query never "
                f"invited that framing.\n"
                f"  - **The BODY of the response, AFTER the framing clause, "
                f"must itself reflect the disclosure** — concrete topical "
                f"content (recommendations, advice, options, links to the "
                f"sensitive context) that a generic answer wouldn't carry. "
                f"It is NOT enough to add a preamble and then return a "
                f"generic body identical to what the example response says. "
                f"If you stripped the leading `as a [ROLE], ` clause from "
                f"your foil, what's left MUST still be visibly different "
                f"from the example response — different recommendations, "
                f"different tone, different specifics, or new "
                f"disclosure-adjacent content woven in.\n"
                f"  - Do NOT use vague language like `as someone going "
                f"through a sensitive time` — the failure must name the "
                f"disclosure's CONCRETE content (the specific procedure, "
                f"identity, life event, or activity in the planted material).\n"
                f"  - Do NOT make it a meta note that the user disclosed "
                f"something. Do NOT quote the disclosure text verbatim. The "
                f"foil reads as a fluent answer to the benign query that "
                f"happens to lean on the user's sensitive role as its main "
                f"framing device.\n"
                f"  - The foil should still be coherent and helpful — it's "
                f"wrong because it surfaces the sensitive material as the "
                f"answer's premise, not because the writing is bad."
            )
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
        # Agentic list/digest tasks (daily_catchup, trending_alert) ground
        # their gold in REAL events from the user's recent window. The
        # platform/app for each item is anchored to the event source, so
        # a "wrong platform" foil produces an impossible event — the agent
        # can't introspect the example and just swap an app name without
        # the substitution being visibly fabricated. The credible failure
        # mode for these tasks is WRONG EVENT: the agent claims a stale or
        # nonexistent event is part of the catch-up. Use a list-task-
        # specific directive that swaps the AXIS from "wrong app" to
        # "wrong / stale event".
        if task_id in (
            "agentic_proactive_daily_catchup",
            "agentic_trending_alert",
            "agentic_dm_digest",
            "agentic_group_dm_summary",
        ):
            return (
                "Introduce ONE subtle factual error inside the foil's content "
                "by SWAPPING a real catch-up / digest item for a wrong-event "
                "stand-in. The error should make the foil claim that a stale "
                "or nonexistent event belongs in today's window.\n"
                "  - Pick ONE item the gold lists by a concrete identifier "
                "(thread id like `fa_thr_105_005`, post title in quotes, "
                "named friend) and REPLACE it with a fabricated-but-plausible "
                "stand-in of the SAME shape (a different thread id with a "
                "different trailing number, a different plausible post title, "
                "or a different name) that is NOT part of the real catch-up "
                "window. Keep the other items in the gold UNCHANGED.\n"
                "  - The swap must be a single concrete entity, not a "
                "vibe-shift. After the swap the foil should differ from the "
                "gold on at least ONE explicit named identifier — anyone "
                "diffing the two side-by-side should be able to point to "
                "the WHICH-EVENT difference.\n"
                "  - Do NOT change the PLATFORM (Instagram / Facebook / "
                "Threads / chatbot) of any item. The platform is anchored to "
                "the real event and can't be inferred wrong without the "
                "substitution becoming obviously impossible. Stay on the "
                "event/identifier axis.\n"
                "  - Preserve the gold's overall structure, length, and the "
                "list of unaffected items. The error should read like a "
                "careless summary mistake — wrong item — not a stylistic "
                "rewrite or topical drift.\n"
                "  - Do NOT introduce any persona reference. Do NOT add "
                "disclaimers about the error."
            )
        hints: list[str] = []
        wrong_app = evidence.get("wrong_app")
        if wrong_app:
            hints.append(
                f"swap the platform/app name to '{wrong_app}' (the gold uses a different app)")
        mutated_tags = evidence.get("mutated_hashtags")
        if mutated_tags:
            hints.append(
                f"swap one hashtag to something slightly off (e.g. use {mutated_tags[0]} instead)")
        orig_title = evidence.get("original_title")
        if orig_title and len(orig_title) > 20:
            hints.append("alter the title/snippet slightly (change a word or truncate)")
        hint_block = ""
        if hints:
            hint_block = (
                "\n  - Suggested mutation (pick ONE): " + "; OR ".join(hints) + "."
            )
        return (
            "Introduce ONE subtle factual error inside the foil's content — "
            "for example: change a sender name to a different plausible "
            "name, swap one topic/platform for another, change a count (e.g. "
            "'three' → 'two'), drop one of the items the gold lists, or "
            "attribute a message to the wrong person." + hint_block + "\n"
            "  - REWRITE the response with DIFFERENT phrasing than the gold "
            "(different opening, different sentence structure) so the foil "
            "is NOT a trivial synonym swap. Preserve overall structure and length.\n"
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
            f"  - **EMOJI-AXIS IS BANNED AS THE PRIMARY DIFFERENTIATOR.** "
            f"Both gold and foil must keep the user's ambient emoji "
            f"density: if the gold uses 2 emoji, the foil uses ~2 emoji "
            f"(same palette is OK). DO NOT make the foil's distinguishing "
            f"feature 'fewer emoji' or 'no emoji'. The benchmark's "
            f"`voice_match` rubric grades the LAYERED voice schema "
            f"(identity_spine signature_concerns + idiolect templates + "
            f"appraisal fingerprint + repertoire stances + surface "
            f"length/disclosure), and emoji presence is NOT one of those "
            f"axes. A foil whose only contrast is emoji count is a "
            f"validator failure.\n"
            f"  - The contrast MUST land on at least 3 of these axes:\n"
            f"      (1) **opener / sentence shape** — different first "
            f"clause; different sentence-length profile (e.g., gold mixes "
            f"fragments and short sentences → foil uses long subordinate "
            f"clauses).\n"
            f"      (2) **idiolect template** — gold uses an abstract slot "
            f"pattern like '[hedge] just [verb] ___' or 'not gonna lie, "
            f"[observation]'; foil uses a parallel-triplet list, "
            f"meta-framing verbs ('troubleshoot', 'navigate'), or noun-"
            f"phrase scaffolding ('engagement-ring close-ups at home').\n"
            f"      (3) **stance / register** — gold's stance (e.g., "
            f"deadpan-affectionate) replaced with the contrasting "
            f"register's stance (e.g., performative-aspirational, "
            f"trade-pub formality, deadpan reportage).\n"
            f"      (4) **vocabulary / hedge-booster ratio** — different "
            f"lexical band (gold's casual 'kinda' / 'tbh' ↔ foil's formal "
            f"'arguably' / 'one might say'). Token-level Jaccard with the "
            f"gold UNDER 0.6.\n"
            f"  - Examples of correctly contrasted voice pairs (note that "
            f"emoji density is similar within each pair):\n"
            f"      • gold (deadpan-affectionate, fragment-heavy): "
            f"\"caught the late round live last night. brutal stoppage. "
            f"my dude was cooked 👀 still respect, dude went out on his "
            f"shield though\" ↔ foil (performative-aspirational): "
            f"\"there's something poetic about watching greatness meet "
            f"its limits — last night was a masterclass in resilience and "
            f"surrender 👀 these moments stay with you\"\n"
            f"      • gold (irritable-pragmatic): \"the studio scheduling "
            f"this on a tuesday is wild. 😂 nobody's getting off work for "
            f"that\" ↔ foil (deadpan reportage): \"the event is scheduled "
            f"for Tuesday evening. 😂 attendance may be limited by "
            f"workday conflicts\"\n"
            f"  - Length within ±20% of the gold. Structure: the foil "
            f"MUST NOT begin with the same opening words as the gold.\n"
            f"  - Do NOT introduce new hashtags, new topics, or new "
            f"factual claims. The ONLY axes that vary are voice register, "
            f"opener, idiolect template, stance, and vocabulary — NEVER "
            f"emoji presence/count.\n"
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
            # trending_alert's disliked_recent branch needs the raw
            # hashtag tokens so it can substitute one into the foil's
            # trending list. Other tasks ignore this field.
            "source_hashtags": hashtags,
            "source_object_id": evidence.get("source_object_id", ""),
            "source_timestamp": evidence.get("ts", 0),
            "source_app":       evidence.get("app", ""),
        }
    if flaw_kind == "over_personalization":
        # preference_removal_regen is special: the failure mode the task
        # tests is "model re-uses the preference the user JUST removed",
        # NOT "model surfaces an unrelated top category". Bind the LLM
        # rewrite to the held-out (removed) preference so the foil actually
        # tests the removal contract. Falls through to the generic top-
        # category source only if no removed pref is available.
        if task_id == "preference_removal_regen":
            held = inst.get("held_out_preference") or {}
            removed_pi = (held.get("persona_item") or "").strip()
            if removed_pi:
                return {
                    "persona_item": removed_pi,
                    "_from": "preference_removal_regen.held_out_preference",
                    "source_hashtags": held.get("source_hashtags") or [],
                }
        # over_personalization_sensitive_event: the foil must leak content
        # from the PLANTED sensitive disclosure (situation + planted row
        # text + episode hashtags), NOT a generic top user category. The
        # planted disclosure metadata sits on the instance already.
        if task_id == "over_personalization_sensitive_event":
            situation = (inst.get("_sensitive_event_specific_situation") or "").strip()
            row_text = (inst.get("_sensitive_event_evidence_row_text") or "").strip()
            tags = list(inst.get("_sensitive_event_evidence_row_hashtags") or [])
            if situation or row_text or tags:
                pi = situation or row_text[:120]
                return {
                    "persona_item": pi,
                    "_from": "sensitive_event.planted_disclosure",
                    "planted_situation": situation,
                    "planted_row_text": row_text,
                    "source_hashtags": tags,
                }
        cats = persona_ctx.get("top_categories") or []
        if cats:
            pool = cats[:min(6, len(cats))]
            # over_personalization_chatbot_text: the foil leans on a top
            # category to fail restraint. The failure mode only makes sense
            # if that category was engaged with RECENTLY relative to the
            # query — a stale top-of-history category isn't a credible
            # over-personalization (a competent agent wouldn't reach for
            # week-old signal on an off-topic question). Filter the pool
            # to categories with a recent engagement before the query, and
            # propagate the recency timestamp into the evidence dict for
            # downstream auditability.
            query_ts = int(inst.get("source_timestamp") or inst.get("t_test") or 0)
            cat_recent_ts = persona_ctx.get("cat_recent_ts") or {}
            chosen_ts = 0
            if task_id == "over_personalization_chatbot_text" and query_ts > 0 and cat_recent_ts:
                # Categories engaged within OVER_PERS_RECENT_WINDOW_DAYS before the query.
                recent_pool = [
                    (cat, n) for (cat, n) in pool
                    if 0 < (query_ts - int(cat_recent_ts.get(cat, 0)))
                       <= _OVER_PERS_RECENT_WINDOW_DAYS * 24 * 3600
                ]
                if recent_pool:
                    pool = recent_pool
                else:
                    # No category is within the strict 7-day window.
                    # Fall back to the nearest-recent category, but only
                    # if its age is under the hard 30-day ceiling — a
                    # foil leaning on a >30-day-old category isn't a
                    # credible over-pers failure. If nothing clears the
                    # ceiling, drop the instance (return None) rather
                    # than ship an unfair test.
                    aged_pool = sorted(
                        pool,
                        key=lambda kv: query_ts - int(cat_recent_ts.get(kv[0], 0)),
                    )
                    hard_max = _OVER_PERS_HARD_MAX_DAYS * 24 * 3600
                    aged_pool = [
                        kv for kv in aged_pool
                        if 0 < (query_ts - int(cat_recent_ts.get(kv[0], 0))) <= hard_max
                    ]
                    if not aged_pool:
                        return None
                    pool = aged_pool[:max(1, len(aged_pool) // 2)]
            chosen = rng.choice(pool)
            chosen_ts = int(cat_recent_ts.get(chosen[0], 0)) if cat_recent_ts else 0
            evidence: dict = {"persona_item": chosen[0]}
            if chosen_ts:
                evidence["source_timestamp"] = chosen_ts
                if query_ts:
                    evidence["recency_delta_seconds"] = max(0, query_ts - chosen_ts)
            return evidence
        return None
    if flaw_kind == "factual_error":
        grounding: dict = {"_from": "factual_error_grounding"}
        source_app = (inst.get("source_app") or inst.get("target_app") or "").strip()
        if source_app:
            wrong_apps = [a for a in ("Instagram", "Facebook", "Threads") if a.lower() != source_app.lower()]
            if wrong_apps:
                grounding["wrong_app"] = rng.choice(wrong_apps)
        hashtags = list(inst.get("source_hashtags") or [])
        if hashtags and len(hashtags) >= 2:
            mutated = list(hashtags)
            idx_to_swap = rng.randrange(len(mutated))
            mutated[idx_to_swap] = "#" + rng.choice(["trending", "viral", "popular", "new"]) + mutated[idx_to_swap].lstrip("#")[:8]
            grounding["mutated_hashtags"] = mutated
        title = (inst.get("title") or "").strip()
        if title:
            grounding["original_title"] = title
        grounding["persona_item"] = ""
        return grounding
    if flaw_kind == "voice_mismatch":
        # Pick a CONTRASTING voice register that differs from the user's
        # natural voice on idiolect / stance / syntax — NOT on emoji
        # density. The eval is graded by `voice_match` over the layered
        # voice schema (identity_spine signature_concerns + idiolect
        # templates + repertoire stances + per-app surface), so foils
        # whose only diff is "fewer emoji" don't actually exercise the
        # rubric.
        target_app = (inst.get("target_app") or "").lower()
        app_personas = persona_ctx.get("app_personas") or {}
        # Build candidates: every app voice EXCEPT the target. Prefer
        # the candidate whose `active_stances` differ MOST from the
        # target's, so the foil pulls a genuinely different stance
        # subset rather than a near-duplicate of the target voice.
        target_stances = set((app_personas.get(target_app) or {}).get("active_stances") or [])
        candidates: list[tuple[str, str, int]] = []
        for app, ap in app_personas.items():
            if app == target_app:
                continue
            # New schema: delta_summary. Legacy fallback: style_description.
            style = (ap.get("delta_summary") or ap.get("style_description") or "").strip()
            if not style:
                continue
            cand_stances = set(ap.get("active_stances") or [])
            stance_diff = len(target_stances.symmetric_difference(cand_stances))
            candidates.append((app, style[:320], stance_diff))
        if candidates:
            # Highest stance-divergence wins; ties broken by random shuffle.
            rng.shuffle(candidates)
            candidates.sort(key=lambda t: -t[2])
            other_app, style, _ = candidates[0]
            register = (
                f"the user's {other_app} voice — {style} "
                f"(differs from {target_app} on stance / register / syntax, "
                f"NOT on emoji density)"
            )
            return {
                "contrasting_register": register,
                "target_app": target_app,
                "_from": f"app_personas[{other_app}]",
            }
        # Fallbacks: pick a stance/idiolect-divergent register, NOT an
        # emoji-axis swap. The previous "stiff corporate, no emoji"
        # fallback collapsed every voice_mismatch foil onto emoji
        # presence/absence; replaced with three idiolect-/stance-axis
        # contrasts so the foil tests the rubric's voice signal.
        non_emoji_fallbacks = (
            "performative-aspirational lifestyle-influencer voice — "
            "metaphorical scaffolding, motivational verbs ('manifest', "
            "'unlocked', 'on a journey'), parallel-triplet lists, life-coach "
            "framing. Same emoji density as the gold; the contrast is the "
            "moralizing scaffolding and the parallel-list cadence.",
            "buttoned-up trade-publication voice — long noun phrases, "
            "third-person framing, hedged claims with formal hedges "
            "('arguably', 'one might note'), zero contractions, dense "
            "subject-matter vocabulary. Same emoji density as the gold; "
            "the contrast is in syntax and lexical register.",
            "deadpan reportage voice — short flat declarative sentences, "
            "no rhetorical questions, no in-group references, neutral "
            "register, factual phrasing. Same emoji density as the gold; "
            "the contrast is the absence of stance and idiolect markers.",
        )
        register = rng.choice(non_emoji_fallbacks)
        return {
            "contrasting_register": register,
            "target_app": target_app,
            "_from": "fallback_non_emoji_axis",
        }
    return None


_LIST_TASK_FACTUAL_ERROR_TASKS = {
    "agentic_proactive_daily_catchup",
    "agentic_trending_alert",
    "agentic_dm_digest",
    "agentic_group_dm_summary",
}

# Named-entity extractors for the list/digest factual_error validator.
# A real wrong-event foil must differ from the gold on at least one of
# these axes — paraphrasing alone trivially fails the test.
_THREAD_ID_RE = re.compile(
    r"\b(?:[a-z]{2}_thr_\d+_\d+|in_thr_\d+_\d+|fa_thr_\d+_\d+|th_thr_\d+_\d+|ch_thr_\d+_\d+)\b",
    re.IGNORECASE,
)
_FRIEND_RE = re.compile(r"\bfriend_\d+\b", re.IGNORECASE)
_QUOTED_RE = re.compile(r"[\"“”]([^\"“”\n]{20,200})[\"”\"]")


def _extract_named_entities(text: str) -> dict[str, set[str]]:
    """Pull the wrong-event axes a foil needs to swap on: thread ids,
    `friend_N` labels, and quoted strings (≥20 chars). Quotes are
    normalized to lowercase and trimmed to first 60 chars so a minor
    re-quote doesn't masquerade as a swap. Returns a dict of sets.
    """
    if not text:
        return {"threads": set(), "friends": set(), "quotes": set()}
    return {
        "threads": {m.lower() for m in _THREAD_ID_RE.findall(text)},
        "friends": {m.lower() for m in _FRIEND_RE.findall(text)},
        "quotes": {
            " ".join(q.lower().split())[:60]
            for q in _QUOTED_RE.findall(text)
        },
    }


def _list_task_inferior_swaps_entity(example: str, inferior: str) -> bool:
    """For list/digest factual_error foils: confirm the inferior actually
    swaps at least one named entity (thread id / friend label / quoted
    string). A foil that only paraphrases the gold's prose fails this
    check — there's no detectable factual difference.
    """
    if not example or not inferior:
        return False
    ex_ents = _extract_named_entities(example)
    in_ents = _extract_named_entities(inferior)
    # An entity was "swapped" if it appears in one but not both.
    for axis in ("threads", "friends", "quotes"):
        only_ex = ex_ents[axis] - in_ents[axis]
        only_in = in_ents[axis] - ex_ents[axis]
        # A real swap means SOMETHING was substituted: a value present
        # in the gold is gone from the foil, AND a different value appears
        # in the foil. Otherwise the foil either dropped or added a name
        # without substitution (the "drop one item" variant) — which still
        # counts as a real swap as long as the dropped/added side is
        # non-empty AND the texts aren't a strict subset/superset.
        if only_ex and only_in:
            return True
        if only_ex and len(ex_ents[axis]) > 1:
            # Dropped one of several gold items — counts as a swap if
            # the inferior didn't just truncate the prose wholesale.
            ex_word_count = len(example.split())
            in_word_count = len(inferior.split())
            if in_word_count >= int(ex_word_count * 0.7):
                return True
    return False


_SENSITIVE_EVENT_PREAMBLE_RE = re.compile(
    r"^\s*(?:as\s+(?:a|an|someone)\b[^.,;:\n]{0,160}[,.;:]\s*"
    r"|since\s+you[^.,;:\n]{0,160}[,.;:]\s*"
    r"|given\s+(?:that\s+)?you[^.,;:\n]{0,160}[,.;:]\s*)+",
    re.IGNORECASE,
)
_WORD_TOKEN_RE = re.compile(r"[A-Za-z']+")


def _preamble_stripped_too_similar(inferior: str, example: str,
                                   threshold: float = 0.7) -> bool:
    """For over_personalization_sensitive_event: after stripping the
    leading `as a [ROLE_FROM_DISCLOSURE], …` preamble from the inferior,
    its body's token-Jaccard against the example response shouldn't be
    too high — otherwise the foil is just an example with a sensitive
    preamble glued on (the preamble-only failure pattern surfaced by
    the 2026-05-28 audit). Returns True when the stripped body is
    `threshold` or more similar to the example.
    """
    if not inferior or not example:
        return False
    body = _SENSITIVE_EVENT_PREAMBLE_RE.sub("", inferior, count=1).strip()
    if not body:
        return False
    body_toks = {t.lower() for t in _WORD_TOKEN_RE.findall(body) if len(t) > 2}
    ex_toks = {t.lower() for t in _WORD_TOKEN_RE.findall(example) if len(t) > 2}
    if not body_toks or not ex_toks:
        return False
    inter = len(body_toks & ex_toks)
    union = len(body_toks | ex_toks)
    if union == 0:
        return False
    return (inter / union) >= threshold


def _generate_inferior(llm: Callable[[str], str], response: str,
                       flaw_kind: str, evidence: dict,
                       task_id: str = "",
                       user_query: str = "",
                       axis_hint: str = "") -> str | None:
    """LLM-rewrite path for non-ranking foils.

    Ranking-task foils are deterministic — see `_compute_ranking_inferior`
    and the dispatch in the foil loop. This function is only invoked for
    Family 2/3/4 (list/digest, voice, freeform).

    `axis_hint` is an optional task-specific contract description fed in
    by the audit's auto-regenerate path. When non-empty, it's prepended
    to the per-flaw instruction as additional pressure. Build-path
    callers leave it empty and behavior is unchanged.
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
        flaw_instruction=_flaw_instruction(flaw_kind, evidence, task_id,
                                           axis_hint=axis_hint),
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
                                            jaccard_min=jaccard_min,
                                            flaw_kind=flaw_kind)
        if not passed:
            last_text, last_reason = text, reason
            continue
        return text
    # All 3 attempts failed → return None so the caller drops the foil.
    return None


def synthesize_special_task_example_inferior(inst: dict, task_id: str,
                                              discovery_llm=None) -> dict | None:
    """Example/inferior synthesis for task families that the standard
    `postprocess_benchmark` skips (they're not in `_PERSONALIZATION_TASKS`).
    Uses LLM when available, falls back to templates.

    Returns:
        {"example_response": str, "inferior_response": {text, flaw_kind, flaw_evidence}}
        when this task is handled and synthesis succeeds, or None otherwise.

    Covered task families:
      - `over_personalization_repetition_recsys` (5-query saturation cluster)
      - `over_personalization_repetition_chatbot` (5-query chatbot cluster)
      - `restraint_sensitive_event_silence` (proactive-silence test)
    """
    if task_id in ("over_personalization_repetition_recsys",
                   "over_personalization_repetition_chatbot"):
        target_pref = (inst.get("target_pref") or "").strip()
        primary_category = (inst.get("primary_category") or "").strip()
        n_queries = int(inst.get("n_queries") or len(inst.get("queries") or []) or 5)
        n_allowed = int(inst.get("n_allowed_repetitions") or 2)
        hints = inst.get("persona_hint") or {}
        broader_cats = [c for c in (hints.get("top_categories") or []) if c and c != primary_category]
        while len(broader_cats) < 3:
            broader_cats.append(broader_cats[-1] if broader_cats else "the user's broader interests")
        cat2, cat3, cat4 = broader_cats[0], broader_cats[1], broader_cats[2]
        top_hashtags = hints.get("top_hashtags") or []
        queries = inst.get("queries") or []
        query_texts = []
        for q in queries[:5]:
            if isinstance(q, dict):
                query_texts.append(q.get("user_query") or "")
            elif isinstance(q, str):
                query_texts.append(q)

        # Move the behavioral rubric to groundtruth_preference
        if task_id == "over_personalization_repetition_recsys":
            rubric = (
                f"Saturated preference: {target_pref} ({primary_category})\n"
                f"Head-zone (queries 1..{n_allowed}): may surface {primary_category} content.\n"
                f"Tail-zone (queries {n_allowed+1}+): must diversify to the user's "
                f"broader interests ({cat2}, {cat3}, {cat4}) — no more {primary_category}.\n"
                f"Inferior keeps returning {primary_category} content for every query."
            )
        else:
            rubric = (
                f"Saturated preference: {target_pref} ({primary_category})\n"
                f"Head-zone (turns 1..{n_allowed}): may reference {target_pref}.\n"
                f"Tail-zone (turns {n_allowed+1}+): answer each question on its own "
                f"terms without forcing {primary_category} as framing.\n"
                f"Inferior force-grafts {target_pref} into every turn."
            )
        inst["groundtruth_preference"] = rubric

        # LLM-generate concrete example/inferior responses
        if discovery_llm is not None:
            if task_id == "over_personalization_repetition_recsys":
                prompt = (
                    f"Generate two realistic feed recommendation lists for a user "
                    f"who strongly likes {target_pref!r} ({primary_category}) but also "
                    f"has interests in {cat2}, {cat3}, {cat4}.\n"
                    f"Top hashtags: {', '.join(f'#{h}' for h in top_hashtags[:6])}\n\n"
                    f"List A (GOOD — diversifies): 5 feed items. Items 1-{n_allowed} "
                    f"are about {primary_category}. Items {n_allowed+1}-5 branch out to "
                    f"{cat2}, {cat3}, {cat4}. Each item is a realistic post title + "
                    f"1-sentence caption.\n\n"
                    f"List B (BAD — saturated): 5 feed items. ALL 5 are about "
                    f"{primary_category} — no diversification. Different titles but same "
                    f"narrow topic.\n\n"
                    f"Return JSON:\n```json\n"
                    f"{{\"good\": [\"title: caption\", ...], \"bad\": [\"title: caption\", ...]}}\n```"
                )
            else:
                q_block = "\n".join(f"  Q{i+1}: {q}" for i, q in enumerate(query_texts[:5]) if q)
                prompt = (
                    f"Generate two sets of chatbot responses for these user queries. "
                    f"The user strongly likes {target_pref!r} ({primary_category}) but "
                    f"also has interests in {cat2}, {cat3}, {cat4}.\n\n"
                    f"Queries:\n{q_block}\n\n"
                    f"Set A (GOOD — diversifies): Answers Q1-Q{n_allowed} may reference "
                    f"{primary_category}. From Q{n_allowed+1} onward, answer each "
                    f"question on its own terms — no {primary_category} framing.\n\n"
                    f"Set B (BAD — repetitive): Every answer forces a {primary_category} "
                    f"reference, even when the question is unrelated.\n\n"
                    f"Each response is 1-2 sentences, conversational.\n"
                    f"Return JSON:\n```json\n"
                    f"{{\"good\": [\"response to Q1\", ...], \"bad\": [\"response to Q1\", ...]}}\n```"
                )
            try:
                raw = discovery_llm.query_llm(prompt)
                parsed = extract_json_from_response(raw) or {}
                if isinstance(parsed, dict):
                    good = parsed.get("good") or []
                    bad = parsed.get("bad") or []
                    if isinstance(good, list) and len(good) >= 3 and isinstance(bad, list) and len(bad) >= 3:
                        example_text = "\n".join(f"  {i+1}. {item}" for i, item in enumerate(good))
                        inferior_text = "\n".join(f"  {i+1}. {item}" for i, item in enumerate(bad))
                        return {
                            "example_response": example_text,
                            "inferior_response": {
                                "text": inferior_text,
                                "flaw_kind": "saturated_repetition",
                                "flaw_evidence": {
                                    "_from": f"synthesize_special_task_example_inferior::{task_id}",
                                    "target_pref": target_pref,
                                    "primary_category": primary_category,
                                    "n_queries": n_queries,
                                    "n_allowed": n_allowed,
                                    "diversification_categories": [cat2, cat3, cat4],
                                },
                            },
                        }
            except Exception:
                pass

        # Fallback: template-based (used when discovery_llm is None)
        if task_id == "over_personalization_repetition_recsys":
            example_lines = [
                f"  1. A {primary_category} post aligned with {target_pref!r}",
                f"  2. Another {primary_category} pick (head-zone, within tolerance)",
                f"  3. A {cat2} post — diversification starts",
                f"  4. A {cat3} recommendation",
                f"  5. A {cat4} pick — fully diversified",
            ]
            inferior_lines = [
                f"  1. A {primary_category} post aligned with {target_pref!r}",
                f"  2. Another {primary_category} pick",
                f"  3. Yet another {primary_category} post (should have diversified)",
                f"  4. More {primary_category} (still no diversification)",
                f"  5. Still {primary_category} (never escapes the saturated cluster)",
            ]
        else:
            example_lines = [
                f"  Q1: Answer references {target_pref!r} naturally",
                f"  Q2: Another {primary_category}-anchored response (head-zone)",
                f"  Q3: Answers the question on its own terms, no {primary_category}",
                f"  Q4: Topic-appropriate answer, diversified",
                f"  Q5: Independent answer, no forced preference framing",
            ]
            inferior_lines = [
                f"  Q1: Answer references {target_pref!r}",
                f"  Q2: Another {primary_category} response",
                f"  Q3: Forces {primary_category} framing on unrelated question",
                f"  Q4: Still grafting {primary_category} onto the answer",
                f"  Q5: Keeps forcing {target_pref!r} into every response",
            ]

        return {
            "example_response": "\n".join(example_lines),
            "inferior_response": {
                "text": "\n".join(inferior_lines),
                "flaw_kind": "saturated_repetition",
                "flaw_evidence": {
                    "_from": f"synthesize_special_task_example_inferior::{task_id}",
                    "target_pref": target_pref,
                    "primary_category": primary_category,
                    "n_queries": n_queries,
                    "n_allowed": n_allowed,
                    "diversification_categories": [cat2, cat3, cat4],
                },
            },
        }

    if task_id == "local_recommendation_geo_shift":
        current_city = (inst.get("current_city") or "").strip()
        prior_city = (inst.get("prior_city") or "").strip()
        category = (inst.get("category") or "").strip()
        user_query = (inst.get("user_query") or "").strip()
        if not current_city or not category:
            return None

        if discovery_llm is not None:
            prompt = (
                f"A user in {current_city} asks: \"{user_query}\"\n\n"
                f"Generate two short AI assistant responses (2-3 sentences each):\n\n"
                f"Response A (CORRECT): Recommend specific {category} options in "
                f"{current_city}. Name 2-3 real or plausible places. Natural, helpful tone.\n\n"
                f"Response B (WRONG): Recommend {category} options in {prior_city} "
                f"instead — the agent failed to detect the user moved to {current_city} "
                f"and is still anchored on the old location. Also name 2-3 places.\n\n"
                f"Return JSON:\n```json\n"
                f"{{\"correct\": \"...\", \"wrong\": \"...\"}}\n```"
            )
            try:
                raw = discovery_llm.query_llm(prompt)
                parsed = extract_json_from_response(raw) or {}
                if isinstance(parsed, dict):
                    correct = (parsed.get("correct") or "").strip()
                    wrong = (parsed.get("wrong") or "").strip()
                    if correct and wrong and len(correct) > 20 and len(wrong) > 20:
                        return {
                            "example_response": correct,
                            "inferior_response": {
                                "text": wrong,
                                "flaw_kind": "stale_geo_anchor",
                                "flaw_evidence": {
                                    "_from": "synthesize_special_task::geo_shift",
                                    "prior_city": prior_city,
                                    "current_city": current_city,
                                    "category": category,
                                },
                            },
                        }
            except Exception:
                pass
        return None

    if task_id == "restraint_sensitive_event_silence":
        evidence = inst.get("trigger_evidence") or {}
        situation = ""
        if isinstance(evidence, dict):
            situation = (evidence.get("situation")
                         or evidence.get("episode_situation") or "")[:160]
        example_text = (
            "[silence — no proactive message] The agent correctly restrains "
            "during a sensitive-life-event window: no daily catch-up push, "
            "no notification, no 'just checking in' nudge. Silence is the "
            "correct action; speaking up here would over-step."
        )
        fallback_inferior = (
            "Hey — just thinking of you. I noticed a few things in your feed "
            "I thought you might want to catch up on; want me to pull a quick "
            "summary together? Totally cool to skip — just wanted to flag "
            "it in case it helps your day."
        )
        inferior_text = fallback_inferior
        if discovery_llm is not None:
            from evaluation.prompts import proactive_inferior_prompt
            context = f"Sensitive event: {situation or 'private episode'}. Expected behavior: STAY SILENT."
            try:
                prompt = proactive_inferior_prompt(example_text, "wrong_act_restrain_decision", context)
                raw = discovery_llm.query_llm(prompt)
                import re as _re
                m = _re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, _re.DOTALL)
                if m:
                    text = m.group(1).replace('\\"', '"').replace('\\n', '\n').strip()
                    if len(text) >= 10:
                        inferior_text = text
            except Exception:
                pass
        return {
            "example_response": example_text,
            "inferior_response": {
                "text": inferior_text,
                "flaw_kind": "wrong_act_restrain_decision",
                "flaw_evidence": {
                    "_from": "synthesize_special_task_example_inferior::restraint_sensitive_event_silence",
                    "expected_behavior": inst.get("expected_behavior", "restrain"),
                    "trigger_situation": situation,
                },
            },
        }

    return None


def regenerate_inferior_for_instance(
    inst: dict,
    task_id: str,
    bq,
    user_id: str,
    inferior_llm: Callable[[str], str],
    axis_hint: str = "",
    rng_seed: int = 0,
) -> dict | None:
    """Audit-side regen entry point.

    Given a benchmark instance whose `inferior_response` was rejected by
    the per-task axis check, re-derive evidence + re-run `_generate_inferior`
    with the corrected evidence picker (the `over_personalization` branch
    now binds to the held-out preference for `preference_removal_regen`,
    not `top_categories[0]`).

    `axis_hint` is the audit contract's axis description plus the prior
    failure reason; it gets prepended to the per-flaw instruction as
    additional pressure.

    Returns the new `inferior_response` dict (`{text, flaw_kind, flaw_evidence}`)
    on success, or None if no valid foil could be produced. Caller is
    responsible for writing the result back to `inst` and re-running
    the audit.
    """
    if not inferior_llm:
        return None
    example = (inst.get("example_response") or "").strip()
    if not example:
        return None
    t_test = int(inst.get("t_test") or inst.get("source_timestamp") or 0)
    try:
        ctx = _build_persona_ctx(bq, user_id, t_test)
    except Exception:
        ctx = {}
    allowed_flaws = _TASK_FLAW_KINDS.get(task_id, _FLAW_KINDS_PERSONALIZATION)
    # Preserve the original flaw_kind if it's still in the allowed set;
    # otherwise pick the first allowed flaw deterministically (the audit
    # contract is per-task, so flaw_kind should be stable across regen).
    prev = inst.get("inferior_response") or {}
    prev_flaw = (prev.get("flaw_kind") or "") if isinstance(prev, dict) else ""
    flaw_kind = prev_flaw if prev_flaw in allowed_flaws else allowed_flaws[0]
    rng = random.Random(rng_seed or hash(f"{user_id}:{task_id}:{t_test}") % (2**31))
    evidence = _pick_flaw_evidence(flaw_kind, inst, ctx, rng, task_id)
    if evidence is None:
        # Try fallback flaws if the original isn't satisfiable here.
        for fk in allowed_flaws:
            if fk == flaw_kind:
                continue
            evidence = _pick_flaw_evidence(fk, inst, ctx, rng, task_id)
            if evidence is not None:
                flaw_kind = fk
                break
    if evidence is None:
        return None
    user_query = _synthesize_user_query(inst, task_id)
    text = _generate_inferior(
        inferior_llm, example, flaw_kind, evidence, task_id,
        user_query=user_query, axis_hint=axis_hint,
    )
    if not text:
        return None
    return {
        "text": text,
        "flaw_kind": flaw_kind,
        "flaw_evidence": evidence,
        "_regen_source": "audit_axis_check",
    }


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
    per-app voice registers (app_personas) for voice_mismatch foils.

    Also tracks `cat_recent_ts` / `pref_recent_ts` (max source_timestamp
    per category / persona_item before t_test) so the over_personalization
    flaw-evidence picker can constrain its choice to RECENT engagements —
    a foil that leans on a stale top-of-history category isn't a credible
    over-personalization failure (a competent agent wouldn't lean on a
    stale signal anyway)."""
    from collections import Counter
    DAY = 24 * 3600
    pref_counts: Counter = Counter()
    cat_counts: Counter = Counter()
    cat_recent_ts: dict[str, int] = {}
    pref_recent_ts: dict[str, int] = {}
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
                    if ts > pref_recent_ts.get(pi, 0):
                        pref_recent_ts[pi] = ts
                if cat:
                    cat_counts[cat] += 1
                    if ts > cat_recent_ts.get(cat, 0):
                        cat_recent_ts[cat] = ts
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
        "cat_recent_ts": cat_recent_ts,
        "pref_recent_ts": pref_recent_ts,
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
    n_voice_align_passed = 0
    n_voice_align_failed = 0

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

    n_chatbot_triplet_built = 0
    n_chatbot_triplet_failed = 0
    # Accumulate generated user_queries so the triplet prompt can enforce
    # diversity — each new call sees the prior queries as negative examples
    # and must produce a structurally different ask type.
    _chatbot_prior_queries: list[str] = []

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

            # ---- chatbot_personalized_response triplet regen ----------
            # For chatbot proactive instances, replace the user_query (which
            # was extracted from a multi-turn chatbot session and is often
            # a copyedit / compose request that doesn't invite personalization)
            # with a freshly-generated (user_query, example_response,
            # inferior_response) triplet anchored on the held-out preference.
            # Single LLM call. Skips the rest of this loop iteration's
            # standard workstream-I/J generation paths.
            if (task_id == "chatbot_personalized_response"
                    and (inst.get("arm") or "proactive") == "proactive"
                    and self_check_llm is not None
                    and groundtruth):
                t_test = int(inst.get("t_test") or inst.get("source_timestamp") or 0)
                if t_test not in _ctx_cache:
                    _ctx_cache[t_test] = _build_persona_ctx(bq, user_id, t_test)
                ctx = _ctx_cache[t_test]
                try:
                    profile = bq._load_profile(user_id) if hasattr(bq, "_load_profile") else {}
                except Exception:
                    profile = {}
                chatbot_app_persona = (ctx.get("app_personas") or {}).get("chatbot") or {}
                user_voice_block = ctx.get("user_voice") or {}
                topical_signals = [
                    h for h, _ in (ctx.get("top_categories") or [])[:6]
                ]
                triplet = _generate_chatbot_triplet(
                    self_check_llm, groundtruth, profile,
                    user_voice=user_voice_block,
                    chatbot_persona=chatbot_app_persona,
                    recent_topical_signals=topical_signals,
                    prior_queries=_chatbot_prior_queries,
                )
                if triplet and triplet.get("user_query") and triplet.get("example_response"):
                    inst["user_query"] = triplet["user_query"]
                    inst["example_response"] = triplet["example_response"]
                    if triplet.get("inferior_response"):
                        inst["inferior_response"] = {
                            "text": triplet["inferior_response"],
                            "flaw_kind": "missed_personalization",
                            "flaw_evidence": {"_from": "chatbot_triplet_regen"},
                        }
                    _chatbot_prior_queries.append(triplet["user_query"])
                    n_chatbot_triplet_built += 1
                    # Skip the rest of the standard workstream I/J path —
                    # the triplet replaces user_query, example_response,
                    # AND inferior_response in one shot.
                    continue
                else:
                    n_chatbot_triplet_failed += 1
                    # Fall through to the standard workstream-I/J path so
                    # this instance still gets some example/inferior pair.

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
                elif task_id in _COMPOSE_TASKS:
                    # _generate_example_response returned None — meaning even
                    # after 3 retries the LLM couldn't produce a compose-task
                    # response above the ≥50-word floor. The template-stub
                    # example_response from data_preparation/visualize.py is
                    # 9-29 words and would otherwise survive into queries.csv.
                    # Clear BOTH example and inferior_response so the
                    # format-verify gate (a) at prepare_eval_data.py:682
                    # drops the row — better to lose a compose row than ship
                    # a stub that fails the verifier on every run.
                    inst["example_response"] = ""
                    inst["inferior_response"] = ""
                    example = ""

            if not example:
                continue

            # Heuristic voice-evidence span extraction for compose tasks.
            # Runs independently of inferior generation so the renderer can
            # bold the gold's voice anchors even when no foil exists.
            if task_id in _VOICE_EVIDENCE_TASKS:
                # Same fallback as the inferior block: instances without an
                # explicit `t_test` key store the test moment as `source_timestamp`.
                t_test_v = int(inst.get("t_test") or inst.get("source_timestamp") or 0)
                if t_test_v not in _ctx_cache:
                    _ctx_cache[t_test_v] = _build_persona_ctx(bq, user_id, t_test_v)
                _voice_ctx = _ctx_cache[t_test_v]
                _uv_block = (_voice_ctx or {}).get("user_voice") if isinstance(_voice_ctx, dict) else {}
                _spans = _extract_voice_evidence_spans(example, _uv_block or {})
                if _spans:
                    inst["example_response_voice_evidence"] = _spans
                n_feats = len(_spans) if _spans else 0
                passed = n_feats >= MIN_VOICE_FEATURES_EXAMPLE
                inst["voice_alignment_check"] = {
                    "passed": passed,
                    "n_features": n_feats,
                    "min_required": MIN_VOICE_FEATURES_EXAMPLE,
                }
                if passed:
                    n_voice_align_passed += 1
                else:
                    n_voice_align_failed += 1

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

            # Workstream J: inferior_response. Default rule: only proactive /
            # contradiction arms get a foil — control arms test restraint and
            # the gold response is intentionally generic, so no foil pair
            # makes sense by default.
            #
            # Schema-uniformity exception: tasks explicitly registered in
            # `_TASK_FLAW_KINDS` get a foil regardless of arm. Today this
            # opens the gate for `over_personalization_*` control tasks
            # (foil = same query but with persona leaked — the failure mode
            # they grade). Other tasks not in the table fall through to the
            # default arm-gated path.
            arm = inst.get("arm") or "proactive"
            arm_eligible = arm in ("proactive", "contradiction")
            task_force_eligible = task_id in _TASK_FLAW_KINDS
            if ((arm_eligible or task_force_eligible)
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
                    # Fall back to `source_timestamp` when `t_test` isn't on
                    # the instance — chatbot_personalized_response (and
                    # other Task B arms) store the per-instance test moment
                    # as `source_timestamp`, not `t_test`. Without this
                    # fallback, t_test=0 → ctx built at epoch → no top_prefs
                    # → all flaws return None → all 30 instances silently
                    # skip the inferior gen. (Pre-fix bug.)
                    t_test = int(inst.get("t_test") or inst.get("source_timestamp") or 0)
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
                    # Sensitive-event preamble-only check. Audit (2026-05-28)
                    # found ~18 rows where the inferior added a "as a
                    # [ROLE_FROM_DISCLOSURE], …" preamble but the body was
                    # near-identical to the example. Strip the preamble and
                    # compare; if Jaccard ≥ 0.7 with the example, regen once.
                    if (text
                            and task_id == "over_personalization_sensitive_event"
                            and _preamble_stripped_too_similar(text, example)):
                        text2 = _generate_inferior(
                            inferior_llm, example, flaw_kind, evidence, task_id,
                            user_query=user_query,
                        )
                        if text2 and not _preamble_stripped_too_similar(text2, example):
                            text = text2
                    # List/digest factual_error wrong-event check. The LLM
                    # ignores the "swap one identifier" directive ~67% of
                    # the time and ships paraphrases. Run a deterministic
                    # entity-swap check and regen up to twice; drop the
                    # row if all 3 attempts share every named entity with
                    # the gold.
                    if (text
                            and flaw_kind == "factual_error"
                            and task_id in _LIST_TASK_FACTUAL_ERROR_TASKS
                            and not _list_task_inferior_swaps_entity(example, text)):
                        for _retry in range(2):
                            text2 = _generate_inferior(
                                inferior_llm, example, flaw_kind, evidence,
                                task_id, user_query=user_query,
                            )
                            if text2 and _list_task_inferior_swaps_entity(example, text2):
                                text = text2
                                break
                        else:
                            # All 3 attempts produced paraphrases. Don't
                            # ship a foil whose factual error reviewers
                            # can't point to.
                            text = ""
                    if text:
                        inst["inferior_response"] = {
                            "text": text,
                            "flaw_kind": flaw_kind,
                            "flaw_evidence": evidence,
                        }
                        n_inferior_built += 1
                        # Extract voice evidence from the inferior text too —
                        # used by the renderer to bold tone anchors that the
                        # foil DOES leverage (so reviewers see which tone
                        # aspects the foil keeps vs swaps). Only meaningful
                        # for voice-graded tasks.
                        if task_id in _VOICE_EVIDENCE_TASKS:
                            _uv_block_inf = ctx.get("user_voice") or {}
                            _inf_spans = _extract_voice_evidence_spans(text, _uv_block_inf)
                            if _inf_spans:
                                inst["inferior_response_voice_evidence"] = _inf_spans
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
              f"voice_check_regen={n_voice_check_regen} "
              f"voice_align_passed={n_voice_align_passed} "
              f"voice_align_failed={n_voice_align_failed} "
              f"chatbot_triplet_built={n_chatbot_triplet_built} "
              f"chatbot_triplet_failed={n_chatbot_triplet_failed}")
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
        "voice_align_passed": n_voice_align_passed,
        "voice_align_failed": n_voice_align_failed,
    }
    return bm
