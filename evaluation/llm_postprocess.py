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
    "context_shift_scenarios",
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
    "agentic_moment_recommendation",
    "agentic_dm_digest",
    "agentic_cross_app_repost",
    "agentic_auto_reply",
    "agentic_vague_refind",
    "agentic_composed_post",
    "agentic_send_post",
    "agentic_collection_curation",
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
#
# Write-and-summarize agentic tasks (compose a post, write a reply, write a
# digest, summarize a thread) used to live here, but the deterministic
# templates produced absurd outputs — e.g. agentic_auto_reply emitted a
# canned "yeah that works, see you saturday" regardless of the inbound DM,
# and agentic_send_post echoed the user's brief verbatim with one hashtag
# tacked on. They now route through the LLM-gen path so the gold actually
# responds in context, grounded in the user's real friends / threads /
# voice via _enrich_groundtruth_for_task.
_TASKS_ALREADY_CONCRETE = {
    # Pure structural responses where the words are tightly constrained by
    # the task — LLM-gen adds little and risks drift.
    "agentic_vague_refind",              # lookup template
    "agentic_wrong_recipient_check",     # confirmation prompt
}

# Tasks where the gold is too short / structural for the inferior-foil
# pipeline to produce a natural rewrite. Skipping foil generation here
# avoids splice-fragment failures like
#   "...two Alexes in your friends list, #NFL, #ZachWilson, ..."
_TASKS_NO_FOIL = {
    "agentic_wrong_recipient_check",
    "agentic_vague_refind",
}

# Ranking tasks — example_response is a deterministic ranked index list,
# computed without LLM.
_RANKING_TASKS = {
    "personalized_feed_ranking",
    "personalized_recommendation",
    "at_ai_directive_followup",
    "short_vs_long_term_lifecycle",
}


_EXAMPLE_GEN_PROMPT = """You are writing the GOLD REFERENCE response that an \
ideal personalized AI agent would emit for a benchmark test instance.

Task type: {task_type}
{task_guidance}

User query / trigger:
\"\"\"{query}\"\"\"

Persona signal the agent may draw on (REAL data — context for YOU, not \
content to quote back to the user):
\"\"\"{groundtruth}\"\"\"
{prior}
Write the actual response — natural, concrete, the words the agent would say.
{length_guidance}

CRITICAL — subtle personalization only:
  - The persona signal above is YOUR context. Do NOT announce it back to the \
    user. NEVER write phrases like "you're interested in X", "you like Y", \
    "you'd like Z", "you'd probably enjoy", "as someone who", "I know you", \
    "I remember", "given your love of", "based on your", "since you're into".
  - Do NOT use benchmark-internal category labels as user-facing language. \
    "comedy video content" → "a comedy clip" / "a short funny video". \
    "boxing fandom" → "a boxing breakdown" / "a fight thread". Translate \
    every category phrase into how a person would actually say it.
  - Personalization should feel invisible: just recommend something whose \
    topic and tone happens to fit the user. The user shouldn't be able to \
    tell from the response that the agent has a persona file on them.
  - Do NOT quote the persona's voice sample back at the user. Use it only \
    to calibrate your own register and word choice.

Output ONE fenced ```json block:
```json
{{"text": "<the actual gold response — not a description of one>"}}
```"""


_TASK_GUIDANCE: dict[str, str] = {
    "chatbot_proactive_personalization": (
        "The user's question implicitly invites the held-out preference. "
        "Weave it in naturally — never parrot the preference verbatim."
    ),
    "over_personalization_chatbot_text": (
        "The user's question is generic. Answer it helpfully WITHOUT "
        "surfacing any of the user's personal preferences."
    ),
    "over_personalization_distractor_reject": (
        "The user's question is generic. Answer it helpfully WITHOUT "
        "surfacing any of the user's personal preferences."
    ),
    "preference_removal_regen": (
        "The user previously asked you to forget the listed preference. "
        "Answer the question WITHOUT drawing on it; produce a "
        "substantively different response."
    ),
    "context_shift_scenarios": (
        "The user's context has shifted. Answer the question respecting "
        "the carve-out — do NOT surface forbidden_items even if they "
        "would normally fit the user's profile."
    ),
    "repetition_fatigue_pairs": (
        "Output two very short responses, one for t_early (showing the "
        "PRE-dominant category) and one for t_late (showing the SHIFT "
        "category). Format: 'Early: ...' / 'Late: ...'."
    ),
    "repetition_fatigue_sequences": (
        "Output a short paragraph showing the agent varies which "
        "preferences it surfaces across the sequence — name 2-3 "
        "different categories rotated turn-to-turn."
    ),
    "active_mistake_prevention": (
        "If polarity=warn: write a respectful warning that flags the "
        "cross-signal contradiction — name the concern, mention items in "
        "must_mention, avoid items in must_not_mention. "
        "If polarity=foil: write a helpful answer with NO warning."
    ),
    "daily_personalized_briefing": (
        "Write a 3-5 item morning briefing that references hashtags / "
        "topics from gt_positive_engagements (post-t_test real engagement) "
        "and AVOIDS topics from gt_avoid_engagements. Tone: light, "
        "conversational. Each item ≤ 1 sentence."
    ),
    # ---- Agentic write/summary tasks (gold must respond IN CONTEXT to the
    # query — not a canned phrase). The LLM sees the inbound brief / DM /
    # update via the {query} field; persona signal goes via {groundtruth}.
    "agentic_user_tone_post": (
        "Compose a short post in the user's voice for the target_app. Use "
        "their typical hashtag pool from groundtruth. Match their tone "
        "(check the voice sample). 1-2 sentences + 1-2 hashtags. Do NOT "
        "describe what to write — actually write the post."
    ),
    "agentic_dm_digest": (
        "Summarize the user's recent DM threads on the target app. Mention "
        "2-4 plausible specific senders/topics drawn from the DM context. "
        "2-3 sentences, conversational tone. Don't list — narrate."
    ),
    "agentic_cross_app_repost": (
        "Paraphrase the source post for the target app, in the user's "
        "voice on that platform. Don't quote verbatim. 1-2 sentences + "
        "1 hashtag from their pool."
    ),
    "agentic_auto_reply": (
        "Write a short reply to the inbound DM in the user's casual tone. "
        "STAY ON TOPIC to the inbound message — if it's a friend confirming "
        "plans, confirm; if it's a stranger pitching something, decline "
        "politely or ignore. Match the relationship signal. 1-2 sentences. "
        "Do NOT compose an unrelated message."
    ),
    "agentic_composed_post": (
        "Take the user's update and rewrite it in their voice on the "
        "target_app. 1-2 sentences + 1 hashtag from their typical pool."
    ),
    "agentic_send_post": (
        "The user just told the chatbot to post the brief in {query} on the "
        "target_app. Compose the post in their voice on that platform — "
        "don't echo the brief verbatim, rewrite it as something they'd "
        "actually publish. 1-2 sentences + 1 hashtag from their pool."
    ),
    "agentic_group_dm_summary": (
        "Summarize the group thread: who said what, the open question, "
        "any disagreement. 2-3 sentences, conversational."
    ),
    "agentic_proactive_daily_catchup": (
        "Brief the user on 2-3 specific catch-up items from the past 24h "
        "(unread DMs, friends' posts, content matching their interests). "
        "Each item ≤ 1 sentence."
    ),
    "agentic_trending_alert": (
        "Alert the user to 2-3 trending hashtags/topics that they'd "
        "naturally care about. Brief, ≤ 1 sentence each. Don't announce "
        "WHY each fits — the relevance should be obvious from the topic."
    ),
    "agentic_moment_recommendation": (
        "Recommend ONE concrete piece of content (a clip, post, or thread) "
        "the user could see right now that fits the moment. Use everyday "
        "user-facing words to describe topics — never use category labels "
        "verbatim (e.g. say 'a short comedy clip', NOT 'comedy video "
        "content'; say 'a boxing breakdown', NOT 'boxing fandom'). 1-2 "
        "sentences."
    ),
    "agentic_collection_curation": (
        "Suggest 3 thematic collection names the user could organize their "
        "saved content into. Names should be evocative and short (3-6 "
        "words), drawn from the user's actual interests in user-facing "
        "language — NEVER benchmark category labels. Format: 'Three "
        "collections from your saves: A, B, C.'"
    ),
}


def _length_guidance(task_type: str) -> str:
    if task_type == "chatbot_proactive_personalization":
        return "Length: 2–3 sentences."
    if task_type in ("over_personalization_chatbot_text",
                     "over_personalization_distractor_reject"):
        return "Length: 1–3 sentences."
    if task_type == "daily_personalized_briefing":
        return "Length: 3–5 short bullet items."
    if task_type == "repetition_fatigue_pairs":
        return "Length: 2 short labelled lines."
    if task_type == "active_mistake_prevention":
        return "Length: 1–3 sentences."
    return "Length: 1–4 sentences."


def _generate_example_response(llm: Callable[[str], str],
                               task_type: str, query: str,
                               groundtruth_preference: str,
                               prior_conversation: list | None) -> str | None:
    if not llm:
        return None
    guidance = _TASK_GUIDANCE.get(task_type, "")
    prior = ""
    if prior_conversation:
        # Compact prior turns for context. Use the last 4 turns max.
        turns = prior_conversation[-4:] if isinstance(prior_conversation, list) else []
        turn_strs = []
        for t in turns:
            role = t.get("role", "?") if isinstance(t, dict) else "?"
            content = (t.get("content") if isinstance(t, dict) else str(t)) or ""
            turn_strs.append(f"  {role}: {content[:120]}")
        if turn_strs:
            prior = "\nPrior conversation:\n" + "\n".join(turn_strs) + "\n"
    raw = llm(_EXAMPLE_GEN_PROMPT.format(
        task_type=task_type,
        task_guidance=guidance,
        query=(query or "")[:1200],
        groundtruth=(groundtruth_preference or "")[:800],
        prior=prior,
        length_guidance=_length_guidance(task_type),
    ))
    parsed = extract_json_from_response(raw) or {}
    text = parsed.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


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
        n = len(cands)
        if not isinstance(held, int) or not cands:
            return ""
        order = [held] + [i for i in range(n)
                          if i != held and i not in hard_negs] + sorted(hard_negs)
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


# ---------------------------------------------------------------------------
# Workstream I — self-check prompt
# ---------------------------------------------------------------------------

_SELF_CHECK_PROMPT = """You are scoring a candidate "gold reference" agent response \
against the rubric `avoid_overpersonalization`.

IMPORTANT — read the task framing first. The rubric only fires when the agent \
surfaces preferences that AREN'T relevant to the query. For some tasks \
personalization IS the point and should NOT be penalized:
  - "compose a post in the user's voice"        → using the user's hashtags is correct.
  - "summarize the user's DMs"                  → mentioning topics from real DMs is correct.
  - "recommend something for {{moment}}"        → drawing on user's known interests is correct.
  - "curate a collection of saves"              → grouping by user's top categories is correct.
  - "alert about trending topics user cares about" → listing matched user interests is correct.
  - "auto-reply to a friend's DM"               → matching user's voice register is correct.

For these tasks, score HIGH if the response uses the persona signal in a way \
the task explicitly asked for, and only score LOW if it crams in unrelated \
preferences (e.g. mentioning "boxing fandom" in an out-of-office reply).

For tasks that don't call for personalization (generic factual queries, \
restraint probes, refind on a non-user topic, scenarios, etc.), penalize \
ANY personalization leak.

Score 0 (worst) to 3 (best):
  3 = task-appropriate; persona used only as far as the task asks for.
  2 = mostly appropriate; one minor stretch.
  1 = visible over-personalization beyond what the task calls for.
  0 = response is dominated by personal trivia the task didn't request.

Respond with ONE fenced ```json block:
```json
{{"score": <0..3>, "reason": "<one short sentence including whether the task itself called for personalization>"}}
```

Task type: {task_type}
User query / trigger: {query}

Candidate gold response:
\"\"\"
{response}
\"\"\""""


def _run_self_check(llm: Callable[[str], str], task_type: str, query: str, response: str) -> dict:
    if not llm or not response:
        return {"score": 3, "passed": True, "reason": "(no llm available; defaulted to pass)"}
    raw = llm(_SELF_CHECK_PROMPT.format(
        task_type=task_type or "unknown",
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

_FLAW_KINDS = ("incorrect_personalization", "disliked_recent", "over_personalization", "factual_error")

# Personalization-flavored flaws (for chatbot / write-a-post tasks where
# the gold leans on persona signal).
_FLAW_KINDS_PERSONALIZATION = ("incorrect_personalization", "disliked_recent", "over_personalization")
# Factual flaws (for summary / lookup tasks where the gold is reporting
# real content — the natural failure mode is a wrong detail, not an
# off-topic persona aside).
_FLAW_KINDS_FACTUAL = ("factual_error",)

# Per-task allowlist. Tasks NOT listed fall back to the personalization set.
# Summarization / digest / lookup tasks need factual flaws; mechanically
# inserting "Follows mixed martial arts as a fan." into a DM digest reads
# absurdly because the gold doesn't have a preference reference to replace.
_TASK_FLAW_KINDS: dict[str, tuple[str, ...]] = {
    "agentic_dm_digest":               _FLAW_KINDS_FACTUAL,
    "agentic_group_dm_summary":        _FLAW_KINDS_FACTUAL,
    "agentic_proactive_daily_catchup": _FLAW_KINDS_FACTUAL,
    "agentic_trending_alert":          _FLAW_KINDS_FACTUAL,
    "agentic_vague_refind":            _FLAW_KINDS_FACTUAL,
}

_INFERIOR_PROMPT = """You are creating a paired *foil* response that mirrors a \
gold reference response but MUST introduce ONE specific flaw — preserving \
the gold's length, tone, sentence count, and structure.

CRITICAL: The foil MUST be visibly different from the gold. If you can't see \
where to make the change, expand the relevant span (one short clause max) \
to include the flaw — but do not make the foil substantially longer.

Gold reference:
\"\"\"
{response}
\"\"\"

Flaw kind: {flaw_kind}
{flaw_instruction}

Rules:
  - Make exactly ONE substantive modification: insert, replace, or augment \
    the targeted detail.
  - The foil must be visibly worse than the gold along the rubric implied by \
    the flaw kind — a reasonable reviewer should rate the gold higher.
  - Keep all other aspects (length, tone, structure, sentence count, hashtag \
    style) identical.
  - The foil MUST be a fluent, grammatical message a real human would send. \
    NEVER produce splice fragments like ", #NFL, #ZachWilson, #MattRyan" or \
    bare third-person fact-statements ("Enjoys X.", "Follows Y."). If the \
    gold is too short to weave the change naturally, REWRITE the gold so \
    the change is a natural clause within a coherent sentence.
  - Subtlety: the foil is supposed to be a wrong-but-plausible response, \
    not an obvious vandalism. A grader who didn't see the gold should still \
    parse the foil as a normal message. Avoid phrasings like "you're \
    interested in X", "you like Y" — the personalization slip should look \
    like the agent reaching for a real preference at the wrong moment, not \
    reciting a profile.
  - Do NOT add disclaimers, parenthetical notes, or commentary about the change.

Output ONE fenced ```json block:
```json
{{"text": "<the rewritten foil response — clearly different from the gold>"}}
```"""


def _texts_too_similar(a: str, b: str) -> bool:
    """True only when the rewrite is byte-identical or a trivial whitespace
    variant. The LLM frequently injects a meaningful clause mid-response
    that preserves head and tail tokens — those ARE legitimate flaws and
    should NOT be flagged as too-similar."""
    a = " ".join((a or "").split())
    b = " ".join((b or "").split())
    if not a or not b:
        return False
    return a == b


def _flaw_instruction(flaw_kind: str, evidence: dict) -> str:
    if flaw_kind == "incorrect_personalization":
        pi = evidence.get("persona_item", "")
        return (
            f"Rewrite ONE clause or sentence in the gold so it naturally "
            f"references this persona item: \"{pi}\". The persona item IS "
            f"true for the user but is irrelevant to the current query — "
            f"the agent has drifted off-topic. The reference must read as "
            f"fluent prose: integrated as part of a clause (e.g. \"btw, did "
            f"you catch the comedy clips this weekend?\"), NOT as a separate "
            f"fact-statement, NOT parenthetical, NOT a third-person "
            f"description like \"Enjoys X.\" or \"Follows Y.\". If the gold "
            f"is too short to weave naturally, you may extend it by one "
            f"short clause — but the result must still sound like one "
            f"coherent message a human would write."
        )
    if flaw_kind == "disliked_recent":
        pi = evidence.get("persona_item", "")
        return (
            f"Rewrite the gold to mention this topic naturally in passing: "
            f"\"{pi}\". The user explicitly DISLIKED this in the last 48 "
            f"hours — the foil agent shouldn't have brought it up. The "
            f"mention must read fluently (a clause woven into an existing "
            f"sentence, or one extra grammatical sentence), NOT a "
            f"third-person fact-statement spliced in."
        )
    if flaw_kind == "over_personalization":
        pi = evidence.get("persona_item", "")
        return (
            f"Rewrite ONE clause or sentence in the gold to add an "
            f"unprompted aside about: \"{pi}\". This is a top user category "
            f"but has zero overlap with the current query — a digression "
            f"the gold correctly omitted. The aside must be a grammatical "
            f"clause integrated into the prose (e.g. \"...and the {pi.split(',')[0] if pi else 'topic'} "
            f"crowd is heating up\"), NOT a separate fact-statement, NOT a "
            f"prepended/appended fragment."
        )
    if flaw_kind == "factual_error":
        return (
            "Introduce ONE subtle factual error inside the gold's own "
            "content — for example: change a sender name to a different "
            "plausible name, swap one topic for another, change a count "
            "(e.g. 'three' → 'two'), drop one of the items the gold lists, "
            "or attribute a message to the wrong person. The error should "
            "look like a careless mistake a hurried summarizer would make, "
            "not an off-topic personalization leak. Do NOT introduce any "
            "persona reference."
        )
    return ""


def _pick_flaw_evidence(flaw_kind: str, inst: dict, persona_ctx: dict,
                        rng: random.Random) -> dict | None:
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
        # Real explicit_negative row in the t_test - 48h window.
        recent_negs = persona_ctx.get("recent_negatives") or []
        if not recent_negs:
            return None
        evidence = rng.choice(recent_negs)
        return {
            "persona_item": evidence.get("persona_item", "") or
                            ", ".join((evidence.get("hashtags") or [])[:3]),
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
    return None


def _generate_inferior(llm: Callable[[str], str], response: str,
                       flaw_kind: str, evidence: dict) -> str | None:
    if not llm or not response or not evidence:
        return None
    prompt = _INFERIOR_PROMPT.format(
        response=response[:1500],
        flaw_kind=flaw_kind,
        flaw_instruction=_flaw_instruction(flaw_kind, evidence),
    )
    # Try up to 2x — sometimes the model copies the original verbatim
    # because the response is short / structured. Retry with a stronger
    # nudge on the second attempt.
    for attempt in range(2):
        raw = llm(prompt if attempt == 0 else prompt + (
            "\n\nReminder: your previous attempt was identical to the gold. "
            "You MUST make a visible textual change introducing the flaw."
        ))
        parsed = extract_json_from_response(raw) or {}
        text = parsed.get("text") or None
        if text and not _texts_too_similar(response, text):
            return text
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


def _enrich_groundtruth_for_task(inst: dict, task_id: str, bq, user_id: str,
                                  base_groundtruth: str) -> str:
    """Some tasks need richer real-data grounding than the default
    persona-context lines provide. Auto-reply needs sender relationship +
    thread history; DM digest needs the actual thread headers; write-post
    tasks need the per-app tone definition. Returns the augmented
    groundtruth string (base + enrichment lines)."""
    try:
        # Per-app tone is the canonical voice source for any task that
        # composes a post or message in the user's voice. Layer it in
        # FIRST so all task-specific enrichments build on top.
        out = _prepend_app_tone(inst, task_id, bq, user_id, base_groundtruth)
        if task_id == "agentic_auto_reply":
            return _enrich_auto_reply(inst, bq, user_id, out)
        if task_id == "agentic_dm_digest":
            return _enrich_dm_digest(inst, bq, user_id, out)
        if task_id == "agentic_group_dm_summary":
            return _enrich_group_dm_summary(inst, bq, user_id, out)
        return out
    except Exception:
        pass
    return base_groundtruth


# Tasks where the agent composes content in the user's voice for a target
# app. The per-app tone is the rubric's authority for voice_match — if it
# isn't piped into the gold-gen prompt the LLM has no consistent reference
# to write against, and grades drift turn-to-turn.
_VOICE_DEPENDENT_TASKS: set[str] = {
    "agentic_user_tone_post",
    "agentic_composed_post",
    "agentic_send_post",
    "agentic_cross_app_repost",
    "agentic_auto_reply",
    "agentic_dm_digest",
    "agentic_group_dm_summary",
    "agentic_proactive_daily_catchup",
    "agentic_trending_alert",
}


def _prepend_app_tone(inst: dict, task_id: str, bq, user_id: str, base: str) -> str:
    """If this task composes content in the user's voice, prepend the
    per-app tone block from profile.app_personas. The pipeline already
    synthesizes a per-app `style_description` (and `topical_focus`,
    `posting_frequency`) — surfacing it here makes voice_match measurable
    against a fixed reference instead of an inferred-from-samples one."""
    if task_id not in _VOICE_DEPENDENT_TASKS:
        return base
    target_app = (inst.get("target_app") or "").strip()
    # Auto-reply / DM digest / group summary use the social app of the
    # message; if not set, fall back to chatbot tone for chat-routed tasks.
    if not target_app:
        target_app = "chatbot" if (inst.get("entry_point") == "chatbot_routed") else ""
    if not target_app:
        return base
    profile = bq.get_full_profile(user_id) or {}
    apps = profile.get("app_personas") or {}
    # app_personas keys are capitalized in the profile; normalize lookup.
    app_persona = None
    for k, v in apps.items():
        if (k or "").lower() == target_app.lower():
            app_persona = v
            break
    if not isinstance(app_persona, dict):
        return base
    style = (app_persona.get("style_description") or "").strip()
    if not style:
        return base
    lines = [
        f"USER'S VOICE ON {target_app.upper()} (canonical reference for tone "
        f"— write to match this, not the voice_sample):",
        f"  Style: {style}",
    ]
    topical = app_persona.get("topical_focus") or []
    if topical:
        lines.append(f"  Typical topics: {', '.join(topical[:5])}")
    freq = app_persona.get("posting_frequency")
    if freq:
        lines.append(f"  Posting cadence: {freq}")
    return "\n".join(lines) + "\n\n" + base


def _enrich_auto_reply(inst: dict, bq, user_id: str, base: str) -> str:
    sender_id = (inst.get("sender_id") or "").strip()
    thread_id = (inst.get("thread_id") or "").strip()
    target_app = (inst.get("target_app") or "").strip()

    profile = bq.get_full_profile(user_id) or {}
    friends = profile.get("friends", []) or []
    friend = None
    for f in friends:
        if (f.get("friend_id") == sender_id or
            (f.get("display_name") or "").lower() == sender_id.lower()):
            friend = f
            break

    if friend:
        rel_line = (
            f"Sender RELATIONSHIP: {friend.get('relationship_depth', 'friend')} "
            f"friend named {friend.get('display_name', sender_id)}; shared interests: "
            f"{', '.join((friend.get('shared_interests') or [])[:3]) or 'none listed'}. "
            f"Reply in the casual tone the user uses with this friend."
        )
    elif "stranger" in sender_id.lower() or sender_id.startswith("unknown"):
        rel_line = (
            f"Sender RELATIONSHIP: STRANGER ({sender_id}) — NOT in the user's "
            f"friends list. If the inbound is unsolicited (spam, scam, sales "
            f"pitch, recruiter cold-DM), the user would either ignore or "
            f"decline briefly — they would NOT respond as if to a friend."
        )
    else:
        rel_line = (
            f"Sender RELATIONSHIP: '{sender_id}' is not in the user's "
            f"friends list. Treat as an acquaintance — reply briefly and "
            f"appropriately to the message content; do NOT volunteer "
            f"personal context."
        )

    thread_lines: list[str] = []
    if thread_id and target_app:
        thread = bq.get_dm_thread(
            user_id=user_id, app=target_app, thread_id=thread_id, limit=10,
        ) or {}
        msgs = thread.get("results") or thread.get("messages") or []
        if msgs:
            thread_lines.append(
                f"Recent thread context (last {min(len(msgs), 4)} messages, "
                f"oldest first):"
            )
            for m in msgs[-4:]:
                role = "User" if m.get("sender") == "self" else (
                    friend.get("display_name") if friend else m.get("sender", "other")
                )
                text = (m.get("text") or "")[:140]
                thread_lines.append(f"  {role}: {text}")

    extras = [rel_line] + thread_lines
    return base + ("\n" + "\n".join(extras) if extras else "")


def _enrich_dm_digest(inst: dict, bq, user_id: str, base: str) -> str:
    target_app = (inst.get("target_app") or "").strip()
    t_test = int(inst.get("t_test") or 0) or None
    if not target_app:
        return base
    page = bq.list_dm_threads(
        user_id=user_id, app=target_app, since_timestamp=t_test, limit=8,
    ) or {}
    threads = page.get("results") or []
    if not threads:
        return base
    profile = bq.get_full_profile(user_id) or {}
    friends_by_id = {
        f.get("friend_id"): f for f in (profile.get("friends") or [])
    }
    lines = [f"Recent DM threads on {target_app} (real headers — pick 2-4 "
             f"to mention in the digest):"]
    for t in threads[:6]:
        parts = t.get("participants") or []
        names = []
        for pid in parts:
            f = friends_by_id.get(pid)
            names.append(f.get("display_name") if f else pid)
        names_str = ", ".join([n for n in names if n][:3]) or "unknown"
        is_group = t.get("is_group")
        prev = (t.get("last_message_preview") or "")[:80]
        kind = "group" if is_group else "1-1"
        lines.append(f"  - {kind} with {names_str}: {prev}")
    return base + "\n" + "\n".join(lines)


def _enrich_group_dm_summary(inst: dict, bq, user_id: str, base: str) -> str:
    target_app = (inst.get("target_app") or "").strip()
    thread_id = (inst.get("thread_id") or "").strip()
    if not (target_app and thread_id):
        return base
    thread = bq.get_dm_thread(
        user_id=user_id, app=target_app, thread_id=thread_id, limit=20,
    ) or {}
    msgs = thread.get("results") or thread.get("messages") or []
    if not msgs:
        return base
    profile = bq.get_full_profile(user_id) or {}
    friends_by_id = {f.get("friend_id"): f for f in (profile.get("friends") or [])}
    lines = ["Recent thread messages (oldest → newest):"]
    for m in msgs[-8:]:
        sid = m.get("sender")
        f = friends_by_id.get(sid)
        name = "User" if sid == "self" else (f.get("display_name") if f else sid)
        text = (m.get("text") or "")[:120]
        lines.append(f"  {name}: {text}")
    return base + "\n" + "\n".join(lines)


def _build_persona_ctx(bq, user_id: str, t_test: int) -> dict:
    """Lightweight persona context for inferior generation: top prefs,
    top categories, real negative engagements in the last 48h."""
    from collections import Counter
    DAY = 24 * 3600
    pref_counts: Counter = Counter()
    cat_counts: Counter = Counter()
    recent_negs: list[dict] = []
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
    }


# ---------------------------------------------------------------------------
# Orchestrator entry point
# ---------------------------------------------------------------------------

def postprocess_benchmark(bm: dict, bq, user_id: str,
                          self_check_llm: Callable[[str], str] | None = None,
                          inferior_llm: Callable[[str], str] | None = None,
                          rng_seed: int = 0,
                          verbose: bool = False) -> dict:
    """Run workstream I (self-check) + J (inferior_response) over every
    personalization instance in the assembled benchmark dict. Mutates
    instances in place; returns the same dict for chaining.
    """
    rng = random.Random(rng_seed or hash(user_id) % (2**31))
    n_example_llm_gen = 0
    n_example_ranking = 0
    n_self_check = 0
    n_self_check_failed = 0
    n_inferior_built = 0
    n_inferior_skipped = 0

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
            # concrete one. Three dispatch paths:
            #   - already-concrete: skip (agentic builders already write
            #     real text via the persona-context templates).
            #   - ranking task: deterministic compute (no LLM).
            #   - everything else (chatbot, restraint, scenarios, E3/E6):
            #     LLM-generate from the persona signal in groundtruth.
            user_query = _synthesize_user_query(inst, task_id)
            prior_conv = inst.get("prior_conversation")
            if task_id in _RANKING_TASKS:
                ranked = _compute_ranking_example(inst, task_id)
                if ranked:
                    example = ranked
                    inst["example_response"] = example
                    n_example_ranking += 1
            elif task_id not in _TASKS_ALREADY_CONCRETE and self_check_llm is not None:
                # Per-task groundtruth enrichment — pull real friend / thread
                # data from the backend so the LLM grounds the reply in the
                # user's actual relationships, not just their public-post
                # voice. Cheap (no LLM); skipped for tasks without a hook.
                enriched_gt = _enrich_groundtruth_for_task(
                    inst, task_id, bq, user_id, groundtruth,
                )
                generated = _generate_example_response(
                    self_check_llm, task_id, user_query, enriched_gt, prior_conv,
                )
                if generated:
                    example = generated
                    inst["example_response"] = example
                    n_example_llm_gen += 1

            if not example:
                continue

            # Workstream I: self-check
            if self_check_llm is not None:
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
            if (inferior_llm is not None
                    and arm in ("proactive", "contradiction")
                    and task_id not in _TASKS_NO_FOIL):
                t_test = int(inst.get("t_test") or 0)
                if t_test not in _ctx_cache:
                    _ctx_cache[t_test] = _build_persona_ctx(bq, user_id, t_test)
                ctx = _ctx_cache[t_test]
                # Task-aware flaw selection: summary / lookup tasks get
                # factual flaws; everything else uses the personalization
                # set. Round-robin within the allowed pool for coverage.
                allowed_flaws = _TASK_FLAW_KINDS.get(task_id, _FLAW_KINDS_PERSONALIZATION)
                flaw_kind = allowed_flaws[(idx + len(items)) % len(allowed_flaws)]
                evidence = _pick_flaw_evidence(flaw_kind, inst, ctx, rng)
                if evidence is None:
                    # Fallback: try the other kinds in the allowed pool.
                    for fk in allowed_flaws:
                        if fk == flaw_kind:
                            continue
                        evidence = _pick_flaw_evidence(fk, inst, ctx, rng)
                        if evidence is not None:
                            flaw_kind = fk
                            break
                if evidence is None:
                    n_inferior_skipped += 1
                    continue
                text = _generate_inferior(inferior_llm, example, flaw_kind, evidence)
                if text:
                    inst["inferior_response"] = {
                        "text": text,
                        "flaw_kind": flaw_kind,
                        "flaw_evidence": evidence,
                    }
                    n_inferior_built += 1
                else:
                    n_inferior_skipped += 1

    if verbose:
        print(f"[llm_postprocess] example_llm_gen={n_example_llm_gen} "
              f"example_ranking={n_example_ranking} "
              f"self_check={n_self_check} self_check_failed={n_self_check_failed} "
              f"inferior_built={n_inferior_built} "
              f"inferior_skipped={n_inferior_skipped}")
    bm["postprocess_stats"] = {
        "example_llm_gen": n_example_llm_gen,
        "example_ranking": n_example_ranking,
        "self_check_total": n_self_check,
        "self_check_failed": n_self_check_failed,
        "inferior_built": n_inferior_built,
        "inferior_skipped": n_inferior_skipped,
    }
    return bm
