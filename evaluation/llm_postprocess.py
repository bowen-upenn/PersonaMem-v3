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
    "agentic_user_voice_post",
    "agentic_moment_recommendation",
    "agentic_dm_digest",
    "agentic_cross_app_repost",
    "agentic_auto_reply",
    "agentic_vague_refind",
    "agentic_composed_post",
    "agentic_chatbot_dispatch",
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
_TASKS_ALREADY_CONCRETE = {
    "agentic_user_voice_post",
    "agentic_moment_recommendation",
    "agentic_dm_digest",
    "agentic_cross_app_repost",
    "agentic_auto_reply",
    "agentic_vague_refind",
    "agentic_composed_post",
    "agentic_chatbot_dispatch",
    "agentic_collection_curation",
    "agentic_group_dm_summary",
    "agentic_wrong_recipient_check",
    "agentic_proactive_daily_catchup",
    "agentic_trending_alert",
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

Persona signal the agent may draw on (REAL data):
\"\"\"{groundtruth}\"\"\"
{prior}
Write the actual response — natural, concrete, the words the agent would say.
{length_guidance}

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

_FLAW_KINDS = ("incorrect_personalization", "disliked_recent", "over_personalization")

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
  - The foil must be a *worse* response than the gold along the
    `avoid_overpersonalization` rubric — a reasonable reviewer should rate
    the gold higher.
  - Keep all other aspects (length, tone, structure, sentence count, hashtag
    style) identical.
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
        return (f"Replace any preference reference in the gold with a single "
                f"reference to: \"{pi}\". This persona item IS true for the "
                f"user but is unrelated to the current query.")
    if flaw_kind == "disliked_recent":
        pi = evidence.get("persona_item", "")
        return (f"Inject one mention of: \"{pi}\". The user explicitly "
                f"DISLIKED this in the last 48 hours. The mention should "
                f"feel natural, not forced — but it must be present.")
    if flaw_kind == "over_personalization":
        pi = evidence.get("persona_item", "")
        return (f"Add one unprompted aside referencing: \"{pi}\". This is "
                f"a top user category but has zero overlap with the current "
                f"query — it's a digression the gold reference correctly "
                f"omitted.")
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
            user_query = inst.get("user_query") or inst.get("query") or ""
            prior_conv = inst.get("prior_conversation")
            if task_id in _RANKING_TASKS:
                ranked = _compute_ranking_example(inst, task_id)
                if ranked:
                    example = ranked
                    inst["example_response"] = example
                    n_example_ranking += 1
            elif task_id not in _TASKS_ALREADY_CONCRETE and self_check_llm is not None:
                # Reuse the same LLM client for example-gen.
                generated = _generate_example_response(
                    self_check_llm, task_id, user_query, groundtruth, prior_conv,
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
            if inferior_llm is not None and arm in ("proactive", "contradiction"):
                t_test = int(inst.get("t_test") or 0)
                if t_test not in _ctx_cache:
                    _ctx_cache[t_test] = _build_persona_ctx(bq, user_id, t_test)
                ctx = _ctx_cache[t_test]
                # Round-robin flaw kind across instances for uniform coverage.
                flaw_kind = _FLAW_KINDS[(idx + len(items)) % len(_FLAW_KINDS)]
                evidence = _pick_flaw_evidence(flaw_kind, inst, ctx, rng)
                if evidence is None:
                    # Fallback: try the other kinds.
                    for fk in _FLAW_KINDS:
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
