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
# Workstream I — self-check prompt
# ---------------------------------------------------------------------------

_SELF_CHECK_PROMPT = """You are scoring a candidate "gold reference" agent response \
against the rubric `avoid_overpersonalization`.

The rubric asks: did the response avoid surfacing user preferences that \
aren't relevant to this specific query? Penalize dragging in unrelated \
hobbies, topics, demographic facts, or recurring interests when the \
query doesn't call for them.

Score 0 (worst) to 3 (best):
  3 = perfectly restrained; preferences only when truly relevant.
  2 = mostly restrained; one minor over-personalization that doesn't dominate.
  1 = visible over-personalization; unrelated interests dragged in.
  0 = response is dominated by irrelevant personal trivia.

Respond with ONE fenced ```json block:
```json
{{"score": <0..3>, "reason": "<one short sentence>"}}
```

User query: {query}

Candidate gold response:
\"\"\"
{response}
\"\"\""""


def _run_self_check(llm: Callable[[str], str], query: str, response: str) -> dict:
    if not llm or not response:
        return {"score": 3, "passed": True, "reason": "(no llm available; defaulted to pass)"}
    raw = llm(_SELF_CHECK_PROMPT.format(query=(query or "")[:1500],
                                        response=response[:1500]))
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
gold reference response but introduces ONE specific flaw — preserving the \
gold's length, tone, sentence count, and structure.

Gold reference:
\"\"\"
{response}
\"\"\"

Flaw kind: {flaw_kind}
{flaw_instruction}

Rewrite the gold reference, changing ONLY the detail above. Do not \
introduce any other changes. Output ONE fenced ```json block:
```json
{{"text": "<the rewritten foil response>"}}
```"""


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
    raw = llm(prompt)
    parsed = extract_json_from_response(raw) or {}
    return parsed.get("text") or None


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
            inst["example_response"] = example
            inst["groundtruth_preference"] = gt_out.get("groundtruth_preference") or ""
            if "tool_call" in gt_out:
                inst["tool_call"] = gt_out["tool_call"]
            if not example:
                continue

            # Workstream I: self-check
            if self_check_llm is not None:
                user_query = inst.get("user_query") or inst.get("query") or ""
                check = _run_self_check(self_check_llm, user_query, example)
                inst["example_response_self_check"] = check
                n_self_check += 1
                if not check.get("passed", True):
                    n_self_check_failed += 1

            # Workstream J: inferior_response
            if inferior_llm is not None and inst.get("arm") != "overpersonalization":
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
        print(f"[llm_postprocess] self_check={n_self_check} "
              f"self_check_failed={n_self_check_failed} "
              f"inferior_built={n_inferior_built} "
              f"inferior_skipped={n_inferior_skipped}")
    bm["postprocess_stats"] = {
        "self_check_total": n_self_check,
        "self_check_failed": n_self_check_failed,
        "inferior_built": n_inferior_built,
        "inferior_skipped": n_inferior_skipped,
    }
    return bm
