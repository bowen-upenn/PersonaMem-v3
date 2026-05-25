"""Step 4.8 — hidden_persona_recommendation.

Ranking task where ALL 16 slate items are LLM-generated general content
(no topic overlap with canonical preferences) and exactly ONE item
subtly resonates with a hidden persona.  Tests whether the agent has
inferred deep motivational patterns well enough to rank the right
content above generic alternatives.

Uses the same slate format, runtime prompt, and deterministic ranking
metrics as ``personalized_recommendation`` — the only difference is how
the slate is built (LLM-synthesised at build time vs. drawn from real
backend events).

Eligibility filter for a hidden persona at T_test:
  - evidence_rows >= 25
  - evidence_row_fraction >= 0.015
  - last_seen_ts within the last 30 days before T_test

Instances per user: one per eligible hidden persona, cap at 6.
Slate size: 16 items (1 target + 15 distractors).
Random recall@1 baseline: 6.25%.

Privacy-flagged types: target resonates via adjacent content only
(same constraint as hidden_persona_implicit_qa).
"""

from __future__ import annotations

import datetime as _dt
import json
import random
import re
from typing import Any

from data_preparation.utils import extract_json_from_response
from evaluation.backend_query import BackendQuery


DAY_SECONDS = 24 * 60 * 60

SLATE_SIZE = 16
INSTANCES_PER_USER_CAP = 6

MIN_EVIDENCE_ROWS = 25
MIN_EVIDENCE_ROW_FRACTION = 0.015
MAX_DAYS_SINCE_LAST_SEEN = 30
RECENT_EVIDENCE_DAYS = 7

_TASK_ID = "hidden_persona_recommendation"

PRIVACY_FLAGGED_TYPES = {
    "intimate_interest",
    "covert_concern",
    "compensatory_need",
    "medical_aesthetic_concern",
    "sensitive_life_event",
}

_DISCOVERY_RETRIES = 1

# ---------------------------------------------------------------------------
# Text helpers (shared with hidden_persona_implicit_qa)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b[\w']+\b")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "his", "her",
    "their", "they", "them", "he", "she", "we", "us", "you", "your", "i",
    "me", "my", "mine", "ours", "yours", "theirs", "not", "no", "do", "does",
    "did", "have", "has", "had", "can", "could", "would", "should", "will",
    "may", "might", "must", "about", "into", "over", "under", "more", "most",
    "than", "then", "so", "if", "when", "while", "what", "who", "whom",
    "where", "why", "how", "any", "all", "some", "such",
}


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def _content_tokens(text: str) -> list[str]:
    return [t for t in _tokenize(text) if t not in _STOPWORDS]


def _word_count(text: str) -> int:
    return len(_tokenize(text))


# ---------------------------------------------------------------------------
# Discovery prompt (build-time mini-tier LLM call)
# ---------------------------------------------------------------------------

DISCOVERY_PROMPT = """You are designing a content recommendation benchmark slate. The slate tests whether an AI agent can identify which piece of general-interest content a specific user would most enjoy, based on a hidden motivational pattern the agent has inferred from the user's behavioral history.

You will receive:
  1. A hidden-persona record — a deeper behavioral pattern inferred from cross-row hashtag analysis
  2. The user's canonical preferences — the slate must NOT overlap with these topics
  3. Other hidden personas for this user — distractors must not accidentally align with any of them

Your job: produce EXACTLY 16 content items + indicate which ONE subtly resonates with the hidden persona.

## Hidden persona record

  label:               {label}
  type:                {type}
  description:         {description}
  inferred_motivation: {inferred_motivation}
  surface_connections: {surface_connections}
  evidence hashtags:   {evidence_hashtags}
  is_privacy_flagged:  {is_privacy_flagged}

## Other hidden personas for this user (distractors must AVOID resonating with any of these)

{other_hidden_personas_block}

## User's canonical preferences (ALL 16 items must AVOID these topics)

{canonical_preferences_block}

## Hard constraints

(A) ALL 16 items must be GENERAL-INTEREST content — topics any adult might find mildly interesting. NONE may overlap with the user's canonical preferences listed above. No boxing, wrestling, MMA, hip-hop, breakdancing, or any topic appearing in the preference list. Check EVERY preference.
(B) Exactly ONE item (at target_index) subtly resonates with the hidden persona's DEEPER MOTIVATIONAL PATTERN. The connection is thematic/motivational, NOT through shared hashtags or topic keywords.
(C) The target must NOT contain any of the persona's evidence hashtags (bare, case-insensitive) in its title, caption, or hashtags.
(D) The target must NOT contain the persona label or type verbatim in its title or caption.
(E) The 15 distractors must each be a plausible general-interest recommendation. NONE may resonate with the target hidden persona OR any of the other hidden personas listed above.
(F) If is_privacy_flagged: the target must reach the deeper need via ADJACENT content only — must not name or evoke the sensitive topic.
(G) The target's resonance must be SUBTLE — a reader unfamiliar with the hidden persona would see it as just another general item. Only an agent that has inferred the motivational pattern would rank it above the rest.
(H) DIVERSITY: use at least 6 different content domains across the 16 items. No two items may share the same narrow topic. Vary tone, scope, and content type (articles, videos, podcasts, documentaries, tutorials, essays, etc.).
(I) QUALITY PARITY: all 16 items must be comparable in specificity and appeal. No strawman distractors (too boring, too niche, too vague). Each description must name specific (fictional) subjects, places, or concepts.
(J) Each item has:
    title    : 5-15 word descriptive title
    caption  : 1-2 sentences, 15-40 words, specific and engaging
    hashtags : 3-5 hashtags (no leading #)

## Per-type guidance for the target item

Calibrate the target's resonance based on `type`:
  - personality_trait        -> content whose FORMAT or STRUCTURE mirrors the trait (e.g., head-to-head elimination format for a competition-admirer)
  - aspiration               -> content touching the aspirational domain from a fresh angle
  - emotional_pattern        -> content creating the same affective scaffolding (e.g., public-recognition narrative for visible-affirmation seeker)
  - identity_anchor          -> content respecting the user's values code via an analogous domain
  - intimate_interest        -> PRIVACY-FLAGGED — adjacent-domain content only
  - intellectual_curiosity   -> content going one layer deeper into mechanism / history
  - private_hobby            -> content inviting craft-process engagement indirectly
  - parasocial_attachment    -> content invoking the figure's world/aesthetic without naming them
  - compensatory_need        -> PRIVACY-FLAGGED — serves underlying need without naming gap
  - covert_concern           -> PRIVACY-FLAGGED — soothes the worry without naming it
  - medical_aesthetic_concern-> PRIVACY-FLAGGED — adjacent content only
  - sensitive_life_event     -> PRIVACY-FLAGGED — respects active window without naming event

## Worked example

Persona: "Battle-format performance admiration beyond sports"
  type: personality_trait
  evidence hashtags: BattleRap, DancePerformance, WWE, Wrestling
Canonical prefs include boxing, wrestling, MMA, breakdancing, hip-hop.

  target_index: 3
  Items [0]-[2], [4]-[15] are diverse general content (Roman gardens, weather stations, color psychology, lighthouses, ice harvesting, origami, composting, sleep science, coral reefs, sourdough history, typewriter design, constellation guides, fermentation, train architecture, sign lettering).
  Item [3]: "The Great British Cheese Rivalry: County Showdowns That Shape Artisan Dairy"
    caption: "A documentary following four cheesemakers through elimination-style head-to-head judging rounds where decades of craft reputation ride on a single blind tasting."
    hashtags: [cheese, competition, artisan, craft, judging]
  resonance_signal: "The cheese competition documentary mirrors the cross-domain pull toward head-to-head formats where identity is proven through craft under pressure — same motivational structure as boxing/wrestling/battle-rap but in an unrelated domain."

## Output

Return EXACTLY one JSON object inside a fence. No prose outside the fence.

```json
{{
  "target_index": <0-15>,
  "resonance_signal": "1-2 sentences explaining the motivational connection",
  "items": [
    {{"title": "...", "caption": "...", "hashtags": ["...", "...", "..."]}},
    ... // exactly 16 items total
  ]
}}
```
"""


def _format_discovery_prompt(
    hp: dict,
    is_privacy_flagged: bool,
    other_personas: list[dict],
    canonical_preferences: list[str],
) -> str:
    evidence_hashtags = (hp.get("evidence_hashtags") or [])[:8]
    surface_connections = hp.get("surface_connections") or []

    other_block_lines = []
    for other in other_personas:
        other_block_lines.append(f"  - {other.get('type', '')}: {other.get('label', '')}")
    other_block = "\n".join(other_block_lines) if other_block_lines else "(none)"

    pref_lines = []
    for p in canonical_preferences:
        text = p.split(" : ", 1)[-1] if " : " in p else p
        pref_lines.append(f"  - {text.strip()}")
    pref_block = "\n".join(pref_lines) if pref_lines else "(none)"

    return DISCOVERY_PROMPT.format(
        label=hp.get("label", ""),
        type=hp.get("type", ""),
        description=hp.get("description", ""),
        inferred_motivation=hp.get("inferred_motivation", ""),
        surface_connections="; ".join(surface_connections) if surface_connections else "(none)",
        evidence_hashtags=", ".join(evidence_hashtags) if evidence_hashtags else "(none)",
        is_privacy_flagged=str(is_privacy_flagged).lower(),
        other_hidden_personas_block=other_block,
        canonical_preferences_block=pref_block,
    )


# ---------------------------------------------------------------------------
# Validation (build-time post-LLM)
# ---------------------------------------------------------------------------

def _validate_discovery_output(
    parsed: dict,
    hp: dict,
    is_privacy_flagged: bool,
    canonical_preferences: list[str],
) -> tuple[bool, str]:
    items = parsed.get("items")
    if not isinstance(items, list) or len(items) != SLATE_SIZE:
        return False, f"items must be a list of exactly {SLATE_SIZE}; got {type(items).__name__} len={len(items) if isinstance(items, list) else '?'}"

    target_index = parsed.get("target_index")
    if not isinstance(target_index, int) or not (0 <= target_index < SLATE_SIZE):
        return False, f"target_index must be int in [0, {SLATE_SIZE - 1}]; got {target_index!r}"

    resonance = parsed.get("resonance_signal")
    if not isinstance(resonance, str) or not resonance.strip():
        return False, "resonance_signal must be a non-empty string"

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            return False, f"items[{i}] is not a dict"
        for key in ("title", "caption"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                return False, f"items[{i}] missing or empty field: {key!r}"
        hashtags = item.get("hashtags")
        if not isinstance(hashtags, list) or not (2 <= len(hashtags) <= 8):
            return False, f"items[{i}] hashtags must be a list of 2-8 strings; got {len(hashtags) if isinstance(hashtags, list) else '?'}"
        for h in hashtags:
            if not isinstance(h, str) or not h.strip():
                return False, f"items[{i}] has empty/non-string hashtag"

        title_wc = _word_count(item["title"])
        if not (3 <= title_wc <= 25):
            return False, f"items[{i}] title has {title_wc} words; expected 3-25"

        caption_wc = _word_count(item["caption"])
        if not (8 <= caption_wc <= 80):
            return False, f"items[{i}] caption has {caption_wc} words; expected 8-80"

    # Hashtag diversity: at least 20 unique hashtags across all items.
    all_tags = set()
    for item in items:
        for h in item.get("hashtags", []):
            all_tags.add(h.lower().strip())
    if len(all_tags) < 20:
        return False, f"only {len(all_tags)} unique hashtags across 16 items; need ≥20 for domain diversity"

    target = items[target_index]
    label = (hp.get("label") or "").strip()
    label_lc = label.lower()
    type_val = (hp.get("type") or "").strip().lower()

    target_text = (target["title"] + " " + target["caption"]).lower()

    if label_lc and label_lc in target_text:
        return False, f"target contains persona label verbatim: {label!r}"

    if type_val and type_val.replace("_", " ") in target_text:
        return False, f"target contains persona type verbatim: {type_val!r}"

    evidence_hashtags = {(h or "").lstrip("#").lower() for h in (hp.get("evidence_hashtags") or []) if h}
    target_hashtags = {(h or "").lstrip("#").lower() for h in target.get("hashtags", []) if h}
    leaked = evidence_hashtags & target_hashtags
    if leaked:
        return False, f"target hashtags contain evidence hashtag(s): {leaked}"

    if is_privacy_flagged:
        for eh in evidence_hashtags:
            if eh and eh in target_text:
                return False, f"privacy-flagged: evidence hashtag {eh!r} appears in target title/caption"

    # Canonical preference keyword leakage check.
    for item in items:
        item_tokens = set(_content_tokens(
            item["title"] + " " + item["caption"] + " " + " ".join(item.get("hashtags", []))
        ))
        for pref in canonical_preferences:
            pref_text = pref.split(" : ", 1)[-1] if " : " in pref else pref
            pref_tokens = set(_content_tokens(pref_text))
            overlap = item_tokens & pref_tokens
            if len(overlap) >= 3:
                return False, (
                    f"item '{item['title'][:40]}' shares ≥3 content tokens "
                    f"with canonical preference '{pref_text[:60]}': {overlap}"
                )

    return True, ""


def _build_corrective_prompt(base_prompt: str, violation: str) -> str:
    return (
        base_prompt
        + "\n\n## Your last attempt failed validation\n\n"
        + f"Violation: {violation}\n\n"
        + "Try again, keeping every other constraint the same. "
        "Return ONE JSON object inside a fenced block, no prose outside."
    )


# ---------------------------------------------------------------------------
# Eligibility + builder
# ---------------------------------------------------------------------------

def _filter_eligible_personas(
    profile: dict, t_test: int,
) -> list[dict]:
    out: list[dict] = []
    for hp in profile.get("hidden_personas") or []:
        if not isinstance(hp, dict):
            continue
        if hp.get("evidence_rows", 0) < MIN_EVIDENCE_ROWS:
            continue
        if hp.get("evidence_row_fraction", 0.0) < MIN_EVIDENCE_ROW_FRACTION:
            continue
        last_seen = hp.get("last_seen_ts") or 0
        if not last_seen:
            continue
        if (t_test - last_seen) > MAX_DAYS_SINCE_LAST_SEEN * DAY_SECONDS:
            continue
        out.append(hp)
    return out


def _hp_is_privacy_flagged(hp: dict) -> bool:
    return (hp.get("type") or "").lower() in PRIVACY_FLAGGED_TYPES


def _t_test_anchor(profile: dict, t_now: int) -> int:
    candidate = t_now - RECENT_EVIDENCE_DAYS * DAY_SECONDS
    return max(candidate, 0) or t_now


def _discover_slate(
    discovery_llm: Any,
    hp: dict,
    is_privacy_flagged: bool,
    other_personas: list[dict],
    canonical_preferences: list[str],
    verbose: bool = False,
) -> dict | None:
    base_prompt = _format_discovery_prompt(
        hp, is_privacy_flagged, other_personas, canonical_preferences,
    )

    raw = discovery_llm.query_llm(base_prompt)
    parsed = extract_json_from_response(raw) or {}
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    if isinstance(parsed, dict):
        ok, why = _validate_discovery_output(parsed, hp, is_privacy_flagged, canonical_preferences)
        if ok:
            return parsed
    else:
        ok, why = False, f"LLM returned non-object JSON: {type(parsed).__name__}"

    if verbose:
        print(
            f"[hidden_persona_recommendation] retry persona "
            f"{hp.get('label')!r}: {why}"
        )

    for _ in range(_DISCOVERY_RETRIES):
        corrective = _build_corrective_prompt(base_prompt, why)
        raw = discovery_llm.query_llm(corrective)
        parsed = extract_json_from_response(raw) or {}
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else {}
        if isinstance(parsed, dict):
            ok, why = _validate_discovery_output(parsed, hp, is_privacy_flagged, canonical_preferences)
            if ok:
                return parsed
        else:
            why = f"LLM returned non-object JSON: {type(parsed).__name__}"

    if verbose:
        print(
            f"[hidden_persona_recommendation] dropping persona "
            f"{hp.get('label')!r} after {_DISCOVERY_RETRIES + 1} attempts: {why}"
        )
    return None


def _items_to_candidates(items: list[dict]) -> list[dict]:
    return [
        {
            "source_object_id": "",
            "title": (item.get("title") or "")[:120],
            "caption": (item.get("caption") or "")[:200],
            "hashtags": [h.lstrip("#") for h in (item.get("hashtags") or [])[:8]],
            "source_app": "",
            "source_timestamp": 0,
        }
        for item in items
    ]


def build_hidden_persona_recommendation(
    bq: BackendQuery,
    user_id: str,
    t_now: int,
    discovery_llm=None,
    rng_seed: int = 0,
    verbose: bool = False,
) -> list[dict]:
    """Build hidden_persona_recommendation instances for one user.

    Requires ``discovery_llm`` to populate the slate.  Without it,
    returns an empty list.
    """
    profile = bq.get_full_profile(user_id) or {}
    if not profile:
        return []
    t_test = _t_test_anchor(profile, t_now)

    eligible = _filter_eligible_personas(profile, t_test)
    if not eligible:
        return []

    if discovery_llm is None:
        print(
            f"[hidden_persona_recommendation] user {user_id}: discovery_llm "
            "not wired; skipping."
        )
        return []

    canonical_preferences = profile.get("preferences") or []
    if isinstance(canonical_preferences, list) and canonical_preferences:
        if not isinstance(canonical_preferences[0], str):
            canonical_preferences = [str(p) for p in canonical_preferences]

    rng = random.Random(rng_seed)
    rng.shuffle(eligible)

    day_label = _dt.datetime.fromtimestamp(
        t_test, tz=_dt.timezone.utc
    ).strftime("%Y-%m-%d")

    out: list[dict] = []
    for hp in eligible:
        if len(out) >= INSTANCES_PER_USER_CAP:
            break

        is_pf = _hp_is_privacy_flagged(hp)
        other_personas = [o for o in eligible if o is not hp]

        parsed = _discover_slate(
            discovery_llm, hp, is_pf, other_personas,
            canonical_preferences, verbose=verbose,
        )
        if parsed is None:
            continue

        items = parsed["items"]
        target_index = parsed["target_index"]

        candidates = _items_to_candidates(items)

        # Shuffle the slate so the target position is unpredictable.
        order = list(range(SLATE_SIZE))
        rng.shuffle(order)
        shuffled = [candidates[j] for j in order]
        held_out_idx = order.index(target_index)

        seq = len(out) + 1
        inst = {
            "instance_id": f"hp_rec_{user_id}_{seq:03d}",
            "task_type": _TASK_ID,
            "task_id": _TASK_ID,
            "entry_point": "chatbot_routed",
            "t_test": t_test,
            "day_label": day_label,
            "anchor_hour_utc": 12,
            "query_text": "",
            "candidates": shuffled,
            "held_out_idx": held_out_idx,
            "hard_negative_idxs": [],
            "groundtruth_preference": {
                "hidden_persona": {
                    "label": hp.get("label", ""),
                    "type": hp.get("type", ""),
                    "is_privacy_flagged": is_pf,
                    "description": hp.get("description", ""),
                    "evidence_hashtags_sample": (hp.get("evidence_hashtags") or [])[:6],
                },
                "resonance_signal": parsed.get("resonance_signal", ""),
            },
        }
        out.append(inst)

    if verbose:
        print(
            f"[hidden_persona_recommendation] user {user_id}: "
            f"emitted {len(out)} instance(s) from {len(eligible)} "
            f"eligible persona(s)."
        )
    return out


# ---------------------------------------------------------------------------
# Runner — deterministic ranking metrics, no LLM judge.
# ---------------------------------------------------------------------------

def run_hidden_persona_recommendation(
    instances,
    user_id,
    bq: BackendQuery,
    llm_client,
    judge_client,
    mode: str,
    snapshot_cache,
    model_name: str | None,
    claude_model: str,
    context_budget: int | None,
    enable_llm_judge: bool,
    dry_run: bool,
    limit: int | None = None,
) -> list[dict]:
    """Runner — mirrors ``run_personalized_recommendation``."""
    from evaluation.tasks.personalized_recommendation import (
        personalized_recommendation_prompt,
        compute_personalized_recommendation_metrics,
    )
    from evaluation.inference_utils import dispatch_agent_run

    if limit is not None:
        instances = instances[:limit]

    results: list[dict] = []
    for inst in instances:
        t = inst["t_test"]
        history_block = None
        history_tokens = 0
        if mode == "llm_longctx" and snapshot_cache is not None:
            history_block, stats = snapshot_cache.get_or_build(
                bq, user_id, t, model_name, context_budget,
            )
            history_tokens = stats.get("total_tokens", 0)

        prompt = personalized_recommendation_prompt(inst, history_block)

        if dry_run:
            results.append({
                "task": _TASK_ID,
                "user_id": user_id,
                "instance_id": inst["instance_id"],
                "day_label": inst.get("day_label"),
                "mode": mode,
                "history_tokens": history_tokens,
                "metrics": None,
            })
            continue

        try:
            raw_response, tool_call_count, subagent_stats = dispatch_agent_run(
                mode, prompt, bq=bq, user_id=user_id, t=t,
                claude_model=claude_model, llm_client=llm_client,
            )
        except Exception as exc:
            results.append({
                "task": _TASK_ID,
                "user_id": user_id,
                "instance_id": inst["instance_id"],
                "mode": mode,
                "metrics": {},
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        parsed = extract_json_from_response(raw_response) or {}
        m = compute_personalized_recommendation_metrics(parsed, inst)

        results.append({
            "task": _TASK_ID,
            "user_id": user_id,
            "instance_id": inst["instance_id"],
            "day_label": inst.get("day_label"),
            "mode": mode,
            "metrics": m,
            "agent_response": raw_response,
            "subagent_stats": subagent_stats,
            "history_tokens": history_tokens,
            "tool_call_count": tool_call_count,
        })
    return results
