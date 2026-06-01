"""Deterministic per-user diversity assignments (seeded by user_id).

Single-user persona generation collapses onto modal defaults — the 2026-06
20-persona audit found 100% "Bachelor's degree", 😂 in 20/20 voice palettes,
"just/kinda/honestly" idiolect everywhere, and 59% of sensitive-life-events
being job_loss/parent_conflict. The LLM, asked per user with no view of the
cohort, picks the safe center every time.

These helpers give each user a STABLE pseudo-random assignment derived from a
hash of the user_id, so the cohort spreads across realistic axes without any
cross-user coordination and fully reproducibly (same user_id → same draw on a
re-run). They are injected into the generation prompts as constraints; the LLM
still authors the content, but the axis is pinned.

Pure-stdlib, no global random state (so it never perturbs other RNG users).
"""
from __future__ import annotations

import hashlib
import random
from typing import Iterable


def user_rng(user_id, salt: str = "") -> random.Random:
    """A reproducible RNG seeded by (user_id, salt). Distinct salts give
    independent draws for different axes of the same user."""
    h = hashlib.sha256(f"{user_id}|{salt}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def weighted_draw(user_id, dist: dict, salt: str = "") -> str:
    rng = user_rng(user_id, salt)
    keys = list(dist.keys())
    weights = list(dist.values())
    return rng.choices(keys, weights=weights, k=1)[0]


# ---- Education ----------------------------------------------------------
def assign_education_level(user_id, distribution: dict) -> str:
    """Seeded draw from the population distribution. Replaces the LLM's modal
    collapse to "Bachelor's degree" while still matching the realistic mix."""
    return weighted_draw(user_id, distribution, salt="education")


# ---- Sensitive-life-event topics ----------------------------------------
def assign_sle_topic_pool(user_id, all_topics: Iterable[str], n: int) -> list:
    """Per-user shuffle of the full topic menu; take the first n as the
    PREFERRED pool. Across the cohort this spreads draws across ALL topics
    (the audit found 8/15 topics never used, 59% job_loss/parent_conflict)."""
    pool = list(all_topics)
    user_rng(user_id, salt="sle_topic").shuffle(pool)
    return pool[: max(1, n)]


# ---- Voice axes (break voice collapse) ----------------------------------
# Each axis is drawn independently so the cohort covers the corners
# (emoji-zero personas, ALL-LOWERCASE personas, formal personas, etc.).
VOICE_CAPITALIZATION = [
    "all_lowercase", "all_lowercase", "standard_sentence_case",
    "standard_sentence_case", "Title-ish frequent caps", "ALL CAPS for emphasis only",
]
VOICE_EMOJI_INTENSITY = [
    "none", "none", "minimal_1to3_palette", "minimal_1to3_palette",
    "moderate", "heavy",
]
VOICE_FORMALITY = [
    "very_casual_slangy", "casual", "casual", "neutral", "neutral", "buttoned_up_formal",
]
VOICE_HUMOR = [
    "earnest_no_jokes", "warm_gentle", "dry_deadpan", "absurdist_silly",
    "sarcastic_edgy", "enthusiastic_exclamatory",
]
VOICE_VERBOSITY = [
    "terse_fragmentary", "terse_fragmentary", "balanced", "balanced",
    "long_winded_run_ons", "long_winded_run_ons",
]
VOICE_PUNCTUATION = [
    "ellipsis_heavy", "exclamation_heavy", "no_terminal_punctuation",
    "em_dash_heavy", "standard", "double_punctuation!!",
]


def assign_voice_axes(user_id) -> dict:
    """A pinned style fingerprint for this user's writing voice. The voice
    prompt must realize THESE axes (and is told NOT to default to 😂 /
    'just/kinda/honestly' / 'dry, avoids-mean')."""
    def pick(options, salt):
        return user_rng(user_id, salt).choice(options)
    return {
        "capitalization": pick(VOICE_CAPITALIZATION, "v_cap"),
        "emoji_intensity": pick(VOICE_EMOJI_INTENSITY, "v_emoji"),
        "formality": pick(VOICE_FORMALITY, "v_form"),
        "humor_mode": pick(VOICE_HUMOR, "v_humor"),
        "verbosity": pick(VOICE_VERBOSITY, "v_verb"),
        "punctuation": pick(VOICE_PUNCTUATION, "v_punct"),
    }


# ---- AI Studio archetype spread -----------------------------------------
# When a user's distinctive signal is `intimate_interest`, the old router sent
# 100% -> romantic_partner (45% of the cohort). Hash-split it across romantic
# and other intimate-compatible archetypes so no archetype dominates at scale.
INTIMATE_ARCHETYPE_SPLIT = [
    "romantic_partner", "romantic_partner",          # ~33% still romantic
    "late_night_best_friend", "niche_expert_creator_ai",
    "anime_or_fandom_character", "hype_affirmation_friend",
]


def intimate_interest_archetype(user_id) -> str:
    return user_rng(user_id, salt="intimate_archetype").choice(INTIMATE_ARCHETYPE_SPLIT)
