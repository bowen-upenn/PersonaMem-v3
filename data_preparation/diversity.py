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


# ---- Big Five (break the personality collapse → MBTI follows) -----------
# Audit: 7 distinct big_five signatures / 20, modal high-O/high-C/med-E/med-A/
# low-N in 9/20, ALL high-openness; MBTI 70% I-S-J, 100% introvert. The LLM
# rates the "interesting open conscientious introvert" by default. Draw each
# trait independently from a spread distribution so the cohort covers the
# personality space (and MBTI, inferred downstream from big_five, spreads too).
_TRAIT_LEVELS = {"low": 0.30, "medium": 0.40, "high": 0.30}


def assign_big_five(user_id) -> dict:
    return {
        trait: weighted_draw(user_id, _TRAIT_LEVELS, salt=f"bf_{trait}")
        for trait in ("openness", "conscientiousness", "extraversion",
                      "agreeableness", "neuroticism")
    }


# ---- Career sector (break the civic/infrastructure skew) ----------------
# Audit: careers skewed municipal/construction/coordinator/analyst (and 100%
# Bachelor's). Pin a per-user sector so the cohort spans the economy. The LLM
# still invents the specific role within the sector.
CAREER_SECTORS = [
    "healthcare / caregiving", "education / academia", "skilled trades / manual",
    "arts / entertainment / media", "food service / hospitality",
    "retail / sales", "finance / accounting", "law / public policy",
    "technology / engineering", "science / research", "agriculture / environment",
    "transportation / logistics", "construction / infrastructure",
    "nonprofit / social services", "small business / self-employed / gig",
    "government / civic", "manufacturing / industrial", "real estate / property",
    "fitness / wellness", "beauty / personal care",
]


def assign_career_sector(user_id) -> str:
    return user_rng(user_id, salt="career_sector").choice(CAREER_SECTORS)


# ---- Names (break the small-pool reuse: Marcus×9, Whitaker×6) -----------
# Audit: persona + friend names reuse a tiny pool. We ban the observed
# overused names and hand the generator a per-user "freshness nudge" so it
# samples widely instead of returning its modal favorites. A driver may ALSO
# pass an explicit `used_names` blocklist (accumulated across the cohort) for
# hard de-duplication of the persona's OWN name.
OVERUSED_FIRST_NAMES = [
    "Marcus", "Kevin", "Daniel", "Rachel", "Jason", "Tasha", "Keisha", "Emily",
    "Priya", "Andre", "Darius", "Brian", "Andrew", "Trevor", "Malik", "Maya",
    "Jamal", "Grace", "Hannah", "Caleb", "Derek", "Mei", "Aaliyah", "Jordan",
]
OVERUSED_SURNAMES = [
    "Whitaker", "Coleman", "Bennett", "Patel", "Miller", "Price", "Carter",
    "Chen", "Brooks", "Ellington", "Vale", "Watkins", "Monroe", "Freeman",
    "Simmons", "Johnson", "Wong", "Ramirez", "Martinez", "Kim", "Nguyen",
    "Liang", "Nair",
]
# Letter buckets the generator is nudged toward (spreads the alphabet so the
# cohort doesn't pile onto M/K/D/J first names and common surnames).
_ALPHA = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def name_freshness_nudge(user_id) -> dict:
    """Per-user nudge: preferred initials + a reminder to avoid overused names.
    Used by both the persona-profile name and the friend-graph names."""
    rng = user_rng(user_id, salt="name_nudge")
    return {
        "first_initial": rng.choice(_ALPHA),
        "surname_initial": rng.choice(_ALPHA),
        "banned_first_names": OVERUSED_FIRST_NAMES,
        "banned_surnames": OVERUSED_SURNAMES,
    }
