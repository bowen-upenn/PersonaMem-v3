"""
PersonaAgent — one instance per user_id.

Holds all interaction data and inferred persona traits for a single user.
Provides the full pipeline: hashtag inference -> cross-referencing -> temporal graph.
Persists results as CSV files in a backend directory.
"""

from __future__ import annotations

import os
import re
import time
import json
import random
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict

try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback when tqdm is not installed
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

# Add repo root to path so query_llm can be imported from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_preparation import utils, prompts, chatbot_conversation


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InteractionRow:
    """One row from the input CSV."""
    interaction_type: str
    user_id: str
    object_id: str
    interaction_time: int
    object_text: str
    interaction_format: str = ""  # e.g., "Instagram: liked"


@dataclass
class AtomicPersona:
    """A single inferred persona trait (output of LLM call #1)."""
    persona_item: str
    category: str
    confidence_score_init: float
    source_interaction_type: str = ""
    source_interaction_format: str = ""
    source_object_id: str = ""
    source_timestamp: int = 0
    formatted_timestamp: str = ""
    source_hashtags: list[str] = field(default_factory=list)


@dataclass
class CrossReferencedPersona:
    """A persona after cross-referencing (output of LLM call #2)."""
    persona_item: str
    category: str
    confidence_score_init: float
    # confidence_cross_referenced = the number of distinct source interaction
    # rows (distinct source_object_id) that independently inferred THIS
    # canonical persona AND whose individual confidence_score_init >= the
    # MIN_PERSONA_INIT_CONFIDENCE threshold. Computed AFTER the init filter
    # on canonicals. An integer
    # stored as float for schema compatibility.
    confidence_cross_referenced: float
    relationship_type: str = "none"          # internal: "similar", "contradictory", "none"
    related_personas: list = field(default_factory=list)  # internal: list of {"persona_item": str, "type": str}
    formatted_timestamp: str = ""
    source_interaction_type: str = ""
    source_interaction_format: str = ""
    # Which app the router assigned this preference to.
    assigned_app: str = ""
    # Temporal update history — how this preference evolved over time.
    # Each entry: {"preference": str, "update_type": str, "timestamp": int, "formatted_timestamp": str}
    # update_type values: "new", "reinforced", "contradicted", "faded"
    update_history: list = field(default_factory=list)
    # Evidence-mix counters set during weighted corroboration. Used by the
    # per-canonical survival threshold (implicit-heavy canonicals need more
    # rows to survive).
    n_explicit_rows: int = 0
    n_implicit_rows: int = 0
    # Time horizon classification (Step 3.5). "long_term" = enduring trait
    # inferable from the observed window (default). "short_term" = bounded
    # intent (trip, event prep, one-time purchase, how-to). Short-term uses
    # a relaxed xref survival threshold.
    time_horizon: str = "long_term"
    # Structured stop-condition for short-term canonicals. Eval tasks auto-
    # expire recommendations past expected_stop_ts. Shape:
    # {"type": "event"|"date"|"mastery"|"relocation",
    #  "description": str,
    #  "expected_stop_ts": int | None}
    # Empty dict for long_term canonicals (no stop condition applies).
    stop_condition: dict = field(default_factory=dict)


@dataclass
class TemporalNode:
    """A persona at a point in time."""
    persona_item: str
    timestamp: int
    formatted_timestamp: str
    confidence_score_init: float
    confidence_cross_referenced: float


@dataclass
class TemporalContradiction:
    """A group of contradictory personas organized as a temporal timeline."""
    topic: str
    timeline: list[TemporalNode] = field(default_factory=list)
    interpretation: str = ""


@dataclass
class AppPersona:
    """How a single user presents themselves and engages on ONE specific app.

    Generated per-user for each of the four supported apps (Instagram, Facebook,
    Threads, Chatbot) AFTER the base UserProfile is created. Drives the
    non-random app routing of preferences: each surviving preference is
    assigned to the app whose AppPersona use_purposes it best matches.
    """
    app_name: str                                          # "Instagram" | "Facebook" | "Threads" | "Chatbot"
    use_purposes: list[str] = field(default_factory=list)  # e.g. ["close friends sharing", "aesthetic personal brand"]
    friend_zones: list[str] = field(default_factory=list)  # e.g. ["close friends", "family", "acquaintances"]
    audience_type: str = "mixed"                           # "private" | "public" | "mixed"
    style_description: str = ""                            # 2-3 sentences on tone/aesthetic
    posting_frequency: str = "weekly"                      # "daily" | "weekly" | "rarely" | "passive viewer only"
    topical_focus: list[str] = field(default_factory=list) # 3-5 broad domains this app is used for
    chatbot_contexts: list[str] = field(default_factory=list)  # Chatbot only; picked from CHATBOT_CONTEXTS


@dataclass
class HiddenPersona:
    """A deeper motivational layer inferred from cross-row hashtag clustering.

    Hidden personas are the 'why' behind surface-level preferences — personality
    traits, aspirations, emotional patterns, identity anchors, intimate interests,
    and private hobbies that explain observable engagement but are not captured by
    individual-row inference.
    """
    label: str                                                              # e.g., "Romantic vulnerability and yearning"
    type: str = ""                                                          # personality_trait | aspiration | emotional_pattern | identity_anchor | intimate_interest | intellectual_curiosity | private_hobby
    description: str = ""                                                   # 2-3 sentence interpretation
    evidence_hashtags: list[str] = field(default_factory=list)              # Top 8-10 hashtags backing this
    evidence_rows: int = 0                                                  # Distinct source rows
    # Sorted list of distinct source_object_ids whose hashtags placed them
    # inside this cluster. Used in Step 16 to label preferences with the
    # hidden persona that motivated them (backward lookup: oid -> cluster).
    evidence_oids: list[str] = field(default_factory=list)
    evidence_row_fraction: float = 0.0                                      # Fraction of user's total rows
    interaction_breakdown: dict = field(default_factory=dict)               # {implicit_positive: N, ...}
    privacy_ratio: float = 0.0                                              # implicit_positive / (implicit_positive + explicit_positive)
    temporal_spread_days: int = 0                                           # Distinct calendar days
    app_distribution: dict = field(default_factory=dict)                    # {Instagram: N, Facebook: N, ...}
    surface_connections: list[str] = field(default_factory=list)            # Which surface preferences this explains
    inferred_motivation: str = ""                                           # 1-2 sentence "why" behind this pattern
    already_captured: bool = False                                          # True if overlaps heavily with surface preferences


# Validation thresholds for hidden persona inference
MIN_HIDDEN_PERSONA_ROWS = 40       # Minimum distinct source rows for a cluster to survive
MIN_HIDDEN_PERSONA_DAYS = 3        # Minimum temporal spread in distinct calendar days
HIDDEN_PERSONA_HASHTAG_MIN_FREQ = 3  # Minimum total occurrences for a hashtag to be considered
HIDDEN_PERSONA_TOP_HASHTAGS = 200  # Number of top hashtags passed to LLM


@dataclass
class UserProfile:
    """Synthetic user profile generated from final personas (output of LLM call #4)."""
    name: str = ""
    gender: str = ""
    race_ethnicity: str = ""
    career: str = ""
    education: str = ""
    big_five: dict = field(default_factory=dict)  # {"openness": "...", "conscientiousness": "...", ...}
    bio: str = ""
    # Per-app sub-personas — filled in by generate_app_personas() after the
    # base profile is written. Keyed by app_name.
    app_personas: dict = field(default_factory=dict)  # dict[str, AppPersona]
    # Hidden personas — deeper motivational layers inferred from cross-row
    # hashtag clustering. Filled in by infer_hidden_personas().
    hidden_personas: list = field(default_factory=list)  # list[HiddenPersona]
    hidden_persona_summary: str = ""                     # cohesive narrative paragraph
    # MBTI inferred from Big Five + hidden personas + top hashtags. Shape:
    # {"type": "INTJ",
    #  "dimensions": {"E_I": {"E": 0.22, "I": 0.78, "reason": "..."}, ...}}
    mbti: dict = field(default_factory=dict)


@dataclass
class AnnotatedPersona:
    """A cross-referenced persona annotated with stereotype mark (output of LLM call #5)."""
    persona_item: str
    category: str
    confidence_score_init: float
    confidence_cross_referenced: float
    stereotype_mark: str = "neutral"  # "neutral", "stereotypical", "anti-stereotypical"


# ---------------------------------------------------------------------------
# High-confidence predicate — used for test-split eligibility and
# distractor shortlisting. Single source of truth; thresholds are tentative
# and will be tuned empirically once we have real-scale stats.
# ---------------------------------------------------------------------------

# Floor on confidence_score_init. Personas below this are dropped after
# cross-ref regardless of cross-ref score or relationship type. This is
# the main knob for preference-list size. Tuneable.
MIN_PERSONA_INIT_CONFIDENCE = 0.75

# High-confidence predicate — used for test-split eligibility and distractor
# shortlisting. init threshold matches the filter floor so "high-confidence"
# at minimum means the persona survived the init filter AND cleared the
# per-canonical xref threshold.
HIGH_CONFIDENCE_INIT_THRESHOLD = 0.75

# Per-canonical xref threshold: the survival bar is interpolated by each
# canonical's evidence mix. Canonicals backed mostly by explicit rows need a
# smaller xref; canonicals backed mostly by implicit rows need a larger one.
#
#   threshold_c = (1 - implicit_fraction) * XREF_THRESHOLD_EXPLICIT
#                 + implicit_fraction     * XREF_THRESHOLD_IMPLICIT
#
# Paired with the 7-day recency window on corroboration counting
# (RECENCY_WINDOW_SECONDS below): only evidence rows within the user's
# trailing 7 days count toward confidence_cross_referenced and the
# evidence-mix counters, so the effective survival bar is "≥ 20 weighted
# corroborating distinct rows within the last 7 days".
XREF_THRESHOLD_EXPLICIT = 20.0
XREF_THRESHOLD_IMPLICIT = 50.0

# Negatives are structurally rarer than positives (source CSVs typically have
# ~5-10x fewer implicit_negative rows than implicit_positive, and often 0
# explicit_negative). The per-canonical xref floor for negatives is therefore
# decoupled from the positive scale — negatives that pass the hot-hashtag gate
# + init ≥ 0.75 + distinct-rows gate already carry enough signal; stacking the
# 50.0 implicit floor on top double-counts the bars.
XREF_THRESHOLD_NEGATIVE = 5.0

# Negatives get a lower init-confidence survival bar than positives.
# `hashtag_to_persona_prompt` tells the LLM to score negatives in the
# 0.55-0.75 range ("direct dislike of the core topic" tops out at 0.75),
# while positives use 0.80-1.0. Applying the 0.75 positive bar to negatives
# was a mathematical mismatch — virtually every negative canonical was
# dropped at the init filter. 0.55 matches the LOW end of "direct dislike".
MIN_NEGATIVE_INIT_CONFIDENCE = 0.55

# Kept for backward-compatibility references. Internal code prefers
# canonical_xref_threshold() so it gets the evidence-mix-dependent value.
HIGH_CONFIDENCE_CROSS_REF_THRESHOLD = XREF_THRESHOLD_EXPLICIT

# Session grouping: source rows with timestamp gaps <= this threshold are
# considered part of the same scrolling session on one app.
SESSION_GAP_SECONDS = 5  # 5 seconds — rows within 5s are same browsing burst

# Implicit-negative handling. A single skipped post is too weak a signal, but
# the same topic skipped many times is a real dislike. We process implicit
# negatives conditionally: a cheap hashtag-signature pre-filter drops obvious
# singletons before LLM inference, and an authoritative gate in the negative
# cross-ref step requires N distinct source rows for any canonical supported
# only by implicit evidence. Canonicals with any explicit-negative evidence
# are unaffected.
MIN_IMPLICIT_NEGATIVE_REPETITION = 5   # distinct source rows for implicit-only negative to survive
IMPLICIT_NEGATIVE_PREFILTER_K = 3      # rows per hashtag signature required to bother with LLM call

# Recency window on cross-reference counting. Only evidence rows whose
# source_timestamp falls within this trailing window (anchored on the user's
# latest interaction, NOT wall-clock time) contribute to confidence_cross_referenced
# and the n_explicit_rows / n_implicit_rows mix that feeds canonical_xref_threshold().
# This replaces raw lifetime-of-account corroboration counting with a recency
# gate, so stale-but-heavily-repeated preferences don't survive on old evidence.
RECENCY_WINDOW_SECONDS = 7 * 86400  # 7 days

# --------------------------------------------------------------------------
# Time horizon classification (Step 3.5).
#
# The observation window is short (~8 days), so horizon classification leans
# on row count + category + span-fraction rather than raw span in days. A
# canonical is eligible for `short_term` iff ALL of:
#   - span_days / obs_window_days <= SHORT_TERM_MAX_SPAN_FRAC
#   - n_explicit_rows + n_implicit_rows < SHORT_TERM_MAX_ROWS
#   - category ∈ SHORT_TERM_ALLOWED_CATEGORIES
# Everything else is `long_term`. The allow-list is the anti-loophole — a
# canonical cannot claim short-term just by having a tight span + few rows
# unless it's in a bounded category. LLM confirmation (run after rule
# pre-label) can demote short→long but NEVER promote long→short.
SHORT_TERM_MAX_SPAN_FRAC: float = 0.35
SHORT_TERM_MAX_ROWS: int = 8
# Survival xref threshold for short-term canonicals. Much lower than the
# long-term explicit/implicit bars because short-term intents leave little
# corroborating evidence (trip hotel search, one-time how-to) yet are still
# legitimate and worth surfacing for personalization during their active
# window.
XREF_THRESHOLD_SHORT_TERM: float = 3.0
# Allowed categories for short_term eligibility. Keep this small — each
# entry names a class of BOUNDED intent (one trip, one event, one purchase,
# one skill acquisition). Category names are matched case-insensitively
# against substrings in the canonical's `category` field (so "travel_planning"
# and "solo travel" both hit "travel").
SHORT_TERM_ALLOWED_CATEGORIES: set[str] = {
    "travel",
    "event_prep",
    "event prep",
    "purchase_intent",
    "purchase intent",
    "how_to",
    "how to",
    "medical_consultation",
    "medical consultation",
    "trip",
}

# --------------------------------------------------------------------------
# Cross-polarity contradiction gate (Step 3b).
#
# Independent positive + negative cross-ref pipelines can both produce
# surviving canonicals about the same topic, creating immediate
# contradictions ("Interested in boxing" alongside "Not interested in
# boxing" ~1h apart, no causal story). Step 3b enforces temporal
# precedent: the later-emerging stance survives only when it has enough
# prior same-polarity evidence to justify the flip.
MIN_STANCE_FLIP_PRIOR: int = 3
MIN_STANCE_FLIP_PRIOR_SHORT: int = 1   # relaxed for short_term canonicals
STANCE_FLIP_WINDOW_HOURS: int = 48     # (informational; all pairs are checked regardless)
HASHTAG_OVERLAP_MIN: int = 2           # pos/neg pair must share ≥ this many hashtags

# NOTE: confidence_cross_referenced is intentionally UNCAPPED on the upper
# side. A preference corroborated by 200 distinct rows should be strictly
# more confident than one corroborated by 10 — they can't both be 1.0. Only
# the lower bound (0.0 floor) is enforced. This makes cross_ref a
# distinguishing signal at scale.


def canonical_xref_threshold(
    n_explicit_rows: int,
    n_implicit_rows: int,
    time_horizon: str = "long_term",
) -> float:
    """Return the survival xref threshold for a canonical.

    Long-term (default): interpolated by evidence mix. A canonical backed
    mostly by explicit rows survives with a smaller xref; mostly implicit
    needs a larger xref. When a canonical has no distinct row evidence, the
    fallback is the explicit threshold (no penalty).

    Short-term: always `XREF_THRESHOLD_SHORT_TERM` regardless of mix. Short-
    term intents leave sparse evidence but are still legitimate; this
    relaxed floor lets them survive without opening a loophole for weak
    long-term signals (eligibility for short_term is gated by category +
    span + row count in `_classify_time_horizon_rule`).
    """
    if time_horizon == "short_term":
        return XREF_THRESHOLD_SHORT_TERM
    total = n_explicit_rows + n_implicit_rows
    if total <= 0:
        return XREF_THRESHOLD_EXPLICIT
    implicit_frac = n_implicit_rows / total
    return (
        (1.0 - implicit_frac) * XREF_THRESHOLD_EXPLICIT
        + implicit_frac * XREF_THRESHOLD_IMPLICIT
    )


def _classify_time_horizon_rule(
    category: str,
    span_days: float,
    obs_window_days: float,
    n_total_rows: int,
) -> str:
    """Rule-based horizon classification. Deterministic, LLM-free.

    Returns "short_term" only when ALL eligibility conditions hold. All
    other inputs return "long_term". Called during cross-referencing BEFORE
    the survival filter so short-term canonicals can use a relaxed xref
    threshold.
    """
    if not category or obs_window_days <= 0:
        return "long_term"
    cat_lower = category.lower()
    # substring match against the allow-list (handles "travel_planning",
    # "solo travel", "medical_consultation", etc.)
    cat_hit = any(
        allowed_lower in cat_lower
        for allowed_lower in (s.lower() for s in SHORT_TERM_ALLOWED_CATEGORIES)
    )
    if not cat_hit:
        return "long_term"
    span_frac = span_days / obs_window_days if obs_window_days > 0 else 0.0
    if span_frac > SHORT_TERM_MAX_SPAN_FRAC:
        return "long_term"
    if n_total_rows >= SHORT_TERM_MAX_ROWS:
        return "long_term"
    return "short_term"


def _compute_recency_cutoff(interactions) -> int:
    """Return the Unix-timestamp cutoff below which rows don't count toward
    cross-reference corroboration.

    Anchored on the user's latest interaction (not wall-clock time) so the
    window is meaningful for synthetic data from any historical period. When
    the interaction list is empty, returns 0 so every row passes.
    """
    if not interactions:
        return 0
    latest = max((r.interaction_time for r in interactions), default=0)
    return latest - RECENCY_WINDOW_SECONDS


def is_high_confidence(
    init_score: float,
    cross_ref_score: float,
    n_explicit_rows: int = 0,
    n_implicit_rows: int = 0,
    time_horizon: str = "long_term",
) -> bool:
    """Return True if a persona's scores qualify as 'high confidence'.

    BOTH conditions must hold:
      - confidence_score_init  >= HIGH_CONFIDENCE_INIT_THRESHOLD
      - confidence_cross_referenced > canonical_xref_threshold(..., time_horizon)
        (evidence-mix-dependent survival bar — stricter for implicit-only
         canonicals than for explicit-supported ones; relaxed to
         XREF_THRESHOLD_SHORT_TERM when time_horizon="short_term").
    """
    return (
        init_score >= HIGH_CONFIDENCE_INIT_THRESHOLD
        and cross_ref_score
        > canonical_xref_threshold(n_explicit_rows, n_implicit_rows, time_horizon)
    )


def _normalize_persona_text(text: str) -> str:
    """Normalize a persona_item string for semantic-equivalence matching.

    Used to deduplicate atomic personas that are lexically identical (after
    case/whitespace normalization) across multiple interaction rows. Those
    duplicates represent the SAME preference corroborated by multiple rows,
    not distinct preferences that happen to be similar to each other.
    """
    if not text:
        return ""
    return " ".join(text.strip().lower().split())


# ---------------------------------------------------------------------------
# Demographic distributions for random sampling
# ---------------------------------------------------------------------------

# Gender × sexual orientation distribution
GENDER_ORIENTATION_DISTRIBUTION = {
    "cisgender female, heterosexual": 0.30,
    "cisgender female, bisexual": 0.04,
    "cisgender female, lesbian": 0.04,
    "cisgender female, queer": 0.01,
    "cisgender male, heterosexual": 0.32,
    "cisgender male, bisexual": 0.02,
    "cisgender male, gay": 0.05,
    "cisgender male, queer": 0.01,
    "transgender female, heterosexual": 0.02,
    "transgender female, lesbian": 0.01,
    "transgender female, bisexual": 0.01,
    "transgender male, heterosexual": 0.02,
    "transgender male, gay": 0.01,
    "transgender male, bisexual": 0.01,
    "non-binary, queer": 0.02,
    "non-binary, bisexual": 0.01,
    "non-binary, pansexual": 0.01,
    "non-binary, asexual": 0.005,
    "genderfluid, queer": 0.005,
    "genderfluid, pansexual": 0.005,
    "agender, asexual": 0.005,
}

# Detailed race/ethnicity distribution (intentionally diversified)
RACE_ETHNICITY_DISTRIBUTION = {
    "White American": 0.15,
    "White European immigrant": 0.02,
    "Russian or Eastern European": 0.02,
    "Jewish American": 0.02,
    "Black or African American": 0.08,
    "African immigrant": 0.02,
    "Afro-Caribbean": 0.02,
    "Mexican American": 0.08,
    "Puerto Rican": 0.02,
    "Cuban American": 0.01,
    "Central American": 0.02,
    "South American": 0.02,
    "Chinese": 0.10,
    "Indian": 0.08,
    "Filipino": 0.04,
    "Vietnamese": 0.04,
    "Korean": 0.04,
    "Japanese": 0.03,
    "Pakistani or Bangladeshi": 0.02,
    "Southeast Asian": 0.02,
    "Central Asian": 0.01,
    "Middle Eastern or North African": 0.03,
    "Native Hawaiian or Pacific Islander": 0.01,
    "American Indian or Alaska Native": 0.01,
    "Multiracial (Black and White)": 0.02,
    "Multiracial (Asian and White)": 0.02,
    "Multiracial (Hispanic and White)": 0.02,
    "Multiracial (other)": 0.03,
}


PLATFORMS = ["Instagram", "Facebook", "Threads", "Chatbot"]

# Per-app action catalog — this is the **single source of truth** for every
# realistic interaction UX affordance on each app. The pipeline does NOT
# invent new actions or labels at generation time: it picks ONE entry from
# the appropriate bucket for each routed preference. Subagents MUST copy
# the `action` identifier and `label` verbatim from this catalog; consistent
# wording across runs is the point.
#
# Each entry also carries a `weight` reflecting real-world relative frequency
# of the action within its polarity bucket. These are rough numbers based
# on publicly-reported engagement benchmarks (passive > active; like >
# comment > share; reactions cluster heavily around 👍 / ❤ on Facebook;
# @ai comments are ~20% of explicit buckets; etc.). At sample time each
# USER gets their own per-user perturbed copy of these weights (lognormal
# noise, see `_perturb_weights` on PersonaAgent), so different users have
# visibly different action distributions while still roughly matching the
# underlying shape.
#
# Two categories of actions carry a natural-language `user_message`:
#
# 1. `at_ai_*` actions on SOCIAL MEDIA apps (Instagram / Facebook / Threads).
#    These model the user @-mentioning an in-feed AI assistant in the
#    comment area of a post. Message starts with `@ai `. They live on the
#    social apps, NOT on the AI Chatbot app.
#
# 2. Conversation-turn actions on the `Chatbot` app (`asked_followup`,
#    `requested_more_detail`, `continued_topic`, `asked_to_change_topic`,
#    `edited_prompt_and_retried`, `regenerated`). The `user_message` is
#    what the user would naturally type — NO `@ai` prefix because the user
#    is already in an AI conversation.
PLATFORM_INTERACTION_FORMATS: dict[str, dict[str, list[dict]]] = {
    "Instagram": {
        "explicit_positive": [
            {"action": "liked", "label": "Liked", "weight": 50.0},
            {"action": "double_tapped", "label": "Double-tapped to like", "weight": 22.0},
            {"action": "saved_to_collection", "label": "Saved to a collection", "weight": 8.0},
            {"action": "reacted_to_story", "label": "Reacted to the story", "weight": 5.0},
            {"action": "commented", "label": "Commented", "weight": 4.0},
            {"action": "followed_creator", "label": "Followed the creator", "weight": 3.0},
            {"action": "dm_to_friend", "label": "Sent via DM to a friend", "weight": 3.0},
            {"action": "shared_to_close_friends_story", "label": "Shared to Close Friends story", "weight": 2.0},
            {"action": "reposted", "label": "Reposted", "weight": 1.0},
            {"action": "at_ai_recommend_more", "label": "@ai comment: asked the in-feed assistant for MORE like this", "weight": 12.2},
            {"action": "at_ai_focus_topic", "label": "@ai comment: asked the in-feed assistant to focus on this topic", "weight": 12.2},
            {"action": "clicked_ad", "label": "Tapped through on a sponsored post", "weight": 1.5},
        ],
        "implicit_positive": [
            {"action": "viewed_reel_75", "label": "Viewed more than 75% of the reel", "weight": 40.0},
            {"action": "lingered_on_image", "label": "Stayed on an image for more than 5 seconds", "weight": 25.0},
            {"action": "lingered_on_story", "label": "Stayed on a story for more than 5 seconds", "weight": 15.0},
            {"action": "tapped_profile", "label": "Tapped through to the creator's profile", "weight": 10.0},
            {"action": "rewatched_reel", "label": "Rewatched the reel", "weight": 8.0},
            {"action": "long_pressed_for_options", "label": "Long-pressed to open context menu", "weight": 2.0},
        ],
        "explicit_negative": [
            {"action": "not_interested", "label": "Marked Not Interested", "weight": 10.0},
            {"action": "hidden", "label": "Hid this post", "weight": 5.0},
            {"action": "muted_user", "label": "Muted the user", "weight": 3.0},
            {"action": "unfollowed", "label": "Unfollowed the creator", "weight": 2.0},
            {"action": "at_ai_stop_recommending", "label": "@ai comment: asked the in-feed assistant to STOP showing this", "weight": 1.7},
            {"action": "at_ai_not_interested", "label": "@ai comment: told the in-feed assistant they're not interested right now", "weight": 1.7},
            {"action": "at_ai_feels_off", "label": "@ai comment: told the in-feed assistant this feels off-base", "weight": 1.7},
            {"action": "hidden_ad", "label": "Tapped 'Hide this ad'", "weight": 2.5},
            {"action": "dismissed_ad", "label": "Dismissed the ad without engagement", "weight": 1.0},
            {"action": "reported", "label": "Reported", "weight": 0.5},
        ],
        "implicit_negative": [
            {"action": "skipped_reel", "label": "Skipped the reel with no interaction", "weight": 50.0},
            {"action": "skipped_image", "label": "Skipped the image with no interaction", "weight": 30.0},
            {"action": "skipped_story", "label": "Skipped the story with no interaction", "weight": 20.0},
        ],
    },
    "Facebook": {
        "explicit_positive": [
            {"action": "reacted_like", "label": "Liked", "weight": 60.0},
            {"action": "reacted_love", "label": "Loved (❤)", "weight": 20.0},
            {"action": "reacted_haha", "label": "Hahaha reaction", "weight": 10.0},
            {"action": "commented", "label": "Commented", "weight": 8.0},
            {"action": "reacted_wow", "label": "Wow reaction", "weight": 4.0},
            {"action": "reacted_sad", "label": "Sad reaction", "weight": 3.0},
            {"action": "saved_post", "label": "Saved the post", "weight": 3.0},
            {"action": "reacted_care", "label": "Care reaction", "weight": 2.0},
            {"action": "shared_to_timeline", "label": "Shared to own timeline", "weight": 2.0},
            {"action": "tagged_friend", "label": "Tagged a friend in the post", "weight": 2.0},
            {"action": "shared_to_group", "label": "Shared to a group", "weight": 1.0},
            {"action": "at_ai_recommend_more", "label": "@ai comment: asked Meta AI in the comments for MORE like this", "weight": 14.4},
            {"action": "at_ai_focus_topic", "label": "@ai comment: asked Meta AI in the comments to focus on this topic", "weight": 14.4},
            {"action": "rsvp_event", "label": "Marked Interested / Going on an event", "weight": 0.5},
            {"action": "clicked_ad", "label": "Tapped through on a sponsored post", "weight": 1.5},
        ],
        "implicit_positive": [
            {"action": "lingered_on_post", "label": "Stayed on a post for more than 5 seconds", "weight": 40.0},
            {"action": "viewed_video_75", "label": "Viewed more than 75% of the video", "weight": 30.0},
            {"action": "expanded_see_more", "label": "Tapped 'See more' to expand the post", "weight": 20.0},
            {"action": "viewed_comments", "label": "Opened the comments thread", "weight": 10.0},
        ],
        "explicit_negative": [
            {"action": "hidden", "label": "Hid the post", "weight": 10.0},
            {"action": "see_fewer_like_this", "label": "Asked to see fewer posts like this", "weight": 8.0},
            {"action": "reacted_angry", "label": "Angry reaction", "weight": 5.0},
            {"action": "snoozed_user", "label": "Snoozed the user for 30 days", "weight": 2.0},
            {"action": "unfollowed", "label": "Unfollowed the page / user", "weight": 2.0},
            {"action": "at_ai_stop_recommending", "label": "@ai comment: asked Meta AI in the comments to STOP showing this", "weight": 2.3},
            {"action": "at_ai_not_interested", "label": "@ai comment: told Meta AI in the comments they're not interested", "weight": 2.3},
            {"action": "at_ai_feels_off", "label": "@ai comment: told Meta AI in the comments this feels off-base", "weight": 2.3},
            {"action": "hidden_ad", "label": "Tapped 'Hide this ad'", "weight": 2.5},
            {"action": "dismissed_ad", "label": "Dismissed the ad without engagement", "weight": 1.0},
            {"action": "reported", "label": "Reported", "weight": 0.5},
        ],
        "implicit_negative": [
            {"action": "skipped_post", "label": "Skipped the post with no interaction", "weight": 60.0},
            {"action": "scrolled_past_video", "label": "Scrolled past the video without watching", "weight": 40.0},
        ],
    },
    "Threads": {
        "explicit_positive": [
            {"action": "liked", "label": "Liked", "weight": 60.0},
            {"action": "replied", "label": "Replied", "weight": 15.0},
            {"action": "reposted", "label": "Reposted", "weight": 10.0},
            {"action": "saved", "label": "Saved the thread", "weight": 5.0},
            {"action": "quote_reposted", "label": "Reposted with a quote", "weight": 4.0},
            {"action": "followed_author", "label": "Followed the author", "weight": 3.0},
            {"action": "shared_externally", "label": "Shared externally (copy link / DM)", "weight": 2.0},
            {"action": "at_ai_recommend_more", "label": "@ai reply: asked the in-feed assistant for MORE like this", "weight": 12.4},
            {"action": "at_ai_focus_topic", "label": "@ai reply: asked the in-feed assistant to focus on this topic", "weight": 12.4},
            {"action": "clicked_ad", "label": "Tapped through on a sponsored post", "weight": 1.5},
        ],
        "implicit_positive": [
            {"action": "lingered_on_thread", "label": "Stayed on the thread for more than 5 seconds", "weight": 40.0},
            {"action": "viewed_video_75", "label": "Viewed more than 75% of the video", "weight": 30.0},
            {"action": "expanded_replies", "label": "Expanded the reply thread", "weight": 20.0},
            {"action": "tapped_author", "label": "Tapped through to the author's profile", "weight": 10.0},
        ],
        "explicit_negative": [
            {"action": "not_interested", "label": "Marked Not Interested", "weight": 10.0},
            {"action": "muted_author", "label": "Muted the author", "weight": 5.0},
            {"action": "hid_replies", "label": "Hid the replies", "weight": 3.0},
            {"action": "at_ai_stop_recommending", "label": "@ai reply: asked the in-feed assistant to STOP showing this", "weight": 1.5},
            {"action": "at_ai_not_interested", "label": "@ai reply: told the in-feed assistant they're not interested", "weight": 1.5},
            {"action": "at_ai_feels_off", "label": "@ai reply: told the in-feed assistant this feels off-base", "weight": 1.5},
            {"action": "hidden_ad", "label": "Tapped 'Hide this ad'", "weight": 2.5},
            {"action": "dismissed_ad", "label": "Dismissed the ad without engagement", "weight": 1.0},
            {"action": "reported", "label": "Reported", "weight": 0.5},
        ],
        "implicit_negative": [
            {"action": "skipped_thread", "label": "Skipped the thread with no interaction", "weight": 100.0},
        ],
    },
    "Chatbot": {
        "explicit_positive": [
            {"action": "thumbs_up", "label": "Thumbs-upped the response", "weight": 30.0},
            {"action": "copied_response", "label": "Copied the response text", "weight": 25.0},
            {"action": "asked_followup", "label": "Asked a follow-up question showing interest", "weight": 20.0},
            {"action": "requested_more_detail", "label": "Requested more detail on the same topic", "weight": 15.0},
            {"action": "saved_to_library", "label": "Saved the response to library", "weight": 8.0},
            {"action": "shared_conversation", "label": "Shared the conversation externally", "weight": 2.0},
        ],
        "implicit_positive": [
            {"action": "continued_topic", "label": "Continued the conversation on the same topic", "weight": 40.0},
            {"action": "read_carefully", "label": "Spent significant time reading the response", "weight": 30.0},
            {"action": "referenced_response", "label": "Copied or referenced part of the response", "weight": 15.0},
            {"action": "positive_language_next_turn", "label": "Positive language in the next turn", "weight": 15.0},
        ],
        "explicit_negative": [
            {"action": "regenerated", "label": "Asked to regenerate the response", "weight": 30.0},
            {"action": "thumbs_down", "label": "Thumbs-downed the response", "weight": 20.0},
            {"action": "asked_to_change_topic", "label": "Explicitly asked to change topic or stop", "weight": 20.0},
            {"action": "edited_prompt_and_retried", "label": "Edited the prompt and retried", "weight": 20.0},
            {"action": "reported_response", "label": "Reported / flagged the response", "weight": 5.0},
            {"action": "asked_to_forget", "label": "Asked the assistant to forget a specific preference", "weight": 5.0},
            {"action": "corrected_assumption", "label": "Corrected the assistant's wrong assumption about a preference", "weight": 5.0},
        ],
        "implicit_negative": [
            {"action": "no_followup", "label": "No active follow-up or response", "weight": 30.0},
            {"action": "abandoned_conversation", "label": "Abandoned the conversation after the response", "weight": 25.0},
            {"action": "changed_topic_immediately", "label": "Immediately changed the topic", "weight": 25.0},
            {"action": "dismissive_reply", "label": "Gave a minimal or dismissive reply", "weight": 20.0},
        ],
    },
}


# ---------------------------------------------------------------------------
# Content-type distribution (Step 13b — synthetic content generation)
# ---------------------------------------------------------------------------
# Per-user content mix is derived by a three-layer process:
#   (1) Platform prior — population baseline (below).
#   (2) Bayesian smoothing — combines the user's observed
#       content-type-implying actions with the prior via PRIOR_PSEUDOCOUNT.
#   (3) Per-user lognormal perturbation — ensures two users with identical
#       observed action histories still land on distinct mixes.
# See _compute_user_content_mix() for the formula.

PLATFORM_CONTENT_PRIOR: dict[str, dict[str, float]] = {
    # Instagram is visual-first: reels + images dominate, very little text-only.
    "Instagram": {"image": 0.45, "short_video": 0.50, "text": 0.05},
    # Facebook balances text status updates, photos, and video.
    "Facebook":  {"image": 0.35, "short_video": 0.30, "text": 0.35},
    # Threads is text-leaning but with substantial image/video mix (not
    # text-dominant as in early days).
    "Threads":   {"image": 0.30, "short_video": 0.20, "text": 0.50},
}

# Smoothing strength for mix derivation. 30 means ~30 observed events
# give the prior ~50% weight. A user with 200+ events is dominated by
# their own action signal.
PRIOR_PSEUDOCOUNT: int = 30

# Lognormal σ for per-user mix perturbation. Kept smaller than the
# action-weight σ (0.6) because the mix is already derived from
# noisy action data.
CONTENT_MIX_NOISE_SIGMA: float = 0.3

# Per-action content-type hints. A hint of "image" / "video" / "text"
# is a hard constraint (deterministic content_type). "story_amb" resolves
# 50/50 between image and video (stories can be either, but not text).
# "ambiguous" (missing or explicit) falls through to per-user mix sampling.
ACTION_CONTENT_HINTS: dict[str, dict[str, str]] = {
    "Instagram": {
        "viewed_reel_75": "video",
        "rewatched_reel": "video",
        "skipped_reel": "video",
        "lingered_on_image": "image",
        "skipped_image": "image",
        "lingered_on_story": "story_amb",
        "skipped_story": "story_amb",
        "reacted_to_story": "story_amb",
        "shared_to_close_friends_story": "story_amb",
        # Everything else (liked, saved_to_collection, commented, tapped_profile,
        # followed_creator, dm_to_friend, reposted, @ai actions, etc.) → ambiguous.
        # Note: Instagram feed does not have pure text posts, so "text" is
        # strongly disfavored by the prior (0.05) and ambiguous events will
        # overwhelmingly resolve to image/video.
    },
    "Facebook": {
        "viewed_video_75": "video",
        "scrolled_past_video": "video",
        "expanded_see_more": "text",   # "See more" implies a long-caption post
    },
    "Threads": {
        "viewed_video_75": "video",
        # Threads has rich image/video content now — most actions (liked,
        # reposted, replied, saved, quote_reposted, lingered_on_thread,
        # expanded_replies, tapped_author, etc.) stay ambiguous and sample
        # from the per-user mix.
    },
}

# Ad actions are added to every social app's ACTION_CONTENT_HINTS as
# "ambiguous" (falls through to per-user mix sampling). The final content_type
# on an ad is decided by `generate_synthetic_content` (step 13b) BEFORE
# `inject_ad_events` runs — 13c reuses whatever content_type was already
# assigned to preserve per-user mix consistency. No hint needed here.


def _perturb_weights(base_weights: list[float], rng: random.Random, noise_strength: float = 0.6) -> list[float]:
    """Perturb a list of action weights with per-user lognormal noise.

    Each weight is multiplied by exp(N(0, noise_strength)), preserving the
    rough shape of the base distribution while introducing visible per-user
    variation. Larger `noise_strength` → more deviation from the baseline.
    Typical range 0.3 (small noise) to 1.0 (large noise). Default 0.6 gives
    a distinct-but-still-recognizable personalized distribution.
    """
    import math
    perturbed = []
    for w in base_weights:
        factor = math.exp(rng.gauss(0.0, noise_strength))
        perturbed.append(max(0.0, w * factor))
    total = sum(perturbed)
    if total <= 0:
        return list(base_weights)
    # Renormalize to preserve the original sum — keeps magnitudes comparable
    scale = sum(base_weights) / total
    return [w * scale for w in perturbed]


def build_user_action_distribution(user_seed: int, noise_strength: float = 0.6) -> dict:
    """Build a per-user perturbed copy of PLATFORM_INTERACTION_FORMATS.

    Returns a dict with the same structure as PLATFORM_INTERACTION_FORMATS
    but each entry's `weight` is a per-user lognormally-perturbed version
    of the base weight. Use this to sample interaction actions for the
    given user in a way that's consistent across their preferences and
    distinct from other users.
    """
    rng = random.Random(user_seed)
    out = {}
    for app, polarities in PLATFORM_INTERACTION_FORMATS.items():
        out[app] = {}
        for polarity, bucket in polarities.items():
            base = [e["weight"] for e in bucket]
            noisy = _perturb_weights(base, rng, noise_strength)
            out[app][polarity] = [
                {**e, "weight": round(w, 3)} for e, w in zip(bucket, noisy)
            ]
    return out


# Action identifiers that REQUIRE a natural-language `user_message` to be
# generated. Two groups:
#
# - `AT_AI_ACTIONS`: social-media @ai comment actions (Instagram / Facebook
#   / Threads only). Message starts with `@ai ` and is first-person,
#   ~15-35 words, grounded in the specific preference topic. NEVER used
#   on the Chatbot app.
#
# - `CHATBOT_TURN_ACTIONS`: natural-chat turn actions on the AI Chatbot.
#   The `user_message` is what the user would type next. NO `@ai` prefix —
#   the user is already conversing with the assistant, there's nothing to
#   @-mention.
AT_AI_ACTIONS: set[str] = {
    "at_ai_recommend_more",
    "at_ai_focus_topic",
    "at_ai_stop_recommending",
    "at_ai_not_interested",
    "at_ai_feels_off",
}

CHATBOT_TURN_ACTIONS: set[str] = {
    "asked_followup",
    "requested_more_detail",
    "continued_topic",
    "asked_to_change_topic",
    "edited_prompt_and_retried",
    "regenerated",
    "asked_to_forget",
    "asked_not_to_personalize",
    "corrected_assumption",
}

# Ad interaction actions — social apps only. Events marked with one of these
# carry `is_ad: true` at the event root and ad-shaped content (with
# `ad_metadata.sponsor_name`, `cta_label`, etc.). The invariant is:
#
#   is_ad == true  ⇔  interaction_format.action ∈ AD_ACTIONS
#
# Ad events are synthesized by `inject_ad_events()` (Step 13c), which converts
# a small fraction (~5-8%) of commerce-adjacent organic events into ads by
# overriding their sampled action and regenerating their content block.
AD_ACTIONS: set[str] = {
    "clicked_ad",
    "hidden_ad",
    "dismissed_ad",
}

# Fixed vocabulary for ad content classification. Keep this list small and
# stable — downstream consumers (including HuggingFace publication and LLM
# judges) rely on it as a controlled category set.
AD_CATEGORIES: list[str] = [
    "food_and_beverage",
    "apparel",
    "electronics",
    "travel",
    "finance",
    "fitness_wellness",
    "education",
    "home_garden",
    "auto",
    "entertainment",
    "services",
]

# Small vocabulary for call-to-action labels. Verbatim list — the prompt
# requires the LLM to pick one of these rather than invent new copy.
AD_CTA_LABELS: list[str] = [
    "Shop now",
    "Learn more",
    "Sign up",
    "Download",
    "Get quote",
    "Book now",
]

AD_CTA_DESTINATION_KINDS: list[str] = [
    "product_page",
    "landing_page",
    "app_store",
    "signup_form",
    "checkout",
]

# Map from lowercased hashtag token (without `#`) to ad_category. A hashtag
# appearing on an event makes that event ad-eligible AND seeds the ad_category
# of any ad we synthesize over it. Eligibility is permissive — any hashtag on
# an event that hits this map suffices. The per-event sampling rate
# (AD_INJECTION_RATE) controls actual ad density.
HASHTAG_TO_AD_CATEGORY: dict[str, str] = {
    # Food / beverage
    "food": "food_and_beverage", "foodie": "food_and_beverage", "coffee": "food_and_beverage",
    "recipe": "food_and_beverage", "cooking": "food_and_beverage", "baking": "food_and_beverage",
    "wine": "food_and_beverage", "cocktails": "food_and_beverage", "brunch": "food_and_beverage",
    "restaurant": "food_and_beverage", "specialtycoffee": "food_and_beverage",
    # Apparel / beauty
    "fashion": "apparel", "ootd": "apparel", "streetwear": "apparel", "style": "apparel",
    "sneakers": "apparel", "makeup": "apparel", "skincare": "apparel", "beauty": "apparel",
    # Electronics
    "tech": "electronics", "gadgets": "electronics", "iphone": "electronics",
    "android": "electronics", "laptop": "electronics", "camera": "electronics",
    "photography": "electronics", "headphones": "electronics",
    # Travel
    "travel": "travel", "wanderlust": "travel", "vacation": "travel", "airbnb": "travel",
    "hotel": "travel", "flights": "travel", "roadtrip": "travel",
    # Finance
    "finance": "finance", "investing": "finance", "crypto": "finance",
    "personalfinance": "finance", "stocks": "finance",
    # Fitness / wellness
    "fitness": "fitness_wellness", "gym": "fitness_wellness", "running": "fitness_wellness",
    "yoga": "fitness_wellness", "workout": "fitness_wellness", "wellness": "fitness_wellness",
    "nutrition": "fitness_wellness", "bjj": "fitness_wellness",
    # Education
    "education": "education", "learning": "education", "course": "education",
    "coding": "education", "programming": "education", "bootcamp": "education",
    # Home / garden
    "home": "home_garden", "homedecor": "home_garden", "interiordesign": "home_garden",
    "gardening": "home_garden", "diy": "home_garden",
    # Auto
    "cars": "auto", "automotive": "auto", "tesla": "auto", "evs": "auto", "bmw": "auto",
    # Entertainment
    "movies": "entertainment", "film": "entertainment", "netflix": "entertainment",
    "gaming": "entertainment", "music": "entertainment", "concert": "entertainment",
    # Services
    "smallbusiness": "services", "entrepreneur": "services", "consulting": "services",
}

# Fraction of ad-eligible events to convert into ad events. 0.06 = ~6% of
# commerce-adjacent events become sponsored. The final overall ad density
# across ALL events depends on how many are commerce-adjacent in the source
# data (typically 15-30%), so the final ad share is ~1-2% of total events.
AD_INJECTION_RATE: float = 0.06

# Polarity split among ad events. 70% clicked_ad, 20% dismissed_ad,
# 10% hidden_ad — aggressive rejection is rare relative to passive dismissal.
AD_POLARITY_WEIGHTS: dict[str, float] = {
    "clicked_ad": 0.70,
    "dismissed_ad": 0.20,
    "hidden_ad": 0.10,
}


def _sample_from_distribution(dist: dict[str, float]) -> str:
    """Randomly sample one key from a weighted distribution dict."""
    keys = list(dist.keys())
    weights = list(dist.values())
    return random.choices(keys, weights=weights, k=1)[0]


CHATBOT_CONTEXTS = [
    "professional emails",
    "personal emails",
    "composing chat messages",
    "composing social media posts",
    "multilingual translation",
    "knowledge exploration",
    "therapy and reflection",
    "medical consultations",
]


def _pick_action_for_app(app: str, interaction_type: str) -> dict:
    """Pick a single action dict (action + label) for a given app/polarity.

    Called by `generate_interaction_formats()` when routing is already done
    and the pipeline just needs to pick a reasonable concrete action on the
    already-chosen app. No randomness is imposed on app assignment here —
    app assignment is done earlier by `route_personas_to_apps()`.
    """
    app_formats = PLATFORM_INTERACTION_FORMATS.get(app)
    if not app_formats:
        return {"action": "unknown", "label": "Unknown"}
    actions = app_formats.get(interaction_type)
    if not actions:
        fallback_key = "implicit_positive" if "positive" in interaction_type else "implicit_negative"
        actions = app_formats.get(fallback_key, [])
    if not actions:
        return {"action": "unknown", "label": "Unknown"}
    return random.choice(actions)


# ---------------------------------------------------------------------------
# PersonaAgent
# ---------------------------------------------------------------------------

class PersonaAgent:
    """Agent representing a single user's persona. One instance per user_id."""

    MAX_RETRIES = 3

    def __init__(
        self,
        user_id: str,
        llm_client=None,
        backend_dir: str = "backend",
        verbose: bool = False,
        max_workers: int = 20,
        llm_client_mini=None,
    ):
        self.user_id = user_id
        self.llm_client = llm_client  # Flagship QueryLLM instance (None in Claude Code mode)
        # Optional mini-tier client for generative/stylistic steps. When None,
        # mini-routed steps fall back to the flagship client. Lets cost-sensitive
        # callers configure the full pipeline with a two-tier model mix.
        self.llm_client_mini = llm_client_mini if llm_client_mini is not None else llm_client
        self.backend_dir = backend_dir
        self.verbose = verbose
        self.max_workers = max_workers  # parallel LLM API calls

        # Instance variables populated by pipeline or load_from_backend
        self.interactions: list[InteractionRow] = []
        self.atomic_personas: list[AtomicPersona] = []              # positive interactions only
        self.negative_personas: list[AtomicPersona] = []            # negative interactions
        self.cross_referenced_personas: list[CrossReferencedPersona] = []
        self.cross_referenced_negatives: list[CrossReferencedPersona] = []  # cross-referenced negative canonicals
        self.temporal_graph: list[TemporalContradiction] = []
        self.user_profile: UserProfile | None = None
        self.annotated_personas: list[AnnotatedPersona] = []

        # Train/test split state populated by build_test_split()
        self.split_labels: dict[str, str] = {}                       # persona_item -> "train" | "test"
        self.test_distractors: dict[str, list[dict]] = {}            # test persona_item -> list of {"persona_item", "category"} distractors

        # Per-user perturbed action distribution (set lazily on first use).
        # The seed is deterministic in user_id so the same user gets the
        # same action distribution across runs.
        self._user_action_distribution: dict | None = None

        # Canonical groups: normalized persona text → list of AtomicPersona instances.
        # Populated by summarize_and_cross_reference() for positive and negative separately.
        self._canonical_groups: dict[str, list] = {}
        self._negative_canonical_groups: dict[str, list] = {}
        self._merge_map: dict[str, str] = {}  # old persona_item → merged representative

        # Session infrastructure — populated by _build_sessions()
        self._sessions: list[list] = []                               # list of session groups (each = list of InteractionRow)
        self._object_id_to_session: dict[str, int] = {}               # object_id → session index
        self._row_app: dict[str, str] = {}                            # object_id → assigned app name

        # Chatbot conversation data generated by generate_chatbot_conversations().
        # Keyed by source_object_id → {"conversation": [...], "conversation_type": str, "ask_to_forget": bool}
        self._chatbot_conversations: dict[str, dict] = {}

        # Synthetic content generated by generate_synthetic_content() (step 13b).
        # Keyed by source_object_id → {"content_type": str, "content": dict}.
        # Only populated for non-Chatbot, non-implicit_negative-stub events.
        self._content_by_oid: dict[str, dict] = {}
        # Per-user content-type mix, keyed by app ("Instagram"/"Facebook"/"Threads")
        # → {"image": p, "short_video": p, "text": p}. Derived from the user's
        # observed actions with Bayesian smoothing + lognormal perturbation.
        self._user_content_mix: dict[str, dict[str, float]] = {}
        # Pre-sampled per-event action metadata populated by step 13b. Keyed by
        # source_object_id → {"action": str, "action_label": str, "itype": str}.
        # Non-Chatbot events only. save_to_backend reads from this dict when
        # available so step 13b's content_type stays consistent with the
        # final displayed action. Empty when step 13b didn't run (legacy path).
        self._action_by_oid: dict[str, dict] = {}

        # Ad events injected by `inject_ad_events()` (Step 13c). Members of
        # this set are emitted with `is_ad: true` at the event root and carry
        # ad-shaped content (with `ad_metadata` block). Non-members have
        # `is_ad: false` by default and ordinary organic content.
        self._ad_oids: set[str] = set()

        # Audit trail for canonicals dropped by the cross-polarity
        # contradiction gate (Step 3b). Each entry records the demoted
        # canonical, the opposing surviving canonical, and the reason.
        # Informational only; never written to disk.
        self._suppressed_stance_flips: list[dict] = []

        # Thread-safe set of known categories, built up during Step 1
        self._known_categories: set[str] = set()

        os.makedirs(backend_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_interactions(self, rows: list[dict]) -> None:
        """Load raw CSV dicts for this user, convert to InteractionRow, sort by time.

        Note: rows no longer get a pre-assigned platform at load time. App
        routing is done later by `route_personas_to_apps()` based on the
        per-app AppPersonas, so the random platform assignment of previous
        versions has been removed. `interaction_format` stays empty on load
        and is populated per-preference downstream.
        """
        self.interactions = []
        for row in rows:
            itype = row.get("interaction_type", "")
            self.interactions.append(InteractionRow(
                interaction_type=itype,
                user_id=row.get("user_id", str(self.user_id)),
                object_id=row.get("object_id", ""),
                interaction_time=int(row.get("interaction_time", 0)),
                object_text=row.get("object_text", ""),
                interaction_format=row.get("interaction_format", ""),
            ))
        self.interactions.sort(key=lambda r: r.interaction_time)
        if self.verbose:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Loaded {len(self.interactions)} interactions{utils.Colors.ENDC}")

    @staticmethod
    def _extract_hashtags(text: str) -> list[str]:
        """Extract hashtags from text."""
        return re.findall(r'#\w+', text)

    @staticmethod
    def _format_timestamp(unix_ts: int) -> str:
        """Convert Unix timestamp to 'HH:MM, MM/DD/YYYY'."""
        return utils.unix_to_formatted(unix_ts)

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _query_llm_with_retry(self, prompt: str) -> str | None:
        """Call the flagship LLM with retry logic. Returns response text or None."""
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.llm_client.query_llm(prompt, verbose=self.verbose)
                if response:
                    return response
            except Exception as e:
                wait = (2 ** attempt) + 1
                print(f"{utils.Colors.WARNING}[User {self.user_id}] LLM error (attempt {attempt + 1}): {e}. Retrying in {wait}s...{utils.Colors.ENDC}")
                time.sleep(wait)
        print(f"{utils.Colors.FAIL}[User {self.user_id}] LLM failed after {self.MAX_RETRIES} attempts.{utils.Colors.ENDC}")
        return None

    def _query_mini_with_retry(self, prompt: str) -> str | None:
        """Call the mini-tier LLM with retry logic. Returns response text or None.

        Falls back to the flagship client when no mini client was provided.
        Use this for generative/stylistic steps where quality bar is lower
        than ground-truth inference.
        """
        client = self.llm_client_mini or self.llm_client
        if client is None:
            return None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = client.query_llm(prompt, verbose=self.verbose)
                if response:
                    return response
            except Exception as e:
                wait = (2 ** attempt) + 1
                print(f"{utils.Colors.WARNING}[User {self.user_id}] Mini LLM error (attempt {attempt + 1}): {e}. Retrying in {wait}s...{utils.Colors.ENDC}")
                time.sleep(wait)
        print(f"{utils.Colors.FAIL}[User {self.user_id}] Mini LLM failed after {self.MAX_RETRIES} attempts.{utils.Colors.ENDC}")
        return None

    # ------------------------------------------------------------------
    # LLM Call #1: Per-interaction persona inference
    # ------------------------------------------------------------------

    def _infer_one_interaction(self, idx: int, interaction: InteractionRow) -> list[AtomicPersona]:
        """Infer atomic personas from a single interaction row (thread-safe).

        One LLM call per row. Rows are parallelised across
        `self.max_workers` threads in `infer_atomic_personas`.
        """
        # Skip implicit_negative — handled separately by hashtag-based promotion
        if interaction.interaction_type == "implicit_negative":
            return []
        hashtags = self._extract_hashtags(interaction.object_text)
        if not hashtags:
            return []

        # Snapshot current categories for context (thread-safe read)
        existing_cats = list(self._known_categories) if self._known_categories else None

        formatted_ts = self._format_timestamp(interaction.interaction_time)
        prompt = prompts.hashtag_to_persona_prompt(
            object_text=interaction.object_text,
            interaction_type=interaction.interaction_type,
            interaction_format=interaction.interaction_format,
            formatted_timestamp=formatted_ts,
            hashtags=hashtags,
            existing_categories=existing_cats,
        )

        response = self._query_llm_with_retry(prompt)
        if not response:
            return []

        parsed = utils.extract_json_from_response(response)
        if not isinstance(parsed, list):
            return []

        results: list[AtomicPersona] = []
        for item in parsed:
            if not isinstance(item, dict) or "persona_item" not in item:
                continue
            raw_confidence = float(item.get("confidence_score_init", 0.3))
            item_hashtags = item.get("source_hashtags", hashtags)
            if not isinstance(item_hashtags, list):
                item_hashtags = hashtags
            cat = item.get("category", "uncategorized")
            results.append(AtomicPersona(
                persona_item=item["persona_item"],
                category=cat,
                confidence_score_init=raw_confidence,
                source_interaction_type=interaction.interaction_type,
                source_interaction_format=interaction.interaction_format,
                source_object_id=interaction.object_id,
                source_timestamp=interaction.interaction_time,
                formatted_timestamp=formatted_ts,
                source_hashtags=item_hashtags,
            ))
            self._known_categories.add(cat.lower())
        return results

    # Weights for net-sentiment scoring of implicit negatives.
    # A hashtag is "hot negative" only when the user consistently skips it
    # AND doesn't engage positively with that topic elsewhere.
    IMPL_NEG_WEIGHT = 1.0    # each implicit_negative row
    EXPL_POS_WEIGHT = 2.0    # each explicit_positive row (strong counter-signal)
    IMPL_POS_WEIGHT = 1.0    # each implicit_positive row (moderate counter-signal)
    MIN_TEMPORAL_DAYS = 1    # must span >= 1 distinct day

    def promote_implicit_negatives(self) -> None:
        """Public entry point for implicit negative promotion (Step 2)."""
        self._promote_implicit_negatives()

    def _promote_implicit_negatives(self) -> None:
        """Promote repeated implicit_negative rows using weighted net-sentiment.

        1. For each hashtag, count occurrences across implicit_negative,
           explicit_positive, and implicit_positive rows.
        2. Compute net_score = neg*1.0 - expl_pos*3.0 - impl_pos*1.5.
           A single like cancels ~3 scroll-pasts; lingering cancels ~1.5.
        3. A hashtag is "hot" only if net_score >= MIN_IMPLICIT_NEGATIVE_REPETITION
           AND the negative rows span >= MIN_TEMPORAL_DAYS distinct days.
        4. ONE LLM call per hot hashtag, passing only that single tag.
        5. Rows with >= 2 hot hashtags are promoted; others stay as stubs.
        6. Fan out inferred preferences; keep FULL original hashtags in output.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from collections import defaultdict as _ddict

        impl_neg_rows = [r for r in self.interactions if r.interaction_type == "implicit_negative"]
        if not impl_neg_rows:
            return

        # Step 1: Count per-hashtag occurrences by interaction type
        tag_neg_oids: dict[str, set[str]] = _ddict(set)
        tag_neg_rows: dict[str, list[InteractionRow]] = _ddict(list)
        tag_neg_days: dict[str, set[int]] = _ddict(set)
        tag_expl_pos_oids: dict[str, set[str]] = _ddict(set)
        tag_impl_pos_oids: dict[str, set[str]] = _ddict(set)

        for row in self.interactions:
            tags = self._extract_hashtags(row.object_text)
            for t in tags:
                key = t.lower()
                if row.interaction_type == "implicit_negative":
                    if row.object_id not in tag_neg_oids[key]:
                        tag_neg_oids[key].add(row.object_id)
                        tag_neg_rows[key].append(row)
                        tag_neg_days[key].add(row.interaction_time // 86400)
                elif row.interaction_type == "explicit_positive":
                    tag_expl_pos_oids[key].add(row.object_id)
                elif row.interaction_type == "implicit_positive":
                    tag_impl_pos_oids[key].add(row.object_id)

        # Step 2: Compute net scores and filter to hot hashtags
        hot_tags: dict[str, list[InteractionRow]] = {}
        hot_scores: dict[str, float] = {}
        n_filtered_pos = 0
        n_filtered_days = 0

        for tag, neg_rows in tag_neg_rows.items():
            n_neg = len(neg_rows)
            n_ep = len(tag_expl_pos_oids.get(tag, set()))
            n_ip = len(tag_impl_pos_oids.get(tag, set()))
            n_days = len(tag_neg_days.get(tag, set()))
            net = (n_neg * self.IMPL_NEG_WEIGHT
                   - n_ep * self.EXPL_POS_WEIGHT
                   - n_ip * self.IMPL_POS_WEIGHT)

            if net < MIN_IMPLICIT_NEGATIVE_REPETITION:
                if n_neg >= MIN_IMPLICIT_NEGATIVE_REPETITION:
                    n_filtered_pos += 1  # would have been hot without counterevidence
                continue
            if n_days < self.MIN_TEMPORAL_DAYS:
                n_filtered_days += 1
                continue
            hot_tags[tag] = neg_rows
            hot_scores[tag] = net

        if not hot_tags:
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Implicit-negative promotion: "
                      f"0 hot hashtags after net-sentiment filter "
                      f"({n_filtered_pos} removed by positive counterevidence, "
                      f"{n_filtered_days} by temporal spread < {self.MIN_TEMPORAL_DAYS} days), "
                      f"{len(impl_neg_rows)} rows → all stubs.{utils.Colors.ENDC}")
            return

        # Step 3: Rows with >= 1 hot hashtag are promoted
        hot_tag_set = set(hot_tags.keys())
        promoted_oids: set[str] = set()
        for row in impl_neg_rows:
            tags = self._extract_hashtags(row.object_text)
            if any(t.lower() in hot_tag_set for t in tags):
                promoted_oids.add(row.object_id)

        # Step 4: Pick ONE representative per hot hashtag (longest text)
        representatives: dict[str, InteractionRow] = {
            tag: max(rows, key=lambda r: len(r.object_text))
            for tag, rows in hot_tags.items()
        }

        if self.verbose:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Implicit-negative promotion: "
                  f"{len(hot_tags)} hot hashtags (net >= {MIN_IMPLICIT_NEGATIVE_REPETITION}, "
                  f">= {self.MIN_TEMPORAL_DAYS} days), "
                  f"{n_filtered_pos} removed by positive counterevidence, "
                  f"{n_filtered_days} by temporal spread, "
                  f"{len(promoted_oids)} rows promoted (>= 1 hot tag), "
                  f"{len(representatives)} LLM calls.{utils.Colors.ENDC}")

        # Step 5: Run LLM — one call per hot hashtag, single hashtag only
        tag_personas: dict[str, list[AtomicPersona]] = {}

        pbar = tqdm(
            total=len(representatives),
            desc=f"[User {self.user_id}] Step 2: Implicit-neg promotion",
            unit="tag",
            disable=not self.verbose,
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._infer_implicit_neg_hashtag, rep, tag
                ): tag
                for tag, rep in representatives.items()
            }
            for future in as_completed(futures):
                tag = futures[future]
                pbar.update(1)
                try:
                    results = future.result()
                except Exception as e:
                    if self.verbose:
                        print(f"\n{utils.Colors.WARNING}[User {self.user_id}] Tag #{tag} error: {e}{utils.Colors.ENDC}")
                    continue
                if results:
                    tag_personas[tag] = results

        pbar.close()

        # Step 6: Fan out — for each hot hashtag, copy preferences to promoted
        # rows that actually contain that hashtag (tag-scoped, no cross-tag
        # leakage). The hot hashtag's own net-sentiment filtering is sufficient
        # corroboration; no per-preference text-matching filter needed.
        # source_hashtags keeps the FULL original set for realism.
        n_atomics = 0
        for tag, personas in tag_personas.items():
            for row in hot_tags[tag]:
                if row.object_id not in promoted_oids:
                    continue
                row_tags_lower = {t.lower() for t in self._extract_hashtags(row.object_text)}
                if tag not in row_tags_lower:
                    continue
                formatted_ts = self._format_timestamp(row.interaction_time)
                all_hashtags = self._extract_hashtags(row.object_text)
                for template in personas:
                    # Promoted rows are upgraded to explicit_negative at the
                    # atomic level — the net-sentiment gate + per-hot-tag LLM
                    # call make this a strong signal, comparable to a real
                    # explicit_negative (hide/mute/unfollow). This changes how
                    # downstream cross-referencing weighs the atomic (explicit
                    # weight 1.0 rather than implicit 0.5) and makes the count
                    # of explicit_negative non-zero at the atomic/canonical
                    # layer.
                    self.negative_personas.append(AtomicPersona(
                        persona_item=template.persona_item,
                        category=template.category,
                        confidence_score_init=template.confidence_score_init,
                        source_interaction_type="explicit_negative",
                        source_interaction_format=row.interaction_format,
                        source_object_id=row.object_id,
                        source_timestamp=row.interaction_time,
                        formatted_timestamp=formatted_ts,
                        source_hashtags=all_hashtags,
                    ))
                    n_atomics += 1

        if self.verbose:
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Implicit-negative promotion: "
                  f"{len(tag_personas)}/{len(hot_tags)} hashtags produced preferences, "
                  f"{n_atomics} atomic negatives fanned out.{utils.Colors.ENDC}")

    def _infer_implicit_neg_hashtag(
        self, interaction: InteractionRow, hot_tag: str,
    ) -> list[AtomicPersona]:
        """Run LLM on a representative row, passing only ONE hot hashtag.

        The LLM infers what the user dislikes based on the single
        repeatedly-skipped topic. Rare co-occurring tags are excluded.
        """
        existing_cats = list(self._known_categories) if self._known_categories else None
        formatted_ts = self._format_timestamp(interaction.interaction_time)
        prompt = prompts.hashtag_to_persona_prompt(
            object_text=interaction.object_text,
            interaction_type=interaction.interaction_type,
            interaction_format=interaction.interaction_format,
            formatted_timestamp=formatted_ts,
            hashtags=[hot_tag],  # single hot hashtag only
            existing_categories=existing_cats,
        )
        response = self._query_llm_with_retry(prompt)
        if not response:
            return []
        parsed = utils.extract_json_from_response(response)
        if not isinstance(parsed, list):
            return []
        results: list[AtomicPersona] = []
        for item in parsed:
            if not isinstance(item, dict) or "persona_item" not in item:
                continue
            raw_confidence = float(item.get("confidence_score_init", 0.3))
            cat = item.get("category", "uncategorized")
            results.append(AtomicPersona(
                persona_item=item["persona_item"],
                category=cat,
                confidence_score_init=raw_confidence,
                source_interaction_type=interaction.interaction_type,
                source_interaction_format=interaction.interaction_format,
                source_object_id=interaction.object_id,
                source_timestamp=interaction.interaction_time,
                formatted_timestamp=formatted_ts,
                source_hashtags=[hot_tag],
            ))
            self._known_categories.add(cat.lower())
        return results

    def infer_personas_from_hashtags(self) -> None:
        """For each interaction, call the LLM to infer atomic persona traits from hashtags.

        Uses ThreadPoolExecutor with self.max_workers parallel API calls.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self.atomic_personas = []
        self.negative_personas = []

        # --- Step 1: Infer positives & explicit negatives (skip implicit_negative) ---
        # One LLM call per row, fanned out across `self.max_workers` threads.
        pbar = tqdm(
            total=len(self.interactions),
            desc=f"[User {self.user_id}] Step 1: Inferring personas",
            unit="row",
            disable=not self.verbose,
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._infer_one_interaction, idx, interaction): (idx, interaction)
                for idx, interaction in enumerate(self.interactions)
            }
            for future in as_completed(futures):
                idx, interaction = futures[future]
                pbar.update(1)
                try:
                    results = future.result()
                except Exception as e:
                    if self.verbose:
                        print(f"\n{utils.Colors.WARNING}[User {self.user_id}] Interaction {idx} error: {e}{utils.Colors.ENDC}")
                    continue
                is_negative = "negative" in interaction.interaction_type
                target_list = self.negative_personas if is_negative else self.atomic_personas
                target_list.extend(results)

        pbar.close()

        if self.verbose:
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Inferred {len(self.atomic_personas)} positive atomic personas, "
                  f"{len(self.negative_personas)} negative (standalone) from {len(self.interactions)} interactions.{utils.Colors.ENDC}")

        if self.verbose:
            # Category statistics
            from collections import Counter as _Ctr
            cat_counts = _Ctr(ap.category.lower() for ap in self.atomic_personas)
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] {len(cat_counts)} unique categories. "
                  f"Top 10: {', '.join(f'{c}({n})' for c, n in cat_counts.most_common(10))}{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # LLM Call #2: Cross-reference and filter
    # ------------------------------------------------------------------

    def summarize_and_cross_reference(self) -> None:
        """Dedupe, filter, count corroboration, then cross-reference.

        Pipeline order:

        1. **Merge duplicates**: Lexically-identical persona_item texts
           across rows collapse into one canonical. `confidence_score_init`
           = max across duplicates.

        2. **Init filter**: Drop canonicals with max `confidence_score_init
           < MIN_PERSONA_INIT_CONFIDENCE` (0.5).

        3. **Count corroboration → confidence_cross_referenced**: For each
           surviving canonical, count the number of distinct source rows
           (source_object_id) that independently produced this persona
           AND whose individual `confidence_score_init >= MIN_PERSONA_INIT_CONFIDENCE`.
           That integer count IS `confidence_cross_referenced`. This is
           done AFTER init filtering so only quality-passing rows count.

        4. **Cross-reference distinct canonicals** via LLM for `similar` /
           `contradictory` relationship discovery. The LLM output populates
           `relationship_type` and `related_personas` but does NOT alter
           `confidence_cross_referenced` — the score is purely the
           filtered corroboration count from step 3.
        """
        if not self.atomic_personas:
            self.cross_referenced_personas = []
            return

        # --- Step 1: Merge duplicates ---
        # Group by normalized text. For each group, track:
        #   - max init (for the canonical's confidence_score_init)
        #   - all individual (source_object_id, individual_init) pairs
        #     (for the post-filter corroboration count)
        #   - earliest metadata (timestamps etc.)
        from collections import defaultdict as _defaultdict

        groups: dict[str, list] = {}  # normalized_key -> list of AtomicPersona
        canonical_order: list[str] = []

        for ap in self.atomic_personas:
            key = _normalize_persona_text(ap.persona_item)
            if not key:
                continue
            if key not in groups:
                groups[key] = []
                canonical_order.append(key)
            groups[key].append(ap)

        # Build canonicals: max init, earliest timestamp, etc.
        canonicals: list[CrossReferencedPersona] = []
        for key in canonical_order:
            atoms = groups[key]
            best = max(atoms, key=lambda a: a.confidence_score_init)
            canonicals.append(CrossReferencedPersona(
                persona_item=best.persona_item,
                category=best.category,
                confidence_score_init=best.confidence_score_init,
                confidence_cross_referenced=0.0,  # set after init filter
                relationship_type="none",
                related_personas=[],
                formatted_timestamp=atoms[0].formatted_timestamp,  # earliest
                source_interaction_type=best.source_interaction_type,
                source_interaction_format=best.source_interaction_format,
            ))

        # Persist canonical groups for later use (output fan-out, update histories, etc.)
        self._canonical_groups = groups

        if self.verbose:
            n_merged = len(self.atomic_personas) - len(canonicals)
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Merged {n_merged} duplicate atomic personas → "
                  f"{len(canonicals)} distinct canonicals.{utils.Colors.ENDC}")

        # --- Step 2: Init filter (strict — no exploration) ---
        above = [c for c in canonicals if c.confidence_score_init >= MIN_PERSONA_INIT_CONFIDENCE]
        below_count = len(canonicals) - len(above)
        survivors = above

        if self.verbose:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] After init >= {MIN_PERSONA_INIT_CONFIDENCE} filter: "
                  f"{len(above)} canonicals ({below_count} dropped below threshold, no exploration).{utils.Colors.ENDC}")

        # --- Step 3: Weighted corroboration → confidence_cross_referenced ---
        # For each surviving canonical, sum weighted contributions from
        # distinct source rows whose individual init >= threshold AND whose
        # timestamp falls within the user's trailing 7-day window.
        # Explicit rows contribute 1.0, implicit rows contribute 0.5.
        # Also record evidence mix (n_explicit_rows, n_implicit_rows) for the
        # per-canonical survival threshold later.
        recency_cutoff = _compute_recency_cutoff(self.interactions)
        for c in survivors:
            key = _normalize_persona_text(c.persona_item)
            atoms = groups.get(key, [])
            seen_sources: set[str] = set()
            base_score = 1.0
            n_expl = 0
            n_impl = 0
            for ap in atoms:
                if (ap.confidence_score_init >= MIN_PERSONA_INIT_CONFIDENCE
                        and ap.source_object_id
                        and ap.source_timestamp >= recency_cutoff):
                    if ap.source_object_id not in seen_sources:
                        seen_sources.add(ap.source_object_id)
                        if "implicit" in ap.source_interaction_type:
                            base_score += 0.5
                            n_impl += 1
                        else:
                            base_score += 1.0
                            n_expl += 1
            c.confidence_cross_referenced = base_score
            c.n_explicit_rows = n_expl
            c.n_implicit_rows = n_impl

        # --- Step 4: Per-category LLM cross-reference for relationship discovery ---
        # Group survivors by category, make one LLM call per category (parallel).

        # Short-circuit: single-row users have nothing to cross-reference.
        unique_objects_all = {ap.source_object_id for ap in self.atomic_personas}
        if len(unique_objects_all) <= 1:
            self.cross_referenced_personas = survivors
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Single interaction row — "
                      f"skipping cross-reference.{utils.Colors.ENDC}")
            return

        from collections import defaultdict as _ddict_xref
        by_category: dict[str, list[CrossReferencedPersona]] = _ddict_xref(list)
        for c in survivors:
            by_category[c.category.lower()].append(c)

        # Only cross-ref categories with 3+ canonicals (raised from 2+).
        # Tiny categories (1-2 items) produce few useful relationships and
        # aren't worth the LLM call.
        MIN_CATEGORY_SIZE = 3
        categories_to_xref = {
            cat: items for cat, items in by_category.items()
            if len(items) >= MIN_CATEGORY_SIZE
        }

        # Split into small (3-9 canonicals) and large (>= 10). Small
        # categories are grouped into batched calls to amortize overhead;
        # large categories get their own dedicated call each (the prompt
        # budget stays focused and the relationship-dense output doesn't
        # overflow).
        SMALL_CATEGORY_MAX = 9
        SMALL_BATCH_GROUP = 5  # up to 5 small categories per batched call
        small_items = [
            (cat, items) for cat, items in categories_to_xref.items()
            if len(items) <= SMALL_CATEGORY_MAX
        ]
        large_items = [
            (cat, items) for cat, items in categories_to_xref.items()
            if len(items) > SMALL_CATEGORY_MAX
        ]
        small_batches = [
            small_items[i:i + SMALL_BATCH_GROUP]
            for i in range(0, len(small_items), SMALL_BATCH_GROUP)
        ]

        if self.verbose:
            n_total_items = sum(len(v) for v in categories_to_xref.values())
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Cross-referencing {len(categories_to_xref)} categories "
                  f"({n_total_items} canonicals, "
                  f"skipping {len(by_category) - len(categories_to_xref)} small categories < {MIN_CATEGORY_SIZE}) — "
                  f"{len(large_items)} large (1 call each), {len(small_items)} small in {len(small_batches)} batched calls.{utils.Colors.ENDC}")

        canonical_by_norm = {_normalize_persona_text(c.persona_item): c for c in survivors}

        def _apply_relationships(personas_block: list) -> int:
            """Apply LLM-returned relationships to the canonicals_by_norm
            lookup. Shared between single-category and batched paths.
            """
            n_rels = 0
            for item in personas_block:
                if not isinstance(item, dict) or "persona_item" not in item:
                    continue
                my_key = _normalize_persona_text(item["persona_item"])
                if my_key not in canonical_by_norm:
                    continue
                canonical = canonical_by_norm[my_key]
                canonical.relationship_type = item.get("relationship_type", canonical.relationship_type)
                raw_related = item.get("related_personas", [])
                related = []
                for r in raw_related:
                    if isinstance(r, dict):
                        related.append(r)
                    elif isinstance(r, str):
                        related.append({"persona_item": r, "type": item.get("relationship_type", "similar")})
                canonical.related_personas = related
                n_rels += len(related)
            return n_rels

        def _xref_one_category(cat: str, items: list[CrossReferencedPersona]) -> int:
            """Cross-reference within one (large) category. Returns #relationships."""
            personas_for_prompt = [{"persona_item": c.persona_item, "category": c.category} for c in items]
            prompt = prompts.summarize_and_cross_reference_prompt(personas_for_prompt)
            response = self._query_llm_with_retry(prompt)
            if not response:
                return 0
            parsed = utils.extract_json_from_response(response)
            if not isinstance(parsed, list):
                return 0
            return _apply_relationships(parsed)

        def _xref_small_batch(batch: list[tuple[str, list[CrossReferencedPersona]]]) -> int:
            """Cross-reference a batch of small categories in ONE LLM call."""
            categories_payload = [
                {
                    "category_name": cat,
                    "personas": [
                        {"persona_item": c.persona_item, "category": c.category}
                        for c in items
                    ],
                }
                for cat, items in batch
            ]
            prompt = prompts.summarize_and_cross_reference_batched_prompt(categories_payload)
            response = self._query_llm_with_retry(prompt)
            if not response:
                return 0
            parsed = utils.extract_json_from_response(response)
            if not isinstance(parsed, list):
                return 0
            n_rels = 0
            for entry in parsed:
                if not isinstance(entry, dict):
                    continue
                n_rels += _apply_relationships(entry.get("personas", []))
            return n_rels

        from concurrent.futures import ThreadPoolExecutor, as_completed
        total_rels = 0
        total_calls = len(large_items) + len(small_batches)
        pbar_xref = tqdm(total=total_calls,
                         desc=f"[User {self.user_id}] Step 2: Cross-referencing",
                         unit="call", disable=not self.verbose)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for cat, items in large_items:
                futures.append(executor.submit(_xref_one_category, cat, items))
            for batch in small_batches:
                futures.append(executor.submit(_xref_small_batch, batch))
            for future in as_completed(futures):
                pbar_xref.update(1)
                try:
                    total_rels += future.result()
                except Exception:
                    pass
        pbar_xref.close()

        if self.verbose:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Cross-ref found {total_rels} relationships.{utils.Colors.ENDC}")

        # --- Sub-step 5: Merge similar preferences into clusters ---
        # Build clusters via union-find: similar preferences merge into one.
        # The representative is the one with the highest init score.
        # Xref scores are summed across the cluster.
        # Contradictory relationships are preserved (penalty applied after merge).
        canonical_by_item = {c.persona_item: c for c in survivors}
        parent: dict[str, str] = {c.persona_item: c.persona_item for c in survivors}

        def _find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: str, b: str) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                # Keep the one with higher init as root
                ca, cb = canonical_by_item.get(ra), canonical_by_item.get(rb)
                if ca and cb and cb.confidence_score_init > ca.confidence_score_init:
                    parent[ra] = rb
                else:
                    parent[rb] = ra

        # Union similar pairs
        for c in survivors:
            for rel in c.related_personas:
                if not isinstance(rel, dict):
                    continue
                other_name = rel.get("persona_item", "")
                if rel.get("type") == "similar" and other_name in canonical_by_item:
                    _union(c.persona_item, other_name)

        # Build clusters
        from collections import defaultdict as _ddict_merge
        clusters: dict[str, list[CrossReferencedPersona]] = _ddict_merge(list)
        for c in survivors:
            clusters[_find(c.persona_item)].append(c)

        # Merge each cluster: representative gets summed xref, max init
        merged_survivors: list[CrossReferencedPersona] = []
        self._merge_map = {}  # old persona_item → merged representative
        merge_map = self._merge_map
        for root, members in clusters.items():
            rep = max(members, key=lambda c: c.confidence_score_init)
            rep.confidence_cross_referenced = sum(m.confidence_cross_referenced for m in members)
            rep.confidence_score_init = max(m.confidence_score_init for m in members)
            # Collect contradictory relationships from all members (skip self-similar)
            contradictions = []
            for m in members:
                for rel in m.related_personas:
                    if isinstance(rel, dict) and rel.get("type") == "contradictory":
                        contradictions.append(rel)
            rep.related_personas = contradictions
            rep.relationship_type = "contradictory" if contradictions else "none"
            merged_survivors.append(rep)
            for m in members:
                merge_map[m.persona_item] = rep.persona_item

        # Update _canonical_groups to point merged atoms to the representative
        for old_name, new_name in merge_map.items():
            if old_name != new_name:
                old_key = _normalize_persona_text(old_name)
                new_key = _normalize_persona_text(new_name)
                if old_key in groups and new_key in groups:
                    groups[new_key].extend(groups[old_key])
                elif old_key in groups:
                    groups[new_key] = groups[old_key]
                # Remove stale absorbed key
                if old_key in groups and old_key != new_key:
                    del groups[old_key]

        survivors = merged_survivors
        self._canonical_groups = groups

        if self.verbose:
            n_merged_clusters = sum(1 for members in clusters.values() if len(members) > 1)
            n_merged_items = sum(len(m) for m in clusters.values() if len(m) > 1)
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Merged {n_merged_items} preferences into "
                  f"{n_merged_clusters} clusters → {len(survivors)} unique preferences.{utils.Colors.ENDC}")

        # Apply contradictory penalties — softened to 0.5 × other_base so a
        # losing canonical in a contradiction pair isn't zeroed out. The
        # contradictory signal itself is a meaningful evaluation target; full
        # subtraction killed every contradictory canonical downstream.
        base_scores: dict[str, float] = {c.persona_item: c.confidence_cross_referenced for c in survivors}
        for c in survivors:
            penalty = 0.0
            for rel in c.related_personas:
                if isinstance(rel, dict) and rel.get("type") == "contradictory":
                    other_base = base_scores.get(rel.get("persona_item", ""), 0.0)
                    penalty += 0.5 * other_base
            if penalty > 0:
                c.confidence_cross_referenced = max(0.0, c.confidence_cross_referenced - penalty)

        # --- Step 5: Log xref distribution, then bottom-20% filter + xref floor ---
        if self.verbose:
            from collections import Counter as _Counter
            xref_vals = [c.confidence_cross_referenced for c in survivors]
            buckets = _Counter()
            for v in xref_vals:
                if v < 1.0: buckets["<1.0"] += 1
                elif v < 2.0: buckets["1.0-1.9"] += 1
                elif v < 3.0: buckets["2.0-2.9"] += 1
                elif v < 5.0: buckets["3.0-4.9"] += 1
                elif v < 10.0: buckets["5.0-9.9"] += 1
                else: buckets["10.0+"] += 1
            dist_str = ", ".join(f"{k}: {buckets.get(k,0)}" for k in ["<1.0","1.0-1.9","2.0-2.9","3.0-4.9","5.0-9.9","10.0+"])
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Xref distribution (before floor): {dist_str} "
                  f"(total {len(survivors)}){utils.Colors.ENDC}")

        # Pre-label time horizons BEFORE the survival filter so short-term
        # canonicals can use the relaxed XREF_THRESHOLD_SHORT_TERM. This is
        # a deterministic rule-based pre-label; LLM refinement runs later
        # as Step 3.5 (`classify_horizons_and_stop_conditions`) and may
        # demote short→long (never promote long→short).
        self._apply_rule_based_time_horizon(survivors, polarity="positive")

        # Bottom-20 filter + xref-floor filter — with an exemption for
        # contradictory canonicals. Contradictions are a meaningful evaluation
        # target (temporal preference change, stance shifts) and are
        # structurally rare; applying the full survival bar on top of the
        # softened-but-still-negative penalty reliably killed them all.
        # Stash contradictories first, run the strict filters on the rest,
        # then re-add the stashed ones.
        contradictories = [c for c in survivors if c.relationship_type == "contradictory"]
        non_contradictory = [c for c in survivors if c.relationship_type != "contradictory"]
        non_contradictory = self._apply_bottom_20_filter(non_contradictory, min_exempt=float("inf"))
        non_contradictory = [
            c for c in non_contradictory
            if c.confidence_cross_referenced
            > canonical_xref_threshold(
                c.n_explicit_rows, c.n_implicit_rows, c.time_horizon
            )
        ]
        survivors = non_contradictory + contradictories
        self.cross_referenced_personas = survivors

        if self.verbose:
            n_contradictions = sum(1 for p in survivors if p.relationship_type == "contradictory")
            cr_vals = [c.confidence_cross_referenced for c in survivors] if survivors else [0.0]
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] {len(survivors)} positive survivors, "
                  f"{n_contradictions} contradictory, "
                  f"cross_ref range {min(cr_vals):.1f}..{max(cr_vals):.1f}{utils.Colors.ENDC}")

        # ==============================================================
        # Negative persona cross-referencing (same pipeline, independent)
        # ==============================================================
        self._cross_reference_negatives(groups_factory=_defaultdict)

    def classify_horizons_and_stop_conditions(self) -> None:
        """Step 3.5: LLM refinement of time-horizon labels + stop conditions.

        Runs AFTER cross-reference (positive + negative) so survival filters
        have already used the rule-based pre-labels. This step does two
        things:

          1. For each canonical the rule pre-labeled as `short_term`, ask the
             LLM to confirm or DEMOTE to `long_term`. The LLM cannot promote
             `long_term` → `short_term` (guards against weak long-term
             signals bypassing the xref floor).
          2. For confirmed short-term canonicals, get a structured
             `stop_condition` so eval tasks can auto-expire recommendations.

        Applies to both `cross_referenced_personas` and
        `cross_referenced_negatives`. One batched LLM call per ~20
        candidates via the mini-tier client.
        """
        client = self.llm_client_mini or self.llm_client
        if client is None:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                      f"Skipping horizon classification (no llm client).{utils.Colors.ENDC}")
            return

        # Collect rule-labeled short-term candidates (positive + negative).
        # Long-term canonicals are untouched by this step.
        obs_window_days = self._obs_window_days()

        def _candidate_payload(cr: CrossReferencedPersona, polarity: str) -> dict:
            groups = (
                self._canonical_groups if polarity == "positive"
                else self._negative_canonical_groups
            )
            key = _normalize_persona_text(cr.persona_item)
            atoms = groups.get(key, []) if groups else []
            tss = [a.source_timestamp for a in atoms if a.source_timestamp]
            first_ts = min(tss) if tss else 0
            last_ts = max(tss) if tss else 0
            return {
                "id": f"{polarity[:1]}:{cr.persona_item[:60]}",
                "persona_item": cr.persona_item,
                "category": cr.category,
                "span_days": (last_ts - first_ts) / 86400.0 if tss else 0.0,
                "n_rows": cr.n_explicit_rows + cr.n_implicit_rows,
                "first_formatted_ts": utils.unix_to_formatted(first_ts) if first_ts else "",
                "last_formatted_ts": utils.unix_to_formatted(last_ts) if last_ts else "",
                "_cr": cr,
                "_polarity": polarity,
            }

        shortterm_candidates: list[dict] = []
        for cr in self.cross_referenced_personas:
            if cr.time_horizon == "short_term":
                shortterm_candidates.append(_candidate_payload(cr, "positive"))
        for cr in self.cross_referenced_negatives:
            if cr.time_horizon == "short_term":
                shortterm_candidates.append(_candidate_payload(cr, "negative"))

        if not shortterm_candidates:
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] "
                      f"Step 3.5: no short-term candidates to refine.{utils.Colors.ENDC}")
            return

        user_profile_dict = {}
        if self.user_profile:
            user_profile_dict = {
                "name": self.user_profile.name,
                "gender": self.user_profile.gender,
                "race_ethnicity": self.user_profile.race_ethnicity,
                "career": self.user_profile.career,
                "education": self.user_profile.education,
                "bio": self.user_profile.bio,
            }

        # Batch candidates in groups of 20 for LLM calls
        BATCH = 20
        n_confirmed = 0
        n_demoted = 0
        for start in range(0, len(shortterm_candidates), BATCH):
            batch = shortterm_candidates[start:start + BATCH]
            # Strip internal references before serializing
            public_batch = [
                {k: v for k, v in c.items() if not k.startswith("_")}
                for c in batch
            ]
            prompt = prompts.horizon_and_stop_prompt(
                candidates=public_batch,
                user_profile=user_profile_dict,
                obs_window_days=obs_window_days,
            )
            response = self._query_mini_with_retry(prompt)
            if not response:
                continue
            parsed = utils.extract_json_from_response(response)
            if not isinstance(parsed, list):
                continue
            # Align by id; if the LLM drops entries or returns duplicates,
            # apply only what we can match exactly.
            by_id = {}
            for entry in parsed:
                if isinstance(entry, dict) and entry.get("id"):
                    by_id[entry["id"]] = entry
            for c in batch:
                cr: CrossReferencedPersona = c["_cr"]
                result = by_id.get(c["id"])
                if not result:
                    continue
                new_horizon = result.get("time_horizon", "short_term")
                if new_horizon == "long_term":
                    cr.time_horizon = "long_term"
                    cr.stop_condition = {}
                    n_demoted += 1
                else:
                    cr.time_horizon = "short_term"
                    sc = result.get("stop_condition")
                    if isinstance(sc, dict):
                        cr.stop_condition = {
                            "type": sc.get("type", "event"),
                            "description": sc.get("description", ""),
                            "expected_stop_ts": sc.get("expected_stop_ts"),
                        }
                    else:
                        cr.stop_condition = {}
                    n_confirmed += 1

        if self.verbose:
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] "
                  f"Horizon classification: {n_confirmed} short_term confirmed, "
                  f"{n_demoted} demoted to long_term.{utils.Colors.ENDC}")

    def resolve_cross_polarity_contradictions(self) -> None:
        """Step 3b: Cross-polarity contradiction causality gate.

        Positive and negative cross-ref pipelines run independently, which
        can produce surviving canonicals about the same topic (one pos,
        one neg) with no temporal-causality check between them. This step
        enforces a temporal-precedent rule:

          - The later-emerging stance (pos or neg) survives only if the
            count of same-polarity rows strictly BEFORE the first
            opposite-polarity row meets `MIN_STANCE_FLIP_PRIOR` (or
            `MIN_STANCE_FLIP_PRIOR_SHORT` for short-term horizons).
          - Otherwise the later canonical is demoted (dropped from
            cross_referenced_{personas,negatives}) and recorded in
            `self._suppressed_stance_flips` for audit. The surviving
            canonical gets a "contradicted" entry in update_history with
            `resolution: "suppressed_insufficient_precedent"`.

        When both stances survive (Case B), mutual "contradicted" entries
        with `resolution: "stance_shift_with_precedent"` are added to both
        canonicals' update_history, keeping the stance shift visible in
        the HTML + eval harness.

        Runs AFTER Step 3.5 (so horizon-aware precedent thresholds are
        available) and BEFORE Step 4 (so the temporal graph sees only
        surviving canonicals). Requires both the positive and negative
        cross-ref lookups already populated.
        """
        if not self.cross_referenced_personas or not self.cross_referenced_negatives:
            return

        # Build hashtag → (positives, negatives) index
        from collections import defaultdict as _ddict

        def _hashtags_for(cr: CrossReferencedPersona, polarity: str) -> set[str]:
            groups = (
                self._canonical_groups if polarity == "positive"
                else self._negative_canonical_groups
            )
            key = _normalize_persona_text(cr.persona_item)
            atoms = groups.get(key, []) if groups else []
            tags: set[str] = set()
            for a in atoms:
                for t in (a.source_hashtags or []):
                    tags.add(t.lstrip("#").lower())
            return tags

        pos_tags: dict[str, set[str]] = {
            cr.persona_item: _hashtags_for(cr, "positive")
            for cr in self.cross_referenced_personas
        }
        neg_tags: dict[str, set[str]] = {
            cr.persona_item: _hashtags_for(cr, "negative")
            for cr in self.cross_referenced_negatives
        }

        # Candidate pairs share ≥ HASHTAG_OVERLAP_MIN hashtags
        candidate_pairs: list[tuple[CrossReferencedPersona, CrossReferencedPersona, set[str]]] = []
        for pos_cr in self.cross_referenced_personas:
            p_tags = pos_tags.get(pos_cr.persona_item, set())
            if not p_tags:
                continue
            for neg_cr in self.cross_referenced_negatives:
                n_tags = neg_tags.get(neg_cr.persona_item, set())
                if not n_tags:
                    continue
                shared = p_tags & n_tags
                if len(shared) >= HASHTAG_OVERLAP_MIN:
                    candidate_pairs.append((pos_cr, neg_cr, shared))

        if not candidate_pairs:
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] "
                      f"Step 3b: no candidate pos/neg pairs share ≥{HASHTAG_OVERLAP_MIN} hashtags.{utils.Colors.ENDC}")
            return

        # LLM-confirm opposition semantics (batched)
        client = self.llm_client_mini or self.llm_client
        confirmed_pairs: list[tuple[CrossReferencedPersona, CrossReferencedPersona, set[str]]] = []
        if client is None:
            # Without an LLM, conservatively treat all shared-hashtag pairs as
            # contradictions. This is the pre-Phase-3 behavior modulo the
            # hashtag-overlap gate, so it still improves over no gate at all.
            confirmed_pairs = list(candidate_pairs)
        else:
            BATCH = 10
            for start in range(0, len(candidate_pairs), BATCH):
                batch = candidate_pairs[start:start + BATCH]
                prompt_pairs = [
                    {
                        "positive": pos.persona_item,
                        "negative": neg.persona_item,
                        "shared_hashtags": sorted(shared),
                    }
                    for (pos, neg, shared) in batch
                ]
                prompt = prompts.contradiction_pair_check_prompt(prompt_pairs)
                resp = self._query_mini_with_retry(prompt)
                if not resp:
                    # On LLM failure, default to treating all as confirmed
                    confirmed_pairs.extend(batch)
                    continue
                parsed = utils.extract_json_from_response(resp)
                if not isinstance(parsed, list):
                    confirmed_pairs.extend(batch)
                    continue
                by_id = {}
                for entry in parsed:
                    if isinstance(entry, dict) and "id" in entry:
                        by_id[int(entry["id"])] = entry
                for i, pair in enumerate(batch):
                    res = by_id.get(i)
                    if res is None or bool(res.get("is_contradiction")):
                        confirmed_pairs.append(pair)

        if not confirmed_pairs:
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] "
                      f"Step 3b: no confirmed cross-polarity contradictions.{utils.Colors.ENDC}")
            return

        # For each confirmed pair, apply the temporal-precedent rule
        def _row_tss(cr: CrossReferencedPersona, polarity: str) -> list[int]:
            groups = (
                self._canonical_groups if polarity == "positive"
                else self._negative_canonical_groups
            )
            key = _normalize_persona_text(cr.persona_item)
            atoms = groups.get(key, []) if groups else []
            return sorted(a.source_timestamp for a in atoms if a.source_timestamp)

        to_drop_pos: set[str] = set()
        to_drop_neg: set[str] = set()
        pair_results: list[dict] = []  # audit

        for pos_cr, neg_cr, shared in confirmed_pairs:
            pos_tss = _row_tss(pos_cr, "positive")
            neg_tss = _row_tss(neg_cr, "negative")
            if not pos_tss or not neg_tss:
                continue

            pos_first = pos_tss[0]
            neg_first = neg_tss[0]

            # Determine which side is the LATER-emerging stance
            if neg_first > pos_first:
                later_cr, later_polarity = neg_cr, "negative"
                earlier_cr, earlier_polarity = pos_cr, "positive"
                earlier_tss = pos_tss
                later_first = neg_first
            elif pos_first > neg_first:
                later_cr, later_polarity = pos_cr, "positive"
                earlier_cr, earlier_polarity = neg_cr, "negative"
                earlier_tss = neg_tss
                later_first = pos_first
            else:
                # Simultaneous first occurrences — treat the negative as
                # "later" since negatives are structurally rarer and more
                # likely to be the inferred-later stance.
                later_cr, later_polarity = neg_cr, "negative"
                earlier_cr, earlier_polarity = pos_cr, "positive"
                earlier_tss = pos_tss
                later_first = neg_first

            prior_count = sum(1 for t in earlier_tss if t < later_first)

            # Short-term horizon uses a relaxed precedent bar
            required = (
                MIN_STANCE_FLIP_PRIOR_SHORT
                if later_cr.time_horizon == "short_term"
                else MIN_STANCE_FLIP_PRIOR
            )

            if prior_count >= required:
                resolution = "stance_shift_with_precedent"
                # Both survive; add mutual "contradicted" entries
                entry_for_later = {
                    "update_type": "contradicted",
                    "preference": earlier_cr.persona_item,
                    "timestamp": later_first,
                    "formatted_timestamp": utils.unix_to_formatted(later_first),
                    "opposing_polarity": earlier_polarity,
                    "resolution": resolution,
                    "prior_corroboration_count": prior_count,
                    "required_precedent": required,
                }
                entry_for_earlier = {
                    "update_type": "contradicted",
                    "preference": later_cr.persona_item,
                    "timestamp": later_first,
                    "formatted_timestamp": utils.unix_to_formatted(later_first),
                    "opposing_polarity": later_polarity,
                    "resolution": resolution,
                    "prior_corroboration_count": prior_count,
                    "required_precedent": required,
                }
                later_cr.update_history = list(later_cr.update_history or []) + [entry_for_later]
                earlier_cr.update_history = list(earlier_cr.update_history or []) + [entry_for_earlier]
                pair_results.append({
                    "pos": pos_cr.persona_item, "neg": neg_cr.persona_item,
                    "resolution": resolution,
                    "prior_count": prior_count, "required": required,
                    "shared_hashtags": sorted(shared),
                })
            else:
                resolution = "suppressed_insufficient_precedent"
                # Drop the later canonical; add an annotation to the survivor
                survivor_entry = {
                    "update_type": "contradicted",
                    "preference": later_cr.persona_item,
                    "timestamp": later_first,
                    "formatted_timestamp": utils.unix_to_formatted(later_first),
                    "opposing_polarity": later_polarity,
                    "resolution": resolution,
                    "prior_corroboration_count": prior_count,
                    "required_precedent": required,
                }
                earlier_cr.update_history = list(earlier_cr.update_history or []) + [survivor_entry]
                if later_polarity == "positive":
                    to_drop_pos.add(later_cr.persona_item)
                else:
                    to_drop_neg.add(later_cr.persona_item)
                self._suppressed_stance_flips.append({
                    "dropped_persona_item": later_cr.persona_item,
                    "dropped_polarity": later_polarity,
                    "kept_persona_item": earlier_cr.persona_item,
                    "kept_polarity": earlier_polarity,
                    "prior_count": prior_count,
                    "required": required,
                    "later_first_ts": later_first,
                    "shared_hashtags": sorted(shared),
                })
                pair_results.append({
                    "pos": pos_cr.persona_item, "neg": neg_cr.persona_item,
                    "resolution": resolution,
                    "prior_count": prior_count, "required": required,
                    "shared_hashtags": sorted(shared),
                })

        # Apply drops
        if to_drop_pos:
            self.cross_referenced_personas = [
                cr for cr in self.cross_referenced_personas
                if cr.persona_item not in to_drop_pos
            ]
        if to_drop_neg:
            self.cross_referenced_negatives = [
                cr for cr in self.cross_referenced_negatives
                if cr.persona_item not in to_drop_neg
            ]

        if self.verbose:
            n_passed = sum(1 for r in pair_results if r["resolution"] == "stance_shift_with_precedent")
            n_suppressed = sum(1 for r in pair_results if r["resolution"] == "suppressed_insufficient_precedent")
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] "
                  f"Cross-polarity gate: {len(candidate_pairs)} candidate pairs, "
                  f"{len(confirmed_pairs)} confirmed, "
                  f"{n_passed} stance-shifts passed, "
                  f"{n_suppressed} suppressed.{utils.Colors.ENDC}")

    def _cross_reference_negatives(self, groups_factory=None) -> None:
        """Run merge → init filter → weighted corroboration → LLM cross-ref
        → relationship adjustment → bottom-20% filter on negative personas.

        Mirrors the positive pipeline but operates on self.negative_personas.
        """
        if not self.negative_personas:
            self.cross_referenced_negatives = []
            self._negative_canonical_groups = {}
            return

        from collections import defaultdict as _ddict

        # Step 1: Merge identical negatives
        neg_groups: dict[str, list] = {}
        neg_order: list[str] = []
        for ap in self.negative_personas:
            key = _normalize_persona_text(ap.persona_item)
            if not key:
                continue
            if key not in neg_groups:
                neg_groups[key] = []
                neg_order.append(key)
            neg_groups[key].append(ap)

        self._negative_canonical_groups = neg_groups

        neg_canonicals: list[CrossReferencedPersona] = []
        for key in neg_order:
            atoms = neg_groups[key]
            best = max(atoms, key=lambda a: a.confidence_score_init)
            neg_canonicals.append(CrossReferencedPersona(
                persona_item=best.persona_item,
                category=best.category,
                confidence_score_init=best.confidence_score_init,
                confidence_cross_referenced=0.0,
                relationship_type="none",
                related_personas=[],
                formatted_timestamp=atoms[0].formatted_timestamp,
                source_interaction_type=best.source_interaction_type,
                source_interaction_format=best.source_interaction_format,
            ))

        if self.verbose:
            n_merged = len(self.negative_personas) - len(neg_canonicals)
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Negatives: merged {n_merged} → "
                  f"{len(neg_canonicals)} distinct canonicals.{utils.Colors.ENDC}")

        # Step 2: Init filter + implicit-only repetition gate. Uses
        # MIN_NEGATIVE_INIT_CONFIDENCE (0.55) instead of the positive 0.75
        # because the hashtag_to_persona prompt caps negative scores at 0.75
        # and most "direct dislike" atoms land in 0.55-0.75. Canonicals
        # supported solely by implicit_negative rows must have at least
        # MIN_IMPLICIT_NEGATIVE_REPETITION distinct source rows; any
        # explicit-negative evidence bypasses the row-count gate.
        neg_survivors: list[CrossReferencedPersona] = []
        n_gated_implicit_only = 0
        n_gated_init = 0
        for c in neg_canonicals:
            if c.confidence_score_init < MIN_NEGATIVE_INIT_CONFIDENCE:
                n_gated_init += 1
                continue
            key = _normalize_persona_text(c.persona_item)
            atoms = neg_groups.get(key, [])
            has_explicit = any("implicit" not in a.source_interaction_type for a in atoms)
            if not has_explicit:
                distinct_rows = {a.source_object_id for a in atoms if a.source_object_id}
                if len(distinct_rows) < MIN_IMPLICIT_NEGATIVE_REPETITION:
                    n_gated_implicit_only += 1
                    continue
            neg_survivors.append(c)

        if self.verbose and n_gated_init:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Gated {n_gated_init} negative canonicals "
                  f"(init < {MIN_NEGATIVE_INIT_CONFIDENCE}).{utils.Colors.ENDC}")

        if self.verbose and n_gated_implicit_only:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Gated {n_gated_implicit_only} implicit-only "
                  f"negative canonicals (< {MIN_IMPLICIT_NEGATIVE_REPETITION} distinct rows).{utils.Colors.ENDC}")

        # Step 3: Weighted corroboration + evidence mix tracking (recency-gated)
        # Only rows within the trailing 7-day window count toward the
        # cross-ref score. The cutoff is computed from the full interactions
        # list so negatives aren't penalized by a later positive-only row
        # defining the anchor (and vice versa).
        recency_cutoff = _compute_recency_cutoff(self.interactions)
        for c in neg_survivors:
            key = _normalize_persona_text(c.persona_item)
            atoms = neg_groups.get(key, [])
            seen: set[str] = set()
            base = 1.0
            n_expl = 0
            n_impl = 0
            for ap in atoms:
                # Use the negative init floor here too — positive 0.75 would
                # discard most corroborating atoms since the prompt tops
                # negatives at 0.75 for "direct dislike".
                if (ap.confidence_score_init >= MIN_NEGATIVE_INIT_CONFIDENCE
                        and ap.source_object_id
                        and ap.source_timestamp >= recency_cutoff):
                    if ap.source_object_id not in seen:
                        seen.add(ap.source_object_id)
                        if "implicit" in ap.source_interaction_type:
                            base += 0.5
                            n_impl += 1
                        else:
                            base += 1.0
                            n_expl += 1
            c.confidence_cross_referenced = base
            c.n_explicit_rows = n_expl
            c.n_implicit_rows = n_impl

        # Step 4: LLM cross-reference for negatives
        unique_neg_objects = {ap.source_object_id for ap in self.negative_personas}
        if len(unique_neg_objects) > 1 and len(neg_survivors) > 1 and self.llm_client is not None:
            neg_for_prompt = [
                {
                    "persona_item": c.persona_item,
                    "category": c.category,
                    "confidence_score_init": c.confidence_score_init,
                    "confidence_cross_referenced": c.confidence_cross_referenced,
                    "formatted_timestamp": c.formatted_timestamp,
                    "source_interaction_type": c.source_interaction_type,
                    "source_interaction_format": c.source_interaction_format,
                }
                for c in neg_survivors
            ]
            prompt = prompts.summarize_and_cross_reference_prompt(neg_for_prompt)
            response = self._query_llm_with_retry(prompt)

            if response:
                neg_by_norm = {_normalize_persona_text(c.persona_item): c for c in neg_survivors}
                parsed = utils.extract_json_from_response(response)
                if isinstance(parsed, list):
                    for item in parsed:
                        if not isinstance(item, dict) or "persona_item" not in item:
                            continue
                        key = _normalize_persona_text(item["persona_item"])
                        if key not in neg_by_norm:
                            continue
                        canonical = neg_by_norm[key]
                        canonical.relationship_type = item.get("relationship_type", canonical.relationship_type)
                        raw_related = item.get("related_personas", [])
                        related = []
                        for r in raw_related:
                            if isinstance(r, dict):
                                related.append(r)
                            elif isinstance(r, str):
                                related.append({"persona_item": r, "type": item.get("relationship_type", "similar")})
                        canonical.related_personas = related

            # Sub-step 5: Adjust for relationships
            base_scores = {c.persona_item: c.confidence_cross_referenced for c in neg_survivors}
            for c in neg_survivors:
                adjustment = 0.0
                for rel in c.related_personas:
                    if not isinstance(rel, dict):
                        continue
                    other_name = rel.get("persona_item", "")
                    rel_type = rel.get("type", "")
                    other_base = base_scores.get(other_name, 0.0)
                    if rel_type == "similar":
                        adjustment += other_base
                    elif rel_type == "contradictory":
                        adjustment -= other_base
                c.confidence_cross_referenced = max(0.0, c.confidence_cross_referenced + adjustment)

        # Skip bottom-20% filter for negatives — keep all that passed the
        # init filter + repetition gate. The promoted negatives are already
        # high-signal (hot-hashtag gate + init ≥ 0.75 + distinct-rows gate).
        # Apply a small dedicated xref floor — decoupled from the positive
        # 20/50 thresholds, which are calibrated for data scales negatives
        # never reach (implicit_negative rows are typically 5-10x rarer than
        # implicit_positive, and explicit_negative is often 0).
        # (XREF_THRESHOLD_NEGATIVE is already close to XREF_THRESHOLD_SHORT_TERM,
        # so no horizon-aware override is needed for negatives at survival time;
        # horizons are still pre-labeled below so downstream consumers see them.)
        neg_survivors = [
            c for c in neg_survivors
            if c.confidence_cross_referenced > XREF_THRESHOLD_NEGATIVE
        ]
        self._apply_rule_based_time_horizon(neg_survivors, polarity="negative")
        self.cross_referenced_negatives = neg_survivors

        if self.verbose:
            cr_vals = [c.confidence_cross_referenced for c in neg_survivors] if neg_survivors else [0.0]
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] {len(neg_survivors)} negative survivors, "
                  f"cross_ref range {min(cr_vals):.1f}..{max(cr_vals):.1f}{utils.Colors.ENDC}")

    def _obs_window_days(self) -> float:
        """User's observation window in days (max - min interaction timestamp).

        Used by the rule-based horizon classifier to normalize each
        canonical's span. Returns 0.0 when no interactions are loaded so
        callers can short-circuit to long_term.
        """
        if not self.interactions:
            return 0.0
        tss = [r.interaction_time for r in self.interactions if r.interaction_time]
        if not tss:
            return 0.0
        span_sec = max(tss) - min(tss)
        return span_sec / 86400.0

    def _apply_rule_based_time_horizon(
        self,
        canonicals: list[CrossReferencedPersona],
        polarity: str = "positive",
    ) -> None:
        """Rule-based pre-label for each canonical's `time_horizon` field.

        Uses `_classify_time_horizon_rule` with the canonical's category,
        span-fraction, and row count. Sets `time_horizon` in place. The
        LLM refinement step (`classify_horizons_and_stop_conditions`)
        runs after cross-ref and may demote short→long.
        """
        if not canonicals:
            return
        obs_window_days = self._obs_window_days()
        groups = (
            self._canonical_groups if polarity == "positive"
            else self._negative_canonical_groups
        )
        for cr in canonicals:
            key = _normalize_persona_text(cr.persona_item)
            atoms = groups.get(key, []) if groups else []
            tss = [a.source_timestamp for a in atoms if a.source_timestamp]
            if tss:
                span_days = (max(tss) - min(tss)) / 86400.0
            else:
                span_days = 0.0
            n_total = cr.n_explicit_rows + cr.n_implicit_rows
            cr.time_horizon = _classify_time_horizon_rule(
                category=cr.category,
                span_days=span_days,
                obs_window_days=obs_window_days,
                n_total_rows=n_total,
            )

    @staticmethod
    def _apply_bottom_20_filter(
        personas: list[CrossReferencedPersona],
        min_exempt: float = 10.0,
    ) -> list[CrossReferencedPersona]:
        """Remove the bottom 20% by cross_ref count, unless count > min_exempt."""
        if len(personas) < 2:
            return personas
        sorted_by_xref = sorted(personas, key=lambda c: c.confidence_cross_referenced)
        n_remove = int(len(sorted_by_xref) * 0.2)
        to_remove: set[str] = set()
        for c in sorted_by_xref[:n_remove]:
            if c.confidence_cross_referenced <= min_exempt:
                to_remove.add(c.persona_item)
        return [c for c in personas if c.persona_item not in to_remove]

    # ------------------------------------------------------------------
    # LLM Call #3: Temporal contradiction graph
    # ------------------------------------------------------------------

    def build_temporal_contradiction_graph(self) -> None:
        """For contradictory personas, build a temporal graph of preference changes."""
        contradictions = [
            {
                "persona_item": p.persona_item,
                "confidence_score_init": p.confidence_score_init,
                "confidence_cross_referenced": p.confidence_cross_referenced,
                "formatted_timestamp": p.formatted_timestamp,
                "related_personas": p.related_personas,
            }
            for p in self.cross_referenced_personas
            if p.relationship_type == "contradictory"
        ]

        if not contradictions:
            self.temporal_graph = []
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] No contradictions found — skipping temporal graph.{utils.Colors.ENDC}")
            return

        prompt = prompts.temporal_contradiction_graph_prompt(contradictions)
        response = self._query_llm_with_retry(prompt)

        if not response:
            print(f"{utils.Colors.WARNING}[User {self.user_id}] Temporal graph LLM call failed.{utils.Colors.ENDC}")
            self.temporal_graph = []
            return

        parsed = utils.extract_json_from_response(response)
        if not isinstance(parsed, list):
            print(f"{utils.Colors.WARNING}[User {self.user_id}] Unparseable temporal graph response.{utils.Colors.ENDC}")
            self.temporal_graph = []
            return

        self.temporal_graph = []
        for group in parsed:
            if not isinstance(group, dict):
                continue
            timeline_nodes = []
            for node in group.get("timeline", []):
                if not isinstance(node, dict):
                    continue
                timeline_nodes.append(TemporalNode(
                    persona_item=node.get("persona_item", ""),
                    timestamp=int(node.get("timestamp") or 0),
                    formatted_timestamp=node.get("formatted_timestamp", ""),
                    confidence_score_init=float(node.get("confidence_score_init") or 0.0),
                    confidence_cross_referenced=float(node.get("confidence_cross_referenced") or 0.0),
                ))
            self.temporal_graph.append(TemporalContradiction(
                topic=group.get("topic", "unknown"),
                timeline=timeline_nodes,
                interpretation=group.get("interpretation", ""),
            ))

        if self.verbose:
            total_nodes = sum(len(tc.timeline) for tc in self.temporal_graph)
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Temporal graph: {len(self.temporal_graph)} topics, {total_nodes} nodes.{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # Build temporal update histories per preference
    # ------------------------------------------------------------------

    def build_update_histories(self) -> None:
        """Build update_history for each surviving canonical preference.

        Three layers of temporal evidence:

        A) **Reinforced**: When the same canonical was independently produced
           by multiple source rows at different timestamps, record sampled
           reinforcement entries (capped at 5 to avoid bloat).
        B) **Algorithmic signals** — contradicted and faded — from cross-ref
           relationships and activity-window analysis.
        C) **LLM evolution narratives** — for categories with ≥2 canonicals,
           ask the LLM to describe deepening, branching, shifting, and
           cross-category patterns. Uses session boundaries (not a fixed
           6-hour gap) as the temporal trigger.
        """
        if not self.cross_referenced_personas or not self.atomic_personas:
            return

        from collections import defaultdict as _ddict

        # Use pre-merged canonical groups (includes atoms from absorbed members)
        # with init-confidence filter applied
        groups: dict[str, list] = {}
        for key, atom_list in self._canonical_groups.items():
            filtered = [ap for ap in atom_list if ap.confidence_score_init >= MIN_PERSONA_INIT_CONFIDENCE]
            if filtered:
                groups[key] = filtered

        # Overall user activity window
        all_timestamps = [ap.source_timestamp for ap in self.atomic_personas if ap.source_timestamp]
        if not all_timestamps:
            return
        user_last_ts = max(all_timestamps)
        FADE_THRESHOLD_SECONDS = 48 * 3600
        MAX_REINFORCED_ENTRIES = 5

        # --- Part A: "reinforced" entries ---
        # For each canonical with >1 distinct source row, sample up to 5
        # recurrence timestamps evenly across the timeline.
        reinforced_entries: dict[str, list] = {}
        for cr in tqdm(self.cross_referenced_personas,
                       desc=f"[User {self.user_id}] Step 5: Update histories",
                       disable=not self.verbose):
            key = _normalize_persona_text(cr.persona_item)
            atoms = groups.get(key, [])
            # Collect distinct (source_object_id, timestamp) pairs
            seen_oids: set[str] = set()
            unique_occurrences: list[tuple[int, str]] = []  # (timestamp, source_object_id)
            for ap in sorted(atoms, key=lambda a: a.source_timestamp):
                if ap.source_object_id and ap.source_object_id not in seen_oids:
                    seen_oids.add(ap.source_object_id)
                    unique_occurrences.append((ap.source_timestamp, ap.source_object_id))

            total = len(unique_occurrences)
            if total <= 1:
                continue

            # Evenly sample up to MAX_REINFORCED_ENTRIES (skip the first — that's the "new")
            subsequent = unique_occurrences[1:]
            if len(subsequent) > MAX_REINFORCED_ENTRIES:
                step = len(subsequent) / MAX_REINFORCED_ENTRIES
                indices = [int(i * step) for i in range(MAX_REINFORCED_ENTRIES)]
                sampled = [subsequent[i] for i in indices]
            else:
                sampled = subsequent

            entries = []
            for i, (ts, oid) in enumerate(sampled, start=2):
                entries.append({
                    "update_type": "reinforced",
                    "timestamp": ts,
                    "formatted_timestamp": utils.unix_to_formatted(ts),
                    "occurrence": i,
                    "total_occurrences": total,
                    "source_object_id": oid,
                })
            reinforced_entries[cr.persona_item] = entries

        # --- Part B: contradicted + faded (algorithmic) ---
        contradicted_by: dict[str, list] = _ddict(list)
        for cr in self.cross_referenced_personas:
            if cr.relationship_type == "contradictory":
                for rel in cr.related_personas:
                    if isinstance(rel, dict) and rel.get("type") == "contradictory":
                        other = rel.get("persona_item", "")
                        if other:
                            contradicted_by[cr.persona_item].append(other)

        # --- Part C: LLM evolution narratives ---
        # Build category data for categories with ≥2 canonicals
        by_category: dict[str, list] = _ddict(list)
        for cr in self.cross_referenced_personas:
            by_category[cr.category].append(cr)

        categories_for_llm: list[dict] = []
        for cat, items in by_category.items():
            if len(items) < 2:
                continue
            prefs_data = []
            for cr in items:
                key = _normalize_persona_text(cr.persona_item)
                atoms = groups.get(key, [])
                timestamps = [a.source_timestamp for a in atoms if a.source_timestamp]
                first_ts = min(timestamps) if timestamps else 0
                last_ts = max(timestamps) if timestamps else 0
                prefs_data.append({
                    "persona_item": cr.persona_item,
                    "first_timestamp": first_ts,
                    "last_timestamp": last_ts,
                    "formatted_first": utils.unix_to_formatted(first_ts) if first_ts else "",
                    "formatted_last": utils.unix_to_formatted(last_ts) if last_ts else "",
                    "occurrence_count": len(set(a.source_object_id for a in atoms if a.source_object_id)),
                })
            categories_for_llm.append({
                "category": cat,
                "preferences": prefs_data,
            })

        # Call LLM for evolution narratives
        evolution_entries: dict[str, list] = _ddict(list)  # source_preference -> list of entries
        if categories_for_llm and self.llm_client is not None:
            prompt = prompts.preference_evolution_prompt(categories_for_llm)
            response = self._query_llm_with_retry(prompt)
            if response:
                parsed = utils.extract_json_from_response(response)
                if isinstance(parsed, list):
                    for item in parsed:
                        if not isinstance(item, dict):
                            continue
                        source = item.get("source_preference", "")
                        target = item.get("target_preference", "")
                        utype = item.get("update_type", "")
                        desc = item.get("description", "")
                        if source and target and utype:
                            evolution_entries[source].append({
                                "preference": target,
                                "update_type": utype,
                                "description": desc,
                                "timestamp": 0,  # populated below
                            })

            # Fill in timestamps for evolution entries from target preference data
            first_ts_cache: dict[str, int] = {}
            for cr in self.cross_referenced_personas:
                key = _normalize_persona_text(cr.persona_item)
                atoms = groups.get(key, [])
                first_ts_cache[cr.persona_item] = min((a.source_timestamp for a in atoms), default=0)

            for source, entries in evolution_entries.items():
                for entry in entries:
                    target = entry.get("preference", "")
                    ts = first_ts_cache.get(target, 0)
                    entry["timestamp"] = ts
                    entry["formatted_timestamp"] = utils.unix_to_formatted(ts) if ts else ""

        # --- Assemble update_history per preference ---
        for cr in self.cross_referenced_personas:
            key = _normalize_persona_text(cr.persona_item)
            atoms = groups.get(key, [])

            history = []

            # Reinforced entries
            history.extend(reinforced_entries.get(cr.persona_item, []))

            # LLM evolution entries (deepened, branched, shifted, intensified)
            history.extend(evolution_entries.get(cr.persona_item, []))

            # Contradicted
            if cr.persona_item in contradicted_by:
                for other_item in contradicted_by[cr.persona_item]:
                    other_key = _normalize_persona_text(other_item)
                    other_atoms = groups.get(other_key, [])
                    if other_atoms:
                        other_first = min(a.source_timestamp for a in other_atoms)
                        history.append({
                            "preference": other_item,
                            "update_type": "contradicted",
                            "timestamp": other_first,
                            "formatted_timestamp": utils.unix_to_formatted(other_first),
                        })

            # Faded
            if atoms:
                last_ts = max(a.source_timestamp for a in atoms)
                if (user_last_ts - last_ts) >= FADE_THRESHOLD_SECONDS:
                    history.append({
                        "update_type": "faded",
                        "timestamp": last_ts,
                        "formatted_timestamp": utils.unix_to_formatted(last_ts),
                    })

            history.sort(key=lambda h: h.get("timestamp", 0))
            cr.update_history = history

        if self.verbose:
            n_with_history = sum(1 for cr in self.cross_referenced_personas if cr.update_history)
            n_reinforced = sum(
                sum(1 for h in cr.update_history if h.get("update_type") == "reinforced")
                for cr in self.cross_referenced_personas
            )
            n_evolved = sum(
                sum(1 for h in cr.update_history if h.get("update_type") in ("deepened", "branched", "shifted", "intensified"))
                for cr in self.cross_referenced_personas
            )
            n_faded = sum(
                sum(1 for h in cr.update_history if h.get("update_type") == "faded")
                for cr in self.cross_referenced_personas
            )
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Update histories: "
                  f"{n_with_history} prefs with entries, "
                  f"{n_reinforced} reinforced, {n_evolved} evolved, {n_faded} faded.{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # LLM Call #4: Generate synthetic user profile
    # ------------------------------------------------------------------

    def generate_user_profile(self) -> None:
        """Generate a synthetic user profile (name, gender, race, career, education,
        Big Five personality, bio) from all available personas (positive + negative).

        Gender and race/ethnicity are randomly sampled from predefined distributions
        and passed to the LLM as constraints. The LLM generates everything else to
        be consistent with the personas — but is explicitly told to be diverse and
        avoid satisfying every persona to prevent stereotyping.
        """
        all_personas = list(self.cross_referenced_personas) + list(self.negative_personas)
        if not all_personas:
            self.user_profile = None
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] No personas — skipping profile generation.{utils.Colors.ENDC}")
            return

        # Sample demographics
        sampled_gender_orientation = _sample_from_distribution(GENDER_ORIENTATION_DISTRIBUTION)
        sampled_race = _sample_from_distribution(RACE_ETHNICITY_DISTRIBUTION)

        personas_summary = [p.persona_item for p in all_personas]
        prompt = prompts.generate_user_profile_prompt(
            personas=personas_summary,
            gender_orientation=sampled_gender_orientation,
            race_ethnicity=sampled_race,
        )

        response = self._query_llm_with_retry(prompt)
        if not response:
            print(f"{utils.Colors.WARNING}[User {self.user_id}] Profile generation LLM call failed.{utils.Colors.ENDC}")
            self.user_profile = None
            return

        parsed = utils.extract_json_from_response(response)
        if not isinstance(parsed, dict):
            print(f"{utils.Colors.WARNING}[User {self.user_id}] Unparseable profile response.{utils.Colors.ENDC}")
            self.user_profile = None
            return

        self.user_profile = UserProfile(
            name=parsed.get("name", ""),
            gender=sampled_gender_orientation,
            race_ethnicity=sampled_race,
            career=parsed.get("career", ""),
            education=parsed.get("education", ""),
            big_five=parsed.get("big_five", {}),
            bio=parsed.get("bio", ""),
        )

        if self.verbose:
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Profile: {self.user_profile.name} "
                  f"({self.user_profile.gender} | {self.user_profile.race_ethnicity}){utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # LLM Call #5: Annotate personas with stereotype marks
    # ------------------------------------------------------------------

    def annotate_stereotype_marks(self) -> None:
        """For each persona (positive and negative), annotate whether it is neutral,
        stereotypical, or anti-stereotypical relative to the user's demographics."""
        all_personas = list(self.cross_referenced_personas) + list(self.cross_referenced_negatives)
        if not self.user_profile or not all_personas:
            self.annotated_personas = []
            return

        personas_for_prompt = [
            {"persona_item": p.persona_item, "category": p.category}
            for p in all_personas
        ]

        prompt = prompts.annotate_stereotype_prompt(
            personas=personas_for_prompt,
            gender=self.user_profile.gender,
            race_ethnicity=self.user_profile.race_ethnicity,
        )

        response = self._query_mini_with_retry(prompt)
        if not response:
            print(f"{utils.Colors.WARNING}[User {self.user_id}] Stereotype annotation LLM call failed — skipping.{utils.Colors.ENDC}")
            self.annotated_personas = []
            return

        parsed = utils.extract_json_from_response(response)
        if not isinstance(parsed, list):
            print(f"{utils.Colors.WARNING}[User {self.user_id}] Unparseable stereotype annotation response — skipping.{utils.Colors.ENDC}")
            self.annotated_personas = []
            return

        # Build lookup from all personas for confidence scores
        all_lookup: dict[str, any] = {}
        for p in self.cross_referenced_personas:
            all_lookup[p.persona_item] = (p.confidence_score_init, p.confidence_cross_referenced, p.category)
        for p in self.cross_referenced_negatives:
            all_lookup[p.persona_item] = (p.confidence_score_init, p.confidence_cross_referenced, p.category)

        self.annotated_personas = []
        for item in parsed:
            if not isinstance(item, dict) or "persona_item" not in item:
                continue
            scores = all_lookup.get(item["persona_item"], (0.0, 0.0, "uncategorized"))
            self.annotated_personas.append(AnnotatedPersona(
                persona_item=item["persona_item"],
                category=item.get("category", scores[2]),
                confidence_score_init=scores[0],
                confidence_cross_referenced=scores[1],
                stereotype_mark=item.get("stereotype_mark", "neutral"),
            ))

        if self.verbose:
            counts = {}
            for ap in self.annotated_personas:
                counts[ap.stereotype_mark] = counts.get(ap.stereotype_mark, 0) + 1
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Stereotype annotations: {counts}{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # Step 7: Hidden Persona Inference (cross-row hashtag clustering)
    # ------------------------------------------------------------------

    def infer_hidden_personas(self) -> None:
        """Infer hidden personas from cross-row hashtag patterns.

        Four phases:
        1. Hashtag frequency census across all interaction rows.
        2. LLM thematic clustering + motivation inference.
        3. Algorithmic validation (row count, temporal spread, privacy ratio).
        4. Deduplicate hidden personas by evidence-hashtag overlap.

        Results are stored on self.user_profile.hidden_personas and
        self.user_profile.hidden_persona_summary.
        """
        if not self.user_profile or not self.interactions:
            return

        from collections import Counter
        from datetime import datetime

        # ── Phase 1: Hashtag Frequency Census ───────────────────────────

        # Per-hashtag, per-interaction-type counters
        hashtag_total: Counter = Counter()
        hashtag_by_type: dict[str, Counter] = {
            "explicit_positive": Counter(),
            "implicit_positive": Counter(),
            "explicit_negative": Counter(),
            "implicit_negative": Counter(),
        }
        # hashtag → set of distinct calendar day strings
        hashtag_days: dict[str, set] = {}
        # hashtag → set of distinct source row object_ids
        hashtag_rows: dict[str, set] = {}

        for row in self.interactions:
            tags = self._extract_hashtags(row.object_text)
            day_str = datetime.utcfromtimestamp(row.interaction_time).strftime("%Y-%m-%d")
            itype = row.interaction_type
            for tag in tags:
                hashtag_total[tag] += 1
                if itype in hashtag_by_type:
                    hashtag_by_type[itype][tag] += 1
                hashtag_days.setdefault(tag, set()).add(day_str)
                hashtag_rows.setdefault(tag, set()).add(row.object_id)

        # Filter to hashtags with >= MIN_FREQ total occurrences
        eligible = [
            (tag, count)
            for tag, count in hashtag_total.most_common()
            if count >= HIDDEN_PERSONA_HASHTAG_MIN_FREQ
        ]

        if len(eligible) < 10:
            # Not enough hashtag diversity for meaningful clustering
            return

        # Take top N for LLM
        top_hashtags = eligible[:HIDDEN_PERSONA_TOP_HASHTAGS]

        # ── Phase 1b: Intimate-Signal Pre-Screen ────────────────────────
        # Ask the LLM to flag any adult/kink/sexually-suggestive hashtags
        # among the user's positive-signal tags. Hashtags it returns get
        # force-included in the table passed to the main clustering LLM
        # (even if below MIN_FREQ), and the MIN_ROWS/MIN_DAYS gates are
        # waived for intimate_interest clusters whose evidence overlaps
        # this set — "one signal is enough" per design.
        positive_tags = sorted({
            tag for tag in hashtag_total
            if hashtag_by_type["explicit_positive"].get(tag, 0) > 0
            or hashtag_by_type["implicit_positive"].get(tag, 0) > 0
        })
        intimate_tags_lower: set[str] = set()
        if self.llm_client and positive_tags:
            screen_prompt = prompts.detect_intimate_hashtags_prompt(positive_tags)
            try:
                screen_resp = self._query_mini_with_retry(screen_prompt)
                flagged = utils.extract_json_from_response(screen_resp)
                if isinstance(flagged, list):
                    intimate_tags_lower = {
                        str(t).lstrip("#").lower() for t in flagged
                    }
            except Exception as e:
                if self.verbose:
                    print(f"{utils.Colors.WARNING}[User {self.user_id}] Intimate-tag "
                          f"screen failed: {e}{utils.Colors.ENDC}")

        # Ensure every flagged intimate tag appears in top_hashtags (even
        # if its count < MIN_FREQ and it would otherwise be dropped).
        if intimate_tags_lower:
            existing_lower = {t.lower() for t, _ in top_hashtags}
            for tag in hashtag_total:
                if tag.lower() in intimate_tags_lower and tag.lower() not in existing_lower:
                    top_hashtags.append((tag, hashtag_total[tag]))
                    existing_lower.add(tag.lower())

        # Build hashtag table for the prompt
        lines = []
        for tag, total in top_hashtags:
            ep = hashtag_by_type["explicit_positive"].get(tag, 0)
            ip = hashtag_by_type["implicit_positive"].get(tag, 0)
            en = hashtag_by_type["explicit_negative"].get(tag, 0)
            inn = hashtag_by_type["implicit_negative"].get(tag, 0)
            lines.append(f"{tag} — {total} | {ep} | {ip} | {en} | {inn}")
        hashtag_table = "\n".join(lines)

        # Build known preference list
        pref_list = []
        for cr in self.cross_referenced_personas:
            if cr.persona_item not in pref_list:
                pref_list.append(cr.persona_item)
        for cr in self.cross_referenced_negatives:
            if cr.persona_item not in pref_list:
                pref_list.append(cr.persona_item)

        # ── Phase 2: LLM Thematic Clustering ────────────────────────────

        prompt_text = prompts.infer_hidden_personas_prompt(
            gender=self.user_profile.gender,
            race_ethnicity=self.user_profile.race_ethnicity,
            career=self.user_profile.career,
            bio=self.user_profile.bio,
            preference_list=pref_list,
            hashtag_table=hashtag_table,
        )

        raw_clusters = None
        for attempt in range(self.MAX_RETRIES):
            try:
                if self.llm_client:
                    response = self.llm_client.query_llm(prompt_text)
                else:
                    # Claude Code subagent mode — this method is called
                    # inline so the LLM reasoning IS the execution.
                    return
                raw_clusters = utils.extract_json_from_response(response)
                if isinstance(raw_clusters, list):
                    break
            except Exception as e:
                if self.verbose:
                    print(f"{utils.Colors.WARNING}[User {self.user_id}] Hidden persona LLM attempt "
                          f"{attempt+1} failed: {e}{utils.Colors.ENDC}")
        if not raw_clusters or not isinstance(raw_clusters, list):
            return

        # ── Phase 3: Algorithmic Validation ──────────────────────────────

        total_rows = len(self.interactions)
        validated: list[HiddenPersona] = []

        for cluster in tqdm(raw_clusters,
                            desc=f"[User {self.user_id}] Step 7: Validating hidden personas",
                            disable=not self.verbose):
            if not isinstance(cluster, dict):
                continue
            evidence_tags = cluster.get("evidence_hashtags", [])
            if len(evidence_tags) < 3:
                continue

            # Compute evidence metrics from raw data
            tag_set_lower = set(t.lower() for t in evidence_tags)
            distinct_row_ids: set[str] = set()
            distinct_days: set[str] = set()
            itype_counts: Counter = Counter()

            for row in self.interactions:
                tags_in_row = self._extract_hashtags(row.object_text)
                tags_lower = set(t.lower() for t in tags_in_row)
                if tags_lower & tag_set_lower:
                    distinct_row_ids.add(row.object_id)
                    day_str = datetime.utcfromtimestamp(row.interaction_time).strftime("%Y-%m-%d")
                    distinct_days.add(day_str)
                    itype_counts[row.interaction_type] += 1

            n_rows = len(distinct_row_ids)
            n_days = len(distinct_days)

            # Gate: minimum rows and temporal spread. Waived for
            # intimate_interest clusters whose evidence overlaps the
            # LLM-flagged intimate hashtag set — a single positive signal
            # is enough to surface an intimate persona.
            is_intimate_exempt = (
                cluster.get("type") == "intimate_interest"
                and intimate_tags_lower
                and bool(tag_set_lower & intimate_tags_lower)
            )
            if not is_intimate_exempt:
                if n_rows < MIN_HIDDEN_PERSONA_ROWS:
                    continue
                if n_days < MIN_HIDDEN_PERSONA_DAYS:
                    continue

            # Compute privacy ratio
            ip = itype_counts.get("implicit_positive", 0)
            ep = itype_counts.get("explicit_positive", 0)
            privacy = ip / (ip + ep) if (ip + ep) > 0 else 0.0

            hp = HiddenPersona(
                label=cluster.get("label", ""),
                type=cluster.get("type", ""),
                description=cluster.get("description", ""),
                evidence_hashtags=evidence_tags,
                evidence_rows=n_rows,
                # Sorted list so JSON output is deterministic across runs.
                evidence_oids=sorted(distinct_row_ids),
                evidence_row_fraction=round(n_rows / total_rows, 4) if total_rows else 0.0,
                interaction_breakdown=dict(itype_counts),
                privacy_ratio=round(privacy, 3),
                temporal_spread_days=n_days,
                app_distribution={},  # Filled retroactively in save_to_backend after routing
                surface_connections=cluster.get("surface_connections", []),
                inferred_motivation=cluster.get("inferred_motivation", ""),
                already_captured=cluster.get("already_captured", False),
            )
            validated.append(hp)

        if not validated:
            return

        # ── Phase 4: Deduplicate by Hashtag Overlap ────────────────────
        # Merge hidden personas whose evidence_hashtags have Jaccard >= 0.5.
        # Iterative: repeat until no merges occur.
        dedup_changed = True
        while dedup_changed:
            dedup_changed = False
            i = 0
            while i < len(validated):
                j = i + 1
                while j < len(validated):
                    tags_i = set(t.lower() for t in validated[i].evidence_hashtags)
                    tags_j = set(t.lower() for t in validated[j].evidence_hashtags)
                    union = tags_i | tags_j
                    jaccard = len(tags_i & tags_j) / len(union) if union else 0.0
                    if jaccard >= 0.5:
                        # Merge: keep the one with more evidence_rows as base
                        if validated[i].evidence_rows >= validated[j].evidence_rows:
                            base_idx, donor_idx = i, j
                        else:
                            base_idx, donor_idx = j, i
                        merged = validated[base_idx]
                        other = validated[donor_idx]

                        # Union evidence_hashtags (preserve casing from base)
                        existing_lower = set(t.lower() for t in merged.evidence_hashtags)
                        for tag in other.evidence_hashtags:
                            if tag.lower() not in existing_lower:
                                merged.evidence_hashtags.append(tag)
                                existing_lower.add(tag.lower())

                        # Union surface_connections
                        existing_conns = set(merged.surface_connections)
                        for conn in other.surface_connections:
                            if conn not in existing_conns:
                                merged.surface_connections.append(conn)

                        # Recompute evidence metrics from merged hashtag set
                        tag_set_lower = set(t.lower() for t in merged.evidence_hashtags)
                        distinct_row_ids: set[str] = set()
                        distinct_days: set[str] = set()
                        itype_counts: Counter = Counter()
                        for row in self.interactions:
                            tags_in_row = self._extract_hashtags(row.object_text)
                            tags_lower = set(t.lower() for t in tags_in_row)
                            if tags_lower & tag_set_lower:
                                distinct_row_ids.add(row.object_id)
                                day_str = datetime.utcfromtimestamp(row.interaction_time).strftime("%Y-%m-%d")
                                distinct_days.add(day_str)
                                itype_counts[row.interaction_type] += 1

                        merged.evidence_rows = len(distinct_row_ids)
                        merged.evidence_oids = sorted(distinct_row_ids)
                        merged.evidence_row_fraction = round(len(distinct_row_ids) / total_rows, 4) if total_rows else 0.0
                        merged.interaction_breakdown = dict(itype_counts)
                        ip = itype_counts.get("implicit_positive", 0)
                        ep = itype_counts.get("explicit_positive", 0)
                        merged.privacy_ratio = round(ip / (ip + ep), 3) if (ip + ep) > 0 else 0.0
                        merged.temporal_spread_days = len(distinct_days)

                        # Keep base at position i, remove donor
                        if base_idx != i:
                            validated[i] = merged
                        validated.pop(j)
                        dedup_changed = True
                        # Don't increment j — list shifted
                    else:
                        j += 1
                i += 1

        if self.verbose:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Hidden persona dedup: "
                  f"{len(validated)} clusters after merging overlapping hashtag sets.{utils.Colors.ENDC}")

        # ── Generate Summary ─────────────────────────────────────────────

        summary_prompt = prompts.hidden_persona_summary_prompt(
            hidden_personas_json=json.dumps(
                [{"label": hp.label, "type": hp.type, "description": hp.description,
                  "evidence_hashtags": hp.evidence_hashtags, "evidence_rows": hp.evidence_rows,
                  "privacy_ratio": hp.privacy_ratio, "inferred_motivation": hp.inferred_motivation}
                 for hp in validated],
                indent=2,
            ),
            preference_list=pref_list,
        )

        summary_text = ""
        for attempt in range(self.MAX_RETRIES):
            try:
                if self.llm_client:
                    summary_text = self.llm_client.query_llm(summary_prompt).strip()
                    if summary_text:
                        break
            except Exception:
                pass

        # Store on profile
        self.user_profile.hidden_personas = validated
        self.user_profile.hidden_persona_summary = summary_text

        if self.verbose:
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Inferred {len(validated)} "
                  f"hidden personas{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # LLM call: MBTI inference
    # ------------------------------------------------------------------

    def infer_mbti(self) -> None:
        """Infer the user's MBTI type from Big Five + hidden personas + top hashtags.

        Single LLM call. Result stored on self.user_profile.mbti as:
            {"type": "INTJ",
             "dimensions": {"E_I": {"E": p, "I": p, "reason": "..."}, ...}}
        """
        if not self.user_profile or not self.interactions:
            return
        if not self.llm_client:
            # Subagent mode handles this inline per skill.md.
            return

        from collections import Counter

        # Top hashtags (quick local census; no threshold).
        hashtag_counter: Counter = Counter()
        for row in self.interactions:
            for tag in self._extract_hashtags(row.object_text):
                hashtag_counter[tag] += 1
        top_tags = [tag for tag, _ in hashtag_counter.most_common(50)]

        hp_brief = [
            {
                "type": hp.type,
                "label": hp.label,
                "description": hp.description,
            }
            for hp in self.user_profile.hidden_personas
        ]

        prompt_text = prompts.infer_mbti_prompt(
            big_five=self.user_profile.big_five,
            hidden_persona_summary=self.user_profile.hidden_persona_summary,
            hidden_personas_brief=hp_brief,
            top_hashtags=top_tags,
        )

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.llm_client.query_llm(prompt_text)
                parsed = utils.extract_json_from_response(response)
                if isinstance(parsed, dict) and "type" in parsed and "dimensions" in parsed:
                    self.user_profile.mbti = parsed
                    if self.verbose:
                        print(f"{utils.Colors.OKGREEN}[User {self.user_id}] MBTI: "
                              f"{parsed.get('type')}{utils.Colors.ENDC}")
                    return
            except Exception as e:
                if self.verbose:
                    print(f"{utils.Colors.WARNING}[User {self.user_id}] MBTI attempt "
                          f"{attempt+1} failed: {e}{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # LLM Call #6: Per-app sub-personas
    # ------------------------------------------------------------------

    def generate_app_personas(self) -> None:
        """Generate four distinct AppPersona objects for this user (one per
        supported app) and attach them to self.user_profile.app_personas.

        Uses the already-generated UserProfile plus a sample of the user's
        strongest preferences as conditioning context. LLM-driven.
        """
        if not self.user_profile:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] No profile — skipping app persona generation.{utils.Colors.ENDC}")
            return
        if self.llm_client is None:
            # Subagent mode does this inline per skill.md; persona_agent.py is
            # only used in API mode. Nothing to do here in subagent mode.
            return

        top_n = 20
        top_personas = [
            p.persona_item
            for p in sorted(
                self.cross_referenced_personas,
                key=lambda x: (x.confidence_score_init + x.confidence_cross_referenced),
                reverse=True,
            )[:top_n]
        ]

        profile_dict = {
            "name": self.user_profile.name,
            "gender": self.user_profile.gender,
            "race_ethnicity": self.user_profile.race_ethnicity,
            "career": self.user_profile.career,
            "education": self.user_profile.education,
            "big_five": self.user_profile.big_five,
            "bio": self.user_profile.bio,
        }

        prompt = prompts.generate_app_personas_prompt(
            profile=profile_dict,
            top_personas=top_personas,
            chatbot_contexts=CHATBOT_CONTEXTS,
        )
        response = self._query_llm_with_retry(prompt)
        if not response:
            print(f"{utils.Colors.WARNING}[User {self.user_id}] App persona generation failed.{utils.Colors.ENDC}")
            return

        parsed = utils.extract_json_from_response(response)
        if not isinstance(parsed, dict):
            print(f"{utils.Colors.WARNING}[User {self.user_id}] Unparseable app persona response.{utils.Colors.ENDC}")
            return

        app_personas = {}
        for app_name in PLATFORMS:
            entry = parsed.get(app_name)
            if not isinstance(entry, dict):
                continue
            app_personas[app_name] = AppPersona(
                app_name=app_name,
                use_purposes=list(entry.get("use_purposes", [])),
                friend_zones=list(entry.get("friend_zones", [])),
                audience_type=entry.get("audience_type", "mixed"),
                style_description=entry.get("style_description", ""),
                posting_frequency=entry.get("posting_frequency", "weekly"),
                topical_focus=list(entry.get("topical_focus", [])),
                chatbot_contexts=list(entry.get("chatbot_contexts", [])) if app_name == "Chatbot" else [],
            )

        self.user_profile.app_personas = {k: asdict(v) for k, v in app_personas.items()}

        if self.verbose:
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Generated {len(app_personas)} app personas.{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # Session grouping + row-level app routing
    # ------------------------------------------------------------------

    def _build_sessions(self) -> None:
        """Group source interactions into temporal sessions.

        Consecutive rows whose timestamp gap <= SESSION_GAP_SECONDS are
        placed in the same session. Populates self._sessions and
        self._object_id_to_session.
        """
        self._sessions = []
        self._object_id_to_session = {}
        if not self.interactions:
            return

        current_session: list = [self.interactions[0]]
        for row in self.interactions[1:]:
            prev = current_session[-1]
            if row.interaction_time - prev.interaction_time <= SESSION_GAP_SECONDS:
                current_session.append(row)
            else:
                self._sessions.append(current_session)
                current_session = [row]
        self._sessions.append(current_session)

        for idx, session in enumerate(self._sessions):
            for row in session:
                self._object_id_to_session[row.object_id] = idx

        if self.verbose:
            sizes = [len(s) for s in self._sessions]
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] {len(self._sessions)} sessions "
                  f"(rows per session: {min(sizes)}-{max(sizes)}, avg {sum(sizes)/len(sizes):.1f}){utils.Colors.ENDC}")

    def _assign_rows_to_apps(self) -> None:
        """Assign each source row to an app using session-level majority vote.

        Must be called AFTER route_personas_to_apps() (which sets
        cr.assigned_app on canonicals) and _build_sessions().

        For each source row, we look at all its atomics' canonical app
        assignments and take the majority. Then for each session, we take the
        majority across all rows and override all rows in the session. Finally,
        8% noise is applied per-session (entire session moves together).
        """
        from collections import Counter as _Counter

        # Build canonical app lookup
        canonical_app: dict[str, str] = {}
        for cr in self.cross_referenced_personas:
            canonical_app[_normalize_persona_text(cr.persona_item)] = cr.assigned_app or ""
        for cr in self.cross_referenced_negatives:
            canonical_app[_normalize_persona_text(cr.persona_item)] = cr.assigned_app or ""

        # Step 1: For each source row, majority vote from its atomics' canonical apps
        row_apps: dict[str, str] = {}  # object_id -> app
        all_atomics = list(self.atomic_personas) + list(self.negative_personas)
        from collections import defaultdict as _ddict
        atomics_by_oid: dict[str, list] = _ddict(list)
        for ap in all_atomics:
            atomics_by_oid[ap.source_object_id].append(ap)

        # Row-level interaction_type lookup for tiebreak logic.
        row_itype: dict[str, str] = {r.object_id: r.interaction_type for r in self.interactions}

        def _pick_with_chatbot_bias(votes: list[str], itype: str | None) -> str:
            """Majority vote with Chatbot tiebreak for non-negative rows.

            If the top vote tally is tied between Chatbot and another app and
            the row/session is positive (not implicit_negative), prefer
            Chatbot. Implicit_negative rows are never routed to Chatbot
            (enforced again in Step 4 below).
            """
            if not votes:
                return random.choice(PLATFORMS)
            tallies = _Counter(votes).most_common()
            top_count = tallies[0][1]
            tied_apps = [a for a, c in tallies if c == top_count]
            if len(tied_apps) == 1:
                return tied_apps[0]
            if "Chatbot" in tied_apps and itype != "implicit_negative":
                return "Chatbot"
            return tied_apps[0]

        for oid, atoms in atomics_by_oid.items():
            app_votes = []
            for ap in atoms:
                key = _normalize_persona_text(ap.persona_item)
                app = canonical_app.get(key, "")
                if app:
                    app_votes.append(app)
            row_apps[oid] = _pick_with_chatbot_bias(app_votes, row_itype.get(oid))

        # Step 2: Session majority vote — override all rows in session.
        # Tiebreak: Chatbot wins ties when at least half the session is
        # positive (not implicit_negative).
        for session in self._sessions:
            session_votes = [row_apps.get(r.object_id, "") for r in session]
            session_votes = [v for v in session_votes if v]
            session_itypes = [row_itype.get(r.object_id, "") for r in session]
            pos_share = sum(1 for it in session_itypes if it and it != "implicit_negative")
            session_is_positive = pos_share >= max(1, len(session_itypes) // 2)
            session_app = _pick_with_chatbot_bias(
                session_votes,
                "implicit_positive" if session_is_positive else "implicit_negative",
            )
            for r in session:
                row_apps[r.object_id] = session_app

        # Step 3: 8% noise per-session
        SOCIAL_PLATFORMS = [p for p in PLATFORMS if p != "Chatbot"]
        for session in self._sessions:
            if random.random() < self.NOISE_REASSIGN_PROBABILITY:
                current_app = row_apps.get(session[0].object_id, PLATFORMS[0])
                alternatives = [a for a in PLATFORMS if a != current_app]
                new_app = random.choice(alternatives)
                for r in session:
                    row_apps[r.object_id] = new_app

        # Step 4: Never route implicit_negative to Chatbot — redirect to social
        for r in self.interactions:
            if r.interaction_type == "implicit_negative" and row_apps.get(r.object_id) == "Chatbot":
                row_apps[r.object_id] = random.choice(SOCIAL_PLATFORMS)

        self._row_app = row_apps

        if self.verbose:
            from collections import Counter as _C2
            counts = _C2(self._row_app.values())
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Row→app distribution: {dict(counts)}{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # LLM Call #7: Route personas to apps (with 8% noise)
    # ------------------------------------------------------------------

    NOISE_REASSIGN_PROBABILITY = 0.08

    def route_personas_to_apps(self) -> None:
        """Assign each surviving preference to a primary app via LLM.

        Uses self.user_profile.app_personas as routing context. After the LLM
        assigns each persona to one app, a small fraction of assignments are
        randomly reassigned to a different app as "noise" to simulate
        real-world cross-app leakage.
        """
        if not self.cross_referenced_personas:
            return
        if not self.user_profile or not self.user_profile.app_personas:
            # Fallback: without app personas, randomly spread (old behavior).
            for cr in self.cross_referenced_personas:
                cr.assigned_app = random.choice(PLATFORMS)
            return
        if self.llm_client is None:
            # Subagent mode — skill.md does it inline.
            return

        preferences_for_prompt = [
            {
                "persona_item": cr.persona_item,
                "category": cr.category,
                "confidence_score_init": cr.confidence_score_init,
                "confidence_cross_referenced": cr.confidence_cross_referenced,
                "source_interaction_type": cr.source_interaction_type,
            }
            for cr in self.cross_referenced_personas
        ]

        prompt = prompts.assign_personas_to_apps_prompt(
            app_personas=self.user_profile.app_personas,
            preferences=preferences_for_prompt,
        )
        response = self._query_mini_with_retry(prompt)

        if response:
            parsed = utils.extract_json_from_response(response)
            if isinstance(parsed, list):
                assignments = {
                    item.get("persona_item", ""): item.get("assigned_app", "")
                    for item in parsed
                    if isinstance(item, dict)
                }
                for cr in self.cross_referenced_personas:
                    app = assignments.get(cr.persona_item, "")
                    if app in PLATFORMS:
                        cr.assigned_app = app

        # Fallback for anything not assigned: route by category heuristics,
        # otherwise random.
        for cr in self.cross_referenced_personas:
            if not cr.assigned_app:
                cr.assigned_app = random.choice(PLATFORMS)

        # Per-canonical noise removed — noise is now applied per-session
        # in _assign_rows_to_apps() to keep close-timestamp rows on same app.

        # Quota rebalance: push Chatbot canonical share up to ~40% by
        # migrating the lowest-xref non-Chatbot canonicals. Session-majority
        # voting downstream washes out the LLM's Chatbot share, so we pre-bias
        # the canonical pool.
        self._quota_rebalance_apps()

        if self.verbose:
            from collections import Counter
            counts = Counter(cr.assigned_app for cr in self.cross_referenced_personas)
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Canonical app routing: {dict(counts)}{utils.Colors.ENDC}")

    # Target shares for the canonical-level distribution before session voting.
    CHATBOT_CANONICAL_TARGET = 0.40
    SOCIAL_CANONICAL_FLOOR = 0.17

    def _quota_rebalance_apps(self) -> None:
        """Enforce soft quotas on the canonical-level app distribution.

        If Chatbot's share is below CHATBOT_CANONICAL_TARGET, migrate the
        lowest-priority non-Chatbot canonicals (lowest xref; introspective /
        knowledge-seeking categories first) into Chatbot until the share
        reaches the target. Symmetrically, if any social app is below
        SOCIAL_CANONICAL_FLOOR, top it up from Chatbot surplus.
        """
        pool = list(self.cross_referenced_personas)
        if not pool:
            return
        n = len(pool)
        target_cb = int(round(n * self.CHATBOT_CANONICAL_TARGET))
        social_floor = int(round(n * self.SOCIAL_CANONICAL_FLOOR))

        from collections import Counter as _C
        counts = _C(cr.assigned_app for cr in pool)
        cb_count = counts.get("Chatbot", 0)

        # Introspective hints (category substrings) that should default to Chatbot.
        introspective_keywords = (
            "knowledge", "learning", "curiosity", "curious",
            "reflection", "identity", "values", "belief",
            "aspiration", "goal", "personal", "private",
            "health", "medical", "therapy", "emotion",
        )

        def _priority_for_chatbot_migration(cr) -> tuple:
            cat = (cr.category or "").lower()
            intro_hit = any(k in cat for k in introspective_keywords)
            # Lower tuple sorts first → migrate first.
            # Prefer introspective categories (True first by using 0) and low xref.
            return (0 if intro_hit else 1, cr.confidence_cross_referenced)

        if cb_count < target_cb:
            non_cb = [cr for cr in pool if cr.assigned_app != "Chatbot"]
            non_cb.sort(key=_priority_for_chatbot_migration)
            deficit = target_cb - cb_count
            for cr in non_cb[:deficit]:
                cr.assigned_app = "Chatbot"

        # Social-app floors: top up starved social apps from Chatbot surplus
        # (only if Chatbot is well above its target).
        counts = _C(cr.assigned_app for cr in pool)
        for app in PLATFORMS:
            if app == "Chatbot":
                continue
            if counts.get(app, 0) >= social_floor:
                continue
            shortfall = social_floor - counts.get(app, 0)
            surplus_cb = counts.get("Chatbot", 0) - target_cb
            if surplus_cb <= 0:
                break
            # Migrate the lowest-xref Chatbot canonicals into this starved app.
            chatbot_crs = [cr for cr in pool if cr.assigned_app == "Chatbot"]
            chatbot_crs.sort(key=lambda cr: cr.confidence_cross_referenced)
            n_move = min(shortfall, surplus_cb)
            for cr in chatbot_crs[:n_move]:
                cr.assigned_app = app
            counts = _C(cr.assigned_app for cr in pool)

    # ------------------------------------------------------------------
    # LLM Call #8: Generate interaction_format objects
    #   - action + action_label come from PLATFORM_INTERACTION_FORMATS
    #     VERBATIM (catalog lookup — no wording regeneration)
    #   - user_message only for AT_AI_ACTIONS (social-media @ai comments)
    #     and CHATBOT_TURN_ACTIONS (natural chat turns on the Chatbot app)
    # ------------------------------------------------------------------

    def _ensure_user_action_distribution(self, noise_strength: float = 0.6) -> dict:
        """Lazily build the per-user perturbed action distribution.

        Seeded deterministically on the user's id so the same user gets
        the same personal distribution across runs.
        """
        if self._user_action_distribution is None:
            try:
                seed = int(str(self.user_id))
            except (ValueError, TypeError):
                seed = abs(hash(str(self.user_id))) % (2**31)
            self._user_action_distribution = build_user_action_distribution(
                seed, noise_strength=noise_strength
            )
        return self._user_action_distribution

    def _sample_action_from_bucket(self, app: str, interaction_type: str, rng: random.Random) -> dict:
        """Sample a single action entry from this user's perturbed bucket
        using weighted random choice."""
        dist = self._ensure_user_action_distribution()
        bucket = dist.get(app, {}).get(interaction_type)
        if not bucket:
            fallback_key = "implicit_positive" if "positive" in interaction_type else "implicit_negative"
            bucket = dist.get(app, {}).get(fallback_key, [])
        if not bucket:
            return {"action": "unknown", "label": "Unknown", "weight": 0.0}
        weights = [e["weight"] for e in bucket]
        if sum(weights) <= 0:
            return bucket[0]
        return rng.choices(bucket, weights=weights, k=1)[0]

    def generate_interaction_formats(self) -> None:
        """For each routed preference, sample a concrete interaction_format
        from this user's perturbed catalog and generate a user_message if
        the chosen action calls for one.

        Action identifiers and labels always come from
        `PLATFORM_INTERACTION_FORMATS` — the sampler picks ONE entry by
        action identifier, never invents new wording. The sampling is
        weighted by per-user perturbed probabilities (see
        `_perturb_weights`) so different users show visibly different
        action distributions while still roughly matching real-world
        patterns (likes >> comments >> shares, etc).

        If `self.llm_client` is set AND an action that requires a
        `user_message` is sampled, we call the LLM once per such item to
        generate the natural-language message. Otherwise user_message is
        left null and can be filled in later.
        """
        if not self.cross_referenced_personas:
            return
        if not self.user_profile or not self.user_profile.app_personas:
            return

        # Seeded RNG for reproducible sampling per user
        try:
            sampler_seed = int(str(self.user_id)) * 7919 + 131
        except (ValueError, TypeError):
            sampler_seed = abs(hash(str(self.user_id))) % (2**31)
        rng = random.Random(sampler_seed)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Pre-sample actions (deterministic, seeded RNG — must be sequential)
        action_plan: list[tuple[CrossReferencedPersona, str, str, str, bool]] = []
        for cr in self.cross_referenced_personas:
            app = cr.assigned_app or random.choice(PLATFORMS)
            entry = self._sample_action_from_bucket(app, cr.source_interaction_type, rng)
            action_id = entry["action"]
            canonical_label = entry["label"]
            needs_msg = action_id in AT_AI_ACTIONS or action_id in CHATBOT_TURN_ACTIONS
            action_plan.append((cr, app, action_id, canonical_label, needs_msg))

        # Parallel LLM calls for user_message generation
        def _gen_format(item):
            cr, app, action_id, canonical_label, needs_msg = item
            user_message = None
            if needs_msg and self.llm_client is not None:
                app_persona_dict = self.user_profile.app_personas.get(app, {})
                prompt = prompts.generate_interaction_format_prompt(
                    persona_item=cr.persona_item,
                    category=cr.category,
                    interaction_type=cr.source_interaction_type,
                    assigned_app=app,
                    app_persona=app_persona_dict,
                    action_catalog=[{"action": action_id, "label": canonical_label}],
                    requires_user_message=True,
                )
                response = self._query_mini_with_retry(prompt)
                if response:
                    parsed = utils.extract_json_from_response(response)
                    if isinstance(parsed, dict):
                        user_message = parsed.get("user_message")
            format_obj = {
                "app": app,
                "action": action_id,
                "action_label": canonical_label,
                "user_message": user_message if needs_msg else None,
            }
            return cr, json.dumps(format_obj)

        pbar = tqdm(total=len(action_plan),
                    desc=f"[User {self.user_id}] Step 8: Interaction formats",
                    unit="pref", disable=not self.verbose)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_gen_format, item): item for item in action_plan}
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    cr, fmt_json = future.result()
                    cr.source_interaction_format = fmt_json
                except Exception:
                    pass

        pbar.close()

        if self.verbose:
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Interaction formats sampled from perturbed catalog.{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # Chatbot multi-turn conversation generation
    # ------------------------------------------------------------------

    def generate_chatbot_conversations(self) -> None:
        """Generate multi-turn conversations for chatbot events.

        Generates one conversation per chatbot EVENT (source row), with each
        conversation naturally embedding ALL surviving preferences for that
        event. This replaces the old per-canonical approach.

        Results are stored in ``self._chatbot_conversations`` (keyed by
        source_object_id) and merged into the Chatbot records at save time.
        """
        if not self.cross_referenced_personas or not self.user_profile:
            return
        if self.llm_client is None:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                      f"Skipping chatbot conversation generation (no llm_client).{utils.Colors.ENDC}")
            return

        chatbot_persona = self.user_profile.app_personas.get("Chatbot")
        if not chatbot_persona:
            return
        if isinstance(chatbot_persona, AppPersona):
            chatbot_persona_dict = {
                "app_name": chatbot_persona.app_name,
                "use_purposes": chatbot_persona.use_purposes,
                "friend_zones": chatbot_persona.friend_zones,
                "audience_type": chatbot_persona.audience_type,
                "style_description": chatbot_persona.style_description,
                "posting_frequency": chatbot_persona.posting_frequency,
                "topical_focus": chatbot_persona.topical_focus,
                "chatbot_contexts": chatbot_persona.chatbot_contexts,
            }
        else:
            chatbot_persona_dict = chatbot_persona

        user_profile_dict = {
            "name": self.user_profile.name,
            "gender": self.user_profile.gender,
            "race_ethnicity": self.user_profile.race_ethnicity,
            "career": self.user_profile.career,
            "education": self.user_profile.education,
            "bio": self.user_profile.bio,
        }

        # Build canonical lookup (same as save_to_backend) to find surviving
        # preferences for each chatbot event.
        canonical_lookup: dict[str, CrossReferencedPersona] = {}
        for cr in self.cross_referenced_personas:
            cr_key = _normalize_persona_text(cr.persona_item)
            canonical_lookup[cr_key] = cr
            for ap in self._canonical_groups.get(cr_key, []):
                ap_key = _normalize_persona_text(ap.persona_item)
                if ap_key not in canonical_lookup:
                    canonical_lookup[ap_key] = cr
        for cr in self.cross_referenced_negatives:
            cr_key = _normalize_persona_text(cr.persona_item)
            canonical_lookup[cr_key] = cr
            for ap in self._negative_canonical_groups.get(cr_key, []):
                ap_key = _normalize_persona_text(ap.persona_item)
                if ap_key not in canonical_lookup:
                    canonical_lookup[ap_key] = cr

        # Group atomics by source_object_id
        from collections import defaultdict
        atomics_by_oid: dict[str, list] = defaultdict(list)
        for ap in self.atomic_personas:
            atomics_by_oid[ap.source_object_id].append(ap)
        for ap in self.negative_personas:
            atomics_by_oid[ap.source_object_id].append(ap)

        # Build per-event records for chatbot-routed rows
        chatbot_records: list[dict] = []
        for oid, atoms in atomics_by_oid.items():
            if self._row_app.get(oid) != "Chatbot":
                continue
            if not atoms:
                continue

            # Collect surviving preferences for this event (deduped)
            seen_items: set[str] = set()
            prefs: list[dict] = []
            for ap in atoms:
                key = _normalize_persona_text(ap.persona_item)
                cr = canonical_lookup.get(key)
                if not cr or not isinstance(cr, CrossReferencedPersona):
                    continue
                if cr.persona_item in seen_items:
                    continue
                seen_items.add(cr.persona_item)
                prefs.append({
                    "persona_item": cr.persona_item,
                    "category": cr.category,
                    "interaction_type": ap.source_interaction_type,
                })

            if not prefs:
                continue

            rep = atoms[0]
            chatbot_records.append({
                "source_object_id": oid,
                "preferences": prefs,
                "source_interaction_type": rep.source_interaction_type,
                "interaction_format": {},
            })

        if not chatbot_records:
            return

        try:
            user_seed = int(str(self.user_id)) * 7919 + 131
        except (ValueError, TypeError):
            user_seed = abs(hash(str(self.user_id))) % (2**31)

        chatbot_conversation.generate_chatbot_conversations(
            chatbot_records=chatbot_records,
            user_profile=user_profile_dict,
            chatbot_persona=chatbot_persona_dict,
            llm_query_fn=self._query_llm_with_retry,
            user_seed=user_seed,
            max_workers=self.max_workers,
        )

        # Store results keyed by source_object_id
        for rec in chatbot_records:
            if rec.get("conversation"):
                self._chatbot_conversations[rec["source_object_id"]] = {
                    "conversation": rec["conversation"],
                    "conversation_type": rec["conversation_type"],
                    "ask_to_forget": rec["ask_to_forget"],
                    "interaction_format_override": rec.get("interaction_format"),
                }

        if self.verbose:
            n_conv = len(self._chatbot_conversations)
            n_total = len(chatbot_records)
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] "
                  f"Generated {n_conv}/{n_total} chatbot conversations.{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # Step 13b — Synthetic per-event content generation
    # ------------------------------------------------------------------

    def _compute_user_content_mix(self) -> None:
        """Derive this user's per-app content-type mix from their observed actions.

        Three layers (see module-level constants):
          (1) Platform prior  (PLATFORM_CONTENT_PRIOR)
          (2) Bayesian smoothing with PRIOR_PSEUDOCOUNT pseudo-events
          (3) Per-user lognormal perturbation (CONTENT_MIX_NOISE_SIGMA)

        Populates self._user_content_mix[app] = {image, short_video, text}.
        Reads from self._action_by_oid, so must run AFTER action pre-sampling.
        """
        import math
        for app in ("Instagram", "Facebook", "Threads"):
            img, vid, txt = 0.0, 0.0, 0.0
            for oid, meta in self._action_by_oid.items():
                if self._row_app.get(oid) != app:
                    continue
                hint = ACTION_CONTENT_HINTS.get(app, {}).get(meta["action"], "ambiguous")
                if hint == "image":
                    img += 1.0
                elif hint == "video":
                    vid += 1.0
                elif hint == "text":
                    txt += 1.0
                elif hint == "story_amb":
                    img += 0.5
                    vid += 0.5
                # "ambiguous" adds no observed weight

            N = img + vid + txt
            prior = PLATFORM_CONTENT_PRIOR[app]
            posterior = {
                "image":       (img + PRIOR_PSEUDOCOUNT * prior["image"])       / (N + PRIOR_PSEUDOCOUNT),
                "short_video": (vid + PRIOR_PSEUDOCOUNT * prior["short_video"]) / (N + PRIOR_PSEUDOCOUNT),
                "text":        (txt + PRIOR_PSEUDOCOUNT * prior["text"])        / (N + PRIOR_PSEUDOCOUNT),
            }

            # Per-user lognormal perturbation so two users with identical
            # action histories still land on distinct mixes.
            mix_rng = random.Random(f"content_mix::{self.user_id}::{app}")
            noisy = {
                k: max(1e-6, v * math.exp(mix_rng.gauss(0.0, CONTENT_MIX_NOISE_SIGMA)))
                for k, v in posterior.items()
            }
            total = sum(noisy.values())
            self._user_content_mix[app] = {k: v / total for k, v in noisy.items()}

    def _resolve_content_type(self, app: str, action: str, oid: str) -> str:
        """Resolve content_type for one event, respecting action constraints.

        Returns one of "text" / "image" / "short_video". Deterministic per
        (user_id, oid) via seeded RNG.
        """
        hint = ACTION_CONTENT_HINTS.get(app, {}).get(action, "ambiguous")
        if hint == "video":
            return "short_video"
        if hint in ("image", "text"):
            return hint

        mix = self._user_content_mix.get(app) or PLATFORM_CONTENT_PRIOR.get(app, {
            "image": 0.4, "short_video": 0.3, "text": 0.3
        })
        rng = random.Random(f"content_type::{self.user_id}::{oid}")

        if hint == "story_amb":
            # Stories can't be text; weight image/video by user's mix ratio.
            denom = mix["image"] + mix["short_video"]
            r = mix["image"] / denom if denom > 0 else 0.5
            return "image" if rng.random() < r else "short_video"

        # "ambiguous" — sample from full 3-way mix.
        return rng.choices(
            ["image", "short_video", "text"],
            weights=[mix["image"], mix["short_video"], mix["text"]],
        )[0]

    @staticmethod
    def _empty_content(content_type: str) -> dict:
        """Minimal placeholder emitted when a content-generation LLM call fails."""
        if content_type == "text":
            return {"text": "(content unavailable)"}
        if content_type == "image":
            return {
                "caption": "(content unavailable)",
                "overall_description": "",
                "parts": [],
                "metadata": {},
            }
        if content_type == "short_video":
            return {
                "title": "(content unavailable)",
                "caption": "",
                "overall_description": "",
                "key_frames": [],
                "audio_transcript": "",
                "metadata": {},
            }
        return {}

    def generate_synthetic_content(self) -> None:
        """Step 13b: Generate synthetic textual content for each non-Chatbot,
        non-stub interaction event.

        One mini-tier LLM call per event — parallelized with ThreadPoolExecutor.
        Also pre-samples action + itype for each non-Chatbot event so save_to_backend
        can display an action consistent with the generated content_type.

        Populates:
          - self._action_by_oid[oid] = {action, action_label, itype}
          - self._user_content_mix[app] = {image, short_video, text}
          - self._content_by_oid[oid] = {content_type, content}

        Skipped when neither a mini nor flagship LLM client is available.
        Chatbot events keep using their own `conversation` field, and
        implicit_negative stubs render as greyscale timeline markers with
        no content attached.
        """
        if not self.user_profile:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                      f"Skipping content gen (no user profile).{utils.Colors.ENDC}")
            return
        client = self.llm_client_mini or self.llm_client
        if client is None:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                      f"Skipping content gen (no llm client).{utils.Colors.ENDC}")
            return

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from collections import defaultdict as _dd

        # --- Build the same canonical lookup save_to_backend uses ---
        canonical_lookup: dict[str, CrossReferencedPersona] = {}
        for cr in self.cross_referenced_personas:
            cr_key = _normalize_persona_text(cr.persona_item)
            canonical_lookup[cr_key] = cr
            for ap in self._canonical_groups.get(cr_key, []):
                ap_key = _normalize_persona_text(ap.persona_item)
                canonical_lookup.setdefault(ap_key, cr)
        for cr in self.cross_referenced_negatives:
            cr_key = _normalize_persona_text(cr.persona_item)
            canonical_lookup[cr_key] = cr
            for ap in self._negative_canonical_groups.get(cr_key, []):
                ap_key = _normalize_persona_text(ap.persona_item)
                canonical_lookup.setdefault(ap_key, cr)

        atomics_by_oid: dict[str, list] = _dd(list)
        for ap in self.atomic_personas:
            atomics_by_oid[ap.source_object_id].append(ap)
        for ap in self.negative_personas:
            atomics_by_oid[ap.source_object_id].append(ap)

        # --- Pass 1: pre-sample action + itype for each event ---
        # Deterministic: same seed as save_to_backend's event_rng.
        try:
            _ev_seed = int(str(self.user_id)) * 7919 + 131
        except (ValueError, TypeError):
            _ev_seed = abs(hash(str(self.user_id))) % (2**31)
        event_rng = random.Random(_ev_seed)

        for oid, atoms in atomics_by_oid.items():
            app = self._row_app.get(oid) or PLATFORMS[0]
            rep = atoms[0]

            # Mirror save_to_backend's "skip events with zero surviving prefs unless implicit_negative"
            surviving_count = sum(
                1 for ap in atoms
                if canonical_lookup.get(_normalize_persona_text(ap.persona_item))
            )
            if surviving_count == 0 and rep.source_interaction_type != "implicit_negative":
                continue  # event will be dropped; do not advance rng for it

            # Mirror save_to_backend's itype promotion + Chatbot 20%-flip.
            itype = rep.source_interaction_type or "implicit_positive"
            if itype == "implicit_negative" and surviving_count > 0:
                itype = "explicit_negative"
            if app == "Chatbot" and itype != "implicit_negative":
                polarity = "negative" if "negative" in itype else "positive"
                itype = f"explicit_{polarity}" if event_rng.random() < 0.20 else f"implicit_{polarity}"

            sampled = self._sample_action_from_bucket(app, itype, event_rng)
            self._action_by_oid[oid] = {
                "action": sampled.get("action", "unknown"),
                "action_label": sampled.get("label", ""),
                "itype": itype,
            }

        # --- Compute per-user content mix from observed actions ---
        self._compute_user_content_mix()

        # --- Pass 2: build event records for LLM content generation ---
        # Skip Chatbot and implicit_negative stubs.
        app_persona_dicts: dict[str, dict] = {}
        for app_name in ("Instagram", "Facebook", "Threads"):
            persona = self.user_profile.app_personas.get(app_name)
            if isinstance(persona, AppPersona):
                app_persona_dicts[app_name] = {
                    "use_purposes": persona.use_purposes,
                    "friend_zones": persona.friend_zones,
                    "audience_type": persona.audience_type,
                    "style_description": persona.style_description,
                    "posting_frequency": persona.posting_frequency,
                    "topical_focus": persona.topical_focus,
                }
            elif isinstance(persona, dict):
                app_persona_dicts[app_name] = persona
            else:
                app_persona_dicts[app_name] = {}

        user_profile_dict = {
            "name": self.user_profile.name,
            "gender": self.user_profile.gender,
            "race_ethnicity": self.user_profile.race_ethnicity,
            "career": self.user_profile.career,
            "education": self.user_profile.education,
            "bio": self.user_profile.bio,
        }

        events_to_generate: list[dict] = []
        for oid, meta in self._action_by_oid.items():
            app = self._row_app.get(oid) or PLATFORMS[0]
            if app == "Chatbot":
                continue
            if meta["itype"] == "implicit_negative":
                continue  # stubs stay content-less
            atoms = atomics_by_oid.get(oid, [])
            if not atoms:
                continue
            rep = atoms[0]

            # Hashtags for this event
            event_hashtags = rep.source_hashtags
            if not event_hashtags:
                all_tags: list[str] = []
                for ap in atoms:
                    all_tags.extend(ap.source_hashtags)
                event_hashtags = list(dict.fromkeys(all_tags))

            # Surviving preferences (deduped)
            seen_items: set[str] = set()
            prefs: list[dict] = []
            for ap in atoms:
                cr = canonical_lookup.get(_normalize_persona_text(ap.persona_item))
                if not cr or not isinstance(cr, CrossReferencedPersona):
                    continue
                if cr.persona_item in seen_items:
                    continue
                seen_items.add(cr.persona_item)
                prefs.append({"persona_item": cr.persona_item, "category": cr.category})

            content_type = self._resolve_content_type(app, meta["action"], oid)
            events_to_generate.append({
                "oid": oid,
                "app": app,
                "action": meta["action"],
                "action_label": meta["action_label"],
                "content_type": content_type,
                "hashtags": event_hashtags,
                "preferences": prefs,
            })

        if not events_to_generate:
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] "
                      f"No events to generate content for.{utils.Colors.ENDC}")
            return

        # --- Pass 3: parallel LLM calls ---
        def _gen_one(ev: dict):
            try:
                prompt = prompts.generate_synthetic_content_prompt(
                    content_type=ev["content_type"],
                    app=ev["app"],
                    app_persona=app_persona_dicts.get(ev["app"], {}),
                    user_profile=user_profile_dict,
                    hashtags=ev["hashtags"],
                    preferences=ev["preferences"],
                    action=ev["action"],
                    action_label=ev["action_label"],
                )
                response = self._query_mini_with_retry(prompt)
                content = None
                if response:
                    parsed = utils.extract_json_from_response(response)
                    if isinstance(parsed, dict):
                        # Accept either {content_type, content} or a flat content dict
                        if "content" in parsed and isinstance(parsed["content"], dict):
                            content = parsed["content"]
                        elif set(parsed.keys()) >= {"text"} or set(parsed.keys()) >= {"caption"}:
                            content = parsed
                return ev["oid"], ev["content_type"], content
            except Exception:
                return ev["oid"], ev["content_type"], None

        pbar = tqdm(total=len(events_to_generate),
                    desc=f"[User {self.user_id}] Step 13b: Synthetic content",
                    unit="event", disable=not self.verbose)

        n_success = 0
        n_placeholder = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_gen_one, ev): ev for ev in events_to_generate}
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    oid, ctype, content = future.result()
                    if content is not None:
                        self._content_by_oid[oid] = {"content_type": ctype, "content": content}
                        n_success += 1
                    else:
                        self._content_by_oid[oid] = {
                            "content_type": ctype,
                            "content": self._empty_content(ctype),
                        }
                        n_placeholder += 1
                except Exception as e:
                    print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                          f"Content-gen future failed: {e}{utils.Colors.ENDC}")

        pbar.close()

        if self.verbose:
            total = len(events_to_generate)
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] "
                  f"Synthetic content: {n_success}/{total} generated, "
                  f"{n_placeholder} placeholders.{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # Ad injection (Step 13c)
    # ------------------------------------------------------------------

    def inject_ad_events(self) -> None:
        """Step 13c: Convert a small fraction of commerce-adjacent events into
        sponsored-ad events with overridden action + ad-shaped content.

        Runs AFTER `generate_synthetic_content` so organic content has already
        been produced. For each event selected as an ad:
          - action is swapped to an AD_ACTIONS entry (70% clicked_ad,
            20% dismissed_ad, 10% hidden_ad)
          - content is regenerated via one LLM call using the
            `synthesize_ad_content_prompt` (includes `ad_metadata` with
            sponsor_name, ad_category, cta_label, etc.)
          - the event's oid is added to `self._ad_oids`, which save_to_backend
            reads to emit `is_ad: true` at the event root.

        Social apps only (Instagram / Facebook / Threads). Skipped entirely
        for Chatbot events and implicit_negative stubs. Eligibility requires
        at least one hashtag mapping into `HASHTAG_TO_AD_CATEGORY`.
        """
        if not self.user_profile:
            return
        client = self.llm_client_mini or self.llm_client
        if client is None:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                      f"Skipping ad injection (no llm client).{utils.Colors.ENDC}")
            return
        if not self._action_by_oid:
            return  # Step 13b didn't run — nothing to inject ads into

        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Deterministic RNG per user — shares the same namespace as Step 13b
        # but offset so ad selection is decoupled from content-type sampling.
        try:
            _ad_seed = int(str(self.user_id)) * 8831 + 419
        except (ValueError, TypeError):
            _ad_seed = abs(hash(str(self.user_id))) % (2**31)
        rng = random.Random(_ad_seed)

        # Build the eligibility roster — one pass over existing events.
        from collections import defaultdict as _dd
        atomics_by_oid: dict[str, list] = _dd(list)
        for ap in self.atomic_personas:
            atomics_by_oid[ap.source_object_id].append(ap)
        for ap in self.negative_personas:
            atomics_by_oid[ap.source_object_id].append(ap)

        def _event_ad_category(oid: str, atoms: list) -> str:
            """Return an ad_category if any hashtag on the event maps to one,
            else empty string. Prefers the first-matching lowercased token."""
            tags: list[str] = []
            for ap in atoms:
                tags.extend(ap.source_hashtags or [])
            for tag in tags:
                t = tag.lstrip("#").lower()
                if t in HASHTAG_TO_AD_CATEGORY:
                    return HASHTAG_TO_AD_CATEGORY[t]
            return ""

        eligible: list[tuple[str, str]] = []  # (oid, ad_category)
        for oid, meta in self._action_by_oid.items():
            app = self._row_app.get(oid) or ""
            if app not in ("Instagram", "Facebook", "Threads"):
                continue
            if meta.get("itype") == "implicit_negative":
                continue
            atoms = atomics_by_oid.get(oid, [])
            if not atoms:
                continue
            cat = _event_ad_category(oid, atoms)
            if not cat:
                continue
            eligible.append((oid, cat))

        if not eligible:
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] "
                      f"Step 13c: no ad-eligible events (no commerce hashtags).{utils.Colors.ENDC}")
            return

        n_target = max(1, int(round(len(eligible) * AD_INJECTION_RATE)))
        rng.shuffle(eligible)
        selected = eligible[:n_target]

        # Pick polarities up front so content-gen prompts know whether to
        # frame the copy as click-worthy (clicked_ad) or dismissable
        # (hidden/dismissed).
        polarity_keys = list(AD_POLARITY_WEIGHTS.keys())
        polarity_weights = [AD_POLARITY_WEIGHTS[k] for k in polarity_keys]
        ad_plan: list[dict] = []
        for oid, ad_category in selected:
            action = rng.choices(polarity_keys, weights=polarity_weights, k=1)[0]
            # Map action → canonical label via the catalog (keeps copy in sync)
            app = self._row_app.get(oid, "Instagram")
            action_label = self._ad_label_for_action(app, action)
            itype = "explicit_positive" if action == "clicked_ad" else "explicit_negative"
            atoms = atomics_by_oid.get(oid, [])
            tags: list[str] = []
            for ap in atoms:
                tags.extend(ap.source_hashtags or [])
            hashtags = list(dict.fromkeys(tags))
            # Keep the content_type from the pre-existing content_by_oid
            # entry when available; default to image otherwise (ads are
            # overwhelmingly visual on social apps).
            existing_content = self._content_by_oid.get(oid, {})
            content_type = existing_content.get("content_type") or "image"
            ad_plan.append({
                "oid": oid,
                "app": app,
                "action": action,
                "action_label": action_label,
                "itype": itype,
                "ad_category": ad_category,
                "content_type": content_type,
                "hashtags": hashtags,
            })

        # Profile context for ad prompt (so the synthesized copy is tailored
        # to the persona's demographic / career / style without betraying
        # specific preferences).
        user_profile_dict = {
            "name": self.user_profile.name,
            "gender": self.user_profile.gender,
            "race_ethnicity": self.user_profile.race_ethnicity,
            "career": self.user_profile.career,
            "education": self.user_profile.education,
            "bio": self.user_profile.bio,
        }

        def _gen_ad(ev: dict):
            try:
                prompt = prompts.synthesize_ad_content_prompt(
                    content_type=ev["content_type"],
                    app=ev["app"],
                    ad_category=ev["ad_category"],
                    action=ev["action"],
                    action_label=ev["action_label"],
                    hashtags=ev["hashtags"],
                    user_profile=user_profile_dict,
                )
                response = self._query_mini_with_retry(prompt)
                content = None
                if response:
                    parsed = utils.extract_json_from_response(response)
                    if isinstance(parsed, dict):
                        if "content" in parsed and isinstance(parsed["content"], dict):
                            content = parsed["content"]
                        elif set(parsed.keys()) >= {"text"} or set(parsed.keys()) >= {"caption"}:
                            content = parsed
                return ev["oid"], ev, content
            except Exception:
                return ev["oid"], ev, None

        pbar = tqdm(total=len(ad_plan),
                    desc=f"[User {self.user_id}] Step 13c: Ad content",
                    unit="ad", disable=not self.verbose)

        n_success = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_gen_ad, ev): ev for ev in ad_plan}
            for future in as_completed(futures):
                pbar.update(1)
                try:
                    oid, ev, content = future.result()
                    if not content:
                        # LLM failed — skip this ad; the organic event stays.
                        continue
                    # Ensure ad_metadata is present; if the LLM dropped it,
                    # build a minimal fallback so the invariant holds.
                    if "ad_metadata" not in content or not isinstance(content.get("ad_metadata"), dict):
                        content["ad_metadata"] = {
                            "sponsor_name": "Sponsored Brand",
                            "ad_category": ev["ad_category"],
                            "cta_label": "Learn more",
                            "cta_destination_kind": "landing_page",
                            "disclosure_label": "Sponsored",
                        }
                    else:
                        # Normalize required fields.
                        md = content["ad_metadata"]
                        md.setdefault("sponsor_name", "Sponsored Brand")
                        md.setdefault("ad_category", ev["ad_category"])
                        md.setdefault("cta_label", "Learn more")
                        md.setdefault("cta_destination_kind", "landing_page")
                        md.setdefault("disclosure_label", "Sponsored")
                    # Override content + action + itype for this event.
                    self._content_by_oid[oid] = {
                        "content_type": ev["content_type"],
                        "content": content,
                    }
                    self._action_by_oid[oid] = {
                        "action": ev["action"],
                        "action_label": ev["action_label"],
                        "itype": ev["itype"],
                    }
                    self._ad_oids.add(oid)
                    n_success += 1
                except Exception as e:
                    print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                          f"Ad-gen future failed: {e}{utils.Colors.ENDC}")
        pbar.close()

        if self.verbose:
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] "
                  f"Ad injection: {n_success}/{len(ad_plan)} ads synthesized "
                  f"({n_success}/{len(eligible)} of eligible events; "
                  f"rate={AD_INJECTION_RATE:.0%}).{utils.Colors.ENDC}")

    def _ad_label_for_action(self, app: str, action: str) -> str:
        """Look up the canonical label for an ad action from the catalog.

        Falls back to a reasonable default if the app/action combination
        isn't present (shouldn't happen — every social app has all three
        ad actions).
        """
        bucket_name = "explicit_positive" if action == "clicked_ad" else "explicit_negative"
        app_cat = PLATFORM_INTERACTION_FORMATS.get(app, {})
        for entry in app_cat.get(bucket_name, []):
            if entry.get("action") == action:
                return entry.get("label", action)
        return action.replace("_", " ").title()

    # ------------------------------------------------------------------
    # Test split — LLM-gated, with LLM-picked distractors
    # ------------------------------------------------------------------

    def build_test_split(
        self,
        fraction: float = 0.2,
        min_test_items: int = 10,
        shortlist_size: int = 15,
        n_distractors: int = 3,
    ) -> None:
        """Pick newest-first high-confidence positives as held-out test items.

        There is no "train" concept in the output — everything that isn't
        test is just interaction history in the app JSONs. split_labels only
        ever holds "test" for items; items that aren't test have no label.

        Test selection rules:
          1. n_test_target = max(min_test_items, int(total_high_conf * fraction)),
             capped at the size of the high-confidence pool.
          2. Walk newest → oldest through high-confidence positives. Batch
             them into gate calls of size n_test_target; the LLM marks each
             as inferrable or not from the remaining (older) pool.
          3. Inferrable items become "test" in strict newest-first order
             until we hit n_test_target or the high-confidence pool runs out.
             Non-inferrable items are NOT deleted — they stay in
             cross_referenced_personas as interaction history.
          4. Distractor pairing per test item:
               Stage A (Python): randomly shortlist `shortlist_size` non-test
                 high-confidence items, causally filtered (first_occurrence
                 <= test item's last_occurrence).
               Stage B (LLM): picks the top `n_distractors` most topically
                 irrelevant items.

        Implicit/explicit negatives live in cross_referenced_negatives, not
        cross_referenced_personas, so they never enter the test candidate
        pool. An explicit guard below also skips any "negative" source type.
        """
        self.split_labels = {}
        self.test_distractors = {}

        if not self.cross_referenced_personas:
            return

        # Build first/last-occurrence lookups per canonical (by normalized key).
        # Canonicals are the rows tracked in self._canonical_groups; a
        # canonical's first_ts is the earliest supporting atom's timestamp,
        # and last_ts is the latest.
        first_ts_by_canon: dict[str, int] = {}
        last_ts_by_canon: dict[str, int] = {}
        for cr in self.cross_referenced_personas:
            key = _normalize_persona_text(cr.persona_item)
            atoms = self._canonical_groups.get(key, [])
            tss = [a.source_timestamp for a in atoms if a.source_timestamp]
            if tss:
                first_ts_by_canon[cr.persona_item] = min(tss)
                last_ts_by_canon[cr.persona_item] = max(tss)

        def _last_ts(cr) -> int:
            return last_ts_by_canon.get(cr.persona_item, 0)

        def _first_ts(cr) -> int:
            return first_ts_by_canon.get(cr.persona_item, 0)

        # High-confidence positive pool, strictly newest → oldest.
        high_conf_pool: list[CrossReferencedPersona] = [
            cr for cr in sorted(self.cross_referenced_personas, key=_last_ts, reverse=True)
            if "negative" not in cr.source_interaction_type
            and is_high_confidence(
                cr.confidence_score_init,
                cr.confidence_cross_referenced,
                getattr(cr, "n_explicit_rows", 0),
                getattr(cr, "n_implicit_rows", 0),
            )
        ]

        if not high_conf_pool:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] No high-confidence positives — "
                      f"no test items this run.{utils.Colors.ENDC}")
            return

        n_test_target = min(
            len(high_conf_pool),
            max(min_test_items, int(len(high_conf_pool) * fraction)),
        )

        # --- LLM inferrability gate, newest-first in batches ---
        # Take n_test_target items at a time from the newest end; for each
        # batch, the gate runs against "everything NOT currently in the
        # batch" as the training reference. Inferrable items land in
        # `kept_test` in strict newest-first order until target reached.
        # Non-inferrable items stay in cross_referenced_personas as
        # interaction history — they're never deleted.
        kept_test: list[CrossReferencedPersona] = []
        non_inferrable_total = 0

        def _run_gate(batch: list[CrossReferencedPersona]) -> tuple[set[str], set[str]]:
            """Return (inferrable_names, non_inferrable_names) for `batch`.
            Reference pool = all cross_referenced_personas minus the batch."""
            if not batch:
                return set(), set()
            batch_names = {c.persona_item for c in batch}
            reference_pool = [
                cr for cr in self.cross_referenced_personas
                if cr.persona_item not in batch_names
            ]
            if self.llm_client is None:
                # Claude Code subagent mode — no gate, keep all.
                return batch_names, set()
            ref_prompt = [
                {
                    "persona_item": cr.persona_item, "category": cr.category,
                    "confidence_score_init": cr.confidence_score_init,
                    "confidence_cross_referenced": cr.confidence_cross_referenced,
                    "formatted_timestamp": cr.formatted_timestamp,
                } for cr in reference_pool
            ]
            batch_prompt = [
                {
                    "persona_item": cr.persona_item, "category": cr.category,
                    "confidence_score_init": cr.confidence_score_init,
                    "confidence_cross_referenced": cr.confidence_cross_referenced,
                    "formatted_timestamp": cr.formatted_timestamp,
                } for cr in batch
            ]
            prompt = prompts.test_inferrability_check_prompt(
                train_personas=ref_prompt, test_candidates=batch_prompt,
            )
            response = self._query_llm_with_retry(prompt)
            if not response:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] Inferrability check LLM call failed — "
                      f"keeping this batch as inferrable by default.{utils.Colors.ENDC}")
                return batch_names, set()
            parsed = utils.extract_json_from_response(response)
            if not isinstance(parsed, list):
                print(f"{utils.Colors.WARNING}[User {self.user_id}] Unparseable test inferrability response — "
                      f"keeping this batch as inferrable by default.{utils.Colors.ENDC}")
                return batch_names, set()
            inferrable, non_inferrable = set(), set()
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                name = item.get("persona_item", "")
                if not name or name not in batch_names:
                    continue
                if bool(item.get("inferrable", False)):
                    inferrable.add(name)
                else:
                    non_inferrable.add(name)
            # Anything the LLM didn't return on — treat as inferrable.
            missing = batch_names - inferrable - non_inferrable
            inferrable |= missing
            return inferrable, non_inferrable

        # Walk the pool newest-first in batches of n_test_target.
        pool_idx = 0
        batch_size = max(n_test_target, 1)
        while len(kept_test) < n_test_target and pool_idx < len(high_conf_pool):
            batch = high_conf_pool[pool_idx : pool_idx + batch_size]
            pool_idx += len(batch)
            inferrable_names, non_inferrable_names = _run_gate(batch)
            non_inferrable_total += len(non_inferrable_names)
            # Preserve newest-first order within the batch.
            for cr in batch:
                if cr.persona_item in inferrable_names:
                    kept_test.append(cr)
                    if len(kept_test) >= n_test_target:
                        break

        # Sort test items chronologically (oldest-first) for downstream consumers.
        kept_test.sort(key=_last_ts)

        # --- Distractor pairing ---
        # Distractor pool = all high-confidence positives that aren't test.
        # Non-inferrable items remain in cross_referenced_personas and are
        # eligible as distractor shortlist members.
        test_items_set = {cr.persona_item for cr in kept_test}
        high_conf_train = [
            cr for cr in self.cross_referenced_personas
            if cr.persona_item not in test_items_set
            and is_high_confidence(
                cr.confidence_score_init,
                cr.confidence_cross_referenced,
                getattr(cr, "n_explicit_rows", 0),
                getattr(cr, "n_implicit_rows", 0),
            )
        ]

        for test_cr in tqdm(kept_test,
                            desc=f"[User {self.user_id}] Step 15: Distractor pairing",
                            disable=not self.verbose):
            self.split_labels[test_cr.persona_item] = "test"

            # Causality: only train items whose first occurrence is at or
            # before this test item's last-occurrence timestamp are eligible
            # as distractors.
            test_cutoff = _last_ts(test_cr)
            causal_train = [cr for cr in high_conf_train if _first_ts(cr) <= test_cutoff]
            if not causal_train:
                if self.verbose:
                    print(f"{utils.Colors.WARNING}[User {self.user_id}] No causally-eligible "
                          f"high-confidence train items for test item "
                          f"'{test_cr.persona_item}' — no distractors assigned.{utils.Colors.ENDC}")
                continue

            n_to_sample = min(shortlist_size, len(causal_train))
            shortlist = random.sample(causal_train, n_to_sample)
            shortlist_for_prompt = [
                {"persona_item": cr.persona_item, "category": cr.category}
                for cr in shortlist
            ]

            # Rank distractors: LLM returns a list of up to n_distractors,
            # Python falls back to the first N from the shortlist on failure.
            n_picks = min(n_distractors, len(shortlist))
            chosen_names: list[str] = []
            if self.llm_client is not None and shortlist_for_prompt:
                prompt = prompts.distractor_selection_prompt(
                    test_persona={"persona_item": test_cr.persona_item, "category": test_cr.category},
                    candidate_distractors=shortlist_for_prompt,
                    n_picks=n_picks,
                )
                response = self._query_llm_with_retry(prompt)
                if response:
                    parsed = utils.extract_json_from_response(response)
                    if isinstance(parsed, list):
                        valid = {cr.persona_item: cr for cr in shortlist}
                        for item in parsed:
                            if not isinstance(item, dict):
                                continue
                            name = item.get("persona_item", "")
                            if name in valid and name not in chosen_names:
                                chosen_names.append(name)
                            if len(chosen_names) >= n_picks:
                                break

            # Top up with shortlist ordering if LLM underdelivered
            for cr in shortlist:
                if len(chosen_names) >= n_picks:
                    break
                if cr.persona_item not in chosen_names:
                    chosen_names.append(cr.persona_item)

            valid_lookup = {cr.persona_item: cr for cr in shortlist}
            self.test_distractors[test_cr.persona_item] = [
                {
                    "persona_item": valid_lookup[name].persona_item,
                    "category": valid_lookup[name].category,
                }
                for name in chosen_names
            ]

        # Non-test items get no split label — they're just interaction history.
        # split_labels only carries "test".

        if self.verbose:
            n_test = sum(1 for v in self.split_labels.values() if v == "test")
            n_history = len(self.cross_referenced_personas) - n_test
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Test split: "
                  f"{n_test} test, {n_history} in interaction history, "
                  f"{non_inferrable_total} non-inferrable (kept as history), "
                  f"{sum(1 for v in self.test_distractors.values())} distractors assigned."
                  f"{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_pipeline(self) -> dict:
        """Run the full persona inference pipeline.

        Order:
          1.  infer atomic personas
          2.  dedupe (lexical) + init filter + count corroboration → cross_ref
          3.  cross-reference & filter
          3.5 classify horizons + stop conditions (LLM refinement of short-term)
          4.  temporal contradiction graph
          5.  build update histories
          5b. resolve cross-polarity contradictions (Step 3b — enforces temporal precedent for stance flips)
          6.  generate user profile (demographics + big_five + bio)
          7.  infer hidden personas (cross-row hashtag clustering)
          8.  generate per-app sub-personas
          9.  build sessions
         10.  route preferences to apps (LLM + 8% noise)
         11.  assign rows to apps (session majority vote)
         12.  generate interaction formats (weighted catalog sampling)
         13.  generate chatbot conversations (multi-turn, implicit embedding)
         13b. generate synthetic per-event content (text/image/short_video)
         13c. inject ad events (convert ~6% of commerce-adjacent events to ads)
         14.  annotate stereotype marks
         15.  build test split (cross-app, latest-20% high-conf by time)
         16.  save to backend/{uid}/ subfolder
        """
        print(f"{utils.Colors.BOLD}[User {self.user_id}] Starting persona pipeline...{utils.Colors.ENDC}")
        pipeline_start = time.time()

        steps = [
            ("1.  Infer atomic personas",          self.infer_personas_from_hashtags),
            ("2.  Promote implicit negatives",      self.promote_implicit_negatives),
            ("3.  Cross-reference & filter",        self.summarize_and_cross_reference),
            ("3.5 Classify horizons + stops",       self.classify_horizons_and_stop_conditions),
            ("4.  Temporal contradiction graph",     self.build_temporal_contradiction_graph),
            ("5.  Build update histories",           self.build_update_histories),
            ("5b. Resolve cross-polarity contradictions", self.resolve_cross_polarity_contradictions),
            ("6.  Generate user profile",            self.generate_user_profile),
            ("7.  Infer hidden personas",            self.infer_hidden_personas),
            ("7b. Infer MBTI",                       self.infer_mbti),
            ("8.  Generate app personas",            self.generate_app_personas),
            ("9.  Build sessions",                   self._build_sessions),
            ("10. Route preferences to apps",        self.route_personas_to_apps),
            ("11. Assign rows to apps",              self._assign_rows_to_apps),
            ("12. Generate interaction formats",     self.generate_interaction_formats),
            ("13. Generate chatbot conversations",   self.generate_chatbot_conversations),
            ("13b. Generate synthetic content",      self.generate_synthetic_content),
            ("13c. Inject ad events",                self.inject_ad_events),
            ("14. Annotate stereotype marks",        self.annotate_stereotype_marks),
            ("15. Build test split",                 self.build_test_split),
            ("16. Save to backend",                  self.save_to_backend),
        ]

        for step_name, step_fn in steps:
            step_start = time.time()
            step_fn()
            elapsed = time.time() - step_start
            total_elapsed = time.time() - pipeline_start
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] {step_name}: "
                  f"{elapsed:.1f}s (total: {total_elapsed:.1f}s){utils.Colors.ENDC}")

        n_test = sum(1 for v in self.split_labels.values() if v == "test")
        n_history = len(self.cross_referenced_personas) - n_test

        summary = {
            "user_id": self.user_id,
            "total_interactions": len(self.interactions),
            "total_atomic_personas": len(self.atomic_personas),
            "total_negative_personas": len(self.negative_personas),
            "total_cross_referenced": len(self.cross_referenced_personas),
            "total_cross_referenced_negatives": len(self.cross_referenced_negatives),
            "total_sessions": len(self._sessions),
            "total_contradictions": sum(
                1 for p in self.cross_referenced_personas if p.relationship_type == "contradictory"
            ),
            "temporal_topics": len(self.temporal_graph),
            "profile_generated": self.user_profile is not None,
            "app_personas_generated": (
                len(self.user_profile.app_personas) if self.user_profile else 0
            ),
            "annotated_personas": len(self.annotated_personas),
            "split_test": n_test,
            "interaction_history_preferences": n_history,
            "distractors_assigned": len(self.test_distractors),
            "total_time_seconds": round(time.time() - pipeline_start, 1),
        }
        total_time = time.time() - pipeline_start
        mins, secs = divmod(total_time, 60)
        print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Pipeline complete in {int(mins)}m {secs:.0f}s: {summary}{utils.Colors.ENDC}")
        return summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _user_dir(self) -> str:
        """Directory for this user's output files."""
        return os.path.join(self.backend_dir, str(self.user_id))

    def save_to_backend(self) -> str:
        """Persist data to backend/{user_id}/:

          - profile.json    — UserProfile + AppPersonas + flat unique list of
                              persona_item strings under "preferences"
          - instagram.json  — interaction events routed to Instagram (time-sorted)
          - facebook.json   — interaction events routed to Facebook (time-sorted)
          - threads.json    — interaction events routed to Threads (time-sorted)
          - chatbot.json    — interaction events routed to Chatbot (time-sorted)

        Each app JSON is a list of **interaction events**. Each event represents
        one piece of content the user engaged with (one source CSV row) and
        contains a nested ``preferences`` list of the surviving inferred
        preferences for that engagement. Events with zero surviving preferences
        are dropped. The same canonical preference naturally appears across
        multiple events (preserving real-world repetition).
        """
        user_dir = self._user_dir()
        os.makedirs(user_dir, exist_ok=True)

        # --- Build lookups ---
        all_annotated_items = {ap.persona_item: ap for ap in self.annotated_personas}

        # Backward lookup: which hidden persona cluster(s) did each
        # source_object_id contribute to? Each hidden persona's
        # `evidence_oids` is the sorted list of oids whose hashtags placed
        # the row inside the cluster during Step 7. A preference's source
        # row being in the cluster IS the motivation trace — we attach the
        # cluster label back onto that preference. When a single oid feeds
        # multiple clusters, the one with the most evidence_rows wins.
        from collections import defaultdict as _ddict_oid
        _oid_to_hp: dict[str, list[tuple[str, int]]] = _ddict_oid(list)
        if self.user_profile and self.user_profile.hidden_personas:
            for hp in self.user_profile.hidden_personas:
                for oid in hp.evidence_oids:
                    _oid_to_hp[oid].append((hp.label, hp.evidence_rows))

        # Canonical metadata lookup (positive + negative)
        # Also map absorbed members' atom texts to the representative canonical
        canonical_lookup: dict[str, CrossReferencedPersona] = {}
        for cr in self.cross_referenced_personas:
            cr_key = _normalize_persona_text(cr.persona_item)
            canonical_lookup[cr_key] = cr
            for ap in self._canonical_groups.get(cr_key, []):
                ap_key = _normalize_persona_text(ap.persona_item)
                if ap_key not in canonical_lookup:
                    canonical_lookup[ap_key] = cr
        for cr in self.cross_referenced_negatives:
            cr_key = _normalize_persona_text(cr.persona_item)
            canonical_lookup[cr_key] = cr
            for ap in self._negative_canonical_groups.get(cr_key, []):
                ap_key = _normalize_persona_text(ap.persona_item)
                if ap_key not in canonical_lookup:
                    canonical_lookup[ap_key] = cr

        def _parse_format(raw: str, fallback_app: str) -> dict:
            """Parse a stored interaction_format (JSON str or legacy plain str)
            into the canonical {app, action, action_label, user_message} dict."""
            if not raw:
                return {"app": fallback_app, "action": "unknown", "action_label": "Unknown", "user_message": None}
            if isinstance(raw, dict):
                return {
                    "app": raw.get("app", fallback_app),
                    "action": raw.get("action", "unknown"),
                    "action_label": raw.get("action_label", ""),
                    "user_message": raw.get("user_message"),
                }
            if isinstance(raw, str):
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        return {
                            "app": obj.get("app", fallback_app),
                            "action": obj.get("action", "unknown"),
                            "action_label": obj.get("action_label", ""),
                            "user_message": obj.get("user_message"),
                        }
                except (ValueError, TypeError):
                    pass
                return {"app": fallback_app, "action": "legacy", "action_label": raw, "user_message": None}
            return {"app": fallback_app, "action": "unknown", "action_label": "Unknown", "user_message": None}

        # --- Group ALL atomics by source_object_id (= one interaction event each) ---
        from collections import defaultdict as _ddict
        atomics_by_oid: dict[str, list] = _ddict(list)
        for ap in self.atomic_personas:
            atomics_by_oid[ap.source_object_id].append(ap)
        for ap in self.negative_personas:
            atomics_by_oid[ap.source_object_id].append(ap)

        # --- Compute latest-20% timestamp cutoff for test labeling ---
        # Test labels only apply to events in the latest 20% of the timeline.
        # Earlier occurrences of the same canonical get no split label.
        all_timestamps = sorted(set(
            ap.source_timestamp for ap in self.atomic_personas if ap.source_timestamp
        ))
        if all_timestamps:
            cutoff_idx = int(len(all_timestamps) * 0.8)
            test_ts_cutoff = all_timestamps[min(cutoff_idx, len(all_timestamps) - 1)]
        else:
            test_ts_cutoff = 0

        # --- Build interaction events ---
        all_events: list[dict] = []
        seen_unique_prefs: list[str] = []  # for profile.json dedup
        # Seeded RNG for per-event action sampling (deterministic per user)
        try:
            _ev_seed = int(str(self.user_id)) * 4219 + 37
        except (ValueError, TypeError):
            _ev_seed = abs(hash(str(self.user_id))) % (2**31)
        event_rng = random.Random(_ev_seed)

        for oid, atoms in tqdm(atomics_by_oid.items(),
                               desc=f"[User {self.user_id}] Step 16: Building events",
                               total=len(atomics_by_oid),
                               disable=not self.verbose):
            if not atoms:
                continue
            # Representative atom for event-level metadata
            rep = atoms[0]
            is_negative_event = "negative" in rep.source_interaction_type

            # Determine app for this event: use session-based row assignment
            app = self._row_app.get(oid, "")
            if not app:
                app = random.choice(PLATFORMS)

            # Build nested preferences list — only surviving canonicals
            preferences: list[dict] = []
            for ap in atoms:
                key = _normalize_persona_text(ap.persona_item)
                cr = canonical_lookup.get(key)
                if not cr:
                    continue  # this atomic's canonical was filtered out

                ann = all_annotated_items.get(cr.persona_item)
                # Only label as "test" if this event is in the latest 20%
                raw_split = self.split_labels.get(cr.persona_item, "")
                split_label = "test" if raw_split == "test" and ap.source_timestamp >= test_ts_cutoff else ""
                distractor_list = (
                    self.test_distractors.get(cr.persona_item, []) if split_label == "test" else []
                )
                # Tolerate the legacy single-dict shape from older cached
                # state so we don't need a migration sweep.
                if isinstance(distractor_list, dict):
                    distractor_list = [distractor_list]

                # Build merged update_history: temporal entries (no raw timestamp)
                # + related_personas folded in as similar/contradictory entries.
                # Causality: only keep entries whose timestamp <= this event's time.
                # Key order: update_type, preference, formatted_timestamp, then extras.
                _HISTORY_KEY_ORDER = ["update_type", "preference", "formatted_timestamp",
                                      "source_app", "occurrence", "total_occurrences", "description",
                                      # Cross-polarity contradiction metadata (Step 3b)
                                      "resolution", "opposing_polarity",
                                      "prior_corroboration_count", "required_precedent"]
                event_ts = ap.source_timestamp  # this atomic's source row timestamp

                merged_history = []
                for h in (cr.update_history or []):
                    raw = dict(h)
                    h_ts = raw.pop("timestamp", 0)
                    if h_ts and h_ts > event_ts:
                        continue  # skip future entries — causality
                    if raw.get("update_type") == "new":
                        continue  # redundant — event timestamp already shows first appearance
                    # Drop self-referencing preference (same as parent persona_item)
                    if raw.get("preference") == cr.persona_item:
                        raw.pop("preference")
                    # Resolve source_object_id → source_app for reinforced entries
                    hist_oid = raw.pop("source_object_id", None)
                    if hist_oid:
                        raw["source_app"] = self._row_app.get(hist_oid, "")
                    # For entries with a preference (evolution/contradicted), use target canonical's app
                    elif raw.get("preference"):
                        pref_cr = canonical_lookup.get(_normalize_persona_text(raw["preference"]))
                        if pref_cr:
                            raw["source_app"] = pref_cr.assigned_app
                    ordered = {k: raw[k] for k in _HISTORY_KEY_ORDER if k in raw}
                    merged_history.append(ordered)
                for rel in (cr.related_personas or []):
                    if not (isinstance(rel, dict) and rel.get("persona_item")):
                        continue
                    rel_key = _normalize_persona_text(rel["persona_item"])
                    rel_cr = canonical_lookup.get(rel_key)
                    # Recover the related preference's first-occurrence timestamp.
                    rel_atoms = self._canonical_groups.get(rel_key, [])
                    if not rel_atoms:
                        rel_atoms = self._negative_canonical_groups.get(rel_key, [])
                    rel_first_ts = (
                        min((a.source_timestamp for a in rel_atoms if a.source_timestamp), default=0)
                        if rel_atoms else 0
                    )
                    # Strict causality: drop any related entry we cannot place
                    # in time, or whose first occurrence is after this event.
                    if not rel_first_ts or rel_first_ts > event_ts:
                        continue
                    # Normalize "contradictory" → "contradicted" for consistent naming
                    rel_type = rel.get("type", "similar")
                    if rel_type == "contradictory":
                        rel_type = "contradicted"
                    merged_history.append({
                        "update_type": rel_type,
                        "preference": rel["persona_item"],
                        "formatted_timestamp": utils.unix_to_formatted(rel_first_ts),
                        "source_app": rel_cr.assigned_app if rel_cr else "",
                    })

                # Backward-link from hidden personas' evidence_oids: label a
                # preference with at most ONE hidden persona — the cluster
                # whose evidence includes this preference's source row.
                # Ties (a row in multiple clusters) resolve to the cluster
                # with the most evidence_rows. Rows that didn't contribute
                # to any cluster stay unlabeled — traceability is required.
                hp_labels: list[str] = []
                matches = _oid_to_hp.get(ap.source_object_id, [])
                if matches:
                    hp_labels = [max(matches, key=lambda m: m[1])[0]]

                pref = {
                    "persona_item": cr.persona_item,
                    "category": cr.category,
                    "confidence_score_init": ap.confidence_score_init,
                    "confidence_cross_referenced": cr.confidence_cross_referenced,
                    "stereotype_mark": ann.stereotype_mark if ann else "neutral",
                    "hidden_persona_labels": hp_labels,
                    "update_history": merged_history,
                    "time_horizon": getattr(cr, "time_horizon", "long_term"),
                }
                # Stop condition only meaningful for short-term canonicals
                if getattr(cr, "time_horizon", "long_term") == "short_term":
                    sc = getattr(cr, "stop_condition", {}) or {}
                    if sc:
                        pref["stop_condition"] = sc
                if split_label == "test":
                    pref["split"] = "test"
                    # A list of distractor dicts, each {persona_item, category}.
                    pref["over_personalization_irrelevant"] = [
                        {
                            "persona_item": d.get("persona_item", ""),
                            "category": d.get("category", ""),
                        }
                        for d in distractor_list
                    ]

                preferences.append(pref)

                # Track unique preferences for profile.json
                if cr.persona_item not in seen_unique_prefs:
                    seen_unique_prefs.append(cr.persona_item)

            # Skip events with zero surviving preferences — UNLESS
            # it's implicit_negative (keep those as empty-preference markers)
            if not preferences and rep.source_interaction_type != "implicit_negative":
                continue

            # Collect all unique hashtags from this event's atoms
            event_hashtags = rep.source_hashtags
            if not event_hashtags:
                all_tags: list[str] = []
                for ap in atoms:
                    all_tags.extend(ap.source_hashtags)
                event_hashtags = list(dict.fromkeys(all_tags))

            # Determine effective interaction type for this event.
            # Step 13b (if it ran) pre-samples action + itype and stores them
            # in self._action_by_oid so the content_type it chose matches
            # the action ultimately shown. Fall back to legacy inline sampling
            # when that step wasn't run (e.g. no LLM client configured).
            pre_meta = self._action_by_oid.get(oid)
            if pre_meta:
                itype = pre_meta["itype"]
                sampled_entry = {
                    "action": pre_meta["action"],
                    "label": pre_meta["action_label"],
                }
            else:
                # Legacy path: promote implicit_negative → explicit_negative when
                # the event carries surviving preferences (≥5 repetition gate).
                itype = rep.source_interaction_type or "implicit_positive"
                if itype == "implicit_negative" and preferences:
                    itype = "explicit_negative"
                # For Chatbot: reassign 20% explicit / 80% implicit (polarity kept).
                if app == "Chatbot" and itype != "implicit_negative":
                    polarity = "negative" if "negative" in itype else "positive"
                    itype = f"explicit_{polarity}" if event_rng.random() < 0.20 else f"implicit_{polarity}"
                sampled_entry = self._sample_action_from_bucket(app, itype, event_rng)

            fmt = {
                "app": app,
                "action": sampled_entry["action"],
                "action_label": sampled_entry["label"],
                "user_message": None,
            }
            # Carry over a stored user_message ONLY when the re-sampled action
            # is one that semantically carries a natural-language message
            # (social-media @ai comments or Chatbot chat turns). Step 12
            # generates user_messages keyed to the canonical's originally
            # sampled action; Step 16 may re-sample a different action for
            # this event (e.g. `viewed_video_75`), in which case the old
            # message must NOT leak onto a passive-view action.
            if sampled_entry["action"] in (AT_AI_ACTIONS | CHATBOT_TURN_ACTIONS):
                first_cr = canonical_lookup.get(_normalize_persona_text(atoms[0].persona_item))
                if first_cr and first_cr.source_interaction_format:
                    stored = _parse_format(first_cr.source_interaction_format, app)
                    if stored.get("user_message"):
                        fmt["user_message"] = stored["user_message"]

            # Build event dict with preferences LAST for readability
            event = {
                "source_object_id": oid,
                "source_timestamp": rep.source_timestamp,
                "formatted_timestamp": rep.formatted_timestamp,
                "source_hashtags": event_hashtags,
                "source_interaction_type": itype,
                "interaction_format": fmt,
            }

            # Ad marker (Step 13c). Invariant: is_ad ⇔ action ∈ AD_ACTIONS.
            if oid in self._ad_oids or sampled_entry["action"] in AD_ACTIONS:
                event["is_ad"] = True
                self._ad_oids.add(oid)  # keep set coherent if only action matched

            # Attach synthetic content (step 13b). Chatbot and stubs are never
            # in self._content_by_oid, so those events render unchanged.
            content_entry = self._content_by_oid.get(oid)
            if content_entry:
                event["content_type"] = content_entry["content_type"]
                event["content"] = content_entry["content"]

            # Merge chatbot conversation data (keyed by source_object_id)
            if app == "Chatbot" and oid in self._chatbot_conversations:
                conv_data = self._chatbot_conversations[oid]
                event["conversation_type"] = conv_data.get("conversation_type")
                event["conversation"] = conv_data.get("conversation")
                event["ask_to_forget"] = conv_data.get("ask_to_forget", False)
                override = conv_data.get("interaction_format_override")
                if override and isinstance(override, dict):
                    action = override.get("action")
                    if action in (
                        "asked_to_forget",
                        "asked_not_to_personalize",
                        "corrected_assumption",
                    ):
                        event["interaction_format"] = override

            event["preferences"] = preferences  # always last

            # Fallback: chatbot events that still lack a conversation (rare —
            # all LLM attempts failed for this event). Try one more direct call.
            if app == "Chatbot" and "conversation" not in event and preferences and self.llm_client:
                fallback_prefs = [{"persona_item": p["persona_item"],
                                   "category": p["category"],
                                   "interaction_type": itype} for p in preferences]
                chatbot_p = self.user_profile.app_personas.get("Chatbot", {})
                if isinstance(chatbot_p, AppPersona):
                    chatbot_p = {"style_description": chatbot_p.style_description,
                                 "chatbot_contexts": chatbot_p.chatbot_contexts}
                fb_prompt = prompts.generate_chatbot_conversation_prompt(
                    preferences=fallback_prefs,
                    conversation_type="knowledge_query",
                    conversation_type_description="User asks a factual or curiosity-driven question.",
                    user_profile={"name": self.user_profile.name, "gender": self.user_profile.gender,
                                  "career": self.user_profile.career, "bio": self.user_profile.bio},
                    chatbot_persona=chatbot_p if isinstance(chatbot_p, dict) else {},
                    interaction_type=itype,
                    num_turns=max(2, min(len(preferences) * 2, 6)),
                )
                for _attempt in range(3):
                    try:
                        resp = self.llm_client.query_llm(fb_prompt)
                        parsed = utils.extract_json_from_response(resp)
                        if isinstance(parsed, list) and len(parsed) >= 2:
                            valid = all(isinstance(t, dict) and t.get("role") in ("user", "assistant")
                                        and t.get("content") for t in parsed)
                            if valid:
                                event["conversation"] = parsed
                                event["conversation_type"] = "knowledge_query"
                                event["ask_to_forget"] = False
                                break
                    except Exception:
                        pass

            all_events.append(event)

        # --- Stub events for implicit_negative rows that produced no atomics ---
        # These rows were pre-filtered (hashtag signature below K) or the LLM
        # returned nothing. They still appear in the app JSONs as empty-preference
        # markers so the timeline is complete, rendered in greyscale in HTML.
        oids_with_atoms = set(atomics_by_oid.keys())
        n_stubs = 0
        for interaction in self.interactions:
            if interaction.interaction_type != "implicit_negative":
                continue
            if interaction.object_id in oids_with_atoms:
                continue  # already has an event from the main loop
            stub_app = self._row_app.get(interaction.object_id, "") or random.choice(PLATFORMS)
            stub_hashtags = self._extract_hashtags(interaction.object_text)
            stub_fmt_ts = self._format_timestamp(interaction.interaction_time)
            sampled_entry = self._sample_action_from_bucket(stub_app, "implicit_negative", event_rng)
            all_events.append({
                "source_object_id": interaction.object_id,
                "source_timestamp": interaction.interaction_time,
                "formatted_timestamp": stub_fmt_ts,
                "source_hashtags": stub_hashtags,
                "source_interaction_type": "implicit_negative",
                "interaction_format": {
                    "app": stub_app,
                    "action": sampled_entry["action"],
                    "action_label": sampled_entry["label"],
                    "user_message": None,
                },
                "preferences": [],
            })
            n_stubs += 1

        if self.verbose and n_stubs:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Added {n_stubs} implicit-negative stub events "
                  f"(no preferences, greyscale in HTML).{utils.Colors.ENDC}")

        # Assertion: no negative interaction event should have test-labeled preferences
        for event in all_events:
            if "negative" in event.get("source_interaction_type", ""):
                assert all(
                    p.get("split") != "test" for p in event["preferences"]
                ), f"BUG: negative interaction {event.get('source_object_id')} leaked into test split"

        # Sort strictly chronological
        all_events.sort(key=lambda e: (int(e.get("source_timestamp") or 0), e.get("source_object_id", "")))

        # Bucket by app (using the session-routed row app)
        per_app: dict[str, list[dict]] = {a: [] for a in PLATFORMS}
        for event in all_events:
            app = self._row_app.get(event.get("source_object_id", ""), "")
            if not app:
                app = PLATFORMS[0]
            if app not in per_app:
                per_app[app] = []
            per_app[app].append(event)

        # --- Write per-app JSONs ---
        for app_name, events in per_app.items():
            filename = app_name.lower() + ".json"
            path = os.path.join(user_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(events, f, indent=2, ensure_ascii=False)

        # --- Retroactively fill app_distribution on hidden personas ---
        if self.user_profile and self.user_profile.hidden_personas and self._row_app:
            from collections import Counter as _Counter
            for hp in self.user_profile.hidden_personas:
                tag_set_lower = set(t.lower() for t in hp.evidence_hashtags)
                app_counts: _Counter = _Counter()
                for row in self.interactions:
                    tags_in_row = self._extract_hashtags(row.object_text)
                    if set(t.lower() for t in tags_in_row) & tag_set_lower:
                        app = self._row_app.get(row.object_id, "")
                        if app:
                            app_counts[app] += 1
                hp.app_distribution = dict(app_counts)

        # --- Write profile.json (flat unique preference list + hidden personas) ---
        if self.user_profile:
            profile_dict = asdict(self.user_profile)
            profile_dict["user_id"] = str(self.user_id)
            # evidence_oids is an internal lookup index used at Step 16 to
            # back-link preferences to their cluster — it's not meant to be
            # consumed downstream. Strip it from each hidden_persona to keep
            # profile.json small and readable.
            for hp in profile_dict.get("hidden_personas", []) or []:
                hp.pop("evidence_oids", None)
            # Each preference is prefixed with its LATEST occurrence timestamp
            # (across all supporting atoms — positive or negative canonical
            # groups). Format: "YYYY-MM-DD HH:MM : <persona_item>".
            # Sorted by latest timestamp descending so recent preferences surface first.
            prefs_with_ts: list[tuple[int, str]] = []
            for pi in seen_unique_prefs:
                key = _normalize_persona_text(pi)
                atoms = (
                    self._canonical_groups.get(key, [])
                    or self._negative_canonical_groups.get(key, [])
                )
                tss = [a.source_timestamp for a in atoms if a.source_timestamp]
                latest_ts = max(tss) if tss else 0
                ts_str = utils.unix_to_formatted(latest_ts) if latest_ts else ""
                prefs_with_ts.append((latest_ts, f"{ts_str} : {pi}" if ts_str else pi))
            prefs_with_ts.sort(key=lambda x: x[0], reverse=True)
            profile_dict["preferences"] = [p for _, p in prefs_with_ts]
            profile_path = os.path.join(user_dir, "profile.json")
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_dict, f, indent=2, ensure_ascii=False)

        if self.verbose:
            total_events = len(all_events)
            total_prefs = sum(len(e["preferences"]) for e in all_events)
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Saved to {user_dir}/ "
                  f"({total_events} events, {total_prefs} preference instances, per-app: "
                  f"{ {k: len(v) for k, v in per_app.items()} }){utils.Colors.ENDC}")
        return user_dir

    def load_from_backend(self) -> bool:
        """Load persisted JSON data back into instance variables.

        Reads backend/{uid}/profile.json and backend/{uid}/{app}.json for each
        supported app. The app JSONs use the **interaction event** format:
        each entry has event-level fields and a nested ``preferences`` list.
        Also supports the legacy flat format (one record per preference) for
        backwards compatibility.

        Returns True if data was found, False otherwise.
        """
        user_dir = self._user_dir()
        if not os.path.isdir(user_dir):
            return False

        profile_path = os.path.join(user_dir, "profile.json")
        if not os.path.exists(profile_path):
            return False

        self.atomic_personas = []
        self.negative_personas = []
        self.cross_referenced_personas = []
        self.cross_referenced_negatives = []
        self.annotated_personas = []
        self.split_labels = {}
        self.test_distractors = {}

        # Track unique canonicals to avoid duplicates in cross_referenced lists
        seen_positive_canonicals: set[str] = set()
        seen_negative_canonicals: set[str] = set()

        # --- Load per-app JSONs ---
        all_events: list[dict] = []
        for app_name in PLATFORMS:
            app_path = os.path.join(user_dir, app_name.lower() + ".json")
            if not os.path.exists(app_path):
                continue
            with open(app_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                rec["_loaded_app"] = app_name
            all_events.extend(records)

        all_events.sort(key=lambda r: (int(r.get("source_timestamp") or 0), r.get("source_object_id", "")))

        for event in all_events:
            # Detect format: new (has "preferences" list) vs legacy (has "persona_item")
            if "preferences" in event:
                # --- New interaction event format ---
                oid = str(event.get("source_object_id", ""))
                ts = int(event.get("source_timestamp", 0))
                fmt_ts = event.get("formatted_timestamp", "")
                interaction_type = event.get("source_interaction_type", "")
                hashtags = list(event.get("source_hashtags", []))
                fmt_obj = event.get("interaction_format") or {}
                fmt_str = json.dumps(fmt_obj) if isinstance(fmt_obj, dict) else str(fmt_obj)
                app = event.get("_loaded_app", "")
                is_negative = "negative" in interaction_type

                for pref in event.get("preferences", []):
                    persona_item = pref.get("persona_item", "")
                    if not persona_item:
                        continue

                    ap = AtomicPersona(
                        persona_item=persona_item,
                        category=pref.get("category", "uncategorized"),
                        confidence_score_init=float(pref.get("confidence_score_init") or 0.0),
                        source_interaction_type=interaction_type,
                        source_interaction_format=fmt_str,
                        source_object_id=oid,
                        source_timestamp=ts,
                        formatted_timestamp=fmt_ts,
                        source_hashtags=hashtags,
                    )

                    if is_negative:
                        self.negative_personas.append(ap)
                        if persona_item not in seen_negative_canonicals:
                            seen_negative_canonicals.add(persona_item)
                            cr = CrossReferencedPersona(
                                persona_item=persona_item,
                                category=pref.get("category", "uncategorized"),
                                confidence_score_init=float(pref.get("confidence_score_init") or 0.0),
                                confidence_cross_referenced=float(pref.get("confidence_cross_referenced") or 0.0),
                                relationship_type=pref.get("relationship_type", "none"),
                                related_personas=list(pref.get("related_personas", [])),
                                formatted_timestamp=fmt_ts,
                                source_interaction_type=interaction_type,
                                source_interaction_format=fmt_str,
                                assigned_app=app,
                                time_horizon=pref.get("time_horizon", "long_term"),
                                stop_condition=dict(pref.get("stop_condition") or {}),
                            )
                            self.cross_referenced_negatives.append(cr)
                    else:
                        self.atomic_personas.append(ap)
                        if persona_item not in seen_positive_canonicals:
                            seen_positive_canonicals.add(persona_item)
                            cr = CrossReferencedPersona(
                                persona_item=persona_item,
                                category=pref.get("category", "uncategorized"),
                                confidence_score_init=float(pref.get("confidence_score_init") or 0.0),
                                confidence_cross_referenced=float(pref.get("confidence_cross_referenced") or 0.0),
                                relationship_type=pref.get("relationship_type", "none"),
                                related_personas=list(pref.get("related_personas", [])),
                                formatted_timestamp=fmt_ts,
                                source_interaction_type=interaction_type,
                                source_interaction_format=fmt_str,
                                assigned_app=app,
                                time_horizon=pref.get("time_horizon", "long_term"),
                                stop_condition=dict(pref.get("stop_condition") or {}),
                            )
                            self.cross_referenced_personas.append(cr)

                            ann = AnnotatedPersona(
                                persona_item=persona_item,
                                category=pref.get("category", "uncategorized"),
                                confidence_score_init=float(pref.get("confidence_score_init") or 0.0),
                                confidence_cross_referenced=float(pref.get("confidence_cross_referenced") or 0.0),
                                stereotype_mark=pref.get("stereotype_mark", "neutral"),
                            )
                            self.annotated_personas.append(ann)

                    split_label = pref.get("split", "")
                    self.split_labels[persona_item] = split_label
                    if split_label == "test":
                        raw_dis = pref.get("over_personalization_irrelevant", None)
                        # New shape: list of {persona_item, category}. Legacy:
                        # a single string + a separate *_category field.
                        if isinstance(raw_dis, list):
                            self.test_distractors[persona_item] = [
                                {
                                    "persona_item": d.get("persona_item", ""),
                                    "category": d.get("category", ""),
                                }
                                for d in raw_dis if isinstance(d, dict)
                            ]
                        elif isinstance(raw_dis, str) and raw_dis:
                            self.test_distractors[persona_item] = [
                                {
                                    "persona_item": raw_dis,
                                    "category": pref.get("over_personalization_irrelevant_category", "") or "",
                                }
                            ]

            elif "persona_item" in event:
                # --- Legacy flat format (backwards compatibility) ---
                rec = event
                interaction_type = rec.get("source_interaction_type", "")
                is_negative = "negative" in interaction_type
                fmt_obj = rec.get("interaction_format") or {}
                interaction_format_str = json.dumps(fmt_obj) if isinstance(fmt_obj, dict) else str(fmt_obj)

                ap = AtomicPersona(
                    persona_item=rec["persona_item"],
                    category=rec.get("category", "uncategorized"),
                    confidence_score_init=float(rec.get("confidence_score_init") or 0.0),
                    source_interaction_type=interaction_type,
                    source_interaction_format=interaction_format_str,
                    source_object_id=str(rec.get("source_object_id", "")),
                    source_timestamp=int(rec.get("source_timestamp", 0)),
                    formatted_timestamp=rec.get("formatted_timestamp", ""),
                    source_hashtags=list(rec.get("source_hashtags", [])),
                )
                if is_negative:
                    self.negative_personas.append(ap)
                else:
                    self.atomic_personas.append(ap)
                    cr = CrossReferencedPersona(
                        persona_item=rec["persona_item"],
                        category=rec.get("category", "uncategorized"),
                        confidence_score_init=float(rec.get("confidence_score_init") or 0.0),
                        confidence_cross_referenced=float(rec.get("confidence_cross_referenced") or 0.0),
                        relationship_type=rec.get("relationship_type", "none"),
                        related_personas=list(rec.get("related_personas", [])),
                        formatted_timestamp=rec.get("formatted_timestamp", ""),
                        source_interaction_type=interaction_type,
                        source_interaction_format=interaction_format_str,
                        assigned_app=rec.get("assigned_app", ""),
                    )
                    self.cross_referenced_personas.append(cr)
                    ann = AnnotatedPersona(
                        persona_item=rec["persona_item"],
                        category=rec.get("category", "uncategorized"),
                        confidence_score_init=float(rec.get("confidence_score_init") or 0.0),
                        confidence_cross_referenced=float(rec.get("confidence_cross_referenced") or 0.0),
                        stereotype_mark=rec.get("stereotype_mark", "neutral"),
                    )
                    self.annotated_personas.append(ann)

                split_label = rec.get("split", "")
                self.split_labels[rec["persona_item"]] = split_label
                if split_label == "test":
                    raw_dis = rec.get("over_personalization_irrelevant", None)
                    if isinstance(raw_dis, list):
                        self.test_distractors[rec["persona_item"]] = [
                            {
                                "persona_item": d.get("persona_item", ""),
                                "category": d.get("category", ""),
                            }
                            for d in raw_dis if isinstance(d, dict)
                        ]
                    elif isinstance(raw_dis, str) and raw_dis:
                        self.test_distractors[rec["persona_item"]] = [
                            {
                                "persona_item": raw_dis,
                                "category": rec.get("over_personalization_irrelevant_category", "") or "",
                            }
                        ]

        # --- Load profile.json ---
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_dict = json.load(f)
        self.user_profile = UserProfile(
            name=profile_dict.get("name", ""),
            gender=profile_dict.get("gender", ""),
            race_ethnicity=profile_dict.get("race_ethnicity", ""),
            career=profile_dict.get("career", ""),
            education=profile_dict.get("education", ""),
            big_five=profile_dict.get("big_five", {}),
            bio=profile_dict.get("bio", ""),
            app_personas=profile_dict.get("app_personas", {}),
        )

        if self.verbose:
            n_test = sum(1 for v in self.split_labels.values() if v == "test")
            n_history = len(self.cross_referenced_personas) - n_test
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Loaded from backend: "
                  f"{len(self.atomic_personas)} positive, "
                  f"{len(self.negative_personas)} negative, "
                  f"{len(self.cross_referenced_personas)} positive canonicals, "
                  f"{len(self.cross_referenced_negatives)} negative canonicals, "
                  f"{n_test} test, {n_history} in interaction history.{utils.Colors.ENDC}")
        return True
