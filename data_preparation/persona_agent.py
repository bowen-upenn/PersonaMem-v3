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
    evidence_row_fraction: float = 0.0                                      # Fraction of user's total rows
    interaction_breakdown: dict = field(default_factory=dict)               # {implicit_positive: N, ...}
    privacy_ratio: float = 0.0                                              # implicit_positive / (implicit_positive + explicit_positive)
    temporal_spread_days: int = 0                                           # Distinct calendar days
    app_distribution: dict = field(default_factory=dict)                    # {Instagram: N, Facebook: N, ...}
    surface_connections: list[str] = field(default_factory=list)            # Which surface preferences this explains
    inferred_motivation: str = ""                                           # 1-2 sentence "why" behind this pattern
    already_captured: bool = False                                          # True if overlaps heavily with surface preferences


# Validation thresholds for hidden persona inference
MIN_HIDDEN_PERSONA_ROWS = 20       # Minimum distinct source rows for a cluster to survive
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
    # Dual personalities — contradictory hidden persona pairs (internal tensions).
    dual_personalities: list = field(default_factory=list)  # list[{"persona_a": str, "persona_b": str, "tension": str}]


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
MIN_PERSONA_INIT_CONFIDENCE = 0.5

# High-confidence predicate — used for test-split eligibility and distractor
# shortlisting. init threshold matches the filter floor so "high-confidence"
# at minimum means the persona survived the init filter AND is corroborated
# by more than a handful of other rows.
HIGH_CONFIDENCE_INIT_THRESHOLD = 0.5
HIGH_CONFIDENCE_CROSS_REF_THRESHOLD = 2.0

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
MIN_IMPLICIT_NEGATIVE_REPETITION = 5  # distinct source rows for implicit-only negative to survive
IMPLICIT_NEGATIVE_PREFILTER_K = 3     # rows per hashtag signature required to bother with LLM call

# NOTE: confidence_cross_referenced is intentionally UNCAPPED on the upper
# side. A preference corroborated by 200 distinct rows should be strictly
# more confident than one corroborated by 10 — they can't both be 1.0. Only
# the lower bound (0.0 floor) is enforced. This makes cross_ref a
# distinguishing signal at scale.


def is_high_confidence(init_score: float, cross_ref_score: float) -> bool:
    """Return True if a persona's scores qualify as 'reasonably high confidence'.

    BOTH conditions must hold:
      - confidence_score_init  >= MIN_PERSONA_INIT_CONFIDENCE (the filter floor)
      - confidence_cross_referenced > HIGH_CONFIDENCE_CROSS_REF_THRESHOLD
        (the persona is independently corroborated by at least ~6 distinct
         interaction rows OR by other semantically-related but distinct personas)

    Since cross_ref is the filtered corroboration count (an integer),
    `> 0.5` means "at least 1 qualifying source row". Most multi-row
    canonicals easily clear this bar.
    """
    return (
        init_score >= HIGH_CONFIDENCE_INIT_THRESHOLD
        and cross_ref_score > HIGH_CONFIDENCE_CROSS_REF_THRESHOLD
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
    "corrected_assumption",
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
    ):
        self.user_id = user_id
        self.llm_client = llm_client  # QueryLLM instance (None in Claude Code mode)
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
        self.test_distractors: dict[str, dict] = {}                  # test persona_item -> {"persona_item": ..., "category": ...}

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
        # Keyed by persona_item → {"conversation": [...], "conversation_type": str, "ask_to_forget": bool}
        self._chatbot_conversations: dict[str, dict] = {}

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
        """Call the LLM with retry logic. Returns response text or None."""
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

    # ------------------------------------------------------------------
    # LLM Call #1: Per-interaction persona inference
    # ------------------------------------------------------------------

    def _infer_one_interaction(self, idx: int, interaction: InteractionRow) -> list[AtomicPersona]:
        """Infer atomic personas from a single interaction row (thread-safe)."""
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
            # Track category for future prompts (thread-safe via set)
            self._known_categories.add(cat.lower())
        return results

    # Weights for net-sentiment scoring of implicit negatives.
    # A hashtag is "hot negative" only when the user consistently skips it
    # AND doesn't engage positively with that topic elsewhere.
    IMPL_NEG_WEIGHT = 1.0    # each implicit_negative row
    EXPL_POS_WEIGHT = 3.0    # each explicit_positive row (strong counter-signal)
    IMPL_POS_WEIGHT = 1.5    # each implicit_positive row (moderate counter-signal)
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
            desc=f"[User {self.user_id}] Step 1b: Implicit-neg promotion",
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
                    self.negative_personas.append(AtomicPersona(
                        persona_item=template.persona_item,
                        category=template.category,
                        confidence_score_init=template.confidence_score_init,
                        source_interaction_type=row.interaction_type,
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

        # --- Step 1a: Infer positives & explicit negatives (skip implicit_negative) ---
        pbar = tqdm(
            total=len(self.interactions),
            desc=f"[User {self.user_id}] Step 1a: Inferring personas",
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

        # --- Step 2: Init filter (with 10% exploration of below-threshold) ---
        above = [c for c in canonicals if c.confidence_score_init >= MIN_PERSONA_INIT_CONFIDENCE]
        below = [c for c in canonicals if c.confidence_score_init < MIN_PERSONA_INIT_CONFIDENCE]
        n_explore = max(1, int(len(below) * 0.10)) if below else 0
        explored = random.sample(below, min(n_explore, len(below))) if n_explore else []
        survivors = above + explored

        if self.verbose:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] After init >= {MIN_PERSONA_INIT_CONFIDENCE} filter: "
                  f"{len(above)} canonicals + {len(explored)} exploratory (10% of {len(below)} below threshold).{utils.Colors.ENDC}")

        # --- Step 3: Weighted corroboration → confidence_cross_referenced ---
        # For each surviving canonical, sum weighted contributions from
        # distinct source rows whose individual init >= threshold.
        # Explicit rows contribute 1.0, implicit rows contribute 0.5.
        for c in survivors:
            key = _normalize_persona_text(c.persona_item)
            atoms = groups.get(key, [])
            seen_sources: set[str] = set()
            base_score = 0.0
            for ap in atoms:
                if ap.confidence_score_init >= MIN_PERSONA_INIT_CONFIDENCE and ap.source_object_id:
                    if ap.source_object_id not in seen_sources:
                        seen_sources.add(ap.source_object_id)
                        base_score += 0.5 if "implicit" in ap.source_interaction_type else 1.0
            c.confidence_cross_referenced = base_score

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

        # Only cross-ref categories with 2+ canonicals
        categories_to_xref = {cat: items for cat, items in by_category.items() if len(items) >= 2}

        if self.verbose:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Cross-referencing {len(categories_to_xref)} categories "
                  f"({sum(len(v) for v in categories_to_xref.values())} canonicals, "
                  f"skipping {len(by_category) - len(categories_to_xref)} single-item categories).{utils.Colors.ENDC}")

        canonical_by_norm = {_normalize_persona_text(c.persona_item): c for c in survivors}

        def _xref_one_category(cat: str, items: list[CrossReferencedPersona]) -> int:
            """Cross-reference within one category. Returns number of relationships found."""
            personas_for_prompt = [{"persona_item": c.persona_item, "category": c.category} for c in items]
            prompt = prompts.summarize_and_cross_reference_prompt(personas_for_prompt)
            response = self._query_llm_with_retry(prompt)
            if not response:
                return 0
            parsed = utils.extract_json_from_response(response)
            if not isinstance(parsed, list):
                return 0
            n_rels = 0
            for item in parsed:
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

        from concurrent.futures import ThreadPoolExecutor, as_completed
        total_rels = 0
        pbar_xref = tqdm(total=len(categories_to_xref),
                         desc=f"[User {self.user_id}] Step 2: Cross-referencing",
                         unit="cat", disable=not self.verbose)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_xref_one_category, cat, items): cat
                for cat, items in categories_to_xref.items()
            }
            for future in as_completed(futures):
                pbar_xref.update(1)
                try:
                    total_rels += future.result()
                except Exception:
                    pass
        pbar_xref.close()

        if self.verbose:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Cross-ref found {total_rels} relationships.{utils.Colors.ENDC}")

        # --- Step 4b: Merge similar preferences into clusters ---
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

        # Apply contradictory penalties
        base_scores: dict[str, float] = {c.persona_item: c.confidence_cross_referenced for c in survivors}
        for c in survivors:
            penalty = 0.0
            for rel in c.related_personas:
                if isinstance(rel, dict) and rel.get("type") == "contradictory":
                    other_base = base_scores.get(rel.get("persona_item", ""), 0.0)
                    penalty += other_base
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

        survivors = self._apply_bottom_20_filter(survivors)
        survivors = [c for c in survivors
                     if c.confidence_cross_referenced >= HIGH_CONFIDENCE_CROSS_REF_THRESHOLD]
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

        # Step 2: Init filter + implicit-only repetition gate. Canonicals
        # supported solely by implicit_negative rows must have at least
        # MIN_IMPLICIT_NEGATIVE_REPETITION distinct source rows; any
        # explicit-negative evidence bypasses the row-count gate.
        neg_survivors: list[CrossReferencedPersona] = []
        n_gated_implicit_only = 0
        for c in neg_canonicals:
            if c.confidence_score_init < MIN_PERSONA_INIT_CONFIDENCE:
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

        if self.verbose and n_gated_implicit_only:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Gated {n_gated_implicit_only} implicit-only "
                  f"negative canonicals (< {MIN_IMPLICIT_NEGATIVE_REPETITION} distinct rows).{utils.Colors.ENDC}")

        # Step 3: Weighted corroboration
        for c in neg_survivors:
            key = _normalize_persona_text(c.persona_item)
            atoms = neg_groups.get(key, [])
            seen: set[str] = set()
            base = 0.0
            for ap in atoms:
                if ap.confidence_score_init >= MIN_PERSONA_INIT_CONFIDENCE and ap.source_object_id:
                    if ap.source_object_id not in seen:
                        seen.add(ap.source_object_id)
                        base += 0.5 if "implicit" in ap.source_interaction_type else 1.0
            c.confidence_cross_referenced = base

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

            # Step 4b: Adjust for relationships
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
        # high-signal (≥5 distinct rows, ≥2 hot hashtags per row).
        self.cross_referenced_negatives = neg_survivors

        if self.verbose:
            cr_vals = [c.confidence_cross_referenced for c in neg_survivors] if neg_survivors else [0.0]
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] {len(neg_survivors)} negative survivors, "
                  f"cross_ref range {min(cr_vals):.1f}..{max(cr_vals):.1f}{utils.Colors.ENDC}")

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
        for cr in self.cross_referenced_personas:
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

        response = self._query_llm_with_retry(prompt)
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
    # Step 5.5: Hidden Persona Inference (cross-row hashtag clustering)
    # ------------------------------------------------------------------

    def infer_hidden_personas(self) -> None:
        """Infer hidden personas from cross-row hashtag patterns.

        Three phases:
        1. Hashtag frequency census across all interaction rows.
        2. LLM thematic clustering + motivation inference.
        3. Algorithmic validation (row count, temporal spread, privacy ratio).

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
                    response = self.llm_client.query(prompt_text)
                else:
                    # Claude Code subagent mode — this method is called
                    # inline so the LLM reasoning IS the execution.
                    return
                raw_clusters = utils.parse_json_response(response)
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

        for cluster in raw_clusters:
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

            # Gate: minimum rows and temporal spread
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
                    summary_text = self.llm_client.query(summary_prompt).strip()
                    if summary_text:
                        break
            except Exception:
                pass

        # ── Detect Dual Personalities ────────────────────────────────────

        dual_personalities: list[dict] = []
        if len(validated) >= 2 and self.llm_client:
            dual_prompt = prompts.identify_dual_personalities_prompt(
                hidden_personas_json=json.dumps(
                    [{"label": hp.label, "type": hp.type, "description": hp.description,
                      "evidence_rows": hp.evidence_rows, "privacy_ratio": hp.privacy_ratio,
                      "interaction_breakdown": hp.interaction_breakdown,
                      "inferred_motivation": hp.inferred_motivation}
                     for hp in validated],
                    indent=2,
                ),
            )
            for attempt in range(self.MAX_RETRIES):
                try:
                    raw_duals = utils.parse_json_response(self.llm_client.query(dual_prompt))
                    if isinstance(raw_duals, list):
                        for d in raw_duals:
                            if isinstance(d, dict) and d.get("persona_a") and d.get("persona_b"):
                                dual_personalities.append({
                                    "persona_a": d["persona_a"],
                                    "persona_b": d["persona_b"],
                                    "tension": d.get("tension", ""),
                                })
                        break
                except Exception:
                    pass

        # Store on profile
        self.user_profile.hidden_personas = validated
        self.user_profile.hidden_persona_summary = summary_text
        self.user_profile.dual_personalities = dual_personalities

        if self.verbose:
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Inferred {len(validated)} "
                  f"hidden personas, {len(dual_personalities)} dual tensions{utils.Colors.ENDC}")

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

        for oid, atoms in atomics_by_oid.items():
            app_votes = []
            for ap in atoms:
                key = _normalize_persona_text(ap.persona_item)
                app = canonical_app.get(key, "")
                if app:
                    app_votes.append(app)
            if app_votes:
                row_apps[oid] = _Counter(app_votes).most_common(1)[0][0]
            else:
                row_apps[oid] = random.choice(PLATFORMS)

        # Step 2: Session majority vote — override all rows in session
        for session in self._sessions:
            session_votes = [row_apps.get(r.object_id, "") for r in session]
            session_votes = [v for v in session_votes if v]
            if session_votes:
                session_app = _Counter(session_votes).most_common(1)[0][0]
            else:
                session_app = random.choice(PLATFORMS)
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
        response = self._query_llm_with_retry(prompt)

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

        if self.verbose:
            from collections import Counter
            counts = Counter(cr.assigned_app for cr in self.cross_referenced_personas)
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Canonical app routing: {dict(counts)}{utils.Colors.ENDC}")

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
                response = self._query_llm_with_retry(prompt)
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
        """Generate multi-turn conversations for preferences routed to Chatbot.

        Delegates to ``chatbot_conversation.generate_chatbot_conversations()``
        which calls the LLM to produce PersonaMem-v2-style task-oriented
        conversations that implicitly embed each preference.

        Results are stored in ``self._chatbot_conversations`` (keyed by
        persona_item) and merged into the Chatbot records at save time.
        """
        if not self.cross_referenced_personas or not self.user_profile:
            return
        if self.llm_client is None:
            # In Claude Code subagent mode, this step is done inline
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

        # Build flat records for chatbot preferences (mirrors save_to_backend format)
        atomic_lookup = {ap.persona_item: ap for ap in self.atomic_personas}
        chatbot_records: list[dict] = []
        for cr in self.cross_referenced_personas:
            if cr.assigned_app != "Chatbot":
                continue
            ap = atomic_lookup.get(cr.persona_item)
            fmt = {}
            if cr.source_interaction_format:
                try:
                    fmt = json.loads(cr.source_interaction_format) if isinstance(
                        cr.source_interaction_format, str) else cr.source_interaction_format
                except (ValueError, TypeError):
                    fmt = {}
            chatbot_records.append({
                "persona_item": cr.persona_item,
                "category": cr.category,
                "source_interaction_type": ap.source_interaction_type if ap else cr.source_interaction_type,
                "interaction_format": fmt if isinstance(fmt, dict) else {},
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

        # Store results keyed by persona_item
        for rec in chatbot_records:
            if rec.get("conversation"):
                self._chatbot_conversations[rec["persona_item"]] = {
                    "conversation": rec["conversation"],
                    "conversation_type": rec["conversation_type"],
                    "ask_to_forget": rec["ask_to_forget"],
                    "interaction_format_override": rec.get("interaction_format"),
                }

        if self.verbose:
            n_conv = len(self._chatbot_conversations)
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] "
                  f"Generated {n_conv} chatbot conversations.{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # Train/test split — LLM-gated, with LLM-picked distractors
    # ------------------------------------------------------------------

    def build_test_split(self, fraction: float = 0.2, shortlist_size: int = 5) -> None:
        """Partition positive cross-referenced personas into train/test splits.

        NOTE: implicit_negative personas are excluded from test by construction:
          (a) they live in self.negative_personas / cross_referenced_negatives,
              not self.cross_referenced_personas,
          (b) the explicit guard below skips any that might leak through, and
          (c) negative personas are always labelled "train" in save_to_backend.

        Test eligibility rules:
          1. Sort positive personas by source_timestamp ascending (early -> latest).
          2. Scan newest -> oldest collecting items that pass `is_high_confidence`
             until we have `fraction * total_positives` candidates (or run out).
             Only truly high-confidence items can be test candidates.
          3. LLM inferrability gate: ask whether each candidate is reasonably
             inferrable from the train 80% (ground truth). Candidates the LLM
             marks as NOT inferrable are REMOVED from self.cross_referenced_personas
             entirely — they fail the fidelity gate.
          4. Distractor pairing for each surviving test item:
               Stage A (Python): randomly shortlist `shortlist_size` train items
                 passing the high-confidence predicate.
               Stage B (LLM): picks the one most topically irrelevant and most
                 annoying/inappropriate as a personalization recommendation.
             The chosen distractor is stored in self.test_distractors.
          5. self.split_labels is populated for every surviving positive persona
             ("train" or "test"). Negative personas are always "train" and are
             not passed through the gate.
        """
        self.split_labels = {}
        self.test_distractors = {}

        if not self.cross_referenced_personas:
            return

        # Build lookup of source_timestamp from atomic_personas (authoritative)
        ts_lookup: dict[str, int] = {ap.persona_item: ap.source_timestamp for ap in self.atomic_personas}

        sorted_positives = sorted(
            self.cross_referenced_personas,
            key=lambda p: ts_lookup.get(p.persona_item, 0),
        )

        total = len(sorted_positives)
        n_test_target = max(1, int(total * fraction))

        # Collect test candidates from the tail of the timeline, newest first,
        # only keeping high-confidence items. Explicit guard: never test negatives.
        test_candidates: list[CrossReferencedPersona] = []
        for cr in reversed(sorted_positives):
            if "negative" in cr.source_interaction_type:
                continue
            if is_high_confidence(cr.confidence_score_init, cr.confidence_cross_referenced):
                test_candidates.append(cr)
                if len(test_candidates) >= n_test_target:
                    break
        # Preserve chronological order (oldest-of-selected first)
        test_candidates.reverse()

        if not test_candidates:
            # No high-confidence tail — everything stays train
            for cr in self.cross_referenced_personas:
                self.split_labels[cr.persona_item] = "train"
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] No high-confidence test candidates — "
                      f"all {total} positives marked train.{utils.Colors.ENDC}")
            return

        test_candidate_set = {cr.persona_item for cr in test_candidates}

        # Build the train pool now (everything else among positives)
        train_pool: list[CrossReferencedPersona] = [
            cr for cr in sorted_positives if cr.persona_item not in test_candidate_set
        ]

        # --- LLM inferrability gate ---
        train_personas_for_prompt = [
            {
                "persona_item": cr.persona_item,
                "category": cr.category,
                "confidence_score_init": cr.confidence_score_init,
                "confidence_cross_referenced": cr.confidence_cross_referenced,
                "formatted_timestamp": cr.formatted_timestamp,
            }
            for cr in train_pool
        ]
        test_candidates_for_prompt = [
            {
                "persona_item": cr.persona_item,
                "category": cr.category,
                "confidence_score_init": cr.confidence_score_init,
                "confidence_cross_referenced": cr.confidence_cross_referenced,
                "formatted_timestamp": cr.formatted_timestamp,
            }
            for cr in test_candidates
        ]

        inferrable_items: set[str] = set()
        non_inferrable_items: set[str] = set()

        if self.llm_client is not None:
            prompt = prompts.test_inferrability_check_prompt(
                train_personas=train_personas_for_prompt,
                test_candidates=test_candidates_for_prompt,
            )
            response = self._query_llm_with_retry(prompt)
            if response:
                parsed = utils.extract_json_from_response(response)
                if isinstance(parsed, list):
                    for item in parsed:
                        if not isinstance(item, dict):
                            continue
                        name = item.get("persona_item", "")
                        if not name:
                            continue
                        if bool(item.get("inferrable", False)):
                            inferrable_items.add(name)
                        else:
                            non_inferrable_items.add(name)
                else:
                    print(f"{utils.Colors.WARNING}[User {self.user_id}] Unparseable test inferrability response — "
                          f"defaulting to keep all candidates.{utils.Colors.ENDC}")
                    inferrable_items = set(test_candidate_set)
            else:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] Inferrability check LLM call failed — "
                      f"defaulting to keep all candidates.{utils.Colors.ENDC}")
                inferrable_items = set(test_candidate_set)
        else:
            # Claude Code subagent mode: persona_agent runs *without* an llm_client;
            # subagents perform this step inline per skill.md. In that mode this
            # method should not normally be called. Fall back to keeping all candidates.
            inferrable_items = set(test_candidate_set)

        # --- Drop non-inferrable test candidates entirely ---
        if non_inferrable_items:
            self.cross_referenced_personas = [
                cr for cr in self.cross_referenced_personas
                if cr.persona_item not in non_inferrable_items
            ]

        kept_test: list[CrossReferencedPersona] = [
            cr for cr in test_candidates if cr.persona_item in inferrable_items
        ]

        # --- Distractor pairing ---
        high_conf_train = [
            cr for cr in train_pool
            if is_high_confidence(cr.confidence_score_init, cr.confidence_cross_referenced)
        ]

        for test_cr in kept_test:
            self.split_labels[test_cr.persona_item] = "test"
            if not high_conf_train:
                if self.verbose:
                    print(f"{utils.Colors.WARNING}[User {self.user_id}] No high-confidence train items — "
                          f"test item '{test_cr.persona_item}' has no distractor.{utils.Colors.ENDC}")
                continue

            n_to_sample = min(shortlist_size, len(high_conf_train))
            shortlist = random.sample(high_conf_train, n_to_sample)
            shortlist_for_prompt = [
                {"persona_item": cr.persona_item, "category": cr.category}
                for cr in shortlist
            ]

            chosen_name: str = ""
            if self.llm_client is not None and shortlist_for_prompt:
                prompt = prompts.distractor_selection_prompt(
                    test_persona={"persona_item": test_cr.persona_item, "category": test_cr.category},
                    candidate_distractors=shortlist_for_prompt,
                )
                response = self._query_llm_with_retry(prompt)
                if response:
                    parsed = utils.extract_json_from_response(response)
                    if isinstance(parsed, dict):
                        chosen_name = parsed.get("chosen_persona_item", "") or ""

            # Fall back: first shortlist item if LLM pick is missing / invalid
            valid_names = {cr.persona_item for cr in shortlist}
            if chosen_name not in valid_names:
                chosen_name = shortlist[0].persona_item

            chosen_cr = next(cr for cr in shortlist if cr.persona_item == chosen_name)
            self.test_distractors[test_cr.persona_item] = {
                "persona_item": chosen_cr.persona_item,
                "category": chosen_cr.category,
            }

        # --- Everything else that survived the gate is train ---
        for cr in self.cross_referenced_personas:
            if cr.persona_item not in self.split_labels:
                self.split_labels[cr.persona_item] = "train"

        if self.verbose:
            n_test = sum(1 for v in self.split_labels.values() if v == "test")
            n_train = sum(1 for v in self.split_labels.values() if v == "train")
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Test split: "
                  f"{n_train} train, {n_test} test, "
                  f"{len(non_inferrable_items)} dropped by inferrability gate, "
                  f"{sum(1 for v in self.test_distractors.values())} distractors assigned."
                  f"{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_pipeline(self) -> dict:
        """Run the full persona inference pipeline.

        Order:
          1. infer atomic personas
          2. dedupe (lexical) + init filter + count corroboration → cross_ref
          3. cross-reference & filter
          4. temporal contradiction graph
          5. build update histories
          6. generate user profile (demographics + big_five + bio)
          7. infer hidden personas (cross-row hashtag clustering)
          8. generate per-app sub-personas
          9. build sessions
         10. route preferences to apps (LLM + 8% noise)
         11. assign rows to apps (session majority vote)
         12. generate interaction formats (weighted catalog sampling)
         13. generate chatbot conversations (multi-turn, implicit embedding)
         14. annotate stereotype marks
         15. build test split (cross-app, latest-20% high-conf by time)
         16. save to backend/{uid}/ subfolder
        """
        print(f"{utils.Colors.BOLD}[User {self.user_id}] Starting persona pipeline...{utils.Colors.ENDC}")
        pipeline_start = time.time()

        steps = [
            ("1.  Infer atomic personas",          self.infer_personas_from_hashtags),
            ("2.  Promote implicit negatives",      self.promote_implicit_negatives),
            ("3.  Cross-reference & filter",        self.summarize_and_cross_reference),
            ("4.  Temporal contradiction graph",     self.build_temporal_contradiction_graph),
            ("5.  Build update histories",           self.build_update_histories),
            ("6.  Generate user profile",            self.generate_user_profile),
            ("7.  Infer hidden personas",            self.infer_hidden_personas),
            ("8.  Generate app personas",            self.generate_app_personas),
            ("9.  Build sessions",                   self._build_sessions),
            ("10. Route preferences to apps",        self.route_personas_to_apps),
            ("11. Assign rows to apps",              self._assign_rows_to_apps),
            ("12. Generate interaction formats",     self.generate_interaction_formats),
            ("13. Generate chatbot conversations",   self.generate_chatbot_conversations),
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
        n_train = sum(1 for v in self.split_labels.values() if v == "train")

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
            "split_train": n_train,
            "split_test": n_test,
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

        # Hidden persona hashtag lookup: for each hidden persona, pre-compute
        # lowercase evidence hashtag set for matching against preference source_hashtags.
        _hp_tag_sets: list[tuple[str, set[str]]] = []
        if self.user_profile and self.user_profile.hidden_personas:
            for hp in self.user_profile.hidden_personas:
                tag_set = set(t.lower() for t in hp.evidence_hashtags)
                _hp_tag_sets.append((hp.label, tag_set))

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

        for oid, atoms in atomics_by_oid.items():
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
                distractor = self.test_distractors.get(cr.persona_item, {}) if split_label == "test" else {}

                # Build merged update_history: temporal entries (no raw timestamp)
                # + related_personas folded in as similar/contradictory entries.
                # Causality: only keep entries whose timestamp <= this event's time.
                # Key order: update_type, preference, formatted_timestamp, then extras.
                _HISTORY_KEY_ORDER = ["update_type", "preference", "formatted_timestamp",
                                      "source_app", "occurrence", "total_occurrences", "description"]
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
                    if isinstance(rel, dict) and rel.get("persona_item"):
                        rel_cr = canonical_lookup.get(_normalize_persona_text(rel["persona_item"]))
                        # Only include if the related preference appeared before this event
                        if rel_cr:
                            rel_atoms = self._canonical_groups.get(
                                _normalize_persona_text(rel["persona_item"]), [])
                            rel_first_ts = min((a.source_timestamp for a in rel_atoms), default=0) if rel_atoms else 0
                            if rel_first_ts > event_ts:
                                continue  # skip future relationships — causality
                        merged_history.append({
                            "update_type": rel.get("type", "similar"),
                            "preference": rel["persona_item"],
                            "formatted_timestamp": rel_cr.formatted_timestamp if rel_cr else "",
                            "source_app": rel_cr.assigned_app if rel_cr else "",
                        })

                # Match preference's source hashtags against hidden personas
                hp_labels: list[str] = []
                if _hp_tag_sets and ap.source_hashtags:
                    pref_tags_lower = set(t.lower() for t in ap.source_hashtags)
                    for hp_label, hp_tags in _hp_tag_sets:
                        if pref_tags_lower & hp_tags:
                            hp_labels.append(hp_label)

                pref = {
                    "persona_item": cr.persona_item,
                    "category": cr.category,
                    "confidence_score_init": ap.confidence_score_init,
                    "confidence_cross_referenced": cr.confidence_cross_referenced,
                    "stereotype_mark": ann.stereotype_mark if ann else "neutral",
                    "hidden_persona_labels": hp_labels,
                    "update_history": merged_history,
                }
                if split_label == "test":
                    pref["split"] = "test"
                    pref["over_personalization_irrelevant"] = distractor.get("persona_item", "")
                    pref["over_personalization_irrelevant_category"] = distractor.get("category", "")

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
            # Promote implicit_negative → explicit_negative when the event
            # carries surviving preferences (i.e. the ≥5 repetition gate
            # passed and real negative preferences were inferred).
            itype = rep.source_interaction_type or "implicit_positive"
            if itype == "implicit_negative" and preferences:
                itype = "explicit_negative"
            # For Chatbot: reassign 20% explicit / 80% implicit (polarity kept).
            if app == "Chatbot" and itype != "implicit_negative":
                polarity = "negative" if "negative" in itype else "positive"
                itype = f"explicit_{polarity}" if event_rng.random() < 0.20 else f"implicit_{polarity}"

            # interaction_format: independently sample one action per event
            sampled_entry = self._sample_action_from_bucket(app, itype, event_rng)
            fmt = {
                "app": app,
                "action": sampled_entry["action"],
                "action_label": sampled_entry["label"],
                "user_message": None,
            }
            # Preserve existing user_message from stored format if available
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

            # Merge chatbot conversation data before preferences
            if app == "Chatbot":
                for ap in atoms:
                    key = _normalize_persona_text(ap.persona_item)
                    cr = canonical_lookup.get(key)
                    if cr and cr.persona_item in self._chatbot_conversations:
                        conv_data = self._chatbot_conversations[cr.persona_item]
                        event["conversation_type"] = conv_data.get("conversation_type")
                        event["conversation"] = conv_data.get("conversation")
                        event["ask_to_forget"] = conv_data.get("ask_to_forget", False)
                        override = conv_data.get("interaction_format_override")
                        if override and isinstance(override, dict):
                            action = override.get("action")
                            if action in ("asked_to_forget", "corrected_assumption"):
                                event["interaction_format"] = override
                        break  # one conversation per event

            event["preferences"] = preferences  # always last

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
            profile_dict["preferences"] = seen_unique_prefs
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
                        distractor_item = pref.get("over_personalization_irrelevant", "") or ""
                        if distractor_item:
                            self.test_distractors[persona_item] = {
                                "persona_item": distractor_item,
                                "category": pref.get("over_personalization_irrelevant_category", "") or "",
                            }

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
                    distractor_item = rec.get("over_personalization_irrelevant", "") or ""
                    if distractor_item:
                        self.test_distractors[rec["persona_item"]] = {
                            "persona_item": distractor_item,
                            "category": rec.get("over_personalization_irrelevant_category", "") or "",
                        }

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
            n_train = sum(1 for v in self.split_labels.values() if v == "train")
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Loaded from backend: "
                  f"{len(self.atomic_personas)} positive, "
                  f"{len(self.negative_personas)} negative, "
                  f"{len(self.cross_referenced_personas)} positive canonicals, "
                  f"{len(self.cross_referenced_negatives)} negative canonicals, "
                  f"{n_train} train, {n_test} test.{utils.Colors.ENDC}")
        return True
