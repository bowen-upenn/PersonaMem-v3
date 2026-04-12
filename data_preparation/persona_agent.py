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

# Add repo root to path so query_llm can be imported from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_preparation import utils, prompts


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
HIGH_CONFIDENCE_CROSS_REF_THRESHOLD = 0.5

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
    "cisgender female, heterosexual": 0.34,
    "cisgender female, bisexual": 0.04,
    "cisgender female, lesbian": 0.02,
    "cisgender female, queer": 0.01,
    "cisgender male, heterosexual": 0.36,
    "cisgender male, bisexual": 0.02,
    "cisgender male, gay": 0.03,
    "cisgender male, queer": 0.01,
    "transgender female, heterosexual": 0.01,
    "transgender female, lesbian": 0.005,
    "transgender female, bisexual": 0.005,
    "transgender male, heterosexual": 0.01,
    "transgender male, gay": 0.005,
    "transgender male, bisexual": 0.005,
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
    "Black or African American": 0.10,
    "African immigrant": 0.02,
    "Afro-Caribbean": 0.02,
    "Mexican American": 0.08,
    "Puerto Rican": 0.02,
    "Cuban American": 0.01,
    "Central American": 0.02,
    "South American": 0.02,
    "Chinese": 0.08,
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
# @ai comments are still rare at ~0.3 weight; etc.). At sample time each
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
            {"action": "at_ai_recommend_more", "label": "@ai comment: asked the in-feed assistant for MORE like this", "weight": 1.0},
            {"action": "at_ai_focus_topic", "label": "@ai comment: asked the in-feed assistant to focus on this topic", "weight": 1.0},
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
            {"action": "at_ai_stop_recommending", "label": "@ai comment: asked the in-feed assistant to STOP showing this", "weight": 1.0},
            {"action": "at_ai_not_interested", "label": "@ai comment: told the in-feed assistant they're not interested right now", "weight": 1.0},
            {"action": "at_ai_feels_off", "label": "@ai comment: told the in-feed assistant this feels off-base", "weight": 1.0},
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
            {"action": "at_ai_recommend_more", "label": "@ai comment: asked Meta AI in the comments for MORE like this", "weight": 1.0},
            {"action": "at_ai_focus_topic", "label": "@ai comment: asked Meta AI in the comments to focus on this topic", "weight": 1.0},
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
            {"action": "at_ai_stop_recommending", "label": "@ai comment: asked Meta AI in the comments to STOP showing this", "weight": 1.0},
            {"action": "at_ai_not_interested", "label": "@ai comment: told Meta AI in the comments they're not interested", "weight": 1.0},
            {"action": "at_ai_feels_off", "label": "@ai comment: told Meta AI in the comments this feels off-base", "weight": 1.0},
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
            {"action": "at_ai_recommend_more", "label": "@ai reply: asked the in-feed assistant for MORE like this", "weight": 1.0},
            {"action": "at_ai_focus_topic", "label": "@ai reply: asked the in-feed assistant to focus on this topic", "weight": 1.0},
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
            {"action": "at_ai_stop_recommending", "label": "@ai reply: asked the in-feed assistant to STOP showing this", "weight": 1.0},
            {"action": "at_ai_not_interested", "label": "@ai reply: told the in-feed assistant they're not interested", "weight": 1.0},
            {"action": "at_ai_feels_off", "label": "@ai reply: told the in-feed assistant this feels off-base", "weight": 1.0},
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
    ):
        self.user_id = user_id
        self.llm_client = llm_client  # QueryLLM instance (None in Claude Code mode)
        self.backend_dir = backend_dir
        self.verbose = verbose

        # Instance variables populated by pipeline or load_from_backend
        self.interactions: list[InteractionRow] = []
        self.atomic_personas: list[AtomicPersona] = []              # positive interactions only
        self.negative_personas: list[AtomicPersona] = []            # negative interactions — standalone, not cross-referenced
        self.cross_referenced_personas: list[CrossReferencedPersona] = []
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

    def infer_personas_from_hashtags(self) -> None:
        """For each interaction, call the LLM to infer atomic persona traits from hashtags.

        Negative interactions (implicit_negative) represent content the user did not click on
        (e.g. promoted content). These are weak signals — their personas are stored separately
        in self.negative_personas and excluded from cross-referencing and temporal analysis.
        """
        self.atomic_personas = []
        self.negative_personas = []

        for idx, interaction in enumerate(self.interactions):
            hashtags = self._extract_hashtags(interaction.object_text)
            if not hashtags:
                continue

            is_negative = "negative" in interaction.interaction_type

            formatted_ts = self._format_timestamp(interaction.interaction_time)
            prompt = prompts.hashtag_to_persona_prompt(
                object_text=interaction.object_text,
                interaction_type=interaction.interaction_type,
                interaction_format=interaction.interaction_format,
                formatted_timestamp=formatted_ts,
                hashtags=hashtags,
            )

            response = self._query_llm_with_retry(prompt)
            if not response:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] Skipping interaction {idx} (no LLM response).{utils.Colors.ENDC}")
                continue

            parsed = utils.extract_json_from_response(response)
            if not isinstance(parsed, list):
                print(f"{utils.Colors.WARNING}[User {self.user_id}] Skipping interaction {idx} (unparseable JSON).{utils.Colors.ENDC}")
                continue

            target_list = self.negative_personas if is_negative else self.atomic_personas

            for item in parsed:
                if not isinstance(item, dict) or "persona_item" not in item:
                    continue
                raw_confidence = float(item.get("confidence_score_init", 0.3))
                # Cap negative interaction confidence — these are unreliable signals
                if is_negative:
                    raw_confidence = min(raw_confidence, 0.2)
                # Use LLM-tagged source hashtags if provided, fall back to full list
                item_hashtags = item.get("source_hashtags", hashtags)
                if not isinstance(item_hashtags, list):
                    item_hashtags = hashtags
                target_list.append(AtomicPersona(
                    persona_item=item["persona_item"],
                    category=item.get("category", "uncategorized"),
                    confidence_score_init=raw_confidence,
                    source_interaction_type=interaction.interaction_type,
                    source_interaction_format=interaction.interaction_format,
                    source_object_id=interaction.object_id,
                    source_timestamp=interaction.interaction_time,
                    formatted_timestamp=formatted_ts,
                    source_hashtags=item_hashtags,
                ))

        if self.verbose:
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Inferred {len(self.atomic_personas)} positive atomic personas, "
                  f"{len(self.negative_personas)} negative (standalone) from {len(self.interactions)} interactions.{utils.Colors.ENDC}")

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

        if self.verbose:
            n_merged = len(self.atomic_personas) - len(canonicals)
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Merged {n_merged} duplicate atomic personas → "
                  f"{len(canonicals)} distinct canonicals.{utils.Colors.ENDC}")

        # --- Step 2: Init filter ---
        survivors = [
            c for c in canonicals
            if c.confidence_score_init >= MIN_PERSONA_INIT_CONFIDENCE
        ]

        if self.verbose:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] After init >= {MIN_PERSONA_INIT_CONFIDENCE} filter: "
                  f"{len(survivors)} canonicals.{utils.Colors.ENDC}")

        # --- Step 3: Count corroboration → confidence_cross_referenced ---
        # For each surviving canonical, count distinct source_object_ids
        # from atomic personas whose individual init >= threshold.
        for c in survivors:
            key = _normalize_persona_text(c.persona_item)
            atoms = groups.get(key, [])
            qualified_sources: set[str] = set()
            for ap in atoms:
                if ap.confidence_score_init >= MIN_PERSONA_INIT_CONFIDENCE and ap.source_object_id:
                    qualified_sources.add(ap.source_object_id)
            c.confidence_cross_referenced = float(len(qualified_sources))

        # --- Step 4: LLM cross-reference for relationship discovery ---
        # (relationship_type + related_personas only; no score changes)

        # Short-circuit: single-row users have nothing to cross-reference.
        unique_objects_all = {ap.source_object_id for ap in self.atomic_personas}
        if len(unique_objects_all) <= 1:
            self.cross_referenced_personas = survivors
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Single interaction row — "
                      f"skipping cross-reference.{utils.Colors.ENDC}")
            return

        personas_for_prompt = [
            {
                "persona_item": c.persona_item,
                "category": c.category,
                "confidence_score_init": c.confidence_score_init,
                "confidence_cross_referenced": c.confidence_cross_referenced,
                "formatted_timestamp": c.formatted_timestamp,
                "source_interaction_type": c.source_interaction_type,
                "source_interaction_format": c.source_interaction_format,
            }
            for c in survivors
        ]

        prompt = prompts.summarize_and_cross_reference_prompt(personas_for_prompt)
        response = self._query_llm_with_retry(prompt)

        canonical_by_norm = {_normalize_persona_text(c.persona_item): c for c in survivors}

        if not response:
            print(f"{utils.Colors.WARNING}[User {self.user_id}] Cross-reference LLM call failed.{utils.Colors.ENDC}")
        else:
            parsed = utils.extract_json_from_response(response)
            if not isinstance(parsed, list):
                print(f"{utils.Colors.WARNING}[User {self.user_id}] Unparseable cross-reference response.{utils.Colors.ENDC}")
            else:
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

        self.cross_referenced_personas = survivors

        if self.verbose:
            n_contradictions = sum(1 for p in survivors if p.relationship_type == "contradictory")
            cr_vals = [c.confidence_cross_referenced for c in survivors]
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] {len(survivors)} survivors, "
                  f"{n_contradictions} contradictory, "
                  f"cross_ref range {min(cr_vals):.0f}..{max(cr_vals):.0f}{utils.Colors.ENDC}")

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
                    timestamp=int(node.get("timestamp", 0)),
                    formatted_timestamp=node.get("formatted_timestamp", ""),
                    confidence_score_init=float(node.get("confidence_score_init", 0.0)),
                    confidence_cross_referenced=float(node.get("confidence_cross_referenced", 0.0)),
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

        Examines how each preference's evidence evolved over time by looking
        at the raw atomic personas (self.atomic_personas). The history only
        records *changes* — not the baseline state:

        - "contradicted": a contradictory preference appeared (with the
          contradicting preference text and its timestamp)
        - "faded": the preference's last qualified occurrence is well before
          the user's overall last interaction, suggesting interest waned

        "new" and "reinforced" entries are NOT stored because they're
        redundant: the item's own position in the time-sorted list and its
        confidence_cross_referenced count already convey when it appeared
        and how frequently it was corroborated.

        The history is stored on each CrossReferencedPersona.update_history
        as a list of dicts sorted by timestamp ascending. Most preferences
        will have an empty list (stable, no changes).
        """
        if not self.cross_referenced_personas or not self.atomic_personas:
            return

        # Group raw atomic personas by normalized key, filtered to init >= threshold
        from collections import defaultdict as _ddict
        groups: dict[str, list] = _ddict(list)
        for ap in self.atomic_personas:
            if ap.confidence_score_init >= MIN_PERSONA_INIT_CONFIDENCE:
                groups[_normalize_persona_text(ap.persona_item)].append(ap)

        # Overall user activity window
        all_timestamps = [ap.source_timestamp for ap in self.atomic_personas if ap.source_timestamp]
        if not all_timestamps:
            return
        user_last_ts = max(all_timestamps)
        FADE_THRESHOLD_SECONDS = 48 * 3600  # 48 hours before end of window = "faded"

        # Build contradictions lookup from cross-ref results
        contradicted_by: dict[str, list] = _ddict(list)
        for cr in self.cross_referenced_personas:
            if cr.relationship_type == "contradictory":
                for rel in cr.related_personas:
                    if isinstance(rel, dict) and rel.get("type") == "contradictory":
                        other = rel.get("persona_item", "")
                        if other:
                            contradicted_by[cr.persona_item].append(other)

        for cr in self.cross_referenced_personas:
            key = _normalize_persona_text(cr.persona_item)
            atoms = groups.get(key, [])
            if not atoms:
                cr.update_history = []
                continue

            atoms_sorted = sorted(atoms, key=lambda a: a.source_timestamp)
            first = atoms_sorted[0]
            last = atoms_sorted[-1]

            history = []

            # "contradicted" if any contradictory relationship exists
            if cr.persona_item in contradicted_by:
                for other_item in contradicted_by[cr.persona_item]:
                    # Find the timestamp of the contradicting preference's first appearance
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

            # "faded" if last occurrence is well before the user's last activity
            if (user_last_ts - last.source_timestamp) >= FADE_THRESHOLD_SECONDS:
                history.append({
                    "preference": cr.persona_item,
                    "update_type": "faded",
                    "timestamp": last.source_timestamp,
                    "formatted_timestamp": utils.unix_to_formatted(last.source_timestamp),
                })

            history.sort(key=lambda h: h["timestamp"])
            cr.update_history = history

        if self.verbose:
            n_multi = sum(1 for cr in self.cross_referenced_personas if len(cr.update_history) > 1)
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Update histories built: "
                  f"{n_multi} preferences have multi-entry histories.{utils.Colors.ENDC}")

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
        all_personas = list(self.cross_referenced_personas) + list(self.negative_personas)
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
            print(f"{utils.Colors.WARNING}[User {self.user_id}] Stereotype annotation LLM call failed.{utils.Colors.ENDC}")
            self.annotated_personas = []
            return

        parsed = utils.extract_json_from_response(response)
        if not isinstance(parsed, list):
            print(f"{utils.Colors.WARNING}[User {self.user_id}] Unparseable stereotype annotation response.{utils.Colors.ENDC}")
            self.annotated_personas = []
            return

        # Build lookup from all personas for confidence scores
        all_lookup: dict[str, any] = {}
        for p in self.cross_referenced_personas:
            all_lookup[p.persona_item] = (p.confidence_score_init, p.confidence_cross_referenced, p.category)
        for p in self.negative_personas:
            all_lookup[p.persona_item] = (p.confidence_score_init, 0.0, p.category)

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

        # Inject 8% noise — deterministic under the agent's random seed.
        for cr in self.cross_referenced_personas:
            if random.random() < self.NOISE_REASSIGN_PROBABILITY:
                alternatives = [a for a in PLATFORMS if a != cr.assigned_app]
                cr.assigned_app = random.choice(alternatives)

        if self.verbose:
            from collections import Counter
            counts = Counter(cr.assigned_app for cr in self.cross_referenced_personas)
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] App routing: {dict(counts)}{utils.Colors.ENDC}")

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

        for cr in self.cross_referenced_personas:
            app = cr.assigned_app or random.choice(PLATFORMS)
            entry = self._sample_action_from_bucket(app, cr.source_interaction_type, rng)
            action_id = entry["action"]
            canonical_label = entry["label"]

            user_message = None
            needs_msg = action_id in AT_AI_ACTIONS or action_id in CHATBOT_TURN_ACTIONS
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
            cr.source_interaction_format = json.dumps(format_obj)

        if self.verbose:
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Interaction formats sampled from perturbed catalog.{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # Train/test split — LLM-gated, with LLM-picked distractors
    # ------------------------------------------------------------------

    def build_test_split(self, fraction: float = 0.2, shortlist_size: int = 5) -> None:
        """Partition positive cross-referenced personas into train/test splits.

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
        # only keeping high-confidence items
        test_candidates: list[CrossReferencedPersona] = []
        for cr in reversed(sorted_positives):
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
          3. LLM cross-reference for relationship discovery (no score changes)
          4. temporal contradiction graph (on surviving canonicals)
          5. generate user profile (demographics + big_five + bio)
          6. generate per-app sub-personas
          7. route preferences to apps (LLM + 8% noise)
          8. generate interaction_format objects per preference (weighted
             sampling from catalog + @ai / chat-turn user_messages)
          9. annotate stereotype marks
         10. build test split (cross-app, global latest-20% high-conf by time)
         11. save to backend/{uid}/ subfolder
        """
        print(f"{utils.Colors.BOLD}[User {self.user_id}] Starting persona pipeline...{utils.Colors.ENDC}")

        self.infer_personas_from_hashtags()
        self.summarize_and_cross_reference()
        self.build_temporal_contradiction_graph()
        self.build_update_histories()
        self.generate_user_profile()
        self.generate_app_personas()
        self.route_personas_to_apps()
        self.generate_interaction_formats()
        self.annotate_stereotype_marks()
        self.build_test_split()
        self.save_to_backend()

        n_test = sum(1 for v in self.split_labels.values() if v == "test")
        n_train = sum(1 for v in self.split_labels.values() if v == "train")

        summary = {
            "user_id": self.user_id,
            "total_interactions": len(self.interactions),
            "total_atomic_personas": len(self.atomic_personas),
            "total_negative_personas": len(self.negative_personas),
            "total_cross_referenced": len(self.cross_referenced_personas),
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
        }
        print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Pipeline complete: {summary}{utils.Colors.ENDC}")
        return summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _user_dir(self) -> str:
        """Directory for this user's output files."""
        return os.path.join(self.backend_dir, str(self.user_id))

    def save_to_backend(self) -> str:
        """Persist data to backend/{user_id}/:

          - profile.json    — UserProfile + AppPersonas + flat list of all
                              persona_item strings under "preferences"
          - instagram.json  — preferences routed to Instagram (time-sorted)
          - facebook.json   — preferences routed to Facebook (time-sorted)
          - threads.json    — preferences routed to Threads (time-sorted)
          - chatbot.json    — preferences routed to Chatbot (time-sorted)

        No semantic redundancy removal is applied — repeated real-world
        signals are meaningful frequency data, and confidence_cross_referenced
        already captures corroboration strength.

        Each app JSON is a list of preference objects. Train/test split is
        global/cross-app (the latest 20% high-confidence items by time carry
        `split: "test"` regardless of which app they live in).
        """
        user_dir = self._user_dir()
        os.makedirs(user_dir, exist_ok=True)

        # --- Build lookups ---
        all_annotated_items = {ap.persona_item: ap for ap in self.annotated_personas}
        atomic_lookup = {ap.persona_item: ap for ap in self.atomic_personas}
        atomic_lookup.update({ap.persona_item: ap for ap in self.negative_personas})

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
                # Legacy "Platform: action" style
                return {"app": fallback_app, "action": "legacy", "action_label": raw, "user_message": None}
            return {"app": fallback_app, "action": "unknown", "action_label": "Unknown", "user_message": None}

        # Assemble one flat list of all surviving preferences (positive +
        # retained negatives), then split by assigned_app. Negatives don't
        # have a routed app in the pipeline; put them on the interaction's
        # implicit home (random spread) OR on the app whose persona best
        # matches. Simple approach: randomly distribute negatives across
        # PLATFORMS with the same noise seed so the distribution is stable.
        all_records: list[dict] = []

        for cr in self.cross_referenced_personas:
            ap = atomic_lookup.get(cr.persona_item)
            ann = all_annotated_items.get(cr.persona_item)
            split_label = self.split_labels.get(cr.persona_item, "train")
            distractor = self.test_distractors.get(cr.persona_item, {}) if split_label == "test" else {}
            app = cr.assigned_app or random.choice(PLATFORMS)
            fmt = _parse_format(cr.source_interaction_format, app)
            record = {
                "persona_item": cr.persona_item,
                "category": cr.category,
                "confidence_score_init": cr.confidence_score_init,
                "confidence_cross_referenced": cr.confidence_cross_referenced,
                "update_history": cr.update_history,
                "stereotype_mark": ann.stereotype_mark if ann else "neutral",
                "split": split_label,
                "source_interaction_type": ap.source_interaction_type if ap else cr.source_interaction_type,
                "source_object_id": ap.source_object_id if ap else "",
                "source_timestamp": ap.source_timestamp if ap else 0,
                "formatted_timestamp": ap.formatted_timestamp if ap else cr.formatted_timestamp,
                "source_hashtags": ap.source_hashtags if ap else [],
                "assigned_app": app,
                "interaction_format": fmt,
            }
            if split_label == "test":
                record["over_personalization_irrelevant"] = distractor.get("persona_item", "")
                record["over_personalization_irrelevant_category"] = distractor.get("category", "")
            all_records.append(record)

        for np_persona in self.negative_personas:
            if np_persona.confidence_score_init <= 0.05:
                continue
            ann = all_annotated_items.get(np_persona.persona_item)
            app = random.choice(PLATFORMS)
            fmt = _parse_format(np_persona.source_interaction_format, app)
            record = {
                "persona_item": np_persona.persona_item,
                "category": np_persona.category,
                "confidence_score_init": np_persona.confidence_score_init,
                "confidence_cross_referenced": 0.0,
                "update_history": [{
                    "preference": np_persona.persona_item,
                    "update_type": "new",
                    "timestamp": np_persona.source_timestamp,
                    "formatted_timestamp": np_persona.formatted_timestamp,
                }],
                "stereotype_mark": ann.stereotype_mark if ann else "neutral",
                "split": "train",
                "source_interaction_type": np_persona.source_interaction_type,
                "source_object_id": np_persona.source_object_id,
                "source_timestamp": np_persona.source_timestamp,
                "formatted_timestamp": np_persona.formatted_timestamp,
                "source_hashtags": np_persona.source_hashtags,
                "assigned_app": app,
                "interaction_format": fmt,
            }
            all_records.append(record)

        # Sort strictly chronological
        all_records.sort(key=lambda r: (int(r.get("source_timestamp") or 0), r.get("persona_item", "")))

        # Bucket by app
        per_app: dict[str, list[dict]] = {a: [] for a in PLATFORMS}
        for rec in all_records:
            app = rec.get("assigned_app") or PLATFORMS[0]
            if app not in per_app:
                per_app[app] = []
            per_app[app].append(rec)

        # --- Write per-app JSONs ---
        for app_name, records in per_app.items():
            filename = app_name.lower() + ".json"  # instagram / facebook / threads / chatbot
            path = os.path.join(user_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

        # --- Write profile.json (includes a flat list of persona_item strings) ---
        if self.user_profile:
            profile_dict = asdict(self.user_profile)
            profile_dict["user_id"] = str(self.user_id)
            profile_dict["preferences"] = [rec["persona_item"] for rec in all_records]
            profile_path = os.path.join(user_dir, "profile.json")
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_dict, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Saved to {user_dir}/ "
                  f"(total prefs: {len(all_records)}, per-app: "
                  f"{ {k: len(v) for k, v in per_app.items()} }){utils.Colors.ENDC}")
        return user_dir

    def load_from_backend(self) -> bool:
        """Load persisted JSON data back into instance variables.

        Reads backend/{uid}/profile.json and backend/{uid}/{app}.json for each
        supported app, merges preferences into in-memory state. Returns True
        if data was found, False otherwise.
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
        self.annotated_personas = []
        self.split_labels = {}
        self.test_distractors = {}

        # --- Load per-app JSONs ---
        all_records: list[dict] = []
        for app_name in PLATFORMS:
            app_path = os.path.join(user_dir, app_name.lower() + ".json")
            if not os.path.exists(app_path):
                continue
            with open(app_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            all_records.extend(records)

        all_records.sort(key=lambda r: (int(r.get("source_timestamp") or 0), r.get("persona_item", "")))

        for rec in all_records:
            interaction_type = rec.get("source_interaction_type", "")
            is_negative = "negative" in interaction_type
            fmt_obj = rec.get("interaction_format") or {}
            interaction_format_str = json.dumps(fmt_obj) if isinstance(fmt_obj, dict) else str(fmt_obj)

            ap = AtomicPersona(
                persona_item=rec["persona_item"],
                category=rec.get("category", "uncategorized"),
                confidence_score_init=float(rec.get("confidence_score_init", 0.0)),
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
                    confidence_score_init=float(rec.get("confidence_score_init", 0.0)),
                    confidence_cross_referenced=float(rec.get("confidence_cross_referenced", 0.0)),
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
                    confidence_score_init=float(rec.get("confidence_score_init", 0.0)),
                    confidence_cross_referenced=float(rec.get("confidence_cross_referenced", 0.0)),
                    stereotype_mark=rec.get("stereotype_mark", "neutral"),
                )
                self.annotated_personas.append(ann)

            split_label = rec.get("split", "train") or "train"
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
                  f"{len(self.annotated_personas)} annotated, "
                  f"{n_train} train, {n_test} test.{utils.Colors.ENDC}")
        return True
