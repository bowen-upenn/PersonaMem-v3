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
    confidence_cross_referenced: float
    relationship_type: str = "none"          # "similar", "contradictory", "none"
    related_personas: list = field(default_factory=list)  # list of {"persona_item": str, "type": str}
    formatted_timestamp: str = ""
    source_interaction_type: str = ""
    source_interaction_format: str = ""
    # Number of DISTINCT interaction rows (distinct source_object_id) that produced
    # this canonical persona — i.e. how many independent rows independently inferred
    # the same preference. `corroboration_count` starts at 1 (the row that first
    # produced it) and grows as duplicate rows are merged in.
    corroboration_count: int = 1
    # Which app the router assigned this preference to.
    assigned_app: str = ""


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

# Strict floor on confidence_score_init. Personas below this are dropped
# after cross-ref regardless of cross-ref score or relationship type. This
# is the main knob for preference-list size. Tuneable.
MIN_PERSONA_INIT_CONFIDENCE = 0.8

# High-confidence predicate — used for test-split eligibility and distractor
# shortlisting. Stricter than the filter: init must be well above the floor,
# cross_ref must show at least some independent corroboration.
HIGH_CONFIDENCE_INIT_THRESHOLD = 0.8
HIGH_CONFIDENCE_CROSS_REF_THRESHOLD = 0.5

# Hard ceiling on confidence_cross_referenced. Even with many similar pairs
# the score cannot exceed 1.0.
CROSS_REF_CAP = 1.0


def is_high_confidence(init_score: float, cross_ref_score: float) -> bool:
    """Return True if a persona's scores qualify as 'reasonably high confidence'.

    BOTH conditions must hold:
      - confidence_score_init  >= MIN_PERSONA_INIT_CONFIDENCE (the filter floor)
      - confidence_cross_referenced > HIGH_CONFIDENCE_CROSS_REF_THRESHOLD
        (the persona is independently corroborated by other distinct interaction rows
         OR by other semantically-related but distinct personas)
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

# Per-app action catalog. Each action is an identifier + a human-readable
# label. The subagent/LLM picks one action from the appropriate polarity
# bucket when generating an interaction_format for a routed preference.
# Expanded from the previous 3-4 generic options to reflect realistic
# modern-day UX affordances on each app.
PLATFORM_INTERACTION_FORMATS: dict[str, dict[str, list[dict]]] = {
    "Instagram": {
        "explicit_positive": [
            {"action": "liked", "label": "Liked"},
            {"action": "double_tapped", "label": "Double-tapped to like"},
            {"action": "reposted", "label": "Reposted"},
            {"action": "commented", "label": "Commented"},
            {"action": "saved_to_collection", "label": "Saved to a collection"},
            {"action": "shared_to_close_friends_story", "label": "Shared to Close Friends story"},
            {"action": "dm_to_friend", "label": "Sent via DM to a friend"},
            {"action": "followed_creator", "label": "Followed the creator"},
            {"action": "reacted_to_story", "label": "Reacted to the story"},
        ],
        "implicit_positive": [
            {"action": "viewed_reel_75", "label": "Viewed more than 75% of the reel"},
            {"action": "rewatched_reel", "label": "Rewatched the reel"},
            {"action": "lingered_on_image", "label": "Stayed on an image for more than 5 seconds"},
            {"action": "lingered_on_story", "label": "Stayed on a story for more than 5 seconds"},
            {"action": "tapped_profile", "label": "Tapped through to the creator's profile"},
            {"action": "long_pressed_for_options", "label": "Long-pressed to open context menu"},
        ],
        "explicit_negative": [
            {"action": "not_interested", "label": "Marked Not Interested"},
            {"action": "hidden", "label": "Hid this post"},
            {"action": "reported", "label": "Reported"},
            {"action": "muted_user", "label": "Muted the user"},
            {"action": "unfollowed", "label": "Unfollowed the creator"},
        ],
        "implicit_negative": [
            {"action": "skipped_reel", "label": "Skipped the reel with no interaction"},
            {"action": "skipped_image", "label": "Skipped the image with no interaction"},
            {"action": "skipped_story", "label": "Skipped the story with no interaction"},
        ],
    },
    "Facebook": {
        "explicit_positive": [
            {"action": "reacted_like", "label": "Liked"},
            {"action": "reacted_love", "label": "Loved (❤)"},
            {"action": "reacted_haha", "label": "Hahaha reaction"},
            {"action": "reacted_wow", "label": "Wow reaction"},
            {"action": "reacted_sad", "label": "Sad reaction"},
            {"action": "reacted_care", "label": "Care reaction"},
            {"action": "commented", "label": "Commented"},
            {"action": "shared_to_timeline", "label": "Shared to own timeline"},
            {"action": "shared_to_group", "label": "Shared to a group"},
            {"action": "tagged_friend", "label": "Tagged a friend in the post"},
            {"action": "saved_post", "label": "Saved the post"},
            {"action": "rsvp_event", "label": "Marked Interested / Going on an event"},
        ],
        "implicit_positive": [
            {"action": "viewed_video_75", "label": "Viewed more than 75% of the video"},
            {"action": "lingered_on_post", "label": "Stayed on a post for more than 5 seconds"},
            {"action": "expanded_see_more", "label": "Tapped 'See more' to expand the post"},
            {"action": "viewed_comments", "label": "Opened the comments thread"},
        ],
        "explicit_negative": [
            {"action": "reacted_angry", "label": "Angry reaction"},
            {"action": "hidden", "label": "Hid the post"},
            {"action": "snoozed_user", "label": "Snoozed the user for 30 days"},
            {"action": "see_fewer_like_this", "label": "Asked to see fewer posts like this"},
            {"action": "unfollowed", "label": "Unfollowed the page / user"},
            {"action": "reported", "label": "Reported"},
        ],
        "implicit_negative": [
            {"action": "skipped_post", "label": "Skipped the post with no interaction"},
            {"action": "scrolled_past_video", "label": "Scrolled past the video without watching"},
        ],
    },
    "Threads": {
        "explicit_positive": [
            {"action": "liked", "label": "Liked"},
            {"action": "reposted", "label": "Reposted"},
            {"action": "quote_reposted", "label": "Reposted with a quote"},
            {"action": "replied", "label": "Replied"},
            {"action": "followed_author", "label": "Followed the author"},
            {"action": "shared_externally", "label": "Shared externally (copy link / DM)"},
            {"action": "saved", "label": "Saved the thread"},
        ],
        "implicit_positive": [
            {"action": "lingered_on_thread", "label": "Stayed on the thread for more than 5 seconds"},
            {"action": "viewed_video_75", "label": "Viewed more than 75% of the video"},
            {"action": "expanded_replies", "label": "Expanded the reply thread"},
            {"action": "tapped_author", "label": "Tapped through to the author's profile"},
        ],
        "explicit_negative": [
            {"action": "not_interested", "label": "Marked Not Interested"},
            {"action": "muted_author", "label": "Muted the author"},
            {"action": "hid_replies", "label": "Hid the replies"},
            {"action": "reported", "label": "Reported"},
        ],
        "implicit_negative": [
            {"action": "skipped_thread", "label": "Skipped the thread with no interaction"},
        ],
    },
    "Chatbot": {
        "explicit_positive": [
            {"action": "thumbs_up", "label": "Thumbs-upped the response"},
            {"action": "saved_to_library", "label": "Saved the response to library"},
            {"action": "copied_response", "label": "Copied the response text"},
            {"action": "asked_followup", "label": "Asked a follow-up question showing interest"},
            {"action": "requested_more_detail", "label": "Requested more detail on the same topic"},
            {"action": "shared_conversation", "label": "Shared the conversation externally"},
            # NEW — `@ai` steering directives. The user_message field on the
            # final interaction_format object will carry the actual text.
            {"action": "at_ai_recommend_more", "label": "@ai steering: asked for MORE of this type"},
            {"action": "at_ai_focus_topic", "label": "@ai steering: asked to focus on this topic"},
        ],
        "implicit_positive": [
            {"action": "continued_topic", "label": "Continued the conversation on the same topic"},
            {"action": "read_carefully", "label": "Spent significant time reading the response"},
            {"action": "referenced_response", "label": "Copied or referenced part of the response"},
            {"action": "positive_language_next_turn", "label": "Positive language in the next turn"},
        ],
        "explicit_negative": [
            {"action": "thumbs_down", "label": "Thumbs-downed the response"},
            {"action": "asked_to_change_topic", "label": "Explicitly asked to change topic or stop"},
            {"action": "edited_prompt_and_retried", "label": "Edited the prompt and retried"},
            {"action": "reported_response", "label": "Reported / flagged the response"},
            {"action": "regenerated", "label": "Asked to regenerate the response"},
            # NEW — `@ai` steering directives (negative side).
            {"action": "at_ai_stop_recommending", "label": "@ai steering: asked to STOP recommending this"},
            {"action": "at_ai_not_interested", "label": "@ai steering: said not interested right now"},
            {"action": "at_ai_feels_off", "label": "@ai steering: said this recommendation feels off-base"},
        ],
        "implicit_negative": [
            {"action": "abandoned_conversation", "label": "Abandoned the conversation after the response"},
            {"action": "changed_topic_immediately", "label": "Immediately changed the topic"},
            {"action": "dismissive_reply", "label": "Gave a minimal or dismissive reply"},
            {"action": "no_followup", "label": "No active follow-up or response"},
        ],
    },
}


# Action identifiers that REQUIRE a natural-language `user_message` to be
# generated by the LLM (chatbot `@ai` directives). The message is first-person,
# ~1-2 sentences, and grounded in the specific preference the directive acts
# on (e.g. "@ai show me more authentic Mexican breakfast recipes this week").
AT_AI_ACTIONS: set[str] = {
    "at_ai_recommend_more",
    "at_ai_focus_topic",
    "at_ai_stop_recommending",
    "at_ai_not_interested",
    "at_ai_feels_off",
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
        """Dedupe atomic personas by semantic equivalence, then cross-reference.

        This method performs three distinct operations in order:

        1. **Merge duplicates**: If multiple atomic personas share the same
           normalized persona_item text, they represent the SAME preference
           corroborated by different interaction rows — not distinct
           preferences similar to each other. Collapse them into one canonical
           persona whose `confidence_score_init` is the max across dupes and
           whose `corroboration_count` records how many distinct rows
           contributed. Corroboration gives a base `confidence_cross_referenced`
           of `0.1 * (corroboration_count - 1)`.

        2. **Cross-reference distinct canonicals** via LLM for `similar` /
           `contradictory` pairs (cross-row only; personas from the same row
           cannot validate each other). The LLM MUST NOT mark identical
           persona_items as similar — those are already merged.

           Python-side additive scoring on top of the merge base:
           - Each `similar` pair adds +0.1 to BOTH personas
           - Each `contradictory` pair subtracts -0.1 from the OLDER persona only
           - Score is floored at 0.0 and capped at CROSS_REF_CAP (1.0)

        3. **Strict filter**: drop canonicals with `confidence_score_init <
           MIN_PERSONA_INIT_CONFIDENCE` (0.8). This is a hard gate — even
           contradictions and high-cross-ref items below the init floor are
           removed. Rationale: at real-world scale we only want strong-signal
           preferences.
        """
        if not self.atomic_personas:
            self.cross_referenced_personas = []
            return

        # --- Step 1: Merge duplicates ---
        # Group atomic personas by normalized persona_item text. For each group,
        # build a canonical CrossReferencedPersona with max init, corroboration
        # count, and the earliest/most-recent metadata preserved.
        canonical_by_key: dict[str, CrossReferencedPersona] = {}
        canonical_source_objects: dict[str, set[str]] = {}  # key -> set of source_object_ids
        canonical_order: list[str] = []  # preserve first-seen order

        for ap in self.atomic_personas:
            key = _normalize_persona_text(ap.persona_item)
            if not key:
                continue
            if key not in canonical_by_key:
                canonical_by_key[key] = CrossReferencedPersona(
                    persona_item=ap.persona_item,
                    category=ap.category,
                    confidence_score_init=ap.confidence_score_init,
                    confidence_cross_referenced=0.0,
                    relationship_type="none",
                    related_personas=[],
                    formatted_timestamp=ap.formatted_timestamp,
                    source_interaction_type=ap.source_interaction_type,
                    source_interaction_format=ap.source_interaction_format,
                    corroboration_count=1,
                )
                canonical_source_objects[key] = {ap.source_object_id} if ap.source_object_id else set()
                canonical_order.append(key)
            else:
                canonical = canonical_by_key[key]
                # Bump init to max across duplicates
                if ap.confidence_score_init > canonical.confidence_score_init:
                    canonical.confidence_score_init = ap.confidence_score_init
                # Count distinct source rows contributing to this canonical
                if ap.source_object_id and ap.source_object_id not in canonical_source_objects[key]:
                    canonical_source_objects[key].add(ap.source_object_id)
                    canonical.corroboration_count += 1
                # Update earliest timestamp (which is just the first-seen one since atomic_personas are sorted early→late)
                # — do nothing; the first one is oldest.

        # Seed confidence_cross_referenced from corroboration count:
        # +0.1 per distinct additional row that produced the same persona.
        for key, canonical in canonical_by_key.items():
            merge_boost = 0.1 * max(0, canonical.corroboration_count - 1)
            canonical.confidence_cross_referenced = round(
                min(CROSS_REF_CAP, merge_boost), 2
            )

        canonicals: list[CrossReferencedPersona] = [canonical_by_key[k] for k in canonical_order]

        if self.verbose:
            n_merged = len(self.atomic_personas) - len(canonicals)
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Merged {n_merged} duplicate atomic personas — "
                  f"{len(canonicals)} distinct canonicals.{utils.Colors.ENDC}")

        # Serialize for the cross-ref prompt (only the canonicals, already merged)
        personas_for_prompt = []
        for c in canonicals:
            personas_for_prompt.append({
                "persona_item": c.persona_item,
                "category": c.category,
                "confidence_score_init": c.confidence_score_init,
                "corroboration_count": c.corroboration_count,
                "formatted_timestamp": c.formatted_timestamp,
                "source_interaction_type": c.source_interaction_type,
                "source_interaction_format": c.source_interaction_format,
            })

        # If all canonicals came from a single row, there's nothing to cross-reference
        # (canonicals already reflect same-row merges). Short-circuit.
        unique_objects_all = {ap.source_object_id for ap in self.atomic_personas}
        if len(unique_objects_all) <= 1:
            self.cross_referenced_personas = [
                c for c in canonicals if c.confidence_score_init >= MIN_PERSONA_INIT_CONFIDENCE
            ]
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Single interaction row — skipping cross-reference. "
                      f"{len(self.cross_referenced_personas)} canonicals kept after init>={MIN_PERSONA_INIT_CONFIDENCE} filter.{utils.Colors.ENDC}")
            return

        prompt = prompts.summarize_and_cross_reference_prompt(personas_for_prompt)
        response = self._query_llm_with_retry(prompt)

        # Index canonicals by normalized key for fast relationship lookups
        canonical_by_norm = {_normalize_persona_text(c.persona_item): c for c in canonicals}

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
                    # Update relationship metadata + related list on the canonical
                    canonical.relationship_type = item.get("relationship_type", canonical.relationship_type)
                    raw_related = item.get("related_personas", [])
                    related = []
                    for r in raw_related:
                        if isinstance(r, dict):
                            related.append(r)
                        elif isinstance(r, str):
                            related.append({"persona_item": r, "type": item.get("relationship_type", "similar")})
                    canonical.related_personas = related

        # --- Compute additive confidence_cross_referenced on top of the merge base ---
        # Index-based ordering: earlier index = older canonical (atomic_personas were
        # sorted early→late, and canonical_order preserves first-seen order)
        persona_order = {canonical.persona_item: idx for idx, canonical in enumerate(canonicals)}
        # Keep the merge-base scores from Step 1 and add on top
        scores: dict[str, float] = {c.persona_item: c.confidence_cross_referenced for c in canonicals}

        for c in canonicals:
            for rel in c.related_personas:
                if not isinstance(rel, dict):
                    continue
                other_item = rel.get("persona_item", "")
                other_key = _normalize_persona_text(other_item)
                rel_type = rel.get("type", "similar")
                if other_key not in canonical_by_norm:
                    continue
                # Guard against self-reference (identical persona_items would be merged —
                # if the LLM still emits a self-loop, ignore it rather than counting it).
                if other_key == _normalize_persona_text(c.persona_item):
                    continue

                if rel_type == "similar":
                    scores[c.persona_item] += 0.1
                elif rel_type == "contradictory":
                    my_idx = persona_order.get(c.persona_item, 0)
                    other_canonical = canonical_by_norm[other_key]
                    other_idx = persona_order.get(other_canonical.persona_item, 0)
                    if my_idx <= other_idx:
                        # I am older — I get penalized
                        scores[c.persona_item] -= 0.1

        # Apply scores (floor at 0.0, ceiling at CROSS_REF_CAP)
        for c in canonicals:
            c.confidence_cross_referenced = round(
                max(0.0, min(CROSS_REF_CAP, scores[c.persona_item])),
                2,
            )

        # --- Step 3: Strict filter on init confidence ---
        # Drop anything with init < MIN_PERSONA_INIT_CONFIDENCE, regardless of
        # cross-ref score or relationship_type.
        self.cross_referenced_personas = [
            c for c in canonicals
            if c.confidence_score_init >= MIN_PERSONA_INIT_CONFIDENCE
        ]

        if self.verbose:
            removed = len(canonicals) - len(self.cross_referenced_personas)
            n_contradictions = sum(1 for p in self.cross_referenced_personas if p.relationship_type == "contradictory")
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] After cross-ref + init>={MIN_PERSONA_INIT_CONFIDENCE} filter: "
                  f"{len(self.cross_referenced_personas)} kept, {removed} dropped, "
                  f"{n_contradictions} contradictory.{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # Post-filter redundancy removal
    # ------------------------------------------------------------------

    def remove_redundant_personas(self) -> None:
        """Collapse semantically-redundant canonical personas after the
        confidence filter has run.

        The merge step in `summarize_and_cross_reference` only combined
        LEXICALLY identical persona_items (e.g. "Enjoys home cooking" ==
        "Enjoys home cooking"). This method operates on the stronger,
        SEMANTIC notion of redundancy: two distinct-wording personas that
        convey essentially the same preference should be collapsed into one.

        Examples of redundant groups:
          - "Enjoys home cooking" + "Likes preparing meals at home" + "Values
             cooking as an act of love and family care"
          - "Follows Detroit Lions" + "Is an NFL fan" + "Supports Detroit
             sports teams" (if the context is clearly Lions-centric)

        The LLM is asked to cluster the surviving personas into redundancy
        groups; Python keeps the *representative* (highest combined score)
        from each group and drops the rest. Related-persona links are
        rewritten to point at the kept representative.

        Subagent mode: the method no-ops here; skill.md instructs the subagent
        to do the equivalent clustering inline before it writes files.
        """
        if self.llm_client is None or not self.cross_referenced_personas:
            return

        candidates = [
            {
                "persona_item": cr.persona_item,
                "category": cr.category,
                "confidence_score_init": cr.confidence_score_init,
                "confidence_cross_referenced": cr.confidence_cross_referenced,
            }
            for cr in self.cross_referenced_personas
        ]
        prompt = prompts.remove_redundant_personas_prompt(candidates)
        response = self._query_llm_with_retry(prompt)
        if not response:
            return
        parsed = utils.extract_json_from_response(response)
        if not isinstance(parsed, list):
            return

        # parsed is expected to be a list of groups, each group is a list of
        # persona_item strings that are semantically redundant. We keep the
        # highest-combined-score item per group, drop the rest, and forward
        # related_personas links from dropped items onto the kept item.
        lookup = {cr.persona_item: cr for cr in self.cross_referenced_personas}
        to_remove: set[str] = set()
        redirect: dict[str, str] = {}  # dropped item -> kept item

        for group in parsed:
            if not isinstance(group, list) or len(group) < 2:
                continue
            # Keep the valid, known ones
            known = [lookup[name] for name in group if name in lookup]
            if len(known) < 2:
                continue
            # Pick the representative: highest (init + cross_ref), ties broken by corroboration_count
            rep = max(
                known,
                key=lambda cr: (
                    cr.confidence_score_init + cr.confidence_cross_referenced,
                    cr.corroboration_count,
                ),
            )
            # Also bump the representative's corroboration_count by the dropped ones'
            # counts, so the cumulative row-backing is preserved.
            extra_corroboration = sum(cr.corroboration_count for cr in known if cr is not rep)
            rep.corroboration_count += extra_corroboration

            for cr in known:
                if cr is rep:
                    continue
                to_remove.add(cr.persona_item)
                redirect[cr.persona_item] = rep.persona_item

        if not to_remove:
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] No semantic redundancies found.{utils.Colors.ENDC}")
            return

        # Rewrite related_personas links on survivors so they don't point at
        # dropped items.
        survivors = [cr for cr in self.cross_referenced_personas if cr.persona_item not in to_remove]
        for cr in survivors:
            new_related = []
            for rel in cr.related_personas:
                if not isinstance(rel, dict):
                    continue
                other = rel.get("persona_item", "")
                if other in to_remove:
                    other = redirect.get(other, "")
                if not other or other == cr.persona_item:
                    continue
                new_related.append({"persona_item": other, "type": rel.get("type", "similar")})
            cr.related_personas = new_related

        self.cross_referenced_personas = survivors

        if self.verbose:
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Removed {len(to_remove)} semantically "
                  f"redundant personas; {len(survivors)} remain.{utils.Colors.ENDC}")

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
    # LLM Call #8: Generate interaction_format objects (with @ai messages
    # for chatbot steering directives)
    # ------------------------------------------------------------------

    def generate_interaction_formats(self) -> None:
        """For each routed preference, generate a concrete interaction_format
        object (action + label + optional user_message).

        In API mode this is one LLM call per preference — expensive at scale,
        so this method is only used when llm_client is set. Subagent mode
        handles this inline.
        """
        if not self.cross_referenced_personas or self.llm_client is None:
            return
        if not self.user_profile or not self.user_profile.app_personas:
            return

        for cr in self.cross_referenced_personas:
            app = cr.assigned_app or random.choice(PLATFORMS)
            app_persona_dict = self.user_profile.app_personas.get(app, {})
            app_formats = PLATFORM_INTERACTION_FORMATS.get(app, {})
            catalog = app_formats.get(cr.source_interaction_type, []) or \
                      app_formats.get("implicit_positive" if "positive" in cr.source_interaction_type else "implicit_negative", [])

            if not catalog:
                cr.source_interaction_format = json.dumps({"app": app, "action": "unknown", "action_label": "Unknown", "user_message": None})
                continue

            # Determine whether an @ai message is required — only for Chatbot
            # negative actions AND a (deterministic) subset of Chatbot positives.
            requires_msg = False
            # We'll let the LLM decide the action first, then check. But the
            # prompt needs to know upfront. Sidestep: in API mode we ALWAYS ask
            # for a user_message on Chatbot and never on others; then post-hoc
            # clear it if the chosen action isn't in AT_AI_ACTIONS.
            requires_msg = (app == "Chatbot")

            prompt = prompts.generate_interaction_format_prompt(
                persona_item=cr.persona_item,
                category=cr.category,
                interaction_type=cr.source_interaction_type,
                assigned_app=app,
                app_persona=app_persona_dict,
                action_catalog=catalog,
                requires_user_message=requires_msg,
            )
            response = self._query_llm_with_retry(prompt)
            format_obj = {"app": app, "action": "unknown", "action_label": "Unknown", "user_message": None}
            if response:
                parsed = utils.extract_json_from_response(response)
                if isinstance(parsed, dict):
                    format_obj = {
                        "app": app,
                        "action": parsed.get("action", "unknown"),
                        "action_label": parsed.get("action_label", ""),
                        "user_message": parsed.get("user_message") if requires_msg else None,
                    }
                    # Only keep user_message for actual @ai actions
                    if format_obj["action"] not in AT_AI_ACTIONS:
                        format_obj["user_message"] = None

            cr.source_interaction_format = json.dumps(format_obj)

        if self.verbose:
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Interaction formats generated.{utils.Colors.ENDC}")

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
          2. dedupe (lexical) + cross-reference + init>=0.8 filter
          3. remove semantic redundancies among survivors (LLM clustering)
          4. temporal contradiction graph (on surviving canonicals)
          5. generate user profile (demographics + big_five + bio)
          6. generate per-app sub-personas
          7. route preferences to apps (LLM + 8% noise)
          8. generate interaction_format objects per preference (with @ai
             messages for Chatbot steering directives)
          9. annotate stereotype marks
         10. build test split (cross-app, global latest-20% high-conf by time)
         11. save to backend/{uid}/ subfolder as per-app JSON files
        """
        print(f"{utils.Colors.BOLD}[User {self.user_id}] Starting persona pipeline...{utils.Colors.ENDC}")

        self.infer_personas_from_hashtags()
        self.summarize_and_cross_reference()
        self.remove_redundant_personas()
        self.build_temporal_contradiction_graph()
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
        """Persist data to 5 JSON files under backend/{user_id}/:

          - profile.json   — UserProfile + AppPersonas (all 4 apps)
          - instagram.json — preferences routed to Instagram (time-sorted)
          - facebook.json  — preferences routed to Facebook (time-sorted)
          - threads.json   — preferences routed to Threads (time-sorted)
          - chatbot.json   — preferences routed to Chatbot (time-sorted, with @ai messages)

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
                "corroboration_count": cr.corroboration_count,
                "relationship_type": cr.relationship_type,
                "related_personas": cr.related_personas,
                "stereotype_mark": ann.stereotype_mark if ann else "neutral",
                "split": split_label,
                "distractor_persona_item": distractor.get("persona_item", ""),
                "distractor_category": distractor.get("category", ""),
                "source_interaction_type": ap.source_interaction_type if ap else cr.source_interaction_type,
                "source_object_id": ap.source_object_id if ap else "",
                "source_timestamp": ap.source_timestamp if ap else 0,
                "formatted_timestamp": ap.formatted_timestamp if ap else cr.formatted_timestamp,
                "source_hashtags": ap.source_hashtags if ap else [],
                "assigned_app": app,
                "interaction_format": fmt,
            }
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
                "corroboration_count": 1,
                "relationship_type": "none",
                "related_personas": [],
                "stereotype_mark": ann.stereotype_mark if ann else "neutral",
                "split": "train",
                "distractor_persona_item": "",
                "distractor_category": "",
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

        # --- Write aggregated preferences.csv (all apps merged, old-style flat layout) ---
        # Re-uses the CSV schema from the pre-refactor era so downstream tools
        # that expect a single-file-per-user view still work. `assigned_app`
        # and `corroboration_count` are appended as new columns. The
        # `interaction_format` cell carries the full JSON object serialized
        # as a string.
        csv_columns = [
            "persona_item",
            "category",
            "confidence_score_init",
            "confidence_cross_referenced",
            "corroboration_count",
            "source_interaction_type",
            "source_object_id",
            "source_timestamp",
            "formatted_timestamp",
            "source_hashtags",
            "assigned_app",
            "interaction_format",
            "relationship_type",
            "related_personas",
            "stereotype_mark",
            "split",
            "distractor_persona_item",
            "distractor_category",
        ]
        csv_rows: list[dict] = []
        for rec in all_records:
            csv_rows.append({
                "persona_item": rec.get("persona_item", ""),
                "category": rec.get("category", ""),
                "confidence_score_init": rec.get("confidence_score_init", 0.0),
                "confidence_cross_referenced": rec.get("confidence_cross_referenced", 0.0),
                "corroboration_count": rec.get("corroboration_count", 1),
                "source_interaction_type": rec.get("source_interaction_type", ""),
                "source_object_id": rec.get("source_object_id", ""),
                "source_timestamp": rec.get("source_timestamp", 0),
                "formatted_timestamp": rec.get("formatted_timestamp", ""),
                "source_hashtags": json.dumps(rec.get("source_hashtags", []), ensure_ascii=False),
                "assigned_app": rec.get("assigned_app", ""),
                "interaction_format": json.dumps(rec.get("interaction_format", {}), ensure_ascii=False),
                "relationship_type": rec.get("relationship_type", "none"),
                "related_personas": json.dumps(rec.get("related_personas", []), ensure_ascii=False),
                "stereotype_mark": rec.get("stereotype_mark", "neutral"),
                "split": rec.get("split", "train"),
                "distractor_persona_item": rec.get("distractor_persona_item", ""),
                "distractor_category": rec.get("distractor_category", ""),
            })
        if csv_rows:
            import csv as _csv
            csv_path = os.path.join(user_dir, "preferences.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = _csv.DictWriter(f, fieldnames=csv_columns)
                writer.writeheader()
                writer.writerows(csv_rows)

        # --- Write profile.json ---
        if self.user_profile:
            profile_dict = asdict(self.user_profile)
            profile_dict["user_id"] = str(self.user_id)
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
                    corroboration_count=int(rec.get("corroboration_count", 1)),
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
                distractor_item = rec.get("distractor_persona_item", "") or ""
                if distractor_item:
                    self.test_distractors[rec["persona_item"]] = {
                        "persona_item": distractor_item,
                        "category": rec.get("distractor_category", "") or "",
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
