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
class UserProfile:
    """Synthetic user profile generated from final personas (output of LLM call #4)."""
    name: str = ""
    gender: str = ""
    race_ethnicity: str = ""
    career: str = ""
    education: str = ""
    big_five: dict = field(default_factory=dict)  # {"openness": "...", "conscientiousness": "...", ...}
    bio: str = ""


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

HIGH_CONFIDENCE_INIT_THRESHOLD = 0.5
HIGH_CONFIDENCE_CROSS_REF_THRESHOLD = 0.5


def is_high_confidence(init_score: float, cross_ref_score: float) -> bool:
    """Return True if a persona's scores qualify as 'reasonably high confidence'.

    BOTH conditions must hold:
      - confidence_score_init  >= 0.5  (LLM's raw inference is solid)
      - confidence_cross_referenced > 0.5  (corroborated by >=6 similar cross-row
                                            pairs under the +0.1/similar scoring)
    """
    return (
        init_score >= HIGH_CONFIDENCE_INIT_THRESHOLD
        and cross_ref_score > HIGH_CONFIDENCE_CROSS_REF_THRESHOLD
    )


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

PLATFORM_INTERACTION_FORMATS = {
    "Instagram": {
        "explicit_positive": ["liked", "reposted", "commented", "shared to friend(s)"],
        "implicit_positive": [
            "viewed more than 75% of the reel",
            "stayed on an image for more than 5 seconds",
            "stayed on a story for more than 5 seconds",
        ],
        "explicit_negative": ["disliked"],
        "implicit_negative": ["skipped the reel with no interactions", "skipped the image with no interactions"],
    },
    "Facebook": {
        "explicit_positive": ["liked", "reposted", "commented", "shared to friend(s)"],
        "implicit_positive": [
            "viewed more than 75% of the video",
            "stayed on a post for more than 5 seconds",
            "stayed on an image for more than 5 seconds",
        ],
        "explicit_negative": ["disliked"],
        "implicit_negative": ["skipped the post with no interactions"],
    },
    "Threads": {
        "explicit_positive": ["liked", "reposted", "commented", "shared to friend(s)", "viewed more than 75% of the long post"],
        "implicit_positive": [
            "stayed on a thread for more than 5 seconds",
            "viewed more than 75% of the video",
        ],
        "explicit_negative": ["disliked"],
        "implicit_negative": ["skipped the thread with no interactions"],
    },
    "Chatbot": {
        "explicit_positive": [
            "liked",
            "positive response/comments to strong extent in next turn",
            "asked follow-up questions showing interest",
            "requested more details or recommendations",
            "saved or bookmarked the response",
        ],
        "implicit_positive": [
            "liked",
            "positive response/comments to weak extent in next turn",
            "continued the conversation on the same topic",
            "spent significant time reading the response",
            "copied or referenced part of the response",
        ],
        "explicit_negative": [
            "disliked",
            "negative response/comments to strong extent in next turn",
            "asked to change topic or stop",
            "reported or flagged the response",
        ],
        "implicit_negative": [
            "no active follow up or response",
            "disliked",
            "negative response/comments to weak extent in next turn",
            "abandoned the conversation after the response",
            "immediately changed the topic",
            "gave a minimal or dismissive reply",
        ],
    },
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


def _assign_interaction_format(interaction_type: str) -> str:
    """Randomly assign a platform and interaction format for a given interaction_type.
    Returns a string like 'Instagram: liked' or 'Chatbot (therapy and reflection): liked'."""
    platform = random.choice(PLATFORMS)
    formats = PLATFORM_INTERACTION_FORMATS[platform].get(interaction_type)
    if not formats:
        key = "implicit_positive" if "positive" in interaction_type else "implicit_negative"
        formats = PLATFORM_INTERACTION_FORMATS[platform][key]
    action = random.choice(formats)
    if platform == "Chatbot":
        context = random.choice(CHATBOT_CONTEXTS)
        return f"Chatbot ({context}): {action}"
    return f"{platform}: {action}"


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
        """Load raw CSV dicts for this user, convert to InteractionRow, sort by time."""
        self.interactions = []
        for row in rows:
            itype = row.get("interaction_type", "")
            self.interactions.append(InteractionRow(
                interaction_type=itype,
                user_id=row.get("user_id", str(self.user_id)),
                object_id=row.get("object_id", ""),
                interaction_time=int(row.get("interaction_time", 0)),
                object_text=row.get("object_text", ""),
                interaction_format=row.get("interaction_format") or _assign_interaction_format(itype),
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
        """Cross-reference all atomic personas, score them, and filter.

        Confidence scoring (computed in Python, not by LLM):
        - Similar relationship: +0.1 to BOTH personas
        - Contradictory relationship: -0.1 to the OLDER persona, no change to the newer one
          (the newer/latest preference can still gain confidence from similar cross-validations)
        - confidence_cross_referenced is floored at 0.0
        """
        if not self.atomic_personas:
            self.cross_referenced_personas = []
            return

        # Serialize atomic personas for the prompt (already sorted early→late)
        personas_for_prompt = []
        for ap in self.atomic_personas:
            personas_for_prompt.append({
                "persona_item": ap.persona_item,
                "category": ap.category,
                "confidence_score_init": ap.confidence_score_init,
                "formatted_timestamp": ap.formatted_timestamp,
                "source_object_id": ap.source_object_id,
                "source_interaction_type": ap.source_interaction_type,
                "source_interaction_format": ap.source_interaction_format,
            })

        # If all personas come from a single row, skip LLM cross-reference
        unique_objects = {ap.source_object_id for ap in self.atomic_personas}
        if len(unique_objects) <= 1:
            # Nothing to cross-reference — personas from same row cannot validate each other
            self.cross_referenced_personas = [
                CrossReferencedPersona(
                    persona_item=ap.persona_item,
                    category=ap.category,
                    confidence_score_init=ap.confidence_score_init,
                    confidence_cross_referenced=0.0,
                    relationship_type="none",
                    related_personas=[],
                    formatted_timestamp=ap.formatted_timestamp,
                    source_interaction_type=ap.source_interaction_type,
                    source_interaction_format=ap.source_interaction_format,
                )
                for ap in self.atomic_personas
            ]
            # Filter: remove items with init < 0.5 and no cross-references
            self.cross_referenced_personas = [
                p for p in self.cross_referenced_personas
                if not (p.confidence_score_init < 0.5 and p.confidence_cross_referenced <= 0.0)
            ]
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Single interaction row — skipping cross-reference. "
                      f"{len(self.cross_referenced_personas)} personas kept after filter.{utils.Colors.ENDC}")
            return

        prompt = prompts.summarize_and_cross_reference_prompt(personas_for_prompt)
        response = self._query_llm_with_retry(prompt)

        all_cross_referenced = []
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
                    # Normalize related_personas: accept both old format (list of strings)
                    # and new format (list of {"persona_item": ..., "type": ...})
                    raw_related = item.get("related_personas", [])
                    related = []
                    for r in raw_related:
                        if isinstance(r, dict):
                            related.append(r)
                        elif isinstance(r, str):
                            # Legacy format — infer type from parent's relationship_type
                            related.append({"persona_item": r, "type": item.get("relationship_type", "similar")})
                    all_cross_referenced.append(CrossReferencedPersona(
                        persona_item=item["persona_item"],
                        category=item.get("category", "uncategorized"),
                        confidence_score_init=float(item.get("confidence_score_init", 0.3)),
                        confidence_cross_referenced=0.0,  # computed below
                        relationship_type=item.get("relationship_type", "none"),
                        related_personas=related,
                        formatted_timestamp=item.get("formatted_timestamp", ""),
                        source_interaction_type=item.get("source_interaction_type", ""),
                        source_interaction_format=item.get("source_interaction_format", ""),
                    ))

        # --- Compute confidence_cross_referenced in Python ---
        # Build index-based ordering: earlier index = older persona (interactions sorted early→late)
        persona_order = {p.persona_item: idx for idx, p in enumerate(all_cross_referenced)}
        scores = {p.persona_item: 0.0 for p in all_cross_referenced}

        # Build lookup for source_object_id to enforce cross-row only
        # Use the original atomic personas which have source_object_id
        atomic_obj_lookup = {ap.persona_item: ap.source_object_id for ap in self.atomic_personas}

        for p in all_cross_referenced:
            my_obj = atomic_obj_lookup.get(p.persona_item, "")
            for rel in p.related_personas:
                if not isinstance(rel, dict):
                    continue
                other_item = rel.get("persona_item", "")
                rel_type = rel.get("type", "similar")
                if other_item not in persona_order:
                    continue

                # Skip same-row relationships — personas from the same interaction
                # row share evidence and cannot cross-validate each other
                other_obj = atomic_obj_lookup.get(other_item, "")
                if my_obj and other_obj and my_obj == other_obj:
                    continue

                if rel_type == "similar":
                    scores[p.persona_item] += 0.1
                elif rel_type == "contradictory":
                    # Subtract from the OLDER one, leave the newer one unchanged
                    my_idx = persona_order.get(p.persona_item, 0)
                    other_idx = persona_order.get(other_item, 0)
                    if my_idx <= other_idx:
                        # I am older or same age — I get penalized
                        scores[p.persona_item] -= 0.1
                    # If I am newer, no change to me (the older one gets penalized
                    # when we process its relationships)

        # Apply scores (floor at 0.0) and deduplicate similar counting
        # Since relationships are listed per-persona, each similar pair is counted once per side
        # which is the intended behavior (+0.1 to each side)
        for p in all_cross_referenced:
            p.confidence_cross_referenced = max(0.0, round(scores[p.persona_item], 2))

        # Filter: remove items with confidence_score_init < 0.5 AND confidence_cross_referenced <= 0.0
        # But KEEP contradictions regardless (they go into temporal graph)
        self.cross_referenced_personas = [
            p for p in all_cross_referenced
            if not (p.confidence_score_init < 0.5 and p.confidence_cross_referenced <= 0.0)
            or p.relationship_type == "contradictory"
        ]

        if self.verbose:
            removed = len(all_cross_referenced) - len(self.cross_referenced_personas)
            n_contradictions = sum(1 for p in self.cross_referenced_personas if p.relationship_type == "contradictory")
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Cross-referenced: {len(self.cross_referenced_personas)} kept, "
                  f"{removed} filtered, {n_contradictions} contradictory.{utils.Colors.ENDC}")

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
        """Run the full persona inference pipeline:
        infer -> cross-ref -> temporal -> profile -> stereotype -> test split -> save."""
        print(f"{utils.Colors.BOLD}[User {self.user_id}] Starting persona pipeline...{utils.Colors.ENDC}")

        self.infer_personas_from_hashtags()
        self.summarize_and_cross_reference()
        self.build_temporal_contradiction_graph()
        self.generate_user_profile()
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

    def _user_path(self, suffix: str) -> str:
        return os.path.join(self.backend_dir, f"{self.user_id}_{suffix}.csv")

    def save_to_backend(self) -> str:
        """Persist data to 2 CSV files per user:

        1. {user_id}_preferences.csv — all personas (positive + negative) sorted
           strictly chronologically by source_timestamp. Positive rows carry a
           `split` label ("train" | "test") from self.split_labels; negative rows
           are always train. Test rows also carry their paired distractor fields.
        2. {user_id}_profile.csv — synthetic user profile.
        """
        # --- Build lookups ---
        all_annotated_items = {ap.persona_item: ap for ap in self.annotated_personas}

        # Raw per-persona metadata (source_timestamp, hashtags, etc.)
        atomic_lookup = {ap.persona_item: ap for ap in self.atomic_personas}
        atomic_lookup.update({ap.persona_item: ap for ap in self.negative_personas})

        rows: list[dict] = []

        # Positive personas that survived filtering + test-split gate
        for cr in self.cross_referenced_personas:
            ap = atomic_lookup.get(cr.persona_item)
            ann = all_annotated_items.get(cr.persona_item)
            split_label = self.split_labels.get(cr.persona_item, "train")
            distractor = self.test_distractors.get(cr.persona_item, {}) if split_label == "test" else {}
            rows.append({
                "persona_item": cr.persona_item,
                "category": cr.category,
                "confidence_score_init": cr.confidence_score_init,
                "source_interaction_type": ap.source_interaction_type if ap else cr.source_interaction_type,
                "source_object_id": ap.source_object_id if ap else "",
                "source_timestamp": ap.source_timestamp if ap else 0,
                "formatted_timestamp": ap.formatted_timestamp if ap else cr.formatted_timestamp,
                "source_hashtags": json.dumps(ap.source_hashtags if ap else []),
                "interaction_format": ap.source_interaction_format if ap else cr.source_interaction_format,
                "confidence_cross_referenced": cr.confidence_cross_referenced,
                "relationship_type": cr.relationship_type,
                "related_personas": json.dumps(cr.related_personas),
                "stereotype_mark": ann.stereotype_mark if ann else "neutral",
                "split": split_label,
                "distractor_persona_item": distractor.get("persona_item", ""),
                "distractor_category": distractor.get("category", ""),
            })

        # Negative personas with confidence > 0.05 — always train
        for np_persona in self.negative_personas:
            if np_persona.confidence_score_init > 0.05:
                ann = all_annotated_items.get(np_persona.persona_item)
                rows.append({
                    "persona_item": np_persona.persona_item,
                    "category": np_persona.category,
                    "confidence_score_init": np_persona.confidence_score_init,
                    "source_interaction_type": np_persona.source_interaction_type,
                    "source_object_id": np_persona.source_object_id,
                    "source_timestamp": np_persona.source_timestamp,
                    "formatted_timestamp": np_persona.formatted_timestamp,
                    "source_hashtags": json.dumps(np_persona.source_hashtags),
                    "interaction_format": np_persona.source_interaction_format,
                    "confidence_cross_referenced": 0.0,
                    "relationship_type": "none",
                    "related_personas": "[]",
                    "stereotype_mark": ann.stereotype_mark if ann else "neutral",
                    "split": "train",
                    "distractor_persona_item": "",
                    "distractor_category": "",
                })

        # Sort strictly chronological (early -> latest) across the combined stream
        rows.sort(key=lambda r: (int(r.get("source_timestamp") or 0), r.get("persona_item", "")))

        if rows:
            utils.save_rows_to_csv(rows, self._user_path("preferences"))

        # --- (2) Profile ---
        if self.user_profile:
            profile_dict = asdict(self.user_profile)
            profile_dict["big_five"] = json.dumps(profile_dict["big_five"])
            utils.save_rows_to_csv([profile_dict], self._user_path("profile"))

        if self.verbose:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Saved to {self.backend_dir}/ "
                  f"(preferences: {len(rows)}, "
                  f"profile: {'yes' if self.user_profile else 'no'}){utils.Colors.ENDC}")
        return self.backend_dir

    def load_from_backend(self) -> bool:
        """Load persisted CSV data back into instance variables. Returns True if data found."""
        prefs_path = self._user_path("preferences")
        if not os.path.exists(prefs_path):
            return False

        pref_rows = utils.load_rows_from_csv(prefs_path)
        self.atomic_personas = []
        self.negative_personas = []
        self.cross_referenced_personas = []
        self.annotated_personas = []
        self.split_labels = {}
        self.test_distractors = {}

        for row in pref_rows:
            interaction_type = row.get("source_interaction_type", "")
            is_negative = "negative" in interaction_type

            # Reconstruct AtomicPersona
            ap = AtomicPersona(
                persona_item=row["persona_item"],
                category=row.get("category", "uncategorized"),
                confidence_score_init=float(row["confidence_score_init"]),
                source_interaction_type=interaction_type,
                source_interaction_format=row.get("interaction_format", ""),
                source_object_id=row.get("source_object_id", ""),
                source_timestamp=int(row.get("source_timestamp", 0)),
                formatted_timestamp=row.get("formatted_timestamp", ""),
                source_hashtags=json.loads(row.get("source_hashtags", "[]")),
            )
            if is_negative:
                self.negative_personas.append(ap)
            else:
                self.atomic_personas.append(ap)

                # Reconstruct CrossReferencedPersona for positive rows
                cr = CrossReferencedPersona(
                    persona_item=row["persona_item"],
                    category=row.get("category", "uncategorized"),
                    confidence_score_init=float(row.get("confidence_score_init", 0.0)),
                    confidence_cross_referenced=float(row.get("confidence_cross_referenced", 0.0)),
                    relationship_type=row.get("relationship_type", "none"),
                    related_personas=json.loads(row.get("related_personas", "[]")),
                    formatted_timestamp=row.get("formatted_timestamp", ""),
                    source_interaction_type=interaction_type,
                    source_interaction_format=row.get("interaction_format", ""),
                )
                self.cross_referenced_personas.append(cr)

                # Reconstruct AnnotatedPersona
                ann = AnnotatedPersona(
                    persona_item=row["persona_item"],
                    category=row.get("category", "uncategorized"),
                    confidence_score_init=float(row.get("confidence_score_init", 0.0)),
                    confidence_cross_referenced=float(row.get("confidence_cross_referenced", 0.0)),
                    stereotype_mark=row.get("stereotype_mark", "neutral"),
                )
                self.annotated_personas.append(ann)

            # Split + distractor state
            split_label = row.get("split", "train") or "train"
            self.split_labels[row["persona_item"]] = split_label
            if split_label == "test":
                distractor_item = row.get("distractor_persona_item", "") or ""
                if distractor_item:
                    self.test_distractors[row["persona_item"]] = {
                        "persona_item": distractor_item,
                        "category": row.get("distractor_category", "") or "",
                    }

        # Load user profile
        profile_rows = utils.load_rows_from_csv(self._user_path("profile"))
        if profile_rows:
            row = profile_rows[0]
            self.user_profile = UserProfile(
                name=row.get("name", ""),
                gender=row.get("gender", ""),
                race_ethnicity=row.get("race_ethnicity", ""),
                career=row.get("career", ""),
                education=row.get("education", ""),
                big_five=json.loads(row.get("big_five", "{}")),
                bio=row.get("bio", ""),
            )
        else:
            self.user_profile = None

        if self.verbose:
            n_test = sum(1 for v in self.split_labels.values() if v == "test")
            n_train = sum(1 for v in self.split_labels.values() if v == "train")
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Loaded from backend: "
                  f"{len(self.atomic_personas)} positive, "
                  f"{len(self.negative_personas)} negative, "
                  f"{len(self.annotated_personas)} annotated, "
                  f"{n_train} train, {n_test} test.{utils.Colors.ENDC}")
        return True
