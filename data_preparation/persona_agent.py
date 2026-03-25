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
    dataset: str
    ds: str
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
        self.overpersonalization_holdout: list[AnnotatedPersona] = []  # 20% held out for overpersonalization study

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
                dataset=row.get("dataset", ""),
                ds=row.get("ds", ""),
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
        Big Five personality, bio) from the final set of cross-referenced personas.

        Gender and race/ethnicity are randomly sampled from predefined distributions
        and passed to the LLM as constraints. The LLM generates everything else to
        be consistent with the personas — but is explicitly told to be diverse and
        avoid satisfying every persona to prevent stereotyping.
        """
        if not self.cross_referenced_personas:
            self.user_profile = None
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] No personas — skipping profile generation.{utils.Colors.ENDC}")
            return

        # Sample demographics
        sampled_gender_orientation = _sample_from_distribution(GENDER_ORIENTATION_DISTRIBUTION)
        sampled_race = _sample_from_distribution(RACE_ETHNICITY_DISTRIBUTION)

        personas_summary = [p.persona_item for p in self.cross_referenced_personas]
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
        """For each cross-referenced persona, annotate whether it is neutral,
        stereotypical, or anti-stereotypical relative to the user profile."""
        if not self.user_profile or not self.cross_referenced_personas:
            self.annotated_personas = []
            return

        personas_for_prompt = [
            {"persona_item": p.persona_item, "category": p.category}
            for p in self.cross_referenced_personas
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

        # Build lookup from cross-referenced personas for confidence scores
        cr_lookup = {p.persona_item: p for p in self.cross_referenced_personas}

        self.annotated_personas = []
        for item in parsed:
            if not isinstance(item, dict) or "persona_item" not in item:
                continue
            cr = cr_lookup.get(item["persona_item"])
            self.annotated_personas.append(AnnotatedPersona(
                persona_item=item["persona_item"],
                category=item.get("category", cr.category if cr else "uncategorized"),
                confidence_score_init=cr.confidence_score_init if cr else 0.0,
                confidence_cross_referenced=cr.confidence_cross_referenced if cr else 0.0,
                stereotype_mark=item.get("stereotype_mark", "neutral"),
            ))

        if self.verbose:
            counts = {}
            for ap in self.annotated_personas:
                counts[ap.stereotype_mark] = counts.get(ap.stereotype_mark, 0) + 1
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Stereotype annotations: {counts}{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # Overpersonalization holdout
    # ------------------------------------------------------------------

    def extract_overpersonalization_holdout(self, fraction: float = 0.2) -> None:
        """Randomly extract a fraction of annotated personas into a holdout set
        for overpersonalization study. Removed items are taken out of
        self.annotated_personas and placed into self.overpersonalization_holdout."""
        if not self.annotated_personas:
            self.overpersonalization_holdout = []
            return

        n = max(1, int(len(self.annotated_personas) * fraction))
        indices = random.sample(range(len(self.annotated_personas)), min(n, len(self.annotated_personas)))
        indices_set = set(indices)

        self.overpersonalization_holdout = [self.annotated_personas[i] for i in indices]
        self.annotated_personas = [p for i, p in enumerate(self.annotated_personas) if i not in indices_set]

        if self.verbose:
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Overpersonalization holdout: "
                  f"{len(self.overpersonalization_holdout)} held out, {len(self.annotated_personas)} remaining.{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_pipeline(self) -> dict:
        """Run the full persona inference pipeline:
        infer -> cross-ref -> temporal -> profile -> stereotype annotation -> holdout -> save."""
        print(f"{utils.Colors.BOLD}[User {self.user_id}] Starting persona pipeline...{utils.Colors.ENDC}")

        self.infer_personas_from_hashtags()
        self.summarize_and_cross_reference()
        self.build_temporal_contradiction_graph()
        self.generate_user_profile()
        self.annotate_stereotype_marks()
        self.extract_overpersonalization_holdout()
        self.save_to_backend()

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
            "overpersonalization_holdout": len(self.overpersonalization_holdout),
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

        1. {user_id}_preferences.csv — all personas (positive + negative) with both
           raw columns and cross-reference/annotation columns in a single file
        2. {user_id}_profile.csv — synthetic user profile
        """
        # --- (1) Preferences: only personas that passed filtering ---
        # Positive: only cross-referenced personas (already filtered)
        # Negative: only those with confidence > 0.05
        cr_lookup = {p.persona_item: p for p in self.cross_referenced_personas}
        holdout_items = {ap.persona_item for ap in self.overpersonalization_holdout}
        all_annotated_items = {ap.persona_item: ap for ap in self.annotated_personas}
        all_annotated_items.update({ap.persona_item: ap for ap in self.overpersonalization_holdout})

        # Build lookup from atomic personas for raw columns
        atomic_lookup = {ap.persona_item: ap for ap in self.atomic_personas}
        atomic_lookup.update({ap.persona_item: ap for ap in self.negative_personas})

        rows = []
        # Positive personas that survived filtering
        for cr in self.cross_referenced_personas:
            ap = atomic_lookup.get(cr.persona_item)
            ann = all_annotated_items.get(cr.persona_item)
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
                "overpersonalization": "yes" if cr.persona_item in holdout_items else "no",
            })

        # Negative personas with confidence > 0.05
        for np in self.negative_personas:
            if np.confidence_score_init > 0.05:
                rows.append({
                    "persona_item": np.persona_item,
                    "category": np.category,
                    "confidence_score_init": np.confidence_score_init,
                    "source_interaction_type": np.source_interaction_type,
                    "source_object_id": np.source_object_id,
                    "source_timestamp": np.source_timestamp,
                    "formatted_timestamp": np.formatted_timestamp,
                    "source_hashtags": json.dumps(np.source_hashtags),
                    "interaction_format": np.source_interaction_format,
                    "confidence_cross_referenced": 0.0,
                    "relationship_type": "none",
                    "related_personas": "[]",
                    "stereotype_mark": "neutral",
                    "overpersonalization": "no",
                })

        if rows:
            utils.save_rows_to_csv(rows, self._user_path("preferences"))

        # --- (2) Profile ---
        if self.user_profile:
            profile_dict = asdict(self.user_profile)
            profile_dict["big_five"] = json.dumps(profile_dict["big_five"])
            utils.save_rows_to_csv([profile_dict], self._user_path("profile"))

        if self.verbose:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Saved to {self.backend_dir}/ "
                  f"(preferences: {len(all_atomic)}, "
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
        self.overpersonalization_holdout = []

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
                if row.get("overpersonalization") == "yes":
                    self.overpersonalization_holdout.append(ann)
                else:
                    self.annotated_personas.append(ann)

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
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Loaded from backend: "
                  f"{len(self.atomic_personas)} positive, "
                  f"{len(self.negative_personas)} negative, "
                  f"{len(self.annotated_personas)} annotated, "
                  f"{len(self.overpersonalization_holdout)} holdout.{utils.Colors.ENDC}")
        return True
