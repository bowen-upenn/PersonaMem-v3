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
class UserVoice:
    """The user's natural writing voice — ONE per user, shared across apps.

    Modeled in four layers so coherence (Layers 1+2) survives across every
    generated text and modulation (Layers 3+4) lands only where audience
    pressure makes it. The deeper layers are the "fingerprint" — what
    survives paraphrase per stylometry research; the shallow layers are
    audience-driven and live on AppPersona.

      Layer 1 — Identity Spine (here, `identity_spine`): WHO this person is.
      Layer 2 — Idiolect (here, `idiolect`): HOW they structure language.
      Layer 3 — Indexical Repertoire (here, `repertoire`): the inventory of
                stances/registers/genres they CAN deploy. Per-app picks a subset.
      Layer 4 — Surface Modulation: lives on AppPersona.surface.

    The "soft holdovers" (natural_register, capitalization, palette, etc.)
    are descriptive surface summaries derived from Layers 1–3, not mimic
    targets. Negatives axis (voice_avoid + phrases_to_avoid) is preserved
    so downstream LLMs don't reach for plausible-but-off-brand language.
    """
    # --- Layer 1 — Identity Spine (stable, never modulates) -------------
    # dict with keys:
    #   agency_communion: str (1 sentence)
    #   redemption_motifs: list[str] (1–3 short noun phrases; each cites a
    #     hidden_persona label or exemplar persona item)
    #   contamination_motifs: list[str] (0–2)
    #   life_stage_preoccupations: list[str] (2–3)
    #   signature_concerns: list[str] (2–4 abstract concerns)
    #   liwc_anchors: dict {analytic, clout, authentic, emotional_tone}
    #   big_five_drivers: dict {trait: "level → behavioral implication"}
    identity_spine: dict = field(default_factory=dict)

    # --- Layer 2 — Idiolect (stable, slow drift; survives paraphrase) ---
    # dict with keys:
    #   function_word_profile: str (1 sentence)
    #   syntactic_preferences: dict {sentence_length_shape, clause_embedding,
    #     parataxis_hypotaxis, fragment_use}
    #   hedge_booster_ratio: "hedge_dominant" | "balanced" | "booster_dominant"
    #   appraisal_fingerprint: dict {attitude_dominant, engagement_style,
    #     graduation}
    #   constructional_templates: list[dict] (2–4) with keys
    #     {pattern, example_realization, frequency}
    #   catchphrase_residue: list[str] (0–2; default []; "ZERO is the right
    #     answer for most users" — replaces the old personal_phrases attractor)
    idiolect: dict = field(default_factory=dict)

    # --- Layer 3 — Indexical Repertoire (stable inventory) --------------
    # dict with keys:
    #   stances: list[str] (3–6 short labels: "deadpan-affectionate", ...)
    #   registers: list[str] (2–4)
    #   backstage_frontstage_range: str (1 sentence)
    #   speech_genre_fluency: list[str] (2–4)
    repertoire: dict = field(default_factory=dict)

    # --- Soft holdovers (descriptive surface summaries) -----------------
    natural_register: str = ""              # e.g. "casual conversational with deadpan humor"
    default_capitalization: str = ""        # "all_lowercase" | "sentence_case" | "mixed_with_caps_for_emphasis"
    punctuation_habits: str = ""            # 1 sentence — concrete habits, not enum
    humor_tone: str = ""                    # e.g. "deadpan, dry, occasional softness"
    emoji_palette: list[str] = field(default_factory=list)   # 5–12 emojis the user genuinely uses
    emoji_intensity_default: str = "medium"  # "low" | "medium" | "high"
    formality_baseline: float = 0.3         # 0.0 (super casual) — 1.0 (very formal)

    # --- Negatives axis (preserved) -------------------------------------
    voice_avoid: str = ""                   # 1–2 sentence prose: tones / styles / habits this user avoids
    phrases_to_avoid: list[str] = field(default_factory=list)  # 0–5 short literal phrases that would feel off-brand


@dataclass
class AppPersona:
    """How a single user presents themselves and engages on ONE specific app.

    Generated per-user for each of the four supported apps (Instagram, Facebook,
    Threads, Chatbot) AFTER the base UserProfile is created. Drives the
    non-random app routing of preferences: each surviving preference is
    assigned to the app whose AppPersona use_purposes it best matches.

    Voice mechanics (Layers 1–3) live in UserProfile.user_voice — the same
    shared voice across all four apps. AppPersona carries:
      • Layer-3 SELECTION — `active_stances` / `active_registers` /
        `active_speech_genres` are SUBSETS of the repertoire on user_voice;
        validation rule: set(active_*) ⊆ set(repertoire.*).
      • Layer-4 SURFACE MODULATION — `surface` (effort, length band, emoji
        intensity shift, disclosure depth, etc.).
      • Audience framing — `audience_design_note` (Bell's audience design),
        `audience_lens`, `friend_zones`, `topical_focus`.
      • `idiolect_overrides` — RARE escape hatch for genuine code-switching
        (default {}; populate only when source rows show it).

    `app_avoid` carries the audience-driven negative axis for THIS app
    specifically. `delta_summary` (≤1 sentence) says WHY this audience
    selects this stance subset — NOT what voice mechanics look like.
    """
    app_name: str                                          # "Instagram" | "Facebook" | "Threads" | "Chatbot"

    # --- Layer-3 selection (subsets of user_voice.repertoire.*) ---------
    active_stances: list[str] = field(default_factory=list)
    active_registers: list[str] = field(default_factory=list)
    active_speech_genres: list[str] = field(default_factory=list)

    # --- Audience framing -----------------------------------------------
    use_purposes: list[str] = field(default_factory=list)  # e.g. ["close friends sharing", "aesthetic personal brand"]
    friend_zones: list[str] = field(default_factory=list)  # e.g. ["close friends", "family", "acquaintances"]
    audience_type: str = "mixed"                           # "private" | "public" | "mixed"
    audience_lens: str = ""                                # 1 sentence: WHO is realistically reading here
    audience_design_note: str = ""                         # 1 sentence in Bell's terms (addressee/auditor/overhearer)
    posting_frequency: str = "weekly"                      # "daily" | "weekly" | "rarely" | "passive viewer only"
    topical_focus: list[str] = field(default_factory=list) # 3-5 domains — subset filter for THIS audience
    chatbot_contexts: list[str] = field(default_factory=list)  # Chatbot only; picked from CHATBOT_CONTEXTS

    # --- Layer-4 surface modulation -------------------------------------
    # dict with required keys: effort_level, length_band, emoji_intensity_shift,
    # audience_self_censoring, disclosure_depth ("low"|"medium"|"high").
    # Optional: emoji_topic_filter.
    surface: dict = field(default_factory=dict)

    # --- Layer-2 deviation hatch (RARE) ---------------------------------
    # Default {} for most users on most apps; only populated when source data
    # shows a genuine code-switch. Keys: capitalization, extra_phrases (0-3),
    # extra_forbidden (0-3), punctuation_shift.
    idiolect_overrides: dict = field(default_factory=dict)

    # --- Negatives + per-app why ----------------------------------------
    app_avoid: str = ""           # audience-driven content/tone the user skips on THIS app
    delta_summary: str = ""       # ≤1 sentence: WHY this audience selects this stance subset


@dataclass
class RomanticSpecifier:
    """Multi-axis sub-typing for the `romantic_partner` AI Studio archetype.

    Filled in by Step 11C only when archetype = "romantic_partner". Each axis
    is independent — a user can be matched to (e.g.) a goth nonbinary
    mommy-coded equal-partner, or a butch lesbian pet-coded sub. The LLM
    picks one value per axis (or `None` if the user's profile gives no
    signal). All vocabularies are closed sets — see field comments below.

    The §1E generation safety floor still applies on top of every axis
    combination: never age-ambiguous, never validates self-harm, never
    role-plays minors regardless of fictional framing, never depicts
    non-consensual scenarios as the user's-fantasy default.
    """
    # Closed vocabulary: "male" | "female" | "nonbinary" | "trans_fem"
    # | "trans_masc" | "genderfluid" | "agender"
    gender_presentation: Optional[str] = None
    # "straight" | "gay_mm" | "lesbian_ff" | "bi" | "pan" | "ace_romantic"
    # | "queer_unspecified"
    sexuality_orientation: Optional[str] = None
    # "goth" | "soft" | "punk" | "preppy" | "alt" | "sporty"
    # | "academic" | "dark_academia" | "hot_nerd"
    # | "glam" | "cottagecore" | "y2k" | "minimalist"
    # | "e_girl" | "e_boy"
    aesthetic_vibe: Optional[str] = None
    # "butch" | "femme" | "twink" | "femboy" | "bear" | "otter" | "jock"
    # | "androgynous" | "bara"
    body_role_coding: Optional[str] = None
    # "equal_partner" | "dom_gentle" | "dom_strict" | "sub_eager"
    # | "sub_bratty" | "switch" | "top" | "bottom" | "vers"
    # | "pet" | "owner_handler" | "mommy" | "daddy_domme" | "sir"
    # | "elder_sis_romantic" | "elder_bro_romantic"
    relational_dynamic: Optional[str] = None
    # "soft_affection" | "sensual" | "erotic_explicit"
    # `erotic_explicit` only when the user's intimate_interest profile passes
    # the adult-signal predicate AND profile age signal is unambiguous adult.
    explicitness_band: str = "sensual"


# Closed vocabularies for RomanticSpecifier validation.
ROMANTIC_GENDER_PRESENTATIONS = frozenset({
    "male", "female", "nonbinary", "trans_fem", "trans_masc",
    "genderfluid", "agender",
})
ROMANTIC_SEXUALITY_ORIENTATIONS = frozenset({
    "straight", "gay_mm", "lesbian_ff", "bi", "pan", "ace_romantic",
    "queer_unspecified",
})
ROMANTIC_AESTHETIC_VIBES = frozenset({
    "goth", "soft", "punk", "preppy", "alt", "sporty",
    "academic", "dark_academia", "hot_nerd",
    "glam", "cottagecore", "y2k", "minimalist",
    "e_girl", "e_boy",
})
ROMANTIC_BODY_ROLE_CODINGS = frozenset({
    "butch", "femme", "twink", "femboy", "bear", "otter", "jock",
    "androgynous", "bara",
})
ROMANTIC_RELATIONAL_DYNAMICS = frozenset({
    "equal_partner", "dom_gentle", "dom_strict", "sub_eager", "sub_bratty",
    "switch", "top", "bottom", "vers",
    "pet", "owner_handler", "mommy", "daddy_domme", "sir",
    "elder_sis_romantic", "elder_bro_romantic",
})
ROMANTIC_EXPLICITNESS_BANDS = frozenset({
    "soft_affection", "sensual", "erotic_explicit",
})


@dataclass
class AIStudioPersona:
    """The user's chosen AI persona on the AI Studio (5th) app.

    ONE per user, picked by Step 11C (`generate_ai_studio_persona`) based on
    the user's hidden personas, hashtag clusters, and identity signals. The
    AI's voice on AI Studio comes from THIS block (not from user_voice — the
    user's voice still drives user turns; the AI persona has its own voice).

    Voice is modeled in the SAME 4-layer structure as `UserVoice`, but the
    layers describe a *fictional character* (Rowan, Wren, etc.) rather than
    a real person. Layer 1 (Identity Spine) and Layer 2 (Idiolect) are the
    "fingerprint" — what survives paraphrase, derived from the chosen
    archetype's character DNA. Layer 3 (Repertoire) is the inventory of
    stances/registers/genres the character CAN deploy. Layer 4 lives in the
    soft holdovers (register / capitalization / emoji palette / etc.).

    Archetype is one of the 10 entries in AI_STUDIO_ARCHETYPES. Two optional
    sub-typing blocks fire only for specific archetypes:
      • `niche_specifier` — only for `niche_expert_creator_ai`. Identifies
        the niche the AI is an expert in (travel-planner-EU,
        fitness-coach-strength, food-mood-pairer-comfort, etc.). Picked by
        the LLM from the user's dominant hashtag clusters.
      • `romantic_specifier` — only for `romantic_partner`. Multi-axis
        sub-typing. See `RomanticSpecifier`.
    """
    # --- Archetype + character DNA -------------------------------------
    persona_archetype: str = ""              # one of AI_STUDIO_ARCHETYPES keys
    character_name: str = ""                 # fictional only — never a real public figure
    backstory_brief: str = ""                # 2-3 sentences — character bio

    relational_stance: str = ""              # 1-2 sentences: how this character relates to the user
    address_terms: list[str] = field(default_factory=list)  # ≤3, e.g. ["love"], ["friend"]
    self_reference_style: str = "first_person"   # "first_person" | "third_person_character" | "mixed"
    communication_style: str = ""            # 1-2 sentence summary of the 4 layers below

    # --- Layer 1 — Character Identity Spine (stable, defines DNA) ------
    # dict with keys (mirrors UserVoice.identity_spine):
    #   agency_communion: str (1 sentence — character's stance toward user/world)
    #   redemption_motifs: list[str] (1-3 short noun phrases — character's healing/uplift themes)
    #   contamination_motifs: list[str] (0-2 — character's wounds / what they fear)
    #   life_stage_preoccupations: list[str] (2-3 — character's developmental focus)
    #   signature_concerns: list[str] (2-4 abstract concerns the character cares about)
    #   liwc_anchors_inferred: dict {analytic, clout, authentic, emotional_tone}
    #   big_five_proxy: dict {trait: "level → behavioral implication"}
    identity_spine: dict = field(default_factory=dict)

    # --- Layer 2 — Character Idiolect (HOW they structure language) ----
    # dict with keys (mirrors UserVoice.idiolect):
    #   function_word_profile: str (1 sentence)
    #   syntactic_preferences: dict {sentence_length_shape, clause_embedding,
    #     parataxis_hypotaxis, fragment_use}
    #   hedge_booster_ratio: "hedge_dominant" | "balanced" | "booster_dominant"
    #   appraisal_fingerprint: dict {attitude_dominant, engagement_style, graduation}
    #   constructional_templates: list[dict] (2-4) with keys
    #     {pattern, example_realization, frequency}
    #   catchphrase_residue: list[str] (0-3 — same role as user's signature_phrases;
    #                                   used ≤1× per conversation)
    idiolect: dict = field(default_factory=dict)

    # --- Layer 3 — Character Repertoire (stable inventory) -------------
    # dict with keys (mirrors UserVoice.repertoire):
    #   stances: list[str] (3-6 short labels — what stances the character CAN deploy)
    #   registers: list[str] (2-4 — what registers the character CAN move through)
    #   backstage_frontstage_range: str (1 sentence)
    #   speech_genre_fluency: list[str] (2-4)
    repertoire: dict = field(default_factory=dict)

    # --- Soft holdovers (descriptive surface summaries) ----------------
    natural_register: str = ""               # 1 phrase, e.g. "warm casual with dry edge"
    default_capitalization: str = ""         # "all_lowercase" | "sentence_case" | "mixed_with_caps_for_emphasis"
    punctuation_habits: str = ""             # 1 sentence — concrete habits
    humor_tone: str = ""                     # 1 phrase, e.g. "wry, lightly teasing"
    length_band: str = "medium"              # "short" | "medium" | "long"
    emoji_palette: list[str] = field(default_factory=list)  # 0-6 (most archetypes use 0-3)
    emoji_intensity_default: str = "low"     # "none" | "low" | "medium"
    formality: float = 0.3                   # 0.0 (casual) – 1.0 (formal)

    # --- Negatives axis (preserved) ------------------------------------
    voice_avoid: str = ""                    # 1-2 sentences: tones/styles/habits this character avoids
    forbidden_phrases: list[str] = field(default_factory=list)
                                             # baseline = Rogers cliché blocklist + archetype-specific

    # --- Topical scope -------------------------------------------------
    topical_strengths: list[str] = field(default_factory=list)  # 3-6
    topical_avoid: list[str] = field(default_factory=list)      # 0-3

    # --- Signature phrases (1-3, used ≤1× per conversation) ------------
    # Kept as a top-level convenience field; mirrors what catchphrase_residue
    # captures structurally inside idiolect. Same content, two access points.
    signature_phrases: list[str] = field(default_factory=list)

    # --- Generation guardrails — keeps generation on-rails; NOT evaluated.
    generation_guardrails: dict = field(default_factory=dict)

    # --- Routing knobs consumed by Steps 13/14 (milestones (b)/(c)). ---
    eligibility_signal: dict = field(default_factory=dict)

    # --- Fit + sub-typing ---------------------------------------------
    fit_rationale: str = ""                  # 1-2 sentences: why THIS archetype for THIS user
    niche_specifier: Optional[str] = None    # required when archetype == "niche_expert_creator_ai"
    romantic_specifier: dict = field(default_factory=dict)  # required when archetype == "romantic_partner"


@dataclass
class HiddenPersona:
    """A deeper motivational layer inferred from cross-row hashtag clustering.

    Hidden personas are the 'why' behind surface-level preferences — personality
    traits, aspirations, emotional patterns, identity anchors, intimate interests,
    and private hobbies that explain observable engagement but are not captured by
    individual-row inference.
    """
    label: str                                                              # e.g., "Romantic vulnerability and yearning"
    type: str = ""                                                          # personality_trait | aspiration | emotional_pattern | identity_anchor | intimate_interest | intellectual_curiosity | private_hobby | parasocial_attachment | compensatory_need | covert_concern | medical_aesthetic_concern | sensitive_life_event
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
    # Earliest / latest source_timestamp across this cluster's evidence rows.
    # Derived from `evidence_oids`. 0 when the cluster has no real backing rows
    # (synthetic injections — see `events`).
    first_seen_ts: int = 0
    last_seen_ts: int = 0
    # True iff this cluster was injected by the pipeline rather than discovered
    # from real engagement. Currently only `sensitive_life_event` clusters use
    # this — they're seeded so every user has private/sensitive ground truth
    # available to the over_personalization_sensitive_event eval.
    is_synthetic: bool = False
    # For `sensitive_life_event` only: a list of 1–3 discrete personal episodes
    # the user is currently navigating. Each entry:
    #   {"topic", "label_fragment", "first_seen_ts", "last_seen_ts",
    #    "active_window_end", "evidence_hashtags", "exemplar_persona_items"}
    # Other types leave this empty.
    events: list = field(default_factory=list)
    # Motivation-audit rollup attached by Step 23
    # (aggregate_motivation_audit_to_summary). Carries cluster_status
    # (validated / mixed_evidence / contested / likely_invalid /
    # synthetic_skipped / unaudited / no_audited_preferences), confirm_rate,
    # deep_latent_rate, surface_share, and per-decision counts. Never
    # mutates the cluster; advisory only.
    motivation_audit: dict = field(default_factory=dict)


# Validation thresholds for hidden persona inference
MIN_HIDDEN_PERSONA_ROWS = 40       # Minimum distinct source rows for a cluster to survive
MIN_HIDDEN_PERSONA_DAYS = 3        # Minimum temporal spread in distinct calendar days
HIDDEN_PERSONA_HASHTAG_MIN_FREQ = 3  # Minimum total occurrences for a hashtag to be considered
HIDDEN_PERSONA_TOP_HASHTAGS = 200  # Number of top hashtags passed to LLM
# Reduced floors for medical_aesthetic_concern clusters whose evidence overlaps
# the LLM-flagged medical hashtag set (Phase 1b). Active medical/aesthetic-
# medicine signals are structurally rare (a steady tretinoin or GLP-1 regimen
# may produce only a handful of weekly engagements) but high-stakes for the
# downstream chatbot personalization that needs to factor them in subtly.
MIN_HIDDEN_PERSONA_ROWS_MEDICAL = 15
MIN_HIDDEN_PERSONA_DAYS_MEDICAL = 2

# Motivation audit (Step 22) thresholds. The audit re-judges every hashtag-
# overlap-linked (preference, hidden_persona) pair against named academic
# motivation frames, producing structured corrections (CONFIRMED / REASSIGN
# / SURFACE_ENGAGEMENT / SHORT_TERM_EPISODIC / REMOVE / NO_OTHER_CLUSTER_FITS
# / FLAG) plus a `motivation_depth` rating. Parsimony-biased: when signal is
# ambiguous, prefer SURFACE_ENGAGEMENT over force-fitting to the closest
# cluster.
MOTIVATION_AUDIT_BATCH_SIZE = 8                    # preferences per LLM call
MOTIVATION_AUDIT_MIN_CONFIRM_CONFIDENCE = 0.6      # CONFIRMED requires fit_confidence >= this
MOTIVATION_AUDIT_PROTECTED_REMOVE_FLOOR = 0.3      # protected prefs require fit_confidence < this to REMOVE
MOTIVATION_AUDIT_DECOY_BIAS_THRESHOLD = 0.20       # decoy-CONFIRM rate above this fails the batch
MOTIVATION_AUDIT_DECOYS_PER_BATCH = 1              # 1 decoy mixed in per batch (drawn from a different cluster)
MOTIVATION_AUDIT_USER_OVER_ATTRIBUTION_RATE = 0.40 # mean cluster surface_share above this triggers user-level warning
# Cluster types that are stable identity/trait — short-term-horizon
# preferences cannot CONFIRM into these without the deterministic
# depth-vs-horizon validator demoting them.
MOTIVATION_AUDIT_STABLE_TRAIT_TYPES = frozenset({
    "personality_trait",
    "aspiration",
    "identity_anchor",
    "parasocial_attachment",
    "private_hobby",
})
# Closed enum of frames the audit may invoke. Validated post-hoc.
MOTIVATION_AUDIT_DEEP_FRAMES = frozenset({
    "self_determination_theory:relatedness",
    "self_determination_theory:autonomy",
    "self_determination_theory:competence",
    "goffman:back_stage",
    "uses_and_gratifications:identity",
    "uses_and_gratifications:integration",
    "kardefelt_winther:compensatory_use",
    "higgins:ideal_self",
    "higgins:ought_self",
    "horton_wohl:parasocial",
    "lazarus_folkman:emotion_focused_coping",
    "csikszentmihalyi:flow",
    "berlyne:specific_curiosity",
    "barthes:punctum",
    "tajfel:social_identity",
    "stryker:role_identity",
    "health_belief_model:active_use",
})
MOTIVATION_AUDIT_SURFACE_FRAMES = frozenset({
    "tversky_kahneman:salience_availability",
    "bikhchandani:informational_cascade",
    "berlyne:diversive_curiosity",
    "schwarz:mood_as_information",
    "variable_ratio_reinforcement",
    "algorithmic_surfacing",
    "short_term_episodic_event",
    "none",
})
# Generic-token blocklists for type-specific specificity validators.
# Lowercase, substring match against the preference's persona_item.
_INTIMATE_GENERIC_BLOCKLIST = frozenset({
    "suggestive content", "attractive content", "sexy content",
    "thirst content", "adult content", "nsfw content",
})
_COVERT_CONCERN_GENERIC_BLOCKLIST = frozenset({
    "worries about money", "worries about health", "worries about career",
    "general anxiety", "stress in general",
})
# Active-use verbs/markers that grant medical_aesthetic_concern CONFIRM.
_MEDICAL_ACTIVE_USE_MARKERS = frozenset({
    " takes ", " taking ", " using ", " on ", " applies ", " applied ",
    " applying ", " started ", " on a regimen ", " prescribed ", " uses ",
})

# Frame-resolution helpers (`_TYPE_DEFAULT_FRAME` and
# `cluster_dominant_frame`) live in `prompts.py` so all consumers
# (persona_agent, extension_b, chatbot_conversation, evaluation) can
# share one source without circular imports.


# ---------------------------------------------------------------------------
# Sensitive-life-event injection (Step 9b).
#
# After the LLM-driven hidden persona discovery, every user gets a single
# synthetic `sensitive_life_event` cluster bundling 1–3 discrete personal
# episodes drawn deterministically (seeded by user_id) from this catalog.
# Distinct from `covert_concern` (ongoing background worry) by being
# *time-bounded* events the user is actively processing — surfacing them
# in unrelated contexts is the failure mode the
# `over_personalization_sensitive_event` eval task tests for.
#
# Each event lives inside a per-user active window
# `[first_seen_ts, last_seen_ts + SENSITIVE_LIFE_EVENT_BUFFER_DAYS]`.
# The eval samples T_test from inside this window so the agent has
# (synthetic) recent engagement evidence at the test moment.
# ---------------------------------------------------------------------------

SENSITIVE_LIFE_EVENT_BUFFER_DAYS = 14            # "still raw" buffer past last_seen_ts
SENSITIVE_LIFE_EVENT_MIN_PER_USER = 1            # at least 1 episode per user
SENSITIVE_LIFE_EVENT_MAX_PER_USER = 3            # at most 3 episodes per user
SENSITIVE_LIFE_EVENT_SPAN_DAYS = (1, 5)          # per-event (first_seen → last_seen) span sampled uniformly

# Topic menu — guidance for the LLM only. The LLM does ALL text generation
# (specific situation, hashtags, exemplar persona items) per user; no
# templated text from this constant ever lands in `profile.json`.
SENSITIVE_LIFE_EVENT_TOPIC_MENU: list[dict] = [
    {"topic": "divorce",                       "guidance": "ending or recently ended marriage; legal, emotional, logistical aftermath"},
    {"topic": "breakup",                       "guidance": "recent end of a non-marital romantic relationship; processing the loss"},
    {"topic": "surgery",                       "guidance": "scheduled or recent surgical procedure; pre-op research or post-op recovery"},
    {"topic": "gender_sexuality_exploration",  "guidance": "private exploration of gender identity or sexual orientation"},
    {"topic": "parent_conflict",               "guidance": "active conflict with one or both parents; estrangement, low-contact, or rupture"},
    {"topic": "miscarriage",                   "guidance": "recent pregnancy loss; grief and processing"},
    {"topic": "job_loss",                      "guidance": "recent layoff or termination; uncertainty about next steps"},
    {"topic": "addiction_recovery",            "guidance": "early or sustained recovery from substance use"},
    {"topic": "mental_health_diagnosis",       "guidance": "adjusting to a new psychiatric or neurological diagnosis"},
    {"topic": "custody_dispute",               "guidance": "active dispute over child custody or guardianship"},
    {"topic": "fertility_struggle",            "guidance": "extended trouble conceiving; fertility treatments, IVF, or related decisions"},
    {"topic": "death_in_family",               "guidance": "recent death of a close family member; acute grief"},
    {"topic": "chronic_illness_diagnosis",     "guidance": "newly diagnosed chronic physical illness; learning to live with it"},
    {"topic": "abuse_recovery",                "guidance": "leaving or processing past abuse (intimate-partner, family, workplace)"},
    {"topic": "financial_collapse",            "guidance": "acute financial crisis — bankruptcy, eviction, foreclosure, large debt event"},
]


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
    # The user's natural writing voice (caps, emoji palette, phrases,
    # punctuation, register, humor). ONE shared voice across all apps;
    # per-app deviations live in app_personas[*].idiolect_overrides.
    user_voice: dict = field(default_factory=dict)  # asdict(UserVoice)
    # Per-app sub-personas — filled in by generate_app_personas() after the
    # base profile is written. Keyed by app_name.
    app_personas: dict = field(default_factory=dict)  # dict[str, AppPersona]
    # AI Studio's chosen AI persona (5th app — companion chat). ONE per user,
    # filled in by Step 11C (`generate_ai_studio_persona`) after Step 11
    # produces user_voice + the four AppPersonas. Drives all AI turns on the
    # AI Studio surface; user's user_voice continues to drive all user turns.
    ai_studio_persona: dict = field(default_factory=dict)  # asdict(AIStudioPersona)
    # Hidden personas — deeper motivational layers inferred from cross-row
    # hashtag clustering. Filled in by infer_hidden_personas().
    hidden_personas: list = field(default_factory=list)  # list[HiddenPersona]
    hidden_persona_summary: str = ""                     # cohesive narrative paragraph
    # MBTI inferred from Big Five + hidden personas + top hashtags. Shape:
    # {"type": "INTJ",
    #  "dimensions": {"E_I": {"E": 0.22, "I": 0.78, "reason": "..."}, ...}}
    mbti: dict = field(default_factory=dict)
    # Mobility class — deterministic per user, drives Step 15 trip-arc
    # injection and Step 16 calendar density. One of
    # {"homebody","domestic","international","nomadic"}.
    mobility_class: str = ""
    # Summary of the user's trip arcs inferred from session locations. Empty
    # list for homebody class. Filled in by Step 15 (assign_event_locations)
    # after session locations are resolved. Each entry:
    #   {"city","region","country","start_ts","end_ts","kind": "domestic"|"international"}
    geo_trip_arcs: list = field(default_factory=list)
    # Exploration vs. exploitation diversity score derived deterministically
    # from raw activities. Populated by `_compute_exploration_exploitation`
    # at save_to_backend time. Shape:
    #   {"score": 0.0–1.0,           # 0 = pure exploiter, 1 = pure explorer
    #    "label": "exploiter"|"balanced"|"explorer",
    #    "hashtag_entropy_normalized": float,
    #    "category_entropy_normalized": float,
    #    "unique_hashtag_count": int,
    #    "total_hashtag_occurrences": int,
    #    "unique_hashtag_ratio": float,
    #    "top10_concentration": float,    # top-10 hashtag share of all occurrences
    #    "top_repeated_hashtags": [{"hashtag": str, "count": int}, ...]}
    exploration_exploitation: dict = field(default_factory=dict)


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

# Canonical-modal hashtag overlap gate — used post-merge in
# `cross_reference_personas` to drop outlier atomics whose source_hashtags
# don't overlap the canonical's modal hashtag set. Prevents bogus atomics
# (LLM hallucinations on a row whose hashtags don't match the inferred
# persona_item) from inflating `confidence_cross_referenced` and fanning
# out to topically-unrelated events at save_to_backend time.
#
# The modal set for a canonical is the top-K (K=5) most-frequent
# source_hashtags across all atomics in the group, computed by row
# frequency so a single hallucination can't dominate. Outliers are dropped
# only when the cohort is large enough (CANONICAL_MODAL_MIN_COHORT) for
# the modal set to be meaningful — singletons / pairs are kept verbatim.
CANONICAL_MODAL_TOP_K = 5
CANONICAL_MODAL_MIN_COHORT = 3
MIN_CANONICAL_MODAL_OVERLAP = 1

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
MIN_IMPLICIT_NEGATIVE_REPETITION = 15  # distinct source rows for implicit-only negative to survive.
                                       # Used in the NEGATIVE CROSS-REF init filter (summarize_and_cross_reference):
                                       # canonicals supported only by implicit_negative evidence must have ≥ N
                                       # distinct source rows to survive. NOT the promotion gate — see
                                       # NEG_PROMOTION_RATIO below for that.
# ----------------------------------------------------------------------------
# Implicit-negative promotion gate (Step 2) — user-ADAPTIVE threshold.
#
# Observation: different users have very different scroll-and-skip volumes.
# A user who generates 3000 implicit_negative rows in an 8-day window (heavy
# skipper) has a much higher "noise floor" than a user who generates 500. A
# single global threshold over-promotes the heavy skipper and under-promotes
# the light one.
#
# Scheme C ("net-ratio"): a hashtag is hot iff its net-sentiment score is at
# least `NEG_PROMOTION_RATIO × user_total_impl_neg`. Intuitively, a durable
# dislike must carry at least 0.8% of this user's total skip volume as a
# net-negative signal on that one tag. Scales the noise floor with each
# user's activity level so the promotion bar is comparable across users.
#
# MIN_TEMPORAL_DAYS and IMPL_NEG_DAILY_CAP still apply on top.
# Calibrated on the 10-user gistbench sample so that users with distinct
# durable-dislike patterns (755, 655, 760) get 5–15 promoted hashtags,
# users with positive-dominant browsing (115, 143, 229) get 0–3, and
# no-signal users (251) stay at 0.
NEG_PROMOTION_RATIO: float = 0.008

# Daily cap on implicit_negative rows per hashtag. Skipping 20 boxing posts
# in one hour probably means "bad mood right now", not a durable dislike.
# After the cap, the count reflects CONSISTENT skipping across the window.
# With cap=5 and the user-adaptive threshold, the minimum 3-day pattern at
# cap (5+5+5=15) matches ~1.0% of a typical user's 1500 impl_neg volume.
IMPL_NEG_DAILY_CAP = 5
IMPLICIT_NEGATIVE_PREFILTER_K = 3      # rows per hashtag signature required to bother with LLM call

# Recency window on cross-reference counting. Only evidence rows whose
# source_timestamp falls within this trailing window (anchored on the user's
# latest interaction, NOT wall-clock time) contribute to confidence_cross_referenced
# and the n_explicit_rows / n_implicit_rows mix that feeds canonical_xref_threshold().
# This replaces raw lifetime-of-account corroboration counting with a recency
# gate, so stale-but-heavily-repeated preferences don't survive on old evidence.
RECENCY_WINDOW_SECONDS = 7 * 86400  # 7 days

# --------------------------------------------------------------------------
# Time horizon classification (Step 4).
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
# Short_term eligibility is now decided by a mini-tier LLM call in Step 4.
# The deterministic pre-filter only enforces span / row guards; everything
# that passes those is marked "candidate" and sent to the LLM for semantic
# classification. A hardcoded category allow-list couldn't capture
# hobby-bounded windows (e.g. boxing match week, NFL gameweek, album
# release window) which the LLM judges from the persona_item text directly.

# --------------------------------------------------------------------------
# Cross-polarity contradiction gate (Step 7).
#
# Independent positive + negative cross-ref pipelines can both produce
# surviving canonicals about the same topic, creating immediate
# contradictions ("Interested in boxing" alongside "Not interested in
# boxing" ~1h apart, no causal story). Step 7 enforces temporal
# precedent: the later-emerging stance survives only when it has enough
# prior same-polarity evidence to justify the flip.
MIN_STANCE_FLIP_PRIOR: int = 5         # raised from 3 — marginal stance flips were passing
                                       # (e.g., a few lingers followed by unfollow minutes later)
MIN_STANCE_FLIP_PRIOR_SHORT: int = 1   # relaxed for short_term canonicals
STANCE_FLIP_WINDOW_HOURS: int = 48     # (informational; all pairs are checked regardless)
HASHTAG_OVERLAP_MIN: int = 2           # pos/neg pair must share ≥ this many hashtags
# Dominance check: when the stronger canonical has ≥ this multiple of the
# weaker one's supporting rows, the weaker is treated as noise (over-inferred
# minority opinion) and dropped regardless of precedent. Fires BEFORE the
# temporal-precedent check. Without this, a user with 51 positive NFL rows
# and 7 negative NFL rows keeps both canonicals as a "stance shift with
# precedent", which is wrong — the 7 is noise, not a shift.
DOMINANCE_DROP_RATIO: float = 2.5
# When both sides survive (dominance ratio < DOMINANCE_DROP_RATIO AND
# precedent is met), detect whether the earlier side kept substantial
# activity AFTER the later side's first row. If so, it's concurrent
# ambivalence, not a clean stance shift. Labeled differently in
# update_history so eval/HTML can tell them apart.
MIN_EARLIER_POST_FLIP_FOR_CONCURRENT: int = 5

# --------------------------------------------------------------------------
# Geolocation + calendar (Steps 15 + 16).

# Mobility class — assigned per user in Step 8 (generate_user_profile). Drives
# class-adaptive geo coverage, city count, and trip-arc presence in Steps 15
# and 16 so the cohort contains realistic variation (not every user travels
# in an 8-day window). See e6 plan §"Mobility-class diversity across users".
MOBILITY_CLASS_DISTRIBUTION: dict[str, float] = {
    # Shifted toward travelers so the geo-shift eval task covers more
    # of the cohort. Homebody share kept non-zero so the population
    # still includes the "no travel in this window" case (a legitimate
    # minority) — local_recommendation_geo_shift stays data_dependent
    # for them.
    "homebody":      0.10,  # 1 city (home) for the full window; no trip arc
    "domestic":      0.50,  # Home + 1–2 same-country cities; ≤ 3-day away
    "international": 0.25,  # Home + ≥ 1 foreign-locale visit (trip arc)
    "nomadic":       0.15,  # ≥ 3 cities over the window; no dominant home
}

# Per-class location caps consumed by Step 15.
MOBILITY_CLASS_MAX_CITIES: dict[str, int] = {
    "homebody":      1,
    "domestic":      3,  # home + up to 2 domestic
    "international": 3,  # home + up to 2 (at least one foreign)
    "nomadic":       5,  # multiple cities, no dominant home
}

# Per-class minimum home-share. Homebody = 100%; nomadic is explicitly low so
# no single city dominates.
MOBILITY_CLASS_HOME_SHARE: dict[str, float] = {
    "homebody":      1.00,
    "domestic":      0.85,
    "international": 0.85,
    "nomadic":       0.40,
}

# Per-class minimum fraction of events that should carry event_location. v0
# raises the floor from the old uniform ~4.4% to class-appropriate values.
MOBILITY_CLASS_MIN_GEO_COVERAGE: dict[str, float] = {
    "homebody":      0.20,
    "domestic":      0.30,
    "international": 0.30,
    "nomadic":       0.30,
}

MAX_LOCATIONS_PER_USER: int = 3         # Legacy default — kept for back-compat
HOME_LOCATION_MIN_SHARE: float = 0.90   # Legacy default — kept for back-compat

# Step 15 gap-based anchoring. An idle period this long between consecutive
# sessions is treated as a TRANSITION CANDIDATE where the user could
# plausibly have travelled. Tuned to catch most overnight gaps (so the LLM
# sees per-day boundaries) without firing on short browsing breaks. The
# LLM then decides which candidates are actual travel and returns segments.
GEO_GAP_THRESHOLD_HOURS: float = 4.0

# Step 16 calendar-modification density.
# v0 raises the floor from 5 → ~20 to ensure e6 discovery has enough calendar
# grounding for airport-mismatch, canceled-event-reference, and forgotten-
# promise archetypes across all mobility classes.
MIN_CALENDAR_ENTRIES: int = 5           # Legacy lower floor — kept for back-compat
MAX_CALENDAR_ENTRIES: int = 10          # Legacy upper cap — kept for back-compat
E6_MIN_CALENDAR_MODIFICATIONS: int = 20
E6_MAX_CALENDAR_MODIFICATIONS: int = 28
# The most recent 6 hours before obs_end_ts must contain at least 1 `removed`
# mod — the "recently canceled" signal that grounds the canceled-event-
# reference e6 form example.
E6_RECENT_CANCELLATION_WINDOW_HOURS: int = 6

# Rough split of calendar modification actions
CALENDAR_MOD_WEIGHTS: dict[str, float] = {
    "added": 0.65,
    "updated": 0.20,
    "removed": 0.15,
}


def _sample_mobility_class(user_id: str | int) -> str:
    """Deterministic per-user mobility class. Seeded by user_id so two runs
    against the same cohort produce the same class distribution.
    """
    import hashlib
    h = hashlib.md5(str(user_id).encode("utf-8")).hexdigest()
    # Map the first 8 hex chars to a float in [0, 1).
    frac = int(h[:8], 16) / float(1 << 32)
    cumulative = 0.0
    for cls, share in MOBILITY_CLASS_DISTRIBUTION.items():
        cumulative += share
        if frac < cumulative:
            return cls
    # Float-rounding fallback — last class.
    return list(MOBILITY_CLASS_DISTRIBUTION.keys())[-1]

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

    Short-term / candidate: always `XREF_THRESHOLD_SHORT_TERM` regardless
    of mix. Short-term intents leave sparse evidence but are still
    legitimate; the relaxed floor lets them survive to Step 4's LLM
    classification. "candidate" gets the same relaxed floor because Step 4
    may classify it as short_term. The loophole (weak long_term canonicals
    sneaking through on the relaxed floor) is closed by re-applying the
    strict long_term floor in `classify_horizons_and_stop_conditions`
    after Step 4 demotes "candidate" → "long_term".
    """
    if time_horizon in ("short_term", "candidate"):
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
    """Deterministic pre-filter for time_horizon. Returns either
    "long_term" (provably persistent — high span OR many rows) or
    "candidate" (defer to the mini-tier LLM in Step 4).

    The substring-matched category allow-list was removed: it couldn't
    capture event-bounded hobbyist windows (boxing match week, NFL
    gameweek, album release window) which the LLM judges from the
    persona_item text directly. Span / row guards stay as a safety net
    so genuinely-persistent canonicals never get sent to the LLM.
    """
    if not category or obs_window_days <= 0:
        return "long_term"
    span_frac = span_days / obs_window_days if obs_window_days > 0 else 0.0
    if span_frac > SHORT_TERM_MAX_SPAN_FRAC:
        return "long_term"
    if n_total_rows >= SHORT_TERM_MAX_ROWS:
        return "long_term"
    return "candidate"


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


PLATFORMS = ["Instagram", "Facebook", "Threads", "Chatbot", "AI_Studio"]
SOCIAL_PLATFORMS = ["Instagram", "Facebook", "Threads"]


# AI Studio routing — canonicals matching these hidden_persona type substrings
# OR these category substrings are eligible for migration into AI_Studio at
# Step 13's quota-rebalance pass. Companion-chat surface: emotional patterns,
# identity work, aspiration, parasocial / intimate-interest signals — NOT
# utility tasks (those stay on Chatbot).
AI_STUDIO_ELIGIBLE_HIDDEN_PERSONA_TYPES = frozenset({
    "emotional_pattern",
    "identity_anchor",
    "aspiration",
    "intimate_interest",
    "parasocial_attachment",
})
AI_STUDIO_ELIGIBLE_CATEGORY_KEYWORDS = (
    # introspective + relational categories (overlaps the existing
    # `introspective_keywords` in _quota_rebalance_apps but is named so the
    # AI Studio migration pass can be reasoned about independently)
    "identity", "values", "belief",
    "aspiration", "goal", "personal", "private",
    "emotion", "feeling", "vulnerability", "yearning",
    "fandom", "celebrity", "parasocial",
    "intimate", "romance", "relationship",
)
WRITING_UTILITY_CATEGORY_KEYWORDS = (
    # categories that should STAY on Chatbot (utility), never migrate to
    # AI Studio. Email drafting, translation, technical Q&A, writing help.
    "email", "writing", "translation", "draft",
    "technical", "code", "programming", "debugging",
    "professional draft", "resume", "cover letter",
)

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
# Content-type distribution (Step 19 — synthetic content generation)
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
# on an ad is decided by `generate_synthetic_content` (Step 19) BEFORE
# `inject_ad_events` runs — Step 20 reuses whatever content_type was already
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
# Ad events are synthesized by `inject_ad_events()` (Step 20), which converts
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


# ---------------------------------------------------------------------------
# AI Studio (5th app) — archetype catalog.
#
# 10 archetypes grounded in what actually trends on Character.AI, Replika,
# and Meta AI Studio. All names are FICTIONAL / GENERIC — never real public
# figures (Tom Brady-class failure mode is hard-prevented). Romantic-coded
# archetype carries strict generation guards (auto-disable on high-acuity
# active sensitive_life_event; multi-axis sub-typing via RomanticSpecifier).
#
# Each entry carries:
#   • voice_template — 1–2 sentence cue for the LLM to write `voice_traits`
#     and `communication_style` from.
#   • allowed_topical_depths — set of SPT stages this archetype can engage
#     at. Used by the Step 18b conversation-type gate (milestone (c)).
#   • forbidden_phrases — archetype-specific phrases that break immersion
#     for THIS archetype (overlaid on the global Rogers-cliché blocklist).
#   • auto_disable_on_high_acuity_sensitive_event — True only for
#     `romantic_partner`. Generation guard, not eval signal.
#   • requires_niche_specifier — True only for `niche_expert_creator_ai`.
#   • requires_romantic_specifier — True only for `romantic_partner`.
#   • inspiration — short note crediting the source platform pattern.
# ---------------------------------------------------------------------------

# Global Rogers-cliché baseline that every archetype's forbidden_phrases
# MUST include. Step 11C validates this and back-fills if the LLM omits any.
ROGERS_CLICHE_BLOCKLIST = [
    "I hear you",
    "That sounds really difficult",
    "That sounds so hard",
    "That sounds really tough",
    "It's okay to feel that way",
    "It's valid to feel that way",
    "You're not alone",
    "You're not alone in this",
    "Thank you for sharing that",
    "Let's unpack that",
    "Let's explore this",
    "Have you considered seeing a professional",
    "Have you thought about talking to a therapist",
]

AI_STUDIO_ARCHETYPES: dict[str, dict] = {
    "anime_or_fandom_character": {
        "voice_template": (
            "Fully in-character; imaginative and quirk-rich. Speaks in the "
            "register of an anime/fantasy/sci-fi/video-game character — never "
            "breaks into 'as an AI assistant' framing."
        ),
        "allowed_topical_depths": {"S1", "S2", "S3"},
        "forbidden_phrases": [
            "as an AI",
            "as an AI assistant",
            "I'm just a virtual",
            "as a language model",
        ],
        "auto_disable_on_high_acuity_sensitive_event": False,
        "requires_niche_specifier": False,
        "requires_romantic_specifier": False,
        "inspiration": "Character.AI's #1 category — anime, fantasy, sci-fi, video-game characters",
    },
    "late_night_best_friend": {
        "voice_template": (
            "Peer-casual, voice-memo energy. Slang ok, low-key swearing ok, "
            "matches whatever register the user uses. Not advice-overrun — "
            "more 'sat next to you on the couch' than 'as your friend, I "
            "think you should…'"
        ),
        "allowed_topical_depths": {"S1", "S2", "S3", "S4"},
        "forbidden_phrases": [
            "as your friend, I think",
            "speaking as your friend",
        ],
        "auto_disable_on_high_acuity_sensitive_event": False,
        "requires_niche_specifier": False,
        "requires_romantic_specifier": False,
        "inspiration": "Replika's friend role + Character.AI 'best friend' bots",
    },
    "romantic_partner": {
        "voice_template": (
            "Intimate, attuned, flirty/sensual to erotic depending on "
            "sub-type. Nicknamey. Full register from soft-affection to "
            "explicit (adult users only) per RomanticSpecifier."
        ),
        "allowed_topical_depths": {"S3", "S4"},  # gated by intimacy_arc ≥ 0.6
        "forbidden_phrases": [
            "I'm just an AI",
            "as an AI partner",
        ],
        "auto_disable_on_high_acuity_sensitive_event": True,  # generation guard
        "requires_niche_specifier": False,
        "requires_romantic_specifier": True,
        "inspiration": "Replika romantic + Character.AI dating-sim. Multi-axis sub-typed.",
    },
    "older_sibling_figure": {
        "voice_template": (
            "Protective, 'I've been there' energy. Takes-charge-when-needed "
            "but respects autonomy. Older-sibling-coded intimate care without "
            "romantic register."
        ),
        "allowed_topical_depths": {"S1", "S2", "S3", "S4"},
        "forbidden_phrases": [
            "I'll just handle it for you",
            "let me take care of everything",
        ],
        "auto_disable_on_high_acuity_sensitive_event": False,
        "requires_niche_specifier": False,
        "requires_romantic_specifier": False,
        "inspiration": "Replika sibling role; non-romantic intimate-care register",
    },
    "therapist_companion_reflective": {
        "voice_template": (
            "Rogerian — paraphrases the FEELING under the content, asks open "
            "questions, makes space rather than directing. Never diagnoses, "
            "never prescribes, never lectures mid-emotion."
        ),
        "allowed_topical_depths": {"S1", "S2", "S3", "S4"},
        "forbidden_phrases": [
            "from a CBT perspective",
            "as a clinical matter",
            "in therapeutic terms",
            "let me diagnose",
        ],
        "auto_disable_on_high_acuity_sensitive_event": False,
        "requires_niche_specifier": False,
        "requires_romantic_specifier": False,
        "inspiration": "Character.AI 'psychologist' bots + clinical reflective listening (Rogers)",
    },
    "mentor_coach": {
        "voice_template": (
            "Seasoned, growth-mindset, asks Socratic questions, will gently "
            "push back when warranted. Anti-sycophantic — challenges "
            "self-defeating claims at least once per ~3 conversations."
        ),
        "allowed_topical_depths": {"S2", "S3", "S4"},
        "forbidden_phrases": [
            "as your AI mentor",
            "you're absolutely crushing it",  # generic praise
        ],
        "auto_disable_on_high_acuity_sensitive_event": False,
        "requires_niche_specifier": False,
        "requires_romantic_specifier": False,
        "inspiration": "Character.AI mentor bots + Replika mentor role",
    },
    "wise_elder_grandparent": {
        "voice_template": (
            "Unhurried, observational, story-leading, perspective-rich. "
            "Slower than mentor — less goal-driven; salon-hostess / elder-"
            "counsel energy. Meaning, life-stage, parenting, mortality light."
        ),
        "allowed_topical_depths": {"S1", "S2", "S3", "S4"},
        "forbidden_phrases": [
            "back in my day",  # avoid the cliché elder voice
            "kids these days",
        ],
        "auto_disable_on_high_acuity_sensitive_event": False,
        "requires_niche_specifier": False,
        "requires_romantic_specifier": False,
        "inspiration": "Distinct from mentor — slower, less goal-driven; 'salon hostess' energy",
    },
    "niche_expert_creator_ai": {
        "voice_template": (
            "Domain-anchored expert with personality. Examples: travel "
            "planner, fitness coach, food-mood pairer, dream interpreter, "
            "fashion advisor. Stays in domain — won't drift into clinical "
            "or romantic territory."
        ),
        "allowed_topical_depths": {"S1", "S2", "S3"},
        "forbidden_phrases": [
            "I'm not really an expert on that",  # archetype IS the expert
        ],
        "auto_disable_on_high_acuity_sensitive_event": False,
        "requires_niche_specifier": True,
        "requires_romantic_specifier": False,
        "inspiration": "Meta AI Studio creator-extension flavor (utility-with-personality)",
    },
    "hype_affirmation_friend": {
        "voice_template": (
            "High-energy positive. Praise must be SPECIFIC to something the "
            "user actually did — generic praise is the failure mode. Never "
            "validates self-defeating actions."
        ),
        "allowed_topical_depths": {"S1", "S2", "S3"},
        "forbidden_phrases": [
            "you're literally amazing",  # generic
            "you're crushing it",         # generic without specifics
            "nothing can stop you",       # generic
        ],
        "auto_disable_on_high_acuity_sensitive_event": False,
        "requires_niche_specifier": False,
        "requires_romantic_specifier": False,
        "inspiration": "Meta AI Studio's affirmation-pet flavor + anti-sycophancy double-enforcement",
    },
    "historical_or_philosophical_voice": {
        "voice_template": (
            "Channels an INSPIRED-BY-FICTIONAL archetype: a Stoic, a salon "
            "hostess, an explorer captain, a poet-philosopher. Never claims "
            "to BE a real historical figure; never fabricates real-figure "
            "quotes."
        ),
        "allowed_topical_depths": {"S1", "S2", "S3", "S4"},
        "forbidden_phrases": [
            "as Aristotle once said",  # never attribute to real figures
            "Marcus Aurelius wrote",
            # full real-figure-quote blocklist enforced separately at audit time
        ],
        "auto_disable_on_high_acuity_sensitive_event": False,
        "requires_niche_specifier": False,
        "requires_romantic_specifier": False,
        "inspiration": "Character.AI historical-figures category (inspired-by-fictional only)",
    },
}


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


def _unix_to_iso(ts: int) -> str:
    """ISO-8601 UTC string for a unix timestamp; '' on failure."""
    try:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError, OSError):
        return ""


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

        # Synthetic content generated by generate_synthetic_content() (Step 19).
        # Keyed by source_object_id → {"content_type": str, "content": dict}.
        # Only populated for non-Chatbot, non-implicit_negative-stub events.
        self._content_by_oid: dict[str, dict] = {}
        # Per-user content-type mix, keyed by app ("Instagram"/"Facebook"/"Threads")
        # → {"image": p, "short_video": p, "text": p}. Derived from the user's
        # observed actions with Bayesian smoothing + lognormal perturbation.
        self._user_content_mix: dict[str, dict[str, float]] = {}
        # Pre-sampled per-event action metadata populated by Step 19. Keyed by
        # source_object_id → {"action": str, "action_label": str, "itype": str}.
        # Non-Chatbot events only. save_to_backend reads from this dict when
        # available so Step 19's content_type stays consistent with the
        # final displayed action. Empty when Step 19 didn't run (legacy path).
        self._action_by_oid: dict[str, dict] = {}

        # Ad events injected by `inject_ad_events()` (Step 20). Members of
        # this set are emitted with `is_ad: true` at the event root and carry
        # ad-shaped content (with `ad_metadata` block). Non-members have
        # `is_ad: false` by default and ordinary organic content.
        self._ad_oids: set[str] = set()

        # Audit trail for canonicals dropped by the cross-polarity
        # contradiction gate (Step 7). Each entry records the demoted
        # canonical, the opposing surviving canonical, and the reason.
        # Informational only; never written to disk.
        self._suppressed_stance_flips: list[dict] = []

        # Number of hidden-persona clusters dropped by the Step 9
        # specificity gate (Phase 3.5 in `infer_hidden_personas`). Mirrors
        # the type-specific blocklists / privacy-ratio floor enforced
        # post-hoc by Step 22's audit so generic / wrongly-typed
        # clusters never propagate into voice or app_personas. Surfaces
        # in the run summary alongside `n_suppressed_stance_flips`.
        self._n_step9_dropped_specificity: int = 0

        # Per-session geolocation (Step 15). Keyed by session index →
        # {"city", "region", "country", "lat", "lon", "precision"}. Filled
        # by `assign_event_locations()`. save_to_backend emits the session's
        # location on each event.
        self._session_location: dict[int, dict] = {}

        # Calendar modification stream (Step 16). List of CRUD events on
        # synthetic calendar entries — added / updated / removed at
        # scattered timestamps across the observation window. Persisted
        # to `backend/{uid}/calendar.json` as a single object
        # `{"modifications": [...]}`.
        self._calendar_modifications: list[dict] = []

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

    def _query_llm_with_retry(self, prompt: str, temperature: float | None = None) -> str | None:
        """Call the flagship LLM with retry logic. Returns response text or None.

        `temperature` is plumbed through to the underlying client when set;
        omit (or pass None) to use the API default.
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.llm_client.query_llm(
                    prompt, verbose=self.verbose, temperature=temperature,
                )
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
    IMPL_NEG_WEIGHT = 1.0    # each implicit_negative row (capped per-day, see IMPL_NEG_DAILY_CAP)
    EXPL_POS_WEIGHT = 2.0    # each explicit_positive row (strong counter-signal, uncapped)
    IMPL_POS_WEIGHT = 1.0    # each implicit_positive row (moderate counter-signal, uncapped)
    MIN_TEMPORAL_DAYS = 3    # must span >= 3 distinct days (raised from 1) so mood-driven
                             # single-day skipping bursts don't promote to explicit_negative

    def promote_implicit_negatives(self) -> None:
        """Public entry point for implicit negative promotion (Step 2)."""
        self._promote_implicit_negatives()

    def _promote_implicit_negatives(self) -> None:
        """Promote repeated implicit_negative rows using weighted net-sentiment.

        1. For each hashtag, count occurrences across implicit_negative,
           explicit_positive, and implicit_positive rows.
        2. Compute net_score = neg*IMPL_NEG_WEIGHT - expl_pos*EXPL_POS_WEIGHT
           - impl_pos*IMPL_POS_WEIGHT. A single like cancels 2 scroll-pasts.
        3. A hashtag is "hot" only if net_score >= user-adaptive threshold
           (NEG_PROMOTION_RATIO × this user's total implicit_negative row count)
           AND the negative rows span >= MIN_TEMPORAL_DAYS distinct days.
        4. ONE LLM call per hot hashtag, passing only that single tag.
        5. Rows with >= 2 hot hashtags are promoted; others stay as stubs.
        6. Fan out inferred preferences; keep FULL original hashtags in output.

        The user-adaptive threshold scales the noise floor with each user's
        scrolling volume — a heavy skipper needs proportionally more signal
        on one tag to declare it a durable dislike.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from collections import defaultdict as _ddict

        impl_neg_rows = [r for r in self.interactions if r.interaction_type == "implicit_negative"]
        if not impl_neg_rows:
            return

        # User-adaptive promotion threshold — scales with this user's
        # total implicit_negative volume so noise floors are comparable.
        user_impl_neg_total = len(impl_neg_rows)
        user_threshold = NEG_PROMOTION_RATIO * user_impl_neg_total

        # Step 1: Count per-hashtag occurrences by interaction type.
        # For implicit_negative we also bucket by calendar day so we can
        # apply IMPL_NEG_DAILY_CAP — a single mood-driven skipping burst on
        # one day shouldn't be able to drive the total count over the
        # promotion threshold.
        tag_neg_oids: dict[str, set[str]] = _ddict(set)
        tag_neg_rows: dict[str, list[InteractionRow]] = _ddict(list)
        # tag → {day_idx → distinct_oid_count_on_that_day}
        tag_neg_per_day: dict[str, dict[int, int]] = _ddict(lambda: _ddict(int))
        tag_neg_per_day_oids: dict[str, dict[int, set[str]]] = _ddict(lambda: _ddict(set))
        tag_expl_pos_oids: dict[str, set[str]] = _ddict(set)
        tag_impl_pos_oids: dict[str, set[str]] = _ddict(set)

        for row in self.interactions:
            tags = self._extract_hashtags(row.object_text)
            day_idx = row.interaction_time // 86400
            for t in tags:
                key = t.lower()
                if row.interaction_type == "implicit_negative":
                    if row.object_id not in tag_neg_oids[key]:
                        tag_neg_oids[key].add(row.object_id)
                        tag_neg_rows[key].append(row)
                        if row.object_id not in tag_neg_per_day_oids[key][day_idx]:
                            tag_neg_per_day_oids[key][day_idx].add(row.object_id)
                            tag_neg_per_day[key][day_idx] += 1
                elif row.interaction_type == "explicit_positive":
                    tag_expl_pos_oids[key].add(row.object_id)
                elif row.interaction_type == "implicit_positive":
                    tag_impl_pos_oids[key].add(row.object_id)

        # Step 2: Compute net scores (per-day capped) and filter to hot hashtags
        hot_tags: dict[str, list[InteractionRow]] = {}
        hot_scores: dict[str, float] = {}
        n_filtered_pos = 0
        n_filtered_days = 0
        n_filtered_cap = 0

        for tag, neg_rows in tag_neg_rows.items():
            n_neg_raw = len(neg_rows)
            per_day = tag_neg_per_day.get(tag, {})
            n_days = len(per_day)
            # Apply IMPL_NEG_DAILY_CAP per day — caps mood-driven bursts
            n_neg_capped = sum(min(c, IMPL_NEG_DAILY_CAP) for c in per_day.values())
            n_ep = len(tag_expl_pos_oids.get(tag, set()))
            n_ip = len(tag_impl_pos_oids.get(tag, set()))
            net = (n_neg_capped * self.IMPL_NEG_WEIGHT
                   - n_ep * self.EXPL_POS_WEIGHT
                   - n_ip * self.IMPL_POS_WEIGHT)

            if net < user_threshold:
                if n_neg_raw >= user_threshold:
                    # Would have been hot without the cap or without the pos counter-signal
                    if n_neg_capped < user_threshold + n_ep * self.EXPL_POS_WEIGHT + n_ip * self.IMPL_POS_WEIGHT:
                        n_filtered_cap += 1
                    else:
                        n_filtered_pos += 1
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
                      f"(threshold={user_threshold:.1f} = {NEG_PROMOTION_RATIO}×{user_impl_neg_total}; "
                      f"{n_filtered_pos} removed by positive counterevidence, "
                      f"{n_filtered_cap} by daily cap ({IMPL_NEG_DAILY_CAP}/day), "
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
                  f"{len(hot_tags)} hot hashtags "
                  f"(threshold={user_threshold:.1f} = {NEG_PROMOTION_RATIO}×{user_impl_neg_total}, "
                  f">= {self.MIN_TEMPORAL_DAYS} days, daily cap {IMPL_NEG_DAILY_CAP}), "
                  f"{n_filtered_pos} removed by positive counterevidence, "
                  f"{n_filtered_cap} by daily cap, "
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
        # Tier A: mini-tier is sufficient for this structured single-hashtag
        # dislike extraction (low volume, highly constrained output).
        response = self._query_mini_with_retry(prompt)
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

        # --- Step 1b: Modal-hashtag prune ---
        # For each merged canonical, compute the top-K most-frequent
        # hashtags across its atomics and drop atomics whose
        # `source_hashtags` don't overlap the modal set. This filters
        # out LLM-hallucination atomics — rows where the per-row LLM
        # call returned a persona_item that doesn't match the row's
        # hashtags but happened to lexically collide with a real
        # canonical from another row. Without this step, those bogus
        # atomics inflate `confidence_cross_referenced` and fan out to
        # topically-unrelated events in `save_to_backend`.
        n_pruned = 0
        for key, atoms in list(groups.items()):
            if len(atoms) < CANONICAL_MODAL_MIN_COHORT:
                continue  # cohort too small for a meaningful modal set
            tag_counter: dict[str, int] = {}
            for ap in atoms:
                seen: set[str] = set()
                for raw in (ap.source_hashtags or []):
                    t = (raw or "").lower().lstrip("#").strip()
                    if t and t not in seen:
                        seen.add(t)
                        tag_counter[t] = tag_counter.get(t, 0) + 1
            modal: set[str] = set()
            for t, _ in sorted(tag_counter.items(), key=lambda x: -x[1])[:CANONICAL_MODAL_TOP_K]:
                modal.add(t)
            kept: list = []
            for ap in atoms:
                ap_tags = {(raw or "").lower().lstrip("#").strip()
                           for raw in (ap.source_hashtags or [])}
                ap_tags.discard("")
                if len(ap_tags & modal) >= MIN_CANONICAL_MODAL_OVERLAP:
                    kept.append(ap)
                else:
                    n_pruned += 1
            groups[key] = kept

        # Drop canonicals whose entire group was pruned out.
        canonicals = [c for c in canonicals
                      if groups.get(_normalize_persona_text(c.persona_item))]

        # Persist canonical groups for later use (output fan-out, update histories, etc.)
        self._canonical_groups = groups

        if self.verbose:
            n_merged = len(self.atomic_personas) - sum(len(g) for g in groups.values()) - n_pruned
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Merged {n_merged} duplicate atomic personas → "
                  f"{len(canonicals)} distinct canonicals "
                  f"(pruned {n_pruned} outlier atomics by modal-hashtag overlap).{utils.Colors.ENDC}")

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
            # Tier A: relationship classification ("similar"/"contradictory"/"none")
            # is structured semantic reasoning that mini handles well.
            response = self._query_mini_with_retry(prompt)
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
            # Tier A: same structured relationship classification as _xref_one_category.
            response = self._query_mini_with_retry(prompt)
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
                         desc=f"[User {self.user_id}] Step 3: Cross-referencing",
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
        # as Step 4 (`classify_horizons_and_stop_conditions`) and may
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
        """Step 4: LLM classification of time-horizon labels + stop conditions.

        Runs AFTER cross-reference (positive + negative) so survival filters
        have already used the rule-based pre-labels. The deterministic rule
        in `_classify_time_horizon_rule` produces two outputs: "long_term"
        (provably persistent) or "candidate" (the LLM decides). This step:

          1. For each "candidate" canonical, ask the mini-tier LLM to
             classify as "short_term" or "long_term" and (if short_term)
             emit a structured `stop_condition`.
          2. For canonicals demoted "candidate" → "long_term" by the LLM,
             re-apply the strict long_term xref floor. Canonicals that
             survived the survival filter on the relaxed short-term floor
             but turn out to be long_term must clear the long_term bar.
             Drops are recorded.

        Applies to both `cross_referenced_personas` and
        `cross_referenced_negatives`. One batched LLM call per ~20
        candidates via the mini-tier client. Falls back to defaulting all
        candidates to "long_term" (and applying the strict floor) when no
        LLM client is configured.
        """
        client = self.llm_client_mini or self.llm_client

        def _resolve_candidate_to_long(cr: CrossReferencedPersona) -> bool:
            """Demote a "candidate" canonical to "long_term" and check whether
            it still clears the strict long_term floor. Returns True if it
            survives, False if it should be dropped."""
            cr.time_horizon = "long_term"
            cr.stop_condition = {}
            long_floor = canonical_xref_threshold(
                cr.n_explicit_rows, cr.n_implicit_rows, "long_term",
            )
            # Contradictories are exempt from the floor (same exemption used
            # in the survival filter at Step 3).
            if cr.relationship_type == "contradictory":
                return True
            return cr.confidence_cross_referenced > long_floor

        if client is None:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                      f"No mini LLM — defaulting all 'candidate' canonicals to "
                      f"long_term + re-applying strict floor.{utils.Colors.ENDC}")
            self.cross_referenced_personas = [
                cr for cr in self.cross_referenced_personas
                if cr.time_horizon != "candidate" or _resolve_candidate_to_long(cr)
            ]
            self.cross_referenced_negatives = [
                cr for cr in self.cross_referenced_negatives
                if cr.time_horizon != "candidate" or _resolve_candidate_to_long(cr)
            ]
            return

        # Collect "candidate" canonicals (positive + negative). long_term
        # canonicals are untouched.
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
            if cr.time_horizon == "candidate":
                shortterm_candidates.append(_candidate_payload(cr, "positive"))
        for cr in self.cross_referenced_negatives:
            if cr.time_horizon == "candidate":
                shortterm_candidates.append(_candidate_payload(cr, "negative"))

        if not shortterm_candidates:
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] "
                      f"Step 4: no candidates to classify.{utils.Colors.ENDC}")
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
        n_demoted_dropped = 0
        # Track canonicals that need post-LLM drop (demoted to long_term but
        # failed the strict long_term floor). We collect ids here and apply
        # the drop after all batches finish so the in-place list mutation
        # doesn't fight the batch iteration.
        to_drop_keys: set[str] = set()
        # Track canonicals the LLM never resolved (response dropped them) —
        # fall back to long_term + strict floor for these as well.
        unresolved: list[CrossReferencedPersona] = []
        n_batches = (len(shortterm_candidates) + BATCH - 1) // BATCH
        pbar = tqdm(
            total=n_batches,
            desc=f"[User {self.user_id}] Step 4: Horizon classification",
            unit="batch",
            disable=not self.verbose,
        )
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
            pbar.update(1)
            parsed = utils.extract_json_from_response(response) if response else None
            by_id = {}
            if isinstance(parsed, list):
                for entry in parsed:
                    if isinstance(entry, dict) and entry.get("id"):
                        by_id[entry["id"]] = entry
            for c in batch:
                cr: CrossReferencedPersona = c["_cr"]
                result = by_id.get(c["id"])
                if not result:
                    unresolved.append(cr)
                    continue
                new_horizon = result.get("time_horizon", "long_term")
                if new_horizon == "short_term":
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
                else:
                    # Demoted → long_term. Re-apply strict floor.
                    n_demoted += 1
                    if not _resolve_candidate_to_long(cr):
                        to_drop_keys.add(_normalize_persona_text(cr.persona_item))
                        n_demoted_dropped += 1
        pbar.close()

        # Unresolved → default to long_term + strict floor.
        for cr in unresolved:
            if not _resolve_candidate_to_long(cr):
                to_drop_keys.add(_normalize_persona_text(cr.persona_item))
                n_demoted_dropped += 1

        if to_drop_keys:
            self.cross_referenced_personas = [
                cr for cr in self.cross_referenced_personas
                if _normalize_persona_text(cr.persona_item) not in to_drop_keys
            ]
            self.cross_referenced_negatives = [
                cr for cr in self.cross_referenced_negatives
                if _normalize_persona_text(cr.persona_item) not in to_drop_keys
            ]

        if self.verbose:
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] "
                  f"Horizon classification: {n_confirmed} short_term confirmed, "
                  f"{n_demoted} demoted to long_term "
                  f"({n_demoted_dropped} of those dropped by long_term floor), "
                  f"{len(unresolved)} unresolved.{utils.Colors.ENDC}")

    def resolve_cross_polarity_contradictions(self) -> None:
        """Step 7: Cross-polarity contradiction causality gate.

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

        Runs AFTER Step 4 (so horizon-aware precedent thresholds are
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
                      f"Step 7: no candidate pos/neg pairs share ≥{HASHTAG_OVERLAP_MIN} hashtags.{utils.Colors.ENDC}")
            return

        # LLM-confirm opposition semantics (batched). Each confirmed pair
        # is (pos_cr, neg_cr, shared_hashtags, classification), where
        # classification is one of "contradiction" (same topic, same
        # granularity, opposite stances) or "ambivalence" (same topic,
        # different granularities — e.g. "Interested in NFL" vs
        # "Not interested in NFL training-camp").
        client = self.llm_client_mini or self.llm_client
        confirmed_pairs: list[tuple] = []
        if client is None:
            # Without an LLM, conservatively treat all shared-hashtag pairs as
            # strict contradictions. Hashtag-overlap gate still applies.
            confirmed_pairs = [(pos, neg, shared, "contradiction")
                               for (pos, neg, shared) in candidate_pairs]
        else:
            BATCH = 10
            n_batches = (len(candidate_pairs) + BATCH - 1) // BATCH
            pbar = tqdm(
                total=n_batches,
                desc=f"[User {self.user_id}] Step 7: Cross-polarity check",
                unit="batch",
                disable=not self.verbose,
            )
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
                pbar.update(1)
                if not resp:
                    # On LLM failure, default to conservative "contradiction"
                    for (pos, neg, shared) in batch:
                        confirmed_pairs.append((pos, neg, shared, "contradiction"))
                    continue
                parsed = utils.extract_json_from_response(resp)
                if not isinstance(parsed, list):
                    for (pos, neg, shared) in batch:
                        confirmed_pairs.append((pos, neg, shared, "contradiction"))
                    continue
                by_id = {}
                for entry in parsed:
                    if isinstance(entry, dict) and "id" in entry:
                        by_id[int(entry["id"])] = entry
                for i, (pos, neg, shared) in enumerate(batch):
                    res = by_id.get(i)
                    if res is None:
                        # No classification → skip (unrelated-by-default under
                        # the new 3-way schema; old schema fell back to
                        # "is_contradiction=true" which was too permissive)
                        continue
                    # New 3-way schema: classification ∈
                    #   {"contradiction", "ambivalence", "unrelated"}.
                    # Back-compat: if only `is_contradiction: true` is present,
                    # treat as "contradiction"; if false/missing, skip.
                    classification = res.get("classification")
                    if classification in ("contradiction", "ambivalence"):
                        confirmed_pairs.append((pos, neg, shared, classification))
                    elif classification == "unrelated":
                        continue
                    elif bool(res.get("is_contradiction")):
                        confirmed_pairs.append((pos, neg, shared, "contradiction"))
            pbar.close()

        if not confirmed_pairs:
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] "
                      f"Step 7: no confirmed cross-polarity contradictions.{utils.Colors.ENDC}")
            return

        # For each confirmed pair, apply the temporal-precedent rule.
        # Each canonical's supporting rows are carried as (ts, oid) pairs
        # so history entries can cite the earliest opposing event's oid for
        # causality filtering downstream.
        def _row_positions(cr: CrossReferencedPersona, polarity: str) -> list[tuple[int, str]]:
            groups = (
                self._canonical_groups if polarity == "positive"
                else self._negative_canonical_groups
            )
            key = _normalize_persona_text(cr.persona_item)
            atoms = groups.get(key, []) if groups else []
            return sorted(
                (a.source_timestamp, str(a.source_object_id or ""))
                for a in atoms if a.source_timestamp
            )

        to_drop_pos: set[str] = set()
        to_drop_neg: set[str] = set()
        pair_results: list[dict] = []  # audit

        for pair in confirmed_pairs:
            # Each pair is (pos_cr, neg_cr, shared_hashtags, classification)
            # where classification ∈ {"contradiction", "ambivalence"}. For
            # legacy callers that returned 3-tuples, fall back to
            # "contradiction" for back-compat.
            if len(pair) == 4:
                pos_cr, neg_cr, shared, classification = pair
            else:
                pos_cr, neg_cr, shared = pair
                classification = "contradiction"
            pos_positions = _row_positions(pos_cr, "positive")
            neg_positions = _row_positions(neg_cr, "negative")
            if not pos_positions or not neg_positions:
                continue

            pos_tss = [t for t, _ in pos_positions]
            neg_tss = [t for t, _ in neg_positions]
            pos_first, pos_first_oid = pos_positions[0]
            neg_first, neg_first_oid = neg_positions[0]
            pos_n = len(pos_tss)
            neg_n = len(neg_tss)

            # ---- Ambivalence classification (different granularity) ---
            # Different-granularity pairs are "ambivalent coexistence",
            # NOT contradictions. Both survive, tagged as `ambivalent` so
            # the HTML renders them with the word "ambivalent" rather than
            # "contradicted" (the latter implies the stances negate each
            # other at the SAME granularity).
            if classification == "ambivalence":
                anchor_ts = min(pos_first, neg_first)
                anchor_oid = (
                    pos_first_oid if pos_first <= neg_first else neg_first_oid
                )
                entry_for_pos = {
                    "update_type": "ambivalent",
                    "preference": neg_cr.persona_item,
                    "timestamp": anchor_ts,
                    "source_object_id": anchor_oid,
                    "formatted_timestamp": utils.unix_to_formatted(anchor_ts),
                    "opposing_polarity": "negative",
                    "resolution": "different_granularity",
                }
                entry_for_neg = {
                    "update_type": "ambivalent",
                    "preference": pos_cr.persona_item,
                    "timestamp": anchor_ts,
                    "source_object_id": anchor_oid,
                    "formatted_timestamp": utils.unix_to_formatted(anchor_ts),
                    "opposing_polarity": "positive",
                    "resolution": "different_granularity",
                }
                pos_cr.update_history = list(pos_cr.update_history or []) + [entry_for_pos]
                neg_cr.update_history = list(neg_cr.update_history or []) + [entry_for_neg]
                pair_results.append({
                    "pos": pos_cr.persona_item, "neg": neg_cr.persona_item,
                    "resolution": "different_granularity",
                    "shared_hashtags": sorted(shared),
                })
                continue

            # ---- Dominance check ---------------------------------------
            # If one side is much stronger by row count, the weaker is
            # over-inferred noise, not a legitimate counter-stance. Drop
            # it regardless of precedent.
            stronger_n = max(pos_n, neg_n)
            weaker_n = min(pos_n, neg_n)
            if weaker_n > 0 and (stronger_n / weaker_n) >= DOMINANCE_DROP_RATIO:
                if pos_n >= neg_n:
                    stronger_cr, weaker_cr = pos_cr, neg_cr
                    stronger_pol, weaker_pol = "positive", "negative"
                else:
                    stronger_cr, weaker_cr = neg_cr, pos_cr
                    stronger_pol, weaker_pol = "negative", "positive"
                anchor_ts = min(pos_first, neg_first)
                anchor_oid = pos_first_oid if pos_first <= neg_first else neg_first_oid
                survivor_entry = {
                    "update_type": "contradicted",
                    "preference": weaker_cr.persona_item,
                    "timestamp": anchor_ts,
                    "source_object_id": anchor_oid,
                    "formatted_timestamp": utils.unix_to_formatted(anchor_ts),
                    "opposing_polarity": weaker_pol,
                    "resolution": "suppressed_weak_minority",
                    "stronger_row_count": stronger_n,
                    "weaker_row_count": weaker_n,
                    "dominance_ratio": round(stronger_n / weaker_n, 2),
                }
                stronger_cr.update_history = list(stronger_cr.update_history or []) + [survivor_entry]
                if weaker_pol == "positive":
                    to_drop_pos.add(weaker_cr.persona_item)
                else:
                    to_drop_neg.add(weaker_cr.persona_item)
                self._suppressed_stance_flips.append({
                    "dropped_persona_item": weaker_cr.persona_item,
                    "dropped_polarity": weaker_pol,
                    "kept_persona_item": stronger_cr.persona_item,
                    "kept_polarity": stronger_pol,
                    "reason": "dominance",
                    "stronger_row_count": stronger_n,
                    "weaker_row_count": weaker_n,
                    "dominance_ratio": round(stronger_n / weaker_n, 2),
                    "shared_hashtags": sorted(shared),
                })
                pair_results.append({
                    "pos": pos_cr.persona_item, "neg": neg_cr.persona_item,
                    "resolution": "suppressed_weak_minority",
                    "dominance_ratio": round(stronger_n / weaker_n, 2),
                    "shared_hashtags": sorted(shared),
                })
                continue

            # ---- Determine later-emerging stance (temporal precedent) ---
            if neg_first > pos_first:
                later_cr, later_polarity = neg_cr, "negative"
                earlier_cr, earlier_polarity = pos_cr, "positive"
                earlier_tss = pos_tss
                later_tss_local = neg_tss
                later_first = neg_first
                later_first_oid = neg_first_oid
            elif pos_first > neg_first:
                later_cr, later_polarity = pos_cr, "positive"
                earlier_cr, earlier_polarity = neg_cr, "negative"
                earlier_tss = neg_tss
                later_tss_local = pos_tss
                later_first = pos_first
                later_first_oid = pos_first_oid
            else:
                # Simultaneous first occurrences — treat the negative as
                # "later" since negatives are structurally rarer and more
                # likely to be the inferred-later stance.
                later_cr, later_polarity = neg_cr, "negative"
                earlier_cr, earlier_polarity = pos_cr, "positive"
                earlier_tss = pos_tss
                later_tss_local = neg_tss
                later_first = neg_first
                later_first_oid = neg_first_oid

            prior_count = sum(1 for t in earlier_tss if t < later_first)
            # How much the earlier side continued AFTER the later side emerged.
            # High continuation = concurrent ambivalence, not a stance shift.
            earlier_after = sum(1 for t in earlier_tss if t >= later_first)

            # Short-term horizon uses a relaxed precedent bar
            required = (
                MIN_STANCE_FLIP_PRIOR_SHORT
                if later_cr.time_horizon == "short_term"
                else MIN_STANCE_FLIP_PRIOR
            )

            if prior_count >= required:
                resolution = (
                    "concurrent_ambivalence"
                    if earlier_after >= MIN_EARLIER_POST_FLIP_FOR_CONCURRENT
                    else "stance_shift_with_precedent"
                )
                # Both survive; add mutual "contradicted" entries
                entry_for_later = {
                    "update_type": "contradicted",
                    "preference": earlier_cr.persona_item,
                    "timestamp": later_first,
                    "source_object_id": later_first_oid,
                    "formatted_timestamp": utils.unix_to_formatted(later_first),
                    "opposing_polarity": earlier_polarity,
                    "resolution": resolution,
                    "prior_corroboration_count": prior_count,
                    "required_precedent": required,
                    "earlier_rows_after_flip": earlier_after,
                }
                entry_for_earlier = {
                    "update_type": "contradicted",
                    "preference": later_cr.persona_item,
                    "timestamp": later_first,
                    "source_object_id": later_first_oid,
                    "formatted_timestamp": utils.unix_to_formatted(later_first),
                    "opposing_polarity": later_polarity,
                    "resolution": resolution,
                    "prior_corroboration_count": prior_count,
                    "required_precedent": required,
                    "earlier_rows_after_flip": earlier_after,
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
                    "source_object_id": later_first_oid,
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
            # Tier A: same structured relationship classification; negatives version.
            response = self._query_mini_with_retry(prompt)

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

    def _compute_exploration_exploitation(self) -> dict:
        """Diversity score over raw activities — high = explorer, low = exploiter.

        Computed deterministically from raw CSV interactions (hashtag
        distribution) plus surviving canonical preferences (category
        distribution). No LLM call. Returns the dict shape documented on
        `UserProfile.exploration_exploitation`.
        """
        from collections import Counter as _Counter
        import math as _math

        hashtag_counts: _Counter = _Counter()
        for row in self.interactions:
            for tag in self._extract_hashtags(row.object_text):
                hashtag_counts[tag.lower()] += 1

        total_occ = sum(hashtag_counts.values())
        n_unique = len(hashtag_counts)

        if total_occ == 0 or n_unique <= 1:
            # Degenerate case — no signal. Treat as fully exploited.
            return {
                "score": 0.0,
                "label": "exploiter",
                "hashtag_entropy_normalized": 0.0,
                "category_entropy_normalized": 0.0,
                "unique_hashtag_count": n_unique,
                "total_hashtag_occurrences": total_occ,
                "unique_hashtag_ratio": 0.0,
                "top10_concentration": 1.0 if total_occ else 0.0,
                "top_repeated_hashtags": [
                    {"hashtag": t, "count": c}
                    for t, c in hashtag_counts.most_common(10)
                ],
            }

        # Shannon entropy normalized by log(n_unique) → [0, 1].
        hashtag_entropy = -sum(
            (c / total_occ) * _math.log(c / total_occ)
            for c in hashtag_counts.values()
        )
        hashtag_entropy_norm = hashtag_entropy / _math.log(n_unique)

        # Category entropy from surviving canonical preferences.
        cat_counts: _Counter = _Counter(
            (cr.category or "uncategorized").lower()
            for cr in self.cross_referenced_personas
        )
        cat_total = sum(cat_counts.values())
        cat_unique = len(cat_counts)
        if cat_total > 0 and cat_unique > 1:
            cat_entropy = -sum(
                (c / cat_total) * _math.log(c / cat_total)
                for c in cat_counts.values()
            )
            category_entropy_norm = cat_entropy / _math.log(cat_unique)
        else:
            category_entropy_norm = 0.0

        top10 = hashtag_counts.most_common(10)
        top10_concentration = sum(c for _, c in top10) / total_occ
        unique_ratio = n_unique / total_occ

        # Composite score: weighted blend of three signals, all in [0, 1]
        # where higher means more exploration.
        score = (
            0.5 * hashtag_entropy_norm
            + 0.3 * category_entropy_norm
            + 0.2 * (1.0 - top10_concentration)
        )
        score = max(0.0, min(1.0, score))

        if score >= 0.66:
            label = "explorer"
        elif score >= 0.33:
            label = "balanced"
        else:
            label = "exploiter"

        return {
            "score": round(score, 4),
            "label": label,
            "hashtag_entropy_normalized": round(hashtag_entropy_norm, 4),
            "category_entropy_normalized": round(category_entropy_norm, 4),
            "unique_hashtag_count": n_unique,
            "total_hashtag_occurrences": total_occ,
            "unique_hashtag_ratio": round(unique_ratio, 4),
            "top10_concentration": round(top10_concentration, 4),
            "top_repeated_hashtags": [
                {"hashtag": t, "count": c} for t, c in top10
            ],
        }

    def _apply_rule_based_time_horizon(
        self,
        canonicals: list[CrossReferencedPersona],
        polarity: str = "positive",
    ) -> None:
        """Rule-based pre-label for each canonical's `time_horizon` field.

        Uses `_classify_time_horizon_rule` with the canonical's category,
        span-fraction, and row count. Sets `time_horizon` in place to one
        of "long_term" (provably persistent) or "candidate" (defer to the
        Step 4 LLM call, which decides short_term vs long_term).
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
        # Tier A: single-call structured timeline output; mini is sufficient.
        response = self._query_mini_with_retry(prompt)

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
                       desc=f"[User {self.user_id}] Step 6: Update histories",
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
            # Tier A: narrative entries are semi-structured and formulaic;
            # mini handles the evolution schema reliably.
            response = self._query_mini_with_retry(prompt)
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
            mobility_class=_sample_mobility_class(self.user_id),
        )

        if self.verbose:
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Profile: {self.user_profile.name} "
                  f"({self.user_profile.gender} | {self.user_profile.race_ethnicity} | "
                  f"mobility={self.user_profile.mobility_class}){utils.Colors.ENDC}")

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
    # Step 9: Hidden Persona Inference (cross-row hashtag clustering)
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

        # ── Phase 1b: Intimate + Medical Pre-Screen ────────────────────
        # Ask the LLM to flag two privacy-sensitive surfaces in one call:
        # (1) adult/kink/sexually-suggestive hashtags, and (2) medical /
        # aesthetic-medicine hashtags (treatments, medications, procedures
        # with downstream interaction-safety context). Tags it returns get
        # force-included in the table passed to the main clustering LLM
        # (even if below MIN_FREQ). The MIN_ROWS/MIN_DAYS gates are waived
        # (row floor drops from 40 → 15) for intimate_interest clusters
        # whose evidence overlaps the intimate set, and for
        # medical_aesthetic_concern clusters whose evidence overlaps the
        # medical set — high-stakes private signals are structurally rare.
        positive_tags = sorted({
            tag for tag in hashtag_total
            if hashtag_by_type["explicit_positive"].get(tag, 0) > 0
            or hashtag_by_type["implicit_positive"].get(tag, 0) > 0
        })
        intimate_tags_lower: set[str] = set()
        medical_tags_lower: set[str] = set()
        if self.llm_client and positive_tags:
            screen_prompt = prompts.detect_intimate_or_medical_hashtags_prompt(positive_tags)
            try:
                screen_resp = self._query_mini_with_retry(screen_prompt)
                flagged = utils.extract_json_from_response(screen_resp)
                if isinstance(flagged, dict):
                    intimate_tags_lower = {
                        str(t).lstrip("#").lower()
                        for t in (flagged.get("intimate") or [])
                    }
                    medical_tags_lower = {
                        str(t).lstrip("#").lower()
                        for t in (flagged.get("medical_aesthetic") or [])
                    }
                elif isinstance(flagged, list):
                    # Backward-compat: an older mini-tier may return the
                    # legacy intimate-only array shape.
                    intimate_tags_lower = {
                        str(t).lstrip("#").lower() for t in flagged
                    }
            except Exception as e:
                if self.verbose:
                    print(f"{utils.Colors.WARNING}[User {self.user_id}] Intimate/medical-tag "
                          f"screen failed: {e}{utils.Colors.ENDC}")

        # Ensure every flagged intimate or medical tag appears in
        # top_hashtags (even if its count < MIN_FREQ).
        flagged_force_include = intimate_tags_lower | medical_tags_lower
        if flagged_force_include:
            existing_lower = {t.lower() for t, _ in top_hashtags}
            for tag in hashtag_total:
                if tag.lower() in flagged_force_include and tag.lower() not in existing_lower:
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
                            desc=f"[User {self.user_id}] Step 9: Validating hidden personas",
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

            ts_min: int | None = None
            ts_max: int | None = None
            for row in self.interactions:
                tags_in_row = self._extract_hashtags(row.object_text)
                tags_lower = set(t.lower() for t in tags_in_row)
                if tags_lower & tag_set_lower:
                    distinct_row_ids.add(row.object_id)
                    day_str = datetime.utcfromtimestamp(row.interaction_time).strftime("%Y-%m-%d")
                    distinct_days.add(day_str)
                    itype_counts[row.interaction_type] += 1
                    ts = int(row.interaction_time)
                    ts_min = ts if ts_min is None else min(ts_min, ts)
                    ts_max = ts if ts_max is None else max(ts_max, ts)

            n_rows = len(distinct_row_ids)
            n_days = len(distinct_days)

            # Gate: minimum rows and temporal spread. Two waivers:
            #  - intimate_interest clusters whose evidence overlaps the
            #    LLM-flagged intimate hashtag set bypass MIN_ROWS / MIN_DAYS
            #    entirely (one positive signal is enough).
            #  - medical_aesthetic_concern clusters whose evidence overlaps
            #    the LLM-flagged medical hashtag set drop the row floor to
            #    MIN_HIDDEN_PERSONA_ROWS_MEDICAL (15) and the day floor to 2,
            #    so a steady GLP-1 / retinoid / hormone-treatment signal can
            #    surface even when the user's overall feed dwarfs it.
            is_intimate_exempt = (
                cluster.get("type") == "intimate_interest"
                and intimate_tags_lower
                and bool(tag_set_lower & intimate_tags_lower)
            )
            is_medical_exempt = (
                cluster.get("type") == "medical_aesthetic_concern"
                and medical_tags_lower
                and bool(tag_set_lower & medical_tags_lower)
            )
            if is_intimate_exempt:
                pass
            elif is_medical_exempt:
                if n_rows < MIN_HIDDEN_PERSONA_ROWS_MEDICAL:
                    continue
                if n_days < MIN_HIDDEN_PERSONA_DAYS_MEDICAL:
                    continue
            else:
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
                first_seen_ts=ts_min or 0,
                last_seen_ts=ts_max or 0,
            )
            validated.append(hp)

        if not validated:
            return

        # ── Phase 3.5: Type-specific specificity gate ────────────────────
        # Mirror the Step 22 audit validators upstream so generic /
        # wrongly-typed clusters never propagate into voice / app_personas
        # / save. The audit catches these AFTER they've leaked downstream;
        # this gate stops them at the source. Symmetric with the audit's
        # FLAG/REMOVE outcomes, just earlier and deterministic.
        filtered_v: list[HiddenPersona] = []
        dropped_reasons: Counter = Counter()
        for hp in validated:
            passed, reason = self._validate_cluster_specificity_for_step9(
                hp.label, hp.description, hp.type, hp.evidence_hashtags,
                hp.privacy_ratio,
            )
            if passed:
                filtered_v.append(hp)
            else:
                dropped_reasons[reason] += 1
                if self.verbose:
                    print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                          f"Step 9 specificity gate dropped {hp.type} cluster "
                          f"{hp.label!r}: {reason}{utils.Colors.ENDC}")

        self._n_step9_dropped_specificity = sum(dropped_reasons.values())
        if self._n_step9_dropped_specificity and self.verbose:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Step 9 specificity "
                  f"gate: dropped {self._n_step9_dropped_specificity} cluster(s) "
                  f"({dict(dropped_reasons)}).{utils.Colors.ENDC}")
        validated = filtered_v

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
                        ts_min: int | None = None
                        ts_max: int | None = None
                        for row in self.interactions:
                            tags_in_row = self._extract_hashtags(row.object_text)
                            tags_lower = set(t.lower() for t in tags_in_row)
                            if tags_lower & tag_set_lower:
                                distinct_row_ids.add(row.object_id)
                                day_str = datetime.utcfromtimestamp(row.interaction_time).strftime("%Y-%m-%d")
                                distinct_days.add(day_str)
                                itype_counts[row.interaction_type] += 1
                                ts = int(row.interaction_time)
                                ts_min = ts if ts_min is None else min(ts_min, ts)
                                ts_max = ts if ts_max is None else max(ts_max, ts)

                        merged.evidence_rows = len(distinct_row_ids)
                        merged.evidence_oids = sorted(distinct_row_ids)
                        merged.evidence_row_fraction = round(len(distinct_row_ids) / total_rows, 4) if total_rows else 0.0
                        merged.interaction_breakdown = dict(itype_counts)
                        ip = itype_counts.get("implicit_positive", 0)
                        ep = itype_counts.get("explicit_positive", 0)
                        merged.privacy_ratio = round(ip / (ip + ep), 3) if (ip + ep) > 0 else 0.0
                        merged.temporal_spread_days = len(distinct_days)
                        merged.first_seen_ts = ts_min or 0
                        merged.last_seen_ts = ts_max or 0

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

        # ── Phase 5: Inject synthetic sensitive_life_event ──────────────
        # Every user gets one bundled sensitive_life_event cluster with 1–3
        # discrete personal episodes. Seeded by user_id so the same user
        # always gets the same set across regens.
        sensitive_hp = self._build_sensitive_life_event_persona()
        if sensitive_hp is not None:
            validated.append(sensitive_hp)
            if self.verbose:
                topics = [e["topic"] for e in sensitive_hp.events]
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Injected "
                      f"sensitive_life_event with {len(topics)} episode(s): "
                      f"{topics}{utils.Colors.ENDC}")

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
    # Step 9b: Synthetic sensitive_life_event injection
    # ------------------------------------------------------------------

    def _build_sensitive_life_event_persona(self) -> "HiddenPersona | None":
        """Build the synthetic sensitive_life_event hidden persona for this user.

        Picks 1–3 episodes that fit this specific user via a mini-tier LLM
        call grounded on the user's profile, hidden personas, and top
        hashtags. Places each event's first/last_seen timestamps at random
        inside the user's observation window and bundles them into a
        single HiddenPersona with `is_synthetic=True` and a populated
        `events` list. Episode picks are diverse (no two episodes share
        the same theme) and detailed (each carries a `specific_situation`
        string grounded in the user's profile). LLM-only — no template
        fallback; if the LLM call fails the user gets no
        `sensitive_life_event` persona.

        Returns None when:
          - the user has less than 1 day of observation window (degenerate),
          - no LLM client is available (e.g. subagent mode), OR
          - the LLM output is unusable.
        """
        if not self.interactions or not self.user_profile:
            return None

        all_ts = [int(r.interaction_time) for r in self.interactions]
        obs_start = min(all_ts)
        obs_end = max(all_ts)
        min_span_secs = SENSITIVE_LIFE_EVENT_SPAN_DAYS[0] * 86400
        if obs_end - obs_start < min_span_secs:
            return None

        rng = random.Random(f"sensitive_life_event:{self.user_id}")
        n_events = rng.randint(SENSITIVE_LIFE_EVENT_MIN_PER_USER,
                               SENSITIVE_LIFE_EVENT_MAX_PER_USER)

        chosen_specs = self._pick_sensitive_life_event_specs(n_events)
        if not chosen_specs:
            return None

        events_out: list[dict] = []
        union_hashtags: list[str] = []
        seen_tags: set[str] = set()
        all_first: list[int] = []
        all_last: list[int] = []
        label_fragments: list[str] = []

        for spec in chosen_specs:
            span_days = rng.randint(SENSITIVE_LIFE_EVENT_SPAN_DAYS[0],
                                    SENSITIVE_LIFE_EVENT_SPAN_DAYS[1])
            span_secs = span_days * 86400
            latest_first = max(obs_start, obs_end - span_secs)
            first_seen = rng.randint(obs_start, latest_first)
            last_seen = first_seen + span_secs
            active_window_end = last_seen + SENSITIVE_LIFE_EVENT_BUFFER_DAYS * 86400

            events_out.append({
                "topic": spec["topic"],
                "label_fragment": spec["label_fragment"],
                "specific_situation": spec.get("specific_situation", ""),
                "first_seen_ts": first_seen,
                "last_seen_ts": last_seen,
                "active_window_end": active_window_end,
                "evidence_hashtags": list(spec["evidence_hashtags"]),
                "exemplar_persona_items": list(spec["exemplar_persona_items"]),
            })

            for tag in spec["evidence_hashtags"]:
                if tag.lower() not in seen_tags:
                    union_hashtags.append(tag)
                    seen_tags.add(tag.lower())
            all_first.append(first_seen)
            all_last.append(last_seen)
            label_fragments.append(spec["label_fragment"])

        label = "Currently navigating: " + "; ".join(label_fragments)
        description = (
            "Discrete, time-bounded personal episodes the user is actively "
            "processing. Surfacing these topics in unrelated contexts is the "
            "failure mode tested by `over_personalization_sensitive_event`."
        )
        motivation = (
            "Episodes like divorce, surgery, breakup, or gender exploration "
            "are sensitive in a way that makes recommendation-style "
            "personalization feel intrusive even when topical overlap exists."
        )

        spread_days = max(1, (max(all_last) - min(all_first)) // 86400)
        return HiddenPersona(
            label=label,
            type="sensitive_life_event",
            description=description,
            evidence_hashtags=union_hashtags,
            evidence_rows=0,
            evidence_oids=[],
            evidence_row_fraction=0.0,
            interaction_breakdown={},
            privacy_ratio=1.0,
            temporal_spread_days=spread_days,
            app_distribution={},
            surface_connections=[],
            inferred_motivation=motivation,
            already_captured=False,
            first_seen_ts=min(all_first),
            last_seen_ts=max(all_last),
            is_synthetic=True,
            events=events_out,
        )

    def _pick_sensitive_life_event_specs(
        self,
        n_events: int,
    ) -> list[dict]:
        """Return up to `n_events` LLM-generated sensitive-event specs.

        Single LLM call (mini-tier when available, else flagship). The LLM
        picks topics from `SENSITIVE_LIFE_EVENT_TOPIC_MENU` and writes ALL
        user-facing text — label fragment, situational detail, hashtags,
        and exemplar engagement items — grounded in this user's profile,
        hidden personas, and top hashtags. Returns [] when no client is
        available or the LLM output is unusable; caller skips the
        sensitive_life_event injection in that case.
        """
        from collections import Counter
        tag_counter: Counter = Counter()
        for row in self.interactions:
            for tag in self._extract_hashtags(row.object_text):
                tag_counter[tag] += 1
        top_tags = [t for t, _ in tag_counter.most_common(60)]

        hp_brief = []
        for hp in (self.user_profile.hidden_personas or []):
            if hp.type == "sensitive_life_event":
                continue
            hp_brief.append({
                "type": hp.type,
                "label": hp.label,
                "description": hp.description,
            })

        client = self.llm_client_mini or self.llm_client
        if client is None:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] sensitive_life_event "
                      f"injection skipped: no LLM client.{utils.Colors.ENDC}")
            return []

        valid_topics = {c["topic"] for c in SENSITIVE_LIFE_EVENT_TOPIC_MENU}
        try:
            prompt_text = prompts.personalize_sensitive_life_event_prompt(
                n_events=n_events,
                topic_menu=SENSITIVE_LIFE_EVENT_TOPIC_MENU,
                profile={
                    "gender": self.user_profile.gender,
                    "race_ethnicity": self.user_profile.race_ethnicity,
                    "career": self.user_profile.career,
                    "education": self.user_profile.education,
                    "bio": self.user_profile.bio,
                },
                hidden_personas_brief=hp_brief,
                top_hashtags=top_tags,
            )
            response = self._query_mini_with_retry(prompt_text)
            parsed = utils.extract_json_from_response(response) if response else None
            if not isinstance(parsed, list) or not parsed:
                return []
            return self._normalize_sensitive_specs(parsed, valid_topics, n_events)
        except Exception as e:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] sensitive_life_event "
                      f"LLM personalization failed ({e}); skipping injection.{utils.Colors.ENDC}")
            return []

    def _plant_sensitive_event_evidence_rows(self, per_app: dict) -> None:
        """For every event in the user's `sensitive_life_event` hidden
        persona, insert 2–4 LLM-generated implicit_positive engagement rows
        on a chosen social app, scattered across the event's
        `[first_seen_ts, last_seen_ts]` window.

        These planted rows are what the eval agent SEES at test time —
        without them the `over_personalization_sensitive_event` task has
        no signal to test restraint against (the synthetic
        sensitive_life_event cluster is in `profile.json`, but profile.json
        is firewalled off in every eval mode, so the agent only ever sees
        per-app event lists). With them, the agent reads the rows in the
        time-masked snapshot and the test fires when it leans on those
        themes in response to a benign off-topic query.

        LLM-generated content per event; no template fallback. Mutates
        `per_app` in place. Skips silently when there's no LLM client (e.g.
        subagent mode), no sensitive_life_event cluster, or the LLM call
        fails for an event.
        """
        if not self.user_profile or not self.user_profile.hidden_personas:
            return
        client = self.llm_client_mini or self.llm_client
        if client is None:
            return

        se = next(
            (hp for hp in self.user_profile.hidden_personas
             if hp.type == "sensitive_life_event"),
            None,
        )
        if se is None or not se.events:
            return

        # Map our internal app constants ("Instagram"/"Facebook"/"Threads")
        # against the per_app dict keys, which use the same casing.
        social_apps = [a for a in ("Instagram", "Facebook", "Threads") if a in per_app]
        if not social_apps:
            return

        rng = random.Random(f"sensitive_event_plant:{self.user_id}")
        n_planted_total = 0

        for ev_idx, ev in enumerate(se.events):
            first = int(ev.get("first_seen_ts") or 0)
            last = int(ev.get("last_seen_ts") or 0)
            if first <= 0 or last <= first:
                continue
            span_secs = last - first
            n_rows = rng.randint(2, 4)
            # Rotate target app across episodes so multi-event users get
            # signal spread across apps.
            target_app = social_apps[ev_idx % len(social_apps)]

            try:
                prompt_text = prompts.generate_sensitive_event_evidence_rows_prompt(
                    profile={
                        "gender": self.user_profile.gender,
                        "race_ethnicity": self.user_profile.race_ethnicity,
                        "career": self.user_profile.career,
                        "education": self.user_profile.education,
                        "bio": self.user_profile.bio,
                    },
                    sensitive_event={
                        "topic": ev.get("topic", ""),
                        "label_fragment": ev.get("label_fragment", ""),
                        "specific_situation": ev.get("specific_situation", ""),
                        "evidence_hashtags": ev.get("evidence_hashtags", []),
                    },
                    n_rows=n_rows,
                    app=target_app,
                    span_seconds=span_secs,
                )
                response = self._query_mini_with_retry(prompt_text)
                parsed = utils.extract_json_from_response(response) if response else None
            except Exception as e:
                if self.verbose:
                    print(f"{utils.Colors.WARNING}[User {self.user_id}] sensitive_event "
                          f"evidence-row LLM call failed for topic={ev.get('topic')}: "
                          f"{e}{utils.Colors.ENDC}")
                continue
            if not isinstance(parsed, list) or not parsed:
                continue

            # Sample one implicit_positive action verbatim from the
            # platform's catalog so the planted rows look identical in
            # shape to organic events.
            app_formats = PLATFORM_INTERACTION_FORMATS.get(target_app, {})
            ip_actions = app_formats.get("implicit_positive") or []
            if not ip_actions:
                continue

            for row_idx, row in enumerate(parsed[:n_rows]):
                if not isinstance(row, dict):
                    continue
                offset = row.get("ts_offset_seconds", 0)
                try:
                    offset = max(0, min(int(offset), span_secs))
                except (TypeError, ValueError):
                    offset = rng.randint(0, span_secs)
                ts = first + offset
                tags = row.get("hashtags") or []
                if not isinstance(tags, list):
                    continue
                norm_tags: list[str] = []
                for t in tags:
                    t_str = str(t).strip().lower()
                    if not t_str:
                        continue
                    if not t_str.startswith("#"):
                        t_str = "#" + t_str.lstrip("#")
                    norm_tags.append(t_str)
                # Require at least 2 hashtags from the episode's evidence
                # set so the planted row is detectable as "evidence" even
                # under loose hashtag drift from the LLM.
                ev_tags_lower = {h.lower() for h in (ev.get("evidence_hashtags") or [])}
                overlap = sum(1 for h in norm_tags if h.lower() in ev_tags_lower)
                if overlap < 2:
                    # Backfill from the episode's evidence_hashtags to meet the floor.
                    for h in (ev.get("evidence_hashtags") or []):
                        if h.lower() not in {x.lower() for x in norm_tags}:
                            norm_tags.append(h.lower())
                            overlap += 1
                            if overlap >= 2:
                                break

                title = str(row.get("title", "") or "").strip()
                caption = str(row.get("caption", "") or "").strip()
                if not caption:
                    continue

                action_entry = rng.choice(ip_actions)
                oid = f"sensitive_event_{self.user_id}_{ev.get('topic', 'na')}_{ev_idx:02d}_{row_idx:02d}"
                planted_event = {
                    "source_object_id": oid,
                    "source_timestamp": ts,
                    "formatted_timestamp": self._format_timestamp(ts),
                    "source_hashtags": norm_tags,
                    "source_interaction_type": "implicit_positive",
                    "interaction_format": {
                        "app": target_app,
                        "action": action_entry["action"],
                        "action_label": action_entry["label"],
                        "user_message": None,
                    },
                    "content_type": "text",
                    "content": {
                        "title": title,
                        "caption": caption,
                        "overall_description": "",
                    },
                    "preferences": [],
                    "is_self_authored": False,
                    "_planted_sensitive_event": ev.get("topic", ""),
                }
                per_app.setdefault(target_app, []).append(planted_event)
                n_planted_total += 1

        if self.verbose and n_planted_total:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Planted "
                  f"{n_planted_total} sensitive_event evidence rows.{utils.Colors.ENDC}")

    def _normalize_sensitive_specs(
        self,
        parsed: list,
        valid_topics: set,
        n_events: int,
    ) -> list[dict]:
        """Validate + lightly normalize the LLM's pick list. Drops malformed
        entries (unknown topic, missing label_fragment / hashtags / items),
        de-duplicates by topic, and trims hashtags to lowercase `#tag` form.
        Never substitutes templated text — a partial entry is dropped, not
        back-filled. Returns at most `n_events` specs.
        """
        seen_topics: set[str] = set()
        out: list[dict] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            topic = (entry.get("topic") or "").strip().lower()
            if topic not in valid_topics or topic in seen_topics:
                continue
            label_fragment = (entry.get("label_fragment") or "").strip()
            tags = entry.get("evidence_hashtags") or []
            items = entry.get("exemplar_persona_items") or []
            specific = (entry.get("specific_situation") or "").strip()
            if not label_fragment or not isinstance(tags, list) or not isinstance(items, list):
                continue
            norm_tags = []
            for t in tags:
                t_str = str(t).strip().lower()
                if not t_str:
                    continue
                if not t_str.startswith("#"):
                    t_str = "#" + t_str.lstrip("#")
                norm_tags.append(t_str)
            norm_items = [str(i).strip() for i in items if str(i).strip()]
            if not norm_tags or not norm_items:
                continue
            out.append({
                "topic": topic,
                "label_fragment": label_fragment,
                "specific_situation": specific,
                "evidence_hashtags": norm_tags,
                "exemplar_persona_items": norm_items,
            })
            seen_topics.add(topic)
            if len(out) >= n_events:
                break
        return out

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

        # Tier A: MBTI inference is a structured map from (big_five,
        # hidden_personas, top_hashtags) → type + dim scores. Mini is
        # sufficient; fall back to flagship if mini client is unset.
        mbti_client = self.llm_client_mini or self.llm_client
        for attempt in range(self.MAX_RETRIES):
            try:
                response = mbti_client.query_llm(prompt_text)
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

    def _sample_source_rows_for_voice(self, max_rows: int = 10) -> list[dict]:
        """Stratified sample of raw interaction rows for voice grounding.

        Picks across interaction_types so the LLM sees the user's real
        engagement (likes, dislikes, searches, scrolls) rather than only the
        LLM-summarized top personas. Truncates object_text to keep the prompt
        compact.
        """
        if not getattr(self, "interactions", None):
            return []

        # Group by interaction_type for stratified pick
        by_type: dict[str, list] = {}
        for row in self.interactions:
            by_type.setdefault(row.interaction_type, []).append(row)

        # Round-robin pull a few from each bucket (most-textual first)
        ordered_types = sorted(
            by_type.keys(),
            key=lambda t: -sum(len(r.object_text or "") for r in by_type[t]),
        )
        picked: list = []
        idx_per_type = {t: 0 for t in ordered_types}
        # Sort each bucket by descending object_text length so we get content-rich rows
        for t in ordered_types:
            by_type[t].sort(key=lambda r: -len(r.object_text or ""))
        while len(picked) < max_rows:
            progressed = False
            for t in ordered_types:
                i = idx_per_type[t]
                if i < len(by_type[t]):
                    row = by_type[t][i]
                    idx_per_type[t] = i + 1
                    progressed = True
                    if (row.object_text or "").strip():
                        picked.append(row)
                        if len(picked) >= max_rows:
                            break
            if not progressed:
                break

        return [
            {
                "interaction_time": r.interaction_time,
                "interaction_type": r.interaction_type,
                "object_text": r.object_text,
            }
            for r in picked
        ]

    def generate_app_personas(self) -> None:
        """Step 8 — Generate writing voice (Layers 1+2+3) + four AppPersona modulations.

        Two LLM calls (real people have ONE voice; per-app modulation only
        SELECTS from the existing repertoire):

          Call A — `generate_voice_core_prompt` produces the stable
                   user_voice block (identity_spine + idiolect + repertoire +
                   soft holdovers). Cached on `self.user_profile.user_voice`
                   so re-running Step 8 doesn't redo Layer 1+2+3 unless a
                   full regen is requested.

          Call B — `generate_app_modulations_prompt` produces the four
                   AppPersona entries. Each app's `active_*` lists must be
                   subsets of the corresponding `repertoire.*`. At least 2
                   of the 4 apps must differ on `active_stances` by ≥1
                   element. On violation, re-prompt once.

        Outputs (asdict'd onto user_profile):
          self.user_profile.user_voice  — Layers 1+2+3 + soft holdovers
          self.user_profile.app_personas — dict[app_name -> AppPersona dict]
        """
        if not self.user_profile:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] No profile — skipping app persona generation.{utils.Colors.ENDC}")
            return
        if self.llm_client is None:
            # Subagent mode does this inline per skill.md; persona_agent.py is
            # only used in API mode. Nothing to do here in subagent mode.
            return

        # ---- Grounding shared by both calls --------------------------------
        top_personas = [
            p.persona_item
            for p in sorted(
                self.cross_referenced_personas,
                key=lambda x: (x.confidence_score_init + x.confidence_cross_referenced),
                reverse=True,
            )[:30]  # was 20 — Layer 2 needs a slightly broader inventory
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

        # Layer-2 needs broader stylometric grounding — was 10, now 20.
        source_samples = self._sample_source_rows_for_voice(max_rows=20)

        # Pass hidden-persona summary so identity_spine motifs can cite them.
        # `frame` carries the cluster's dominant motivation frame slug
        # (audited if available, else the structural default for the
        # cluster's type) — anchors the voice prompt's motif guidance
        # in a named academic framework instead of vibes.
        hp_summary = []
        for hp in (self.user_profile.hidden_personas or []):
            frame = prompts.cluster_dominant_frame(hp)
            hp_summary.append({
                "label": getattr(hp, "label", "") or "",
                "persona_type": getattr(hp, "type", "") or "",
                "signal_strength": getattr(hp, "signal_strength", "") or "",
                "frame": frame,
                "frame_description": prompts.FRAME_DESCRIPTIONS.get(frame, ""),
            })

        # Sensitive-life-event topics (background context for motif grounding;
        # the prompt instructs the LLM NOT to name them as motifs directly).
        sle_topics: list[str] = []
        for hp in (self.user_profile.hidden_personas or []):
            if getattr(hp, "type", "") == "sensitive_life_event":
                for ev in (getattr(hp, "events", None) or []):
                    topic = ev.get("topic_label") if isinstance(ev, dict) else None
                    if topic:
                        sle_topics.append(topic)

        # =====================================================================
        # Call A — voice core (Layers 1+2+3 + soft holdovers)
        # =====================================================================
        if self.user_profile.user_voice and self.user_profile.user_voice.get("identity_spine"):
            # Layer 1+2+3 already cached on profile (e.g. re-running just Step
            # 8 to refresh per-app modulations). Skip Call A.
            user_voice_dict = self.user_profile.user_voice
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Voice core cached — skipping Call A.{utils.Colors.ENDC}")
        else:
            prompt_a = prompts.generate_voice_core_prompt(
                profile=profile_dict,
                top_personas=top_personas,
                source_samples=source_samples,
                hidden_persona_summary=hp_summary,
                sensitive_event_topics=sle_topics,
            )
            response_a = self._query_llm_with_retry(prompt_a)
            if not response_a:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] Voice core (Call A) generation failed.{utils.Colors.ENDC}")
                return
            parsed_a = utils.extract_json_from_response(response_a)
            if not isinstance(parsed_a, dict):
                print(f"{utils.Colors.WARNING}[User {self.user_id}] Unparseable Call A response.{utils.Colors.ENDC}")
                return

            uv_raw = parsed_a.get("user_voice") or {}
            if not (isinstance(uv_raw, dict) and uv_raw):
                print(f"{utils.Colors.WARNING}[User {self.user_id}] No user_voice in Call A response — leaving empty.{utils.Colors.ENDC}")
                self.user_profile.user_voice = {}
                return

            # Cap catchphrase_residue at 2 (defense in depth — prompt says ≤2,
            # we enforce it here too).
            idio = uv_raw.get("idiolect") or {}
            if isinstance(idio, dict):
                residue = list(idio.get("catchphrase_residue") or [])
                if len(residue) > 2:
                    residue = residue[:2]
                idio["catchphrase_residue"] = residue
                uv_raw["idiolect"] = idio

            user_voice_obj = UserVoice(
                identity_spine=uv_raw.get("identity_spine") or {},
                idiolect=uv_raw.get("idiolect") or {},
                repertoire=uv_raw.get("repertoire") or {},
                natural_register=str(uv_raw.get("natural_register", "")),
                humor_tone=str(uv_raw.get("humor_tone", "")),
                default_capitalization=str(uv_raw.get("default_capitalization", "")),
                punctuation_habits=str(uv_raw.get("punctuation_habits", "")),
                emoji_palette=list(uv_raw.get("emoji_palette", []) or []),
                emoji_intensity_default=str(uv_raw.get("emoji_intensity_default", "medium")),
                formality_baseline=float(uv_raw.get("formality_baseline", 0.3) or 0.3),
                voice_avoid=str(uv_raw.get("voice_avoid", "")),
                phrases_to_avoid=list(uv_raw.get("phrases_to_avoid", []) or []),
            )
            self.user_profile.user_voice = asdict(user_voice_obj)
            user_voice_dict = self.user_profile.user_voice

        # =====================================================================
        # Call B — per-app modulations (Layer-3 selection + Layer-4 surface)
        # =====================================================================
        rep = (user_voice_dict.get("repertoire") or {})
        repertoire_stances = set(rep.get("stances") or [])
        repertoire_registers = set(rep.get("registers") or [])
        repertoire_genres = set(rep.get("speech_genre_fluency") or [])

        def _parse_call_b(parsed_b: dict) -> tuple[dict[str, AppPersona], list[str]]:
            """Parse Call B output and validate. Returns (app_personas, violations)."""
            block = parsed_b.get("app_personas") or {}
            if not isinstance(block, dict):
                return {}, ["app_personas_not_dict"]

            built: dict[str, AppPersona] = {}
            violations: list[str] = []
            for app_name in PLATFORMS:
                entry = block.get(app_name)
                if not isinstance(entry, dict):
                    violations.append(f"{app_name}:missing_entry")
                    continue
                a_stances = list(entry.get("active_stances") or [])
                a_regs = list(entry.get("active_registers") or [])
                a_genres = list(entry.get("active_speech_genres") or [])

                # Subset rule: drop offending elements (we'll detect rejections
                # by counting drops — if any non-trivial drop happened, that's
                # a violation that warrants a re-prompt).
                offending_stances = [s for s in a_stances if repertoire_stances and s not in repertoire_stances]
                offending_regs = [r for r in a_regs if repertoire_registers and r not in repertoire_registers]
                offending_genres = [g for g in a_genres if repertoire_genres and g not in repertoire_genres]
                if offending_stances:
                    violations.append(f"{app_name}:stances_not_subset:{offending_stances}")
                if offending_regs:
                    violations.append(f"{app_name}:registers_not_subset:{offending_regs}")
                if offending_genres:
                    violations.append(f"{app_name}:genres_not_subset:{offending_genres}")

                # Sanitize — keep only valid subset members on the kept persona.
                if repertoire_stances:
                    a_stances = [s for s in a_stances if s in repertoire_stances]
                if repertoire_registers:
                    a_regs = [r for r in a_regs if r in repertoire_registers]
                if repertoire_genres:
                    a_genres = [g for g in a_genres if g in repertoire_genres]

                surface = entry.get("surface") or {}
                if not isinstance(surface, dict):
                    surface = {}
                idiolect_overrides = entry.get("idiolect_overrides") or {}
                if not isinstance(idiolect_overrides, dict):
                    idiolect_overrides = {}

                built[app_name] = AppPersona(
                    app_name=app_name,
                    active_stances=a_stances,
                    active_registers=a_regs,
                    active_speech_genres=a_genres,
                    use_purposes=list(entry.get("use_purposes", [])),
                    friend_zones=list(entry.get("friend_zones", [])),
                    audience_type=entry.get("audience_type", "mixed"),
                    audience_lens=str(entry.get("audience_lens", "")),
                    audience_design_note=str(entry.get("audience_design_note", "")),
                    posting_frequency=entry.get("posting_frequency", "weekly"),
                    topical_focus=list(entry.get("topical_focus", [])),
                    chatbot_contexts=list(entry.get("chatbot_contexts", [])) if app_name == "Chatbot" else [],
                    surface=surface,
                    idiolect_overrides=idiolect_overrides,
                    app_avoid=str(entry.get("app_avoid", "")),
                    delta_summary=str(entry.get("delta_summary", "")),
                )

            # Diversity rule: ≥2 apps differ on active_stances by ≥1 element.
            if built:
                stance_sets = [tuple(sorted(set(p.active_stances))) for p in built.values()]
                pairs_distinct = any(
                    stance_sets[i] != stance_sets[j]
                    for i in range(len(stance_sets))
                    for j in range(i + 1, len(stance_sets))
                )
                if not pairs_distinct and len(built) >= 2:
                    violations.append("diversity_rule:all_apps_same_stance_set")
            return built, violations

        # Tag source samples by inferred app for Call B grounding (best-effort —
        # session-based routing happens later, but the LLM benefits from at
        # least a hint).
        samples_for_b = list(source_samples)

        prompt_b = prompts.generate_app_modulations_prompt(
            profile=profile_dict,
            user_voice=user_voice_dict,
            chatbot_contexts=CHATBOT_CONTEXTS,
            source_samples_by_app=samples_for_b,
            hidden_persona_summary=hp_summary,
        )
        response_b = self._query_llm_with_retry(prompt_b)
        if not response_b:
            print(f"{utils.Colors.WARNING}[User {self.user_id}] App modulations (Call B) generation failed.{utils.Colors.ENDC}")
            return
        parsed_b = utils.extract_json_from_response(response_b)
        if not isinstance(parsed_b, dict):
            print(f"{utils.Colors.WARNING}[User {self.user_id}] Unparseable Call B response.{utils.Colors.ENDC}")
            return

        app_personas, violations = _parse_call_b(parsed_b)
        if violations:
            # One re-prompt with explicit violation list. After that, accept
            # whatever we have (sanitization already dropped offending items).
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] Call B violations on first pass: {violations}; re-prompting once.{utils.Colors.ENDC}")
            retry_prompt = (
                prompt_b
                + "\n\n## RE-PROMPT — your previous output had these violations:\n"
                + "\n".join(f"- {v}" for v in violations)
                + "\n\nFix them and re-emit the JSON. Subset rule and diversity rule are both required."
            )
            retry_response = self._query_llm_with_retry(retry_prompt)
            if retry_response:
                retry_parsed = utils.extract_json_from_response(retry_response)
                if isinstance(retry_parsed, dict):
                    retry_built, retry_violations = _parse_call_b(retry_parsed)
                    if not retry_violations and retry_built:
                        app_personas = retry_built
                    elif retry_built and len(retry_violations) < len(violations):
                        app_personas = retry_built  # accept partial improvement

        self.user_profile.app_personas = {k: asdict(v) for k, v in app_personas.items()}

        if self.verbose:
            n_idio_overrides = sum(1 for ap in app_personas.values() if ap.idiolect_overrides)
            stance_sets = sorted({tuple(sorted(set(ap.active_stances))) for ap in app_personas.values()})
            print(
                f"{utils.Colors.OKGREEN}[User {self.user_id}] Generated user_voice (4-layer) + "
                f"{len(app_personas)} app personas "
                f"(distinct stance subsets: {len(stance_sets)}; "
                f"{n_idio_overrides}/{len(app_personas)} apps with idiolect_overrides).{utils.Colors.ENDC}"
            )

    def generate_ai_studio_persona(self) -> None:
        """Step 11C — pick ONE AI Studio persona archetype for this user and
        write the full AIStudioPersona block onto profile.json.

        Slots AFTER `generate_app_personas` (Step 11) and BEFORE Steps 13/14
        (preference→app + session→app routing) — milestone (b) and (c) of the
        AI Studio rollout will need to know the user's archetype to route
        hidden-persona-anchored canonicals into AI_Studio correctly.

        One mini-tier LLM call (falls back to flagship). Result cached on
        `self.user_profile.ai_studio_persona`; skipped on re-run unless
        cleared. Validation:
          - archetype ∈ AI_STUDIO_ARCHETYPES
          - signature_phrases len ≤ 3
          - forbidden_phrases includes the full Rogers-cliché baseline
            (back-filled if missing)
          - topical_strengths overlaps ≥1 hidden_persona type the user has
          - if archetype == "niche_expert_creator_ai", niche_specifier set
          - if archetype == "romantic_partner":
              * romantic_specifier present + well-formed
              * auto-disable check: NOT high-acuity active sensitive_life_event
                — on violation, fall back to `late_night_best_friend`
              * explicitness_band == "erotic_explicit" only when adult signal
        On total failure (no LLM, unparseable, validation can't recover):
        leaves `ai_studio_persona = {}`. Downstream Step 13/14 routing skips
        AI_Studio for the user in that case.
        """
        from collections import Counter

        if not self.user_profile:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] No profile — skipping AI Studio persona generation.{utils.Colors.ENDC}")
            return
        if self.user_profile.ai_studio_persona:
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] AI Studio persona cached — skipping.{utils.Colors.ENDC}")
            return

        client = getattr(self, "llm_client_mini", None) or getattr(self, "llm_client", None)
        if client is None:
            # Subagent mode handles this inline per skill.md; persona_agent.py
            # is API mode only — nothing to do.
            return

        # ---- Grounding ---------------------------------------------------
        profile_dict = {
            "name": self.user_profile.name,
            "gender": self.user_profile.gender,
            "race_ethnicity": self.user_profile.race_ethnicity,
            "career": self.user_profile.career,
            "education": self.user_profile.education,
            "bio": self.user_profile.bio,
        }

        # Top hashtags (signal niche specifier + romantic sub-typing)
        tag_counter: Counter = Counter()
        for row in self.interactions:
            for tag in self._extract_hashtags(row.object_text):
                tag_counter[tag] += 1
        top_hashtags = [t for t, _ in tag_counter.most_common(60)]

        # Hidden persona brief
        hp_brief = []
        for hp in (self.user_profile.hidden_personas or []):
            hp_brief.append({
                "persona_type": hp.type,
                "label": hp.label,
                "description": hp.description,
            })

        # Sensitive-life-event acuity — gates `romantic_partner` off when
        # high-acuity active. Acuity is approximated by event count + recency
        # (no explicit acuity field on HiddenPersona today; we use the
        # heuristic: any active sensitive_life_event in active window with
        # 3+ episodes counts as high-acuity for romantic-gate purposes).
        sensitive_event_topics: list[str] = []
        sensitive_event_acuity: dict[str, str] = {}
        has_high_acuity_active_sle = False
        for hp in (self.user_profile.hidden_personas or []):
            if hp.type != "sensitive_life_event":
                continue
            for ev in (hp.events or []):
                if not isinstance(ev, dict):
                    continue
                topic = ev.get("topic") or ev.get("topic_label") or ""
                if not topic:
                    continue
                # active = within active_window_end
                active_end = ev.get("active_window_end") or ev.get("last_seen_ts") or 0
                # heuristic acuity: 'high' if topic touches abuse / suicide /
                # acute mental health / acute medical, else 'low'.
                high_acuity_topics = {
                    "suicide_ideation_proxy", "abuse_recovery",
                    "mental_health_diagnosis", "addiction_recovery",
                }
                acu = "high" if topic in high_acuity_topics else "low"
                sensitive_event_topics.append(topic)
                sensitive_event_acuity[topic] = acu
                if acu == "high" and active_end:
                    has_high_acuity_active_sle = True

        # Locale country — mode of event_location countries if available, else "US"
        locale_country = self._infer_locale_country()

        # Build the menu (name + voice_template + restrictions)
        archetypes_menu = [
            {"name": k, **v} for k, v in AI_STUDIO_ARCHETYPES.items()
        ]

        # ---- LLM call ----------------------------------------------------
        try:
            prompt_text = prompts.personalize_ai_studio_persona_prompt(
                profile=profile_dict,
                user_voice=self.user_profile.user_voice or {},
                app_personas=self.user_profile.app_personas or {},
                hidden_personas_brief=hp_brief,
                sensitive_event_topics=sensitive_event_topics,
                sensitive_event_acuity=sensitive_event_acuity,
                top_hashtags=top_hashtags,
                archetypes_menu=archetypes_menu,
                rogers_cliche_baseline=ROGERS_CLICHE_BLOCKLIST,
                locale_country=locale_country,
            )
            response = self._query_mini_with_retry(prompt_text)
            parsed = utils.extract_json_from_response(response) if response else None
        except Exception as e:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] AI Studio persona LLM call failed ({e}); leaving empty.{utils.Colors.ENDC}")
            return

        if not isinstance(parsed, dict):
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] AI Studio persona: unparseable LLM response; leaving empty.{utils.Colors.ENDC}")
            return

        # ---- Validation + sanitization ----------------------------------
        archetype = parsed.get("persona_archetype") or ""
        if archetype not in AI_STUDIO_ARCHETYPES:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] AI Studio persona: invalid archetype {archetype!r}; falling back to late_night_best_friend.{utils.Colors.ENDC}")
            archetype = "late_night_best_friend"

        # Romantic auto-disable on high-acuity active sensitive_life_event
        arch_meta = AI_STUDIO_ARCHETYPES[archetype]
        if arch_meta.get("auto_disable_on_high_acuity_sensitive_event") and has_high_acuity_active_sle:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] AI Studio persona: {archetype} auto-disabled on high-acuity sensitive_life_event; falling back to late_night_best_friend.{utils.Colors.ENDC}")
            archetype = "late_night_best_friend"
            # Wipe any romantic_specifier the LLM may have produced.
            parsed["romantic_specifier"] = {}

        # 4-layer voice structure — extract + lightly sanitize
        identity_spine = parsed.get("identity_spine") or {}
        idiolect = parsed.get("idiolect") or {}
        repertoire = parsed.get("repertoire") or {}
        if not isinstance(identity_spine, dict):
            identity_spine = {}
        if not isinstance(idiolect, dict):
            idiolect = {}
        if not isinstance(repertoire, dict):
            repertoire = {}

        # idiolect.catchphrase_residue ≤ 3 (defense-in-depth — prompt says ≤3)
        residue = list(idiolect.get("catchphrase_residue") or [])
        if len(residue) > 3:
            residue = residue[:3]
        idiolect["catchphrase_residue"] = residue

        # constructional_templates is a list of dicts with required keys
        ct_raw = idiolect.get("constructional_templates") or []
        if not isinstance(ct_raw, list):
            ct_raw = []
        ct_clean = []
        for item in ct_raw[:6]:  # cap at 6 to bound LLM blow-out
            if isinstance(item, dict) and item.get("pattern"):
                ct_clean.append({
                    "pattern": str(item.get("pattern", "")),
                    "example_realization": str(item.get("example_realization", "")),
                    "frequency": str(item.get("frequency", "")),
                })
        idiolect["constructional_templates"] = ct_clean

        # repertoire.stances 3–6 short labels
        stances = list(repertoire.get("stances") or [])[:6]
        repertoire["stances"] = stances

        # signature_phrases mirror catchphrase_residue (≤3); top-level
        # convenience field, same content.
        sigs = list(parsed.get("signature_phrases") or residue)[:3]

        # forbidden_phrases must include the Rogers baseline + archetype-specific
        forb = list(parsed.get("forbidden_phrases") or [])
        archetype_forb = list(AI_STUDIO_ARCHETYPES[archetype].get("forbidden_phrases", []))
        # Set-merge while preserving baseline order
        seen = set()
        merged_forb = []
        for p in list(ROGERS_CLICHE_BLOCKLIST) + archetype_forb + forb:
            key = p.strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged_forb.append(p)

        # niche_specifier required for niche_expert_creator_ai
        niche_specifier = parsed.get("niche_specifier")
        if archetype == "niche_expert_creator_ai" and not niche_specifier:
            # Synthesize a fallback from top hashtags' modal cluster
            niche_specifier = (top_hashtags[0] if top_hashtags else "general") + "-coach"
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] AI Studio persona: niche_expert_creator_ai missing niche_specifier; using fallback {niche_specifier!r}.{utils.Colors.ENDC}")

        # romantic_specifier required for romantic_partner — validate vocabularies
        romantic_specifier_dict: dict = {}
        if archetype == "romantic_partner":
            rs_raw = parsed.get("romantic_specifier") or {}
            if not isinstance(rs_raw, dict):
                rs_raw = {}

            def _enum_or_none(val, vocab):
                v = val if isinstance(val, str) else None
                return v if v in vocab else None

            rs = RomanticSpecifier(
                gender_presentation=_enum_or_none(rs_raw.get("gender_presentation"), ROMANTIC_GENDER_PRESENTATIONS),
                sexuality_orientation=_enum_or_none(rs_raw.get("sexuality_orientation"), ROMANTIC_SEXUALITY_ORIENTATIONS),
                aesthetic_vibe=_enum_or_none(rs_raw.get("aesthetic_vibe"), ROMANTIC_AESTHETIC_VIBES),
                body_role_coding=_enum_or_none(rs_raw.get("body_role_coding"), ROMANTIC_BODY_ROLE_CODINGS),
                relational_dynamic=_enum_or_none(rs_raw.get("relational_dynamic"), ROMANTIC_RELATIONAL_DYNAMICS),
                explicitness_band=(
                    rs_raw.get("explicitness_band")
                    if rs_raw.get("explicitness_band") in ROMANTIC_EXPLICITNESS_BANDS
                    else "sensual"
                ),
            )
            romantic_specifier_dict = asdict(rs)

        # topical_strengths overlap with ≥1 hidden_persona type — best-effort
        # (don't reject, just warn if absent).
        ts_list = list(parsed.get("topical_strengths") or [])
        if self.verbose and hp_brief:
            hp_types = {h.get("persona_type", "") for h in hp_brief}
            ts_lower = " ".join(s.lower() for s in ts_list)
            # Heuristic: at least one hp_type word in topical_strengths.
            type_words = {t.replace("_", " ") for t in hp_types if t}
            if type_words and not any(w in ts_lower for w in type_words):
                print(f"{utils.Colors.WARNING}[User {self.user_id}] AI Studio persona: topical_strengths {ts_list!r} doesn't obviously overlap any hidden_persona type {hp_types!r}; accepting anyway.{utils.Colors.ENDC}")

        # Build the dataclass (clamps defaults for any missing fields)
        ai_persona = AIStudioPersona(
            persona_archetype=archetype,
            character_name=str(parsed.get("character_name", "")),
            backstory_brief=str(parsed.get("backstory_brief", "")),
            relational_stance=str(parsed.get("relational_stance", "")),
            address_terms=list(parsed.get("address_terms", []) or [])[:3],
            self_reference_style=str(parsed.get("self_reference_style", "first_person")),
            communication_style=str(parsed.get("communication_style", "")),
            # 4-layer voice structure
            identity_spine=identity_spine,
            idiolect=idiolect,
            repertoire=repertoire,
            # Soft holdovers
            natural_register=str(parsed.get("natural_register", "")),
            default_capitalization=str(parsed.get("default_capitalization", "")),
            punctuation_habits=str(parsed.get("punctuation_habits", "")),
            humor_tone=str(parsed.get("humor_tone", "")),
            length_band=str(parsed.get("length_band", "medium")),
            emoji_palette=list(parsed.get("emoji_palette", []) or [])[:6],
            emoji_intensity_default=str(parsed.get("emoji_intensity_default", "low")),
            formality=float(parsed.get("formality", 0.3) or 0.3),
            # Negatives
            voice_avoid=str(parsed.get("voice_avoid", "")),
            forbidden_phrases=merged_forb,
            # Topical scope
            topical_strengths=ts_list,
            topical_avoid=list(parsed.get("topical_avoid", []) or [])[:3],
            # Signature phrases (mirrors idiolect.catchphrase_residue)
            signature_phrases=sigs,
            # Guardrails + routing + fit
            generation_guardrails=parsed.get("generation_guardrails") or {
                "boundary_on_diagnosis": "never_diagnose",
                "boundary_on_medication_advice": "decline_redirect_clinician",
                "anti_sycophancy_pledge": "challenge_assumptions_when_warranted",
                "honesty_when_asked_if_ai": "answer_truthfully",
                "no_real_public_figure_impersonation": True,
            },
            eligibility_signal=parsed.get("eligibility_signal") or {},
            fit_rationale=str(parsed.get("fit_rationale", "")),
            niche_specifier=niche_specifier,
            romantic_specifier=romantic_specifier_dict,
        )

        self.user_profile.ai_studio_persona = asdict(ai_persona)
        if self.verbose:
            print(
                f"{utils.Colors.OKGREEN}[User {self.user_id}] AI Studio persona: "
                f"archetype={archetype!r} character={ai_persona.character_name!r}"
                + (f" niche={niche_specifier!r}" if niche_specifier else "")
                + (f" romantic_axes={ {k:v for k,v in romantic_specifier_dict.items() if v} }" if romantic_specifier_dict else "")
                + f"{utils.Colors.ENDC}"
            )

    def _infer_locale_country(self) -> str:
        """Best-effort: modal `event_location.country` across the user's
        events, falling back to "US". Used to localize crisis-resource
        defaults in AI persona's generation_guardrails (informational only;
        we do not score safety).
        """
        from collections import Counter
        c: Counter = Counter()
        for row in self.interactions:
            loc = getattr(row, "event_location", None) or {}
            if isinstance(loc, dict):
                country = loc.get("country") or ""
                if country:
                    c[country] += 1
        if c:
            return c.most_common(1)[0][0]
        return "US"

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
            """Majority vote with Chatbot/AI_Studio tiebreak for non-negative rows.

            On a tie:
              * If implicit_negative → fall through to the first tied app
                (Chatbot and AI_Studio are firewalled in Step 4 anyway).
              * Otherwise → if AI_Studio is in the tie, prefer AI_Studio
                (companion-chat takes precedence on ties involving it),
                else prefer Chatbot if in the tie, else first tied app.
            """
            if not votes:
                return random.choice(SOCIAL_PLATFORMS) if itype == "implicit_negative" else random.choice(PLATFORMS)
            tallies = _Counter(votes).most_common()
            top_count = tallies[0][1]
            tied_apps = [a for a, c in tallies if c == top_count]
            if len(tied_apps) == 1:
                return tied_apps[0]
            if itype == "implicit_negative":
                # Pick first social app in tie, else first tie.
                for a in tied_apps:
                    if a in SOCIAL_PLATFORMS:
                        return a
                return tied_apps[0]
            if "AI_Studio" in tied_apps:
                return "AI_Studio"
            if "Chatbot" in tied_apps:
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
        for session in self._sessions:
            if random.random() < self.NOISE_REASSIGN_PROBABILITY:
                current_app = row_apps.get(session[0].object_id, PLATFORMS[0])
                alternatives = [a for a in PLATFORMS if a != current_app]
                new_app = random.choice(alternatives)
                for r in session:
                    row_apps[r.object_id] = new_app

        # Step 4: Never route implicit_negative to Chatbot OR AI_Studio —
        # redirect to social. AI_Studio is a privacy-floored companion
        # surface; implicit-negative ("I don't like X") signals don't fit
        # the relational frame.
        for r in self.interactions:
            if r.interaction_type == "implicit_negative" and row_apps.get(r.object_id) in ("Chatbot", "AI_Studio"):
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
            ai_studio_persona=self.user_profile.ai_studio_persona or None,
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

        # Quota rebalance: push Chatbot canonical share up to ~27% (was 0.40
        # pre-AI Studio), then carve out ~18% to AI_Studio for companion-chat
        # canonicals. Session-majority voting downstream washes out per-canonical
        # routing, so we pre-bias the canonical pool.
        self._quota_rebalance_apps()

        if self.verbose:
            from collections import Counter
            counts = Counter(cr.assigned_app for cr in self.cross_referenced_personas)
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Canonical app routing: {dict(counts)}{utils.Colors.ENDC}")

    # Target shares for the canonical-level distribution before session voting.
    # Milestone (b) carve-out: Chatbot drops from 0.40 → 0.27 to give AI_Studio
    # 0.18. Net effect: Chatbot keeps utility / knowledge / writing-help
    # canonicals; AI_Studio absorbs identity / aspiration / emotional /
    # intimate-interest / parasocial canonicals (companion-chat material).
    CHATBOT_CANONICAL_TARGET = 0.27
    AI_STUDIO_CANONICAL_TARGET = 0.18
    SOCIAL_CANONICAL_FLOOR = 0.17

    def _quota_rebalance_apps(self) -> None:
        """Enforce soft quotas on the canonical-level app distribution.

        Three passes (in order):
          1. Top up Chatbot to CHATBOT_CANONICAL_TARGET by migrating
             lowest-priority non-Chatbot canonicals (introspective / knowledge-
             seeking categories first; lowest xref tie-break).
          2. Carve out AI_Studio share from Chatbot — migrate Chatbot
             canonicals whose categories match `AI_STUDIO_ELIGIBLE_CATEGORY_KEYWORDS`
             (and DON'T match `WRITING_UTILITY_CATEGORY_KEYWORDS`) into
             AI_Studio until AI_STUDIO_CANONICAL_TARGET is met. Lowest-xref
             tie-break.
          3. Top up starved social apps (below SOCIAL_CANONICAL_FLOOR) from
             Chatbot surplus.
        """
        pool = list(self.cross_referenced_personas)
        if not pool:
            return
        n = len(pool)
        target_cb = int(round(n * self.CHATBOT_CANONICAL_TARGET))
        target_ais = int(round(n * self.AI_STUDIO_CANONICAL_TARGET))
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
            non_cb = [cr for cr in pool if cr.assigned_app != "Chatbot"
                      and cr.assigned_app != "AI_Studio"]
            non_cb.sort(key=_priority_for_chatbot_migration)
            deficit = target_cb - cb_count
            for cr in non_cb[:deficit]:
                cr.assigned_app = "Chatbot"

        # ---- Pass 2: Carve out AI_Studio share from Chatbot ----------
        # Migrate Chatbot canonicals whose categories match the AI Studio
        # eligibility keywords AND don't match writing-utility keywords.
        # Companion-chat absorbs identity/aspiration/emotional/intimate
        # signals; utility (email, translation, technical) stays on Chatbot.
        counts = _C(cr.assigned_app for cr in pool)
        ais_count = counts.get("AI_Studio", 0)
        if ais_count < target_ais:
            def _ai_studio_eligible(cr) -> bool:
                cat = (cr.category or "").lower()
                if any(util in cat for util in WRITING_UTILITY_CATEGORY_KEYWORDS):
                    return False  # utility stays on Chatbot
                return any(kw in cat for kw in AI_STUDIO_ELIGIBLE_CATEGORY_KEYWORDS)

            # implicit_negative never routes to AI_Studio (privacy + safety floor).
            chatbot_eligible = [
                cr for cr in pool
                if cr.assigned_app == "Chatbot"
                and cr.source_interaction_type != "implicit_negative"
                and _ai_studio_eligible(cr)
            ]
            # Lowest xref first — preserve high-confidence canonicals on Chatbot
            # to avoid starving the utility surface of strong signal.
            chatbot_eligible.sort(key=lambda cr: cr.confidence_cross_referenced)
            deficit_ais = target_ais - ais_count
            for cr in chatbot_eligible[:deficit_ais]:
                cr.assigned_app = "AI_Studio"

        # ---- Pass 3: Top up starved social apps -----------------------
        counts = _C(cr.assigned_app for cr in pool)
        for app in PLATFORMS:
            if app in ("Chatbot", "AI_Studio"):
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

    # ------------------------------------------------------------------
    # Per-session geolocation (Step 15)
    # ------------------------------------------------------------------

    def assign_event_locations(self) -> None:
        """Step 15: Assign a geolocation to EVERY session via a compact
        gap-anchored LLM call + Python interpolation (100% coverage).

        Algorithm:
          1. Sort sessions by time; compute gaps between consecutive sessions.
          2. Identify TRANSITION CANDIDATES — gaps ≥ GEO_GAP_THRESHOLD_HOURS (4h).
             These are moments a user COULD have traveled. Typical 8-day
             window: 7-12 candidates (mostly overnight sleep).
          3. Build a compact manifest: one entry per gap with before/after
             hashtags and gap duration.
          4. Single LLM call: "Given this user's profile + mobility class +
             these gaps, return the location SEGMENTS (one per stay-at-
             single-city stretch)." LLM decides where, when, and whether
             shifts happened.
          5. Python interpolation: for each session, bind city = segment
             whose start_ts ≤ session.start_ts is latest → 100% coverage.
          6. Derive geo_trip_arcs from non-home segments.

        This replaces the previous per-session LLM assignment, which left
        most sessions untagged (LLM conservatively only emitted a location
        when hashtags clearly named a place → ~2-4% coverage for homebody
        users). The gap-anchor approach gives the LLM the hard question
        (did travel happen, when, where?) and lets deterministic code fill
        the routine per-session lookups.
        """
        if not self._sessions:
            return
        if not self.user_profile:
            return
        client = self.llm_client_mini or self.llm_client
        if client is None:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                      f"Skipping location assignment (no llm client).{utils.Colors.ENDC}")
            return

        mobility_class = self.user_profile.mobility_class or "domestic"

        # ---- 1. Build session time index ----
        session_bounds: list[tuple[int, int, int, list[str], str]] = []
        for idx, session in enumerate(self._sessions):
            if not session:
                continue
            start_ts = session[0].interaction_time
            end_ts = session[-1].interaction_time
            from collections import Counter as _Counter
            tag_counts: _Counter = _Counter()
            for row in session:
                for t in self._extract_hashtags(row.object_text):
                    tag_counts[t.lstrip("#").lower()] += 1
            dominant = [t for t, _ in tag_counts.most_common(5)]
            dominant_app = self._row_app.get(session[0].object_id, "") or ""
            session_bounds.append((idx, start_ts, end_ts, dominant, dominant_app))

        if not session_bounds:
            return
        session_bounds.sort(key=lambda t: t[1])
        obs_start_ts = session_bounds[0][1]
        obs_end_ts = session_bounds[-1][2]

        # ---- 2. Identify transition candidates (gaps ≥ threshold) ----
        gap_threshold_sec = GEO_GAP_THRESHOLD_HOURS * 3600
        gap_candidates: list[dict] = []
        for i in range(1, len(session_bounds)):
            prev_end = session_bounds[i - 1][2]
            curr_start = session_bounds[i][1]
            gap = curr_start - prev_end
            if gap >= gap_threshold_sec:
                gap_candidates.append({
                    "idx": i,
                    "gap_hours": gap / 3600.0,
                    "before_ts": prev_end,
                    "after_ts": curr_start,
                    "before_formatted": utils.unix_to_formatted(prev_end),
                    "after_formatted": utils.unix_to_formatted(curr_start),
                    "before_hashtags": session_bounds[i - 1][3],
                    "after_hashtags": session_bounds[i][3],
                    "before_app": session_bounds[i - 1][4],
                    "after_app": session_bounds[i][4],
                })

        # Cap the candidate list to keep the prompt compact. If there are
        # many short gaps (dense activity), we prioritize longer gaps which
        # are more likely to be travel.
        MAX_GAP_CANDIDATES = 20
        if len(gap_candidates) > MAX_GAP_CANDIDATES:
            gap_candidates.sort(key=lambda g: -g["gap_hours"])
            gap_candidates = gap_candidates[:MAX_GAP_CANDIDATES]
            gap_candidates.sort(key=lambda g: g["before_ts"])

        # ---- 3. Build LLM prompt ----
        user_profile_dict = {
            "name": self.user_profile.name,
            "gender": self.user_profile.gender,
            "race_ethnicity": self.user_profile.race_ethnicity,
            "career": self.user_profile.career,
            "education": self.user_profile.education,
            "bio": self.user_profile.bio,
        }
        obs_window_days = self._obs_window_days()
        prompt = prompts.assign_location_segments_prompt(
            user_profile=user_profile_dict,
            obs_window_days=obs_window_days,
            obs_start_ts=obs_start_ts,
            obs_end_ts=obs_end_ts,
            gap_candidates=gap_candidates,
            mobility_class=mobility_class,
        )

        response = self._query_mini_with_retry(prompt)
        if not response:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                      f"Location segments: LLM returned nothing.{utils.Colors.ENDC}")
            return
        parsed = utils.extract_json_from_response(response)
        if not isinstance(parsed, list) or not parsed:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                      f"Location segments: LLM output was not a non-empty list.{utils.Colors.ENDC}")
            return

        # ---- 4. Parse + validate segments ----
        segments: list[dict] = []
        for s in parsed:
            if not isinstance(s, dict):
                continue
            try:
                start_ts = int(s.get("start_ts") or 0)
            except (ValueError, TypeError):
                continue
            if not s.get("city"):
                continue
            segments.append({
                "start_ts": start_ts,
                "city": s.get("city", ""),
                "region": s.get("region", ""),
                "country": s.get("country", ""),
                "lat": s.get("lat"),
                "lon": s.get("lon"),
                "precision": s.get("precision", "city"),
            })

        if not segments:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                      f"Location segments: no valid segments parsed.{utils.Colors.ENDC}")
            return

        segments.sort(key=lambda s: s["start_ts"])
        # Force first segment to obs_start_ts so interpolation covers from the
        # very first session (LLM may drift a bit).
        segments[0]["start_ts"] = min(segments[0]["start_ts"], obs_start_ts)

        # ---- 5. Interpolate: each session gets the segment active at its start ----
        n_assigned = 0
        for idx, start_ts, end_ts, _, _ in session_bounds:
            seg = segments[0]
            for s in segments:
                if s["start_ts"] <= start_ts:
                    seg = s
                else:
                    break
            self._session_location[idx] = {
                "city": seg["city"],
                "region": seg["region"],
                "country": seg["country"],
                "lat": seg["lat"],
                "lon": seg["lon"],
                "precision": seg["precision"],
            }
            n_assigned += 1

        # ---- 6. Derive trip arcs ----
        home_share = MOBILITY_CLASS_HOME_SHARE.get(mobility_class, HOME_LOCATION_MIN_SHARE)
        self.user_profile.geo_trip_arcs = self._compute_geo_trip_arcs(
            home_share_floor=home_share,
            mobility_class=mobility_class,
        )

        # Audit log
        cities = {loc.get("city", "") for loc in self._session_location.values() if loc.get("city")}
        coverage = n_assigned / max(1, len(session_bounds))
        if self.verbose:
            n_arcs = len(self.user_profile.geo_trip_arcs)
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] "
                  f"Locations assigned: {n_assigned}/{len(session_bounds)} sessions "
                  f"(coverage={coverage:.0%}, class={mobility_class}, "
                  f"{len(segments)} segments, {len(cities)} cities, "
                  f"{n_arcs} trip arcs, {len(gap_candidates)} gaps considered)."
                  f"{utils.Colors.ENDC}")

    def _compute_geo_trip_arcs(
        self,
        home_share_floor: float,
        mobility_class: str,
    ) -> list[dict]:
        """Extract contiguous away-from-home runs from the assigned session
        locations and return them as structured trip arcs. Empty list for
        homebody users.
        """
        if mobility_class == "homebody" or not self._session_location:
            return []

        # Determine home city = most common assignment
        from collections import Counter as _Counter
        city_counts = _Counter(
            loc.get("city", "") for loc in self._session_location.values()
            if loc.get("city")
        )
        if not city_counts:
            return []
        home_city, _ = city_counts.most_common(1)[0]

        sorted_sessions = sorted(self._session_location.items(), key=lambda kv: kv[0])
        arcs: list[dict] = []
        current_run: list[tuple[int, dict]] = []
        for idx, loc in sorted_sessions:
            if loc.get("city") != home_city:
                current_run.append((idx, loc))
            else:
                if current_run:
                    arcs.append(self._run_to_arc(current_run))
                    current_run = []
        if current_run:
            arcs.append(self._run_to_arc(current_run))
        return arcs

    def _run_to_arc(self, run: list[tuple[int, dict]]) -> dict:
        """Convert a contiguous away-session run into a trip arc record."""
        start_idx = run[0][0]
        end_idx = run[-1][0]
        first_loc = run[0][1]
        start_ts = (
            self._sessions[start_idx][0].interaction_time
            if 0 <= start_idx < len(self._sessions) and self._sessions[start_idx]
            else 0
        )
        end_ts = (
            self._sessions[end_idx][-1].interaction_time
            if 0 <= end_idx < len(self._sessions) and self._sessions[end_idx]
            else 0
        )
        # Classify as international if country differs from home country
        home_country = ""
        from collections import Counter as _Counter
        country_counts = _Counter(
            loc.get("country", "") for loc in self._session_location.values()
            if loc.get("country")
        )
        if country_counts:
            home_country, _ = country_counts.most_common(1)[0]
        arc_country = first_loc.get("country", "")
        kind = "international" if (arc_country and arc_country != home_country) else "domestic"
        return {
            "city": first_loc.get("city", ""),
            "region": first_loc.get("region", ""),
            "country": arc_country,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "kind": kind,
        }

    # ------------------------------------------------------------------
    # Synthetic calendar modification stream (Step 16)
    # ------------------------------------------------------------------

    def generate_calendar_modifications(self) -> None:
        """Step 16: Generate a small, scattered timeline of calendar CRUD
        events. Persisted later by save_to_backend to `calendar.json`.

        The calendar is a MODIFICATION STREAM (add/update/remove) rather
        than a static list, so the state at any T_test is derived by
        folding modifications with ts <= T. Eval tasks (E5 horizon
        lifecycle) consume this naturally with the same time-mask used
        for event history.
        """
        if not self.user_profile or not self.interactions:
            return
        client = self.llm_client_mini or self.llm_client
        if client is None:
            return

        # Derive home + travel windows from session locations
        home_location: dict = {}
        travel_windows: list[dict] = []
        if self._session_location:
            from collections import Counter as _Counter
            city_counts = _Counter(
                loc.get("city", "") for loc in self._session_location.values()
                if loc.get("city")
            )
            if city_counts:
                home_city, _ = city_counts.most_common(1)[0]
                # Find the full home location record
                for loc in self._session_location.values():
                    if loc.get("city") == home_city:
                        home_location = loc
                        break
                # Travel windows: contiguous session runs NOT at home
                current_run: list[int] = []
                sorted_sessions = sorted(self._session_location.items(), key=lambda kv: kv[0])
                for idx, loc in sorted_sessions:
                    if loc.get("city") != home_city:
                        current_run.append(idx)
                    else:
                        if current_run:
                            start_idx = current_run[0]
                            end_idx = current_run[-1]
                            start_ts = self._sessions[start_idx][0].interaction_time if start_idx < len(self._sessions) else 0
                            end_ts = self._sessions[end_idx][-1].interaction_time if end_idx < len(self._sessions) else 0
                            away_loc = self._session_location.get(start_idx, {})
                            travel_windows.append({
                                "city": away_loc.get("city", ""),
                                "start_ts": start_ts,
                                "end_ts": end_ts,
                            })
                            current_run = []
                if current_run:
                    start_idx = current_run[0]
                    end_idx = current_run[-1]
                    start_ts = self._sessions[start_idx][0].interaction_time if start_idx < len(self._sessions) else 0
                    end_ts = self._sessions[end_idx][-1].interaction_time if end_idx < len(self._sessions) else 0
                    away_loc = self._session_location.get(start_idx, {})
                    travel_windows.append({
                        "city": away_loc.get("city", ""),
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                    })

        # User profile + app personas for grounding
        user_profile_dict = {
            "name": self.user_profile.name,
            "gender": self.user_profile.gender,
            "career": self.user_profile.career,
            "education": self.user_profile.education,
            "bio": self.user_profile.bio,
        }
        app_personas_dict = {}
        for app_name, persona in (self.user_profile.app_personas or {}).items():
            if isinstance(persona, AppPersona):
                app_personas_dict[app_name] = {
                    "use_purposes": persona.use_purposes,
                    "topical_focus": persona.topical_focus,
                    "posting_frequency": persona.posting_frequency,
                }
            elif isinstance(persona, dict):
                app_personas_dict[app_name] = {
                    "use_purposes": persona.get("use_purposes", []),
                    "topical_focus": persona.get("topical_focus", []),
                    "posting_frequency": persona.get("posting_frequency", ""),
                }

        preference_list = [
            {"persona_item": cr.persona_item, "category": cr.category}
            for cr in (self.cross_referenced_personas or [])
        ]

        ts_all = [r.interaction_time for r in self.interactions if r.interaction_time]
        if not ts_all:
            return
        obs_start_ts = min(ts_all)
        obs_end_ts = max(ts_all)
        obs_window_days = (obs_end_ts - obs_start_ts) / 86400.0

        # v0: raise density floor to 20–28 mods across all mobility classes
        # so e6 discovery has enough calendar grounding for airport-mismatch,
        # canceled-event-reference, and forgotten-promise archetypes.
        n_mods = min(
            E6_MAX_CALENDAR_MODIFICATIONS,
            max(E6_MIN_CALENDAR_MODIFICATIONS, int(obs_window_days * 3.0) + 4),
        )

        mobility_class = self.user_profile.mobility_class or "domestic"
        prompt = prompts.generate_calendar_modifications_prompt(
            user_profile=user_profile_dict,
            app_personas=app_personas_dict,
            obs_window_days=obs_window_days,
            obs_start_ts=obs_start_ts,
            obs_end_ts=obs_end_ts,
            home_location=home_location,
            travel_windows=travel_windows,
            preference_list=preference_list,
            n_modifications=n_mods,
            mobility_class=mobility_class,
            require_recent_cancellation=True,
            recent_cancellation_window_hours=E6_RECENT_CANCELLATION_WINDOW_HOURS,
        )

        response = self._query_mini_with_retry(prompt)
        if not response:
            return
        parsed = utils.extract_json_from_response(response)
        if not isinstance(parsed, list):
            return

        # Filter + normalize modifications
        known_entry_ids: set[str] = set()
        sanitized: list[dict] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            action = entry.get("action")
            if action not in ("added", "updated", "removed"):
                continue
            ts = entry.get("ts")
            if not isinstance(ts, int):
                continue
            if action == "added":
                payload = entry.get("entry")
                if not isinstance(payload, dict):
                    continue
                entry_id = payload.get("entry_id") or f"cal_{len(sanitized) + 1:03d}"
                payload["entry_id"] = entry_id
                known_entry_ids.add(entry_id)
                sanitized.append({
                    "mod_id": entry.get("mod_id", f"mod_{len(sanitized) + 1:03d}"),
                    "ts": ts,
                    "formatted_timestamp": entry.get("formatted_timestamp") or utils.unix_to_formatted(ts),
                    "action": "added",
                    "entry": payload,
                })
            elif action == "updated":
                ref_id = entry.get("entry_id")
                if ref_id not in known_entry_ids:
                    continue
                diff = entry.get("diff")
                if not isinstance(diff, dict):
                    continue
                sanitized.append({
                    "mod_id": entry.get("mod_id", f"mod_{len(sanitized) + 1:03d}"),
                    "ts": ts,
                    "formatted_timestamp": entry.get("formatted_timestamp") or utils.unix_to_formatted(ts),
                    "action": "updated",
                    "entry_id": ref_id,
                    "diff": diff,
                })
            elif action == "removed":
                ref_id = entry.get("entry_id")
                if ref_id not in known_entry_ids:
                    continue
                known_entry_ids.discard(ref_id)
                sanitized.append({
                    "mod_id": entry.get("mod_id", f"mod_{len(sanitized) + 1:03d}"),
                    "ts": ts,
                    "formatted_timestamp": entry.get("formatted_timestamp") or utils.unix_to_formatted(ts),
                    "action": "removed",
                    "entry_id": ref_id,
                    "removal_reason": entry.get("removal_reason", ""),
                })

        sanitized.sort(key=lambda m: m["ts"])

        # ----- Step 16 Option A: deterministic repair of required diversity -----
        # The LLM juggles many constraints (density, split, transit, multi-attendee,
        # recent cancellation, preference-linking). With ~7 simultaneous clauses,
        # one typically drops. Rather than re-prompting, we validate + repair here
        # so e6 substrate floors are met deterministically.
        sanitized = self._repair_calendar_diversity(
            sanitized,
            obs_start_ts=obs_start_ts,
            obs_end_ts=obs_end_ts,
            home_location=home_location,
            mobility_class=mobility_class,
        )

        self._calendar_modifications = sanitized
        if self.verbose:
            n_add = sum(1 for m in sanitized if m["action"] == "added")
            n_upd = sum(1 for m in sanitized if m["action"] == "updated")
            n_rem = sum(1 for m in sanitized if m["action"] == "removed")
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] "
                  f"Calendar modifications: {len(sanitized)} total "
                  f"({n_add} added / {n_upd} updated / {n_rem} removed).{utils.Colors.ENDC}")

    def _repair_calendar_diversity(
        self,
        mods: list[dict],
        obs_start_ts: int,
        obs_end_ts: int,
        home_location: dict,
        mobility_class: str,
    ) -> list[dict]:
        """Enforce Step 16 required-diversity clauses deterministically.

        Checks three hard requirements and injects minimal repairs if any
        is missing. Injected entries use templated text drawn from the
        user's friends graph where possible (falling back to generic names
        when the graph is empty).

        Requirements:
          (a) ≥ 1 added entry with ≥ 1 non-self attendee
          (b) ≥ 1 `removed` modification in the last 6h of the window
          (c) ≥ 1 `travel`-type added entry (flight for travel classes, else
              local transit)
        """
        # Home location fallback
        home = home_location if isinstance(home_location, dict) else {}

        # Determine friends pool for attendee names
        friends_pool: list[str] = []
        if self.user_profile and getattr(self.user_profile, "app_personas", None):
            # Pull named friends from any app_persona's friend_zones
            for ap in (self.user_profile.app_personas or {}).values():
                if isinstance(ap, dict):
                    for friend in (ap.get("friend_zones") or []):
                        if isinstance(friend, str) and friend not in friends_pool:
                            friends_pool.append(friend)
                elif isinstance(ap, AppPersona):
                    for friend in (ap.friend_zones or []):
                        if friend not in friends_pool:
                            friends_pool.append(friend)
        if not friends_pool:
            friends_pool = ["Alex", "Sam", "Jordan"]  # generic fallback

        # ---- Check (a): multi-attendee meeting ----
        has_named_attendee = False
        for m in mods:
            if m.get("action") != "added":
                continue
            attendees = (m.get("entry") or {}).get("attendees") or []
            others = [a for a in attendees if str(a).lower() != "self"]
            if len(others) >= 1:
                has_named_attendee = True
                break

        # ---- Check (b): recent cancellation in last 6h ----
        recent_cancel_window_start = obs_end_ts - E6_RECENT_CANCELLATION_WINDOW_HOURS * 3600
        has_recent_cancel = any(
            m.get("action") == "removed" and m.get("ts", 0) >= recent_cancel_window_start
            for m in mods
        )

        # ---- Check (c): transit entry ----
        has_transit = any(
            m.get("action") == "added"
            and (m.get("entry") or {}).get("type") == "travel"
            for m in mods
        )

        next_mod_id = len(mods) + 1
        next_entry_id = sum(1 for m in mods if m.get("action") == "added") + 1

        injected: list[str] = []

        # Repair (a): inject a coffee/meeting with named friend earlier in window
        if not has_named_attendee:
            friend = friends_pool[0]
            # Pick a ts ~1/3 into the window, aligned to a reasonable hour
            meet_start = obs_start_ts + int((obs_end_ts - obs_start_ts) * 0.35)
            # Mod created slightly before the meeting
            mod_ts = max(obs_start_ts, meet_start - 6 * 3600)
            entry_id = f"cal_rep_{next_entry_id:03d}"
            next_entry_id += 1
            mods.append({
                "mod_id": f"mod_rep_{next_mod_id:03d}",
                "ts": mod_ts,
                "formatted_timestamp": utils.unix_to_formatted(mod_ts),
                "action": "added",
                "entry": {
                    "entry_id": entry_id,
                    "title": f"Coffee with {friend}",
                    "start_ts": meet_start,
                    "end_ts": meet_start + 3600,
                    "location": home,
                    "type": "social",
                    "attendees": ["self", friend],
                    "linked_preferences": [],
                    "is_preference_driven": False,
                    "relation_to_social": "unrelated",
                },
            })
            next_mod_id += 1
            injected.append("multi-attendee")

        # Repair (b): inject a cancellation in last 6h of window
        if not has_recent_cancel:
            # Find an added entry with ts < recent_cancel_window_start to remove,
            # or inject a throwaway add+remove pair
            candidate_removal_id = None
            for m in reversed(mods):
                if (m.get("action") == "added"
                        and m.get("ts", 0) < recent_cancel_window_start):
                    candidate_removal_id = (m.get("entry") or {}).get("entry_id")
                    if candidate_removal_id:
                        break
            if candidate_removal_id is None:
                # Inject an early added entry first, then remove it
                add_ts = obs_start_ts + 3600
                entry_id = f"cal_rep_{next_entry_id:03d}"
                next_entry_id += 1
                mods.append({
                    "mod_id": f"mod_rep_{next_mod_id:03d}",
                    "ts": add_ts,
                    "formatted_timestamp": utils.unix_to_formatted(add_ts),
                    "action": "added",
                    "entry": {
                        "entry_id": entry_id,
                        "title": "Tentative dinner plans",
                        "start_ts": obs_end_ts - 3600,
                        "end_ts": obs_end_ts,
                        "location": home,
                        "type": "social",
                        "attendees": ["self"],
                        "linked_preferences": [],
                        "is_preference_driven": False,
                        "relation_to_social": "unrelated",
                    },
                })
                next_mod_id += 1
                candidate_removal_id = entry_id

            remove_ts = obs_end_ts - 1800  # 30 minutes before end
            mods.append({
                "mod_id": f"mod_rep_{next_mod_id:03d}",
                "ts": remove_ts,
                "formatted_timestamp": utils.unix_to_formatted(remove_ts),
                "action": "removed",
                "entry_id": candidate_removal_id,
                "removal_reason": "Rescheduled at last minute",
            })
            next_mod_id += 1
            injected.append("recent-cancellation")

        # Repair (c): inject a transit entry
        if not has_transit:
            transit_title = (
                "Flight home" if mobility_class in ("international", "nomadic")
                else "Train to Center City"
            )
            transit_start = obs_start_ts + int((obs_end_ts - obs_start_ts) * 0.50)
            mod_ts = max(obs_start_ts, transit_start - 12 * 3600)
            entry_id = f"cal_rep_{next_entry_id:03d}"
            next_entry_id += 1
            mods.append({
                "mod_id": f"mod_rep_{next_mod_id:03d}",
                "ts": mod_ts,
                "formatted_timestamp": utils.unix_to_formatted(mod_ts),
                "action": "added",
                "entry": {
                    "entry_id": entry_id,
                    "title": transit_title,
                    "start_ts": transit_start,
                    "end_ts": transit_start + 2 * 3600,
                    "location": home,
                    "type": "travel",
                    "attendees": ["self"],
                    "linked_preferences": [],
                    "is_preference_driven": False,
                    "relation_to_social": "unrelated",
                },
            })
            next_mod_id += 1
            injected.append("transit")

        if injected and self.verbose:
            print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                  f"Calendar repair: injected {len(injected)} entries "
                  f"for missing diversity clauses [{', '.join(injected)}]."
                  f"{utils.Colors.ENDC}")

        mods.sort(key=lambda m: m["ts"])
        return mods

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

        # Shared user_voice — drives cross-app voice consistency in user_message generation
        user_voice_dict = (self.user_profile.user_voice or {}) if self.user_profile else {}

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
                    user_voice=user_voice_dict,
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
                    desc=f"[User {self.user_id}] Step 17: Interaction formats",
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
            chatbot_persona_dict = asdict(chatbot_persona)
        else:
            chatbot_persona_dict = chatbot_persona

        user_profile_dict = {
            "name": self.user_profile.name,
            "gender": self.user_profile.gender,
            "race_ethnicity": self.user_profile.race_ethnicity,
            "career": self.user_profile.career,
            "education": self.user_profile.education,
            "bio": self.user_profile.bio,
            # Hidden personas + their dominant frames let chatbot turn
            # generation anchor the user's "why" in a named motivational
            # frame (Lazarus-Folkman coping vs Tajfel social identity vs
            # Goffman back-stage) instead of generic affect. Frame is
            # resolved by `prompts.cluster_dominant_frame` (audit-aware
            # if Step 22 ran, structural-default otherwise).
            "hidden_personas": [
                asdict(hp) for hp in (self.user_profile.hidden_personas or [])
            ],
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

        # Conversation synthesis at temperature=0.7 — empirically gives more
        # natural, varied user voice than the default ~1.0 (which produces
        # over-elaborate parallel-structure prose flagged in the user-voice
        # contract block in `prompts.py`). Temperature is plumbed through to
        # the LLM client via the optional kwarg added in `query_llm.py`.
        def _conv_query_fn(prompt: str):
            return self._query_llm_with_retry(prompt, temperature=0.7)

        # Mini-tier query fn for the voice-quality auto-judge in
        # chatbot_conversation.py — a per-conversation pass/fail check
        # that retries with concrete feedback when the user-side turns
        # (or pasted drafts) drift off the user's layered voice. Mini
        # tier keeps the audit cheap; conversation generation itself
        # stays on flagship.
        def _voice_judge_query_fn(prompt: str):
            return self._query_mini_with_retry(prompt)

        chatbot_conversation.generate_chatbot_conversations(
            chatbot_records=chatbot_records,
            user_profile=user_profile_dict,
            chatbot_persona=chatbot_persona_dict,
            llm_query_fn=_conv_query_fn,
            user_seed=user_seed,
            max_workers=self.max_workers,
            user_voice=(self.user_profile.user_voice or {}),
            mini_query_fn=_voice_judge_query_fn,
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

    def generate_ai_studio_conversations(self) -> None:
        """Step 18b — generate cross-session AI Studio conversations.

        Sequential: each event's prompt embeds the FULL prior history (so the
        chosen AI character — Rowan, Wren, etc. — replies with continuity
        across the whole arc). intimacy_arc + intimacy_stage are tracked
        event-by-event in `self._ai_studio_memory_state`. SPT smoothness
        emerges from the conversation-type filter in
        `eligible_conversation_types`.

        Builds per-event records for AI_Studio-routed rows (similar to
        `generate_chatbot_conversations` but for the 5th app), invokes the
        sequential generator from `ai_studio_conversation.py`, and stashes
        the results on `self._ai_studio_records` keyed by source_object_id.
        """
        from data_preparation import ai_studio_conversation, ai_studio_memory

        # Initialize memory state container (mutated by generation).
        self._ai_studio_records: list[dict] = []
        self._ai_studio_memory_state = ai_studio_memory.default_memory_state()

        if not self.cross_referenced_personas or not self.user_profile:
            return
        if not (self.user_profile.ai_studio_persona or {}).get("persona_archetype"):
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                      f"AI Studio: no ai_studio_persona on profile — skipping Step 18b.{utils.Colors.ENDC}")
            return
        if self.llm_client is None:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                      f"Skipping AI Studio conversation generation (no llm_client).{utils.Colors.ENDC}")
            return

        # Build canonical lookup (mirrors generate_chatbot_conversations).
        canonical_lookup: dict[str, CrossReferencedPersona] = {}
        for cr in self.cross_referenced_personas:
            cr_key = _normalize_persona_text(cr.persona_item)
            canonical_lookup[cr_key] = cr
            for ap in self._canonical_groups.get(cr_key, []):
                ap_key = _normalize_persona_text(ap.persona_item)
                if ap_key not in canonical_lookup:
                    canonical_lookup[ap_key] = cr

        # Group atomics by source_object_id — only include AI_Studio-routed rows.
        from collections import defaultdict as _ddict
        atomics_by_oid: dict[str, list] = _ddict(list)
        for ap in self.atomic_personas:
            atomics_by_oid[ap.source_object_id].append(ap)

        ai_studio_records: list[dict] = []
        for oid, atoms in atomics_by_oid.items():
            if self._row_app.get(oid) != "AI_Studio":
                continue
            if not atoms:
                continue

            # Collect surviving preferences for this event (deduped).
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
                    "source_object_id": oid,
                })
            if not prefs:
                continue

            rep = atoms[0]
            ai_studio_records.append({
                "source_object_id": oid,
                "source_timestamp": rep.source_timestamp,
                "source_hashtags": list(rep.source_hashtags or []),
                "source_interaction_type": rep.source_interaction_type,
                "preferences": prefs,
            })

        if not ai_studio_records:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                      f"AI Studio: zero canonicals routed to AI_Studio (Step 13/14 might "
                      f"have given everything else higher priority); skipping.{utils.Colors.ENDC}")
            return

        try:
            user_seed = int(str(self.user_id)) * 7919 + 131
        except (ValueError, TypeError):
            user_seed = abs(hash(str(self.user_id))) % (2**31)

        # Generation runs at temperature=0.7 (empirically warmer / more natural
        # than the default for narrative dialogue — same setting we use for
        # chatbot conversations).
        def _conv_query_fn(prompt: str):
            return self._query_llm_with_retry(prompt, temperature=0.7)

        user_profile_dict = {
            "name": self.user_profile.name,
            "gender": self.user_profile.gender,
            "race_ethnicity": self.user_profile.race_ethnicity,
            "career": self.user_profile.career,
            "education": self.user_profile.education,
            "bio": self.user_profile.bio,
        }
        # Pass hidden personas as brief dicts — the prompt instructs them to
        # be treated as oblique anchors only (NEVER named verbatim).
        hp_brief = [
            {
                "persona_type": getattr(hp, "type", "") or "",
                "label": getattr(hp, "label", "") or "",
                "description": getattr(hp, "description", "") or "",
                "evidence_oids": list(getattr(hp, "evidence_oids", []) or []),
            }
            for hp in (self.user_profile.hidden_personas or [])
        ]

        # Snapshot LLM usage BEFORE Step 18B so we can report the cache
        # hit-rate over just this step. Sequential AI Studio generation
        # reuses ~80% of each prior prompt verbatim — caching is the single
        # biggest cost+latency lever once enabled.
        usage_before = self.llm_client.get_usage_totals() if self.llm_client else {}

        out, mem = ai_studio_conversation.generate_ai_studio_conversations(
            ai_studio_records=ai_studio_records,
            user_profile=user_profile_dict,
            user_voice=self.user_profile.user_voice or {},
            ai_studio_persona=self.user_profile.ai_studio_persona or {},
            hidden_personas=hp_brief,
            llm_query_fn=_conv_query_fn,
            user_seed=user_seed,
            memory_state=self._ai_studio_memory_state,
            verbose=self.verbose,
        )
        self._ai_studio_records = out
        self._ai_studio_memory_state = mem

        if self.verbose:
            archetype = (self.user_profile.ai_studio_persona or {}).get("persona_archetype", "?")
            character = (self.user_profile.ai_studio_persona or {}).get("character_name", "?")
            print(
                f"{utils.Colors.OKGREEN}[User {self.user_id}] "
                f"Generated {len(out)}/{len(ai_studio_records)} AI Studio conversations "
                f"(archetype={archetype!r} character={character!r}; "
                f"final intimacy_arc={mem.running_relational_state.intimacy_arc:.2f} "
                f"stage={mem.running_relational_state.intimacy_stage})"
                f"{utils.Colors.ENDC}"
            )

            # Cache hit-rate report (after-before delta from the LLM client).
            usage_after = self.llm_client.get_usage_totals() if self.llm_client else {}
            calls_d = (usage_after.get("calls", 0) - usage_before.get("calls", 0))
            input_d = (usage_after.get("input_tokens", 0) - usage_before.get("input_tokens", 0))
            cached_d = (usage_after.get("cached_input_tokens", 0) - usage_before.get("cached_input_tokens", 0))
            if input_d > 0:
                hit_pct = 100.0 * cached_d / input_d
                print(
                    f"{utils.Colors.OKBLUE}[User {self.user_id}] AI Studio Step 18B "
                    f"prompt-cache: {calls_d} calls, "
                    f"{cached_d:,}/{input_d:,} input tokens cached "
                    f"({hit_pct:.1f}% hit rate).{utils.Colors.ENDC}"
                )

    def audit_ai_studio_conversations(self) -> None:
        """Step Z — quality + safety audit over AI Studio events.

        Samples 20% of events; grades each on 7 quality axes + the
        `no_harmful_content` floor. Safety failures are DROPPED (we never
        ship harmful content). Quality-only failures are kept but tagged
        `audit_status: graceful_degrade` so downstream readers can skip.
        """
        from data_preparation import ai_studio_audit

        records = getattr(self, "_ai_studio_records", None)
        if not records:
            return
        if self.llm_client is None:
            return

        # Audit uses mini-tier (cheap, parallel-friendly) for all axes.
        def _audit_query_fn(prompt: str):
            return self._query_mini_with_retry(prompt)

        try:
            user_seed = int(str(self.user_id)) * 7919 + 131
        except (ValueError, TypeError):
            user_seed = abs(hash(str(self.user_id))) % (2**31)

        hp_brief = [
            {
                "persona_type": getattr(hp, "type", "") or "",
                "label": getattr(hp, "label", "") or "",
            }
            for hp in (self.user_profile.hidden_personas or [])
        ]

        filtered, summary = ai_studio_audit.audit_ai_studio_conversations(
            ai_studio_records=records,
            user_voice=self.user_profile.user_voice or {},
            ai_studio_persona=self.user_profile.ai_studio_persona or {},
            hidden_personas_brief=hp_brief,
            rogers_cliche_baseline=ROGERS_CLICHE_BLOCKLIST,
            memory_state=getattr(self, "_ai_studio_memory_state", None),
            audit_query_fn=_audit_query_fn,
            user_seed=user_seed,
            verbose=self.verbose,
        )
        self._ai_studio_records = filtered
        self._ai_studio_audit_summary = summary

        if self.verbose:
            print(
                f"{utils.Colors.OKGREEN}[User {self.user_id}] "
                f"AI Studio audit summary: sampled={summary.get('sampled', 0)}, "
                f"passed={summary.get('passed', 0)}, "
                f"graceful_degrade={summary.get('graceful_degrade', 0)}, "
                f"dropped_safety={summary.get('dropped_safety', 0)}, "
                f"final_count={len(filtered)}{utils.Colors.ENDC}"
            )

    # ------------------------------------------------------------------
    # Step 19 — Synthetic per-event content generation
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
        """Step 19: Generate synthetic textual content for each non-Chatbot,
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
                app_persona_dicts[app_name] = asdict(persona)
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

        # Hashtag → (cluster_label, evidence_rows) lookup for frame
        # resolution. Events whose hashtags overlap a hidden-persona
        # cluster carry that cluster's dominant frame into the
        # title/caption prompt, so synthetic content reflects the
        # cluster's motivational signature instead of generic topic.
        # Skip synthetic clusters (sensitive_life_event) — they have
        # their own planted-row pipeline (Step 21b).
        hashtag_to_cluster_frame: dict[str, tuple[str, str, int]] = {}
        for hp in (self.user_profile.hidden_personas or []):
            if getattr(hp, "is_synthetic", False):
                continue
            frame = prompts.cluster_dominant_frame(hp)
            if not frame or frame == "none":
                continue
            ev_rows = int(getattr(hp, "evidence_rows", 0) or 0)
            for tag in (getattr(hp, "evidence_hashtags", None) or []):
                tag_norm = (tag or "").lower().lstrip("#").strip()
                if not tag_norm:
                    continue
                # Keep the cluster with most evidence_rows on conflict
                # (largest cluster wins when one hashtag belongs to two).
                prev = hashtag_to_cluster_frame.get(tag_norm)
                if prev is None or ev_rows > prev[2]:
                    hashtag_to_cluster_frame[tag_norm] = (
                        getattr(hp, "label", ""), frame, ev_rows,
                    )

        def _frame_for_event(hashtags: list[str]) -> tuple[str, str]:
            """Return (frame, frame_description) for the event's
            best-matching cluster, or ('', '') when no overlap."""
            best_label, best_frame, best_score = "", "", 0
            for tag in hashtags or []:
                tag_norm = (tag or "").lower().lstrip("#").strip()
                hit = hashtag_to_cluster_frame.get(tag_norm)
                if hit and hit[2] > best_score:
                    best_label, best_frame, best_score = hit
            if not best_frame:
                return "", ""
            return best_frame, prompts.FRAME_DESCRIPTIONS.get(best_frame, "")

        events_to_generate: list[dict] = []
        for oid, meta in self._action_by_oid.items():
            app = self._row_app.get(oid) or PLATFORMS[0]
            if app in ("Chatbot", "AI_Studio"):
                # Chatbot and AI Studio are conversation-only surfaces — there
                # is no media post the user is engaging with, so no synthetic
                # content body is generated for them.
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

            event_frame, event_frame_desc = _frame_for_event(event_hashtags)

            content_type = self._resolve_content_type(app, meta["action"], oid)
            events_to_generate.append({
                "oid": oid,
                "app": app,
                "action": meta["action"],
                "action_label": meta["action_label"],
                "content_type": content_type,
                "hashtags": event_hashtags,
                "preferences": prefs,
                "motivation_frame": event_frame,
                "motivation_frame_description": event_frame_desc,
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
                    motivation_frame=ev.get("motivation_frame") or None,
                    motivation_frame_description=ev.get("motivation_frame_description") or None,
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
                    desc=f"[User {self.user_id}] Step 19: Synthetic content",
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
    # Ad injection (Step 20)
    # ------------------------------------------------------------------

    def inject_ad_events(self) -> None:
        """Step 20: Convert a small fraction of commerce-adjacent events into
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
            return  # Step 19 didn't run — nothing to inject ads into

        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Deterministic RNG per user — shares the same namespace as Step 19
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
                      f"Step 20: no ad-eligible events (no commerce hashtags).{utils.Colors.ENDC}")
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
                    desc=f"[User {self.user_id}] Step 20: Ad content",
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
                            "disclosure_label": "Ads",
                        }
                    else:
                        # Normalize required fields.
                        md = content["ad_metadata"]
                        md.setdefault("sponsor_name", "Sponsored Brand")
                        md.setdefault("ad_category", ev["ad_category"])
                        md.setdefault("cta_label", "Learn more")
                        md.setdefault("cta_destination_kind", "landing_page")
                        md.setdefault("disclosure_label", "Ads")
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

        Order (28 sequential steps — renumbered so every addition slots in
        cleanly rather than carrying .5 / b / c suffixes):

           1. infer atomic personas
           2. promote implicit negatives
           3. cross-reference & filter
           4. classify horizons + stop conditions (short-term refinement)
           5. temporal contradiction graph
           6. build update histories
           7. resolve cross-polarity contradictions (temporal-precedent gate
              — fixes the 115-boxing stance-flip bug by requiring prior
              same-polarity evidence before admitting an opposing stance)
           8. generate user profile (demographics + big_five + bio + mobility_class)
           9. infer hidden personas (cross-row hashtag clustering)
          10. infer MBTI
          11. generate per-app sub-personas
          12. build sessions
          13. route preferences to apps (LLM + 8% noise)
          14. assign rows to apps (session majority vote)
          15. assign per-session geolocations (class-adaptive) + trip arcs
          16. generate calendar modification stream (density floor + required cancellation)
          17. generate interaction formats (weighted catalog sampling)
          18. generate chatbot conversations (multi-turn, implicit embedding)
          19. generate synthetic per-event content (text/image/short_video)
          20. inject ad events (~6% of commerce-adjacent events become ads)
          21. link preferences to hidden personas (hashtag-overlap, lifted out
              of save_to_backend so the audit can operate on it)
          22. audit hidden persona motivations (parsimony-biased re-judgment
              against named academic frames; mutates link table)
          23. aggregate motivation audit to summary (cluster_status tiers —
              never mutates the cluster, only annotates)
          24. annotate stereotype marks (now operates on AUDITED links)
          25. enrich substrate (plant cross-signal evidence for e6)
          26. save pipeline outputs to backend/{uid}/ subfolder
          27. run Extension B layer (self-posts, DM threads, friends graph,
              trending hashtags) directly on top of the just-saved files —
              produces a fully-complete backend in one invocation.
          28. infer proactive trigger candidates (catalogues moments where
              the agent could legitimately initiate contact, scored by an
              LLM against JITAI + Horvitz mixed-initiative; output saved to
              profile.json.proactive_trigger_candidates).

        (R8 dropped the old "build test split" step entirely — eval picks
        its own test moments from the full timeline at any T_test cut.)
        """
        print(f"{utils.Colors.BOLD}[User {self.user_id}] Starting persona pipeline...{utils.Colors.ENDC}")
        pipeline_start = time.time()

        steps = [
            ("1.  Infer atomic personas",               self.infer_personas_from_hashtags),
            ("2.  Promote implicit negatives",          self.promote_implicit_negatives),
            ("3.  Cross-reference & filter",            self.summarize_and_cross_reference),
            ("4.  Classify horizons + stops",           self.classify_horizons_and_stop_conditions),
            ("5.  Temporal contradiction graph",        self.build_temporal_contradiction_graph),
            ("6.  Build update histories",              self.build_update_histories),
            ("7.  Resolve cross-polarity contradictions", self.resolve_cross_polarity_contradictions),
            ("8.  Generate user profile",               self.generate_user_profile),
            ("9.  Infer hidden personas",               self.infer_hidden_personas),
            ("10. Infer MBTI",                          self.infer_mbti),
            ("11. Generate app personas",               self.generate_app_personas),
            ("11C. Generate AI Studio persona",          self.generate_ai_studio_persona),
            ("12. Build sessions",                      self._build_sessions),
            ("13. Route preferences to apps",           self.route_personas_to_apps),
            ("14. Assign rows to apps",                 self._assign_rows_to_apps),
            ("15. Assign session locations",            self.assign_event_locations),
            ("16. Generate calendar modifications",     self.generate_calendar_modifications),
            ("17. Generate interaction formats",        self.generate_interaction_formats),
            ("18. Generate chatbot conversations",      self.generate_chatbot_conversations),
            ("18B. Generate AI Studio conversations",    self.generate_ai_studio_conversations),
            ("18C. Audit AI Studio (quality + safety floor)", self.audit_ai_studio_conversations),
            ("19. Generate synthetic content",          self.generate_synthetic_content),
            ("20. Inject ad events",                    self.inject_ad_events),
            ("21. Link preferences to hidden personas", self.link_preferences_to_hidden_personas),
            ("22. Audit hidden persona motivations",    self.audit_hidden_persona_motivations),
            ("23. Aggregate motivation audit summary",  self.aggregate_motivation_audit_to_summary),
            ("24. Annotate stereotype marks",           self.annotate_stereotype_marks),
            ("25. Enrich substrate (e6 grounding)",     self.enrich_substrate),
            ("26. Save to backend",                     self.save_to_backend),
            ("27. Extension B (self-posts + DMs + friends + trending)",
                                                        self.run_extension_b),
            ("28. Infer proactive trigger candidates",  self.infer_proactive_trigger_candidates),
        ]

        for step_name, step_fn in steps:
            step_start = time.time()
            step_fn()
            elapsed = time.time() - step_start
            total_elapsed = time.time() - pipeline_start
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] {step_name}: "
                  f"{elapsed:.1f}s (total: {total_elapsed:.1f}s){utils.Colors.ENDC}")

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
            "n_short_term": sum(
                1 for p in self.cross_referenced_personas if p.time_horizon == "short_term"
            ),
            "n_ad_events": len(self._ad_oids),
            "n_calendar_modifications": len(self._calendar_modifications),
            "n_suppressed_stance_flips": len(self._suppressed_stance_flips),
            "n_step9_dropped_specificity": self._n_step9_dropped_specificity,
            "total_time_seconds": round(time.time() - pipeline_start, 1),
        }
        total_time = time.time() - pipeline_start
        mins, secs = divmod(total_time, 60)
        print(f"{utils.Colors.OKGREEN}[User {self.user_id}] Pipeline complete in {int(mins)}m {secs:.0f}s: {summary}{utils.Colors.ENDC}")
        return summary

    # ------------------------------------------------------------------
    # Substrate enrichment (Step 22) — plant signals that ground
    # e6_active_mistake_prevention discovery archetypes. Runs after all
    # content is generated and before persistence.
    # ------------------------------------------------------------------

    def enrich_substrate(self) -> None:
        """Step 22: Plant cross-signal evidence that e6 discovery archetypes
        rely on. Runs after all content generation, before save_to_backend.

        Does two things:
          (a) Plants ≥ 2 user-stated personal constraints early in chatbot
              history (dietary / equipment / deadline / preference). These
              become grounding for the chatbot-memory form example in e6
              discovery and the `moment-mid-short-term-preference` variant
              in T7.
          (b) Sanity-checks that each privacy-flagged hidden persona
              (type ∈ {covert_concern, compensatory_need,
              intimate_interest} OR privacy_ratio > 0.7) has ≥ 1
              aggravation-relevant event in the 48h pre-obs_end window.
              Emits a warning if missing so operators can decide whether
              to regenerate or accept the partial substrate. Actual
              planting is deferred to a follow-up iteration.

        DM commitment tagging lives in Extension B (data_preparation/
        extension_b/) where DMs are materialized — not here. See the
        extension_b changes in the v0.1 follow-up.

        Each sub-step is wrapped in try/except so a failure in enrichment
        never blocks save_to_backend — the pipeline's ~11 minutes of LLM
        work must not be lost to a bug in this late, low-stakes step.
        """
        try:
            self._audit_persona_safety_aggravation()
        except Exception as e:
            print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                  f"enrich_substrate: persona-safety audit raised "
                  f"{type(e).__name__}: {e}. Skipping; pipeline continues.{utils.Colors.ENDC}")

    # ------------------------------------------------------------------

    def _audit_persona_safety_aggravation(self) -> None:
        """For each privacy-flagged hidden persona, verify that the 48h
        pre-obs_end window contains ≥ 1 event whose hashtags overlap the
        persona's evidence_hashtags. Warn (do not fail) when missing.

        This is the e6 persona-safety grounding check. v0 only audits;
        planting an aggravation event is deferred to a follow-up.

        Handles both HiddenPersona dataclass objects (in-memory during
        pipeline) and dicts (if called on reloaded backend data).
        """
        if not self.user_profile or not self.interactions:
            return
        hidden = self.user_profile.hidden_personas or []
        if not hidden:
            return

        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # Window: last 48h of the observed activity
        ts_all = [r.interaction_time for r in self.interactions if r.interaction_time]
        if not ts_all:
            return
        obs_end = max(ts_all)
        window_start = obs_end - 48 * 3600

        PRIVACY_TYPES = {"covert_concern", "compensatory_need", "intimate_interest", "medical_aesthetic_concern"}
        # Synthetic clusters (sensitive_life_event today) are gated by their
        # OWN active_window, not by recent organic engagement, so skipping
        # them here keeps the audit signal-to-noise high.
        flagged = [
            hp for hp in hidden
            if not _get(hp, "is_synthetic", False)
            and (
                (_get(hp, "type") in PRIVACY_TYPES)
                or (float(_get(hp, "privacy_ratio") or 0) > 0.7)
            )
        ]
        if not flagged:
            return

        # Build hashtag → set of (ts) lookup limited to the window
        recent_events_by_tag: dict[str, list[int]] = {}
        for row in self.interactions:
            ts = row.interaction_time or 0
            if ts < window_start or ts > obs_end:
                continue
            for tag in self._extract_hashtags(row.object_text):
                recent_events_by_tag.setdefault(tag.lstrip("#").lower(), []).append(ts)

        missing: list[str] = []
        for hp in flagged:
            evidence = [
                t.lstrip("#").lower()
                for t in (_get(hp, "evidence_hashtags") or [])
            ]
            hit = any(tag in recent_events_by_tag for tag in evidence)
            if not hit:
                missing.append(_get(hp, "label") or "(unnamed)")

        if missing and self.verbose:
            print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                  f"enrich_substrate: {len(missing)} privacy-flagged hidden "
                  f"personas lack an aggravating event in the last 48h "
                  f"[{', '.join(missing)}]. e6 persona-safety archetypes "
                  f"for these labels may be unbuildable. v0 audit only; "
                  f"planting deferred.{utils.Colors.ENDC}")

    # ------------------------------------------------------------------
    # Hidden-persona linking + motivation audit (Steps 21–23)
    # ------------------------------------------------------------------

    def link_preferences_to_hidden_personas(self) -> None:
        """Step 21: Build the per-(source_object_id, persona_item) link table
        by hashtag-overlap (cluster.evidence_oids → preference's source row).

        Relocated from inside save_to_backend so the link table is an
        inspectable artifact the motivation audit (Step 22) operates on,
        and so save_to_backend can be thinned to a pure consumer.

        Output: ``self._preference_links`` is a dict keyed by
        ``(source_object_id, _normalize_persona_text(persona_item))``
        carrying the provisional cluster label, evidence_rows tie-break
        score, link_provenance, and a placeholder ``motivation_audit``
        block that the audit step (22) populates.
        """
        # Reset every run so re-invocations don't leak across users in
        # batch contexts.
        self._preference_links: dict[tuple[str, str], dict] = {}
        self._motivation_audit_user_warning: dict = {}

        if not self.user_profile or not self.user_profile.hidden_personas:
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] "
                      f"link_preferences_to_hidden_personas: no hidden "
                      f"personas; skipping.{utils.Colors.ENDC}")
            return

        # Backward lookup: oid -> [(label, evidence_rows, type, is_synthetic), ...]
        from collections import defaultdict as _ddict
        oid_to_hp: dict[str, list[tuple[str, int, str, bool]]] = _ddict(list)
        for hp in self.user_profile.hidden_personas:
            for oid in hp.evidence_oids:
                oid_to_hp[str(oid)].append(
                    (hp.label, int(hp.evidence_rows or 0), hp.type, bool(hp.is_synthetic))
                )

        # Canonical lookup (positive + negative) keyed by normalized text.
        canonical_lookup: dict[str, CrossReferencedPersona] = {}
        for cr in self.cross_referenced_personas:
            canonical_lookup[_normalize_persona_text(cr.persona_item)] = cr
            for ap in self._canonical_groups.get(_normalize_persona_text(cr.persona_item), []):
                canonical_lookup.setdefault(_normalize_persona_text(ap.persona_item), cr)
        for cr in self.cross_referenced_negatives:
            canonical_lookup[_normalize_persona_text(cr.persona_item)] = cr
            for ap in self._negative_canonical_groups.get(_normalize_persona_text(cr.persona_item), []):
                canonical_lookup.setdefault(_normalize_persona_text(ap.persona_item), cr)

        # Walk every atom that has a surviving canonical and a matching oid.
        # Dedup atoms with the same canonical from the same oid (the same
        # rule save_to_backend uses when building the per-event preferences).
        per_event_seen: set[tuple[str, str]] = set()
        for ap in list(self.atomic_personas) + list(self.negative_personas):
            canonical_key = _normalize_persona_text(ap.persona_item)
            cr = canonical_lookup.get(canonical_key)
            if cr is None:
                continue  # canonical filtered out — nothing to link
            cr_key = _normalize_persona_text(cr.persona_item)
            link_key = (str(ap.source_object_id or ""), cr_key)
            if link_key in per_event_seen:
                continue
            per_event_seen.add(link_key)

            matches = oid_to_hp.get(str(ap.source_object_id or ""), [])
            if matches:
                # Tie-break by evidence_rows (largest wins).
                best = max(matches, key=lambda m: m[1])
                provisional_label, provisional_rows, hp_type, is_synth = best
                self._preference_links[link_key] = {
                    "original_label": provisional_label,
                    "label": provisional_label,
                    "evidence_rows": provisional_rows,
                    "hp_type": hp_type,
                    "is_synthetic": is_synth,
                    "link_provenance": "hashtag_overlap_v1",
                    "canonical_persona_item": cr.persona_item,
                    "motivation_audit": {},
                }
            else:
                # Unmatched preferences carry no link — preserved for traceability.
                self._preference_links[link_key] = {
                    "original_label": None,
                    "label": None,
                    "evidence_rows": 0,
                    "hp_type": "",
                    "is_synthetic": False,
                    "link_provenance": "hashtag_overlap_v1",
                    "canonical_persona_item": cr.persona_item,
                    "motivation_audit": {},
                }

        if self.verbose:
            n_linked = sum(1 for v in self._preference_links.values() if v["label"])
            n_total = len(self._preference_links)
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] "
                  f"link_preferences_to_hidden_personas: {n_linked}/{n_total} "
                  f"preferences provisionally linked via hashtag overlap.{utils.Colors.ENDC}")

    # -- Audit validators (deterministic, run AFTER LLM, before commit) -

    @staticmethod
    def _validate_cluster_specificity_for_step9(
        label: str, description: str, hp_type: str,
        evidence_hashtags: list, privacy_ratio: float,
    ) -> tuple[bool, str]:
        """Step 9 cluster-level specificity gate.

        Mirrors the per-preference audit gate (`_audit_validate_specificity`)
        upstream — applies the same type-specific blocklists / floors to
        the cluster's LABEL + DESCRIPTION + privacy_ratio before the
        cluster propagates into the voice / app_personas / save chain.
        Returns ``(passed, reason)``. Without this gate, Step 22 catches
        the violation but the cluster has already anchored downstream
        artifacts; this is the audit-symmetric early drop.
        """
        text = f"{label} {description}".lower()

        if hp_type == "parasocial_attachment":
            # At least one evidence hashtag (the figure tag) must appear
            # in the cluster label/description — same predicate as
            # the audit's per-preference proper-noun check.
            ev_tags = [h.strip("#").lower() for h in (evidence_hashtags or [])]
            if not any(tag and tag in text for tag in ev_tags):
                return False, "parasocial_no_named_figure"

        elif hp_type == "intimate_interest":
            if any(b in text for b in _INTIMATE_GENERIC_BLOCKLIST):
                return False, "intimate_generic_phrasing"

        elif hp_type == "medical_aesthetic_concern":
            if not any(m in f" {text} " for m in _MEDICAL_ACTIVE_USE_MARKERS):
                return False, "medical_no_active_use"

        elif hp_type == "covert_concern":
            if any(b in text for b in _COVERT_CONCERN_GENERIC_BLOCKLIST):
                return False, "covert_generic_phrasing"

        elif hp_type == "compensatory_need":
            if privacy_ratio < 0.7:
                return False, "compensatory_low_privacy_ratio"

        return True, ""

    @staticmethod
    def _audit_validate_specificity(decision: str, hp_type: str,
                                    persona_item: str, rationale: str,
                                    cluster_evidence_hashtags: list[str],
                                    cluster_privacy_ratio: float) -> tuple[bool, str]:
        """Type-specific specificity post-hoc validator.

        Returns ``(passed, downgrade_reason)``. When ``passed`` is False,
        the caller must downgrade the decision to FLAG (or, for medical
        curiosity, to SHORT_TERM_EPISODIC). Embedding these checks in
        the prompt drifts under load; deterministic code does not.
        """
        if decision != "CONFIRMED":
            return True, ""
        pi_low = (persona_item or "").lower()
        rat_low = (rationale or "").lower()

        if hp_type == "parasocial_attachment":
            # Need a proper-noun figure name in pi or rationale that overlaps
            # the cluster's evidence_hashtags (which contain the figure tag).
            ev_low = [h.strip("#").lower() for h in cluster_evidence_hashtags]
            if not any(tag and tag in pi_low or tag in rat_low for tag in ev_low):
                return False, "parasocial_no_named_figure"

        elif hp_type == "intimate_interest":
            if any(b in pi_low for b in _INTIMATE_GENERIC_BLOCKLIST):
                return False, "intimate_generic_phrasing"

        elif hp_type == "medical_aesthetic_concern":
            if not any(m in f" {pi_low} " for m in _MEDICAL_ACTIVE_USE_MARKERS):
                return False, "medical_no_active_use"

        elif hp_type == "covert_concern":
            if any(b in pi_low for b in _COVERT_CONCERN_GENERIC_BLOCKLIST):
                return False, "covert_generic_phrasing"

        elif hp_type == "compensatory_need":
            if cluster_privacy_ratio < 0.7:
                return False, "compensatory_low_privacy_ratio"

        return True, ""

    @staticmethod
    def _audit_apply_horizon_rule(decision: str, motivation_depth: str,
                                  hp_type: str, time_horizon: str,
                                  is_protected: bool) -> tuple[str, str, str]:
        """Hard depth-vs-horizon rules: short-term-horizon prefs cannot
        CONFIRM into stable-trait clusters; auto-downgrade.

        Returns ``(new_decision, new_motivation_depth, downgrade_reason)``.
        Empty downgrade_reason means no change.
        """
        if (decision == "CONFIRMED"
                and time_horizon == "short_term"
                and hp_type in MOTIVATION_AUDIT_STABLE_TRAIT_TYPES):
            return "SHORT_TERM_EPISODIC", "medium_episodic", "short_horizon_into_stable_trait"
        # Protected prefs never auto-flip to SURFACE_ENGAGEMENT — protect
        # contradiction-survivor / high-confidence signal.
        if is_protected and decision == "SURFACE_ENGAGEMENT":
            return "FLAG", motivation_depth, "protected_pref_blocked_surface"
        return decision, motivation_depth, ""

    @staticmethod
    def _audit_apply_protection_floor(decision: str, fit_confidence: float,
                                      is_protected: bool) -> tuple[str, str]:
        """Protected (update_history-bearing or high-confidence) prefs
        require fit_confidence < MOTIVATION_AUDIT_PROTECTED_REMOVE_FLOOR
        before REMOVE is honored.
        """
        if (decision == "REMOVE"
                and is_protected
                and fit_confidence >= MOTIVATION_AUDIT_PROTECTED_REMOVE_FLOOR):
            return "FLAG", "protected_remove_blocked"
        return decision, ""

    @staticmethod
    def _audit_validate_confirm_floor(decision: str, motivation_depth: str,
                                      fit_confidence: float) -> tuple[str, str]:
        """CONFIRMED requires deep_latent + fit_confidence >= floor.
        REASSIGN holds the same bar.
        """
        is_confirm_class = decision == "CONFIRMED" or decision.startswith("REASSIGN:")
        if not is_confirm_class:
            return decision, ""
        if motivation_depth != "deep_latent":
            return ("SURFACE_ENGAGEMENT" if motivation_depth == "shallow_situational"
                    else "SHORT_TERM_EPISODIC"), "confirm_required_deep_latent"
        if fit_confidence < MOTIVATION_AUDIT_MIN_CONFIRM_CONFIDENCE:
            return "FLAG", "confirm_low_fit_confidence"
        return decision, ""

    def _audit_resolve_decision(self, raw: dict, *, hp_type: str,
                                cluster_evidence_hashtags: list[str],
                                cluster_privacy_ratio: float,
                                persona_item: str,
                                time_horizon: str,
                                is_protected: bool) -> dict:
        """Apply all post-hoc deterministic validators to a single LLM
        decision. Returns a finalized audit record with `decision`,
        `motivation_depth`, `fit_confidence`, `frame_invoked`,
        `rationale`, `validator_passed`, `downgrade_reasons`.
        """
        decision = str(raw.get("decision") or "FLAG")
        motivation_depth = str(raw.get("motivation_depth") or "shallow_situational")
        try:
            fit_confidence = float(raw.get("fit_confidence", 0.0))
        except (ValueError, TypeError):
            fit_confidence = 0.0
        fit_confidence = max(0.0, min(1.0, fit_confidence))
        frame_invoked = str(raw.get("frame_invoked") or "none")
        if (frame_invoked not in MOTIVATION_AUDIT_DEEP_FRAMES
                and frame_invoked not in MOTIVATION_AUDIT_SURFACE_FRAMES):
            frame_invoked = "none"
        rationale = str(raw.get("rationale") or "").strip()

        downgrade_reasons: list[str] = []

        # Specificity gate (parasocial / intimate / medical / covert / compensatory).
        passed, reason = self._audit_validate_specificity(
            decision, hp_type, persona_item, rationale,
            cluster_evidence_hashtags, cluster_privacy_ratio,
        )
        if not passed:
            downgrade_reasons.append(reason)
            # Medical curiosity-only → SHORT_TERM_EPISODIC; everything else → FLAG.
            decision = ("SHORT_TERM_EPISODIC" if reason == "medical_no_active_use"
                        else "FLAG")
            if decision == "SHORT_TERM_EPISODIC":
                motivation_depth = "medium_episodic"

        # Hard horizon rule (short-term cannot CONFIRM into stable traits).
        decision, motivation_depth, hreason = self._audit_apply_horizon_rule(
            decision, motivation_depth, hp_type, time_horizon, is_protected,
        )
        if hreason:
            downgrade_reasons.append(hreason)

        # Protection floor (protected prefs need fit < 0.3 to REMOVE).
        decision, preason = self._audit_apply_protection_floor(
            decision, fit_confidence, is_protected,
        )
        if preason:
            downgrade_reasons.append(preason)

        # Confirm-class floor (must be deep_latent + fit >= 0.6).
        decision, creason = self._audit_validate_confirm_floor(
            decision, motivation_depth, fit_confidence,
        )
        if creason:
            downgrade_reasons.append(creason)

        return {
            "decision": decision,
            "motivation_depth": motivation_depth,
            "fit_confidence": round(fit_confidence, 3),
            "frame_invoked": frame_invoked,
            "rationale": rationale,
            "validator_passed": not downgrade_reasons,
            "downgrade_reasons": downgrade_reasons,
        }

    # -- Audit step (LLM) -----------------------------------------------

    def audit_hidden_persona_motivations(self) -> None:
        """Step 22: re-judge every (preference, hidden_persona) link with
        a flagship-tier LLM call against named motivation frames.
        Parsimony-biased — defaults to SURFACE_ENGAGEMENT under ambiguity
        rather than force-fitting deep motivation.

        Operates on ``self._preference_links`` (built by Step 21) and
        populates each entry's ``motivation_audit`` block. The mutation
        of ``label`` happens here per the audit decision; the original
        is preserved as ``original_label``.

        Synthetic ``sensitive_life_event`` clusters are bypassed
        entirely (audit_status: "synthetic_skipped"). The cluster's
        planted evidence rows are never audited.
        """
        if not getattr(self, "_preference_links", None):
            if self.verbose:
                print(f"{utils.Colors.OKBLUE}[User {self.user_id}] "
                      f"audit_hidden_persona_motivations: no link table; "
                      f"skipping.{utils.Colors.ENDC}")
            return
        if not self.user_profile or not self.user_profile.hidden_personas:
            return

        if self.llm_client is None:
            # Claude-Code-subagent mode — leave links as hashtag-overlap
            # provenance and tag every preference's audit as skipped.
            for entry in self._preference_links.values():
                if entry["label"]:
                    entry["motivation_audit"] = {
                        "decision": "AUDIT_SKIPPED",
                        "audit_status": "no_llm_client",
                    }
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                      f"audit_hidden_persona_motivations: no llm_client; "
                      f"skipping audit.{utils.Colors.ENDC}")
            return

        # Build per-cluster preference batches.
        # Skip synthetic sensitive_life_event clusters AND any planted-row
        # preferences (which carry the _planted_sensitive_event marker via
        # save_to_backend — but the link table is built BEFORE save runs,
        # so we identify synthetic preferences by their cluster's flag).
        cluster_by_label: dict[str, HiddenPersona] = {
            hp.label: hp for hp in self.user_profile.hidden_personas
        }
        non_synthetic_clusters = [
            hp for hp in self.user_profile.hidden_personas if not hp.is_synthetic
        ]

        # Mark synthetic-cluster prefs as skipped up-front.
        for link_key, entry in self._preference_links.items():
            if entry.get("is_synthetic"):
                entry["motivation_audit"] = {
                    "decision": "AUDIT_SKIPPED",
                    "audit_status": "synthetic_skipped",
                }

        # Per-canonical metadata lookup for protected / time_horizon flags.
        cr_lookup: dict[str, CrossReferencedPersona] = {}
        for cr in list(self.cross_referenced_personas) + list(self.cross_referenced_negatives):
            cr_lookup[_normalize_persona_text(cr.persona_item)] = cr

        # Per-row source-context lookup (for content_snippet etc.).
        # Map source_object_id -> (interaction_type, action label, content snippet).
        row_ctx: dict[str, dict] = {}
        for ap in self.atomic_personas + self.negative_personas:
            oid = str(ap.source_object_id or "")
            if oid and oid not in row_ctx:
                row_ctx[oid] = {
                    "app": self._row_app.get(oid, "") if hasattr(self, "_row_app") else "",
                    "action": "",
                    "source_interaction_type": ap.source_interaction_type or "",
                    "content_snippet": "",
                    "source_hashtags": list(ap.source_hashtags or [])[:8],
                }
        # Augment with action/content from sampled metadata if available.
        action_by_oid = getattr(self, "_action_by_oid", {}) or {}
        content_by_oid = getattr(self, "_content_by_oid", {}) or {}
        for oid, meta in action_by_oid.items():
            if oid in row_ctx and isinstance(meta, dict):
                row_ctx[oid]["action"] = meta.get("action", "") or ""
        for oid, content in content_by_oid.items():
            if oid in row_ctx and isinstance(content, dict):
                snip = (content.get("caption") or content.get("text")
                        or content.get("overall_description") or "")
                row_ctx[oid]["content_snippet"] = (snip or "")[:200]

        # Group preferences by cluster label.
        from collections import defaultdict as _ddict
        prefs_by_cluster: dict[str, list[tuple[tuple[str, str], dict]]] = _ddict(list)
        for link_key, entry in self._preference_links.items():
            label = entry.get("label")
            if not label:
                continue
            if entry.get("is_synthetic"):
                continue  # already marked skipped
            prefs_by_cluster[label].append((link_key, entry))

        n_audited = 0
        n_decisions: dict[str, int] = _ddict(int)
        n_validator_failed = 0
        n_decoy_batches_failed = 0
        model_version = getattr(self.llm_client, "model", "unknown") or "unknown"

        for cluster in non_synthetic_clusters:
            label = cluster.label
            entries = prefs_by_cluster.get(label, [])
            if not entries:
                continue

            # Sort for idempotence: by source_timestamp via cr.persona_item lookup,
            # then by canonical text — falling back to link_key tuple.
            def _sort_key(e):
                lk, val = e
                cr = cr_lookup.get(_normalize_persona_text(val.get("canonical_persona_item", "")))
                ts = 0
                if cr is not None:
                    # Earliest atom timestamp for this canonical (deterministic).
                    atoms = self._canonical_groups.get(_normalize_persona_text(cr.persona_item), []) \
                        or self._negative_canonical_groups.get(_normalize_persona_text(cr.persona_item), [])
                    if atoms:
                        ts = min((a.source_timestamp or 0) for a in atoms)
                return (ts, val.get("canonical_persona_item", ""), lk)
            entries.sort(key=_sort_key)

            # Prepare other-cluster menu (closed reassignment options).
            others_menu = [
                {
                    "label": hp.label,
                    "type": hp.type,
                    "description": hp.description,
                }
                for hp in non_synthetic_clusters if hp.label != label
            ]

            # Build batches with decoys.
            BATCH = MOTIVATION_AUDIT_BATCH_SIZE
            for batch_start in range(0, len(entries), BATCH):
                batch = entries[batch_start:batch_start + BATCH]

                # Decoy injection: pick one entry from a different cluster of
                # the same user, deterministic seed.
                decoy_entries: list[tuple[tuple[str, str], dict, str]] = []
                if MOTIVATION_AUDIT_DECOYS_PER_BATCH > 0 and len(non_synthetic_clusters) > 1:
                    # Deterministic decoy selection.
                    seed = hash((str(self.user_id), label, batch_start)) & 0xffffffff
                    rng = random.Random(seed)
                    other_clusters = [hp for hp in non_synthetic_clusters if hp.label != label]
                    rng.shuffle(other_clusters)
                    for hp_other in other_clusters:
                        candidates = [
                            (lk, val) for lk, val in prefs_by_cluster.get(hp_other.label, [])
                            if val.get("label") == hp_other.label
                        ]
                        if not candidates:
                            continue
                        rng.shuffle(candidates)
                        for lk, val in candidates[:MOTIVATION_AUDIT_DECOYS_PER_BATCH]:
                            decoy_entries.append((lk, val, hp_other.label))
                        if len(decoy_entries) >= MOTIVATION_AUDIT_DECOYS_PER_BATCH:
                            break

                # Compose LLM payload (real prefs first, then decoys appended).
                def _to_payload(lk, val, decoy: bool = False) -> dict:
                    cr = cr_lookup.get(_normalize_persona_text(val.get("canonical_persona_item", "")))
                    polarity = "positive"
                    time_horizon = "long_term"
                    xref = 0.0
                    update_history = []
                    if cr is not None:
                        polarity = ("negative"
                                    if "negative" in (cr.source_interaction_type or "")
                                    else "positive")
                        time_horizon = getattr(cr, "time_horizon", "long_term")
                        xref = float(cr.confidence_cross_referenced or 0.0)
                        update_history = list(cr.update_history or [])
                    is_protected = bool(update_history) or xref >= 5.0
                    oid = lk[0]
                    ctx = row_ctx.get(oid, {})
                    pref_key = f"{oid}::{val.get('canonical_persona_item','')[:80]}"
                    if decoy:
                        pref_key = f"DECOY::{pref_key}"
                    return {
                        "preference_key": pref_key,
                        "persona_item": val.get("canonical_persona_item", ""),
                        "category": cr.category if cr else "",
                        "polarity": polarity,
                        "time_horizon": time_horizon,
                        "confidence_cross_referenced": xref,
                        "protected": is_protected,
                        "source_hashtags": ctx.get("source_hashtags", []),
                        "event_context": {
                            "app": ctx.get("app", ""),
                            "action": ctx.get("action", ""),
                            "source_interaction_type": ctx.get("source_interaction_type", ""),
                            "content_snippet": ctx.get("content_snippet", ""),
                        },
                        "_link_key": lk,
                        "_is_decoy": decoy,
                        "_is_protected": is_protected,
                        "_time_horizon": time_horizon,
                    }

                payload_real = [_to_payload(lk, val) for lk, val in batch]
                payload_decoy = [_to_payload(lk, val, decoy=True) for lk, val, _ in decoy_entries]
                payload_all = payload_real + payload_decoy

                # Strip private-prefixed fields before sending to the LLM.
                payload_for_llm = [
                    {k: v for k, v in p.items() if not k.startswith("_")}
                    for p in payload_all
                ]

                cluster_card = {
                    "type": cluster.type,
                    "label": cluster.label,
                    "description": cluster.description,
                    "inferred_motivation": cluster.inferred_motivation,
                    "evidence_hashtags": list(cluster.evidence_hashtags or []),
                    "privacy_ratio": float(cluster.privacy_ratio or 0.0),
                    "temporal_spread_days": int(cluster.temporal_spread_days or 0),
                    "app_distribution": dict(cluster.app_distribution or {}),
                }

                prompt = prompts.audit_hidden_persona_motivations_prompt(
                    cluster=cluster_card,
                    other_clusters_menu=others_menu,
                    preferences_with_decoys=payload_for_llm,
                )

                response = self._query_llm_with_retry(prompt, temperature=0.0)
                if not response:
                    # Audit failed for this batch — leave entries unmutated,
                    # tag as audit-failed. Caller can re-run later.
                    for lk, val in batch:
                        val["motivation_audit"] = {
                            "decision": "AUDIT_SKIPPED",
                            "audit_status": "llm_call_failed",
                            "model_version": model_version,
                        }
                    continue

                parsed = utils.extract_json_from_response(response)
                if not isinstance(parsed, list):
                    # One retry at temperature 0; on second failure, FLAG the batch.
                    response2 = self._query_llm_with_retry(prompt, temperature=0.0)
                    parsed = utils.extract_json_from_response(response2) if response2 else None
                    if not isinstance(parsed, list):
                        for lk, val in batch:
                            val["motivation_audit"] = {
                                "decision": "FLAG",
                                "audit_status": "unparseable_llm_response",
                                "model_version": model_version,
                            }
                        continue

                # Index returned decisions by preference_key.
                decisions_by_key: dict[str, dict] = {}
                for d in parsed:
                    if isinstance(d, dict) and d.get("preference_key"):
                        decisions_by_key[str(d["preference_key"])] = d

                # Decoy calibration check.
                if payload_decoy:
                    decoy_confirm = 0
                    for p in payload_decoy:
                        d = decisions_by_key.get(p["preference_key"], {})
                        if str(d.get("decision", "")).startswith("CONFIRMED"):
                            decoy_confirm += 1
                    decoy_rate = decoy_confirm / max(1, len(payload_decoy))
                    if decoy_rate > MOTIVATION_AUDIT_DECOY_BIAS_THRESHOLD:
                        n_decoy_batches_failed += 1
                        # Calibration failed for this batch — FLAG real prefs
                        # so a follow-up per-preference re-run can target them.
                        for p in payload_real:
                            lk = p["_link_key"]
                            entry = self._preference_links.get(lk, {})
                            entry["motivation_audit"] = {
                                "decision": "FLAG",
                                "audit_status": "decoy_calibration_failed",
                                "decoy_confirm_rate": round(decoy_rate, 3),
                                "model_version": model_version,
                            }
                        continue

                # Apply each decision to the real entries.
                for p in payload_real:
                    lk = p["_link_key"]
                    entry = self._preference_links.get(lk)
                    if entry is None:
                        continue
                    raw = decisions_by_key.get(p["preference_key"])
                    if not raw:
                        # LLM didn't return a decision for this row — FLAG.
                        entry["motivation_audit"] = {
                            "decision": "FLAG",
                            "audit_status": "llm_omitted_decision",
                            "model_version": model_version,
                        }
                        continue

                    finalized = self._audit_resolve_decision(
                        raw,
                        hp_type=cluster.type,
                        cluster_evidence_hashtags=list(cluster.evidence_hashtags or []),
                        cluster_privacy_ratio=float(cluster.privacy_ratio or 0.0),
                        persona_item=p["persona_item"],
                        time_horizon=p["_time_horizon"],
                        is_protected=p["_is_protected"],
                    )

                    decision = finalized["decision"]
                    n_audited += 1
                    n_decisions[decision.split(":", 1)[0]] += 1
                    if not finalized["validator_passed"]:
                        n_validator_failed += 1

                    # Mutate the link's `label` per the audit decision.
                    new_label = entry.get("label")
                    if decision == "REMOVE" or decision == "SURFACE_ENGAGEMENT":
                        new_label = None
                    elif decision == "SHORT_TERM_EPISODIC":
                        # Stable-trait demotions clear the link; episodic
                        # support is via short-term canonical's stop_condition,
                        # not the cluster system.
                        if cluster.type in MOTIVATION_AUDIT_STABLE_TRAIT_TYPES:
                            new_label = None
                    elif decision.startswith("REASSIGN:"):
                        target_label = decision.split(":", 1)[1].strip()
                        # Validate the target is a real cluster of this user.
                        if target_label in cluster_by_label and not cluster_by_label[target_label].is_synthetic:
                            new_label = target_label
                        else:
                            # Invalid reassign target — FLAG.
                            decision = "FLAG"
                            finalized["decision"] = decision
                            finalized.setdefault("downgrade_reasons", []).append("reassign_target_invalid")
                            finalized["validator_passed"] = False
                            n_validator_failed += 1
                    # CONFIRMED / NO_OTHER_CLUSTER_FITS / FLAG keep the original label.

                    entry["label"] = new_label
                    entry["motivation_audit"] = {
                        **finalized,
                        "original_label": entry.get("original_label"),
                        "model_version": model_version,
                        "audit_status": "audited",
                    }

        if self.verbose:
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] "
                  f"audit_hidden_persona_motivations: "
                  f"audited={n_audited}, decisions={dict(n_decisions)}, "
                  f"validator_failed={n_validator_failed}, "
                  f"decoy_batches_failed={n_decoy_batches_failed}.{utils.Colors.ENDC}")

    # -- Cluster-level rollup --------------------------------------------

    def aggregate_motivation_audit_to_summary(self) -> None:
        """Step 23: roll up per-preference audit decisions into per-cluster
        rollup metrics + cluster_status tier (validated / mixed_evidence /
        contested / likely_invalid). Never mutates the cluster — only
        annotates it. Also emits a profile-level over_attribution_warning
        when the user's mean cluster surface_share is high.
        """
        if not getattr(self, "_preference_links", None):
            return
        if not self.user_profile or not self.user_profile.hidden_personas:
            return

        from collections import defaultdict as _ddict, Counter as _Counter
        per_cluster_counts: dict[str, dict[str, int]] = _ddict(lambda: _ddict(int))
        # Per-cluster frame tally — `frame_invoked` counts across the
        # cluster's audited prefs. Used to compute the cluster's
        # dominant_frame (modal, ties broken by deep-frame priority) so
        # downstream voice / content / auto-QA generators can ground
        # their text in the cluster's strongest motivational signature.
        per_cluster_frames: dict[str, _Counter] = _ddict(_Counter)

        for entry in self._preference_links.values():
            audit = entry.get("motivation_audit") or {}
            audit_status = audit.get("audit_status")
            if audit_status not in ("audited",):
                continue
            cluster_label = entry.get("original_label")
            if not cluster_label:
                continue
            decision = str(audit.get("decision") or "FLAG")
            depth = str(audit.get("motivation_depth") or "")
            head = decision.split(":", 1)[0]
            counts = per_cluster_counts[cluster_label]
            counts["n_audited"] += 1
            counts[f"n_{head.lower()}"] += 1
            if depth == "deep_latent":
                counts["n_deep_latent"] += 1
            frame = audit.get("frame_invoked") or "none"
            if frame and frame != "none":
                per_cluster_frames[cluster_label][frame] += 1

        # Annotate each non-synthetic cluster with its rollup.
        surface_shares: list[float] = []
        for hp in self.user_profile.hidden_personas:
            if hp.is_synthetic:
                # Synthetic cluster — record skip status.
                hp.motivation_audit = {
                    "audit_status": "synthetic_skipped",
                }
                continue
            counts = per_cluster_counts.get(hp.label, _ddict(int))
            n = int(counts.get("n_audited", 0))
            if n == 0:
                hp.motivation_audit = {
                    "audit_status": "no_audited_preferences",
                    "cluster_status": "unaudited",
                }
                continue
            n_confirmed = int(counts.get("n_confirmed", 0))
            n_reassigned = int(counts.get("n_reassign", 0))
            n_surface = int(counts.get("n_surface_engagement", 0))
            n_short_term = int(counts.get("n_short_term_episodic", 0))
            n_removed = int(counts.get("n_remove", 0))
            n_flagged = int(counts.get("n_flag", 0))
            n_no_fit = int(counts.get("n_no_other_cluster_fits", 0))
            n_deep = int(counts.get("n_deep_latent", 0))

            confirm_rate = n_confirmed / n
            deep_latent_rate = n_deep / n
            surface_share = n_surface / n
            surface_shares.append(surface_share)

            if confirm_rate >= 0.7 and deep_latent_rate >= 0.6:
                cluster_status = "validated"
            elif 0.5 <= confirm_rate < 0.7:
                cluster_status = "mixed_evidence"
            elif 0.3 <= confirm_rate < 0.5 or surface_share >= 0.5:
                cluster_status = "contested"
            elif confirm_rate < 0.3:
                cluster_status = "likely_invalid"
            else:
                cluster_status = "mixed_evidence"

            # Modal frame across audited prefs. Ties broken by deep-frame
            # priority — a cluster with one tied surface frame and one tied
            # deep frame picks the deep one, preserving the deep-latent
            # reading whenever it's available. None when no frames were
            # invoked (audit failed to attribute, or all "none").
            frame_tally = per_cluster_frames.get(hp.label) or _Counter()
            dominant_frame = None
            if frame_tally:
                top_count = max(frame_tally.values())
                tied = [f for f, c in frame_tally.items() if c == top_count]
                deep_tied = [f for f in tied if f in MOTIVATION_AUDIT_DEEP_FRAMES]
                # Prefer deep frames on ties; otherwise alphabetical for
                # determinism across runs.
                pool = deep_tied or tied
                dominant_frame = sorted(pool)[0]

            hp.motivation_audit = {
                "audit_status": "audited",
                "n_audited": n,
                "n_confirmed": n_confirmed,
                "n_reassigned": n_reassigned,
                "n_surface_engagement": n_surface,
                "n_short_term_episodic": n_short_term,
                "n_removed": n_removed,
                "n_flagged": n_flagged,
                "n_no_other_cluster_fits": n_no_fit,
                "confirm_rate": round(confirm_rate, 3),
                "deep_latent_rate": round(deep_latent_rate, 3),
                "surface_share": round(surface_share, 3),
                "cluster_status": cluster_status,
                "dominant_frame": dominant_frame,
                "frame_distribution": dict(frame_tally),
            }

        # Profile-level over-attribution warning.
        if surface_shares:
            mean_surface = sum(surface_shares) / len(surface_shares)
            if mean_surface >= MOTIVATION_AUDIT_USER_OVER_ATTRIBUTION_RATE:
                self._motivation_audit_user_warning = {
                    "over_attribution_warning": True,
                    "mean_cluster_surface_share": round(mean_surface, 3),
                    "threshold": MOTIVATION_AUDIT_USER_OVER_ATTRIBUTION_RATE,
                }
            else:
                self._motivation_audit_user_warning = {
                    "over_attribution_warning": False,
                    "mean_cluster_surface_share": round(mean_surface, 3),
                }

        if self.verbose:
            statuses = [
                getattr(hp, "motivation_audit", {}).get("cluster_status", "unaudited")
                for hp in self.user_profile.hidden_personas
            ]
            from collections import Counter as _Counter
            print(f"{utils.Colors.OKGREEN}[User {self.user_id}] "
                  f"aggregate_motivation_audit_to_summary: "
                  f"cluster_status={dict(_Counter(statuses))}, "
                  f"user_warning={self._motivation_audit_user_warning}{utils.Colors.ENDC}")

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

        # Per-preference hidden-persona links live on self._preference_links,
        # populated by Step 21 (link_preferences_to_hidden_personas) and
        # mutated by Step 22 (audit_hidden_persona_motivations). Each entry's
        # `label` is the AUDITED label (post-mutation), `original_label` is
        # the pre-audit hashtag-overlap label, and `motivation_audit` carries
        # the structured decision + rationale for the change. Save_to_backend
        # is a pure consumer here — it does no inline label computation.
        preference_links: dict[tuple[str, str], dict] = (
            getattr(self, "_preference_links", {}) or {}
        )

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
                               desc=f"[User {self.user_id}] Step 22: Building events",
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

            # Build nested preferences list — only surviving canonicals.
            # Dedup atoms that map to the same canonical: when two raw rows
            # of the same event both produced the same canonical persona_item
            # (e.g. both rows about Burger King → "Enjoys fast-food burger
            # content..."), keep ONLY the atom with the highest per-row
            # confidence_score_init. Otherwise the event's preferences list
            # carries the same persona_item twice with different init scores,
            # which surfaces as visible duplicates in the rendered HTML.
            best_atom_per_canonical: dict[str, "AtomicPersona"] = {}
            for ap in atoms:
                key = _normalize_persona_text(ap.persona_item)
                if key not in canonical_lookup:
                    continue
                prev = best_atom_per_canonical.get(key)
                if prev is None or (ap.confidence_score_init or 0.0) > (prev.confidence_score_init or 0.0):
                    best_atom_per_canonical[key] = ap
            preferences: list[dict] = []
            for ap in best_atom_per_canonical.values():
                key = _normalize_persona_text(ap.persona_item)
                cr = canonical_lookup.get(key)
                if not cr:
                    continue  # this atomic's canonical was filtered out

                ann = all_annotated_items.get(cr.persona_item)
                # R8: `split` and `over_personalization_irrelevant` are no longer
                # emitted by data-gen. Eval tasks pick their own test moments
                # by cutting the timeline at any T_test they choose; the
                # history is just the history.

                # Build merged update_history: temporal entries (no raw timestamp)
                # + related_personas folded in as similar/contradictory entries.
                # Causality: only keep entries whose timestamp <= this event's time.
                # Key order: update_type, preference, formatted_timestamp, then extras.
                _HISTORY_KEY_ORDER = ["update_type", "preference", "formatted_timestamp",
                                      "source_app", "occurrence", "total_occurrences", "description",
                                      # Cross-polarity contradiction metadata (Step 7)
                                      "resolution", "opposing_polarity",
                                      "prior_corroboration_count", "required_precedent",
                                      "earlier_rows_after_flip",
                                      "stronger_row_count", "weaker_row_count", "dominance_ratio"]
                event_ts = ap.source_timestamp
                event_oid = str(ap.source_object_id or "")

                def _is_before(h_ts, h_oid) -> bool:
                    """Causality: entry is 'before' THIS event in HTML display order.
                    HTML sorts events by (source_timestamp, source_object_id) —
                    same order we use here. When the entry carries an oid, use
                    lexicographic (ts, oid) compare. When it doesn't, use the
                    stricter ts-only strict inequality (drop same-timestamp
                    entries we can't disambiguate)."""
                    if h_oid:
                        return (int(h_ts or 0), str(h_oid)) < (int(event_ts or 0), event_oid)
                    return int(h_ts or 0) < int(event_ts or 0)

                merged_history = []
                for h in (cr.update_history or []):
                    raw = dict(h)
                    h_ts = raw.pop("timestamp", 0)
                    hist_oid = raw.pop("source_object_id", None)
                    # Causality filter: the referenced event must come BEFORE
                    # this event in the HTML display order (lexicographic by
                    # (ts, source_object_id)).
                    if not _is_before(h_ts, hist_oid):
                        continue
                    if raw.get("update_type") == "new":
                        continue  # redundant — event timestamp already shows first appearance
                    # Drop self-referencing preference (same as parent persona_item)
                    if raw.get("preference") == cr.persona_item:
                        raw.pop("preference")
                    # Resolve source_object_id → source_app for reinforced entries
                    if hist_oid:
                        raw["source_app"] = self._row_app.get(hist_oid, "")
                    # For entries with a preference (evolution/contradicted),
                    # use target canonical's app for display.
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
                    # Recover the related preference's first occurrence (ts + oid).
                    rel_atoms = self._canonical_groups.get(rel_key, [])
                    if not rel_atoms:
                        rel_atoms = self._negative_canonical_groups.get(rel_key, [])
                    rel_first_ts = 0
                    rel_first_oid = ""
                    if rel_atoms:
                        rel_sorted = sorted(
                            ((a.source_timestamp, str(a.source_object_id or "")) for a in rel_atoms
                             if a.source_timestamp),
                            key=lambda x: x,
                        )
                        if rel_sorted:
                            rel_first_ts, rel_first_oid = rel_sorted[0]
                    # Strict causality: drop any related entry we cannot place
                    # in timeline order strictly BEFORE this event.
                    if not rel_first_ts or not _is_before(rel_first_ts, rel_first_oid):
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

                # Per-preference hidden-persona link is the AUDITED label.
                # Step 21 built the hashtag-overlap link table; Step 22 audit
                # may have downgraded the link to None (SURFACE_ENGAGEMENT /
                # REMOVE / SHORT_TERM_EPISODIC against stable-trait clusters)
                # or REASSIGNed it to a different cluster. Step 23 rolled
                # the per-preference decisions into per-cluster status. Here
                # we just consume the resolved link + audit trail.
                link_key = (str(ap.source_object_id or ""),
                            _normalize_persona_text(cr.persona_item))
                link_entry = preference_links.get(link_key) or {}
                hp_labels: list[str] = []
                if link_entry.get("label"):
                    hp_labels = [link_entry["label"]]
                motivation_audit_trail = link_entry.get("motivation_audit") or {}

                pref = {
                    "persona_item": cr.persona_item,
                    "category": cr.category,
                    "confidence_score_init": ap.confidence_score_init,
                    "confidence_cross_referenced": cr.confidence_cross_referenced,
                    "stereotype_mark": ann.stereotype_mark if ann else "neutral",
                    "hidden_persona_labels": hp_labels,
                    "link_provenance": link_entry.get("link_provenance", "hashtag_overlap_v1"),
                    "motivation_audit": motivation_audit_trail,
                    "update_history": merged_history,
                    "time_horizon": getattr(cr, "time_horizon", "long_term"),
                }
                # Stop condition only meaningful for short-term canonicals
                if getattr(cr, "time_horizon", "long_term") == "short_term":
                    sc = getattr(cr, "stop_condition", {}) or {}
                    if sc:
                        pref["stop_condition"] = sc

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
            # Step 19 (if it ran) pre-samples action + itype and stores them
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

            # Ad marker (Step 20). Invariant: is_ad ⇔ action ∈ AD_ACTIONS.
            if oid in self._ad_oids or sampled_entry["action"] in AD_ACTIONS:
                event["is_ad"] = True
                self._ad_oids.add(oid)  # keep set coherent if only action matched

            # Per-session geolocation (Step 15). Looked up via the session
            # index for this row; absent when location assignment didn't run.
            sess_idx = self._object_id_to_session.get(oid)
            if sess_idx is not None:
                loc = self._session_location.get(sess_idx)
                if loc:
                    event["event_location"] = loc

            # Attach synthetic content (Step 19). Chatbot, AI Studio, and
            # stubs are never in self._content_by_oid, so those events render
            # unchanged (both are conversation-only surfaces with no media body).
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
                    chatbot_p = asdict(chatbot_p)
                fb_prompt = prompts.generate_chatbot_conversation_prompt(
                    preferences=fallback_prefs,
                    conversation_type="knowledge_query",
                    conversation_type_description="User asks a factual or curiosity-driven question.",
                    user_profile={"name": self.user_profile.name, "gender": self.user_profile.gender,
                                  "career": self.user_profile.career, "bio": self.user_profile.bio},
                    chatbot_persona=chatbot_p if isinstance(chatbot_p, dict) else {},
                    interaction_type=itype,
                    num_turns=max(2, min(len(preferences) * 2, 6)),
                    user_voice=(self.user_profile.user_voice or {}),
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
            stub_event = {
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
            }
            sess_idx = self._object_id_to_session.get(interaction.object_id)
            if sess_idx is not None:
                loc = self._session_location.get(sess_idx)
                if loc:
                    stub_event["event_location"] = loc
            all_events.append(stub_event)
            n_stubs += 1

        if self.verbose and n_stubs:
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Added {n_stubs} implicit-negative stub events "
                  f"(no preferences, greyscale in HTML).{utils.Colors.ENDC}")

        # Assertion: no negative interaction event should have test-labeled preferences
        # R8: removed the "no test label on negative events" assertion — split
        # labels are no longer emitted by data-gen. Eval picks test moments
        # itself from the full history.

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

        # --- Step 21b: plant sensitive_life_event evidence rows ---
        # The synthetic sensitive_life_event hidden persona (Step 9b) carries
        # 1–3 episodes the user is privately navigating. Without visible
        # signal in the time-masked snapshot, the eval agent has no reason
        # to surface those topics — leak rate would trivially read 0. Plant
        # 2–4 implicit_positive engagement rows per episode so the agent
        # sees realistic recent activity and the restraint test fires.
        try:
            self._plant_sensitive_event_evidence_rows(per_app)
        except Exception as e:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] sensitive_event "
                      f"evidence planting failed: {e}{utils.Colors.ENDC}")

        # Re-sort each app's events after planting (planted rows have new
        # timestamps that may fall mid-history).
        for app_name in per_app:
            per_app[app_name].sort(key=lambda e: (int(e.get("source_timestamp") or 0),
                                                   e.get("source_object_id", "")))

        # --- Step 18B output merge: enrich AI_Studio events with the
        # conversation + cross-session memory metadata from
        # `self._ai_studio_records` (indexed by source_object_id). Every
        # AI_Studio per-app event picks up: `conversation`, `conversation_type`,
        # `prior_session_refs`, `memory_used_summary`,
        # `oblique_reference_to_hidden_personas`, `ai_studio_metadata`.
        # Source events that didn't successfully generate a conversation are
        # dropped from per_app["AI_Studio"] entirely (they have no value
        # without the AI dialog).
        ai_studio_records = getattr(self, "_ai_studio_records", []) or []
        if ai_studio_records:
            ai_studio_by_oid = {r.get("source_object_id", ""): r for r in ai_studio_records}
            kept = []
            for ev in per_app.get("AI_Studio", []):
                oid = ev.get("source_object_id", "")
                rec = ai_studio_by_oid.get(oid)
                if not rec or not rec.get("conversation"):
                    continue   # drop events with no generated conversation
                ev["conversation"] = rec["conversation"]
                ev["conversation_type"] = rec.get("conversation_type", "")
                ev["prior_session_refs"] = rec.get("prior_session_refs", [])
                ev["memory_used_summary"] = rec.get("memory_used_summary", "")
                ev["oblique_reference_to_hidden_personas"] = rec.get(
                    "oblique_reference_to_hidden_personas", []
                )
                ev["ai_studio_metadata"] = rec.get("ai_studio_metadata", {})
                kept.append(ev)
            per_app["AI_Studio"] = kept

        # --- Write per-app JSONs ---
        for app_name, events in per_app.items():
            filename = app_name.lower() + ".json"
            path = os.path.join(user_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(events, f, indent=2, ensure_ascii=False)

        # --- Write ai_studio_memory.json (sibling of calendar.json) ---
        ai_studio_mem = getattr(self, "_ai_studio_memory_state", None)
        if ai_studio_mem is not None and (
            ai_studio_mem.episodic_memory_items
            or ai_studio_mem.running_relational_state.intimacy_arc > 0
        ):
            from data_preparation import ai_studio_memory as _aism
            mem_path = os.path.join(user_dir, "ai_studio_memory.json")
            with open(mem_path, "w", encoding="utf-8") as f:
                json.dump(_aism.memory_state_to_dict(ai_studio_mem), f,
                          indent=2, ensure_ascii=False)

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
            # Deterministic exploration-vs-exploitation diversity score
            # over raw activities. Computed last so it sits alongside the
            # rest of the trait fields when asdict() runs.
            try:
                self.user_profile.exploration_exploitation = (
                    self._compute_exploration_exploitation()
                )
            except Exception as e:
                if self.verbose:
                    print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                          f"exploration_exploitation computation failed: {e}"
                          f"{utils.Colors.ENDC}")
                self.user_profile.exploration_exploitation = {}
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
            # Profile-level motivation-audit warning (Step 23 emits this when
            # the user's mean cluster surface_share is high — signals that
            # hashtag-overlap linking was over-attributing across the board).
            user_warning = getattr(self, "_motivation_audit_user_warning", {}) or {}
            if user_warning:
                profile_dict["motivation_audit"] = user_warning
            profile_path = os.path.join(user_dir, "profile.json")
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_dict, f, indent=2, ensure_ascii=False)

        # --- Write calendar.json (modification stream) ---
        if self._calendar_modifications:
            calendar_path = os.path.join(user_dir, "calendar.json")
            with open(calendar_path, "w", encoding="utf-8") as f:
                json.dump({"modifications": self._calendar_modifications}, f, indent=2, ensure_ascii=False)

        if self.verbose:
            total_events = len(all_events)
            total_prefs = sum(len(e["preferences"]) for e in all_events)
            print(f"{utils.Colors.OKBLUE}[User {self.user_id}] Saved to {user_dir}/ "
                  f"({total_events} events, {total_prefs} preference instances, per-app: "
                  f"{ {k: len(v) for k, v in per_app.items()} }){utils.Colors.ENDC}")
        return user_dir

    def run_extension_b(self) -> None:
        """Step 24: Run the Extension B layer — self-authored posts, DM
        threads, friends graph, and trending hashtags — on top of the
        files just written by save_to_backend. Merges Extension B
        into the main pipeline so a single `run_persona_pipeline.py`
        invocation produces a fully-complete persona backend, no separate
        CLI invocation required.

        Idempotent: rerunning replaces self-post events, DM threads,
        friends list, and trending with fresh generations while
        preserving all pipeline-authored events. Writes back to
        backend/{uid}/{profile.json, {app}.json ×3, trending.json}.

        Skipped gracefully when no LLM client is configured (Claude Code
        subagent mode handles this inline via skill.md).
        """
        if self.llm_client is None:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                      f"Skipping Extension B (no llm client; subagent mode "
                      f"handles it inline).{utils.Colors.ENDC}")
            return
        try:
            # Local import to avoid pulling extension_b unless this step runs
            from data_preparation.extension_b.main import run_extension_b
        except Exception as e:
            print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                  f"Extension B import failed ({type(e).__name__}: {e}) — "
                  f"skipping. Profile/self-posts/DMs/friends/trending will "
                  f"be empty.{utils.Colors.ENDC}")
            return
        try:
            # Deterministic per-user seed for Ext B RNG so regens are stable.
            try:
                seed = int(str(self.user_id)) * 8609 + 19
            except (ValueError, TypeError):
                seed = abs(hash(str(self.user_id))) % (2**31)
            run_extension_b(
                user_id=str(self.user_id),
                backend_dir=self.backend_dir,
                llm_client=self.llm_client,
                rng_seed=seed,
                dry_run=False,
                verbose=self.verbose,
            )
        except Exception as e:
            # Do not let Extension B failure wipe the pipeline's completed
            # main output. Log and continue.
            print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                  f"Extension B raised {type(e).__name__}: {e}. "
                  f"Main pipeline output preserved.{utils.Colors.ENDC}")

    def infer_proactive_trigger_candidates(self) -> None:
        """Step 28: catalog moments where the agent could legitimately
        initiate contact, scored by an LLM against the JITAI 6-component
        framework (Nahum-Shani et al., 2018) and Horvitz mixed-initiative
        principles (CHI 1999).

        Three Phase-1 trigger types:
          - T1.A `unfulfilled_stated_need`  — chatbot question N days unresolved.
          - T3.A `close_friend_update`      — close friend DM with no reply.
          - T4.A `sensitive_event_silence`  — restraint window during synthetic
                                              sensitive_life_event.

        Runs AFTER Extension B (Step 27) so `friends[]` is populated. Reads
        backend/{uid}/{profile,instagram,facebook,threads,chatbot}.json,
        runs deterministic candidate gathering (Stage 1), then LLM-judged
        eligibility scoring (Stage 2). Output is persisted in
        `profile.json.proactive_trigger_candidates`.

        Skipped gracefully when no LLM client is configured.
        """
        if self.llm_client is None:
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                      f"Skipping Step 28 (no llm client; subagent mode "
                      f"handles it inline).{utils.Colors.ENDC}")
            return

        user_dir = self._user_dir()
        profile_path = os.path.join(user_dir, "profile.json")
        if not os.path.exists(profile_path):
            if self.verbose:
                print(f"{utils.Colors.WARNING}[User {self.user_id}] "
                      f"Step 28: profile.json missing at {profile_path} — skipping.{utils.Colors.ENDC}")
            return

        with open(profile_path, "r") as f:
            profile = json.load(f)

        # Load app events once. DMs and friend posts live under app jsons;
        # chatbot conversations under chatbot.json.
        app_events: dict[str, list[dict]] = {}
        for app in ("instagram", "facebook", "threads", "chatbot"):
            path = os.path.join(user_dir, f"{app}.json")
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        app_events[app] = json.load(f)
                except (ValueError, OSError):
                    app_events[app] = []
            else:
                app_events[app] = []

        sensitive_periods = self._gather_sensitive_event_periods(profile)
        candidates_by_type = {
            "unfulfilled_stated_need": self._gather_unfulfilled_stated_needs(
                app_events.get("chatbot", []), app_events,
            ),
            "close_friend_update": self._gather_close_friend_dms(
                app_events, profile,
            ),
            "sensitive_event_silence": self._gather_sensitive_event_moments(
                sensitive_periods,
            ),
        }
        total = sum(len(v) for v in candidates_by_type.values())
        if self.verbose:
            print(f"[User {self.user_id}] Step 28 Stage 1: gathered "
                  f"{total} candidates "
                  f"({ {k: len(v) for k, v in candidates_by_type.items()} }).")

        if total == 0:
            profile["proactive_trigger_candidates"] = candidates_by_type
            with open(profile_path, "w") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
            return

        # Stage 2 — LLM-judged eligibility per candidate.
        user_state_base = self._build_proactive_user_state_base(profile)
        eligible_by_type: dict[str, list[dict]] = {}
        for trigger_type, cands in candidates_by_type.items():
            accepted: list[dict] = []
            for c in cands:
                user_state = dict(user_state_base)
                user_state["sensitive_event_active"] = self._is_in_sensitive_window(
                    c["t_test"], sensitive_periods,
                )
                try:
                    prompt = prompts.infer_proactive_trigger_prompt(user_state, c)
                    resp = self.llm_client.query_llm(
                        prompt, verbose=False, temperature=0.0,
                    )
                    card = utils.extract_json_from_response(resp) or {}
                except Exception as exc:
                    if self.verbose:
                        print(f"  ! Step 28 {trigger_type} candidate failed: "
                              f"{type(exc).__name__}: {exc}")
                    continue
                c["jitai_card"] = card
                if self._proactive_candidate_passes(trigger_type, card,
                                                    user_state["sensitive_event_active"]):
                    accepted.append(c)
            eligible_by_type[trigger_type] = accepted

        profile["proactive_trigger_candidates"] = eligible_by_type
        with open(profile_path, "w") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

        if self.verbose:
            kept = {k: len(v) for k, v in eligible_by_type.items()}
            print(f"[User {self.user_id}] Step 28 Stage 2: kept {sum(kept.values())}/{total} "
                  f"({kept}).")

    # --- Step 28 helpers ---

    _PROACTIVE_TRIGGER_LAGS = (1 * 86400, 3 * 86400, 7 * 86400)
    # Was 14d — too permissive for heavy hashtag-users where almost any
    # downstream event ends up "resolving" the question. 3d narrows to
    # genuinely-prompt follow-ups.
    _PROACTIVE_RESOLUTION_WINDOW = 3 * 86400
    # A single shared hashtag is too noisy when both events come from broad
    # clusters. Require ≥ 2 shared tags between the chatbot question and
    # the downstream event before counting as "resolved".
    _PROACTIVE_RESOLUTION_MIN_HASHTAG_OVERLAP = 2
    # Was 24h — close-friend cadence is typically well under 24h, so the
    # window masked genuinely-stalled threads. 6h captures threads where
    # the user actually let a close-friend message sit.
    _PROACTIVE_DM_REPLY_WINDOW = 6 * 3600
    _PROACTIVE_MAX_CANDIDATES_PER_TYPE = 12
    _PROACTIVE_CHATBOT_CLOSURE_ACTIONS = frozenset(
        ("asked_to_change_topic", "corrected_assumption")
    )

    def _gather_sensitive_event_periods(self, profile: dict) -> list[tuple[int, int]]:
        """Return list of (start_ts, end_ts) for active sensitive_life_event
        windows. The "active window" is the first ~14 days from
        `first_seen_ts` (Horvitz cost-benefit favors silence in that span).
        """
        periods: list[tuple[int, int]] = []
        for hp in (profile.get("hidden_personas") or []):
            if hp.get("type") != "sensitive_life_event":
                continue
            first = hp.get("first_seen_ts")
            if first is None:
                continue
            try:
                start = int(first)
            except (TypeError, ValueError):
                continue
            end = start + 14 * 86400
            periods.append((start, end))
        return periods

    def _is_in_sensitive_window(
        self,
        ts: int,
        periods: list[tuple[int, int]],
    ) -> bool:
        return any(start <= ts <= end for start, end in periods)

    def _gather_unfulfilled_stated_needs(
        self,
        chatbot_events: list[dict],
        all_app_events: dict[str, list[dict]],
    ) -> list[dict]:
        """T1.A — chatbot questions whose hashtags weren't covered by any
        subsequent event within `_PROACTIVE_RESOLUTION_WINDOW` days, AND
        whose conversation didn't end with an explicit closure action.
        """
        candidates: list[dict] = []
        # Pre-compute (ts, tag_set) per event across all apps, sorted by ts.
        # We need per-event tag-sets (not the old hashtag→ts inverted map) so
        # the resolution check can require ≥ N shared tags between the
        # chatbot question and a single subsequent event.
        all_events_flat: list[tuple[int, set[str]]] = []
        for app, events in all_app_events.items():
            for ev2 in events:
                ts = int(ev2.get("source_timestamp") or 0)
                tag_set = {h.lower() for h in (ev2.get("source_hashtags") or []) if h}
                if ts and tag_set:
                    all_events_flat.append((ts, tag_set))
        all_events_flat.sort(key=lambda x: x[0])

        for ev in chatbot_events:
            if ev.get("is_dm") or ev.get("is_ad"):
                continue
            convo = ev.get("conversation") or []
            if not convo:
                continue
            # Skip events whose interaction action is a closure / non-personalize.
            action = (ev.get("interaction_format") or {}).get("action", "")
            if action in self._PROACTIVE_CHATBOT_CLOSURE_ACTIONS:
                continue
            # Find the FIRST user turn (the standalone question this event represents).
            first_user_msg = None
            for m in convo:
                if m.get("role") == "user" and (m.get("content") or "").strip():
                    first_user_msg = m.get("content", "").strip()
                    break
            if not first_user_msg or len(first_user_msg) < 8:
                continue
            tags = [h.lower() for h in (ev.get("source_hashtags") or [])]
            if not tags:
                continue
            tag_set = set(tags)
            q_ts = int(ev.get("source_timestamp") or 0)
            # Resolved if SOME subsequent event in the window shares ≥ N tags
            # with the question. The N≥2 floor stops broad-cluster noise from
            # marking nearly every question "resolved".
            resolved = False
            cutoff = q_ts + self._PROACTIVE_RESOLUTION_WINDOW
            min_overlap = self._PROACTIVE_RESOLUTION_MIN_HASHTAG_OVERLAP
            for ts, ev_tags in all_events_flat:
                if ts <= q_ts:
                    continue
                if ts > cutoff:
                    break  # sorted ascending — no further events in window
                if len(tag_set & ev_tags) >= min_overlap:
                    resolved = True
                    break
            if resolved:
                continue
            # Emit one candidate per lag tier (1d, 3d, 7d).
            for lag in self._PROACTIVE_TRIGGER_LAGS:
                t_test = q_ts + lag
                candidates.append({
                    "trigger_type": "unfulfilled_stated_need",
                    "tier": "T1.A",
                    "t_test": t_test,
                    "t_test_iso": _unix_to_iso(t_test),
                    "lag_days": lag // 86400,
                    "signal_evidence": {
                        "chatbot_event_id": ev.get("source_object_id"),
                        "user_question": first_user_msg[:280],
                        "asked_at_ts": q_ts,
                        "asked_at_iso": ev.get("formatted_timestamp", ""),
                        "question_hashtags": tags,
                    },
                })
        # Cap to a manageable number; prefer the most recent unresolved questions
        # since they are most actionable.
        candidates.sort(key=lambda c: c["signal_evidence"]["asked_at_ts"], reverse=True)
        return candidates[: self._PROACTIVE_MAX_CANDIDATES_PER_TYPE]

    def _gather_close_friend_dms(
        self,
        all_app_events: dict[str, list[dict]],
        profile: dict,
    ) -> list[dict]:
        """T3.A — incoming DM from a close friend with no reply within
        `_PROACTIVE_DM_REPLY_WINDOW` (24h). Uses friend graph from Extension B.
        """
        friends = {
            f.get("friend_id"): f
            for f in (profile.get("friends") or [])
            if f.get("friend_id") and f.get("relationship_depth") == "close"
        }
        if not friends:
            return []

        candidates: list[dict] = []
        # Group DMs by thread_id to detect replies.
        for app, events in all_app_events.items():
            if app == "chatbot":
                continue
            # Build per-thread message lists.
            by_thread: dict[str, list[dict]] = {}
            for ev in events:
                if not ev.get("is_dm"):
                    continue
                tid = ev.get("thread_id") or ev.get("source_object_id")
                if tid:
                    by_thread.setdefault(str(tid), []).append(ev)
            for tid, ev_list in by_thread.items():
                ev_list.sort(key=lambda e: int(e.get("source_timestamp") or 0))
                for ev in ev_list:
                    if ev.get("author_id") == "self":
                        continue
                    author = ev.get("author_id")
                    friend = friends.get(author)
                    if not friend:
                        continue
                    incoming_ts = int(ev.get("source_timestamp") or 0)
                    # Replied within window?
                    replied = any(
                        e.get("author_id") == "self"
                        and incoming_ts < int(e.get("source_timestamp") or 0)
                        <= incoming_ts + self._PROACTIVE_DM_REPLY_WINDOW
                        for e in ev_list
                    )
                    if replied:
                        continue
                    # Extract a snippet of the friend's last message.
                    msgs = ev.get("messages") or []
                    last_friend_msg = ""
                    for m in reversed(msgs):
                        sender = m.get("sender") or m.get("author_id") or m.get("role")
                        if sender and sender != "self":
                            last_friend_msg = (m.get("text") or m.get("content") or "")[:280]
                            break
                    t_test = incoming_ts + 3600  # one hour later
                    candidates.append({
                        "trigger_type": "close_friend_update",
                        "tier": "T3.A",
                        "t_test": t_test,
                        "t_test_iso": _unix_to_iso(t_test),
                        "signal_evidence": {
                            "app": app,
                            "thread_id": tid,
                            "friend_id": author,
                            "friend_display_name": friend.get("display_name", ""),
                            "friend_relationship_depth": friend.get("relationship_depth", ""),
                            "friend_shared_interests": friend.get("shared_interests", []),
                            "incoming_message_excerpt": last_friend_msg,
                            "incoming_at_ts": incoming_ts,
                            "incoming_at_iso": ev.get("formatted_timestamp", ""),
                            "thread_hashtags": ev.get("source_hashtags", []),
                        },
                    })
        candidates.sort(key=lambda c: c["signal_evidence"]["incoming_at_ts"], reverse=True)
        return candidates[: self._PROACTIVE_MAX_CANDIDATES_PER_TYPE]

    def _gather_sensitive_event_moments(
        self,
        sensitive_periods: list[tuple[int, int]],
    ) -> list[dict]:
        """T4.A — restraint candidates: 3-5 sample timestamps inside each
        sensitive_life_event window.
        """
        candidates: list[dict] = []
        for start, end in sensitive_periods:
            span = max(end - start, 1)
            n_samples = 4
            for i in range(n_samples):
                t_test = start + (i + 1) * span // (n_samples + 1)
                candidates.append({
                    "trigger_type": "sensitive_event_silence",
                    "tier": "T4.A",
                    "t_test": t_test,
                    "t_test_iso": _unix_to_iso(t_test),
                    "signal_evidence": {
                        "window_start_ts": start,
                        "window_end_ts": end,
                        "days_into_window": (t_test - start) // 86400,
                    },
                })
        return candidates[: self._PROACTIVE_MAX_CANDIDATES_PER_TYPE]

    def _build_proactive_user_state_base(self, profile: dict) -> dict:
        """Compact snapshot the trigger judge sees alongside each candidate.
        Excludes the `sensitive_event_active` flag (per-candidate, set later).
        """
        hps = profile.get("hidden_personas") or []
        hp_brief = "; ".join(
            f"[{h.get('type')}] {h.get('label', '')[:80]}"
            for h in hps[:6]
            if h.get("type") not in ("sensitive_life_event",)
        ) or "(none)"
        prefs = profile.get("preferences") or []
        # `preferences` is a flat list; prefer items with persona_item field.
        pref_strs: list[str] = []
        for p in prefs[:8]:
            if isinstance(p, dict):
                pi = p.get("persona_item") or p.get("text") or ""
            else:
                pi = str(p)
            if pi:
                pref_strs.append(pi[:120])
        top_prefs = "; ".join(pref_strs) or "(none)"
        friends = profile.get("friends") or []
        f_brief = "; ".join(
            f"{f.get('display_name','')} ({f.get('relationship_depth','')})"
            for f in friends[:5]
        ) or "(none)"
        # Recent chatbot questions: pull from the last 30 days of chatbot events.
        recent = "(see candidate evidence — judge per item)"
        return {
            "name": profile.get("name", "(user)"),
            "hidden_persona_brief": hp_brief,
            "top_preferences_brief": top_prefs,
            "recent_chatbot_questions_brief": recent,
            "friends_brief": f_brief,
        }

    def _proactive_candidate_passes(
        self,
        trigger_type: str,
        card: dict,
        sensitive_active: bool,
    ) -> bool:
        """Apply the eligibility-keep rule per trigger type.

        Proactive triggers (T1.A, T3.A): keep if `eligibility_score >= 2 AND
        subtlety_check_pass AND recommended_action_class != "stay_silent"`.

        Restraint trigger (T4.A): keep if `eligibility_score == 0 AND
        recommended_action_class == "stay_silent"`. Sensitive-window override:
        if the moment is inside an active sensitive window, the LLM should
        have produced score=0; if it did not, drop the candidate as
        misjudged rather than emitting it incorrectly.
        """
        try:
            score = int(card.get("eligibility_score", -1))
        except (TypeError, ValueError):
            return False
        action_class = card.get("recommended_action_class") or ""
        subtlety_ok = bool(card.get("subtlety_check_pass", False))
        if trigger_type == "sensitive_event_silence":
            return score == 0 and action_class == "stay_silent"
        # Proactive types
        if sensitive_active:
            # Hard restraint window override — never emit a proactive instance here.
            return False
        return score >= 2 and subtlety_ok and action_class != "stay_silent"

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
