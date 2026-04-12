"""
Chatbot conversation generation module.

Generates multi-turn user-chatbot conversations for preferences routed to the
Chatbot app.  Each conversation implicitly embeds a user preference through
task-oriented dialogue (PersonaMem-v2 style), rather than having the user
directly state their preference.

Used by PersonaAgent after interaction-format sampling (step 8) to enrich
Chatbot records with realistic conversation histories.
"""

from __future__ import annotations

import random
from typing import Callable

from data_preparation import prompts, utils


# ---------------------------------------------------------------------------
# Conversation-type catalog
# ---------------------------------------------------------------------------

CHATBOT_CONVERSATION_TYPES: dict[str, dict] = {
    "writing_help": {
        "description": "User asks the chatbot to draft, edit, or improve written text (emails, captions, messages). "
                       "The preference is hidden in the content the user provides or the style they request.",
        "compatible_contexts": [
            "professional emails",
            "personal emails",
            "composing chat messages",
            "composing social media posts",
        ],
        "weight": 25.0,
    },
    "knowledge_query": {
        "description": "User asks a factual, how-to, or nuanced knowledge question. "
                       "The preference is revealed through the specificity of the question and domain details.",
        "compatible_contexts": [
            "knowledge exploration",
            "medical consultations",
        ],
        "weight": 30.0,
    },
    "therapy_reflection": {
        "description": "User seeks advice, vents, or engages in personal reflection. "
                       "The preference emerges through the personal context and feelings shared.",
        "compatible_contexts": [
            "therapy and reflection",
        ],
        "weight": 20.0,
    },
    "troubleshooting": {
        "description": "User describes a problem and seeks a practical solution. "
                       "The preference is concealed within the problem description.",
        "compatible_contexts": [
            "knowledge exploration",
            "medical consultations",
        ],
        "weight": 10.0,
    },
    "casual_chat": {
        "description": "Natural back-and-forth conversation. "
                       "The preference is woven into the flow of casual dialogue.",
        "compatible_contexts": [
            "composing chat messages",
            "knowledge exploration",
            "therapy and reflection",
        ],
        "weight": 5.0,
    },
    "translation": {
        "description": "User asks for translation or cross-language help. "
                       "The preference is embedded in the source material chosen.",
        "compatible_contexts": [
            "multilingual translation",
        ],
        "weight": 10.0,
    },
    "health_consultation": {
        "description": "User asks health or medical questions. "
                       "The preference is revealed through symptoms, conditions, or health concerns described.",
        "compatible_contexts": [
            "medical consultations",
        ],
        "weight": 15.0,
    },
}

# Probability of generating a full multi-turn conversation vs keeping the
# existing single-action format.
MULTITURN_PROBABILITY = 0.80

# Fraction of explicit_negative chatbot samples that get ask-to-forget or
# correction treatment (combined).
EXPLICIT_NEG_SPECIAL_FRACTION = 0.70


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perturb_weights(base_weights: list[float], rng: random.Random,
                     noise_strength: float = 0.6) -> list[float]:
    """Lognormal perturbation — mirrors persona_agent._perturb_weights."""
    import math
    perturbed = []
    for w in base_weights:
        factor = math.exp(rng.gauss(0.0, noise_strength))
        perturbed.append(max(0.0, w * factor))
    total = sum(perturbed)
    if total <= 0:
        return list(base_weights)
    scale = sum(base_weights) / total
    return [w * scale for w in perturbed]


def select_conversation_type(
    user_chatbot_contexts: list[str],
    rng: random.Random,
) -> str:
    """Pick a conversation type compatible with the user's chatbot contexts."""
    eligible: list[tuple[str, float]] = []
    for ctype, spec in CHATBOT_CONVERSATION_TYPES.items():
        if any(ctx in spec["compatible_contexts"] for ctx in user_chatbot_contexts):
            eligible.append((ctype, spec["weight"]))

    if not eligible:
        # Fallback: knowledge_query is always reasonable
        return "knowledge_query"

    names = [e[0] for e in eligible]
    weights = _perturb_weights([e[1] for e in eligible], rng)
    return rng.choices(names, weights=weights, k=1)[0]


def select_num_turns(interaction_type: str, rng: random.Random) -> int:
    """Pick 2-4 turns based on the interaction type.

    implicit_negative gets fewer turns (user disengages quickly).
    """
    if "negative" in interaction_type:
        return rng.choices([2, 3], weights=[0.7, 0.3], k=1)[0]
    return rng.choices([2, 3, 4], weights=[0.3, 0.45, 0.25], k=1)[0]


# Keywords indicating self-referencing preferences (eligible for ask-to-forget)
_SELF_PREF_PATTERNS = [
    "may have", "works in", "is a ", "has a ", "likely ",
    "currently", "identifies as", "partner", "lives in",
    "raising", "enrolled", "employed", "married", "single",
    "parent", "caregiver", "incarcerated", "nursing",
    "connected to", "navigating",
]


def is_self_preference(persona_item: str) -> bool:
    """Heuristic: does this preference describe a personal fact about the user
    (eligible for ask-to-forget) vs a content-consumption pattern?

    Content-consumption patterns ('Enjoys X content', 'Follows X creators')
    are less likely to trigger a 'don't remember that' request.
    """
    lower = persona_item.lower()
    # Content consumption patterns — skip these
    consumption_keywords = [
        "enjoys", "follows", "watches", "engages with",
        "appreciates", "looks for", "seeks", "keeps up with",
        "gravitates toward", "interested in", "invested in",
        "supports", "responds to", "finds", "active across",
        "spends time",
    ]
    if any(lower.startswith(kw) for kw in consumption_keywords):
        return False
    return any(pat in lower for pat in _SELF_PREF_PATTERNS)


def should_generate_multiturn(rng: random.Random) -> bool:
    """~80% chance of full multi-turn conversation."""
    return rng.random() < MULTITURN_PROBABILITY


def pick_explicit_neg_variant(rng: random.Random) -> str | None:
    """For explicit_negative records, decide if this gets a special treatment.

    Returns 'ask_to_forget', 'correction', or None (standard format).
    ~70% get special treatment, split roughly evenly.
    """
    if rng.random() >= EXPLICIT_NEG_SPECIAL_FRACTION:
        return None
    return rng.choice(["ask_to_forget", "correction"])


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_chatbot_conversations(
    chatbot_records: list[dict],
    user_profile: dict,
    chatbot_persona: dict,
    llm_query_fn: Callable[[str], str | None],
    user_seed: int,
) -> list[dict]:
    """Generate multi-turn conversations for chatbot preference records.

    Args:
        chatbot_records: list of preference dicts already routed to Chatbot
            (loaded from chatbot.json or built in-memory).
        user_profile: the user's profile dict (name, bio, career, etc.)
        chatbot_persona: the Chatbot AppPersona dict (use_purposes,
            style_description, chatbot_contexts, etc.)
        llm_query_fn: callable that takes a prompt string and returns the
            LLM response text (or None on failure).  This is typically
            PersonaAgent._query_llm_with_retry or a Claude Code subagent
            wrapper.
        user_seed: deterministic seed derived from user_id.

    Returns:
        The same chatbot_records list, with ``conversation``,
        ``conversation_type``, and ``ask_to_forget`` added to each record.
    """
    rng = random.Random(user_seed * 1301 + 7)
    user_contexts = chatbot_persona.get("chatbot_contexts", [])

    for rec in chatbot_records:
        interaction_type = rec.get("source_interaction_type", "implicit_positive")
        persona_item = rec.get("persona_item", "")
        category = rec.get("category", "")

        # Default: no conversation
        rec["conversation"] = None
        rec["conversation_type"] = None
        rec["ask_to_forget"] = False

        if not should_generate_multiturn(rng):
            # ~20%: keep existing single-action format, no conversation
            continue

        # --- Decide conversation variant ---
        conv_type = select_conversation_type(user_contexts, rng)
        variant: str | None = None  # "ask_to_forget", "correction", or None

        if interaction_type == "explicit_negative":
            variant = pick_explicit_neg_variant(rng)
            if variant == "ask_to_forget" and not is_self_preference(persona_item):
                # ask-to-forget only makes sense for self-referencing prefs;
                # fall back to correction instead
                variant = "correction"

        # --- Build prompt ---
        if variant == "ask_to_forget":
            prompt = prompts.generate_ask_to_forget_conversation_prompt(
                persona_item=persona_item,
                category=category,
                user_profile=user_profile,
                chatbot_persona=chatbot_persona,
            )
        elif variant == "correction":
            prompt = prompts.generate_correction_conversation_prompt(
                persona_item=persona_item,
                category=category,
                user_profile=user_profile,
                chatbot_persona=chatbot_persona,
            )
        else:
            num_turns = select_num_turns(interaction_type, rng)
            prompt = prompts.generate_chatbot_conversation_prompt(
                persona_item=persona_item,
                category=category,
                conversation_type=conv_type,
                conversation_type_description=CHATBOT_CONVERSATION_TYPES[conv_type]["description"],
                user_profile=user_profile,
                chatbot_persona=chatbot_persona,
                interaction_type=interaction_type,
                num_turns=num_turns,
            )

        # --- Call LLM ---
        response = llm_query_fn(prompt)
        if not response:
            continue

        parsed = utils.extract_json_from_response(response)
        if not isinstance(parsed, list):
            continue

        # Validate conversation structure
        valid = True
        for turn in parsed:
            if not isinstance(turn, dict) or "role" not in turn or "content" not in turn:
                valid = False
                break
            if turn["role"] not in ("user", "assistant"):
                valid = False
                break
        if not valid or len(parsed) < 2:
            continue

        # --- Attach to record ---
        rec["conversation"] = parsed
        rec["conversation_type"] = conv_type
        rec["ask_to_forget"] = (variant == "ask_to_forget")

        # Update interaction_format action for special variants
        if variant == "ask_to_forget":
            rec["interaction_format"]["action"] = "asked_to_forget"
            rec["interaction_format"]["action_label"] = (
                "Asked the assistant to forget a specific preference"
            )
            rec["interaction_format"]["user_message"] = parsed[-2]["content"] if len(parsed) >= 2 else None
        elif variant == "correction":
            rec["interaction_format"]["action"] = "corrected_assumption"
            rec["interaction_format"]["action_label"] = (
                "Corrected the assistant's wrong assumption about a preference"
            )
            rec["interaction_format"]["user_message"] = parsed[-2]["content"] if len(parsed) >= 2 else None

    return chatbot_records
