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
    """Pick an even number of turns (2-10) uniformly at random.

    Even turn count guarantees every user message gets a chatbot reply.
    Negative interactions skew shorter (2-6).
    """
    if "negative" in interaction_type:
        return rng.choice([2, 4, 6])
    return rng.choice([2, 4, 6, 8, 10])


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
    max_workers: int = 20,
) -> list[dict]:
    """Generate multi-turn conversations for chatbot preference records.

    Uses ThreadPoolExecutor with ``max_workers`` parallel LLM calls.
    Deterministic RNG decisions are pre-computed sequentially, then LLM
    calls are fanned out in parallel.

    Args:
        chatbot_records: list of preference dicts already routed to Chatbot.
        user_profile: the user's profile dict (name, bio, career, etc.)
        chatbot_persona: the Chatbot AppPersona dict.
        llm_query_fn: callable that takes a prompt string and returns the
            LLM response text (or None on failure).
        user_seed: deterministic seed derived from user_id.
        max_workers: number of parallel LLM API calls.

    Returns:
        The same chatbot_records list, with ``conversation``,
        ``conversation_type``, and ``ask_to_forget`` added to each record.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rng = random.Random(user_seed * 1301 + 7)
    user_contexts = chatbot_persona.get("chatbot_contexts", [])

    # --- Phase 1: Sequential RNG decisions + prompt building ---
    # (Must be sequential to keep RNG deterministic)
    work_items: list[tuple[int, str, str, str | None]] = []
    # work_items: (record_index, prompt, conv_type, variant)

    for i, rec in enumerate(chatbot_records):
        interaction_type = rec.get("source_interaction_type", "implicit_positive")
        persona_item = rec.get("persona_item", "")
        category = rec.get("category", "")

        rec["conversation"] = None
        rec["conversation_type"] = None
        rec["ask_to_forget"] = False

        if not should_generate_multiturn(rng):
            continue

        conv_type = select_conversation_type(user_contexts, rng)
        variant: str | None = None

        if interaction_type == "explicit_negative":
            variant = pick_explicit_neg_variant(rng)
            if variant == "ask_to_forget" and not is_self_preference(persona_item):
                variant = "correction"

        if variant == "ask_to_forget":
            prompt = prompts.generate_ask_to_forget_conversation_prompt(
                persona_item=persona_item, category=category,
                user_profile=user_profile, chatbot_persona=chatbot_persona,
            )
        elif variant == "correction":
            prompt = prompts.generate_correction_conversation_prompt(
                persona_item=persona_item, category=category,
                user_profile=user_profile, chatbot_persona=chatbot_persona,
            )
        else:
            num_turns = select_num_turns(interaction_type, rng)
            prompt = prompts.generate_chatbot_conversation_prompt(
                persona_item=persona_item, category=category,
                conversation_type=conv_type,
                conversation_type_description=CHATBOT_CONVERSATION_TYPES[conv_type]["description"],
                user_profile=user_profile, chatbot_persona=chatbot_persona,
                interaction_type=interaction_type, num_turns=num_turns,
            )

        work_items.append((i, prompt, conv_type, variant))

    # --- Phase 2: Parallel LLM calls ---
    def _call_llm(item):
        idx, prompt, conv_type, variant = item
        response = llm_query_fn(prompt)
        return idx, response, conv_type, variant

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_call_llm, item): item for item in work_items}
        for future in as_completed(futures):
            try:
                idx, response, conv_type, variant = future.result()
            except Exception:
                continue

            if not response:
                continue

            parsed = utils.extract_json_from_response(response)
            if not isinstance(parsed, list):
                continue

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

            rec = chatbot_records[idx]
            rec["conversation"] = parsed
            rec["conversation_type"] = conv_type
            rec["ask_to_forget"] = (variant == "ask_to_forget")

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
