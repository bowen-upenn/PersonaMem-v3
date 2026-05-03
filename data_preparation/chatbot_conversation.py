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

try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback when tqdm is not installed — silent no-op wrapper.
    def tqdm(iterable=None, **kwargs):  # type: ignore[misc]
        if iterable is not None:
            return iterable
        class _Stub:
            def update(self, _n=1): pass
            def close(self): pass
        return _Stub()


# ---------------------------------------------------------------------------
# Conversation-type catalog
# ---------------------------------------------------------------------------

CHATBOT_CONVERSATION_TYPES: dict[str, dict] = {
    "writing_help": {
        "description": "User pastes a draft (email, caption, message, cover letter, post) and asks the chatbot "
                       "to improve the language, fix grammar, or adjust tone. The preference is embedded INSIDE "
                       "the user-provided draft text — e.g., the email body mentions a hobby, the caption "
                       "references a lifestyle detail. The user's explicit request is about writing quality, "
                       "NOT about the preference itself. The chatbot should help improve the text without "
                       "calling out the preference.",
        "compatible_contexts": [
            "professional emails",
            "personal emails",
            "composing chat messages",
            "composing social media posts",
        ],
        "weight": 25.0,
    },
    "knowledge_query": {
        "description": "User asks a specific factual, how-to, or nuanced question that reveals hidden curiosity. "
                       "The preference is inferred from WHAT the user chooses to ask about and the level of "
                       "domain-specific detail in their question — not from any direct statement. For example, "
                       "a user who enjoys Korean cooking might ask 'What's the difference between gochugaru "
                       "and gochujang in terms of fermentation?' without ever saying they like Korean food.",
        "compatible_contexts": [
            "knowledge exploration",
            "medical consultations",
        ],
        "weight": 30.0,
    },
    "therapy_reflection": {
        "description": "User discusses personal concerns, vents, or seeks life advice. The preference is NOT "
                       "the topic of concern — it surfaces as incidental context while the user talks about "
                       "something else. For example, a user who values fitness might say 'I've been stressed "
                       "about work deadlines and it's cutting into my morning runs' — the concern is stress, "
                       "the preference (running/fitness) is mentioned naturally as background.",
        "compatible_contexts": [
            "therapy and reflection",
        ],
        "weight": 20.0,
    },
    "troubleshooting": {
        "description": "User describes a practical problem and asks for a solution. The preference is embedded "
                       "in the problem context, NOT as the problem itself. For example, a user into home "
                       "gardening might ask 'My raised bed drainage isn't working after the last rain — what "
                       "could be wrong?' The preference (gardening) is the backdrop, the troubleshooting "
                       "request is the explicit task.",
        "compatible_contexts": [
            "knowledge exploration",
            "medical consultations",
        ],
        "weight": 10.0,
    },
    "casual_chat": {
        "description": "User asks the chatbot to help compose a chat message or social reply to a friend, "
                       "family member, or colleague. The preference is embedded in the content of the message "
                       "being composed — e.g., 'Help me reply to my friend who asked what I did this weekend' "
                       "and the user's draft mentions an activity revealing the preference.",
        "compatible_contexts": [
            "composing chat messages",
            "knowledge exploration",
            "therapy and reflection",
        ],
        "weight": 5.0,
    },
    "translation": {
        "description": "User provides text in another language and asks the chatbot to translate or rephrase it. "
                       "The preference is embedded in the SOURCE TEXT being translated — e.g., a recipe, a forum "
                       "post, a product review, an article excerpt — whose subject matter reveals the preference. "
                       "The user's request is about translation accuracy, not the preference.",
        "compatible_contexts": [
            "multilingual translation",
        ],
        "weight": 10.0,
    },
    "health_consultation": {
        "description": "User asks a health or medical question. The preference surfaces through the specific "
                       "lifestyle details, symptoms, or health context the user provides — e.g., a user who "
                       "does rock climbing might mention 'I've had this recurring pain in my forearm after "
                       "long sessions at the crag.' The consultation is about the symptom, not the preference.",
        "compatible_contexts": [
            "medical consultations",
        ],
        "weight": 15.0,
    },
}

# Fraction of ALL chatbot conversations (any polarity) that get either an
# ask-to-forget or a don't-personalize variant. Sampling is 50/50 between
# the two sub-variants. Applies regardless of interaction_type.
ASK_TO_FORGET_FRACTION = 0.20

# Fraction of explicit_negative chatbot samples that — if they did NOT get
# an ask_to_forget / do_not_personalize variant above — instead get the
# "wrong assumption → correction" treatment. Applied only to negatives.
CORRECTION_FRACTION_NEGATIVE = 0.50


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
    """Pick an even number of turns (2-8) uniformly at random.

    Even turn count guarantees every user message gets a chatbot reply.
    Negative interactions skew shorter (2-6).
    """
    if "negative" in interaction_type:
        return rng.choice([2, 4, 6])
    return rng.choice([2, 4, 6, 8])


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


def pick_conversation_variant(interaction_type: str, rng: random.Random) -> str | None:
    """Decide whether a chatbot conversation gets a special variant.

    Returns one of:
      - "ask_to_forget"       (user reveals a preference then asks assistant to forget it)
      - "do_not_personalize"  (user reveals a preference then asks not to be personalized on it)
      - "correction"          (explicit_negative only — assistant wrong assumption, user corrects)
      - None                   (standard multi-turn conversation)

    With probability ASK_TO_FORGET_FRACTION (20%), pick 50/50 between
    ask_to_forget and do_not_personalize — applied to ALL polarities.
    Otherwise, for explicit_negative only, with probability
    CORRECTION_FRACTION_NEGATIVE (50% of the remaining 80%), pick "correction".
    """
    if rng.random() < ASK_TO_FORGET_FRACTION:
        return rng.choice(["ask_to_forget", "do_not_personalize"])
    if interaction_type == "explicit_negative" and rng.random() < CORRECTION_FRACTION_NEGATIVE:
        return "correction"
    return None


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
    user_voice: dict | None = None,
) -> list[dict]:
    """Generate multi-turn conversations for chatbot event records.

    Each record represents one chatbot event (source row) with a
    ``preferences`` list of dicts (each having ``persona_item``,
    ``category``, and optionally ``interaction_type``). The generated
    conversation weaves in ALL preferences for that event.

    Uses ThreadPoolExecutor with ``max_workers`` parallel LLM calls.
    Deterministic RNG decisions are pre-computed sequentially, then LLM
    calls are fanned out in parallel.

    Args:
        chatbot_records: list of event dicts with ``preferences``,
            ``source_interaction_type``, and ``source_object_id``.
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
        preferences = rec.get("preferences", [])
        if not preferences:
            continue

        rec["conversation"] = None
        rec["conversation_type"] = None
        rec["ask_to_forget"] = False

        conv_type = select_conversation_type(user_contexts, rng)
        variant: str | None = pick_conversation_variant(interaction_type, rng)

        # Pick the "primary" preference that the variant acts on.
        # For ask_to_forget, prefer a self-referencing preference; fall back
        # to the first preference if none found (no fallback to correction
        # since do_not_personalize works for any preference kind).
        primary = preferences[0]
        if variant == "ask_to_forget":
            self_pref = next(
                (p for p in preferences if is_self_preference(p.get("persona_item", ""))),
                None,
            )
            if self_pref is not None:
                primary = self_pref

        if variant in ("ask_to_forget", "do_not_personalize"):
            additional = [p for p in preferences if p.get("persona_item") != primary.get("persona_item")]
            if variant == "ask_to_forget":
                prompt = prompts.generate_ask_to_forget_conversation_prompt(
                    persona_item=primary.get("persona_item", ""),
                    category=primary.get("category", ""),
                    user_profile=user_profile, chatbot_persona=chatbot_persona,
                    additional_preferences=additional if additional else None,
                    user_voice=user_voice,
                )
            else:
                prompt = prompts.generate_do_not_personalize_conversation_prompt(
                    persona_item=primary.get("persona_item", ""),
                    category=primary.get("category", ""),
                    user_profile=user_profile, chatbot_persona=chatbot_persona,
                    additional_preferences=additional if additional else None,
                    user_voice=user_voice,
                )
        elif variant == "correction":
            additional = [p for p in preferences if p.get("persona_item") != primary.get("persona_item")]
            prompt = prompts.generate_correction_conversation_prompt(
                persona_item=primary.get("persona_item", ""),
                category=primary.get("category", ""),
                user_profile=user_profile, chatbot_persona=chatbot_persona,
                additional_preferences=additional if additional else None,
                user_voice=user_voice,
            )
        else:
            # Standard conversation — scale turn count based on number of preferences,
            # clamped to the [2, 8] even-turn range.
            n_prefs = len(preferences)
            base_turns = select_num_turns(interaction_type, rng)
            num_turns = min(max(base_turns, min(n_prefs * 2, 8)), 8)
            if num_turns % 2 != 0:
                num_turns += 1

            # Build preference list with per-preference interaction types
            prefs_for_prompt = []
            for p in preferences:
                prefs_for_prompt.append({
                    "persona_item": p.get("persona_item", ""),
                    "category": p.get("category", ""),
                    "interaction_type": p.get("interaction_type", interaction_type),
                })

            prompt = prompts.generate_chatbot_conversation_prompt(
                preferences=prefs_for_prompt,
                conversation_type=conv_type,
                conversation_type_description=CHATBOT_CONVERSATION_TYPES[conv_type]["description"],
                user_profile=user_profile, chatbot_persona=chatbot_persona,
                interaction_type=interaction_type, num_turns=num_turns,
                user_voice=user_voice,
            )

        work_items.append((i, prompt, conv_type, variant))

    # --- Phase 2: Parallel LLM calls (with up to 4 retries) ---
    MAX_RETRIES = 4

    def _validate_conversation(response: str | None, n_prefs: int = 0) -> list[dict] | None:
        """Parse and validate a conversation response. Returns turns or None.

        Sanitizes per-turn `embeds_pref_idx` (1-based pref indices on user
        turns). Tolerates missing field for legacy back-compat — but malformed
        entries (non-list, non-int, out-of-range indices) get stripped, not
        rejected. The downstream extractor in evaluation/build_benchmark.py
        falls back to the legacy interaction_format.user_message path if
        no user turn carries an `embeds_pref_idx`.
        """
        if not response:
            return None
        parsed = utils.extract_json_from_response(response)
        if not isinstance(parsed, list) or len(parsed) < 2:
            return None
        for turn in parsed:
            if not isinstance(turn, dict) or "role" not in turn or "content" not in turn:
                return None
            if turn["role"] not in ("user", "assistant"):
                return None
            # Sanitize embeds_pref_idx (only meaningful on user turns).
            if turn["role"] == "user":
                raw = turn.get("embeds_pref_idx")
                if raw is not None:
                    if not isinstance(raw, list):
                        turn.pop("embeds_pref_idx", None)
                    else:
                        clean = []
                        for v in raw:
                            try:
                                idx = int(v)
                            except (TypeError, ValueError):
                                continue
                            # 1-based; if n_prefs known, clamp to range
                            if n_prefs > 0 and not (1 <= idx <= n_prefs):
                                continue
                            if idx not in clean:
                                clean.append(idx)
                        if clean:
                            turn["embeds_pref_idx"] = clean
                        else:
                            turn.pop("embeds_pref_idx", None)
            else:
                # Assistant turns should not carry pref indices; strip if present.
                turn.pop("embeds_pref_idx", None)
        return parsed

    def _call_llm_with_retry(item):
        idx, prompt, conv_type, variant = item
        for attempt in range(1 + MAX_RETRIES):
            try:
                response = llm_query_fn(prompt)
            except Exception:
                continue
            parsed = _validate_conversation(response)
            if parsed is not None:
                return idx, parsed, conv_type, variant
        return idx, None, conv_type, variant

    failed_count = 0
    pbar = tqdm(
        total=len(work_items),
        desc="Step 18: Chatbot conversations",
        unit="conv",
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_call_llm_with_retry, item): item for item in work_items}
        for future in as_completed(futures):
            pbar.update(1)
            try:
                idx, parsed, conv_type, variant = future.result()
            except Exception:
                failed_count += 1
                continue

            if parsed is None:
                failed_count += 1
                continue

            rec = chatbot_records[idx]
            rec["conversation"] = parsed
            rec["conversation_type"] = conv_type
            # Shared flag: both ask_to_forget and do_not_personalize set this True.
            rec["ask_to_forget"] = variant in ("ask_to_forget", "do_not_personalize")

            if variant == "ask_to_forget":
                rec["interaction_format"]["action"] = "asked_to_forget"
                rec["interaction_format"]["action_label"] = (
                    "Asked the assistant to forget a specific preference"
                )
                rec["interaction_format"]["user_message"] = parsed[-2]["content"] if len(parsed) >= 2 else None
            elif variant == "do_not_personalize":
                rec["interaction_format"]["action"] = "asked_not_to_personalize"
                rec["interaction_format"]["action_label"] = (
                    "Asked the assistant not to personalize recommendations around a preference"
                )
                rec["interaction_format"]["user_message"] = parsed[-2]["content"] if len(parsed) >= 2 else None
            elif variant == "correction":
                rec["interaction_format"]["action"] = "corrected_assumption"
                rec["interaction_format"]["action_label"] = (
                    "Corrected the assistant's wrong assumption about a preference"
                )
                rec["interaction_format"]["user_message"] = parsed[-2]["content"] if len(parsed) >= 2 else None
    pbar.close()

    if failed_count > 0:
        print(f"{utils.Colors.WARNING}[chatbot_conversation] "
              f"{failed_count}/{len(work_items)} conversation generations failed "
              f"after {1 + MAX_RETRIES} attempts each.{utils.Colors.ENDC}")

    return chatbot_records
