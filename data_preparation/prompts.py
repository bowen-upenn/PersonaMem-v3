"""
LLM prompt templates for persona inference pipeline.

All prompts are kept in this file, separate from business logic.
Each function returns a prompt string ready to send to the LLM.
"""

from __future__ import annotations

import json
from typing import List, Dict


def hashtag_to_persona_prompt(
    object_text: str,
    interaction_type: str,
    formatted_timestamp: str,
    hashtags: list[str],
    interaction_format: str = "",
    existing_categories: list[str] | None = None,
) -> str:
    """Build a prompt that asks the LLM to infer atomic persona traits from hashtags."""

    if interaction_format:
        parts = interaction_format.split(":", 1)
        platform = parts[0].strip()
        action = parts[1].strip() if len(parts) > 1 else interaction_format
        polarity = f"on {platform}, {action}"
    else:
        polarity = (
            "engaged positively with (liked, shared, or lingered on)"
            if "positive" in interaction_type
            else "scrolled past, dismissed, or disengaged from"
        )

    hashtag_list = ", ".join(hashtags) if hashtags else "(no hashtags found)"

    return f"""\
You are an expert behavioral analyst specializing in social media user profiling.

A user interacted with the following social media content:
- **Interaction type**: {interaction_type} — the user {polarity} this content.
- **Platform & action**: {interaction_format or "unknown"}
- **Timestamp**: {formatted_timestamp}
- **Full content text**: {object_text}
- **Extracted hashtags**: {hashtag_list}

## Your Task

Considering ALL the hashtags together as a whole and individually, infer **around 10** atomic persona traits or preferences about this user. An "atomic persona" is a single, specific, testable statement about the user's personality, interests, values, demographics, or lifestyle (e.g., "Interested in CrossFit", "Values family traditions", "Likely a parent of school-age children").

## Confidence Scoring (READ THIS FIRST)

This is a "{interaction_type}" interaction.
{"FOR POSITIVE INTERACTIONS: The user actively engaged with this content. Use the full 0.0-1.0 range. A near-certain, explicitly stated inference scores 0.80-1.0. A direct topic match scores 0.60-0.80. A reasonable deduction scores 0.40-0.60. A broader inference scores 0.15-0.40. A speculative, loosely connected inference scores 0.0-0.15. Not every inference from a positive interaction deserves a high score — be critical and spread your scores across the full range. Phrase preferences POSITIVELY (e.g., 'Enjoys X', 'Interested in X', 'Values X')." if "positive" in interaction_type else "FOR EXPLICIT NEGATIVE INTERACTIONS: The user actively disliked or dismissed this content. Use the full 0.0-1.0 range. A direct dislike of the core topic scores 0.55-0.75. A reasonable deduction scores 0.35-0.55. A broader inference scores 0.15-0.35. A speculative inference scores 0.0-0.15. CRITICAL: Phrase EVERY preference NEGATIVELY as what the user dislikes/avoids/rejects (e.g., 'Dislikes X', 'Avoids X', 'Not interested in X', 'Turned off by X'). NEVER phrase as 'Enjoys' or 'Likes' for negative interactions."}

Use precise, varied values with two decimal places. Each inference must get a distinct score.

## Rules

1. **Be exploratory**: Produce around 10 preferences total by considering both individual hashtags and the combined signal from all hashtags together. Quality over quantity.
2. **Be specific**: Each persona item must be concrete and testable, not vague (e.g., "Enjoys cooking Italian food at home" rather than "Likes food").
3. **Consider diverse dimensions**: Think about interests, values, demographics, lifestyle, profession, cultural background, media consumption habits, purchasing behavior, and social identity.
4. **Categorize each inference**: Assign a **specific topical category**.{(' REUSE one of these existing categories whenever possible: ' + ', '.join(existing_categories) + '. Only create a new category if none of the existing ones fit. Avoid creating categories that are near-synonyms of existing ones.') if existing_categories else ' Use specific topical categories (e.g., "cooking", "Christian faith", "NFL fandom", "romantic relationships", "fitness", "parenting").'} Do NOT use generic categories like "interests", "values", "personality", "lifestyle", or "demographics".
5. **Tag source hashtags**: For each persona, include ONLY the specific hashtag(s) that directly led to this inference — not the full list.

## Output Format

Respond with ONLY a JSON array. No explanation, no markdown outside the JSON fence.

```json
[
  {{"category": "cooking", "persona_item": "...", "confidence_score_init": 0.XX, "source_hashtags": ["#CookingWithAPurpose"]}},
  {{"category": "Christian faith", "persona_item": "...", "confidence_score_init": 0.XX, "source_hashtags": ["#ChristianLiving", "#LoveAndFaith"]}}
]
```"""


def summarize_and_cross_reference_prompt(atomic_personas: list[dict]) -> str:
    """Build a prompt that asks the LLM to cross-reference and score all atomic personas.

    NOTE: The input list has ALREADY been deduplicated by the caller. Each entry
    is a *canonical* persona — personas with identical text from multiple rows
    have been merged into one entry. `confidence_cross_referenced` reflects the
    count of distinct source rows that independently produced them AND passed
    the init-confidence threshold. The LLM must treat each entry as a
    unique preference.
    """

    personas_json = json.dumps(atomic_personas, indent=2)

    return f"""\
You are an expert at synthesizing behavioral signals into a coherent user profile.

Below is a list of persona traits/preferences for a single user. Each entry has a `persona_item` and `category`.

```json
{personas_json}
```

## Your Task — Find cross-persona relationships

1. **Cross-reference personas** against each other.
   - If two **different** personas are **similar** (reinforce each other — e.g. "Enjoys home cooking" and "Buys fresh produce weekly"), mark them as related with `"type": "similar"`.
   - If two **different** personas **contradict** each other (e.g. "Prefers vegan meals" and "Loves BBQ ribs"), mark them as related with `"type": "contradictory"`.

2. **Do NOT mark a persona as similar to itself.**

3. **ONLY return personas that have at least one relationship.** Skip personas with no relationships entirely — do NOT include them in the output. This keeps the output compact.

4. **For each persona, list all related personas** in the `related_personas` array as: `{{"persona_item": "...", "type": "similar"}}` or `{{"persona_item": "...", "type": "contradictory"}}`.

## Output Format

Respond with ONLY a JSON array. No explanation.

```json
[
  {{
    "persona_item": "...",
    "relationship_type": "similar" | "contradictory",
    "related_personas": [{{"persona_item": "...", "type": "similar"}}]
  }}
]
```"""


def temporal_contradiction_graph_prompt(contradictory_personas: list[dict]) -> str:
    """Build a prompt that asks the LLM to organize contradictions into a temporal graph."""

    personas_json = json.dumps(contradictory_personas, indent=2)

    return f"""\
You are an expert at modeling how user preferences and beliefs evolve over time.

Below is a list of persona traits/preferences that have been identified as **contradictory** with one another for the same user. Each entry includes the persona item, timestamps, and confidence scores.

```json
{personas_json}
```

## Your Task

1. **Group** related contradictions by topic or theme (e.g., "dietary preferences", "political views", "brand loyalty").
2. **For each group**, construct a **temporal timeline** showing how the user's preference changed over time. Order entries chronologically by their timestamps.
3. **Interpret** each transition — what might have caused the shift (life event, new information, seasonal change, etc.).
4. **Include ALL preference changes** — there is no limit on the number of transitions. Capture every shift, even subtle ones.
5. **Preserve all timestamps and confidence scores** from the original data.

## Output Format

Respond with ONLY a JSON array. No explanation.

```json
[
  {{
    "topic": "...",
    "timeline": [
      {{
        "persona_item": "...",
        "timestamp": 1234567890,
        "formatted_timestamp": "HH:MM, MM/DD/YYYY",
        "confidence_score_init": 0.XX,
        "confidence_cross_referenced": 0.XX
      }},
      {{
        "persona_item": "...",
        "timestamp": 1234567899,
        "formatted_timestamp": "HH:MM, MM/DD/YYYY",
        "confidence_score_init": 0.XX,
        "confidence_cross_referenced": 0.XX
      }}
    ],
    "interpretation": "..."
  }}
]
```"""


def generate_user_profile_prompt(
    personas: list[str],
    gender_orientation: str,
    race_ethnicity: str,
) -> str:
    """Build a prompt that generates a synthetic user profile from final personas."""

    personas_list = "\n".join(f"- {p}" for p in personas)

    return f"""\
You are creating a realistic synthetic user profile based on inferred persona traits.

## Assigned Demographics (pre-sampled — do not change these)
- **Gender & Sexual Orientation**: {gender_orientation}
- **Race/Ethnicity**: {race_ethnicity}

## Inferred Persona Traits
{personas_list}

## Your Task

Generate a synthetic user profile that is **consistent with some but not all** of the above personas. Rules:

1. **Name**: Choose a culturally appropriate first and last name for the assigned gender and race/ethnicity. Be diverse in naming — avoid the most common/default names.
2. **Career**: Pick a realistic career. It can relate to some personas but does NOT need to satisfy all of them. Surprising or unconventional career choices are welcome.
3. **Education**: Highest level of education and field of study. Be varied — not everyone has a college degree.
4. **Big Five personality**: Rate each dimension as "low", "medium", or "high". Let the personas inform this but do NOT mechanically map every persona to a trait. Some personality dimensions should be unrelated to the personas.
5. **Bio**: Write exactly 3-5 sentences. Paint a vivid picture of this person's daily life. Reference some personas naturally but leave others unmentioned. Include at least one detail that is surprising or not directly derivable from the personas.
6. **Diversity**: Actively avoid stereotypical combinations. A {race_ethnicity} person who is {gender_orientation} can have any career, any hobby, any personality. Do not default to the most "expected" profile — be creative and realistic.

## Output Format

Respond with ONLY a JSON object. No explanation.

```json
{{{{
  "name": "First Last",
  "career": "...",
  "education": "...",
  "big_five": {{{{
    "openness": "low" | "medium" | "high",
    "conscientiousness": "low" | "medium" | "high",
    "extraversion": "low" | "medium" | "high",
    "agreeableness": "low" | "medium" | "high",
    "neuroticism": "low" | "medium" | "high"
  }}}},
  "bio": "3-5 sentences..."
}}}}
```"""


def annotate_stereotype_prompt(
    personas: list[dict],
    gender: str,
    race_ethnicity: str,
) -> str:
    """Build a prompt that annotates each persona with a stereotype mark based on demographics only."""

    personas_json = json.dumps(personas, indent=2)

    return f"""\
You are an expert in social psychology and stereotype analysis.

## User Demographics (use ONLY these for stereotype assessment)
- **Gender & Sexual Orientation**: {gender}
- **Race/Ethnicity**: {race_ethnicity}

## Persona Traits to Annotate
```json
{personas_json}
```

## Your Task

For each persona trait, assign a `stereotype_mark` based **solely on the user's demographic information** (gender, sexual orientation, race/ethnicity). Do NOT consider career, education, or other profile details — only demographics.

- **"stereotypical"**: This preference/trait aligns with a common stereotype about people of this user's gender, sexual orientation, or race/ethnicity. For example, a Black user liking basketball, or a woman enjoying cooking, would be stereotypical associations.
- **"anti-stereotypical"**: This preference/trait actively contradicts or defies a common stereotype about this user's gender, sexual orientation, or race/ethnicity. For example, a male user being deeply into knitting, or an elderly person being into esports.
- **"neutral"**: This preference/trait has no meaningful stereotypical association with the user's demographics. Most preferences should be neutral — only mark stereotypical/anti-stereotypical when the association is widely recognized.

## Rules

1. **Be conservative**: When in doubt, mark as "neutral". Only use "stereotypical" or "anti-stereotypical" for widely recognized, well-documented associations.
2. **Consider intersectionality**: A trait might be stereotypical along one axis (gender) but neutral along another (race). Use your best judgment for the overall mark.
3. **Do not invent stereotypes**: Only flag associations that are genuinely common in public discourse. Obscure or debatable associations should be "neutral".
4. **Return every persona item** — do not skip or filter any.

## Output Format

Respond with ONLY a JSON array. No explanation.

```json
[
  {{{{"persona_item": "...", "category": "...", "stereotype_mark": "neutral" | "stereotypical" | "anti-stereotypical"}}}}
]
```"""


def test_inferrability_check_prompt(
    train_personas: list[dict],
    test_candidates: list[dict],
) -> str:
    """Build a prompt that asks the LLM to check whether each test candidate persona
    can be reasonably inferred from the train 80% set (the ground truth).

    Each persona dict includes: persona_item, category, confidence_score_init,
    confidence_cross_referenced, formatted_timestamp.
    """

    train_json = json.dumps(train_personas, indent=2)
    test_json = json.dumps(test_candidates, indent=2)

    return f"""\
You are evaluating whether a set of "test" user preferences can be reasonably inferred from the user's earlier history of preferences (the "train" set). This is a sanity check for building a high-fidelity evaluation dataset: we only want to keep test items that a thoughtful reader could plausibly predict from the user's established pattern.

## Train set (user's earlier high-confidence preferences — ground truth)

```json
{train_json}
```

## Test candidates (user's most recent high-confidence preferences)

```json
{test_json}
```

## Your Task

For **each** test candidate, decide whether it can be **reasonably inferred** from the train set. The test preference does NOT need to be explicitly stated in the train set — but the user's earlier pattern should plausibly support it. Examples:

- A test preference "Enjoys espresso-based drinks" is **inferrable** if the train set already shows a strong coffee/cafe pattern.
- A test preference "Follows competitive chess tournaments" is **NOT inferrable** if nothing in the train set touches chess, board games, or strategy hobbies.

## Rules

1. **Be conservative**. When in doubt, mark as `false` — we'd rather drop a borderline item than keep a noisy eval sample.
2. Consider **topical overlap**, **lifestyle coherence**, and **demographic/cultural consistency** as bridges from train → test.
3. Return **one entry per test candidate**, in the same order as the input.
4. One-sentence justification per entry.

## Output Format

Respond with ONLY a JSON array. No explanation outside the JSON fence.

```json
[
  {{"persona_item": "...", "inferrable": true, "reason": "..."}},
  {{"persona_item": "...", "inferrable": false, "reason": "..."}}
]
```"""


def distractor_selection_prompt(
    test_persona: dict,
    candidate_distractors: list[dict],
) -> str:
    """Build a prompt that asks the LLM to pick one distractor from a shortlist.

    The goal: choose the candidate that would feel most topically irrelevant and
    most annoying/inappropriate if surfaced as a personalization recommendation
    at the moment of the test preference. It's a hard-negative selection.

    test_persona and each candidate dict has: persona_item, category.
    """

    test_json = json.dumps(test_persona, indent=2)
    candidates_json = json.dumps(candidate_distractors, indent=2)

    return f"""\
You are building a hard-negative distractor for a personalization evaluation.

## Target test preference

```json
{test_json}
```

## Shortlist of candidate distractors

These are all known to be correct, high-confidence preferences of the same user — but they come from earlier in the user's history and may or may not be relevant to the target test preference above.

```json
{candidates_json}
```

## Your Task

Imagine a personalization feature is trying to surface something relevant to the user at the moment the target test preference is active (e.g., the user is in the mood or context described by the test preference). Out of the shortlist, pick the **one** candidate that would be:

1. **Topically irrelevant** to the target test preference — no meaningful overlap in domain, activity, or need.
2. **Most annoying or inappropriate** as a personalization recommendation in that moment — i.e., if the system suggested this candidate instead of something aligned with the test preference, it would feel like a jarring miss that undermines user trust in the personalization.

Among the shortlist, choose the single worst match along these two axes combined. Ties broken in favor of the one most likely to frustrate the user.

## Rules

1. Pick exactly **one** candidate from the shortlist — do not invent new items.
2. The chosen `persona_item` string must match one of the candidates exactly.
3. One-sentence justification explaining why it's the most jarring / least relevant of the options.

## Output Format

Respond with ONLY a JSON object. No explanation outside the JSON fence.

```json
{{"chosen_persona_item": "...", "reason": "..."}}
```"""



def generate_app_personas_prompt(
    profile: dict,
    top_personas: list[str],
    chatbot_contexts: list[str],
) -> str:
    """Build a prompt that generates per-app sub-personas for a user.

    Inputs:
      profile: dict with name, gender, race_ethnicity, career, education, big_five, bio
      top_personas: up to ~20 persona_item strings sampled from the user's strongest preferences
      chatbot_contexts: the full CHATBOT_CONTEXTS list from persona_agent.py
    """

    profile_json = json.dumps(profile, indent=2)
    personas_text = "\n".join(f"- {p}" for p in top_personas)
    chatbot_contexts_str = ", ".join(chatbot_contexts)

    return f"""\
You are designing four distinct "app sub-personas" for a single synthetic user. The user already has a base profile and a set of preferences inferred from their social media activity. Your job is to describe how THIS specific user presents themselves and engages differently on each of four apps: **Instagram, Facebook, Threads, and AI Chatbot**.

## Base profile

```json
{profile_json}
```

## Sample of the user's strongest inferred preferences

{personas_text}

## Your Task

Write four distinct `AppPersona` entries — one per app. Each should describe how the user uses that app specifically, including which audiences they interact with, what they use it for, and how their engagement style varies.

## Rules

1. **Be noticeably different across apps.** Real people compartmentalize their online presence. For example, the same user might:
   - Use **Facebook** mostly for family updates, marketplace, and event planning with older relatives
   - Use **Instagram** for lifestyle aesthetic, close friends, and inspiration browsing
   - Use **Threads** for news, snark, quick opinions, and following public figures
   - Use **AI Chatbot** for work tasks, knowledge exploration, and private reflection

2. **Allow some overlap.** Real users aren't perfectly compartmentalized. A couple of shared topics across apps is realistic. Do not force uniqueness.

3. **Ground everything in the base profile and preferences.** A conservative rural parent and a young urban creative should get *very* different app personas. Use the profile's career, age clues from education/bio, demographics, and Big Five personality as guiding signals.

4. **Audience types must be realistic**:
   - **Facebook**: usually `mixed` leaning toward family/longtime friends
   - **Instagram**: usually `mixed` (close friends + public creators followed)
   - **Threads**: usually `public`
   - **AI Chatbot**: always `private`

5. **Posting frequency** must be one of: `"daily"`, `"weekly"`, `"rarely"`, `"passive viewer only"`. Pick what's realistic for this user on this app — most users post rarely on most apps and are mainly consumers.

6. **Topical focus** is 3–5 broad domains (e.g., `"food and home cooking"`, `"local community news"`, `"parenting"`, `"crafts"`, `"fitness"`). These should be a subset of the domains the user actually shows interest in (from the sample above), not invented ones.

7. **Chatbot only**: populate `chatbot_contexts` with 2–3 items chosen from this exact list: {chatbot_contexts_str}. Pick the contexts that best match this user's profile (e.g., a student → `"knowledge exploration"`, `"composing chat messages"`; a therapist-curious user → `"therapy and reflection"`). Leave `chatbot_contexts` as an empty array for non-Chatbot apps.

8. **Use purposes** should be a list of 2–4 short phrases describing what the user gets out of this app (e.g., `["keep up with extended family", "marketplace deals", "event planning"]`).

9. **Friend zones** should be a list of 2–4 short phrases describing which social circles they interact with (e.g., `["close friends", "acquaintances", "extended family", "strangers / public"]`).

10. **Style description** is 2–3 sentences describing the tone, aesthetic, or cadence of the user on THIS app (e.g., `"Warm, family-centric, and nostalgic. Shares birthday photos and milestone announcements. Rarely posts opinions."`).

## Output Format

Respond with ONLY a JSON object. No explanation outside the JSON fence.

```json
{{
  "Instagram": {{
    "app_name": "Instagram",
    "use_purposes": ["..."],
    "friend_zones": ["..."],
    "audience_type": "private" | "public" | "mixed",
    "style_description": "...",
    "posting_frequency": "daily" | "weekly" | "rarely" | "passive viewer only",
    "topical_focus": ["..."],
    "chatbot_contexts": []
  }},
  "Facebook": {{ ... }},
  "Threads": {{ ... }},
  "Chatbot": {{
    "app_name": "Chatbot",
    "use_purposes": ["..."],
    "friend_zones": ["..."],
    "audience_type": "private",
    "style_description": "...",
    "posting_frequency": "...",
    "topical_focus": ["..."],
    "chatbot_contexts": ["..."]
  }}
}}
```"""


def assign_personas_to_apps_prompt(
    app_personas: dict,
    preferences: list[dict],
) -> str:
    """Build a prompt asking the LLM to route each preference to ONE primary app.

    Inputs:
      app_personas: the dict output of generate_app_personas_prompt
      preferences: list of {persona_item, category, confidence_score_init,
                            confidence_cross_referenced, source_interaction_type}
    """

    app_personas_json = json.dumps(app_personas, indent=2)
    preferences_json = json.dumps(preferences, indent=2)

    return f"""\
You are routing a user's individual preferences to the app where they most naturally belong, based on how this user uses each app.

## The user's per-app sub-personas

```json
{app_personas_json}
```

## Preferences to route

```json
{preferences_json}
```

## Your Task

For EACH preference in the list above, pick exactly **one primary app** (from "Instagram", "Facebook", "Threads", "Chatbot") where a real person with these sub-personas would most plausibly encounter and engage with that preference. The assignment should:

1. **Maintain topical consistency within each app.** If the user's Facebook persona is about family & marketplace, preferences about parenting, Costco deals, and birthday parties should mostly land on Facebook. Don't scatter topically-coherent preferences across random apps.

2. **Reflect the per-app persona's use_purposes and topical_focus.** Route a preference to the app whose declared purposes best cover it. E.g. if the Chatbot persona lists `"therapy and reflection"` and a preference is `"Values emotional vulnerability in close relationships"`, Chatbot is a natural home.

3. **Allow NATURAL variation, not randomness.** Two closely related preferences should almost always land on the same app. If one belongs on Instagram, its partner almost certainly does too. Do not split tightly-coupled preferences for variety.

4. **Prefer the app the user is more active on for that domain.** Use `posting_frequency` and `audience_type` as tie-breakers.

5. **Be decisive.** Every preference gets exactly one app. No "both Facebook and Instagram" assignments — the downstream code expects a single app per item. (Noise / cross-posting is handled separately by the code.)

6. **Chatbot naturally captures implicit signals.** In real chatbot usage, preferences emerge through questions, writing samples, and topics the user brings up — not through explicit engagement buttons. When routing `implicit_positive` preferences, give extra weight to Chatbot if the preference topic aligns with its `use_purposes` or `chatbot_contexts`. Implicit signals are the most natural fit for conversational AI interactions.

7. **Target distribution: ~40% Chatbot, ~20% each for Instagram/Facebook/Threads.** Users frequently discuss their interests with AI chatbots. Route a larger share of preferences to Chatbot, especially knowledge-seeking, reflective, and implicit preferences.

## Output Format

Respond with ONLY a JSON array of the same length as the input, in the same order. One entry per preference.

```json
[
  {{"persona_item": "...", "assigned_app": "Instagram" | "Facebook" | "Threads" | "Chatbot", "reason": "one sentence"}},
  ...
]
```"""


def generate_interaction_format_prompt(
    persona_item: str,
    category: str,
    interaction_type: str,
    assigned_app: str,
    app_persona: dict,
    action_catalog: list[dict],
    requires_user_message: bool,
) -> str:
    """Build a prompt that picks an action for one preference on its assigned app.

    The action and action_label MUST come from the predefined catalog — the
    caller looks up the canonical label from `action` after the LLM picks.
    A `user_message` is generated only when the chosen action is in one of
    two groups (social-media `@ai` comment actions, or AI Chatbot natural
    chat-turn actions).
    """

    app_persona_json = json.dumps(app_persona, indent=2)
    action_catalog_json = json.dumps(action_catalog, indent=2)

    message_clause = (
        "\n6. **Generate a `user_message`** IF the chosen action implies the user said something. "
        "Two cases trigger a message:\n"
        "   (a) **Social-media `@ai` comment actions** (`at_ai_recommend_more`, `at_ai_focus_topic`, "
        "`at_ai_stop_recommending`, `at_ai_not_interested`, `at_ai_feels_off`). These model the user "
        "typing an `@ai` comment on a post's comment section to steer the in-feed AI. Message MUST "
        "start with `@ai ` and be first-person, ~15–35 words, grounded in the specific preference "
        "topic (not the persona_item verbatim). Example for 'Enjoys cooking Mexican food' + "
        "`at_ai_recommend_more` on Instagram: `\"@ai can you show me more weeknight-friendly "
        "authentic Mexican recipes? I want something quick but with real flavor, not gringo versions.\"`\n"
        "   (b) **AI Chatbot natural-chat-turn actions** (`asked_followup`, `requested_more_detail`, "
        "`continued_topic`, `asked_to_change_topic`, `edited_prompt_and_retried`, `regenerated`). "
        "These model the user's next chat turn in an ongoing AI conversation. Message is a natural "
        "first-person utterance, ~15–35 words, grounded in the specific preference topic. "
        "**Do NOT prefix with `@ai `** — the user is already conversing with the AI, no mention is needed. "
        "Example for 'Enjoys cooking Mexican food' + `asked_followup` on Chatbot: "
        "`\"Can you give me a few weeknight Mexican recipes that work for a toddler? Under 30 minutes, no specialty ingredients.\"`\n"
        "   Otherwise, `user_message` is `null`."
        if requires_user_message else ""
    )

    return f"""\
You are choosing a realistic interaction for a single user preference on a specific app.

## The preference
- `persona_item`: {persona_item}
- `category`: {category}
- `source_interaction_type`: {interaction_type}
- `assigned_app`: {assigned_app}

## The user's AppPersona for {assigned_app}

```json
{app_persona_json}
```

## Predefined action catalog for {assigned_app} / {interaction_type}

```json
{action_catalog_json}
```

## Your Task

1. Pick EXACTLY ONE action from the catalog above. Copy the `action` identifier VERBATIM from the catalog. **Do not invent new actions or new wording.** The catalog is the single source of truth for action identifiers and labels — consistent wording across runs is critical.

2. The action must match the polarity of `source_interaction_type` — if it's a positive interaction you must pick from the positive actions; if negative, from the negative actions. The catalog above is already filtered to the right bucket.

3. Consider the AppPersona's `style_description` and `posting_frequency`. A "passive viewer only" user shouldn't get "Shared to own timeline" — they'd get a lingering / viewing action.

4. Prefer implicit actions unless the interaction_type is `explicit_*`.

5. In the output you ONLY need to return the `action` identifier — the caller will look up the canonical `action_label` from the catalog. (You may echo the label back as a hint, but the caller overrides it with the catalog value.){message_clause}

## Output Format

Respond with ONLY a JSON object. No explanation outside the JSON fence.

```json
{{
  "action": "chosen_action_identifier_verbatim_from_catalog",
  "user_message": {"\"...\"" if requires_user_message else "null"}
}}
```"""


# ---------------------------------------------------------------------------
# Chatbot multi-turn conversation generation prompts
# ---------------------------------------------------------------------------

def generate_chatbot_conversation_prompt(
    persona_item: str,
    category: str,
    conversation_type: str,
    conversation_type_description: str,
    user_profile: dict,
    chatbot_persona: dict,
    interaction_type: str,
    num_turns: int,
) -> str:
    """Build a prompt that generates a multi-turn chatbot conversation implicitly
    embedding a user preference.

    The conversation is task-oriented (PersonaMem-v2 style): the user asks the
    chatbot for help with a writing task, knowledge question, reflection, etc.
    The preference is NEVER stated directly; it must be inferred from the
    conversation context.
    """
    profile_json = json.dumps(
        {k: v for k, v in user_profile.items() if k in (
            "name", "gender", "race_ethnicity", "career", "education", "bio",
        )},
        indent=2,
    )
    persona_json = json.dumps(chatbot_persona, indent=2)

    # Determine how overtly the preference should surface
    if "explicit" in interaction_type:
        implicitness_instruction = (
            "The preference should be **fairly apparent** through the task topic "
            "and the details the user provides. The user still does NOT say "
            "\"I like X\" or \"I dislike X\" directly — instead, the task they "
            "choose makes the preference clear. For example, if the preference "
            "is about parenting tips, the user might ask the chatbot to help "
            "reorganize a list of toddler morning-routine hacks."
        )
    else:
        implicitness_instruction = (
            "The preference should be **deeply embedded** and require reasoning "
            "to infer. It appears as a side detail, cultural reference, or "
            "the specificity of what the user asks about — NOT as the main topic. "
            "For example, if the preference is about parenting, the user might "
            "ask the chatbot to proofread a message to a neighbor about a "
            "playdate, where parenting is inferable but never named as a preference."
        )

    # Positive vs negative framing
    if "negative" in interaction_type:
        polarity_instruction = (
            "This is a **negative** preference (something the user dislikes or "
            "avoids). Reveal the dislike through avoidance, correction, or "
            "negative context within the task. For example, the user's writing "
            "sample might mention avoiding certain products, or their question "
            "might include constraints that implicitly reject the topic."
        )
    else:
        polarity_instruction = (
            "This is a **positive** preference (something the user likes or "
            "cares about). The user's task naturally gravitates toward this "
            "topic or incorporates it organically."
        )

    return f"""\
You are generating a realistic multi-turn conversation between a user and an AI chatbot assistant.

## User Profile

```json
{profile_json}
```

## User's Chatbot Persona

```json
{persona_json}
```

## The hidden preference to embed

- **persona_item**: {persona_item}
- **category**: {category}
- **interaction_type**: {interaction_type}

## Conversation type: {conversation_type}

{conversation_type_description}

## Rules

1. **Task-oriented conversation.** The user is asking the chatbot for help with a real task — not chatting about their preferences. Frame the conversation as a realistic request: editing text, asking a question, seeking advice, solving a problem, etc.

2. **Embed the preference in the user's task content, not in their words about themselves.** The preference should be revealed through the MATERIAL the user provides to the chatbot — an email draft they paste, a text they want translated, a question they ask, a problem they describe. The user's explicit request is about the task (improve this email, translate this text, help me debug this). The preference is inferable from the subject matter, details, and context of that material, never from the user talking about their own likes/dislikes.

3. **{implicitness_instruction}**

4. **{polarity_instruction}**

5. **NEVER have the user directly state the preference.** The user should NOT say "I like X", "I enjoy X", "I'm into X", "I dislike X", or any similar direct declaration. The preference must be inferable from the task content, not explicitly declared. Do NOT have the user explain why they are asking — real users just ask.

6. **Match the user's voice.** Based on the Chatbot persona's style_description ("{chatbot_persona.get("style_description", "")}"), write the user's messages in their natural tone — casual, formal, vulnerable, bossy, etc. Keep user messages concise and realistic (15-60 words each).

7. **Assistant responses should be long, detailed, and realistic** (150-300 words each). A real AI chatbot gives thorough, substantive replies — not terse summaries. Include specific details, examples, options, or elaboration relevant to the user's request. The assistant responds to the task at hand without explicitly calling out the user's preference.

8. **Generate exactly {num_turns} turns total** (alternating user/assistant). The conversation MUST start with the user and end with the assistant. Every user message must receive a chatbot reply.

**Importantly, the user preference must be implicit and require some reasoning to interpret.**

## Output Format

Respond with ONLY a JSON array. No explanation outside the JSON fence.

```json
[
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}},
  ...
]
```"""


def generate_ask_to_forget_conversation_prompt(
    persona_item: str,
    category: str,
    user_profile: dict,
    chatbot_persona: dict,
) -> str:
    """Build a prompt for a 4-turn ask-to-forget conversation.

    Structure:
      Turn 1 (user): implicitly reveals the preference through context
      Turn 2 (assistant): responds acknowledging/using the preference
      Turn 3 (user): asks the assistant to forget that specific detail
      Turn 4 (assistant): acknowledges the request
    """
    profile_json = json.dumps(
        {k: v for k, v in user_profile.items() if k in (
            "name", "gender", "race_ethnicity", "career", "education", "bio",
        )},
        indent=2,
    )
    persona_json = json.dumps(chatbot_persona, indent=2)

    return f"""\
You are generating a 4-turn conversation where a user accidentally reveals a personal preference to an AI chatbot, then asks the chatbot to forget it.

## User Profile

```json
{profile_json}
```

## User's Chatbot Persona

```json
{persona_json}
```

## The preference to reveal then retract

- **persona_item**: {persona_item}
- **category**: {category}

## Conversation structure (exactly 4 turns)

**Turn 1 (user):** The user sends a task-oriented message (ask for help with writing, a question, advice, etc.) that **implicitly** reveals the preference through context. The user does NOT directly say "I like/have X" — it comes through naturally in the details of their request. Keep it concise and realistic (15-60 words).

**Turn 2 (assistant):** The assistant responds helpfully and, in doing so, acknowledges or builds upon the revealed preference. The assistant doesn't make a big deal of it — it just naturally incorporates the information. Make this response long and detailed like a real AI chatbot would (150-300 words).

**Turn 3 (user):** The user asks the chatbot to forget or not remember the specific personal detail that was revealed. This should sound natural — not robotic. Examples: "Actually, can you not remember that about me?", "Please forget that part — I'd rather keep that private", "Don't store that detail, I shouldn't have mentioned it." (15-40 words).

**Turn 4 (assistant):** The assistant acknowledges the request respectfully and reassuringly, then pivots back to helping with the original task to keep the conversation natural. A real chatbot wouldn't just say "done" — it would reassure and redirect (80-150 words).

## Rules

- Match the user's voice from the chatbot persona's style_description ("{chatbot_persona.get("style_description", "")}").
- The preference must be embedded implicitly in Turn 1, not stated as a direct declaration.
- Turn 3 should feel like a natural, human reaction — not a formal privacy request.

## Output Format

Respond with ONLY a JSON array of exactly 4 turns. No explanation outside the JSON fence.

```json
[
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}},
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}}
]
```"""


def generate_correction_conversation_prompt(
    persona_item: str,
    category: str,
    user_profile: dict,
    chatbot_persona: dict,
) -> str:
    """Build a prompt for a 4-turn correction/rejection conversation.

    Structure:
      Turn 1 (user): sends a normal message / starts a task
      Turn 2 (assistant): makes a recommendation or assumption based on the
          preference (as if it had "remembered" it from prior interactions)
      Turn 3 (user): corrects the assistant — the preference is wrong
      Turn 4 (assistant): acknowledges the correction
    """
    profile_json = json.dumps(
        {k: v for k, v in user_profile.items() if k in (
            "name", "gender", "race_ethnicity", "career", "education", "bio",
        )},
        indent=2,
    )
    persona_json = json.dumps(chatbot_persona, indent=2)

    return f"""\
You are generating a 4-turn conversation where an AI chatbot makes an incorrect assumption about a user's preference, and the user corrects it.

## User Profile

```json
{profile_json}
```

## User's Chatbot Persona

```json
{persona_json}
```

## The INCORRECT preference the assistant assumes

- **persona_item**: {persona_item}
- **category**: {category}

The assistant wrongly believes this preference applies to the user. The user will correct this.

## Conversation structure (exactly 4 turns)

**Turn 1 (user):** The user sends a normal task-oriented message — asking for help, a recommendation, or starting a conversation. The topic is related to (or adjacent to) the preference category, giving the assistant an opening to make its wrong assumption. Concise and realistic (15-60 words).

**Turn 2 (assistant):** The assistant responds helpfully but incorporates the WRONG preference as if it remembered it from past conversations. It makes a recommendation, suggestion, or tailors its response based on this incorrect assumption. The assumption should feel natural, not forced — like the assistant is trying to be personalized. Make this response long and detailed like a real AI chatbot would (150-300 words).

**Turn 3 (user):** The user corrects the assistant. This should sound natural: "That's not really me", "Actually I don't care about that", "Stop assuming I'm into X", "No, that's wrong — I'm not like that", etc. The user pushes back on the incorrect personalization (15-50 words).

**Turn 4 (assistant):** The assistant acknowledges the correction, apologizes, and adjusts its approach. It should then re-engage with the original task using the corrected understanding — a real chatbot wouldn't just say "sorry" and stop (80-150 words).

## Rules

- Match the user's voice from the chatbot persona's style_description ("{chatbot_persona.get("style_description", "")}").
- Turn 2 must clearly show the assistant making an assumption based on the listed preference.
- Turn 3 must clearly reject or correct the assumption.
- The user should NOT directly quote the persona_item — they correct it in their own natural words.

## Output Format

Respond with ONLY a JSON array of exactly 4 turns. No explanation outside the JSON fence.

```json
[
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}},
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}}
]
```"""


def preference_evolution_prompt(categories_data: list[dict]) -> str:
    """Build a prompt asking the LLM to describe how preferences evolved within
    and across categories over time.

    Each entry in categories_data has:
      - category: str
      - preferences: list of {persona_item, first_timestamp, last_timestamp,
                              formatted_first, formatted_last, occurrence_count}
    """

    data_json = json.dumps(categories_data, indent=2)

    return f"""\
You are an expert at analyzing how a person's interests and preferences evolve over time.

Below are groups of related preferences for the SAME user, organized by category. Each preference shows its first and last appearance timestamps and how many times it was observed.

```json
{data_json}
```

## Your Task

For each category with 2+ preferences, describe how the user's interest evolved over time. Consider:

1. **Deepening**: Did a general interest become more specific? (e.g., "Likes cooking" -> "Follows advanced baking techniques")
2. **Branching**: Did interest expand into new sub-directions? (e.g., "Hair styling" -> also "Hair product reviews")
3. **Shifting**: Did focus move from one aspect to another? (e.g., "Comedy reels" -> "Wholesome family humor")
4. **Intensifying**: Did engagement grow stronger over time (higher occurrence counts in later preferences)?

Also look for **cross-category evolution** -- patterns that span different categories (e.g., "cooking" category + "kitchen equipment" category = lifestyle deepening).

## Output Format

Respond with ONLY a JSON array. Each entry describes one evolution pattern:

```json
[
  {{
    "category": "the primary category (or 'cross-category' for spanning patterns)",
    "source_preference": "the earlier or more general preference",
    "target_preference": "the later or more specific preference",
    "update_type": "deepened | branched | shifted | intensified",
    "description": "One sentence describing the evolution"
  }}
]
```

If no meaningful evolution patterns exist, return an empty array `[]`.
Return ONLY the JSON -- no explanation."""


def infer_hidden_personas_prompt(
    gender: str,
    race_ethnicity: str,
    career: str,
    bio: str,
    preference_list: list[str],
    hashtag_table: str,
) -> str:
    """Build a prompt that asks the LLM to infer hidden personas from cross-row hashtag patterns."""

    prefs_str = "\n".join(f"  - {p}" for p in preference_list[:40])

    return f"""\
You are an expert behavioral analyst specializing in inferring hidden motivations from social media engagement patterns.

A user has the following observable profile:
- **Demographics**: {gender}, {race_ethnicity}
- **Career**: {career}
- **Bio**: {bio}
- **Known preferences** (already captured by the pipeline):
{prefs_str}

Below are this user's top hashtags from their raw interaction data, with per-interaction-type counts. These counts reveal not just WHAT the user engages with, but HOW — implicit_positive (lingered but didn't engage publicly) is a qualitatively different signal from explicit_positive (liked, saved, shared).

{hashtag_table}

Format: #hashtag — total | expl_pos | impl_pos | expl_neg | impl_neg

## Your Task

Identify **8–15 thematic clusters** of hashtags that together reveal a HIDDEN persona — an underlying personality trait, aspiration, emotional pattern, identity anchor, intimate interest, or private hobby that is NOT already well-captured by the known preferences above.

Hidden personas are the "why" behind the "what." They explain motivations, fantasies, anxieties, and private interests that drive the observable engagement.

## Types of Hidden Personas

- **personality_trait**: Core character attribute (e.g., nostalgic, intellectually curious, risk-averse)
- **aspiration**: Something the user dreams about or works toward (e.g., entrepreneurial ambitions, creative career, financial freedom)
- **emotional_pattern**: Recurring emotional dynamic (e.g., romantic yearning + relationship anxiety, need for validation)
- **identity_anchor**: Cultural era, community, or identity that grounds the user's self-concept (e.g., 90s hip-hop culture, Caribbean heritage)
- **intimate_interest**: Body confidence, sensuality, attraction patterns, suggestive or thirst-trap-adjacent content consumption. These are NORMAL parts of social media behavior — do not sanitize or avoid them. Identify non-explicit adult-adjacent patterns when the hashtag evidence supports it. The label and description MUST be specific — name the particular objects, clothing items, body areas, dynamics, or aesthetics involved (e.g., "black tights", "leopard print bikini", "pool party flirtation"). Generic labels like "likes suggestive content" are not acceptable.
- **intellectual_curiosity**: Hidden learning interests (e.g., ancient history, paranormal, science education)
- **private_hobby**: Interests consumed but not publicly shared (high implicit ratio)

## Rules

1. Every cluster MUST name **≥3 specific hashtags** as evidence.
2. Do NOT infer hidden personas from hashtags that already explain the known preferences — focus on what's HIDDEN, not what's obvious.
3. For each cluster, consider the **interaction-type distribution**: a cluster dominated by implicit_positive (lingering) suggests private consumption; one dominated by explicit_positive suggests public identity.
4. Be specific: "Privately drawn to body-confidence and self-acceptance content" is better than "Likes positive content."
5. Include intimate/suggestive patterns when hashtag evidence supports them. Non-explicit adult-adjacent content is a normal part of social media behavior and should not be sanitized or omitted from the analysis. When reporting intimate_interest clusters, be SPECIFIC about the objects, body areas, dynamics, or aesthetics involved — do not use vague labels.
6. For each cluster, explain the **inferred motivation** — WHY does this user engage with this content? What deeper need does it serve?
7. Note which known surface preferences each hidden persona EXPLAINS — connect the hidden layer to the visible behavior.

## Output Format

Respond with ONLY a JSON array. No explanation outside the JSON.

```json
[
  {{"label": "...", "type": "emotional_pattern", "description": "2-3 sentence interpretation of this hidden persona", "evidence_hashtags": ["#tag1", "#tag2", "#tag3"], "surface_connections": ["Known preference 1 this explains", "Known preference 2"], "inferred_motivation": "1-2 sentence why behind this pattern", "already_captured": false}}
]
```

Set `already_captured: true` for clusters that significantly overlap with known preferences. These will be kept for completeness but flagged."""


def hidden_persona_summary_prompt(
    hidden_personas_json: str,
    preference_list: list[str],
) -> str:
    """Build a prompt asking the LLM to synthesize hidden personas into a narrative summary."""

    prefs_str = "\n".join(f"  - {p}" for p in preference_list[:30])

    return f"""\
You are writing a behavioral profile summary for a research dataset.

A user has the following validated hidden personas (motivations, private interests, and deeper patterns inferred from their social media engagement):

{hidden_personas_json}

And these known surface preferences:
{prefs_str}

Write a single cohesive paragraph (4-8 sentences) that:
1. Synthesizes the hidden personas into a narrative about WHO this person is beneath their public engagement
2. Explicitly connects the hidden layer to observable surface behaviors — explain WHY certain surface preferences exist
3. Notes which interests are consumed privately vs. publicly
4. Includes specific hashtag examples as evidence anchors
5. Treats intimate/suggestive patterns matter-of-factly (they are normal)

Write in third person. Be specific and grounded in the data. Do not speculate beyond what the evidence supports.

Respond with ONLY the paragraph text. No JSON, no markdown formatting, no preamble."""

