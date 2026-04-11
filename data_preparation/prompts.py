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

For EACH hashtag and the overall content context, infer as many **atomic persona traits or preferences** as possible about this user. An "atomic persona" is a single, specific, testable statement about the user's personality, interests, values, demographics, or lifestyle (e.g., "Interested in CrossFit", "Values family traditions", "Likely a parent of school-age children").

## Rules

1. **Be comprehensive**: Generate 3-5 atomic personas per hashtag, plus additional ones from the overall content context. More is better — cast a wide net of plausible inferences.
2. **Be specific**: Each persona item must be concrete and testable, not vague (e.g., "Enjoys cooking Italian food at home" rather than "Likes food").
3. **Calibrate confidence on a 0.0 to 1.0 scale**:
   - 0.0-0.2: Very speculative, loosely connected inference
   - 0.2-0.4: Plausible but based on a single weak signal
   - 0.4-0.6: Supported by moderate evidence in the content
   - 0.6-0.8: Strong evidence from multiple converging signals
   - 0.8-1.0: Near-certain, directly and unambiguously stated
   Use the full range. Assign higher scores when the evidence is strong.
4. **Handle negative interactions**: For "{interaction_type}" interactions, the user scrolled past or did not click on promoted content. This is a very weak signal — not clicking an ad does not reliably indicate dislike. Infer what the user might not prefer, but keep ALL confidence scores very low (0.05-0.15 range). Phrase the persona as what they DO prefer instead (e.g., if they ignored fast-food ads, infer "May prefer home-cooked or health-conscious meals").
5. **Consider diverse dimensions**: Think about interests, values, demographics, lifestyle, profession, cultural background, media consumption habits, purchasing behavior, and social identity.
6. **Categorize each inference**: Assign a **specific topical category** that describes the domain of the persona (e.g., "cooking", "Christian faith", "NFL fandom", "laundry products", "romantic relationships", "fitness", "parenting"). Do NOT use generic categories like "interests", "values", "personality", "lifestyle", or "demographics". The category should tell you what real-world topic the persona is about.
7. **Tag source hashtags**: For each persona, include ONLY the specific hashtag(s) that directly led to this inference — not the full list.

## Output Format

Respond with ONLY a JSON array. No explanation, no markdown outside the JSON fence.

```json
[
  {{"category": "cooking", "persona_item": "...", "confidence_score_init": 0.XX, "source_hashtags": ["#CookingWithAPurpose"]}},
  {{"category": "Christian faith", "persona_item": "...", "confidence_score_init": 0.XX, "source_hashtags": ["#ChristianLiving", "#LoveAndFaith"]}}
]
```"""


def summarize_and_cross_reference_prompt(atomic_personas: list[dict]) -> str:
    """Build a prompt that asks the LLM to cross-reference and score all atomic personas."""

    personas_json = json.dumps(atomic_personas, indent=2)

    return f"""\
You are an expert at synthesizing behavioral signals into a coherent user profile.

Below is a list of atomic persona traits/preferences inferred from a single user's social media interactions over time. Each entry includes the category, persona item, an initial confidence score, the source interaction type, the timestamp, and the `source_object_id` identifying which interaction row it came from.

```json
{personas_json}
```

## Your Task

1. **Cross-reference** every persona item against personas **from different interaction rows only** (different `source_object_id`). Personas inferred from the same row (same `source_object_id`) must NOT be marked as related to each other — they share the same evidence and cannot validate each other.
   - If two personas (from different rows) are **similar** (reinforce each other), mark them as related with relationship_type "similar".
   - If two personas (from different rows) **contradict** each other, mark them as related with relationship_type "contradictory".
   - If a persona has no meaningful cross-row relationship, mark it as "none".

2. **Do NOT compute confidence scores** — set `confidence_cross_referenced` to 0.0 for every item. Scoring is computed downstream.

3. **Keep the original `confidence_score_init` unchanged** — do not modify it.

4. **Do NOT filter anything out** — return every persona item, even contradictions. Filtering happens downstream.

5. **For each persona, list all related personas** in the `related_personas` array. Each entry must include the persona text AND its relationship type as an object: `{{"persona_item": "...", "type": "similar"}}` or `{{"persona_item": "...", "type": "contradictory"}}`.

6. **Preserve the `category`** from the input for each persona item.

## Output Format

Respond with ONLY a JSON array. No explanation.

```json
[
  {{
    "category": "...",
    "persona_item": "...",
    "confidence_score_init": 0.XX,
    "confidence_cross_referenced": 0.0,
    "relationship_type": "similar" | "contradictory" | "none",
    "related_personas": [{{"persona_item": "...", "type": "similar"}}, {{"persona_item": "...", "type": "contradictory"}}],
    "formatted_timestamp": "...",
    "source_interaction_type": "..."
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
