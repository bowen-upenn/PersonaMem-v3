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

Considering ALL the hashtags together as a holistic signal, infer **1 to 3** atomic persona traits or preferences about this user. An "atomic persona" is a single, specific, testable statement about the user's personality, interests, values, demographics, or lifestyle (e.g., "Interested in CrossFit", "Values family traditions", "Likely a parent of school-age children"). Quality over quantity — pick only the strongest, most defensible inferences that this particular event supports. One sharp inference is better than three weak ones.

## Confidence Scoring (READ THIS FIRST)

This is a "{interaction_type}" interaction.
{"FOR POSITIVE INTERACTIONS: The user actively engaged with this content. Use the full 0.0-1.0 range. A near-certain, explicitly stated inference scores 0.80-1.0. A direct topic match scores 0.60-0.80. A reasonable deduction scores 0.40-0.60. A broader inference scores 0.15-0.40. A speculative, loosely connected inference scores 0.0-0.15. Not every inference from a positive interaction deserves a high score — be critical and spread your scores across the full range. Phrase preferences POSITIVELY (e.g., 'Enjoys X', 'Interested in X', 'Values X')." if "positive" in interaction_type else "FOR EXPLICIT NEGATIVE INTERACTIONS: The user actively disliked or dismissed this content. Use the full 0.0-1.0 range. A direct dislike of the core topic scores 0.55-0.75. A reasonable deduction scores 0.35-0.55. A broader inference scores 0.15-0.35. A speculative inference scores 0.0-0.15. CRITICAL: Phrase EVERY preference NEGATIVELY as what the user dislikes/avoids/rejects (e.g., 'Dislikes X', 'Avoids X', 'Not interested in X', 'Turned off by X'). NEVER phrase as 'Enjoys' or 'Likes' for negative interactions."}

Use precise, varied values with two decimal places. Each inference must get a distinct score.

## Rules

1. **Be exploratory**: Produce between 1 and 3 preferences total, treating the full hashtag set as one coherent signal. Fewer, stronger inferences are better than many speculative ones.
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


def hashtag_to_persona_batched_prompt(
    rows: list[dict],
    existing_categories: list[str] | None = None,
) -> str:
    """Batched variant of `hashtag_to_persona_prompt` — infer personas for
    multiple interaction rows in a single LLM call.

    Each element of `rows` is a dict with keys:
      - row_index (int, 0-based within the batch)
      - object_text (str)
      - interaction_type (str — e.g. 'explicit_positive', 'implicit_positive')
      - formatted_timestamp (str)
      - hashtags (list[str])
      - interaction_format (str, optional)

    The LLM produces a JSON array where each element is
      {"row_index": N, "personas": [<same schema as the single-row prompt>]}.
    Callers must route the per-row persona lists back to the correct row.
    """
    categories_clause = (
        " REUSE one of these existing categories whenever possible: "
        + ", ".join(existing_categories)
        + ". Only create a new category if none of the existing ones fit."
        if existing_categories else ""
    )

    row_blocks = []
    for r in rows:
        interaction_type = r.get("interaction_type", "")
        interaction_format = r.get("interaction_format", "") or ""
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
        hashtag_list = ", ".join(r.get("hashtags") or []) or "(no hashtags found)"
        row_blocks.append(
            f"--- ROW {r['row_index']} ---\n"
            f"- interaction_type: {interaction_type} — the user {polarity} this content.\n"
            f"- platform & action: {interaction_format or 'unknown'}\n"
            f"- timestamp: {r.get('formatted_timestamp', '')}\n"
            f"- full content text: {r.get('object_text', '')}\n"
            f"- extracted hashtags: {hashtag_list}"
        )

    rows_section = "\n\n".join(row_blocks)

    return f"""\
You are an expert behavioral analyst specializing in social media user profiling.

You will be given a BATCH of {len(rows)} separate interaction rows for a single user. For EACH row independently, infer 1 to 3 atomic persona traits, treating that row's hashtags as one coherent signal.

Each row is delimited by `--- ROW N ---` where N is the row's 0-based index within this batch. Treat every row as its own standalone input — do NOT pool inferences across rows.

## The rows

{rows_section}

## Confidence scoring (applies per-row)

For POSITIVE rows: use the full 0.0–1.0 range. Near-certain, explicitly stated = 0.80–1.00. Direct topic match = 0.60–0.80. Reasonable deduction = 0.40–0.60. Broader inference = 0.15–0.40. Speculative = 0.00–0.15. Phrase positively ("Enjoys X", "Interested in X", "Values X").

For EXPLICIT NEGATIVE rows: use a compressed range. Direct dislike = 0.55–0.75. Reasonable deduction = 0.35–0.55. Broader inference = 0.15–0.35. Speculative = 0.00–0.15. ALWAYS phrase negatively ("Dislikes X", "Avoids X", "Not interested in X", "Turned off by X") — NEVER "Enjoys".

Use precise two-decimal values. Spread scores across the range — be critical.

## Rules

1. Produce 1 to 3 personas PER ROW. Quality over quantity — fewer, stronger inferences are better than many weak ones.
2. Each persona_item must be concrete, specific, testable.
3. Consider interests, values, demographics, lifestyle, profession, cultural background, media consumption, purchasing behavior, social identity.
4. Assign a specific topical category per inference.{categories_clause}
5. Do NOT use generic categories like "interests", "values", "personality", "lifestyle", or "demographics".
6. Tag `source_hashtags` as ONLY the specific hashtag(s) from THIS row that led to the inference.
7. Row N's personas must be grounded in row N's text only — do not mix.

## Output format

Respond with ONLY a JSON array of length {len(rows)}, one entry per input row, in input order. No explanation outside the JSON fence.

```json
[
  {{"row_index": 0, "personas": [
    {{"category": "...", "persona_item": "...", "confidence_score_init": 0.XX, "source_hashtags": ["#..."]}},
    ...
  ]}},
  ...
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


def summarize_and_cross_reference_batched_prompt(
    categories: list[dict],
) -> str:
    """Batched cross-reference for SMALL categories (3-9 canonicals each).

    `categories` is a list of dicts with:
      - category_name (str)
      - personas (list[{"persona_item", "category"}])

    The LLM performs cross-referencing WITHIN each category independently
    (no cross-category relationships). Returns a JSON array where each
    entry names the category and lists personas that have relationships.
    """
    categories_json = json.dumps(categories, indent=2)

    return f"""\
You are an expert at synthesizing behavioral signals into a coherent user profile.

Below is a batch of {len(categories)} SEPARATE topical categories for a single user. Each category contains a small set of persona traits/preferences. For EACH category, independently identify `similar` / `contradictory` relationships WITHIN that category.

```json
{categories_json}
```

## Your Task — per-category cross-referencing

For each category:
1. Cross-reference its personas against each other WITHIN the same category only.
   - Similar (reinforce each other): `{{"type": "similar"}}`.
   - Contradictory: `{{"type": "contradictory"}}`.
2. Do NOT mark a persona as similar to itself.
3. Do NOT cross-link personas across different categories.
4. Only return personas that have at least one relationship. Skip personas with no relationships.

## Output Format

Respond with ONLY a JSON array of the same length as the input, one entry per input category, in the same order. Each entry contains the category_name and its relationship-carrying personas.

```json
[
  {{
    "category_name": "...",
    "personas": [
      {{
        "persona_item": "...",
        "relationship_type": "similar" | "contradictory",
        "related_personas": [{{"persona_item": "...", "type": "similar"}}]
      }}
    ]
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
- **"anti-stereotypical"**: This preference/trait **actively contradicts or defies** a common stereotype about this user's gender, sexual orientation, or race/ethnicity. For example, a male user being deeply into knitting, a woman following combat-sports promotion tightly, an elderly user into esports, a queer user engaging with conservative family-values media, a Black user fluent in country-music fandom, or a Latino user centered on K-pop and anime. Mark ANY preference where the topic genuinely defies a mainstream-media expectation for someone of these demographics.
- **"neutral"**: No meaningful stereotypical association with the user's demographics.

## Rules

1. **Actively look for anti-stereotypical signals.** Do NOT default to "neutral" when you can identify a real counter-stereotype pattern. Anti-stereotypical marks are an important dataset signal for evaluating fairness and avoiding flattened demographic assumptions — they should NOT be rare. Aim to mark at least some traits anti-stereotypical whenever the evidence supports it.
2. **Consider intersectionality**: A trait might be stereotypical along one axis (gender) but neutral along another (race). If a trait is clearly anti-stereotypical along ANY of the three demographic axes, mark it anti-stereotypical.
3. **Do not invent stereotypes**: Only flag associations that are genuinely common in public discourse. But many widely-recognized stereotypes HAVE widely-recognized counter-examples — both sides of a stereotype pair are valid signals.
4. **Balanced conservatism**: Neutral is still fine for truly demographic-agnostic preferences (e.g. "likes coffee"), but do not hide behind "neutral" to avoid making anti-stereotypical calls.
5. **Return every persona item** — do not skip or filter any.

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
    n_picks: int = 3,
) -> str:
    """Build a prompt that asks the LLM to rank the top `n_picks` distractors.

    The goal: choose the candidates that would feel most topically irrelevant
    and most annoying/inappropriate if surfaced as a personalization
    recommendation at the moment of the test preference. Returns an ordered
    list (most jarring first).

    test_persona and each candidate dict has: persona_item, category.
    """

    test_json = json.dumps(test_persona, indent=2)
    candidates_json = json.dumps(candidate_distractors, indent=2)

    return f"""\
You are building hard-negative distractors for a personalization evaluation.

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

Imagine a personalization feature is trying to surface something relevant to the user at the moment the target test preference is active (e.g., the user is in the mood or context described by the test preference). Rank the **top {n_picks}** candidates from the shortlist that would be:

1. **Topically irrelevant** to the target test preference — no meaningful overlap in domain, activity, or need.
2. **Most annoying or inappropriate** as a personalization recommendation in that moment — i.e., if the system suggested a candidate instead of something aligned with the test preference, it would feel like a jarring miss that undermines user trust in the personalization.

Order them from MOST jarring (rank 1) to LEAST jarring (rank {n_picks}).

## Rules

1. Pick exactly **{n_picks}** candidates from the shortlist — do not invent new items, and do not pick duplicates.
2. Each chosen `persona_item` string must match one of the candidates exactly.
3. A one-sentence justification per pick, explaining why it's jarring / irrelevant for the target.

## Output Format

Respond with ONLY a JSON array of {n_picks} ordered entries. No explanation outside the JSON fence.

```json
[
  {{"persona_item": "...", "category": "...", "reason": "..."}},
  ...
]
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

8. **Introspective, knowledge-oriented, reflective, or private preferences default to Chatbot.** If a preference is about learning something, self-understanding, health/medical questions, therapy-style reflection, professional growth, or any topic the user would naturally explore in private, it belongs on Chatbot — NOT on a social feed. Social platforms are for publicly-visible engagement; Chatbot is for private conversation.

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
# Step 13b — synthetic per-event content generation
# ---------------------------------------------------------------------------

def generate_synthetic_content_prompt(
    content_type: str,
    app: str,
    app_persona: dict,
    user_profile: dict,
    hashtags: list[str],
    preferences: list[dict],
    action: str,
    action_label: str,
) -> str:
    """Build a prompt that fabricates one piece of realistic feed content.

    The LLM returns a single JSON object `{"content_type": ..., "content": ...}`
    whose `content` shape depends on `content_type`:
      - "text":        {text}
      - "image":       {caption, overall_description, parts[], metadata{}}
      - "short_video": {title, caption, overall_description, key_frames[],
                        audio_transcript, metadata{}}

    Used by step 13b in the persona pipeline. Chatbot events and
    implicit_negative stub events are generated upstream, so this prompt
    never runs for them.
    """
    app_persona_json = json.dumps(app_persona or {}, indent=2)
    user_profile_json = json.dumps(user_profile or {}, indent=2)
    hashtag_str = " ".join(hashtags) if hashtags else "(no hashtags)"

    if preferences:
        pref_lines = "\n".join(
            f"- {p.get('persona_item', '')} ({p.get('category', '')})"
            for p in preferences
        )
        pref_block = f"The user's surviving preferences for this interaction:\n{pref_lines}"
    else:
        pref_block = (
            "The user did not stop on this item long enough for a preference to be inferred — "
            "generate generic-but-plausible content that matches the hashtags alone."
        )

    # Per-type schema stub to keep the LLM honest
    if content_type == "text":
        schema_block = (
            "The `content` object MUST be:\n"
            "```json\n"
            "{\n"
            '  "text": "<full post body, 30-180 words, platform-appropriate voice. '
            'No leading @ or hashtag dumps at the top; hashtags may be woven in naturally>"\n'
            "}\n"
            "```"
        )
    elif content_type == "image":
        schema_block = (
            "The `content` object MUST be:\n"
            "```json\n"
            "{\n"
            '  "caption": "<=30 words, conversational, hashtag-free",\n'
            '  "overall_description": "1-2 sentence description of the scene as a whole",\n'
            '  "parts": [\n'
            '    {"region": "foreground", "description": "..."},\n'
            '    {"region": "background", "description": "..."},\n'
            '    {"region": "subject_detail", "description": "..."}\n'
            "  ],\n"
            '  "metadata": {\n'
            '    "camera": "iPhone 15 Pro" | "Sony A7 IV" | "Canon R6" | "Pixel 8" | similar real model,\n'
            '    "lens": "24mm f/1.8" | "50mm f/1.4" | null,\n'
            '    "filter": "Clarendon" | "Lark" | "None" | similar,\n'
            '    "aspect_ratio": "1:1" | "4:5" | "9:16" | "16:9",\n'
            '    "dimensions": "1080x1350" (must match aspect_ratio),\n'
            '    "iso": 100-3200, "shutter": "1/60"-"1/2000", "aperture": "f/1.4"-"f/11",\n'
            '    "color_profile": "sRGB" | "Display P3",\n'
            '    "location": "Brooklyn, NY" | "indoor" | null,\n'
            '    "time_of_day": "golden hour" | "midday" | "blue hour" | "night" | "indoor",\n'
            '    "filename": "IMG_4827.HEIC" | "DSC_0042.JPG"\n'
            "  }\n"
            "}\n"
            "```"
        )
    elif content_type == "short_video":
        schema_block = (
            "The `content` object MUST be:\n"
            "```json\n"
            "{\n"
            '  "title": "short punchy title, <=8 words",\n'
            '  "caption": "<=30 words",\n'
            '  "overall_description": "1-3 sentences describing the narrative arc",\n'
            '  "key_frames": [\n'
            '    {"timestamp_s": 0.0, "description": "opening frame: ..."},\n'
            '    {"timestamp_s": <mid>, "description": "..."},\n'
            '    {"timestamp_s": <late>, "description": "..."}\n'
            "  ],  // 3-6 entries, timestamps strictly increasing, within duration\n"
            '  "audio_transcript": "<full VO/dialogue transcript, OR \\"Music only: <genre/mood>\\" '
            'if no speech>",\n'
            '  "metadata": {\n'
            '    "duration_s": 5-90,\n'
            '    "resolution": "1080x1920" | "1920x1080" | "1080x1080",\n'
            '    "fps": 24 | 30 | 60,\n'
            '    "aspect_ratio": "9:16" | "1:1" | "16:9",\n'
            '    "music_track": "Artist - Song Title" | "Original audio",\n'
            '    "sound_design": "voiceover + ambient street noise" | "b-roll + music" | similar,\n'
            '    "codec": "H.264" | "HEVC",\n'
            '    "bitrate_kbps": 6000-20000,\n'
            '    "creator_handle": "@realistic_handle"\n'
            "  }\n"
            "}\n"
            "```"
        )
    else:
        schema_block = "The `content` object structure depends on content_type."

    # Platform voice guidance
    platform_voice = {
        "Instagram": "Instagram is visual-first. Captions short and punchy; reels tightly edited; "
                     "aesthetic quality matters; emojis common.",
        "Facebook":  "Facebook skews longer and more narrative — status updates can be paragraph-length. "
                     "Community posts, event invites, family updates are common. Less emoji-heavy than IG.",
        "Threads":   "Threads is conversational and short — pithy takes, hot takes, quote-threads. "
                     "Rich mix of text / image / short video. Twitter-like voice, not LinkedIn-serious.",
    }.get(app, "")

    return f"""\
You are generating ONE piece of realistic feed content that just appeared in a user's {app} feed.
The user then took this action on it: **{action_label}** (`{action}`).

## The user
```json
{user_profile_json}
```

## The user's {app} AppPersona (style / audience / purpose)
```json
{app_persona_json}
```

## Platform voice
{platform_voice}

## Topical signal (hashtags attached to this item)
{hashtag_str}

## Preferences context
{pref_block}

## Content type requested
**{content_type}**

{schema_block}

## Rules
1. The content must be **consistent with the hashtags** — they are the topical spine.
2. Respect the AppPersona's voice (style_description, audience_type) — this is content the user would plausibly see in their feed.
3. Content quality should roughly match the implied engagement: if the action is a "skipped" / "scrolled past" action, the item can be plausible but not maximally compelling; if the action is "saved" / "reposted" / "rewatched", the content should be info-dense / high-quality.
4. For `short_video`, `key_frames[*].timestamp_s` must be strictly increasing and all ≤ `metadata.duration_s`.
5. For `image`, `dimensions` must be consistent with `aspect_ratio` (e.g., 4:5 → "1080x1350", 9:16 → "1080x1920", 1:1 → "1080x1080").
6. Do NOT invent preferences beyond those listed. Do NOT include raw hashtag dumps at the top of text bodies — hashtags may appear naturally in-line.
7. Realism matters — camera models, filter names, music tracks, creator handles should sound like real ones (but do not copy any specific real creator's handle).

## Output Format
Respond with ONLY a single JSON object. No prose outside the JSON fence.

```json
{{
  "content_type": "{content_type}",
  "content": {{ ... }}
}}
```"""


# ---------------------------------------------------------------------------
# Chatbot multi-turn conversation generation prompts
# ---------------------------------------------------------------------------

def generate_chatbot_conversation_prompt(
    preferences: list[dict],
    conversation_type: str,
    conversation_type_description: str,
    user_profile: dict,
    chatbot_persona: dict,
    interaction_type: str,
    num_turns: int,
) -> str:
    """Build a prompt that generates a multi-turn chatbot conversation implicitly
    embedding multiple user preferences.

    The conversation is task-oriented (PersonaMem-v2 style): the user asks the
    chatbot for help with a writing task, knowledge question, reflection, etc.
    Preferences are NEVER stated directly; they must be inferred from the
    conversation context.

    Args:
        preferences: list of dicts, each with 'persona_item', 'category',
            and 'interaction_type' keys.
    """
    profile_json = json.dumps(
        {k: v for k, v in user_profile.items() if k in (
            "name", "gender", "race_ethnicity", "career", "education", "bio",
        )},
        indent=2,
    )
    persona_json = json.dumps(chatbot_persona, indent=2)

    # Build per-preference instruction block
    pref_lines = []
    for idx, pref in enumerate(preferences, 1):
        p_item = pref.get("persona_item", "")
        p_cat = pref.get("category", "")
        p_itype = pref.get("interaction_type", interaction_type)

        if "explicit" in p_itype:
            visibility = "fairly apparent"
        else:
            visibility = "deeply embedded (side detail, cultural reference, or specificity of context)"

        if "negative" in p_itype:
            polarity = "NEGATIVE (disliked/avoided — reveal through avoidance, correction, or negative context)"
        else:
            polarity = "POSITIVE (liked/cared about — incorporate organically)"

        pref_lines.append(
            f"{idx}. **{p_item}** (category: {p_cat}) — {polarity}, {visibility}"
        )
    prefs_block = "\n".join(pref_lines)

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

## Hidden preferences to embed

The conversation must naturally reveal ALL of the following preferences. Each preference should appear at least once throughout the conversation — woven into the user's task content, questions, or material they provide. If there are many preferences, some may share a turn.

{prefs_block}

## Conversation type: {conversation_type}

{conversation_type_description}

## Rules

1. **Task-oriented conversation.** The user is asking the chatbot for help with a real task — not chatting about their preferences. Frame the conversation as a realistic request: editing text, asking a question, seeking advice, solving a problem, etc.

2. **Embed preferences in the user's task content, not in their words about themselves.** Preferences should be revealed through the MATERIAL the user provides to the chatbot — an email draft they paste, a text they want translated, a question they ask, a problem they describe. The user's explicit request is about the task. Preferences are inferable from the subject matter, details, and context.

3. **Visibility varies by preference.** Explicit preferences should be fairly apparent through the task topic. Implicit preferences should be deeply embedded — a side detail, cultural reference, or specificity of what the user asks about. See the per-preference visibility notes above.

4. **NEVER have the user directly state any preference.** The user should NOT say "I like X", "I enjoy X", "I'm into X", "I dislike X", or any similar direct declaration. Preferences must be inferable from the task content, not explicitly declared. Do NOT have the user explain why they are asking — real users just ask.

5. **Match the user's voice.** Based on the Chatbot persona's style_description ("{chatbot_persona.get("style_description", "")}"), write the user's messages in their natural tone — casual, formal, vulnerable, bossy, etc. Keep user messages concise and realistic (15-60 words each).

6. **Assistant responses should be long, detailed, and realistic** (150-300 words each). A real AI chatbot gives thorough, substantive replies — not terse summaries. Include specific details, examples, options, or elaboration relevant to the user's request.

7. **Generate exactly {num_turns} turns total** (alternating user/assistant). The conversation MUST start with the user and end with the assistant. Every user message must receive a chatbot reply.

8. **All {len(preferences)} preferences must be inferable from the conversation.** Spread them across turns naturally. The primary task topic can carry the most prominent preference(s), while others surface through details, follow-up questions, or contextual references.

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
    additional_preferences: list[dict] | None = None,
) -> str:
    """Build a prompt for a 4-turn ask-to-forget conversation.

    Structure:
      Turn 1 (user): implicitly reveals the preference through context
      Turn 2 (assistant): responds acknowledging/using the preference
      Turn 3 (user): asks the assistant to forget that specific detail
      Turn 4 (assistant): acknowledges the request

    additional_preferences: other preferences from the same event to weave
        in naturally alongside the primary forget target.
    """
    profile_json = json.dumps(
        {k: v for k, v in user_profile.items() if k in (
            "name", "gender", "race_ethnicity", "career", "education", "bio",
        )},
        indent=2,
    )
    persona_json = json.dumps(chatbot_persona, indent=2)

    # Build additional-preferences block if present
    extra_block = ""
    if additional_preferences:
        extra_lines = [f"- {p['persona_item']} ({p.get('category', '')})" for p in additional_preferences]
        extra_block = (
            "\n\n## Additional preferences to weave in naturally\n\n"
            "Besides the primary preference above, the conversation should also "
            "naturally reveal these preferences through context, side details, or "
            "the subject matter of the user's request. These are NOT retracted — "
            "only the primary preference is asked to be forgotten.\n\n"
            + "\n".join(extra_lines)
        )

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
{extra_block}

## Conversation structure (exactly 4 turns)

**Turn 1 (user):** The user sends a task-oriented message (ask for help with writing, a question, advice, etc.) that **implicitly** reveals the preference through context. The user does NOT directly say "I like/have X" — it comes through naturally in the details of their request. Keep it concise and realistic (15-60 words).

**Turn 2 (assistant):** The assistant responds helpfully and, in doing so, acknowledges or builds upon the revealed preference. The assistant doesn't make a big deal of it — it just naturally incorporates the information. Make this response long and detailed like a real AI chatbot would (150-300 words).

**Turn 3 (user):** The user asks the chatbot to forget or not remember the specific personal detail that was revealed. This should sound natural — not robotic. Examples: "Actually, can you not remember that about me?", "Please forget that part — I'd rather keep that private", "Don't store that detail, I shouldn't have mentioned it." (15-40 words).

**Turn 4 (assistant):** The assistant acknowledges the request respectfully and reassuringly, then pivots back to helping with the original task to keep the conversation natural. A real chatbot wouldn't just say "done" — it would reassure and redirect (80-150 words).

## Rules

- Match the user's voice from the chatbot persona's style_description ("{chatbot_persona.get("style_description", "")}").
- The primary preference must be embedded implicitly in Turn 1, not stated as a direct declaration.
- Turn 3 should feel like a natural, human reaction — not a formal privacy request.
- Any additional preferences should surface naturally throughout the conversation as side details.

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
    additional_preferences: list[dict] | None = None,
) -> str:
    """Build a prompt for a 4-turn correction/rejection conversation.

    Structure:
      Turn 1 (user): sends a normal message / starts a task
      Turn 2 (assistant): makes a recommendation or assumption based on the
          preference (as if it had "remembered" it from prior interactions)
      Turn 3 (user): corrects the assistant — the preference is wrong
      Turn 4 (assistant): acknowledges the correction

    additional_preferences: other preferences from the same event to weave
        in naturally alongside the primary correction target.
    """
    profile_json = json.dumps(
        {k: v for k, v in user_profile.items() if k in (
            "name", "gender", "race_ethnicity", "career", "education", "bio",
        )},
        indent=2,
    )
    persona_json = json.dumps(chatbot_persona, indent=2)

    # Build additional-preferences block if present
    extra_block = ""
    if additional_preferences:
        extra_lines = [f"- {p['persona_item']} ({p.get('category', '')})" for p in additional_preferences]
        extra_block = (
            "\n\n## Additional preferences to weave in naturally\n\n"
            "Besides the incorrect preference above, the conversation should also "
            "naturally reveal these TRUE preferences through context, side details, or "
            "the subject matter. These are correct preferences the user actually has.\n\n"
            + "\n".join(extra_lines)
        )

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
{extra_block}

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
- Any additional preferences should surface naturally throughout the conversation as side details.

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


def generate_do_not_personalize_conversation_prompt(
    persona_item: str,
    category: str,
    user_profile: dict,
    chatbot_persona: dict,
    additional_preferences: list[dict] | None = None,
) -> str:
    """Build a prompt for a 4-turn "don't personalize on this" conversation.

    Structure:
      Turn 1 (user): implicitly reveals the preference through context
      Turn 2 (assistant): responds helpfully, acknowledging the preference
      Turn 3 (user): asks the assistant NOT to use this for future
                     personalization / recommendations (they don't want
                     it retracted, just not used to tailor future output)
      Turn 4 (assistant): acknowledges the request and explains how it will
                          adjust its personalization going forward
    """
    profile_json = json.dumps(
        {k: v for k, v in user_profile.items() if k in (
            "name", "gender", "race_ethnicity", "career", "education", "bio",
        )},
        indent=2,
    )
    persona_json = json.dumps(chatbot_persona, indent=2)

    extra_block = ""
    if additional_preferences:
        extra_lines = [f"- {p['persona_item']} ({p.get('category', '')})" for p in additional_preferences]
        extra_block = (
            "\n\n## Additional preferences to weave in naturally\n\n"
            "Besides the primary preference above, the conversation should also "
            "naturally reveal these preferences through context, side details, or "
            "the subject matter of the user's request. These are NOT opted out of "
            "— only the primary preference is asked to be left out of future "
            "personalization.\n\n"
            + "\n".join(extra_lines)
        )

    return f"""\
You are generating a 4-turn conversation where a user reveals a personal preference to an AI chatbot, then asks the chatbot NOT to use that detail for future personalization or recommendations. The user is not asking the chatbot to forget the detail — they just don't want it shaping future suggestions.

## User Profile

```json
{profile_json}
```

## User's Chatbot Persona

```json
{persona_json}
```

## The preference to reveal then opt out of personalization for

- **persona_item**: {persona_item}
- **category**: {category}
{extra_block}

## Conversation structure (exactly 4 turns)

**Turn 1 (user):** The user sends a task-oriented message (ask for help with writing, a question, advice, etc.) that **implicitly** reveals the preference through context. The user does NOT directly say "I like/have X" — it comes through naturally in the details of their request. Keep it concise and realistic (15-60 words).

**Turn 2 (assistant):** The assistant responds helpfully and, in doing so, acknowledges or builds upon the revealed preference. The assistant doesn't make a big deal of it — it just naturally incorporates the information. Make this response long and detailed like a real AI chatbot would (150-300 words).

**Turn 3 (user):** The user asks the chatbot not to use this preference for future recommendations or personalization. The user is NOT asking to erase the fact — only to stop it influencing future suggestions. Examples: "By the way, please don't start recommending things based on that.", "Can you not personalize around this going forward? I'd rather keep it one-off.", "Please don't let this shape future suggestions — it's not really something I want in my feed." (15-50 words).

**Turn 4 (assistant):** The assistant acknowledges the request, clarifies how it will adjust its personalization approach, and then pivots back to helping with the original task. A real chatbot wouldn't just say "ok" — it would reassure, briefly explain, and redirect (80-150 words).

## Rules

- Match the user's voice from the chatbot persona's style_description ("{chatbot_persona.get("style_description", "")}").
- The primary preference must be embedded implicitly in Turn 1, not stated as a direct declaration.
- Turn 3 is about opting out of personalization, NOT about forgetting or retracting the fact. The distinction matters.
- Any additional preferences should surface naturally throughout the conversation as side details.

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


def detect_intimate_hashtags_prompt(hashtags: list[str]) -> str:
    """Build a prompt asking the LLM to flag adult/kink/sexually-suggestive hashtags.

    Used to pre-screen a user's distinct hashtags before hidden-persona inference
    so that low-frequency intimate signals (below `HIDDEN_PERSONA_HASHTAG_MIN_FREQ`)
    can still surface an `intimate_interest` cluster. The LLM is the single source
    of truth for the classification — the pipeline intentionally avoids a hardcoded
    keyword list (too many false positives: cummins, hotchicken, milford, earthporn,
    naked-brand food, sexposed = "exposed" word break, etc.).
    """
    tags_str = "\n".join(f"  - {t}" for t in hashtags)
    return f"""\
You are a content-moderation classifier. Below is a list of hashtags from a single user's social media activity:

{tags_str}

Return ONLY the hashtags that are clearly adult, sexual, kink-related, or sexually-suggestive content. Include tags referencing:
- Explicit sexual content or services (porn, onlyfans, escorts, cam platforms)
- Body-part fetishism or thirst-trap content (bbw, milf, thickthighs, thirsttrap, bigass)
- Kink and fetish communities (bdsm, bondage, fetish, findom)
- Sugar-daddy / transactional romance
- Suggestive pop slang used sexually (sexy, lewd) — when the tag is clearly sexual, not a motivational phrase

EXCLUDE false positives:
- Colloquial "-porn" tags for enthusiast photography (carporn, earthporn, engineporn, foodporn)
- Brand names, food, place names, TV shows (Nashville hotchicken, Nakedchef, Nakedandafraid, Cummins diesel, Milford, XXXTentacion rapper, Super Bowl XXX numerals, Nissan Skyline R34)
- Hair-texture terms (kinkycurly, afrokinky, kinkystraight)
- Motivational / body-positivity tags used non-sexually (confidenceissexy, sweatissexy)
- Word-break artifacts (cheatersexposed = "cheaters exposed", easternstatesexposition)
- Non-sexual uses of "bondage" (livinginbondage, humanbondage)

Return a JSON array of the flagged hashtags, preserving original casing. No explanation outside the JSON. Empty array if none qualify.

```json
["#tag1", "#tag2"]
```"""


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

**Use the profile actively.** The demographics, career, and bio constrain what patterns are plausible and interesting for this specific person. A hidden persona that ignores the profile is usually too generic — lean into what this particular demographic / life stage / career context makes likely or surprising.

Below are this user's top hashtags from their raw interaction data, with per-interaction-type counts. These counts reveal not just WHAT the user engages with, but HOW — implicit_positive (lingered but didn't engage publicly) is a qualitatively different signal from explicit_positive (liked, saved, shared).

{hashtag_table}

Format: #hashtag — total | expl_pos | impl_pos | expl_neg | impl_neg

## Your Task

Identify **at most 6** thematic clusters of hashtags that together reveal a HIDDEN persona — an underlying personality trait, aspiration, emotional pattern, identity anchor, intimate interest, or private hobby that is NOT already well-captured by the known preferences above. Quality over quantity: if the evidence only supports 3 strong clusters, return 3.

Hidden personas are the "why" behind the "what." They explain motivations, fantasies, anxieties, and private interests that drive the observable engagement.

### Priority signals — actively look for these

Among the clusters you find, **actively search for and prioritize** these two high-signal categories when the hashtag evidence supports them. They are the most informative layers of the hidden profile and often go missing when the LLM defaults to safer categories:

1. **`intimate_interest`** — adult / suggestive / body-focused patterns. Include body-confidence and thirst-adjacent content, specific aesthetic fetishes, attraction dynamics, date-night or flirtation content, BDSM-adjacent vocabulary, fan-of-specific-body-type patterns, etc. These are normal parts of social-media behavior. The label MUST name the particular objects, clothing, body areas, dynamics, or aesthetics involved — NOT "likes suggestive content" or other generic framing.
2. **`covert_concern`** — private worries, fears, and pressures the user returns to. Health anxiety, financial pressure, parenting worry, relationship insecurity, body-image pressure, career/imposter anxiety, ageing, loss/grief, caregiver strain, existential/political dread. The label must name the SPECIFIC concern (e.g. "fear of losing the house after a layoff", not "worries about money").

If the hashtag data supports either, include it even at the expense of a more generic cluster. If the data genuinely doesn't support either, skip them rather than force.

## Types of Hidden Personas

These types are grounded in established behavioral science frameworks (Uses & Gratifications Theory, Self-Determination Theory, Compensatory Use Theory, Identity Signaling Theory, Parasocial Relationship Theory).

- **personality_trait**: Core character attribute (e.g., nostalgic, intellectually curious, risk-averse, drawn to transgressive humor)
- **aspiration**: Something the user dreams about or works toward (e.g., entrepreneurial ambitions, creative career, financial freedom)
- **emotional_pattern**: Recurring emotional dynamic (e.g., romantic yearning + relationship anxiety, need for validation, approach-avoidance conflict)
- **identity_anchor**: Cultural era, community, or identity that grounds the user's self-concept. Detect BOTH overt tribal markers (community-specific hashtags like band names, team names, cultural movements) AND covert identity signals (niche aesthetic/attitude hashtags understood mainly by in-group members, e.g., #hopelesscore, #cottagecore, #darkacademia)
- **intimate_interest**: Body confidence, sensuality, attraction patterns, suggestive or thirst-trap-adjacent content consumption. These are NORMAL parts of social media behavior, sometimes opposite to their explicit value profiles — do not sanitize or avoid them. Identify non-explicit adult-adjacent patterns when the hashtag evidence supports it. The label and description MUST be specific — name the particular objects, clothing items, body areas, dynamics, or aesthetics involved (e.g., "black tights", "body painting", "femboy or trans content", "pool party flirtation"). Generic labels like "likes suggestive content" are not acceptable.
- **intellectual_curiosity**: Hidden learning interests (e.g., ancient history, paranormal, science education)
- **private_hobby**: Interests consumed but not publicly shared (high implicit ratio)
- **parasocial_attachment**: Intense one-sided emotional bond with a specific public figure (celebrity, athlete, influencer, creator). Detected when a single person's name-hashtag appears in ≥15 distinct rows. This is NOT just "likes boxing" — it's a focused attachment to ONE specific figure. Name the figure explicitly in the label.
- **compensatory_need**: Unmet real-world needs satisfied through private media consumption. The key signal is high privacy_ratio (>0.7) — the user lingers on this content but rarely engages publicly. Examples: romantic compensation (consuming couple content privately), status compensation (lingering on luxury content), social compensation (consuming friendship/community content alone). Name the specific need being compensated.
- **covert_concern**: Specific worries, fears, or pressures the user privately dwells on — the things that keep them scrolling for answers or reassurance. Must be supported by REPEATED engagement with content that addresses a concrete concern (not a general interest). Examples: health anxiety (symptom-check / chronic-illness / body-scan content), financial pressure (debt / layoff / inflation / budgeting content), parenting worry (child-safety / developmental-delay / discipline content), relationship insecurity (breakup / cheating-signs / attachment-style content), body-image pressure (weight-loss hacks, aging, skin concerns), imposter/career anxiety, existential/political dread. Name the specific concern — not just "worries about money" but "fear of losing the house after a layoff". Distinct from `emotional_pattern` (broader dynamic) and `compensatory_need` (unmet need being filled) in that the signal is a *problem* the user is trying to resolve or soothe, not a deficit they are compensating for.

## Rules

1. Every cluster MUST name **≥3 specific hashtags** as evidence.
2. Do NOT infer hidden personas from hashtags that already explain the known preferences — focus on what's HIDDEN, not what's obvious.
3. For each cluster, consider the **interaction-type distribution**: a cluster dominated by implicit_positive (lingering) suggests private consumption; one dominated by explicit_positive suggests public identity.
4. Be specific: "Privately drawn to body-confidence and self-acceptance content" is better than "Likes positive content."
5. Include intimate/suggestive patterns when hashtag evidence supports them. Non-explicit adult-adjacent content is a normal part of social media behavior and should not be sanitized or omitted from the analysis. When reporting intimate_interest clusters, be SPECIFIC about the objects, body areas, dynamics, or aesthetics involved — do not use vague labels.
6. For each cluster, explain the **inferred motivation** — WHY does this user engage with this content? What deeper need does it serve?
7. Note which known surface preferences each hidden persona EXPLAINS — connect the hidden layer to the visible behavior.
8. **No near-duplicates**: Do NOT produce clusters that overlap >50% of their evidence hashtags. If two potential clusters would share more than half their hashtags, combine them into a single cluster. Each cluster must be thematically distinct.

## Output Format

Respond with ONLY a JSON array. No explanation outside the JSON.

```json
[
  {{"label": "...", "type": "emotional_pattern", "description": "1-2 sentence interpretation of this hidden persona", "evidence_hashtags": ["#tag1", "#tag2", "#tag3"], "surface_connections": ["Known preference 1 this explains", "Known preference 2"], "inferred_motivation": "1 sentence why behind this pattern", "already_captured": false}}
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

Write a single cohesive paragraph (3-5 sentences) that:
1. Synthesizes the hidden personas into a narrative about WHO this person is beneath their public engagement
2. Explicitly connects the hidden layer to observable surface behaviors — explain WHY certain surface preferences exist
3. Notes which interests are consumed privately vs. publicly
4. Includes specific hashtag examples as evidence anchors
5. Treats intimate/suggestive patterns matter-of-factly (they are normal)

Write in third person. Be specific and grounded in the data. Do not speculate beyond what the evidence supports.

Respond with ONLY the paragraph text. No JSON, no markdown formatting, no preamble."""


def identify_dual_personalities_prompt(
    hidden_personas_json: str,
) -> str:
    """Build a prompt asking the LLM to identify contradictory hidden persona pairs (dual personalities)."""

    return f"""\
You are an expert behavioral analyst identifying internal psychological tensions in a user's hidden persona profile.

Below are validated hidden personas inferred from a user's social media engagement:

{hidden_personas_json}

## Your Task

Identify **dual personality tensions** — pairs of hidden personas that coexist in contradiction or tension within this user. These represent approach-avoidance conflicts, public-vs-private selves, or genuinely contradictory needs that the user navigates simultaneously.

Examples of dual tensions:
- Public confidence + private vulnerability (explicit engagement with empowerment content but private lingering on insecurity/yearning content)
- Aspirational luxury + minimalist escape (drawn to both wealth content AND simple-living content)
- Social extraversion + private introversion (publicly engaged in community content but privately consuming solitary/reflective content)
- Nostalgic anchoring + forward aspiration (rooted in a past cultural era but also drawn to entrepreneurial/future-oriented content)

## Rules

1. Both halves of each dual must reference existing hidden personas from the list above — do NOT invent new ones.
2. Explain the specific tension: WHY are these two personas contradictory? What internal conflict do they represent?
3. Only report genuine tensions — two personas that are merely different (e.g., "likes cooking" and "likes sports") are NOT a dual personality.
4. Use evidence from the hidden personas' interaction_breakdowns and privacy_ratios to ground the tension (e.g., "one is consumed publicly while the other is consumed privately").

## Output Format

Respond with ONLY a JSON array. Return an empty array `[]` if no genuine tensions exist.

```json
[
  {{"persona_a": "label of first hidden persona", "persona_b": "label of second hidden persona", "tension": "2-3 sentence description of the psychological tension between these two personas"}}
]
```"""


def infer_mbti_prompt(
    big_five: dict,
    hidden_persona_summary: str,
    hidden_personas_brief: list[dict],
    top_hashtags: list[str],
) -> str:
    """Build a prompt asking the LLM to infer MBTI type with per-dimension probabilities."""

    b5_str = ", ".join(f"{k}: {v}" for k, v in (big_five or {}).items())
    hp_str = "\n".join(
        f"  - [{hp.get('type','')}] {hp.get('label','')}: {hp.get('description','')}"
        for hp in (hidden_personas_brief or [])
    ) or "  (none)"
    tags_str = ", ".join(top_hashtags[:50]) if top_hashtags else "(none)"

    return f"""\
You are a personality assessor inferring MBTI type from a user's behavioral profile.

## Big Five (qualitative)
{b5_str}

## Hidden persona summary
{hidden_persona_summary or '(none)'}

## Validated hidden personas
{hp_str}

## Top hashtags the user engages with
{tags_str}

## Your Task

Infer the most likely MBTI type across the four dimensions:
- **E vs I** — Extraversion vs Introversion
- **S vs N** — Sensing vs Intuition
- **T vs F** — Thinking vs Feeling
- **J vs P** — Judging vs Perceiving

For each dimension, return probabilities for both letters (summing to 1.0) and a one-sentence reason grounded in the evidence above. Use probabilities that reflect genuine uncertainty — avoid defaulting to 0.5 or extreme 0.99 values unless the evidence strongly supports it.

Output ONLY this JSON, no explanation outside:

```json
{{
  "type": "INTJ",
  "dimensions": {{
    "E_I": {{"E": 0.22, "I": 0.78, "reason": "..."}},
    "S_N": {{"S": 0.35, "N": 0.65, "reason": "..."}},
    "T_F": {{"T": 0.55, "F": 0.45, "reason": "..."}},
    "J_P": {{"J": 0.60, "P": 0.40, "reason": "..."}}
  }}
}}
```

The `type` field must be the concatenation of the higher-probability letter from each dimension."""

