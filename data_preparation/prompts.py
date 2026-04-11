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
    """Build a prompt that asks the LLM to cross-reference and score all atomic personas.

    NOTE: The input list has ALREADY been deduplicated by the caller. Each entry
    is a *canonical* persona — personas with identical text from multiple rows
    have been merged into one entry and their `corroboration_count` reflects how
    many independent rows produced them. The LLM must treat each entry as a
    unique preference.
    """

    personas_json = json.dumps(atomic_personas, indent=2)

    return f"""\
You are an expert at synthesizing behavioral signals into a coherent user profile.

Below is a list of canonical persona traits/preferences for a single user. These have already been deduplicated — if two interaction rows produced the exact same persona text, they were merged into one canonical entry, and `corroboration_count` records how many distinct rows contributed. Each entry includes the category, persona item, initial confidence score, corroboration count, and source interaction metadata.

```json
{personas_json}
```

## Your Task — Find cross-persona relationships

1. **Cross-reference DISTINCT canonical personas** against each other. Because the input is already deduplicated, you will NEVER see two entries with identical `persona_item` text — so you must NEVER mark a persona as `similar` or `contradictory` to itself.
   - If two **different** personas are **similar** (reinforce each other — e.g. "Enjoys home cooking" and "Buys fresh produce weekly"), mark them as related with `"type": "similar"`.
   - If two **different** personas **contradict** each other (e.g. "Prefers vegan meals" and "Loves BBQ ribs"), mark them as related with `"type": "contradictory"`.
   - If a persona has no meaningful cross-persona relationship, set `relationship_type` to `"none"` with an empty `related_personas` list.

2. **Do NOT mark identical persona_items as similar to each other.** Identical-text preferences are the same preference corroborated by multiple rows — that corroboration is already captured in `corroboration_count` and scored upstream. Marking them as "similar" would double-count.

3. **Do NOT compute confidence scores** — the `confidence_cross_referenced` field is computed downstream from your relationships. Do not include it in your output.

4. **Keep the original `confidence_score_init` unchanged.**

5. **Return EVERY canonical persona** — one entry per input, even if its `relationship_type` is `"none"`.

6. **For each persona, list all related personas** in the `related_personas` array. Each entry must include the other persona's text AND its relationship type as an object: `{{"persona_item": "...", "type": "similar"}}` or `{{"persona_item": "...", "type": "contradictory"}}`.

7. **Preserve the `category`** from the input for each persona item.

## Output Format

Respond with ONLY a JSON array. No explanation.

```json
[
  {{
    "category": "...",
    "persona_item": "...",
    "confidence_score_init": 0.XX,
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


def remove_redundant_personas_prompt(candidates: list[dict]) -> str:
    """Build a prompt that asks the LLM to cluster semantically-redundant personas.

    The caller has already filtered the candidates to only those above the
    init-confidence floor. The LLM's job is to find GROUPS of personas that
    convey essentially the same preference in different words, so downstream
    code can keep one representative per group and drop the rest.
    """

    candidates_json = json.dumps(candidates, indent=2)

    return f"""\
You are reducing redundancy in a user's preference list. The entries below have already passed a confidence threshold, but some of them almost certainly describe the SAME underlying preference with different wording.

## Candidates

```json
{candidates_json}
```

## Your Task

Cluster the candidates into **redundancy groups**. Each group is a set of two or more persona_items that describe the SAME preference. Downstream code will keep the strongest one from each group and drop the rest.

## Rules

1. **Only group items that truly mean the same thing.** Err on the side of NOT grouping — if two personas are related but describe different aspects, keep them separate. Example:
   - ✅ Group together: `"Enjoys home cooking"` + `"Likes preparing meals at home"` (same preference, different wording)
   - ✅ Group together: `"Follows Detroit Lions"` + `"Supports Detroit NFL team"` (same preference)
   - ❌ Do NOT group: `"Enjoys home cooking"` + `"Owns multiple cast iron pans"` (related but distinct)
   - ❌ Do NOT group: `"Follows Detroit Lions"` + `"Is an NFL fan"` (the NFL one is broader)

2. **Every group has 2 or more items.** Singletons are not redundancies — they don't need to appear in the output.

3. **Each persona_item appears in AT MOST ONE group.** No overlapping clusters.

4. **Skip personas that have no redundant counterpart.** They simply stay as-is — just don't include them in your output.

5. **Return ONLY the groups that have redundancies.** Do not return every input persona.

## Output Format

Respond with ONLY a JSON array of arrays. Each inner array lists the persona_item strings of one redundancy group. No explanation outside the JSON fence.

```json
[
  ["persona_item A", "persona_item B", "persona_item C"],
  ["persona_item D", "persona_item E"]
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

