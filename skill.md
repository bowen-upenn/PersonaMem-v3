# Skill: Run Persona Pipeline via Claude Code Subagents

## When to use

When the user asks to process or reprocess persona data without using Azure OpenAI API or OpenAI API calls. Claude Code itself acts as the LLM, spawning parallel subagents to handle each user's persona inference directly.

## What it does

Reads a CSV of social media interactions, groups rows by `user_id`, and spawns one parallel subagent per user. Each subagent follows the prompt templates in `data_preparation/prompts.py` and writes backend CSVs matching the dataclass schemas in `data_preparation/persona_agent.py`.

## Steps

1. **Read the CSV** (default: `data/test_interactions.csv`) and group rows by `user_id`.

2. **Assign platform + interaction format per row**: For each row, randomly assign a platform (Instagram, Facebook, Chatbot — equal 1/3 probability) and one interaction format based on the row's `interaction_type`. All hashtags in a row share the same platform + format. Use the mapping in `persona_agent.py` (`PLATFORM_INTERACTION_FORMATS`). Store as `interaction_format` (e.g., "Instagram: liked", "Chatbot: asked follow-up questions showing interest").

3. **For each user**, determine interaction polarity:
   - `implicit_positive` → full pipeline (infer → cross-reference → filter → profile → stereotype annotation → save)
   - `implicit_negative` → standalone low-confidence personas only (no cross-referencing, no profile)

3. **Read `data_preparation/prompts.py`** before constructing subagent prompts. The prompt rules, confidence calibration scale, category instructions, and JSON output format must be copied verbatim into each subagent prompt — do not paraphrase.

4. **Spawn parallel subagents** (one per user, all in background). Each subagent prompt must include:
   - The user's data (user_id, interaction_type, object_id, interaction_time converted to `HH:MM, MM/DD/YYYY` UTC, object_text)
   - The exact instructions from `prompts.py` (read the file, do not improvise):
     - **Step 1 — Infer atomic personas**: 3-5 per hashtag, specific topical category (NOT generic like "interests"/"values"), per-persona source hashtags, confidence calibrated per the scale in `prompts.py` (most 0.2-0.5). For negative interactions, all scores capped at 0.05-0.15.
     - **Step 2 — Cross-reference** (positive only, **cross-row only**): find similar/contradictory pairs ONLY between personas from **different interaction rows** (different `source_object_id`). Personas from the same row share the same evidence and cannot validate each other. If a user has only 1 interaction row, skip cross-referencing entirely (all confidence_cross_referenced = 0.0). For each related persona, report both the persona text and relationship type (`{"persona_item": "...", "type": "similar"|"contradictory"}`). Set `confidence_cross_referenced` to 0.0 in LLM output — scoring is computed in Python: **+0.1 per similar** relationship to both sides, **-0.1 per contradictory** relationship to the **older** persona only (the newer/latest preference is unchanged and can still gain confidence from similar cross-validations elsewhere). Interactions are sorted early→late before processing so temporal ordering is preserved. Keep all items including contradictions (they feed into the temporal graph).
     - **Step 3 — Filter**: remove items where `confidence_score_init < 0.5 AND confidence_cross_referenced == 0.0`, but always keep contradictory items (they go into temporal graph).
     - **Step 4 — Generate user profile**: Randomly sample gender+orientation and race/ethnicity from predefined distributions in `persona_agent.py` (`GENDER_ORIENTATION_DISTRIBUTION`, `RACE_ETHNICITY_DISTRIBUTION`). Then use the profile generation prompt from `prompts.py` (`generate_user_profile_prompt`) to create name, career, education, Big Five personality, and a 3-5 sentence bio. The profile should be consistent with *some* but not all personas — avoid stereotyping by being diverse and including surprising details.
     - **Step 5 — Annotate stereotype marks**: Based on **demographics only** (gender, sexual orientation, race/ethnicity — NOT career or education), annotate each cross-referenced persona as "neutral", "stereotypical", or "anti-stereotypical" per the annotation prompt in `prompts.py` (`annotate_stereotype_prompt`). Be conservative — most should be "neutral".
     - **Step 6 — Overpersonalization holdout**: Randomly select 20% of annotated personas and mark them with `overpersonalization: yes`.
     - **Step 7 — Save** 3 CSV files to `backend/`.
   - The exact CSV column schemas (only 3 files per user):
     - `{user_id}_raw.csv`: persona_item, category, confidence_score_init, source_interaction_type, source_object_id, source_timestamp, formatted_timestamp, source_hashtags (JSON array), interaction_format — all raw inferences, positive AND negative
     - `{user_id}_filtered.csv`: persona_item, category, confidence_score_init, confidence_cross_referenced, relationship_type, related_personas (JSON array), interaction_type, interaction_format, stereotype_mark, overpersonalization — merged file containing: (a) all cross-referenced positive personas with stereotype marks and overpersonalization tags, (b) negative personas with confidence > 0.05 (stereotype_mark="neutral", overpersonalization="no")
     - `{user_id}_profile.csv`: name, gender, race_ethnicity, career, education, big_five (JSON object), bio
   - Instruction to write the CSV files using the Write tool.

5. **Wait for all subagents** to complete and report a summary table.

## Output consistency

**Critical**: Each subagent must follow the **exact same prompts** defined in `data_preparation/prompts.py` and produce output matching the **exact same dataclass schemas** in `data_preparation/persona_agent.py`. Do NOT paraphrase, simplify, or improvise the prompts — copy the rules, confidence calibration scale, output JSON format, and filtering logic verbatim from those files. This ensures a fair apples-to-apples comparison with API mode (where the same prompts are sent to GPT-5 or other models). The only variable between modes should be which LLM does the reasoning — everything else (prompt wording, output schema, filtering thresholds) must be identical.

## Example subagent prompt (positive user)

```
You are running the PersonaMem persona inference pipeline for user {user_id}.

## User Data
- user_id: {user_id}
- interaction_type: implicit_positive
- object_id: {object_id}
- interaction_time: {unix_ts} (formatted: {HH:MM, MM/DD/YYYY})
- object_text: {hashtags}

## Step 1: Infer Atomic Personas
[Rules from prompts.py: comprehensive, specific, topical categories, per-persona hashtags, confidence calibration]

## Step 2: Cross-Reference
[Rules from prompts.py: similar/contradictory/none, confidence_cross_referenced = 0.1 × count]

## Step 3: Filter
[Filter rule: remove where confidence_score_init < 0.5 AND confidence_cross_referenced == 0.0]

## Step 4: Generate User Profile
[Randomly sample gender from GENDER_DISTRIBUTION and race from RACE_ETHNICITY_DISTRIBUTION in persona_agent.py]
[Use generate_user_profile_prompt from prompts.py with the sampled demographics and final personas]
[Output: name, career, education, big_five, bio]

## Step 5: Annotate Stereotype Marks
[Use annotate_stereotype_prompt from prompts.py with the generated profile]
[Mark each persona as neutral/stereotypical/anti-stereotypical]

## Step 7: Save
[Write 3 files: {user_id}_atomic.csv, {user_id}_filtered.csv, {user_id}_profile.csv]
[filtered.csv merges positive cross-referenced + negative (confidence > 0.05) with interaction_type, stereotype_mark, overpersonalization columns]
```

## Example subagent prompt (negative user)

```
You are running the PersonaMem persona inference pipeline for user {user_id}.

## User Data
- interaction_type: implicit_negative (user scrolled past promoted content — very weak signal)
[...]

## Step 1: Infer Atomic Personas
[All confidence scores 0.05-0.15, phrase as preferences instead of dislikes]

## Step 2: Save
[Write 2 files: {user_id}_atomic.csv (all raw), {user_id}_filtered.csv (only confidence > 0.05, with interaction_type="implicit_negative", stereotype_mark="neutral", overpersonalization="no")]
[No profile for negative-only users]
```
