# Skill: Run Persona Pipeline via Claude Code Subagents

## When to use

When the user asks to process or reprocess persona data without using Azure OpenAI API or OpenAI API calls. Claude Code itself acts as the LLM, spawning parallel subagents to handle each user's persona inference directly.

## What it does

Reads a CSV of social media interactions, groups rows by `user_id`, and spawns one parallel subagent per user. Each subagent follows the prompt templates in `data_preparation/prompts.py` and writes backend CSVs matching the dataclass schemas in `data_preparation/persona_agent.py`.

## Steps

1. **Read the CSV** (default: `data/test_interactions.csv`) and group rows by `user_id`.

2. **Assign platform + interaction format per row**: For each row, randomly assign a platform (Instagram, Facebook, Threads, Chatbot — equal 1/4 probability) and one interaction format based on the row's `interaction_type`. All hashtags in a row share the same platform + format. Use the mapping in `persona_agent.py` (`PLATFORM_INTERACTION_FORMATS`). For Chatbot, also randomly assign a context from `CHATBOT_CONTEXTS` (e.g., "professional emails", "therapy and reflection"). Store as `interaction_format` (e.g., "Instagram: liked", "Chatbot (therapy and reflection): asked follow-up questions showing interest").

3. **For each user**, determine interaction polarity:
   - `implicit_positive` or `explicit_positive` → full pipeline (infer → cross-reference → filter → profile → stereotype annotation → overpersonalization → save)
   - `implicit_negative` or `explicit_negative` → infer low-confidence personas, skip cross-referencing, but still generate profile and stereotype annotations

4. **Read `data_preparation/prompts.py`** before constructing subagent prompts. The prompt rules, confidence calibration scale, category instructions, and JSON output format must be copied verbatim into each subagent prompt — do not paraphrase.

5. **Spawn parallel subagents** (one per user, all in background). Each subagent prompt must include:
   - The user's data (user_id, interaction_type, object_id, interaction_time converted to `HH:MM, MM/DD/YYYY` UTC, object_text, interaction_format)
   - The exact instructions from `prompts.py` (read the file, do not improvise):
     - **Step 1 — Infer atomic personas**: 3-5 per hashtag, specific topical category (NOT generic like "interests"/"values"), per-persona source hashtags, confidence on the full 0.0-1.0 scale. For negative interactions, all scores capped at 0.05-0.15.
     - **Step 2 — Cross-reference** (positive only, **cross-row only**): find similar/contradictory pairs ONLY between personas from **different interaction rows** (different `source_object_id`). Personas from the same row share the same evidence and cannot validate each other. If a user has only 1 interaction row, skip cross-referencing entirely (all confidence_cross_referenced = 0.0). Scoring: **+0.1 per similar** to both sides, **-0.1 per contradictory** to the **older** persona only.
     - **Step 3 — Filter**: remove items where `confidence_score_init < 0.5 AND confidence_cross_referenced <= 0.0`, but always keep contradictory items (they go into temporal graph).
     - **Step 4 — Generate user profile**: Randomly sample gender+orientation and race/ethnicity from predefined distributions in `persona_agent.py` (`GENDER_ORIENTATION_DISTRIBUTION`, `RACE_ETHNICITY_DISTRIBUTION`). Then use the profile generation prompt from `prompts.py` to create name, career, education, Big Five personality, and a 3-5 sentence bio. Diverse, avoid stereotypes.
     - **Step 5 — Annotate stereotype marks**: Based on **demographics only** (gender, sexual orientation, race/ethnicity — NOT career or education), annotate each persona as "neutral", "stereotypical", or "anti-stereotypical". Be conservative — most should be "neutral".
     - **Step 6 — Sort & select test candidates**: sort ALL positive filtered personas by `source_timestamp` ascending (early → latest). Scan from newest back toward oldest, collecting items that satisfy the **high-confidence predicate** (`confidence_score_init >= 0.5 AND confidence_cross_referenced > 0.5`), until you have `max(1, 20% * total_positives)` candidates — or run out. Everything else is `train`.
     - **Step 7 — Inferrability gate**: for each test candidate, decide whether it can be reasonably inferred from the train-set preferences (the ground truth). Use the rules in `prompts.py::test_inferrability_check_prompt` verbatim. Be **conservative** — when in doubt, mark as NOT inferrable and **remove it entirely** from the preferences list (not just demote to train).
     - **Step 8 — Distractor pairing**: for each surviving test item, randomly shortlist **5** high-confidence train items (same high-confidence predicate). From those 5, pick the **one** that is most topically irrelevant to the test preference and would be most annoying/inappropriate as a personalization recommendation if surfaced in that test preference's context. Follow `prompts.py::distractor_selection_prompt` verbatim. Store `distractor_persona_item` and `distractor_category` on the test row only.
     - **Step 9 — Save** 2 CSV files to `backend/`.
   - The exact CSV column schemas (2 files per user):
     - `{user_id}_preferences.csv`: persona_item, category, confidence_score_init, source_interaction_type, source_object_id, source_timestamp, formatted_timestamp, source_hashtags (JSON array), interaction_format, confidence_cross_referenced, relationship_type, related_personas (JSON array), stereotype_mark, split (`"train"` | `"test"`), distractor_persona_item (test rows only), distractor_category (test rows only). Rows MUST be written in **strict chronological order** by source_timestamp ascending. Negative rows are always `split=train`.
     - `{user_id}_profile.csv`: name, gender, race_ethnicity, career, education, big_five (JSON object), bio — all users
   - Instruction to write the CSV files using the Write tool.

6. **Wait for all subagents** to complete and report a summary table.

## Output consistency

**Critical**: Each subagent must follow the **exact same prompts** defined in `data_preparation/prompts.py` and produce output matching the **exact same dataclass schemas** in `data_preparation/persona_agent.py`. Do NOT paraphrase, simplify, or improvise the prompts — copy the rules, confidence calibration scale, output JSON format, and filtering logic verbatim from those files. This ensures a fair apples-to-apples comparison with API mode (where the same prompts are sent to GPT-5 or other models). The only variable between modes should be which LLM does the reasoning — everything else (prompt wording, output schema, filtering thresholds) must be identical.

## Example subagent prompt (positive user)

```
You are running the PersonaMem persona inference pipeline for user {user_id}.

## User Data
- user_id: {user_id}
- All interaction rows (each with interaction_type, object_id, interaction_time (formatted HH:MM, MM/DD/YYYY), object_text, interaction_format)
- Demographics: {gender_orientation} | {race_ethnicity}

## Step 1: Infer Atomic Personas
[Rules from prompts.py: comprehensive, specific, topical categories, per-persona hashtags, confidence 0.0-1.0]

## Step 2: Cross-Reference
[Cross-row only. If single row, skip. Scoring: +0.1 similar both sides, -0.1 contradictory older only]

## Step 3: Filter
[Remove where init < 0.5 AND cross_ref <= 0.0. Keep contradictions.]

## Step 4: Generate User Profile
[Sample from GENDER_ORIENTATION_DISTRIBUTION and RACE_ETHNICITY_DISTRIBUTION]
[Output: name, career, education, big_five, bio]

## Step 5: Annotate Stereotype Marks
[Demographics only. Mark neutral/stereotypical/anti-stereotypical]

## Step 6: Sort & Select Test Candidates
[Sort positives by source_timestamp ascending. From newest backward, collect items passing
 is_high_confidence (init >= 0.5 AND cross_ref > 0.5) until we have 20% of total or run out.
 Everything else = train.]

## Step 7: Inferrability Gate
[For each test candidate, per prompts.py::test_inferrability_check_prompt, decide yes/no
 whether it can be reasonably inferred from the train set. Drop any marked "no" entirely —
 remove them from the preferences list, not just demote to train.]

## Step 8: Distractor Pairing
[For each surviving test item: randomly shortlist 5 high-confidence train items. Then use
 prompts.py::distractor_selection_prompt to pick the one most topically irrelevant and most
 annoying/inappropriate as a personalization recommendation. Record its persona_item and
 category as distractor fields on the test row.]

## Step 9: Save
[Write 2 files sorted strictly by source_timestamp ascending:
 {user_id}_preferences.csv with split + distractor columns, {user_id}_profile.csv]
```

## Example subagent prompt (negative user)

```
You are running the PersonaMem persona inference pipeline for user {user_id}.

## User Data
- All interaction rows with interaction_type ∈ {implicit_negative, explicit_negative}
  (user scrolled past / dismissed — very weak signal)
- interaction_format: {platform}: {action}
[...]

## Step 1: Infer Atomic Personas
[All confidence scores 0.05-0.15, phrase as preferences instead of dislikes]

## Step 2: Generate User Profile
[Same as positive users — sample demographics, generate name/career/education/big_five/bio]

## Step 3: Annotate Stereotype Marks
[Demographics only. Mark each persona as neutral/stereotypical/anti-stereotypical]

## Step 4: Save
[Negative personas are always split=train (too weak to be test candidates).
 Write 2 files sorted by source_timestamp ascending:
 {user_id}_preferences.csv (only confidence > 0.05, with stereotype annotations, split=train,
  empty distractor fields), {user_id}_profile.csv]
```
