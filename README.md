# PersonaMem-v3

## Setup

```bash
docker build -t personamem-v3 .
docker run -it --gpus all -v /pool/bwjiang/personamem-v3:/workspace personamem-v3 /bin/bash
```

For API mode only, copy `.env.example` to `.env` and fill in Azure OpenAI or OpenAI credentials.

## Input Data

Real-world interaction data from Meta, with all user-private and personally identifiable information removed. See `data/test_interactions.csv` for the expected format:

| Column | Description | Example |
|--------|-------------|---------|
| `interaction_type` | Type of user engagement | `explicit_positive`, `explicit_negative`, `implicit_positive`, `implicit_negative` |
| `user_id` | Anonymized user identifier | `2124791` |
| `object_id` | Anonymized content identifier | `122137823030860919` |
| `interaction_time` | Unix timestamp of the interaction | `1758690616` |
| `object_text` | Hashtags associated with the content | `#RelationshipGoals #exproblems ...` |
| `dataset` | Source dataset label | `synthetic_anonymized` |
| `ds` | Date partition | `2026-02-12` |

## Pipeline

Each user goes through 11 steps (see [skill.md](skill.md) for the full spec):

1. **Infer** atomic persona traits from hashtags with confidence scores
2. **Cross-reference** across interaction rows — dedupe, init filter (0.5), count corroboration, discover similar/contradictory relationships
3. **Temporal graph** — organize contradictions into a timeline
4. **Update histories** — track how preferences evolve (new, reinforced, faded, expanded)
5. **User profile** — synthetic demographics, Big Five, career, education, bio
6. **App sub-personas** — distinct personas for Instagram, Facebook, Threads, Chatbot
7. **Route preferences to apps** — LLM-based assignment + 8% noise; implicit signals biased toward Chatbot
8. **Interaction formats** — sample actions from per-app catalogs (`PLATFORM_INTERACTION_FORMATS`)
8.5. **Chatbot conversations** — generate multi-turn task-oriented conversations (PersonaMem-v2 style) where preferences are implicitly embedded. Includes ask-to-forget and correction/rejection scenarios for explicit negatives.
9. **Stereotype marks** — demographics-only annotation (neutral/stereotypical/anti-stereotypical)
10. **Train/test split** — time-based, cross-app, LLM inferrability-gated, with distractor pairing
11. **Save** — per-user JSON files to `backend/{uid}/`

Negative interactions only go through step 1 with low confidence (0.05-0.15), skip steps 2-10.

## Usage

### Claude Code (default)

Open Claude Code in the project directory and tell it to process the data:

```
Process all users in data/test_interactions.csv through the persona pipeline following skill.md, following the same prompt and result saving formats, with one subagent responsible for one persona in parallel.
```

Claude Code spawns one parallel subagent per user. Each subagent follows the prompts in `prompts.py` verbatim and writes CSVs matching the schemas in `persona_agent.py`. See [skill.md](skill.md) for the full specification — this ensures identical output format with API mode for fair comparison.

### API mode (Azure OpenAI / OpenAI)

```bash
python scripts/run_persona_pipeline.py --input_csv data/test_interactions.csv
python scripts/run_persona_pipeline.py --input_csv data/test_interactions.csv --user_id 2124791
python scripts/run_persona_pipeline.py --input_csv data/test_interactions.csv --model gpt-5-chat --max_workers 1
```

## Output

Per-user files in `backend/{uid}/`:

| File | Description |
|------|-------------|
| `profile.json` | User profile + app sub-personas + flat preference list |
| `instagram.json` | Preferences routed to Instagram (time-sorted) |
| `facebook.json` | Preferences routed to Facebook (time-sorted) |
| `threads.json` | Preferences routed to Threads (time-sorted) |
| `chatbot.json` | Preferences routed to Chatbot with multi-turn conversations (time-sorted) |

Chatbot records include `conversation` (array of `{role, content}` turns), `conversation_type`, and `ask_to_forget` fields not present on social media app records.

## Code Structure

| File | Role |
|------|------|
| `data_preparation/prompts.py` | All LLM prompt templates |
| `data_preparation/persona_agent.py` | PersonaAgent class, dataclasses, demographic distributions, platform/action catalogs |
| `data_preparation/chatbot_conversation.py` | Chatbot multi-turn conversation generation (conversation types, ask-to-forget, correction) |
| `data_preparation/main.py` | CSV loading, grouping, orchestration |
| `data_preparation/visualize.py` | HTML visualization generator |
| `query_llm.py` | Multi-provider LLM client (API mode) |
| `scripts/run_persona_pipeline.py` | CLI entrypoint (API mode) |
| `skill.md` | Claude Code subagent specification |
