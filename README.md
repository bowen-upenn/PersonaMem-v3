# PersonaMem-v3

Infers user personas from social media interaction data. Given a CSV of user-hashtag interactions, the system back-engineers atomic persona traits per user, cross-references them across interactions, tracks how preferences change over time, and crafts synthetic user profile descriptions with stereotype and overpersonalization annotations.

## Setup

```bash
docker build -t personamem-v3 .
docker run -it --gpus all -v /pool/bwjiang/personamem-v3:/workspace personamem-v3 /bin/bash
```

For API mode only, copy `.env.example` to `.env` and fill in Azure OpenAI or OpenAI credentials.

## Pipeline

Each user goes through up to 5 steps:

1. **Infer** — associate all hashtags in each activity to a randomly assigned platform (Instagram, Facebook, Threads, or Chatbot) and user engagement format, then infer atomic persona traits with confidence scores (0.0-1.0) and topical categories
2. **Cross-reference** — find similar/contradictory pairs across different interaction rows (not within the same row), boost confidence for corroborated ones (+0.1 per similar), reduce confidence on older contradictory ones (-0.1), filter out weak isolated guesses (init < 0.5 and cross_ref <= 0.0)
3. **Temporal graph** — organize contradictions into a timeline showing how preferences shifted
4. **User profile** — generate a synthetic user description: name, career, education, Big Five personality, and a 3-5 sentence bio, with demographics (gender, sexual orientation, race/ethnicity) sampled from predefined distributions
5. **Annotate** — mark each preference as neutral, stereotypical, or anti-stereotypical based on demographics only, then randomly hold out 20% for overpersonalization study

Negative interactions (`implicit_negative`) only go through step 1 with low confidence (0.05-0.15), skip steps 2-5.

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
python scripts/run_persona_pipeline.py --input_csv data/test_interactions.csv --model gpt-4o --max_workers 1
```

## Output

Per-user files in `backend/`:

| File | Description |
|------|-------------|
| `{uid}_preferences.csv` | Filtered personas with confidence scores, cross-reference data, platform/interaction format, stereotype marks, and overpersonalization tags |
| `{uid}_profile.csv` | Synthetic user profile: name, gender, race/ethnicity, career, education, Big Five, bio (positive users only) |

## Code Structure

| File | Role |
|------|------|
| `data_preparation/prompts.py` | All LLM prompt templates |
| `data_preparation/persona_agent.py` | PersonaAgent class, dataclasses, demographic distributions, platform mappings |
| `data_preparation/main.py` | CSV loading, grouping, orchestration |
| `data_preparation/visualize.py` | HTML visualization generator |
| `query_llm.py` | Multi-provider LLM client (API mode) |
| `scripts/run_persona_pipeline.py` | CLI entrypoint (API mode) |
| `skill.md` | Claude Code subagent specification |
