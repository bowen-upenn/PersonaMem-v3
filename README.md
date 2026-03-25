# PersonaMem-v3

Infers user personas from social media interaction data. Given a CSV of user-hashtag interactions, the system back-engineers atomic persona traits per user, cross-references them, and tracks how preferences change over time.

## Setup

```bash
docker build -t personamem-v3 .
docker run -it --gpus all -v /pool/bwjiang/personamem-v3:/workspace personamem-v3 /bin/bash
```

For API mode only, copy `.env.example` to `.env` and fill in Azure OpenAI or OpenAI credentials.

## Pipeline

Each user goes through 3 LLM calls:

1. **Infer** — extract hashtags, guess atomic personas with confidence scores and topical categories
2. **Cross-reference** — find similar/contradictory pairs, boost confidence for corroborated ones, filter out weak isolated guesses
3. **Temporal graph** — organize contradictions into a timeline showing how preferences shifted

Negative interactions (`implicit_negative`) skip steps 2-3 and are saved separately with low confidence.

## Usage

### Claude Code (default)

Open Claude Code in the project directory and tell it to process the data:

```
Process all users in data/test_interactions.csv through the persona pipeline
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
| `{uid}_atomic.csv` | Raw inferred personas (positive interactions) |
| `{uid}_negative.csv` | Negative-interaction personas (standalone, low confidence) |
| `{uid}_cross_referenced.csv` | Cross-referenced and filtered personas |
| `{uid}_temporal.csv` | Temporal contradiction timeline |
| `{uid}_persona.html` | HTML visualization |

## Code Structure

| File | Role |
|------|------|
| `data_preparation/prompts.py` | All LLM prompt templates |
| `data_preparation/persona_agent.py` | PersonaAgent class + dataclasses |
| `data_preparation/main.py` | CSV loading, grouping, orchestration |
| `data_preparation/visualize.py` | HTML visualization generator |
| `query_llm.py` | Multi-provider LLM client (API mode) |
| `scripts/run_persona_pipeline.py` | CLI entrypoint (API mode) |
| `skill.md` | Claude Code subagent specification |
