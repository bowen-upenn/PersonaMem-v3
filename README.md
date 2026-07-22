# PersonaMem-v3

Cross-platform personalization benchmark. Three commands take you from
raw interaction logs to evaluation results.

## Setup

```bash
docker build -t personamem-v3 .
docker run -it --gpus all -v /pool/bwjiang/personamem-v3:/workspace personamem-v3 /bin/bash
```

For API mode, copy `.env.example` to `.env` and fill in your Azure OpenAI
or OpenAI credentials.

## Input

Place your raw interaction CSV at `data/test_interactions.csv`. Required
columns:

| Column | Example |
|---|---|
| `interaction_type` | `explicit_positive`, `implicit_negative`, … |
| `user_id` | `2124791` |
| `object_id` | `122137823030860919` |
| `interaction_time` | unix timestamp |
| `object_text` | `#RelationshipGoals #exproblems …` |

## The three commands

### 1. Generate persona data

```bash
python scripts/run_persona_pipeline.py --input_csv data/test_interactions.csv
```

Reads `data/test_interactions.csv`, runs the 28-step persona pipeline
(see [skill.md](skill.md) for the spec), writes per-user files to
`backend/{uid}/`:

```
backend/{uid}/
├── profile.json      demographics + Big Five + sub-personas + flat preferences
├── instagram.json    interaction events with nested preferences
├── facebook.json
├── threads.json
├── chatbot.json      events + multi-turn conversations
├── calendar.json     calendar add/update/remove stream
└── persona.html      self-contained interactive visualization
```

### 2. Generate eval data

```bash
python scripts/prepare_eval_data.py --all
```

One command, every artifact. For each user with persona data, this
writes the frozen test queries (~220 across 27 task types), a JSON
dump for human review, a re-rendered HTML, and a structural audit:

```
benchmark/{uid}/queries.csv             frozen test queries (eval input)
backend/{uid}/test.json                 same queries in human-readable JSON
backend/{uid}/persona.html              re-rendered with the new test cards
backend/{uid}/test_audit_snapshot.md    audit report (realism, label honesty,
                                        distribution coverage)
```

No follow-up commands needed — this scales cleanly to 1000+ personas.

### 3. Run the evaluation

```bash
scripts/run_eval_all.sh --mode mcp_agent
```

Runs the eval harness across every persona. Output per persona:
`benchmark/{uid}/runs/{timestamp}/results.csv` plus a summary.

`--mode` chooses the agent setup. The modes isolate which
component (framework, retrieval, structured API) drives performance:

| Mode | What it tests |
|---|---|
| `mcp_agent` | Claude Code agent + structured per-app MCP tools — closest to a real app integration |
| `agent_tools` | Claude Code agent reading time-masked filesystem snapshots |
| `codex_agent` | Codex CLI agent (`codex exec --model gpt-5.5`) reading the same time-masked filesystem snapshots |
| `llm_longctx` | Direct LLM call (no agent framework) — pure long-context baseline |

Single-persona variant: `scripts/run_eval.sh 115 --mode mcp_agent`.
Codex agent variant over the configured cohort: `scripts/run_codex_agent.sh`.

## Aggregating results

```bash
python scripts/aggregate_eval.py
```

Writes `benchmark/_aggregate/` with per-task accuracy and the
macro/micro headline numbers across all personas and modes.

## More

- [skill.md](skill.md) — full persona-pipeline specification (28 steps; final step seeds proactive-agent triggers).
- [EVAL.md](EVAL.md) — task families, rubric dimensions, metric definitions.
