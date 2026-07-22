# PersonaMem-v3: Toward Omni-Platform Personal Intelligence for Holistic User Understanding, Recommendation, and Agentic Tasks

Third release in the PersonaMem series:

- **PersonaMem (v1)** — [COLM 2025] *Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale* · [code](https://github.com/bowen-upenn/PersonaMem) · [paper](https://arxiv.org/abs/2504.14225)
- **PersonaMem-v2** — *Towards Personalized Intelligence via Learning Implicit User Personas and Agentic Memory* · [code](https://github.com/bowen-upenn/PersonaMem-v2) · [paper](https://arxiv.org/abs/2512.06688)

## What's new in v3

| Dimension | PersonaMem-v1 | PersonaMem-v2 | PersonaMem-v3 |
|---|---|---|---|
| **Data source** | 20 fully synthetic users | 1000 fully synthetic users with more comprehensive personas | 200 anonymized **real-world** users with 4,000,000 engagement histories |
| **Explicit vs. implicit** | Explicit user preferences | **Implicit** user preferences | Around 95% **implicit** user behavior signals |
| **Scenarios** | Chatbot conversations | Chatbot conversations | **Omni-platform**, including chatbot, social media recommendation, **agentic tasks**, and proactiveness |
| **Restraint** | Personalization | Personalization | Personalization and **over-personalization** |
| **User privacy** | No mentioning of user private information | Including personally identifiable information and user-initiated ask-to-forget scenarios | Including psychology-anchored hidden persona and **socially inappropriate** scenarios |
| **Dynamics** | Fully synthesized preference updates | Fully synthesized preference updates | Reinforced, emerging, diminishing, bursting, and varied attention shifts from the **real world** |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in Azure OpenAI / Gemini credentials
```

Engagement-history CSVs (`data/`) and generated personas (`backend/`) are distributed separately (see the [dataset](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v3)) — they are intentionally not tracked in git.

## 1. Build personas and their queries

One persona, end to end (28-step generation pipeline → `backend/115/` with `profile.json`, five app histories, `calendar.json`, `persona.html`):

```bash
python scripts/run_persona_pipeline.py --user_id 115 --input_csv data/all180_input.csv --verbose
python scripts/prepare_eval_data.py --user_id 115        # → backend/115/test.json (the eval queries)
```

Multiple personas:

```bash
scripts/run_next80_personas.sh                            # batch driver: resumable, bounded concurrency
python scripts/prepare_eval_data.py --user_range 17-118 --parallel 4
```

(Omit `--user_id` on `run_persona_pipeline.py` to process every user in the input CSV.)

## 2. Run evaluations and show results

Single persona, single mode:

```bash
python evaluation/run_eval.py --user_id 115 --mode llm_longctx \
    --model gpt-5.5 --judge_model gpt-5.5 --run_dir results/llm_longctx_gpt5.5/115
```

`--mode` ∈ `llm_longctx` (long-context baseline) · `llm_memory` (textual memory) · `mem0` (mem0 memory) · `agent_tools` (Claude Code agent over filesystem snapshots) · `mcp_agent` (Claude Code agent over per-app MCP tools) · `codex_agent` (Codex CLI agent). The judge is always `gpt-5.5`.

Full matrix over a cohort of personas and modes:

```bash
scripts/run_eval_matrix.sh --personas "1 2 3 5 6" --modes "llm_longctx llm_memory mem0"
```

Results land in `results/{mode}/{uid}/results.csv`. Aggregate and render the summary tables:

```bash
python scripts/aggregate_eval.py --results_root results   # → results/aggregate/ (CSV/JSON summaries)
```

Final comparison tables: `results/aggregate/html/results_tables.html`.

> **Note for agents (Claude Code / Codex):** all commands above are directly runnable from the repo root; evaluation runs make real LLM API calls, so confirm with the user before launching them.
