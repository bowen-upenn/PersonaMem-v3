# PersonaMem-v3: Toward Omni-Platform Personal Intelligence for Holistic User Understanding, Recommendation, and Agentic Tasks

<!-- TODO: point the Paper badge at the arXiv abs page once the v3 paper is up -->
[![Paper](https://img.shields.io/badge/Paper-preview-b31b1b.svg)](PersonaMem_v3.pdf)
[![alphaXiv](https://img.shields.io/badge/alphaXiv-PersonaMem--v3-b31b1b.svg)](https://www.alphaxiv.org/abs/2607.personamem-v3-omni-platform-personal-intelligence)
[![Dataset](https://img.shields.io/badge/HuggingFace-PersonaMem--v3-ffd21e.svg)](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v3)

Bowen Jiang, Yuan Yuan, Zhuoqun Hao, Yuchen Liu, Maohao Shen, Sihao Chen, Gregory Wornell, Chris Callison-Burch, Lyle Ungar, Dan Roth, Qi Guo, Xiangjun Fan, Camillo J. Taylor, Hanchao Yu

A collaboration between <img src="assets/meta.png" height="16" alt="Meta"> **Meta Recommendation Systems**, <img src="assets/upenn.png" height="16" alt="UPenn"> **University of Pennsylvania**, and <img src="assets/mit.png" height="16" alt="MIT"> **MIT**.

![PersonaMem-v3](assets/header.png)

Third release in the PersonaMem series:

- **PersonaMem (v1)** — *[COLM 2025] Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale* · [code](https://github.com/bowen-upenn/PersonaMem) · [paper](https://arxiv.org/abs/2504.14225) · [data](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1)
- **PersonaMem-v2** — *Towards Personalized Intelligence via Learning Implicit User Personas and Agentic Memory* · [code](https://github.com/bowen-upenn/PersonaMem-v2) · [paper](https://arxiv.org/abs/2512.06688) · [data](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2)

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
# download the full source engagement data (facebook/gistbench) and convert it to the input CSV:
python scripts/download_gistbench.py                # → data/gistbench_input.csv
```

Source data comes from [facebook/gistbench](https://huggingface.co/datasets/facebook/gistbench); pre-built personas are distributed via the [PersonaMem-v3 dataset](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v3). Neither is tracked in git — but a ready-made 10-user sample input ships with the repo at `data/gistbench_sample_10users.csv` for a quick start.

## 1. Build personas, user histories, their queries

One persona, end to end (full generation pipeline → `backend/115/` with `profile.json`, five app histories, `calendar.json`, `persona.html`):

```bash
python scripts/run_persona_pipeline.py --user_id 115 --input_csv data/gistbench_input.csv --verbose
python scripts/prepare_eval_data.py --user_id 115        # → backend/115/test.json (the eval queries)
```

Multiple personas:

```bash
bash scripts/run_persona_batch.sh                         # every user in the input CSV; resumable
NUM_USERS=25 bash scripts/run_persona_batch.sh            # only the first 25 user ids in the CSV
USERS="17 18 115" bash scripts/run_persona_batch.sh       # explicit persona ids
# other knobs: INPUT_CSV=path/to/input.csv  CONCURRENCY=3 (personas generated simultaneously)
python scripts/prepare_eval_data.py --all --parallel 4    # queries for every generated persona
```

## 2. Run evaluations and show results

All artifacts are plain JSON/CSV/HTML on disk:

- **Personas and their data** — `backend/{uid}/`: `profile.json` (persona definition), `instagram.json` / `facebook.json` / `threads.json` / `chatbot.json` / `ai_studio.json` (time-sorted interaction-event histories per app), `calendar.json` (calendar modification stream), `test.json` (the eval queries), `persona.html` (self-contained human-readable review page).
- **Eval runs** — `results/{mode}/{uid}/`: `results.csv` (one row per query: `query_id, seq, user_id, task_type, ts, metrics_json, status, duration_ms, error, agent_response`), `writes.jsonl` (agentic write actions), `summary.json`.
- **Aggregates** — `results/aggregate/` (per-mode CSV/JSON summaries); final comparison tables at `results/aggregate/html/results_tables.html`.

Single persona, single mode:

```bash
python evaluation/run_eval.py --user_id 115 --mode llm_longctx \
    --model gpt-5.5 --judge_model gpt-5.5 --run_dir results/llm_longctx_gpt5.5/115
```

`--mode` ∈ `llm_longctx` (long-context baseline) · `llm_memory` (textual memory) · `mem0` (mem0 memory) · `claude_code` (Claude Code agent over time-masked filesystem snapshots) · `codex` (Codex CLI agent over the same snapshots). The judge is always `gpt-5.5`.

Full matrix over a cohort of personas and modes:

```bash
scripts/run_eval_matrix.sh --personas "101 102 103" --modes "llm_longctx llm_memory mem0"
```

Aggregate and render the summary tables:

```bash
python scripts/aggregate_eval.py --results_root results   # → results/aggregate/ (CSV/JSON summaries)
```

> **Note for agents (Claude Code / Codex):** all commands above are directly runnable from the repo root; evaluation runs make real LLM API calls, so confirm with the user before launching them.

## License

Code is released under the [MIT License](LICENSE). The sample input data and all personas derived from [facebook/gistbench](https://huggingface.co/datasets/facebook/gistbench) inherit its **CC-BY-NC-4.0** license (attribution, non-commercial).
