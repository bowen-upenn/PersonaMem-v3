#!/usr/bin/env bash
# Surgical eval of the newly-generated short_vs_long_term_lifecycle (e5) task
# across all 8 paper model-mode configs, 5 personas {1,2,3,5,6}.
# --task + --resume fills ONLY the e5 rows into each existing canonical run_dir
# without touching any other task's rows. Gemini configs run with
# EVAL_GEMINI_BATCH=0 so the long-context prefix is served from a context cache.
# e5 is NDCG-scored (deterministic) — no LLM judge cost.
set -u
PERSONAS="1 2 3 5 6"
TASK="short_vs_long_term_lifecycle"
JUDGE="gpt-5.5"
LOGD=/tmp/eval_regen/e5
mkdir -p "$LOGD"

run_cfg() {  # $1=config_dir ; rest=run_eval flags
  local cfg="$1"; shift
  for uid in $PERSONAS; do
    echo "[$(date +%H:%M:%S)] $cfg u$uid START" >> "$LOGD/${cfg}.log"
    python evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
      --run_dir "results/$cfg/$uid" --task "$TASK" --resume \
      --judge_model "$JUDGE" "$@" >> "$LOGD/${cfg}.log" 2>&1
    echo "[$(date +%H:%M:%S)] $cfg u$uid DONE rc=$?" >> "$LOGD/${cfg}.log"
  done
}

# Wave 1 — fast LLM-baseline modes (parallel across configs).
run_cfg llm_longctx_gpt5.5    --mode llm_longctx --model gpt-5.5 --workers 8 &
EVAL_GEMINI_BATCH=0 run_cfg llm_longctx_gemini3.5flash --mode llm_longctx --model gemini-3.5-flash --workers 8 &
run_cfg llm_memory_gpt5.5     --mode llm_memory --model gpt-5.5 --workers 8 &
EVAL_GEMINI_BATCH=0 run_cfg llm_memory_gemini3.5flash --mode llm_memory --model gemini-3.5-flash --workers 8 &
run_cfg mem0_gpt5.5           --mode mem0 --model gpt-5.5 --workers 1 &
wait
echo "[$(date +%H:%M:%S)] WAVE1 (llm modes) complete" >> "$LOGD/_driver.log"

# Wave 2 — agent modes (slower; spawn CLI subprocesses per query).
run_cfg codex_agent_gpt5.5    --mode codex_agent --model gpt-5.5 --workers 4 &
run_cfg agent_tools_opus4.8   --mode agent_tools --claude_model opus --workers 4 &
run_cfg agent_tools_sonnet4.6 --mode agent_tools --claude_model sonnet --workers 4 &
wait
echo "[$(date +%H:%M:%S)] WAVE2 (agent modes) complete — ALL DONE" >> "$LOGD/_driver.log"
