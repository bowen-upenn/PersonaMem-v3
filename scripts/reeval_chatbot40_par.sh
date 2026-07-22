#!/usr/bin/env bash
# Parallel-persona re-eval of the 40-instance new_suggestions_chatbot set.
# Configs run in parallel AND the 5 personas within each config run in parallel
# (was sequential — the agent modes' bottleneck). --resume makes already-done
# personas instant no-ops. Agent modes use --workers 2 so 5 parallel personas
# stay at ~10 concurrent subagents (bounded, to avoid re-triggering 529s).
set -u
PERSONAS="1 2 3 5 6"
TASK="new_suggestions_chatbot"
JUDGE="gpt-5.5"
LOGD=/tmp/eval_regen/cb40
mkdir -p "$LOGD"

run_cfg() {  # $1=cfg ; rest=flags ; runs all personas in PARALLEL
  local cfg="$1"; shift
  for uid in $PERSONAS; do
    (
      echo "[$(date +%H:%M:%S)] $cfg u$uid START" >> "$LOGD/${cfg}.log"
      python evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
        --run_dir "results/$cfg/$uid" --task "$TASK" --resume \
        --judge_model "$JUDGE" "$@" >> "$LOGD/${cfg}.log" 2>&1
      echo "[$(date +%H:%M:%S)] $cfg u$uid DONE rc=$?" >> "$LOGD/${cfg}.log"
    ) &
  done
  wait
  echo "[$(date +%H:%M:%S)] $cfg ALL-DONE" >> "$LOGD/_driver.log"
}

run_cfg llm_longctx_gpt5.5    --mode llm_longctx --model gpt-5.5 --workers 8 &
EVAL_GEMINI_BATCH=0 run_cfg llm_longctx_gemini3.5flash --mode llm_longctx --model gemini-3.5-flash --workers 8 &
run_cfg llm_memory_gpt5.5     --mode llm_memory --model gpt-5.5 --workers 8 &
EVAL_GEMINI_BATCH=0 run_cfg llm_memory_gemini3.5flash --mode llm_memory --model gemini-3.5-flash --workers 8 &
run_cfg mem0_gpt5.5           --mode mem0 --model gpt-5.5 --workers 1 --reuse_mem0_store &
run_cfg agent_tools_opus4.8   --mode agent_tools --claude_model opus --workers 2 &
run_cfg agent_tools_sonnet4.6 --mode agent_tools --claude_model sonnet --workers 2 &
wait
echo "[$(date +%H:%M:%S)] ALL CONFIGS DONE (parallel personas)" >> "$LOGD/_driver.log"
