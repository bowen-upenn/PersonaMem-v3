#!/usr/bin/env bash
# Combined surgical eval of BOTH new tasks across all 8 paper configs, 5 personas.
# All configs run IN PARALLEL (no wave gating) so mem0's slow per-persona store
# build doesn't block the agent modes. --resume skips already-done rows (the 4
# LLM modes' e5 is already complete → they only add new_suggestions_chatbot).
# Gemini configs use EVAL_GEMINI_BATCH=0 for the context cache. e5 = NDCG (no
# judge); new_suggestions_chatbot uses the gpt-5.5 judge.
set -u
PERSONAS="1 2 3 5 6"
TASKS="short_vs_long_term_lifecycle,new_suggestions_chatbot"
JUDGE="gpt-5.5"
LOGD=/tmp/eval_regen/e5c
mkdir -p "$LOGD"

run_cfg() {  # $1=config_dir ; rest=run_eval flags
  local cfg="$1"; shift
  for uid in $PERSONAS; do
    echo "[$(date +%H:%M:%S)] $cfg u$uid START" >> "$LOGD/${cfg}.log"
    python evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
      --run_dir "results/$cfg/$uid" --task "$TASKS" --resume \
      --judge_model "$JUDGE" "$@" >> "$LOGD/${cfg}.log" 2>&1
    echo "[$(date +%H:%M:%S)] $cfg u$uid DONE rc=$?" >> "$LOGD/${cfg}.log"
  done
  echo "[$(date +%H:%M:%S)] $cfg ALL-PERSONAS-DONE" >> "$LOGD/_driver.log"
}

run_cfg llm_longctx_gpt5.5    --mode llm_longctx --model gpt-5.5 --workers 8 &
EVAL_GEMINI_BATCH=0 run_cfg llm_longctx_gemini3.5flash --mode llm_longctx --model gemini-3.5-flash --workers 8 &
run_cfg llm_memory_gpt5.5     --mode llm_memory --model gpt-5.5 --workers 8 &
EVAL_GEMINI_BATCH=0 run_cfg llm_memory_gemini3.5flash --mode llm_memory --model gemini-3.5-flash --workers 8 &
run_cfg mem0_gpt5.5           --mode mem0 --model gpt-5.5 --workers 1 --reuse_mem0_store &
run_cfg codex_agent_gpt5.5    --mode codex_agent --model gpt-5.5 --workers 4 &
run_cfg agent_tools_opus4.8   --mode agent_tools --claude_model opus --workers 4 &
run_cfg agent_tools_sonnet4.6 --mode agent_tools --claude_model sonnet --workers 4 &
wait
echo "[$(date +%H:%M:%S)] ALL CONFIGS DONE" >> "$LOGD/_driver.log"
