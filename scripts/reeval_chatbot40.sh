#!/usr/bin/env bash
# Re-eval the expanded new_suggestions_chatbot set (40 instances, 5 personas)
# across the 7 configs Claude Code can run (codex is run separately by the user).
# --resume keeps the ~2 overlapping old anchors; the ~30 new instances evaluate
# fresh under the FIXED scoring. mem0 reuses its store; gemini uses the cache.
set -u
PERSONAS="1 2 3 5 6"
TASK="new_suggestions_chatbot"
JUDGE="gpt-5.5"
LOGD=/tmp/eval_regen/cb40
mkdir -p "$LOGD"

run_cfg() {
  local cfg="$1"; shift
  for uid in $PERSONAS; do
    echo "[$(date +%H:%M:%S)] $cfg u$uid START" >> "$LOGD/${cfg}.log"
    python evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
      --run_dir "results/$cfg/$uid" --task "$TASK" --resume \
      --judge_model "$JUDGE" "$@" >> "$LOGD/${cfg}.log" 2>&1
    echo "[$(date +%H:%M:%S)] $cfg u$uid DONE rc=$?" >> "$LOGD/${cfg}.log"
  done
  echo "[$(date +%H:%M:%S)] $cfg ALL-DONE" >> "$LOGD/_driver.log"
}

run_cfg llm_longctx_gpt5.5    --mode llm_longctx --model gpt-5.5 --workers 8 &
EVAL_GEMINI_BATCH=0 run_cfg llm_longctx_gemini3.5flash --mode llm_longctx --model gemini-3.5-flash --workers 8 &
run_cfg llm_memory_gpt5.5     --mode llm_memory --model gpt-5.5 --workers 8 &
EVAL_GEMINI_BATCH=0 run_cfg llm_memory_gemini3.5flash --mode llm_memory --model gemini-3.5-flash --workers 8 &
run_cfg mem0_gpt5.5           --mode mem0 --model gpt-5.5 --workers 1 --reuse_mem0_store &
run_cfg agent_tools_opus4.8   --mode agent_tools --claude_model opus --workers 4 &
run_cfg agent_tools_sonnet4.6 --mode agent_tools --claude_model sonnet --workers 4 &
wait
echo "[$(date +%H:%M:%S)] ALL CONFIGS DONE" >> "$LOGD/_driver.log"
