#!/usr/bin/env bash
# Finish the 3 remaining configs for both new tasks (mem0, opus, sonnet).
# The 4 LLM modes + codex are already complete. mem0 reuses its prebuilt store.
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
  echo "[$(date +%H:%M:%S)] $cfg ALL-PERSONAS-DONE" >> "$LOGD/_driver2.log"
}

run_cfg mem0_gpt5.5           --mode mem0 --model gpt-5.5 --workers 1 --reuse_mem0_store &
run_cfg agent_tools_opus4.8   --mode agent_tools --claude_model opus --workers 4 &
run_cfg agent_tools_sonnet4.6 --mode agent_tools --claude_model sonnet --workers 4 &
wait
echo "[$(date +%H:%M:%S)] REMAINING-3 ALL DONE" >> "$LOGD/_driver2.log"
