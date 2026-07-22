#!/usr/bin/env bash
# Full eval matrix (post audit-fix + query rebuild). User-authorized, no approval.
#   gpt-5.5  × { llm_longctx, llm_memory, mem0 }   (Azure)
#   opus-4.8 × { agent_tools }                     (Claude Code subagent)
# The configured persona cohort → results/{mode}/{uid}; gpt-5.5 judge; --prune_invalid
# so the headline is over COMPLETED rows only. The Azure-gpt chain (3 modes, serialized
# to share the rate limit) runs in PARALLEL with the Claude agent chain. Ends
# with a per-mode aggregate + cross-mode comparison.csv.
set -uo pipefail
cd "$(dirname "$0")/.."

# Persona cohort is defined in a local, untracked file (see .gitignore).
[ -f scripts/personas.local.sh ] && . scripts/personas.local.sh
PERSONAS="${PERSONAS:-${PERSONAS_EXTENDED:-}}"
[ -n "$PERSONAS" ] || { echo "ERROR: set PERSONAS=... or create scripts/personas.local.sh" >&2; exit 2; }
GPT=gpt-5.5
JUDGE=gpt-5.5
OPUS=opus
WORKERS=8          # intra-persona answer concurrency for longctx/memory
CONC_GPT=3         # personas at a time for longctx/memory (rate-limit safe)
JOBS_MEM0=3        # mem0 forces --workers 1 in-process → parallelize across personas

ts(){ date +%H:%M:%S; }
mkdir -p results/_logs

# Back up the prior results/ once (the old longctx/memory are stale vs new queries).
if [ ! -d results/_prefix_backup ]; then
  mkdir -p results/_prefix_backup
  for m in llm_longctx llm_memory mem0 agent_tools; do
    [ -d "results/$m" ] && cp -r "results/$m" "results/_prefix_backup/$m" 2>/dev/null
  done
  echo "[matrix $(ts)] backed up prior results/ → results/_prefix_backup/"
fi

run_gpt_mode(){   # $1=mode  $2=conc
  local mode="$1" conc="$2" running=0
  for uid in $PERSONAS; do
    mkdir -p "results/$mode/$uid"
    ( python -u evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
        --run_dir "results/$mode/$uid" --mode "$mode" --model "$GPT" --judge_model "$JUDGE" \
        --workers "$WORKERS" --memory_token_cap 4096 --prune_invalid \
        > "results/_logs/$mode.$uid.stdout" 2> "results/_logs/$mode.$uid.stderr" ) &
    running=$((running+1)); if [ "$running" -ge "$conc" ]; then wait -n; running=$((running-1)); fi
  done; wait; echo "[matrix $(ts)] DONE $mode"
}

run_mem0(){
  local running=0
  for uid in $PERSONAS; do
    mkdir -p "results/mem0/$uid"
    ( python -u evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
        --run_dir "results/mem0/$uid" --mode mem0 --model "$GPT" --judge_model "$JUDGE" \
        --memory_token_cap 4096 --prune_invalid \
        > "results/_logs/mem0.$uid.stdout" 2> "results/_logs/mem0.$uid.stderr" ) &
    running=$((running+1)); if [ "$running" -ge "$JOBS_MEM0" ]; then wait -n; running=$((running-1)); fi
  done; wait; echo "[matrix $(ts)] DONE mem0"
}

run_agent_tools(){   # Claude Code subagent — one persona at a time (subscription)
  for uid in $PERSONAS; do
    mkdir -p "results/agent_tools/$uid"
    python -u evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
        --run_dir "results/agent_tools/$uid" --mode agent_tools --claude_model "$OPUS" \
        --judge_model "$JUDGE" --prune_invalid \
        > "results/_logs/agent_tools.$uid.stdout" 2> "results/_logs/agent_tools.$uid.stderr" || true
  done; echo "[matrix $(ts)] DONE agent_tools"
}

echo "[matrix $(ts)] START: gpt-5.5 {longctx,memory,mem0} ‖ opus agent_tools"
# Azure-gpt chain (serialized — shared rate limit) ‖ Claude agent chain (other API)
( run_gpt_mode llm_longctx "$CONC_GPT"; run_gpt_mode llm_memory "$CONC_GPT"; run_mem0 ) &
GPT_PID=$!
run_agent_tools &
AGENT_PID=$!
wait "$GPT_PID"; echo "[matrix $(ts)] gpt-5.5 chain done"
wait "$AGENT_PID"; echo "[matrix $(ts)] agent_tools done"

echo "[matrix $(ts)] aggregating ..."
python scripts/aggregate_eval.py --results_root results \
    --modes llm_longctx,llm_memory,mem0,agent_tools \
    > results/_logs/aggregate.matrix.log 2>&1 || true
echo "[matrix $(ts)] ALL DONE → results/aggregate/{mode}/token_accuracy_table.csv + comparison.csv"
