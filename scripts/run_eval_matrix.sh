#!/usr/bin/env bash
# 5-config evaluation matrix over the PersonaMem-v3 personas.
#
#   modes  : llm_longctx  llm_memory  mem0        (Azure gpt-5.5 baselines)
#            agent_tools   mcp_agent              (Claude Code, opus 4.8)
#   judge  : gpt-5.5 (Azure) for all
#   layout : results/{mode}/{uid}/results.csv   (+ per-(mode,uid) logs under results/_logs/)
#
# Usage:
#   scripts/run_eval_matrix.sh [--personas "1 2 3"] [--modes "llm_longctx mem0"] \
#       [--limit N] [--gpt-model gpt-5-chat] [--claude-model opus] \
#       [--gpt-workers 4] [--agent-workers 1] [--jobs 1] [--no-resume]
#
# --limit N         cap rows per (mode,persona) — use for the smoke test.
# --jobs N          run up to N gpt-mode (mode,persona) jobs concurrently.
#                   Agent modes (agent_tools/mcp_agent) ALWAYS run one at a time
#                   to protect the Claude subscription + avoid overlay races.
# --no-resume       ignore prior partial results.csv (default: --resume).
#
# NOTE: this launches real LLM/eval runs (evaluation/run_eval.py). Per project
# rules it must only be run with explicit user approval.
set -uo pipefail
cd "$(dirname "$0")/.."

PERSONAS="1 2 3 5 6 8 9 10 13 14 26 105 115 209 229 282 461 655 760 835"
MODES="llm_longctx llm_memory mem0 agent_tools mcp_agent"
GPT_MODEL="gpt-5-chat"          # QueryLLM maps this to AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5.5
CLAUDE_MODEL="opus"             # alias -> latest opus (opus-4.8)
JUDGE_MODEL="gpt-5.5"
GPT_WORKERS=4
AGENT_WORKERS=1
JOBS=1
LIMIT=""
RESUME="--resume"

while [ $# -gt 0 ]; do
  case "$1" in
    --personas)      PERSONAS="$2"; shift 2;;
    --modes)         MODES="$2"; shift 2;;
    --limit)         LIMIT="--limit $2"; shift 2;;
    --gpt-model)     GPT_MODEL="$2"; shift 2;;
    --claude-model)  CLAUDE_MODEL="$2"; shift 2;;
    --gpt-workers)   GPT_WORKERS="$2"; shift 2;;
    --agent-workers) AGENT_WORKERS="$2"; shift 2;;
    --jobs)          JOBS="$2"; shift 2;;
    --no-resume)     RESUME=""; shift 1;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

LOGDIR="results/_logs"
mkdir -p "$LOGDIR"
echo "[matrix] personas=[$PERSONAS]"
echo "[matrix] modes=[$MODES]  gpt_model=$GPT_MODEL  claude_model=$CLAUDE_MODEL  judge=$JUDGE_MODEL"
echo "[matrix] gpt_workers=$GPT_WORKERS agent_workers=$AGENT_WORKERS jobs=$JOBS limit='${LIMIT:-none}' resume='${RESUME:-off}'"

is_agent_mode() { case "$1" in agent_tools|mcp_agent) return 0;; *) return 1;; esac; }

run_one() {
  local mode="$1" uid="$2"
  local rundir="results/$mode/$uid"
  local out="$LOGDIR/$mode.$uid.stdout" err="$LOGDIR/$mode.$uid.stderr"
  mkdir -p "$rundir"
  local args=(--user_id "$uid" --backend_dir backend --run_dir "$rundir"
              --mode "$mode" --judge_model "$JUDGE_MODEL" --enable-llm-judge
              --memory_token_cap 2048)
  [ -n "$RESUME" ] && args+=($RESUME)
  [ -n "$LIMIT" ]  && args+=($LIMIT)
  if is_agent_mode "$mode"; then
    args+=(--claude_model "$CLAUDE_MODEL" --workers "$AGENT_WORKERS")
  else
    args+=(--model "$GPT_MODEL" --workers "$GPT_WORKERS")
  fi
  echo "[matrix] START $mode/$uid -> $rundir  (log: $out)"
  if python -m evaluation.run_eval "${args[@]}" >"$out" 2>"$err"; then
    echo "[matrix] DONE  $mode/$uid"
  else
    echo "[matrix] FAIL  $mode/$uid (exit $?) — see $err" >&2
  fi
}

for mode in $MODES; do
  if is_agent_mode "$mode"; then
    # Agent modes: strictly one (mode,persona) at a time.
    for uid in $PERSONAS; do run_one "$mode" "$uid"; done
  else
    # GPT modes: up to $JOBS personas concurrently.
    running=0
    for uid in $PERSONAS; do
      run_one "$mode" "$uid" &
      running=$((running+1))
      if [ "$running" -ge "$JOBS" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
    done
    wait
  fi
done

echo "[matrix] all jobs finished — aggregating"
python scripts/aggregate_eval.py --results_root results --modes "$(echo "$MODES" | tr ' ' ',')"
echo "[matrix] done. cross-mode comparison: results/aggregate/comparison.csv"
