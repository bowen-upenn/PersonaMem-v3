#!/usr/bin/env bash
# Run the Codex-harness GPT-5.5 filesystem-agent eval on the primary persona
# cohort used for matched cross-model comparison (agent_tools_sonnet4.6 / _opus4.8).
#
# This launches real Codex CLI calls. It sends each row's time-masked snapshot
# prompt to the Codex/OpenAI service via `codex exec`.
set -uo pipefail
cd "$(dirname "$0")/.."

# Persona cohort is defined in a local, untracked file (see .gitignore).
[ -f scripts/personas.local.sh ] && . scripts/personas.local.sh
PERSONAS="${PERSONAS:-${PERSONAS_PRIMARY:-}}"
[ -n "$PERSONAS" ] || { echo "ERROR: set PERSONAS=... or create scripts/personas.local.sh" >&2; exit 2; }
MODEL="${MODEL:-gpt-5.5}"
JUDGE="${JUDGE:-gpt-5.5}"
OUT_ROOT="${OUT_ROOT:-results/codex_agent_gpt5.5}"
WORKERS="${CODEX_WORKERS:-1}"
RATE_LIMIT="${RATE_LIMIT:-50}"
RESUME="${RESUME:---resume}"
RETRY_EMPTY="${RETRY_EMPTY:---retry_empty}"
LIMIT="${LIMIT:-}"

mkdir -p results/_logs

ts(){ date +%H:%M:%S; }

echo "[codex-agent $(ts)] personas=[$PERSONAS] model=$MODEL judge=$JUDGE workers=$WORKERS out=$OUT_ROOT"

for uid in $PERSONAS; do
  rundir="$OUT_ROOT/$uid"
  mkdir -p "$rundir"
  args=(--user_id "$uid" --backend_dir backend --run_dir "$rundir"
        --mode codex_agent --model "$MODEL" --judge_model "$JUDGE"
        --workers "$WORKERS" --rate_limit "$RATE_LIMIT" --prune_invalid)
  [ -n "$RESUME" ] && args+=($RESUME)
  [ -n "$RETRY_EMPTY" ] && args+=($RETRY_EMPTY)
  [ -n "$LIMIT" ] && args+=(--limit "$LIMIT")

  echo "[codex-agent $(ts)] START uid=$uid -> $rundir"
  if python -u -m evaluation.run_eval "${args[@]}" \
      > "results/_logs/codex_agent_gpt5.5.$uid.stdout" \
      2> "results/_logs/codex_agent_gpt5.5.$uid.stderr"; then
    echo "[codex-agent $(ts)] DONE  uid=$uid"
  else
    echo "[codex-agent $(ts)] FAIL  uid=$uid -- see results/_logs/codex_agent_gpt5.5.$uid.stderr" >&2
    if grep -qi "usage limit" "results/_logs/codex_agent_gpt5.5.$uid.stderr"; then
      echo "[codex-agent $(ts)] STOP  Codex usage limit hit; resume after reset with the same command." >&2
      exit 75
    fi
  fi
done

echo "[codex-agent $(ts)] aggregating matched comparison"
python scripts/aggregate_eval.py --results_root results \
  --modes agent_tools_sonnet4.6,agent_tools_opus4.8,llm_longctx_gpt5.5_judged,codex_agent_gpt5.5
echo "[codex-agent $(ts)] done"
