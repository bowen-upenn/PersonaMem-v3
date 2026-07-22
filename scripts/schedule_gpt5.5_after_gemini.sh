#!/bin/bash
# ============================================================================
# Scheduled gpt-5.5 llm_longctx RE-eval on the current benchmark (primary cohort).
# Waits until the gemini-3.5-flash run_eval processes (both modes) have exited,
# then clears the STALE old-benchmark results in results/llm_longctx_gpt5.5 and
# launches a fresh gpt-5.5 long-context run on the SAME cohort.
#
# gpt-5.5 = Azure (different provider from gemini), judge OFF (a single gpt-5.5
# judge-replay pass scores gemini + gpt-5.5 together later). Cache via
# --workers 1 ascending-T (engages Azure prompt caching). CONCURRENCY kept low
# for Azure TPM.
#
# Cancel before it fires: TaskStop this background task (or kill this PID).
# Logs: /tmp/eval_gemini/_gpt5.5_scheduled.log + per-persona llm_longctx_*.log
# ============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p /tmp/eval_gemini

# Persona cohort is defined in a local, untracked file (see .gitignore).
[ -f scripts/personas.local.sh ] && . scripts/personas.local.sh
PERSONAS="${PERSONAS:-${PERSONAS_PRIMARY:-}}"
[ -n "$PERSONAS" ] || { echo "ERROR: set PERSONAS=... or create scripts/personas.local.sh" >&2; exit 2; }

echo "[schedule] $(date) waiting for gemini-3.5-flash run_eval to finish..."
deadline=$(( $(date +%s) + 6*3600 ))   # 6h safety cap
# Poll until no gemini run_eval worker remains.
while pgrep -f "run_eval.*--model gemini-3.5-flash" >/dev/null 2>&1; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[schedule] deadline exceeded — gemini still running; aborting gpt-5.5 launch."
    exit 1
  fi
  sleep 60
done
echo "[schedule] $(date) gemini done."

# Clear STALE old-benchmark results so the reeval is clean current-benchmark only.
for u in $PERSONAS; do
  rm -f "results/llm_longctx_gpt5.5/$u/results.csv" "results/llm_longctx_gpt5.5/$u/summary.json"
done
echo "[schedule] cleared stale llm_longctx_gpt5.5. launching gpt-5.5 longctx reeval..."

MODE=llm_longctx MODEL=gpt-5.5 OUT_ROOT=results/llm_longctx_gpt5.5 \
  CONCURRENCY="${CONCURRENCY:-3}" JUDGE_FLAG="--no-enable_llm_judge" \
  bash scripts/run_gemini35_eval.sh
echo "[schedule] $(date) gpt-5.5 longctx reeval finished."
