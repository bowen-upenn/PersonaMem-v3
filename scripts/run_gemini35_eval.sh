#!/bin/bash
# ============================================================================
# Run gemini-3.5-flash eval on the 10 Sonnet-covered personas, WITH CACHE.
#   MODE=llm_longctx (default) or llm_memory
# Cache requires --workers 1 per process (sequential ascending-T + in-process
# API cost tracking). Personas run in PARALLEL at the shell level (independent
# caches), bounded by CONCURRENCY to avoid Gemini TPM thrash.
#
# llm_memory: reuses the gpt-5.5-built ledgers (results/llm_memory_gpt5.5/{uid}/
# memory_states) by seeding the gemini run_dir — build_checkpoints' resume
# fast-path then skips the (gpt-5.5) rebuild. So no rebuild cost.
#
# Per-persona log: /tmp/eval_gemini/{MODE}_{uid}.log
# Output:          results/{MODE}_gemini3.5flash/{uid}/
# ============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${MODE:-llm_longctx}"
MODEL="${MODEL:-gemini-3.5-flash}"
CONCURRENCY="${CONCURRENCY:-5}"
JUDGE_FLAG="${JUDGE_FLAG:---enable_llm_judge}"      # default: JUDGE ON (gpt-5.5 judge). Scored runs are the norm.
PERSONAS="${PERSONAS:-1 2 3 5 6 8 9 10 13 14}"
# OUT_ROOT overridable (e.g. results/llm_longctx_gpt5.5 for the gpt-5.5 reeval).
OUT_ROOT="${OUT_ROOT:-results/${MODE}_gemini3.5flash}"
LOG_DIR="/tmp/eval_gemini"
mkdir -p "$LOG_DIR"

echo "MODE=$MODE MODEL=$MODEL CONCURRENCY=$CONCURRENCY JUDGE=$JUDGE_FLAG"
echo "personas: $PERSONAS"

run_one() {
  local uid="$1"
  local rd="$OUT_ROOT/$uid"
  mkdir -p "$rd"
  # llm_memory: seed the reusable gpt-5.5 ledgers so the build is skipped.
  if [ "$MODE" = "llm_memory" ]; then
    local src="results/llm_memory_gpt5.5/$uid/memory_states"
    if [ -d "$src" ] && [ ! -d "$rd/memory_states" ]; then
      cp -r "$src" "$rd/memory_states"
    fi
  fi
  EVAL_GEMINI_BATCH=0 EVAL_CHRONO_HISTORY=1 \
  python -m evaluation.run_eval \
    --user_id "$uid" --mode "$MODE" --model "$MODEL" \
    --judge_model "${JUDGE_MODEL:-gpt-5.5}" \
    --workers 1 $JUDGE_FLAG --resume ${RETRY_FLAG:-} \
    ${REPLAY_FROM:+--replay_from "$REPLAY_FROM"} \
    --run_dir "$rd" > "$LOG_DIR/${MODE}_${uid}.log" 2>&1
  echo "[done] uid=$uid mode=$MODE exit=$?"
}
export -f run_one
export MODE MODEL OUT_ROOT LOG_DIR JUDGE_FLAG

# Bounded parallelism.
printf '%s\n' $PERSONAS | xargs -P "$CONCURRENCY" -I{} bash -c 'run_one "$@"' _ {}
echo "ALL PERSONAS LAUNCHED/COMPLETED for MODE=$MODE"
