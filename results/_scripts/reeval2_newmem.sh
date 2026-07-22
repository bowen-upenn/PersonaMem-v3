#!/usr/bin/env bash
# Phase 2: full textual-memory re-eval, both models, seeded NEW memory (--resume → no rebuild),
# Gemini batch + cache ON (no EVAL_GEMINI_BATCH override), Azure prompt cache, judge gpt-5.5.
set -u
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
PERS="1 2 3 5 6 8 9 10 13 14"
LOG=results/_logs/reeval2; mkdir -p "$LOG"
run_one(){ local model=$1 mdir=$2 u=$3
  python -m evaluation.run_eval --user_id "$u" --backend_dir backend \
    --run_dir "results/_reeval_newmem/$mdir/$u" \
    --mode llm_memory --model "$model" --memory_builder_model "$model" \
    --judge_model gpt-5.5 --memory_token_cap 4096 --resume --workers 8 \
    > "$LOG/$mdir.$u.stdout" 2> "$LOG/$mdir.$u.stderr" \
    && echo "[reeval2] DONE $mdir/$u" || echo "[reeval2] FAIL $mdir/$u (exit $?)"; }
run_model(){ local model=$1 mdir=$2 jobs=$3; local r=0
  for u in $PERS; do run_one "$model" "$mdir" "$u" & r=$((r+1));
    if [ "$r" -ge "$jobs" ]; then wait -n 2>/dev/null||wait; r=$((r-1)); fi; done; wait
  echo "[reeval2] $mdir ALL DONE"; }
( run_model gpt-5.5 gpt_5_5 1 ) &
( run_model gemini-3.5-flash gemini_3_5_flash 2 ) &
wait
echo "[reeval2] PHASE 2 COMPLETE"
