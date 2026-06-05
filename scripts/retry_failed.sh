#!/usr/bin/env bash
# Retry ONLY the failed rows (429 / transient API errors) of the existing
# gpt-5.5 eval, prune anything that still fails, then re-aggregate.
# GPT-5.5 ONLY (results/) — no gemini-3.5 (results_gemini/ is left untouched).
#
# Why this works now: query_llm._openai_create_with_retry absorbs Azure 429s
# with exponential backoff (previously only Gemini retried; gpt-5.5 rows just
# errored and the aggregator scored them 0). run_eval --retry_failed drops the
# non-ok rows so resume re-runs exactly them; --prune_invalid removes any that
# still fail (e.g. a hard Azure content-filter 400). workers=8 (12 tripped the
# rate limit); CONC personas run at a time.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=gpt-5.5
JUDGE=gpt-5.5
ROOT=results
WORKERS=8
CONC=2            # personas in parallel (each fires up to WORKERS answer calls)

ts() { date +%H:%M:%S; }
mkdir -p "$ROOT/_logs"

run_mode () {
  local mode="$1"
  local running=0
  for d in "$ROOT/$mode"/*/; do
    [ -f "${d}results.csv" ] || continue
    local uid; uid="$(basename "$d")"
    # Skip personas whose rows are all already ok (avoids reloading the memory
    # ledger for nothing). Proceed only if at least one row is non-ok.
    python3 -c "import csv,sys; sys.exit(0 if any((r.get('status') or '')!='ok' for r in csv.DictReader(open('${d}results.csv'))) else 1)" \
      || { echo "[retry $(ts)] skip $mode/$uid (all rows ok)"; continue; }
    echo "[retry $(ts)] queue $mode/$uid"
    ( python -u evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
        --run_dir "$ROOT/$mode/$uid" --mode "$mode" --model "$MODEL" \
        --judge_model "$JUDGE" --workers "$WORKERS" --memory_token_cap 4096 \
        --retry_failed --prune_invalid \
        > "$ROOT/_logs/$mode.$uid.retry.stdout" 2> "$ROOT/_logs/$mode.$uid.retry.stderr" ) &
    running=$((running+1))
    if [ "$running" -ge "$CONC" ]; then wait -n; running=$((running-1)); fi
  done
  wait
  echo "[retry $(ts)] DONE mode=$mode"
}

echo "[retry $(ts)] retrying FAILED gpt-5.5 rows (results/), both modes, workers=$WORKERS conc=$CONC"
run_mode llm_longctx
run_mode llm_memory

echo "[retry $(ts)] re-aggregating gpt-5.5 (results/) ..."
python scripts/aggregate_eval.py --results_root results --modes llm_longctx,llm_memory \
  > "$ROOT/_logs/aggregate.retry.log" 2>&1 || true
echo "[retry $(ts)] ALL DONE → results/aggregate/{llm_longctx,llm_memory}/token_accuracy_table.csv"
