#!/usr/bin/env bash
# Auto eval chain: gpt-5.5 + gemini-3.5-flash  x  llm_longctx + llm_memory, over
# all 20 personas, judged by gpt-5.5. Per-model result roots so they don't collide:
#   gpt-5.5  -> results/{mode}/{uid}
#   gemini   -> results_gemini/{mode}/{uid}
# gpt (Azure) and gemini (Google) run in PARALLEL (different APIs); within each,
# CONC personas run concurrently; modes run sequentially (llm_memory builds a ledger).
set -uo pipefail
cd "$(dirname "$0")/.."

PERSONAS="1 2 3 5 6 8 9 10 13 14 26 105 115 209 229 282 461 655 760 835"
JUDGE=gpt-5.5
WORKERS=8
CONC_GPT=3        # cross-persona concurrency for Azure gpt-5.5 (rate-limit safe)
CONC_GEM=2        # cross-persona concurrency for gemini (Tier 3)

run_pass () {
  local model="$1" root="$2" mode="$3" conc="$4"
  mkdir -p "$root/_logs"
  local running=0
  for uid in $PERSONAS; do
    mkdir -p "$root/$mode/$uid"
    ( python -u evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
        --run_dir "$root/$mode/$uid" --mode "$mode" --model "$model" \
        --judge_model "$JUDGE" --workers "$WORKERS" --memory_token_cap 4096 \
        > "$root/_logs/$mode.$uid.stdout" 2> "$root/_logs/$mode.$uid.stderr" ) &
    running=$((running+1))
    if [ "$running" -ge "$conc" ]; then wait -n; running=$((running-1)); fi
  done
  wait
  echo "[chain] DONE pass: model=$model mode=$mode -> $root/$mode/"
}

ts() { date +%H:%M:%S; }
echo "[chain $(ts)] launching gpt-5.5 (results/) + gemini-3.5-flash (results_gemini/) x {llm_longctx, llm_memory}"

(
  run_pass gpt-5.5        results        llm_longctx "$CONC_GPT"
  run_pass gpt-5.5        results        llm_memory  "$CONC_GPT"
) &
GPT_PID=$!
(
  run_pass gemini-3.5-flash  results_gemini llm_longctx "$CONC_GEM"
  run_pass gemini-3.5-flash  results_gemini llm_memory  "$CONC_GEM"
) &
GEM_PID=$!
wait "$GPT_PID"; echo "[chain $(ts)] gpt-5.5 BOTH modes done"
wait "$GEM_PID"; echo "[chain $(ts)] gemini BOTH modes done"

echo "[chain $(ts)] aggregating..."
python scripts/aggregate_eval.py --results_dir results        > results/_logs/aggregate.log 2>&1 || true
python scripts/aggregate_eval.py --results_dir results_gemini > results_gemini/_logs/aggregate.log 2>&1 || true
echo "[chain $(ts)] ALL DONE. aggregates: results/aggregate/, results_gemini/aggregate/"
