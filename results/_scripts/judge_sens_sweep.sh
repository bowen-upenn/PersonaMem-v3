#!/usr/bin/env bash
# Judge-sensitivity sweep: re-judge the saved matrix responses under two
# ALTERNATIVE judges (gpt-5.4-mini, claude-opus-4.8) to measure how far the
# reported accuracy cells move vs the live GPT-5.5 judge. Saved responses only
# (no model-under-test calls). NO --write_back: the live GPT-5.5 table is left
# untouched; alt-judge scores go only to /tmp/eval_regen/judge_sens/.
#
# Baseline (GPT-5.5) is read for free from each live results.csv
# (pr_query_score_0_10, written by the identical Jun-12 rejudge_existing path),
# so we only spend LLM calls on the two alt judges.
set -uo pipefail
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
ts(){ date +%H:%M:%S; }

OUT=/tmp/eval_regen/judge_sens
LOG=results/_logs/judge_sens
mkdir -p "$OUT" "$LOG"

CFGS=(
  llm_longctx_gpt5.5_judged
  llm_memory_gpt5.5
  mem0_gpt5.5
  codex_agent_gpt5.5
  llm_longctx_gemini3.5flash_judged
  llm_memory_gemini3.5flash_judged
  agent_tools_opus4.8
  agent_tools_sonnet4.6
)
JUDGES=(gpt-5.4-mini claude-opus-4.8)
USERS="1,2,3,5,6,8,9,10,13,14"

run_one(){
  local judge="$1" cfg="$2"
  local tag="${judge}.${cfg}"
  local log="$LOG/${tag}.log"
  : > "$log"
  echo "[js $(ts)] start $tag" | tee -a "$log"
  python -u scripts/rejudge_existing.py \
    --results_dir "results/$cfg" --users "$USERS" \
    --judge_model "$judge" --workers 8 \
    --out "$OUT/${tag}.json" \
    >> "$log" 2>&1
  echo "[js $(ts)] done  $tag -> $OUT/${tag}.json" | tee -a "$log"
}

# Two judges hit different providers (Azure gpt-5.4-mini vs Anthropic opus), so
# running both in parallel splits load by endpoint. Within a judge, 8 configs
# in parallel * 50/min limiter = ~400/min per endpoint — under caps.
for judge in "${JUDGES[@]}"; do
  for cfg in "${CFGS[@]}"; do
    run_one "$judge" "$cfg" &
  done
done
wait
echo "[js $(ts)] JUDGE-SENS SWEEP DONE" | tee -a "$LOG/driver.log"
