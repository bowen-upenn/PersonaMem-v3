#!/bin/bash
# ============================================================================
# gpt-5.5 judge-replay for the three judge-OFF runs. Re-scores each model's
# SAVED responses (no regeneration) with the gpt-5.5 judge, writing to a
# *_judged sibling dir. Runs the modes sequentially (shared Azure gpt-5.5 judge
# deployment); judge prompts are small (focused evidence, not full history), so
# CONCURRENCY can be higher than the 400K-token gen phase without 429 storms.
#
# Output: results/<run>_judged/{uid}/   Logs: /tmp/eval_gemini/<mode>_<uid>.log
# ============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p /tmp/eval_gemini
CONC="${CONCURRENCY:-4}"

run_mode () {  # $1=MODE $2=MODEL $3=SRC_ROOT $4=OUT_ROOT
  echo "=== judge-replay: $3 -> $4 (mode=$1 model=$2 judge=gpt-5.5) $(date) ==="
  MODE="$1" MODEL="$2" REPLAY_FROM="$3" OUT_ROOT="$4" \
    JUDGE_FLAG="--enable_llm_judge" JUDGE_MODEL="gpt-5.5" CONCURRENCY="$CONC" \
    bash scripts/run_gemini35_eval.sh
}

run_mode llm_longctx gemini-3.5-flash results/llm_longctx_gemini3.5flash  results/llm_longctx_gemini3.5flash_judged
run_mode llm_memory  gemini-3.5-flash results/llm_memory_gemini3.5flash   results/llm_memory_gemini3.5flash_judged
run_mode llm_longctx gpt-5.5          results/llm_longctx_gpt5.5          results/llm_longctx_gpt5.5_judged
echo "=== ALL judge-replay modes complete $(date) ==="
