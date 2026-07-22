#!/usr/bin/env bash
# Re-eval ONLY the newly-added chatbot_personalized_response rows across the 7
# runnable configs (codex skipped — CLI missing), for the matched-10 personas.
# --task + --resume => existing rows (unchanged query_ids) are skipped; only the
# new rows run and are appended to each results.csv. Memory/Mem0 stores are
# reused (no rebuild). Bounded parallel pool keeps the shared gpt-5.5 judge
# under the Azure rate cap.
set -u
cd "$(dirname "$0")/.."
PERSONAS=(1 2 3 5 6 8 9 10 13 14)
MAXPAR="${MAXPAR:-6}"
TASK=chatbot_personalized_response
mkdir -p /tmp/cpr_eval

# config_dir | mode | model-flag | workers
CONFIGS=(
  "llm_longctx_gpt5.5_judged|llm_longctx|--model gpt-5.5|4"
  "llm_memory_gpt5.5|llm_memory|--model gpt-5.5|4"
  "mem0_gpt5.5|mem0|--model gpt-5.5|4"
  "llm_longctx_gemini3.5flash_judged|llm_longctx|--model gemini-3.5-flash|6"
  "llm_memory_gemini3.5flash_judged|llm_memory|--model gemini-3.5-flash|6"
  "agent_tools_opus4.8|agent_tools|--claude_model opus|1"
  "agent_tools_sonnet4.6|agent_tools|--claude_model sonnet|1"
)

running=0
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r config mode mflag workers <<< "$cfg"
  for uid in "${PERSONAS[@]}"; do
    (
      python -m evaluation.run_eval --user_id "$uid" --backend_dir backend \
        --run_dir "results/$config/$uid" --mode "$mode" $mflag \
        --judge_model gpt-5.5 --workers "$workers" --memory_token_cap 4096 \
        --task "$TASK" --resume
    ) > "/tmp/cpr_eval/${config}.${uid}.log" 2>&1 &
    running=$((running + 1))
    if (( running >= MAXPAR )); then wait -n; running=$((running - 1)); fi
  done
done
wait
echo "ALL CPR EVAL JOBS DONE"
