#!/usr/bin/env bash
# Re-eval the 3 over-personalization tasks affected by the minimal chatbot prompt,
# on the matched 10 personas, 5 LLM configs. Fresh run_dirs (does NOT touch existing
# results). Memory modes seed prebuilt checkpoints (+ --resume → fast-path skip rebuild,
# answers run fresh under the new prompt). Judge = gpt-5.5.
set -u
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
PERS="1 2 3 5 6 8 9 10 13 14"
TASKS="over_personalization_chatbot_text,over_personalization_sensitive_event,over_personalization_sycophancy"
OUT=results/_reeval_minprompt
LOG=results/_logs/reeval
mkdir -p "$LOG"

run_one() {
  local cfg="$1" u="$2" mode="$3" model="$4"; shift 4; local extra=("$@")
  local rd="$OUT/$cfg/$u"; mkdir -p "$rd"
  case "$cfg" in
    gpt_memory) mkdir -p "$rd/memory_states"; cp results/llm_memory_gpt5.5/$u/memory_states/*.json "$rd/memory_states/" 2>/dev/null;;
    gem_memory) mkdir -p "$rd/memory_states"; cp results/llm_memory_gemini3.5flash_ownmem/$u/memory_states/*.json "$rd/memory_states/" 2>/dev/null;;
  esac
  if [ "$cfg" = "gpt_mem0" ]; then export MEM0_DIR="$PWD/results/mem0_gpt5.5/$u/.mem0dir"; else unset MEM0_DIR; fi
  python -m evaluation.run_eval --user_id "$u" --backend_dir backend --run_dir "$rd" \
    --mode "$mode" --model "$model" --judge_model gpt-5.5 --memory_token_cap 4096 \
    --task "$TASKS" "${extra[@]}" \
    > "$LOG/$cfg.$u.stdout" 2> "$LOG/$cfg.$u.stderr" \
    && echo "[reeval] DONE $cfg/$u" || echo "[reeval] FAIL $cfg/$u (exit $?)"
}

run_cfg() {  # cfg mode model jobs -- extra...
  local cfg="$1" mode="$2" model="$3" jobs="$4"; shift 4; local extra=("$@")
  echo "[reeval] === $cfg ($mode/$model) start ==="
  local running=0
  for u in $PERS; do
    run_one "$cfg" "$u" "$mode" "$model" "${extra[@]}" &
    running=$((running+1))
    if [ "$running" -ge "$jobs" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
  done
  wait
  echo "[reeval] === $cfg done ==="
}

# GPT group (Azure — configs sequential to respect the rate limit)
(
  run_cfg gpt_longctx llm_longctx gpt-5.5 1 --workers 8
  run_cfg gpt_memory  llm_memory  gpt-5.5 1 --memory_builder_model gpt-5.5 --resume --workers 8
  run_cfg gpt_mem0    mem0        gpt-5.5 5 --workers 1
) &
GPT_PID=$!
# Gemini group (separate API — runs in parallel with GPT)
(
  run_cfg gem_longctx llm_longctx gemini-3.5-flash 2 --workers 8
  run_cfg gem_memory  llm_memory  gemini-3.5-flash 2 --memory_builder_model gemini-3.5-flash --resume --workers 8
) &
GEM_PID=$!
wait $GPT_PID $GEM_PID
echo "[reeval] ALL CONFIGS DONE"
