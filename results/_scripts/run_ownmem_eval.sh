#!/usr/bin/env bash
# Re-eval gemini-3.5-flash llm_memory using gemini's OWN prebuilt memory
# (seeded into run_dir/memory_states; --resume → 0 build calls). Judge = gpt-5.5.
set -u
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
PERS="2 3 5 6 8 9 10 13 14"
JOBS=4
run_one() {
  local u=$1
  python -m evaluation.run_eval --user_id "$u" --backend_dir backend \
    --run_dir "results/llm_memory_gemini3.5flash_ownmem/$u" \
    --mode llm_memory --model gemini-3.5-flash --memory_builder_model gemini-3.5-flash \
    --judge_model gpt-5.5 --resume --workers 8 \
    > "results/_logs/ownmem.$u.stdout" 2> "results/_logs/ownmem.$u.stderr" \
    && echo "[ownmem] DONE u$u" || echo "[ownmem] FAIL u$u (exit $?)"
}
running=0
for u in $PERS; do
  run_one "$u" &
  running=$((running+1))
  if [ "$running" -ge "$JOBS" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
done
wait
echo "[ownmem] all 9 remaining personas finished"
