#!/usr/bin/env bash
# Memory-size ablation (GPT-5.5 only): rebuild the textual memory at HALF (2048)
# and DOUBLE (8192) the default 4096 token cap, eval ALL tasks, matched-10
# personas. Baseline = existing results/llm_memory_gpt5.5 (cap 4096). Throttled
# parallel pool.
set -u
PERS="1 2 3 5 6 8 9 10 13 14"
CAPS="2048 8192"
LOG=/tmp/eval_regen; mkdir -p "$LOG"
MAXJ="${MAXJ:-5}"            # concurrent run_eval processes (each uses --workers 8)

run_job() {
  local cap="$1" uid="$2"
  local rd="results/llm_memory_gpt5.5_cap${cap}/${uid}"
  mkdir -p "$rd"
  echo "[abl] START cap=$cap uid=$uid -> $rd"
  python -m evaluation.run_eval --user_id "$uid" --backend_dir backend --run_dir "$rd" \
    --mode llm_memory --model gpt-5.5 --memory_builder_model gpt-5.5 \
    --memory_token_cap "$cap" --judge_model gpt-5.5 --workers 8 \
    > "$LOG/cap${cap}.${uid}.stdout" 2> "$LOG/cap${cap}.${uid}.stderr"
  echo "[abl] DONE  cap=$cap uid=$uid rc=$?"
}

echo "[abl] caps=[$CAPS] personas=[$PERS] maxj=$MAXJ  $(date)"
for cap in $CAPS; do
  for uid in $PERS; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJ" ]; do wait -n; done
    run_job "$cap" "$uid" &
  done
done
wait
echo "[abl] ALL DONE  $(date)"
