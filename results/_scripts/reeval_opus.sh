#!/usr/bin/env bash
set -u
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
PERS="1 2 3 5 6 8 9 10 13 14"
TASKS="over_personalization_chatbot_text,over_personalization_sensitive_event,over_personalization_sycophancy"
LOG=results/_logs/reeval
JOBS=2
run_one(){ local u="$1"
  python -m evaluation.run_eval --user_id "$u" --backend_dir backend \
    --run_dir "results/_reeval_minprompt/opus_cc/$u" \
    --mode agent_tools --claude_model opus --judge_model gpt-5.5 \
    --task "$TASKS" --workers 4 \
    > "$LOG/opus_cc.$u.stdout" 2> "$LOG/opus_cc.$u.stderr" \
    && echo "[reeval-opus] DONE $u" || echo "[reeval-opus] FAIL $u (exit $?)"; }
running=0
for u in $PERS; do
  run_one "$u" &
  running=$((running+1))
  if [ "$running" -ge "$JOBS" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
done
wait
echo "[reeval-opus] ALL DONE"
