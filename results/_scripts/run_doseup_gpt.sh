#!/bin/bash
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
TASKS="over_personalization_sensitive_event,over_personalization_chatbot_text"
echo "[doseupgpt] START $(date +%H:%M)" > results/_logs/abl/doseupgpt_driver.log
for u in 1 10; do
  while [ "$(jobs -r|wc -l)" -ge 2 ]; do sleep 5; done
  python -m evaluation.run_eval --user_id $u --backend_dir backend --run_dir results/_abl/DoseUp_gpt/$u \
    --mode llm_memory --model gpt-5.5 --memory_builder_model gpt-5.5 --judge_model gpt-5.5 \
    --memory_token_cap 8192 --resume --task "$TASKS" --workers 6 > results/_logs/abl/DoseUp_gpt_$u.log 2>&1 &
done
wait
echo "[doseupgpt] ALL COMPLETE $(date +%H:%M)" >> results/_logs/abl/doseupgpt_driver.log
