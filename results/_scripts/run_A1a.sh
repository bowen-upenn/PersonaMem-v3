#!/bin/bash
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
OP="over_personalization_sensitive_event,over_personalization_chatbot_text,over_personalization_context_shift"
echo "[A1a] START $(date +%H:%M)" > results/_logs/abl/A1a_driver.log
for u in 1 10; do
  while [ "$(jobs -r | wc -l)" -ge 2 ]; do sleep 5; done
  python -m evaluation.run_eval --user_id $u --backend_dir backend \
    --run_dir results/_abl/A1a/$u --mode llm_memory --model gemini-3.5-flash --memory_builder_model gemini-3.5-flash \
    --judge_model gpt-5.5 --memory_token_cap 4096 --resume --task "$OP" --workers 6 \
    > results/_logs/abl/A1a_$u.log 2>&1 &
done
wait
echo "[A1a] ALL COMPLETE $(date +%H:%M)" >> results/_logs/abl/A1a_driver.log
