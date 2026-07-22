#!/bin/bash
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
OP="over_personalization_chatbot_text,over_personalization_sensitive_event,over_personalization_context_shift,over_personalization_repetition_recsys,over_personalization_repetition_chatbot,over_personalization_sycophancy"
run_one(){
  u=$1
  python -m evaluation.run_eval --user_id $u --backend_dir backend \
    --run_dir results/_reeval_newmem/gemini_3_5_flash/$u \
    --mode llm_memory --model gemini-3.5-flash --memory_builder_model gemini-3.5-flash \
    --judge_model gpt-5.5 --memory_token_cap 4096 --resume --workers 8 \
    --task "$OP" \
    > results/_logs/gem_finish/$u.log 2>&1
  echo "[gem_finish] DONE persona $u $(date +%H:%M)" >> results/_logs/gem_finish/driver.log
}
echo "[gem_finish] START $(date +%H:%M) personas 6 8 9 10 13 14 | OVER-PERS tasks only | batch on, judge=gpt-5.5" > results/_logs/gem_finish/driver.log
for u in 6 8 9 10 13 14; do
  while [ "$(jobs -r | wc -l)" -ge 2 ]; do sleep 5; done
  run_one $u &
done
wait
echo "[gem_finish] ALL 6 COMPLETE $(date +%H:%M)" >> results/_logs/gem_finish/driver.log
