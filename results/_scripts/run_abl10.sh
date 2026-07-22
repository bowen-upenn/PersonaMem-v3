#!/bin/bash
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
echo "[abl10] START $(date +%H:%M)" > results/_logs/abl10/driver.log
for arm in counts p3 norules p40; do
  for mk in gem gpt; do
    model=$([ "$mk" = gpt ] && echo gpt-5.5 || echo gemini-3.5-flash)
    for u in 1 2 3 5 6 8 9 10 13 14; do
      while [ "$(jobs -r | wc -l)" -ge 6 ]; do sleep 4; done
      python -m evaluation.run_eval --user_id $u --backend_dir backend \
        --run_dir results/_abl10/${arm}_${mk}/$u --mode llm_memory --model $model \
        --memory_builder_model $model --judge_model gpt-5.5 --memory_token_cap 8192 --resume \
        --task over_personalization_sensitive_event,over_personalization_chatbot_text --workers 6 \
        > results/_logs/abl10/${arm}_${mk}_${u}.log 2>&1 &
    done
  done
done
wait
echo "[abl10] ALL COMPLETE $(date +%H:%M)" >> results/_logs/abl10/driver.log
