#!/bin/bash
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
echo "[mirror_gpt] START $(date +%H:%M)" > results/_logs/abl/mirror_gpt_driver.log
for u in 1 10; do
  while [ "$(jobs -r|wc -l)" -ge 2 ]; do sleep 5; done
  python -m evaluation.run_eval --user_id $u --backend_dir backend_MirrorR \
    --run_dir results/_abl/MirrorR_gpt/$u --mode llm_longctx --model gpt-5.5 --judge_model gpt-5.5 \
    --task over_personalization_sensitive_event,over_personalization_chatbot_text --workers 6 \
    > results/_logs/abl/MirrorR_gpt_$u.log 2>&1 &
done
wait
echo "[mirror_gpt] ALL COMPLETE $(date +%H:%M)" >> results/_logs/abl/mirror_gpt_driver.log
