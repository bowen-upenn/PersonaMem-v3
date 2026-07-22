#!/bin/bash
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
echo "[lccs] START $(date +%H:%M)" > results/_logs/abl6/lccs_driver.log
for mk in gem gpt; do model=$([ $mk = gpt ] && echo gpt-5.5 || echo gemini-3.5-flash)
  for u in 1 2 3 5 6 8 9 10 13 14; do
    while [ "$(jobs -r|wc -l)" -ge 6 ]; do sleep 4; done
    python -m evaluation.run_eval --user_id $u --backend_dir backend --run_dir results/_abl6/lcfull_cs_$mk/$u \
      --mode llm_longctx --model $model --judge_model gpt-5.5 --task over_personalization_context_shift --workers 6 \
      > results/_logs/abl6/lccs_${mk}_$u.log 2>&1 &
  done
done
wait
echo "[lccs] ALL COMPLETE $(date +%H:%M)" >> results/_logs/abl6/lccs_driver.log
