#!/bin/bash
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
OP6="over_personalization_chatbot_text,over_personalization_sensitive_event,over_personalization_context_shift"
echo "[abl6] START $(date +%H:%M)" > results/_logs/abl6/driver.log
mem(){ python -m evaluation.run_eval --user_id $3 --backend_dir backend --run_dir results/_abl6/$1/$3 \
  --mode llm_memory --model $2 --memory_builder_model $2 --judge_model gpt-5.5 --memory_token_cap 16384 --resume \
  --task "$OP6" --workers 6 > results/_logs/abl6/$1_$3.log 2>&1; }
lc(){ python -m evaluation.run_eval --user_id $3 --backend_dir backend_Mirror6 --run_dir results/_abl6/$1/$3 \
  --mode llm_longctx --model $2 --judge_model gpt-5.5 --task "$OP6" --workers 6 > results/_logs/abl6/$1_$3.log 2>&1; }
for u in 1 2 3 5 6 8 9 10 13 14; do
  for c in "dose3_gem gemini-3.5-flash mem" "dose3_gpt gpt-5.5 mem" "dose40_gem gemini-3.5-flash mem" "dose40_gpt gpt-5.5 mem" \
           "counts_gem gemini-3.5-flash mem" "counts_gpt gpt-5.5 mem" "swap_gem gemini-3.5-flash mem" "swap_gpt gpt-5.5 mem" \
           "mirror_gem gemini-3.5-flash lc" "mirror_gpt gpt-5.5 lc"; do
    set -- $c
    while [ "$(jobs -r|wc -l)" -ge 6 ]; do sleep 4; done
    if [ "$3" = lc ]; then lc $1 $2 $u & else mem $1 $2 $u & fi
  done
done
wait
echo "[abl6] ALL COMPLETE $(date +%H:%M)" >> results/_logs/abl6/driver.log
