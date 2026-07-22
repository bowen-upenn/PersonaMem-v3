#!/bin/bash
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
OP3="over_personalization_chatbot_text,over_personalization_sensitive_event,over_personalization_context_shift"
echo "[ablOLD] START $(date +%H:%M)" > results/_logs/ablOLD/driver.log
mem(){ python -m evaluation.run_eval --user_id $3 --backend_dir backend --run_dir results/_ablOLD/$1/$3 \
  --mode llm_memory --model $2 --memory_builder_model $2 --judge_model gpt-5.5 --memory_token_cap 16384 --resume \
  --task "$OP3" --workers 6 > results/_logs/ablOLD/$1_$3.log 2>&1; }
for u in 1 2 3 5 6 8 9 10 13 14; do
  for c in "base_gem gemini-3.5-flash" "base_gpt gpt-5.5" "dpriv_gem gemini-3.5-flash" "dpriv_gpt gpt-5.5" \
           "dord_gem gemini-3.5-flash" "dord_gpt gpt-5.5" "counts_gem gemini-3.5-flash" "counts_gpt gpt-5.5" \
           "swap_gem gemini-3.5-flash" "swap_gpt gpt-5.5"; do
    set -- $c
    while [ "$(jobs -r|wc -l)" -ge 6 ]; do sleep 4; done
    mem $1 $2 $u &
  done
done
wait
echo "[ablOLD] ALL COMPLETE $(date +%H:%M)" >> results/_logs/ablOLD/driver.log
