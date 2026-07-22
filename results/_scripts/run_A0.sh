#!/bin/bash
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
OP="over_personalization_sensitive_event,over_personalization_chatbot_text,over_personalization_context_shift"
echo "[A0] START $(date +%H:%M)" > results/_logs/abl/A0_driver.log
launch(){ # arm model run_root
  for u in 1 10; do
    while [ "$(jobs -r | wc -l)" -ge 2 ]; do sleep 5; done
    python -m evaluation.run_eval --user_id $u --backend_dir backend \
      --run_dir $3/$u --mode llm_memory --model $2 --memory_builder_model $2 \
      --judge_model gpt-5.5 --memory_token_cap 4096 --resume --task "$OP" --workers 6 \
      > results/_logs/abl/A0_$1_$u.log 2>&1 &
  done
}
launch gem_ans_gptdoc gemini-3.5-flash results/_abl/A0_gem_ans_gptdoc
launch gpt_ans_gemdoc gpt-5.5          results/_abl/A0_gpt_ans_gemdoc
wait
echo "[A0] ALL COMPLETE $(date +%H:%M)" >> results/_logs/abl/A0_driver.log
