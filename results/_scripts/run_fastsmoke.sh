#!/bin/bash
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
TASKS="over_personalization_sensitive_event,over_personalization_chatbot_text"
echo "[fastsmoke] START $(date +%H:%M)" > results/_logs/abl/fastsmoke_driver.log
mirror(){ u=$1; python -m evaluation.run_eval --user_id $u --backend_dir backend_MirrorR --run_dir results/_abl/MirrorR/$u --mode llm_longctx --model gemini-3.5-flash --judge_model gpt-5.5 --task "$TASKS" --workers 6 > results/_logs/abl/MirrorR_$u.log 2>&1; }
doseup(){ u=$1; python -m evaluation.run_eval --user_id $u --backend_dir backend --run_dir results/_abl/DoseUp/$u --mode llm_memory --model gemini-3.5-flash --memory_builder_model gemini-3.5-flash --judge_model gpt-5.5 --memory_token_cap 8192 --resume --task "$TASKS" --workers 6 > results/_logs/abl/DoseUp_$u.log 2>&1; }
for u in 1 10; do while [ "$(jobs -r|wc -l)" -ge 2 ]; do sleep 5; done; mirror $u & done
wait
for u in 1 10; do while [ "$(jobs -r|wc -l)" -ge 2 ]; do sleep 5; done; doseup $u & done
wait
echo "[fastsmoke] ALL COMPLETE $(date +%H:%M)" >> results/_logs/abl/fastsmoke_driver.log
