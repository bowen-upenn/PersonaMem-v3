#!/usr/bin/env bash
set -u
J=gpt-5.5; T=new_suggestions_chatbot; LOGD=/tmp/eval_regen/cb40
mkdir -p "$LOGD"
go(){ python evaluation/run_eval.py --user_id "$2" --backend_dir backend \
       --run_dir "results/$1/$2" --task "$T" --resume --judge_model "$J" $3 \
       > "$LOGD/$1.$2.log" 2>&1; echo "[$(date +%H:%M:%S)] $1 u$2 rc=$?" >> "$LOGD/_remain.log"; }
go mem0_gpt5.5 6 "--mode mem0 --model gpt-5.5 --workers 1 --reuse_mem0_store" &
go agent_tools_opus4.8 5 "--mode agent_tools --claude_model opus --workers 4" &
go agent_tools_opus4.8 6 "--mode agent_tools --claude_model opus --workers 4" &
go agent_tools_sonnet4.6 6 "--mode agent_tools --claude_model sonnet --workers 4" &
wait
echo "[$(date +%H:%M:%S)] REMAINING DONE" >> "$LOGD/_remain.log"
