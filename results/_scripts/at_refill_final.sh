#!/usr/bin/env bash
# Final agent_tools refill: the 194 limit-stripped rows (sonnet/14, opus/13,
# opus/14). Subscription-based claude subagents; rate-limit guard (0c0074f)
# errors rows instead of poisoning if the limit re-trips.
set -uo pipefail
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
ts(){ date +%H:%M:%S; }
run_one(){  # $1=suffix $2=model $3=uid
  local suf="$1" model="$2" uid="$3"
  local d="results/agent_tools_${suf}/$uid"
  local log="results/_logs/agent_tools_${suf}.$uid.stdout"
  : > "$log"; : > "results/_logs/agent_tools_${suf}.$uid.stderr"
  python -u evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
    --run_dir "$d" --mode agent_tools --claude_model "$model" --judge_model gpt-5.5 \
    --workers 6 --resume > "$log" 2> "results/_logs/agent_tools_${suf}.$uid.stderr" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q "wrote .*summary.json" "$log" 2>/dev/null; then
      sleep 3; pkill -9 -P "$pid" 2>/dev/null||true; kill -9 "$pid" 2>/dev/null||true; break
    fi
    sleep 5
  done
  wait "$pid" 2>/dev/null||true
  echo "[at $(ts)] done ${suf}/$uid"
}
echo "[at $(ts)] refilling sonnet/14 + opus/13 (parallel), then opus/14"
run_one sonnet4.6 sonnet 14 &
run_one opus4.8 opus 13 &
wait
run_one opus4.8 opus 14
echo "[at $(ts)] AT REFILL DONE"
