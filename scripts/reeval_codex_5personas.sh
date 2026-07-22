#!/bin/bash
# Extend codex_agent_gpt5.5 to the full matched-10 by running the 5 MISSING
# personas {8,9,10,13,14} on the FULL task suite (no --task filter), so codex's
# whole column (incl. Overall) becomes 10-persona comparable. Personas in
# parallel, 2 workers each (~10 concurrent codex calls).
set -u
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
ts(){ date +%H:%M:%S; }
echo "[$(ts)] codex 5-persona extension START"
for u in 8 9 10 13 14; do
  python -u -m evaluation.run_eval --user_id "$u" --backend_dir backend \
    --run_dir "results/codex_agent_gpt5.5/$u" --mode codex_agent --model gpt-5.5 \
    --judge_model gpt-5.5 --workers 2 --rate_limit 50 --prune_invalid --resume \
    > "/tmp/codex5/$u.log" 2>&1 &
done
wait
echo "[$(ts)] codex 5-persona extension COMPLETE"
