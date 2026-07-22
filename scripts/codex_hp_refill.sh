#!/bin/bash
set -u
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
ts(){ date +%H:%M:%S; }
for u in 1 2 3 5 6; do
  echo "[$(ts)] codex hp retry user $u"
  python -u -m evaluation.run_eval --user_id "$u" --backend_dir backend \
    --run_dir "results/codex_agent_gpt5.5/$u" --mode codex_agent --model gpt-5.5 \
    --judge_model gpt-5.5 --workers 1 --rate_limit 50 \
    --task hidden_persona_recommendation --retry_failed \
    > "/tmp/codex5/retry_$u.log" 2>&1
  hp_ok=$(python3 -c "import csv;csv.field_size_limit(2**31-1);print(sum(1 for r in csv.DictReader(open('results/codex_agent_gpt5.5/$u/results.csv')) if r['task_type']=='hidden_persona_recommendation' and r.get('status')=='ok'))")
  echo "[$(ts)]   user $u hp ok=$hp_ok"
done
echo "[$(ts)] codex hp refill COMPLETE"
