#!/usr/bin/env bash
set -uo pipefail
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
# wait for round-4 replay to finish
i=0
until grep -q "PROACTIVE REPLAY DONE" results/_logs/replay.driver.log 2>/dev/null; do
  sleep 20; i=$((i+1)); [ "$i" -ge 90 ] && { echo "chain: replay wait timeout"; exit 1; }
done
echo "chain: replay done, launching gen pass"
bash results/_scripts/gen_appended_and_judged_repetition.sh
