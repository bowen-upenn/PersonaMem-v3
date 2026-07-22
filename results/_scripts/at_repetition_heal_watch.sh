#!/usr/bin/env bash
# Watch the in-flight agent_tools strip-refill (other session's driver), then:
#  1) verify completeness per persona vs test.json,
#  2) refill any rows it left behind (--resume; rate-limit guard 0c0074f active,
#     so a re-trip errors rows instead of poisoning),
#  3) final re-aggregate.
set -uo pipefail
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
ts(){ date +%H:%M:%S; }

# 1) wait until no agent_tools eval procs remain (other session's refill done)
echo "[heal $(ts)] waiting for in-flight agent_tools refill to finish..."
i=0
while pgrep -f "run_eval[.]py.*agent_tools" >/dev/null 2>&1; do
  sleep 60; i=$((i+1))
  [ "$i" -ge 300 ] && { echo "[heal $(ts)] still running after 5h — bailing to report"; break; }
done
echo "[heal $(ts)] no agent_tools procs; settling 60s"; sleep 60

# 2) completeness check + targeted refill of any gaps
missing_runs=$(python - <<'PY'
import csv, json, os
csv.field_size_limit(10_000_000)
out=[]
for suf in ("sonnet4.6","opus4.8"):
    for u in (1,2,3,5,6,8,9,10,13,14):
        f=f"results/agent_tools_{suf}/{u}/results.csv"
        try: have={r["query_id"] for r in csv.DictReader(open(f))}
        except FileNotFoundError: have=set()
        cur={it.get("query_id") for it in json.load(open(f"backend/{u}/test.json")) if isinstance(it,dict)}
        if cur-have: out.append(f"{suf}:{u}:{len(cur-have)}")
print(" ".join(out))
PY
)
if [ -n "$missing_runs" ]; then
  echo "[heal $(ts)] gaps remain: $missing_runs — refilling"
  for spec in $missing_runs; do
    suf="${spec%%:*}"; rest="${spec#*:}"; uid="${rest%%:*}"
    model="sonnet"; [ "$suf" = "opus4.8" ] && model="opus"
    d="results/agent_tools_${suf}/$uid"
    log="results/_logs/heal.$suf.$uid.stdout"
    : > "$log"; : > "results/_logs/heal.$suf.$uid.stderr"
    python -u evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
      --run_dir "$d" --mode agent_tools --claude_model "$model" --judge_model gpt-5.5 \
      --workers 6 --resume > "$log" 2> "results/_logs/heal.$suf.$uid.stderr" &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
      grep -q "wrote .*summary.json" "$log" 2>/dev/null && { sleep 3; pkill -9 -P "$pid" 2>/dev/null||true; kill -9 "$pid" 2>/dev/null||true; break; }
      sleep 5
    done
    wait "$pid" 2>/dev/null||true
    echo "[heal $(ts)] refilled $suf/$uid"
  done
else
  echo "[heal $(ts)] no gaps — other session's refill covered everything"
fi

# 3) final aggregate
python scripts/aggregate_eval.py --results_root results \
  --modes llm_longctx_gpt5.5_judged,llm_memory_gpt5.5,mem0_gpt5.5,llm_longctx_gemini3.5flash_judged,llm_memory_gemini3.5flash_judged,agent_tools_sonnet4.6,agent_tools_opus4.8 \
  > results/_logs/aggregate.heal.log 2>&1 || true
echo "[heal $(ts)] HEAL DONE"
