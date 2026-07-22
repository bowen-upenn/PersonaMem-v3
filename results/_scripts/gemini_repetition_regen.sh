#!/usr/bin/env bash
# Gemini repetition regen (user-approved): real Gemini generation, gpt-5.5
# judge, 700K ctx budget. Cache-maximizing: workers=1 per persona so the
# per-process c1c/c1d cluster caches + same-prefix implicit prompt caching
# are fully reused across each persona's 5 rows; memory ledgers are cached
# on disk (0 build calls); parallelism comes from CONC across personas.
set -uo pipefail
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
ts(){ date +%H:%M:%S; }
CONC=5

strip_rep(){ python - "$1" <<'PY'
import csv, os, sys
csv.field_size_limit(10_000_000)
REP={"over_personalization_repetition_chatbot","over_personalization_repetition_recsys"}
COLS=["query_id","seq","user_id","task_type","ts","metrics_json","status","duration_ms","error","agent_response"]
f=sys.argv[1]
rows=list(csv.DictReader(open(f)))
keep=[r for r in rows if r["task_type"] not in REP]
if len(keep)<len(rows):
    tmp=f+".tmp"
    with open(tmp,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=COLS,extrasaction="ignore"); w.writeheader()
        for r in keep: w.writerow(r)
    os.replace(tmp,f)
print(len(rows)-len(keep), end="")
PY
}

run_one(){  # $1=mdir $2=mode $3=uid
  local mdir="$1" mode="$2" uid="$3"
  local d="results/$mdir/$uid"
  [ -f "$d/results.csv" ] || { echo "[gem $(ts)] skip $mdir/$uid"; return; }
  local n; n=$(strip_rep "$d/results.csv")
  local log="results/_logs/gem.$mdir.$uid.stdout"
  : > "$log"; : > "results/_logs/gem.$mdir.$uid.stderr"
  python -u evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
    --run_dir "$d" --mode "$mode" --model gemini-3.5-flash --judge_model gpt-5.5 \
    --workers 1 --resume > "$log" 2> "results/_logs/gem.$mdir.$uid.stderr" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q "wrote .*summary.json" "$log" 2>/dev/null; then
      sleep 3; pkill -9 -P "$pid" 2>/dev/null||true; kill -9 "$pid" 2>/dev/null||true; break
    fi
    sleep 5
  done
  wait "$pid" 2>/dev/null||true
  echo "[gem $(ts)] done $mdir/$uid (stripped $n)"
}

P10="1 2 3 5 6 8 9 10 13 14"
running=0
for spec in "llm_longctx_gemini3.5flash:llm_longctx" "llm_longctx_gemini3.5flash_judged:llm_longctx" "llm_memory_gemini3.5flash_judged:llm_memory"; do
  mdir="${spec%%:*}"; mode="${spec#*:}"
  for uid in $P10; do
    run_one "$mdir" "$mode" "$uid" &
    running=$((running+1))
    if [ "$running" -ge "$CONC" ]; then wait -n; running=$((running-1)); fi
  done
done
wait
echo "[gem $(ts)] GEMINI REPETITION DONE"
