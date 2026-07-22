#!/usr/bin/env bash
# v2: finish raw longctx gemini repetition regen (gpt-5.5 judge ON), regen
# memory _judged (tiny), then COPY raw's freshly judged repetition rows into
# longctx _judged instead of regenerating them (~78M Gemini tokens saved —
# identical generation+judge regime, consistent with the replay philosophy).
set -uo pipefail
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
ts(){ date +%H:%M:%S; }

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

run_one(){  # $1=mdir $2=mode $3=uid $4=dostrip
  local mdir="$1" mode="$2" uid="$3" dostrip="$4"
  local d="results/$mdir/$uid"
  local n=0
  [ "$dostrip" = "strip" ] && n=$(strip_rep "$d/results.csv")
  local log="results/_logs/gem.$mdir.$uid.stdout"
  : > "$log"; : > "results/_logs/gem.$mdir.$uid.stderr"
  python -u evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
    --run_dir "$d" --mode "$mode" --model gemini-3.5-flash --judge_model gpt-5.5 \
    --workers 1 --resume > "$log" 2> "results/_logs/gem.$mdir.$uid.stderr" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    grep -q "wrote .*summary.json" "$log" 2>/dev/null && { sleep 3; pkill -9 -P "$pid" 2>/dev/null||true; kill -9 "$pid" 2>/dev/null||true; break; }
    sleep 5
  done
  wait "$pid" 2>/dev/null||true
  echo "[gem2 $(ts)] done $mdir/$uid (stripped $n)"
}

# A) finish raw longctx (resume picks up only missing repetition rows)
running=0
for uid in 1 2 3 5 6 8 9 10 13 14; do
  run_one llm_longctx_gemini3.5flash llm_longctx "$uid" nostrip &
  running=$((running+1)); [ "$running" -ge 5 ] && { wait -n; running=$((running-1)); }
done
wait

# B) memory _judged regen (tiny)
running=0
for uid in 1 2 3 5 6 8 9 10 13 14; do
  run_one llm_memory_gemini3.5flash_judged llm_memory "$uid" strip &
  running=$((running+1)); [ "$running" -ge 5 ] && { wait -n; running=$((running-1)); }
done
wait

# C) copy raw's judged repetition rows into longctx _judged (replace stale)
python - <<'PY'
import csv, os
csv.field_size_limit(10_000_000)
REP={"over_personalization_repetition_chatbot","over_personalization_repetition_recsys"}
COLS=["query_id","seq","user_id","task_type","ts","metrics_json","status","duration_ms","error","agent_response"]
tot=0
for u in (1,2,3,5,6,8,9,10,13,14):
    src=f"results/llm_longctx_gemini3.5flash/{u}/results.csv"
    dst=f"results/llm_longctx_gemini3.5flash_judged/{u}/results.csv"
    fresh=[r for r in csv.DictReader(open(src)) if r["task_type"] in REP]
    rows=[r for r in csv.DictReader(open(dst)) if r["task_type"] not in REP]
    rows.extend(fresh)
    rows.sort(key=lambda r:int(r["seq"]))
    tmp=dst+".tmp"
    with open(tmp,"w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=COLS,extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow(r)
    os.replace(tmp,dst); tot+=len(fresh)
print(f"copied {tot} judged repetition rows raw -> longctx _judged")
PY
echo "[gem2 $(ts)] GEMINI V2 DONE"
