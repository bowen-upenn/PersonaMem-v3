#!/usr/bin/env bash
# Two REAL-GENERATION fixes (Azure gpt-5.5 only — no Claude Code, no Gemini):
#  A) llm_longctx_gpt5.5: generate the 21 appended proactive_overactive_check
#     rows (added to second-10 personas' test.json by the Step-28 regen; no
#     saved responses exist). --resume fills exactly the missing qids.
#  B) llm_longctx_gpt5.5_judged: strip the 50 repetition rows (empty-replay
#     0-scores from the overflow era) and regenerate fresh with the 700K
#     context budget + gpt-5.5 judge (user-approved "longctx repetition
#     re-run"). NOTE: these become fresh generations inside a replay-built
#     config — same model + judge, documented in EVAL.md.
set -uo pipefail
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
ts(){ date +%H:%M:%S; }

run_resume(){  # $1=mdir $2=uid $3=workers
  local mdir="$1" uid="$2" wk="$3"
  local d="results/$mdir/$uid"
  local log="results/_logs/gen.$mdir.$uid.stdout"
  : > "$log"; : > "results/_logs/gen.$mdir.$uid.stderr"
  python -u evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
    --run_dir "$d" --mode llm_longctx --model gpt-5.5 --judge_model gpt-5.5 \
    --workers "$wk" --resume > "$log" 2> "results/_logs/gen.$mdir.$uid.stderr" &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q "wrote .*summary.json" "$log" 2>/dev/null; then
      sleep 3; pkill -9 -P "$pid" 2>/dev/null||true; kill -9 "$pid" 2>/dev/null||true; break
    fi
    sleep 5
  done
  wait "$pid" 2>/dev/null||true
  echo "[gen $(ts)] done $mdir/$uid"
}

echo "[gen $(ts)] A) appended overactive rows (21) in llm_longctx_gpt5.5"
running=0
for uid in 105 115 209 229 282 461 655 760 835; do
  run_resume llm_longctx_gpt5.5 "$uid" 2 &
  running=$((running+1))
  if [ "$running" -ge 5 ]; then wait -n; running=$((running-1)); fi
done
wait

echo "[gen $(ts)] B) strip + regenerate repetition rows in llm_longctx_gpt5.5_judged"
python - <<'PY'
import csv, os
csv.field_size_limit(10_000_000)
REP={"over_personalization_repetition_chatbot","over_personalization_repetition_recsys"}
COLS=["query_id","seq","user_id","task_type","ts","metrics_json","status","duration_ms","error","agent_response"]
tot=0
for u in (1,2,3,5,6,8,9,10,13,14):
    f=f"results/llm_longctx_gpt5.5_judged/{u}/results.csv"
    if not os.path.exists(f): continue
    rows=list(csv.DictReader(open(f)))
    keep=[r for r in rows if r["task_type"] not in REP]
    n=len(rows)-len(keep)
    if n:
        tmp=f+".tmp"
        with open(tmp,"w",newline="",encoding="utf-8") as fh:
            w=csv.DictWriter(fh,fieldnames=COLS,extrasaction="ignore"); w.writeheader()
            for r in keep: w.writerow(r)
        os.replace(tmp,f); tot+=n
print(f"stripped {tot} repetition rows from _judged")
PY
running=0
for uid in 1 2 3 5 6 8 9 10 13 14; do
  run_resume llm_longctx_gpt5.5_judged "$uid" 2 &
  running=$((running+1))
  if [ "$running" -ge 3 ]; then wait -n; running=$((running-1)); fi
done
wait
echo "[gen $(ts)] GEN PASS DONE"
