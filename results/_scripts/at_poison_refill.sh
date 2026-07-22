#!/usr/bin/env bash
# Refill the 254 rate-limit-poisoned agent_tools rows (2026-06-11 incident).
# Poisoned = proactive/restraint/repetition row with status=ok but
# cost_usd==0 AND cache_read_tokens==0: the subscription limit tripped and
# `claude -p` returned the ~18-token limit notice as the result; the
# proactive runner stored the parsed `{}` so text-based limit-stripping
# missed them. Detection is therefore usage-based, not marker-based.
#
# Phase 0 backs up + strips those rows (backup under
# results/_poison_backup_20260612/, never overwritten). Phase 1 re-runs each
# persona with --resume so ONLY the stripped query_ids regenerate. Sequential
# (sonnet personas, then opus) to stay under the subscription limit; the
# ClaudeZeroWorkError guard in claude_subagent.py now errors rows instead of
# poisoning if the limit re-trips (recover with --retry_failed).
set -uo pipefail
cd /vast/projects/cjtaylor/occam/bwjiang/PersonaMem-v3
BK=results/_poison_backup_20260612
ts(){ date +%H:%M:%S; }

strip_poisoned(){  # $1 = live results.csv  -> prints number stripped
  python - "$1" <<'PY'
import csv, json, os, sys
csv.field_size_limit(10_000_000)
FAM_PREFIXES = ("proactive_", "over_personalization_repetition_")
FAM_EXACT = {"restraint_sensitive_event_silence"}
COLS = ["query_id","seq","user_id","task_type","ts","metrics_json","status",
        "duration_ms","error","agent_response"]
f = sys.argv[1]
rows = list(csv.DictReader(open(f)))
def poisoned(r):
    t = r["task_type"]
    if not (t in FAM_EXACT or t.startswith(FAM_PREFIXES)):
        return False
    if r.get("status") != "ok":
        return False
    try:
        m = json.loads(r.get("metrics_json") or "{}")
    except Exception:
        return False
    return (m.get("cost_usd") or 0) == 0 and (m.get("cache_read_tokens") or 0) == 0
keep = [r for r in rows if not poisoned(r)]
if len(keep) < len(rows):
    tmp = f + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in keep:
            w.writerow(r)
    os.replace(tmp, f)
print(len(rows) - len(keep), end="")
PY
}

run_one(){  # $1=suffix $2=model $3=uid
  local suf="$1" model="$2" uid="$3"
  local d="results/agent_tools_${suf}/$uid"
  local log="results/_logs/at_poison.${suf}.$uid.stdout"
  : > "$log"; : > "results/_logs/at_poison.${suf}.$uid.stderr"
  python -u evaluation/run_eval.py --user_id "$uid" --backend_dir backend \
    --run_dir "$d" --mode agent_tools --claude_model "$model" --judge_model gpt-5.5 \
    --workers 6 --resume > "$log" 2> "results/_logs/at_poison.${suf}.$uid.stderr" &
  local pid=$!
  # run_eval occasionally lingers after writing summary.json — watch + kill.
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q "wrote .*summary.json" "$log" 2>/dev/null; then
      sleep 3; pkill -9 -P "$pid" 2>/dev/null||true; kill -9 "$pid" 2>/dev/null||true; break
    fi
    sleep 5
  done
  wait "$pid" 2>/dev/null||true
  echo "[atpoison $(ts)] done ${suf}/$uid"
}

SONNET_P="5 6 8 9 10 13 14"
OPUS_P="3 5 6 8 9 10 13 14"

echo "[atpoison $(ts)] phase 0: backup + strip"
total=0
for cfg in "sonnet4.6:$SONNET_P" "opus4.8:$OPUS_P"; do
  suf="${cfg%%:*}"; personas="${cfg#*:}"
  for uid in $personas; do
    d="results/agent_tools_${suf}/$uid"
    [ -f "$d/results.csv" ] || { echo "[atpoison $(ts)] MISSING $d/results.csv — skip"; continue; }
    mkdir -p "$BK/agent_tools_${suf}/$uid"
    if [ ! -f "$BK/agent_tools_${suf}/$uid/results.csv" ]; then
      cp "$d/results.csv" "$BK/agent_tools_${suf}/$uid/results.csv"
    fi
    n=$(strip_poisoned "$d/results.csv")
    total=$((total+n))
    echo "[atpoison $(ts)] stripped ${suf}/$uid: $n rows"
  done
done
echo "[atpoison $(ts)] phase 0 done: $total rows stripped (expected 254)"

echo "[atpoison $(ts)] phase 1: refill sonnet then opus (sequential)"
for uid in $SONNET_P; do run_one sonnet4.6 sonnet "$uid"; done
for uid in $OPUS_P;   do run_one opus4.8  opus   "$uid"; done
echo "[atpoison $(ts)] AT POISON REFILL DONE"
