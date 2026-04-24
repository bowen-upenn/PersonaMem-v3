#!/usr/bin/env bash
# Cross-persona eval driver — runs scripts/run_eval.sh concurrently across
# all personas that have a benchmark/{uid}/queries.csv file.
#
# Usage:
#   scripts/run_eval_all.sh [--mode MODE] [--limit N] [--resume] [--dry_run]
# Env:
#   PARALLEL — concurrent personas (default 8)
#
# Each persona is processed independently — they do NOT share an overlay;
# every run_eval.sh invocation creates its own run_dir + writes.jsonl.
set -euo pipefail

PARALLEL="${PARALLEL:-8}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PERSONA_IDS=$(find benchmark -mindepth 2 -maxdepth 2 -name queries.csv \
    | sed -E 's|benchmark/([^/]+)/queries.csv|\1|' \
    | sort)

if [[ -z "$PERSONA_IDS" ]]; then
    echo "[run_eval_all] no benchmark/*/queries.csv found — run prepare_eval_data.py first" >&2
    exit 2
fi

COUNT=$(echo "$PERSONA_IDS" | wc -l | tr -d ' ')
echo "[run_eval_all] dispatching $COUNT personas, parallel=$PARALLEL, extra_args=$*"

# `-I{}` substitutes persona id; extra harness args are forwarded verbatim.
# Each subshell tees its own stdout/stderr into per-persona log files so
# a long run is debuggable without interleaving.
LOG_ROOT="${PM3_EVAL_LOG_DIR:-/tmp/pm3_eval_logs}"
mkdir -p "$LOG_ROOT"
echo "[run_eval_all] per-persona logs → $LOG_ROOT/<uid>.{stdout,stderr}"

echo "$PERSONA_IDS" | xargs -n1 -P "$PARALLEL" -I{} bash -c '
    uid="$1"; shift
    log="'"$LOG_ROOT"'/${uid}"
    scripts/run_eval.sh "$uid" "$@" >"${log}.stdout" 2>"${log}.stderr"
' _ {} "$@"

echo "[run_eval_all] done"
