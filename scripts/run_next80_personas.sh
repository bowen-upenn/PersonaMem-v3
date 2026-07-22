#!/bin/bash
# ============================================================================
# Generate the "next 80" personas (extend the benchmark to 100 users).
# ============================================================================
#
# Backend: Azure OpenAI
#   flagship  = gpt-5.5        (deployment AZURE_OPENAI_DEPLOYMENT_NAME)
#   mini-tier = gpt-5.4-mini   (deployment literally named "gpt-5.4-mini")
# Both read from .env (AZURE_OPENAI_ENDPOINT / _KEY / _API_VERSION).
#
# User set: the 80 lowest-numbered IDs in data/all180_input.csv (17..118).
# These are fully DISJOINT from the existing evaluation cohort — nothing is overwritten.
#
# Properties:
#   * RESUMABLE  — skips any user whose backend/<uid>/profile.json already exists.
#   * PER-USER LOGS — /tmp/persona_regen/<uid>.{stdout,stderr} (CLAUDE.md convention).
#   * BOUNDED CONCURRENCY — CONCURRENCY users at a time (Azure dev-resource quota).
#   * No subset CSV is built (avoids the queries.csv-style csv round-trip footgun);
#     each invocation re-parses all180 and filters with --user_id. Parse overhead
#     (~8s/user) is negligible next to LLM wall-time.
#
# !!! DO NOT run without an explicit go-ahead — LLM calls are expensive. !!!
#
# USAGE:
#   # calibration smoke: one user, watch cost + quality first
#   USERS="17" CONCURRENCY=1 bash scripts/run_next80_personas.sh
#
#   # full run (background + tmux dashboard recommended — see footer)
#   bash scripts/run_next80_personas.sh
#
#   # knobs
#   CONCURRENCY=2 PARALLEL=30 bash scripts/run_next80_personas.sh
#   USERS="17 18 19" bash scripts/run_next80_personas.sh
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

INPUT_CSV="${INPUT_CSV:-data/all180_input.csv}"
MODEL="${MODEL:-gpt-5.5}"
MINI_MODEL="${MINI_MODEL:-gpt-5.4-mini}"
PARALLEL="${PARALLEL:-30}"        # internal concurrent LLM calls per user
CONCURRENCY="${CONCURRENCY:-3}"   # number of users processed simultaneously
RATE_LIMIT="${RATE_LIMIT:-50}"    # per-min budget passed to each QueryLLM client
LOGDIR="${LOGDIR:-/tmp/persona_regen}"
MASTER_LOG="$LOGDIR/_next80.master.log"

# The 80 lowest IDs in all180 (derived: sorted numeric, first 80).
DEFAULT_USERS="17 18 19 20 21 23 25 27 29 32 34 35 36 37 38 41 43 44 45 46 \
48 49 51 52 53 55 56 58 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 \
77 78 79 80 81 82 83 85 86 87 89 90 91 93 94 96 97 98 99 100 101 102 103 104 \
106 107 108 109 111 112 113 114 116 117 118"
USERS="${USERS:-$DEFAULT_USERS}"

mkdir -p "$LOGDIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

run_one() {
  local uid="$1"
  if [ -s "backend/$uid/profile.json" ]; then
    log "[skip] user $uid — backend/$uid/profile.json already exists"
    return 0
  fi
  log "[start] user $uid -> $LOGDIR/$uid.{stdout,stderr}"
  /usr/bin/time -v python scripts/run_persona_pipeline.py \
      --input_csv "$INPUT_CSV" --user_id "$uid" \
      --model "$MODEL" --mini_model "$MINI_MODEL" \
      --parallel "$PARALLEL" --rate_limit "$RATE_LIMIT" --verbose \
      > "$LOGDIR/$uid.stdout" 2> "$LOGDIR/$uid.stderr"
  local rc=$?
  if [ "$rc" -eq 0 ] && [ -s "backend/$uid/profile.json" ]; then
    log "[done]  user $uid"
  else
    log "[FAIL]  user $uid (rc=$rc) — tail $LOGDIR/$uid.stderr"
  fi
}

log "=== next80 run: ${CONCURRENCY} concurrent, parallel=${PARALLEL}, model=${MODEL}/${MINI_MODEL} ==="
n_total=$(echo $USERS | wc -w)
log "Requested $n_total users; resumable (existing profiles are skipped)."

# Bounded-concurrency dispatch using bash job control (wait -n).
for uid in $USERS; do
  while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do wait -n; done
  run_one "$uid" &
done
wait

# Summary.
done_n=0; fail_n=0
for uid in $USERS; do
  if [ -s "backend/$uid/profile.json" ]; then done_n=$((done_n+1)); else fail_n=$((fail_n+1)); fi
done
log "=== next80 complete: $done_n with profile.json, $fail_n missing (rerun to resume) ==="

# ----------------------------------------------------------------------------
# Recommended launch (background + live tmux dashboard tailing the master log):
#
#   nohup bash scripts/run_next80_personas.sh > /tmp/persona_regen/_next80.nohup 2>&1 &
#   tmux new-session -d -s next80 -x 220 -y 50 \
#       "tail -F /tmp/persona_regen/_next80.master.log"
#   tmux attach -t next80        # read-only: add -r ; detach: Ctrl-b d
#
# Per-user detail lives in /tmp/persona_regen/<uid>.{stdout,stderr}.
# ----------------------------------------------------------------------------
