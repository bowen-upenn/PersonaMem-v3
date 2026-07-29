#!/usr/bin/env bash
# Cross-platform context ablation driver (EVAL.md "Cross-platform context
# ablation"): full cross-platform history vs current-platform-only history,
# llm_longctx gpt-5.5, the 10 matched eval personas, chatbot +
# social-feed-recommendation scenario tasks only.
#
# Usage:
#   scripts/run_platform_ablation.sh full_ctx          # arm 1: --history_scope full
#   scripts/run_platform_ablation.sh single_platform   # arm 2: --history_scope current_platform
#
# Arms run users sequentially (shared 50/min Azure rate budget; workers=8
# within a user per the eval default). Logs: /tmp/eval_regen/platform_ablation/.
set -euo pipefail
cd "$(dirname "$0")/.."

USERS=(1 2 3 5 6 8 9 10 13 14)
TASKS="chatbot_personalized_response,new_suggestions_chatbot,local_recommendation_geo_shift,personal_qa_hallucination,personalized_recommendation,at_ai_directive_followup,short_vs_long_term_lifecycle"
ROOT=results/ablation_platform_context
LOGDIR=/tmp/eval_regen/platform_ablation
mkdir -p "$LOGDIR"

ARM="${1:?arm required: full_ctx | single_platform}"
case "$ARM" in
  full_ctx)        SCOPE=full ;;
  single_platform) SCOPE=current_platform ;;
  *) echo "unknown arm: $ARM (want full_ctx | single_platform)" >&2; exit 1 ;;
esac

for u in "${USERS[@]}"; do
  echo "=== arm=$ARM user=$u start $(date '+%F %T') ==="
  python -m evaluation.run_eval \
    --user_id "$u" \
    --run_dir "$ROOT/$ARM/$u" \
    --mode llm_longctx --model gpt-5.5 --judge_model gpt-5.5 \
    --workers 8 \
    --task "$TASKS" \
    --history_scope "$SCOPE" \
    --resume \
    >> "$LOGDIR/${ARM}_${u}.stdout" 2>> "$LOGDIR/${ARM}_${u}.stderr"
  echo "=== arm=$ARM user=$u done  $(date '+%F %T') ==="
done
echo "=== arm=$ARM COMPLETE $(date '+%F %T') ==="
