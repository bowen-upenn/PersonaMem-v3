# qa_pair_audit — adversarially-verified example/inferior pair audit

Standardized quality check for the two headline chatbot-response tasks —
`chatbot_personalized_response` (should personalize) and
`over_personalization_chatbot_text` (should restrain). It verifies, for every
shipped pair, the three things a good pair must satisfy:

1. **Query naturalness** — the `user_query` reads like a real message from THIS persona.
2. **GOLD validity** — personalize-side: the `example_response` uses the held-out
   preference and is genuinely more helpful; restraint-side: the query truly needs
   no personalization and the GOLD stays generic.
3. **FOIL validity** — personalize-side: the `inferior_response` is a fair,
   clearly-worse *missed-personalization* foil (not broken for unrelated reasons);
   restraint-side: it genuinely *over-personalizes* with a real, unwarranted persona item.

See `AUDIT.md` → Slice A (dimensions #11, #16, #22–#25) for the specific failure
modes this surfaces. Methodology only; write findings to a separate report.

## Why two stages

A single LLM judge over-flags style nits and mis-reads on-topic injections as
leaks. Every stage-1 flag is therefore re-checked by **2 independent adversarial
verifiers** whose default stance is "the row is fine". A flag is kept only if it
survives: both agree ⇒ **confirmed**, one ⇒ **plausible**, both refute ⇒ dropped.

## Run (3 steps)

```bash
WORK=/tmp/qa_pair_audit/$(date +%s)     # any scratch dir

# 1. extract self-contained per-persona work-units (persona card + rows)
python scripts/qa_pair_audit/build_workunits.py --out "$WORK"
#    optional: --users 1 2 3 ...   --batch 5

# 2a. emit the multi-agent verify workflow (manifest embedded inline)
python scripts/qa_pair_audit/gen_workflow.py --workdir "$WORK"

# 2b. run "$WORK/verify_pairs_workflow.js" via the Claude Code Workflow tool,
#     and save its returned JSON object to "$WORK/wf_result.json".
#     (~170 agents for the 20-persona cohort; judge + 2x adversarial verify.)

# 3. aggregate -> per-axis pass rates + enriched findings.json
python scripts/qa_pair_audit/aggregate.py --result "$WORK/wf_result.json" --workdir "$WORK"
```

`findings.json` carries every surviving problem (status, axis, judge + verifier
reasons, and the real query / GOLD / FOIL text) — group by failure mode for the
report. The extractor and aggregator are deterministic; only step 2b needs the
harness.
