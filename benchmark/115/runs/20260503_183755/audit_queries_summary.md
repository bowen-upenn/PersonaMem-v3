# Per-query Audit Summary
- model: `gpt-5.4-mini`
- total queries: 3

| task_type | dim | passed | failed | skipped | pass_rate |
| --- | --- | ---: | ---: | ---: | ---: |
| over_personalization_context_shift | context_required | 0 | 0 | 3 | — |
| over_personalization_context_shift | context_restraint | 0 | 3 | 0 | 0.0% |
| over_personalization_context_shift | example_vs_inferior | 0 | 3 | 0 | 0.0% |
| over_personalization_context_shift | gt_alignment | 0 | 0 | 3 | — |
| over_personalization_context_shift | naturalness | 0 | 3 | 0 | 0.0% |
| over_personalization_context_shift | privacy_leak | 0 | 0 | 3 | — |
| over_personalization_context_shift | schema_sanity | 3 | 0 | 0 | 100.0% |
| over_personalization_context_shift | sensitive_probe_placement | 0 | 0 | 3 | — |
