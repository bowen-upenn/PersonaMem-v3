# Per-query Audit Summary
- model: `gpt-5.4-mini`
- total queries: 4

| task_type | dim | passed | failed | skipped | pass_rate |
| --- | --- | ---: | ---: | ---: | ---: |
| preference_removal_regen | context_required | 0 | 0 | 4 | — |
| preference_removal_regen | context_restraint | 0 | 0 | 4 | — |
| preference_removal_regen | example_vs_inferior | 0 | 0 | 4 | — |
| preference_removal_regen | gt_alignment | 0 | 0 | 4 | — |
| preference_removal_regen | naturalness | 0 | 0 | 4 | — |
| preference_removal_regen | privacy_leak | 0 | 0 | 4 | — |
| preference_removal_regen | schema_sanity | 4 | 0 | 0 | 100.0% |
| preference_removal_regen | sensitive_probe_placement | 0 | 0 | 4 | — |
