# Per-query Audit Summary
- model: `gpt-5.4-mini`
- total queries: 2

| task_type | dim | passed | failed | skipped | pass_rate |
| --- | --- | ---: | ---: | ---: | ---: |
| active_mistake_prevention | context_required | 0 | 0 | 2 | — |
| active_mistake_prevention | context_restraint | 0 | 0 | 2 | — |
| active_mistake_prevention | example_vs_inferior | 0 | 2 | 0 | 0.0% |
| active_mistake_prevention | gt_alignment | 0 | 0 | 2 | — |
| active_mistake_prevention | naturalness | 0 | 2 | 0 | 0.0% |
| active_mistake_prevention | privacy_leak | 0 | 0 | 2 | — |
| active_mistake_prevention | schema_sanity | 2 | 0 | 0 | 100.0% |
| active_mistake_prevention | sensitive_probe_placement | 0 | 0 | 2 | — |
