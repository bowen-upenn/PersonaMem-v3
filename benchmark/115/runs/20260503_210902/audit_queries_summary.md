# Per-query Audit Summary
- model: `gpt-5.4-mini`
- total queries: 6

| task_type | dim | passed | failed | skipped | pass_rate |
| --- | --- | ---: | ---: | ---: | ---: |
| agentic_composed_post | context_required | 0 | 0 | 6 | — |
| agentic_composed_post | context_restraint | 0 | 0 | 6 | — |
| agentic_composed_post | example_vs_inferior | 0 | 6 | 0 | 0.0% |
| agentic_composed_post | gt_alignment | 0 | 0 | 6 | — |
| agentic_composed_post | naturalness | 0 | 0 | 6 | — |
| agentic_composed_post | privacy_leak | 0 | 0 | 6 | — |
| agentic_composed_post | schema_sanity | 6 | 0 | 0 | 100.0% |
| agentic_composed_post | sensitive_probe_placement | 0 | 0 | 6 | — |
| agentic_composed_post | tool_call_validity | 0 | 0 | 6 | — |
