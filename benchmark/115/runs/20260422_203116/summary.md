# PersonaMem-v3 Evaluation Summary
- user_id: 115
- mode: mcp_agent
- model: gpt-5-chat
- judge: disabled
- benchmark: v2 built_at=2026-04-23T00:30:43.060180+00:00 backend_hash=413f2666dbede64b rng_seed=0
- instance counts: {'test_items': 66, 'slate_ranking': 39, 'chatbot_response_proactive': 43, 'chatbot_response_control': 3, 'c1a_pairs': 5, 'c1b_sequences': 1, 'c2_scenarios': 5, 'c3_restraint': 39, 'c4_button_regen': 3}

## slate_ranking
- n: 1
- hit@1: 1.000
- hit@3: 1.000
- ild_topk: 0.996
- irrelevant_in_top1: 0.000
- mrr: 1.000
- ndcg@k: 1.000
- negative_in_top1: 0.000
- negative_in_top3: 1.000
- recall@1: 1.000
- recall@3: 1.000
- recall@5: 1.000

## d_negative_avoidance
_no results_
