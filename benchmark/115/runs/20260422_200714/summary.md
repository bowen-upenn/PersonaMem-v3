# PersonaMem-v3 Evaluation Summary
- user_id: 115
- mode: agent_tools
- model: gpt-5-chat
- judge: disabled
- benchmark: v2 built_at=2026-04-22T21:30:07.451004+00:00 backend_hash=cd65d068e4193e77 rng_seed=0
- instance counts: {'test_items': 66, 'slate_ranking': 39, 'chatbot_response_proactive': 43, 'chatbot_response_control': 3, 'c1a_pairs': 5, 'c1b_sequences': 1, 'c2_scenarios': 5, 'c3_restraint': 39, 'c4_button_regen': 3}

## chatbot_response_proactive
- n: 1
- avoid_leak_flag: 0.000
- avoid_leak_rate: 0.000
- behavioral_false_hit_rate: 0.000
- behavioral_hit_rate: 0.023
- behavioral_miss_rate: 0.977
- carve_out_respect: 1.000
- held_out_hit: 0.000
- held_out_score: 0.061
- num_avoid: 69.000
- num_target: 246.000
- privacy_leak_hard_fail: 0.000
- privacy_leak_rate: 0.000
- target_match_recall: 0.016
- target_secondary_match: 0.016
- top_k_alignment_rate: 0.000
- top_k_matched: 0.000
