# PersonaMem-v3 Evaluation Summary
- user_id: 115
- mode: mcp_agent
- model: gpt-5-chat
- judge: disabled
- benchmark: v2 built_at=2026-04-23T01:05:24.552159+00:00 backend_hash=413f2666dbede64b rng_seed=0
- instance counts: {'test_items': 66, 'slate_ranking': 39, 'chatbot_response_proactive': 43, 'chatbot_response_control': 3, 'c1a_pairs': 5, 'c1b_sequences': 1, 'c2_scenarios': 5, 'c3_restraint': 39, 'c4_button_regen': 3, 't6_community_digest': 4, 't7_moment_recommendation': 4, 't8_dm_digest': 3, 't9_cross_app_repost': 1, 't10_auto_reply': 6, 't11_vague_refind': 4, 't12_agent_composed_post': 9, 't13_chatbot_dispatch': 3, 't14_draft_audit': 3, 't15_collection_curation': 3, 't16_group_dm_summary': 4, 't17_wrong_recipient': 1, 't18_proactive_daily': 1, 't19_trending_alert': 1}

## t14_draft_audit
- n: 1
- final_state_rules_failed: 0.000
- final_state_rules_passed: 3.000
- pr_personalization_hard_fail_count: 1.000
- pr_privacy_leak_hard_fail: 0.000
- pr_privacy_leak_rate: 0.000
- pr_stale_preference_use_hard_fail: 1.000
- pr_stale_preference_use_rate: 0.333
- tool_call_rules_fail: 0.000
- tool_call_rules_pass: 2.000
