#!/usr/bin/env python3
"""Debug why all proactive_unfulfilled_stated_need candidates get killed.

Reproduces Step 29 Stage 1 (gather candidates) + Stage 2 (LLM judge) for
ONE user, but prints every candidate's evidence + the LLM's JITAI card
so we can see the actual rejection reasons.

No backend writes — pure read + log.

Usage:
    python scripts/debug_unfulfilled_judge.py --user_id 115 [--model gpt-5-chat]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user_id", required=True)
    ap.add_argument("--backend_dir", default="backend")
    ap.add_argument("--model", default="gpt-5-chat")
    ap.add_argument("--rate_limit", type=int, default=30)
    args = ap.parse_args()

    from data_preparation.persona_agent import PersonaAgent
    from data_preparation import prompts, utils
    from query_llm import QueryLLM

    llm_client = QueryLLM(
        {"models": {"llm_model": args.model}},
        rate_limit_per_min=args.rate_limit,
    )

    agent = PersonaAgent(
        user_id=args.user_id,
        llm_client=llm_client,
        backend_dir=args.backend_dir,
        verbose=False,
    )

    # Load backend files
    base = Path(args.backend_dir) / args.user_id
    profile = json.loads((base / "profile.json").read_text())
    app_events = {}
    for app in ("instagram", "facebook", "threads", "chatbot"):
        p = base / f"{app}.json"
        app_events[app] = json.loads(p.read_text()) if p.exists() else []

    # Stage 1 — gather candidates
    sensitive_periods = agent._gather_sensitive_event_periods(profile)
    candidates = agent._gather_unfulfilled_stated_needs(
        app_events.get("chatbot", []), app_events,
    )
    print(f"Stage 1: {len(candidates)} candidates gathered\n")

    # Stage 2 — JITAI judge for each
    user_state_base = agent._build_proactive_user_state_base(profile)
    summary = {"kept": 0, "rejected": 0, "by_score": {}, "by_pass": {}}
    for i, c in enumerate(candidates):
        user_state = dict(user_state_base)
        user_state["sensitive_event_active"] = agent._is_in_sensitive_window(
            c["t_test"], sensitive_periods,
        )
        prompt = prompts.infer_proactive_trigger_prompt(user_state, c)
        try:
            resp = llm_client.query_llm(prompt, verbose=False, temperature=0.0)
            card = utils.extract_json_from_response(resp) or {}
        except Exception as exc:
            print(f"--- candidate {i} ({c.get('t_test_iso')}) ---")
            print(f"  ERROR: {exc}")
            continue

        score = card.get("eligibility_score")
        subt_pass = card.get("subtlety_check_pass")
        rule_pass = card.get("decision_rule_pass")
        rec = card.get("recommended_action_class")
        keep = agent._proactive_candidate_passes(
            "unfulfilled_stated_need", card, user_state["sensitive_event_active"]
        )

        sig = c.get("signal_evidence", {})
        q = (sig.get("user_question") or "").strip()
        lag = c.get("lag_days") or sig.get("lag_days")
        print(f"--- candidate {i} ---")
        print(f"  lag: {lag} day(s); asked_at: {sig.get('asked_at_iso','?')}")
        print(f"  user question: {q[:200]!r}")
        print(f"  hashtags: {sig.get('question_hashtags', [])}")
        print(f"  JITAI score: {score}  |  subtlety_pass: {subt_pass}  |  decision_rule_pass: {rule_pass}")
        print(f"  recommended_action_class: {rec}")
        print(f"  tailoring_variable: {(card.get('tailoring_variable') or '')[:160]}")
        print(f"  reasoning: {(card.get('reasoning') or '')[:300]}")
        print(f"  KEPT: {keep}")
        print()

        if keep: summary["kept"] += 1
        else:    summary["rejected"] += 1
        summary["by_score"][str(score)] = summary["by_score"].get(str(score), 0) + 1
        summary["by_pass"][f"subt={subt_pass},rule={rule_pass}"] = summary["by_pass"].get(
            f"subt={subt_pass},rule={rule_pass}", 0
        ) + 1

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nLLM usage: {llm_client.get_usage_totals()}")


if __name__ == "__main__":
    main()
