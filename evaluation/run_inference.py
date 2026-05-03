"""LEGACY CLI orchestrator for the eval harness.

Reads the pre-built `benchmark/{user_id}/benchmark.json` artifact, which
the consolidated `scripts/prepare_eval_data.py` no longer writes (single
source of truth is now `benchmark/{user_id}/queries.csv`). Use
`evaluation/run_eval.py` for new work; this module is preserved for
backward compatibility with snapshots that still have a benchmark.json
sidecar on disk.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.backend_query import BackendQuery
from evaluation.build_benchmark import (
    BENCHMARK_VERSION,
    compute_backend_hash,
    default_benchmark_path,
)
from evaluation.inference_utils import SnapshotCache
from evaluation.tasks import slate_ranking, chatbot_response, over_personalization, agentic_tasks
from evaluation import metrics as metrics_mod


AGENTIC_TASK_IDS = [
    "t6_community_digest", "t7_moment_recommendation", "t8_dm_digest",
    "t9_cross_app_repost", "t10_auto_reply", "t11_vague_refind",
    "t12_agent_composed_post", "t13_send_post", "t14_draft_audit",
    "t16_group_dm_summary", "t17_wrong_recipient",
    "t18_proactive_daily", "t19_trending_alert",
]

TASK_ALIASES = {
    "all": [
        "slate_ranking",
        "chatbot_response_proactive",
        "chatbot_response_control",
        "c1a_pairs",
        "c1b_sequences",
        "c2_scenarios",
        "c3_restraint",
        "c4_button_regen",
        "e2_at_ai_followup",
        "e3_daily_briefing_multi",
        "e5_horizon_lifecycle",
        # e4_google_search is NOT in "all" by default — opt in via --task e4
        # or --task all_with_e4 (requires --enable_e4 too).
        *AGENTIC_TASK_IDS,
    ],
    "all_with_e4": [
        "slate_ranking",
        "chatbot_response_proactive",
        "chatbot_response_control",
        "c1a_pairs",
        "c1b_sequences",
        "c2_scenarios",
        "c3_restraint",
        "c4_button_regen",
        "e2_at_ai_followup",
        "e3_daily_briefing_multi",
        "e4_google_search",
        "e5_horizon_lifecycle",
        *AGENTIC_TASK_IDS,
    ],
    "a": ["slate_ranking"],
    "b": ["chatbot_response_proactive", "chatbot_response_control"],
    "b_proactive": ["chatbot_response_proactive"],
    "b_control": ["chatbot_response_control"],
    "c": ["c1a_pairs", "c1b_sequences", "c2_scenarios", "c3_restraint", "c4_button_regen"],
    "c1": ["c1a_pairs", "c1b_sequences"],
    "c1a": ["c1a_pairs"],
    "c1b": ["c1b_sequences"],
    "c2": ["c2_scenarios"],
    "c3": ["c3_restraint"],
    "c4": ["c4_button_regen"],
    "e": ["e2_at_ai_followup", "e3_daily_briefing_multi", "e5_horizon_lifecycle"],
    "e2": ["e2_at_ai_followup"],
    "e3": ["e3_daily_briefing_multi"],
    "e4": ["e4_google_search"],
    "e5": ["e5_horizon_lifecycle"],
    "agentic": AGENTIC_TASK_IDS,
    # Individual agentic shortcuts: t6, t7, ..., t19.
    **{tid.split("_", 1)[0]: [tid] for tid in AGENTIC_TASK_IDS},
}

MODES = ("agent_tools", "mcp_agent", "agent_longctx", "llm_longctx")


def _build_llm_clients(args):
    """Only the llm_longctx mode + optional judge go through QueryLLM; the
    Claude Code modes use the subscription-authed `claude` CLI directly."""
    if args.dry_run or args.mode in ("agent_tools", "agent_longctx", "mcp_agent"):
        baseline_client = None
    else:
        from query_llm import QueryLLM
        baseline_client = QueryLLM({"models": {"llm_model": args.model}}, rate_limit_per_min=args.rate_limit)
    judge_client = None
    if args.enable_llm_judge and not args.dry_run:
        from query_llm import QueryLLM
        judge_client = QueryLLM({"models": {"llm_model": args.judge_model}}, rate_limit_per_min=args.rate_limit)
    return baseline_client, judge_client


def _resolve_tasks(task_arg: str) -> list[str]:
    if task_arg in TASK_ALIASES:
        return TASK_ALIASES[task_arg]
    return [task_arg]


BENCHMARK_TASK_KEYS = {
    "slate_ranking": "slate_ranking",
    # Task B v2 — two arms.
    "chatbot_response_proactive": "chatbot_response_proactive",
    "chatbot_response_control":   "chatbot_response_control",
    # Task C v2.
    "c1a_pairs":        "c1a_pairs",
    "c1b_sequences":    "c1b_sequences",
    "c2_scenarios":     "c2_scenarios",
    "c3_restraint":     "c3_restraint",
    "c4_button_regen":  "c4_button_regen",
    # Task E (R9+): @ai proactive recommendation, multi-day briefing, etc.
    "e2_at_ai_followup":"e2_at_ai_followup",
    "e3_daily_briefing_multi":"e3_daily_briefing_multi",
    "e4_google_search":"e4_google_search",
    "e5_horizon_lifecycle":"e5_horizon_lifecycle",
    # Agentic T6-T19 — key in benchmark JSON is same as task_id.
    **{tid: tid for tid in AGENTIC_TASK_IDS},
}


def _run_task(name, instances, user_id, bq, llm_client, judge_client, args, snapshot_cache):
    common = dict(
        instances=instances,
        user_id=user_id,
        bq=bq,
        llm_client=llm_client,
        judge_client=judge_client,
        mode=args.mode,
        snapshot_cache=snapshot_cache,
        model_name=args.model,
        claude_model=args.claude_model,
        context_budget=args.context_budget,
        enable_llm_judge=args.enable_llm_judge,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    if name == "slate_ranking":
        return slate_ranking.run_task_a(**common)
    if name in ("chatbot_response_proactive", "chatbot_response_control"):
        return chatbot_response.run_task_b(**common)
    if name == "c1a_pairs":
        return over_personalization.run_task_c1a(**common)
    if name == "c1b_sequences":
        return over_personalization.run_task_c1b(**common)
    if name == "c2_scenarios":
        return over_personalization.run_task_c2(**common)
    if name == "c3_restraint":
        return over_personalization.run_task_c3(**common)
    if name == "c4_button_regen":
        return over_personalization.run_task_c4(**common)
    if name == "e2_at_ai_followup":
        from evaluation.tasks import e2_at_ai_followup as _e2
        return _e2.run_e2_at_ai_followup(**common)
    if name == "e3_daily_briefing_multi":
        from evaluation.tasks import e3_daily_briefing_multi as _e3
        return _e3.run_e3_daily_briefing_multi(**common)
    if name == "e4_google_search":
        from evaluation.tasks import e4_google_search as _e4
        return _e4.run_e4_google_search(**common)
    if name == "e5_horizon_lifecycle":
        from evaluation.tasks import e5_horizon_lifecycle as _e5
        return _e5.run_e5_horizon_lifecycle(**common)
    if name in AGENTIC_TASK_IDS:
        return agentic_tasks.run_task(task_id=name, **common)
    raise ValueError(f"unknown task: {name}")


def _summarize(all_results: dict[str, list[dict]]) -> dict:
    summary: dict = {}
    for task, rows in all_results.items():
        rows = [r for r in rows if r.get("metrics")]
        if not rows:
            summary[task] = {"n": 0}
            continue
        keys = set()
        for r in rows:
            keys.update(k for k, v in r["metrics"].items() if isinstance(v, (int, float)))
        summary[task] = {"n": len(rows)}
        for k in sorted(keys):
            summary[task][k] = metrics_mod.mean(r["metrics"].get(k, 0) or 0 for r in rows)
    return summary


def _render_markdown(summary: dict, args, benchmark_meta: dict) -> str:
    lines = [
        "# PersonaMem-v3 Evaluation Summary",
        f"- user_id: {args.user_id}",
        f"- mode: {args.mode}",
        f"- model: {args.model}",
        f"- judge: {args.judge_model if args.enable_llm_judge else 'disabled'}",
        f"- benchmark: {benchmark_meta['benchmark_version']} built_at={benchmark_meta['built_at']} backend_hash={benchmark_meta['backend_hash']} rng_seed={benchmark_meta['rng_seed']}",
        f"- instance counts: {benchmark_meta.get('counts')}",
        "",
    ]
    for task, metrics_dict in summary.items():
        lines.append(f"## {task}")
        if not metrics_dict or metrics_dict.get("n") == 0:
            lines.append("_no results_\n")
            continue
        for k, v in metrics_dict.items():
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.3f}")
            else:
                lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines)


def _load_benchmark(args) -> dict:
    path = Path(args.benchmark) if args.benchmark else default_benchmark_path(args.user_id)
    if not path.exists():
        sys.exit(
            f"Benchmark file not found at {path}. Build it first:\n"
            f"  python scripts/prepare_eval_data.py --user_id {args.user_id}"
        )
    with path.open() as f:
        bm = json.load(f)

    if bm.get("user_id") != args.user_id:
        sys.exit(f"Benchmark user_id {bm.get('user_id')!r} != --user_id {args.user_id!r}")
    if bm.get("benchmark_version") != BENCHMARK_VERSION:
        sys.exit(
            f"Benchmark version mismatch: file is {bm.get('benchmark_version')!r}, harness expects {BENCHMARK_VERSION!r}. "
            f"Rebuild with `python scripts/prepare_eval_data.py --user_id {args.user_id}`."
        )

    current_hash = compute_backend_hash(args.backend_dir, args.user_id)
    if bm.get("backend_hash") != current_hash:
        msg = (
            f"Backend data has changed since this benchmark was built "
            f"({bm.get('backend_hash')!r} → {current_hash!r}).\n"
            f"  Rebuild: python scripts/prepare_eval_data.py --user_id {args.user_id}\n"
            f"  Or override with --allow_stale to run against the frozen inputs anyway."
        )
        if not args.allow_stale:
            sys.exit(msg)
        print(f"[warn] {msg}", file=sys.stderr)
    return bm


def main():
    parser = argparse.ArgumentParser(description="PersonaMem-v3 evaluation harness.")
    parser.add_argument("--user_id", required=True, help="User id under backend/, e.g. 115")
    parser.add_argument("--backend_dir", default="backend", help="Path to backend/ root")
    parser.add_argument("--benchmark", default=None, help="Path to frozen benchmark.json (default: evaluation/benchmarks/{user_id}/benchmark.json)")
    parser.add_argument("--allow_stale", action="store_true", help="Run even if backend_hash has drifted since benchmark was built")
    parser.add_argument("--mode", choices=MODES, default="llm_longctx", help="Inference mode")
    parser.add_argument("--task", default="all", help=f"Task or alias ({', '.join(TASK_ALIASES)})")
    parser.add_argument("--limit", type=int, default=None, help="Cap per-task item count (for quick runs)")
    parser.add_argument("--enable_llm_judge", action="store_true", help="Enable LLM-as-judge optional layer")
    parser.add_argument("--model", default=os.getenv("EVAL_MODEL", "gpt-5-chat"), help="Baseline (llm_longctx) model — QueryLLM backend (Azure/OpenAI/Claude/Gemini)")
    parser.add_argument("--claude_model", default=os.getenv("EVAL_CLAUDE_MODEL", "sonnet"), help="Claude Code subagent model for agent_tools / agent_longctx (haiku, sonnet, opus)")
    parser.add_argument("--judge_model", default=os.getenv("EVAL_JUDGE_MODEL", "claude-opus"), help="Judge model (QueryLLM)")
    parser.add_argument("--context_budget", type=int, default=None, help="Token budget for long-context modes")
    parser.add_argument("--rate_limit", type=int, default=50, help="LLM rate limit per minute")
    parser.add_argument("--dry_run", action="store_true", help="Build prompts without calling LLMs")
    parser.add_argument("--output_dir", default=None, help="Where to write results (default: benchmark/{user_id}/runs/)")
    # E4 (Google Search) gating — default OFF. See evaluation/tasks/e4_google_search.py.
    parser.add_argument("--enable_e4", action="store_true",
                        help="Enable task e4_google_search (external Google Custom Search API).")
    parser.add_argument("--e4_allow_live", action="store_true",
                        help="Allow live Google API calls on cache miss (requires GOOGLE_API_KEY + GOOGLE_CSE_ID).")
    parser.add_argument("--e4_quota_per_day", type=int, default=20,
                        help="Per-user daily live-call cap for E4 (default 20).")
    args = parser.parse_args()

    # Propagate E4 gating into env so the MCP server + runner see it
    if args.enable_e4:
        os.environ["PM3_E4_ENABLED"] = "1"
    if args.e4_allow_live:
        os.environ["PM3_E4_ALLOW_LIVE"] = "1"
    os.environ["PM3_E4_QUOTA_PER_DAY"] = str(args.e4_quota_per_day)

    bm = _load_benchmark(args)

    bq = BackendQuery(args.backend_dir)
    llm_client, judge_client = _build_llm_clients(args)
    snapshot_cache = SnapshotCache()

    tasks = _resolve_tasks(args.task)
    all_results: dict[str, list[dict]] = {}
    for task_name in tasks:
        instances = bm.get(BENCHMARK_TASK_KEYS[task_name], [])
        print(f"[eval] running {task_name} (mode={args.mode}) — {len(instances)} frozen instances")
        results = _run_task(task_name, instances, args.user_id, bq, llm_client, judge_client, args, snapshot_cache)
        all_results[task_name] = results
        print(f"[eval]   {task_name}: {len(results)} rows")

    if "slate_ranking" in all_results:
        all_results["d_negative_avoidance"] = [over_personalization.aggregate_task_d(all_results["slate_ranking"])]

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = Path(args.output_dir) if args.output_dir else Path("benchmark") / args.user_id / "runs"
    out_dir = base_out / timestamp if args.output_dir else base_out / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_name, rows in all_results.items():
        with (out_dir / f"{task_name}.json").open("w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    summary = _summarize(all_results)
    run_meta = {
        "benchmark_version": bm["benchmark_version"],
        "built_at": bm["built_at"],
        "backend_hash": bm["backend_hash"],
        "rng_seed": bm["rng_seed"],
        "counts": bm.get("counts"),
        "mode": args.mode,
        "model": args.model,
        "judge_model": args.judge_model if args.enable_llm_judge else None,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump({"run": run_meta, "metrics": summary}, f, ensure_ascii=False, indent=2)
    with (out_dir / "summary.md").open("w") as f:
        f.write(_render_markdown(summary, args, run_meta))

    print(f"\n[eval] wrote results to {out_dir}")
    print(f"[eval] benchmark: {bm['benchmark_version']} hash={bm['backend_hash']} seed={bm['rng_seed']}")
    print(f"[eval] summary:")
    for task, m in summary.items():
        print(f"  {task}: {m}")


if __name__ == "__main__":
    main()
