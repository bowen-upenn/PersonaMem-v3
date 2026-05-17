"""Per-persona sequential evaluation harness.

Reads `benchmark/{uid}/queries.csv`, iterates rows in `seq` order, dispatches
each query to its task-specific runner via `run_eval_dispatch.dispatch_single`,
and writes per-row results to `{run_dir}/results.csv` and a per-persona summary.

Strictly sequential within a persona — agentic writes accumulate across queries
via a single persistent MCP overlay file. Cross-persona parallelism happens at
the shell level (see `scripts/run_eval_all.sh`).

CLI:
    python -m evaluation.run_eval --user_id 115 --run_dir benchmark/115/runs/<ts>
        [--mode llm_longctx|mcp_agent|agent_longctx]
        [--limit N] [--resume] [--dry_run]
        [--enable_llm_judge]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Load .env early so AZURE_OPENAI_* are visible to QueryLLM
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
except Exception:
    pass

from evaluation.backend_query import BackendQuery
from evaluation.inference_utils import SnapshotCache
from evaluation.run_eval_dispatch import DispatchContext, dispatch_single
from evaluation.task_registry import QUERIES_CSV_VERSION

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# Runner-side CSV limit — some instance_json cells are ~256 KB.
csv.field_size_limit(10_000_000)


RESULTS_COLUMNS: list[str] = [
    "query_id",
    "seq",
    "user_id",
    "task_type",
    "ts",
    "metrics_json",
    "status",
    "duration_ms",
    "error",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-persona sequential evaluation harness."
    )
    p.add_argument("--user_id", required=True)
    p.add_argument("--backend_dir", default="backend")
    p.add_argument("--run_dir", required=True,
                   help="Output directory for results.csv + writes.jsonl + summary files")
    p.add_argument("--mode",
                   choices=("llm_longctx", "mcp_agent", "agent_tools", "agent_longctx"),
                   default="llm_longctx")
    p.add_argument("--model", default=os.getenv("EVAL_MODEL", "gpt-5-chat"),
                   help="Baseline LLM model for llm_longctx mode")
    p.add_argument("--claude_model", default=os.getenv("EVAL_CLAUDE_MODEL", "sonnet"),
                   help="Claude Code subagent model (haiku/sonnet/opus)")
    p.add_argument("--judge_model", default=os.getenv("EVAL_JUDGE_MODEL", "claude-opus"))
    p.add_argument("--rate_limit", type=int, default=50)
    # Phase I.1: judge is ON by default — chatbot tasks need pr_held_out_score
    # which is judge-based; without it, chatbot_personalized_response
    # scored 5.4% in Phase F purely because the judge wasn't running.
    # Use --no_llm_judge to opt out (e.g., for cheap dry runs).
    p.add_argument("--enable_llm_judge", action=argparse.BooleanOptionalAction, default=True,
                   help="Run the LLM judge for pr_* dimensions (default: on). --no-enable_llm_judge to disable.")
    p.add_argument("--context_budget", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap total query rows (for quick smoke tests)")
    p.add_argument("--resume", action="store_true",
                   help="Skip queries already present in {run_dir}/results.csv")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def _build_llm_clients(args: argparse.Namespace):
    """Mirror the clients set up by run_inference.py._build_llm_clients."""
    if args.dry_run or args.mode in ("agent_tools", "agent_longctx", "mcp_agent"):
        baseline = None
    else:
        from query_llm import QueryLLM
        baseline = QueryLLM(
            {"models": {"llm_model": args.model}}, rate_limit_per_min=args.rate_limit,
        )
    judge = None
    if args.enable_llm_judge and not args.dry_run:
        from query_llm import QueryLLM
        judge = QueryLLM(
            {"models": {"llm_model": args.judge_model}}, rate_limit_per_min=args.rate_limit,
        )
    return baseline, judge


def _load_queries(queries_path: Path) -> list[dict]:
    with queries_path.open("r", encoding="utf-8") as f:
        first = f.readline().rstrip("\n")
        if not first.startswith("#"):
            # No version header — treat first line as data header.
            f.seek(0)
        else:
            # Sanity: queries_csv_version=N
            if f"queries_csv_version={QUERIES_CSV_VERSION}" not in first:
                print(f"[run_eval] WARN: CSV version mismatch — header={first!r}, "
                      f"expected queries_csv_version={QUERIES_CSV_VERSION}")
        reader = csv.DictReader(f)
        rows = list(reader)
    # Assert sort-by-seq
    seqs = [int(r["seq"]) for r in rows]
    if seqs != sorted(seqs):
        print("[run_eval] WARN: queries.csv is not sorted by seq ascending — "
              "the runner will still iterate in file order.")
    return rows


def _already_done(results_csv: Path) -> set[str]:
    """Return the set of query_ids already present in results.csv."""
    if not results_csv.exists():
        return set()
    out: set[str] = set()
    with results_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            qid = r.get("query_id")
            if qid:
                out.add(qid)
    return out


def _open_results_writer(results_csv: Path, resume: bool):
    """Append mode when resuming; overwrite with header otherwise."""
    need_header = not (resume and results_csv.exists())
    mode = "a" if resume and results_csv.exists() else "w"
    f = results_csv.open(mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
    if need_header:
        writer.writeheader()
        f.flush()
    return f, writer


def _set_query_env(row: dict, run_dir: Path, user_id: str, backend_dir: str) -> None:
    """Set the per-query env vars the MCP servers + overlay read.

    PM3_T_TEST  — time mask for this query.
    PM3_OVERLAY_PATH — single accumulating overlay across the persona run.
    PM3_USER_ID / PM3_BACKEND_DIR — standard pointers.
    """
    os.environ["PM3_T_TEST"] = str(row["ts"])
    os.environ["PM3_USER_ID"] = user_id
    os.environ["PM3_BACKEND_DIR"] = backend_dir
    os.environ["PM3_OVERLAY_PATH"] = str(run_dir / "writes.jsonl")


def _summarize_by_task(rows: list[dict]) -> dict:
    """Per-task: count + mean of each numeric metric field."""
    from collections import defaultdict
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r.get("task_type", "")].append(r)

    out: dict[str, dict] = {}
    for task, task_rows in by_task.items():
        m = {"n": len(task_rows)}
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        for r in task_rows:
            mj = r.get("metrics_json") or ""
            if not mj:
                continue
            try:
                metrics = json.loads(mj)
            except Exception:
                continue
            for k, v in (metrics or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
                    metric_counts[k] = metric_counts.get(k, 0) + 1
        for k, s in metric_sums.items():
            m[k] = s / max(1, metric_counts[k])
        out[task] = m
    return out


def main() -> int:
    args = _parse_args()

    queries_path = Path(args.backend_dir).parent / "benchmark" / args.user_id / "queries.csv"
    # Real on-disk location is benchmark/{uid}/queries.csv — not under backend_dir.
    queries_path = Path("benchmark") / args.user_id / "queries.csv"
    if not queries_path.exists():
        print(f"[run_eval] queries.csv missing at {queries_path} — "
              f"build it first: python scripts/prepare_eval_data.py --user_id {args.user_id}",
              file=sys.stderr)
        return 2

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_csv = run_dir / "results.csv"

    # Persistent overlay for this persona-run — MCP writes from all queries
    # accumulate here so later queries see earlier writes via OverlayView.
    overlay_path = run_dir / "writes.jsonl"
    overlay_path.touch(exist_ok=True)

    rows = _load_queries(queries_path)
    if args.limit:
        rows = rows[: args.limit]

    done: set[str] = _already_done(results_csv) if args.resume else set()
    print(f"[run_eval] user={args.user_id}  mode={args.mode}  rows={len(rows)}  "
          f"resume_skip={len(done)}  run_dir={run_dir}")

    # Shared dispatch objects
    bq = BackendQuery(args.backend_dir)
    llm_client, judge_client = _build_llm_clients(args)
    snapshot_cache = SnapshotCache()

    ctx = DispatchContext(
        user_id=args.user_id,
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
    )

    out_file, writer = _open_results_writer(results_csv, args.resume)
    written_results: list[dict] = []

    # Progress bar — starts at the number of already-done rows when resuming
    # so the bar reflects true completion (not just this-session's work).
    if tqdm is not None:
        bar = tqdm(
            total=len(rows),
            initial=len(done),
            desc=f"eval user={args.user_id} mode={args.mode}",
            unit="q",
            dynamic_ncols=True,
            smoothing=0.1,
        )
    else:
        bar = None

    try:
        for i, row in enumerate(rows):
            qid = row["query_id"]
            if qid in done:
                continue
            try:
                inst = json.loads(row["instance_json"])
            except Exception as e:
                rec = {
                    "query_id": qid, "seq": row["seq"],
                    "user_id": args.user_id, "task_type": row["task_type"],
                    "ts": row["ts"],
                    "metrics_json": "", "status": "error", "duration_ms": 0,
                    "error": f"instance_json parse: {type(e).__name__}: {e}",
                }
                writer.writerow(rec)
                out_file.flush()
                written_results.append(rec)
                continue

            _set_query_env(row, run_dir, args.user_id, args.backend_dir)
            t0 = time.time()
            try:
                result = dispatch_single(row["task_type"], inst, ctx)
                duration_ms = int((time.time() - t0) * 1000)
                if result is None:
                    rec = {
                        "query_id": qid, "seq": row["seq"],
                        "user_id": args.user_id, "task_type": row["task_type"],
                        "ts": row["ts"],
                        "metrics_json": "", "status": "no_result",
                        "duration_ms": duration_ms, "error": "",
                    }
                else:
                    # Pull token counts from subagent_stats (Claude Code modes
                    # populate them) and inject into metrics_json so the
                    # aggregator can build the token-vs-accuracy table without
                    # having to peek into nested dicts. Per-task runners that
                    # already populate these (slate, chatbot, agentic) get a
                    # no-op merge — the keys overwrite to the same values.
                    metrics_dict = dict(result.get("metrics") or {})
                    sub = result.get("subagent_stats") or {}
                    for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cost_usd"):
                        v = sub.get(k)
                        if v is not None and k not in metrics_dict:
                            metrics_dict[k] = v
                    rec = {
                        "query_id": qid, "seq": row["seq"],
                        "user_id": args.user_id, "task_type": row["task_type"],
                        "ts": row["ts"],
                        "metrics_json": json.dumps(metrics_dict, ensure_ascii=False),
                        "status": result.get("status", "ok"),
                        "duration_ms": duration_ms,
                        "error": result.get("error", "") or "",
                    }
            except Exception as e:
                duration_ms = int((time.time() - t0) * 1000)
                rec = {
                    "query_id": qid, "seq": row["seq"],
                    "user_id": args.user_id, "task_type": row["task_type"],
                    "ts": row["ts"],
                    "metrics_json": "", "status": "error",
                    "duration_ms": duration_ms,
                    "error": f"{type(e).__name__}: {e}",
                }
            writer.writerow(rec)
            out_file.flush()
            written_results.append(rec)

            if bar is not None:
                bar.set_postfix_str(
                    f"{row['task_type']}/{rec['status']} {rec['duration_ms']}ms",
                    refresh=False,
                )
                bar.update(1)
            elif (i + 1) % 10 == 0 or i == len(rows) - 1:
                print(f"[run_eval] {i + 1}/{len(rows)} done "
                      f"(latest: {row['task_type']} {row['seq']} {rec['status']} "
                      f"{rec['duration_ms']}ms)")
    finally:
        if bar is not None:
            bar.close()
        out_file.close()

    # Per-persona summary
    summary = _summarize_by_task(written_results)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "user_id": args.user_id,
                "mode": args.mode,
                "model": args.model,
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "n_queries_run": len(written_results),
                "by_task": summary,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(f"[run_eval] wrote {results_csv} + summary.json")
    for t, m in sorted(summary.items()):
        print(f"  {t}: n={m['n']}  "
              + "  ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
