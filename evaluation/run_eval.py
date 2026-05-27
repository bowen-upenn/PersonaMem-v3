"""Per-persona sequential evaluation harness.

Reads `benchmark/{uid}/queries.csv`, iterates rows in `seq` order, dispatches
each query to its task-specific runner via `run_eval_dispatch.dispatch_single`,
and writes per-row results to `{run_dir}/results.csv` and a per-persona summary.

Strictly sequential within a persona — agentic writes accumulate across queries
via a single persistent MCP overlay file. Cross-persona parallelism happens at
the shell level (see `scripts/run_eval_all.sh`).

CLI:
    python -m evaluation.run_eval --user_id 115 --run_dir benchmark/115/runs/<ts>
        [--mode llm_longctx|mcp_agent|agent_tools]
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
    "agent_response",
]


# Truncation cap for agent_response in results.csv. Some agentic
# responses (especially in mcp_agent mode) can be many KB of JSON;
# we keep enough for downstream audits (privacy-leak detector
# false-positive sampling, voice-match calibration) without
# bloating the CSV beyond what spreadsheet tools handle.
_AGENT_RESPONSE_TRUNCATE_BYTES = 4096


def _truncate_agent_response(resp) -> str:
    """Coerce runner-emitted `agent_response` (str | dict | None) to a
    truncated UTF-8 string suitable for the CSV column."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        s = resp
    else:
        try:
            s = json.dumps(resp, ensure_ascii=False)
        except (TypeError, ValueError):
            s = str(resp)
    if len(s.encode("utf-8")) <= _AGENT_RESPONSE_TRUNCATE_BYTES:
        return s
    # Encode-then-decode-with-replace to land on a valid char boundary.
    truncated = s.encode("utf-8")[:_AGENT_RESPONSE_TRUNCATE_BYTES].decode(
        "utf-8", errors="replace"
    )
    return truncated + "…[truncated]"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-persona sequential evaluation harness."
    )
    p.add_argument("--user_id", required=True)
    p.add_argument("--backend_dir", default="backend")
    p.add_argument("--run_dir", required=True,
                   help="Output directory for results.csv + writes.jsonl + summary files")
    p.add_argument("--mode",
                   choices=("llm_longctx", "mcp_agent", "agent_tools"),
                   default="llm_longctx")
    p.add_argument("--model", default=os.getenv("EVAL_MODEL", "gpt-5-chat"),
                   help="Baseline LLM model for llm_longctx mode")
    p.add_argument("--claude_model", default=os.getenv("EVAL_CLAUDE_MODEL", "sonnet"),
                   help="Claude Code subagent model (haiku/sonnet/opus)")
    p.add_argument("--judge_model", default=os.getenv("EVAL_JUDGE_MODEL", "gpt-5.4"))
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
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel worker count for non-agentic rows. "
                        "Agentic rows (T6-T19) always run sequentially in a "
                        "dedicated worker because they share writes.jsonl. "
                        "--workers 1 disables parallelism (original behavior).")
    return p.parse_args()


def _build_llm_clients(args: argparse.Namespace):
    """Mirror the clients set up by run_inference.py._build_llm_clients."""
    if args.dry_run or args.mode in ("agent_tools", "mcp_agent"):
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


def _is_sequential(task_type: str, mode: str = "") -> bool:
    """Rows that must run in seq order within a persona.

    Sequential execution is only needed when BOTH conditions hold:
      (a) the task has `state_write_policy == "writes_ok"` (appends to
          the shared `writes.jsonl` overlay), AND
      (b) the eval mode is `mcp_agent` (the only mode where writes
          actually happen).

    In `agent_tools` mode, the agent is read-only (filesystem snapshot,
    no MCP tools, no overlay). In `llm_longctx` mode, there's no agent
    framework at all. So in both non-mcp modes, ALL tasks can safely
    run in parallel — the sequential constraint is unnecessary.

    This alone eliminates the sequential bottleneck for agent_tools
    mode: the 26-min agentic queue drops to ~4 min with 16 workers.
    """
    if mode != "mcp_agent":
        return False
    from evaluation.task_registry import get_meta, normalize_task_type
    meta = get_meta(normalize_task_type(task_type))
    return meta.get("state_write_policy") == "writes_ok"


def _pack_rec(qid: str, seq, user_id: str, task_type: str, ts,
              result: dict | None, duration_ms: int) -> dict:
    """Build the per-row CSV record from a runner's return value.

    Shared between the sequential and parallel-worker code paths so the
    record shape (status, metrics, token counts, agent_response) is
    identical regardless of which queue serviced the row.
    """
    if result is None:
        return {
            "query_id": qid, "seq": seq,
            "user_id": user_id, "task_type": task_type, "ts": ts,
            "metrics_json": "", "status": "no_result",
            "duration_ms": duration_ms, "error": "",
            "agent_response": "",
        }
    metrics_dict = dict(result.get("metrics") or {})
    sub = result.get("subagent_stats") or {}
    # Repetition runners (c1c/c1d) emit `subagent_stats` as a LIST
    # (one entry per query in the cluster); collapse by summing token
    # counts so the token-vs-accuracy aggregator works uniformly.
    if isinstance(sub, list):
        agg: dict = {}
        for s in sub:
            if not isinstance(s, dict):
                continue
            for k in ("input_tokens", "output_tokens",
                      "cache_read_tokens", "cost_usd"):
                v = s.get(k)
                if isinstance(v, (int, float)):
                    agg[k] = agg.get(k, 0) + v
        sub = agg
    elif not isinstance(sub, dict):
        sub = {}
    for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cost_usd"):
        v = sub.get(k)
        if v is not None and k not in metrics_dict:
            metrics_dict[k] = v
    # Prefer the runner's explicit agent_response; fall back to
    # agent_response_raw (proactive_actions emits this) when present.
    raw_resp = result.get("agent_response")
    if raw_resp is None:
        raw_resp = result.get("agent_response_raw")
    return {
        "query_id": qid, "seq": seq,
        "user_id": user_id, "task_type": task_type, "ts": ts,
        "metrics_json": json.dumps(metrics_dict, ensure_ascii=False),
        "status": result.get("status", "ok"),
        "duration_ms": duration_ms,
        "error": result.get("error", "") or "",
        "agent_response": _truncate_agent_response(raw_resp),
    }


def _run_one_in_worker(payload: dict) -> dict:
    """Top-level function so ProcessPoolExecutor can pickle it.

    Each child process inherits a copy of the parent env at spawn time
    and mutates its OWN copy of os.environ — safe sibling-isolation that
    threads cannot provide (PM3_T_TEST would race across threads).

    Parallel workers should never dispatch agentic_* task types — the
    caller filters those out — so the overlay file is never touched here.
    The PM3_OVERLAY_PATH env var is still set for parity with the
    sequential path and because the MCP server config reads it
    unconditionally at import time.
    """
    import os  # noqa: F811 — explicit so this module ships standalone
    import json  # noqa: F811
    import time  # noqa: F811
    import traceback
    from pathlib import Path  # noqa: F811
    from evaluation.backend_query import BackendQuery
    from evaluation.inference_utils import SnapshotCache
    from evaluation.run_eval_dispatch import DispatchContext, dispatch_single

    row = payload["row"]
    run_dir = Path(payload["run_dir_str"])
    user_id = payload["user_id"]

    os.environ["PM3_T_TEST"] = str(row["ts"])
    os.environ["PM3_USER_ID"] = user_id
    os.environ["PM3_BACKEND_DIR"] = payload["backend_dir"]
    os.environ["PM3_OVERLAY_PATH"] = str(run_dir / "writes.jsonl")

    # Build per-worker LLM clients. Each worker keeps its own QueryLLM
    # connection — instantiation is cheap relative to the per-row wall
    # time (~25s). The per-worker rate limit is scaled down so the
    # aggregate doesn't exceed the parent's intended bound.
    judge = None
    if payload["enable_llm_judge"]:
        from query_llm import QueryLLM
        judge = QueryLLM(
            {"models": {"llm_model": payload["judge_model_name"]}},
            rate_limit_per_min=payload["per_worker_rate_limit"],
        )
    baseline = None
    if payload["mode"] == "llm_longctx" and not payload["dry_run"]:
        from query_llm import QueryLLM
        baseline = QueryLLM(
            {"models": {"llm_model": payload["model_name"]}},
            rate_limit_per_min=payload["per_worker_rate_limit"],
        )

    bq = BackendQuery(payload["backend_dir"])
    ctx = DispatchContext(
        user_id=user_id, bq=bq,
        llm_client=baseline, judge_client=judge,
        mode=payload["mode"], snapshot_cache=SnapshotCache(),
        model_name=payload["model_name"],
        claude_model=payload["claude_model"],
        context_budget=payload["context_budget"],
        enable_llm_judge=payload["enable_llm_judge"],
        dry_run=payload["dry_run"],
    )

    qid = row["query_id"]
    try:
        inst = json.loads(row["instance_json"])
    except Exception as e:
        return {
            "query_id": qid, "seq": row["seq"],
            "user_id": user_id, "task_type": row["task_type"],
            "ts": row["ts"],
            "metrics_json": "", "status": "error", "duration_ms": 0,
            "error": f"instance_json parse: {type(e).__name__}: {e}",
            "agent_response": "",
        }

    t0 = time.time()
    try:
        result = dispatch_single(row["task_type"], inst, ctx)
        return _pack_rec(qid, row["seq"], user_id, row["task_type"],
                         row["ts"], result, int((time.time() - t0) * 1000))
    except Exception as e:
        tb = traceback.format_exc()
        last_frame_loc = ""
        for line in reversed(tb.splitlines()):
            if line.startswith("  File "):
                last_frame_loc = line.strip()
                break
        # Print to the worker's stderr so the parent's tail captures it.
        print(
            f"[run_eval/worker] ERROR on query_id={qid} task_type={row['task_type']}:\n{tb}",
            file=sys.stderr, flush=True,
        )
        return {
            "query_id": qid, "seq": row["seq"],
            "user_id": user_id, "task_type": row["task_type"],
            "ts": row["ts"],
            "metrics_json": "", "status": "error",
            "duration_ms": int((time.time() - t0) * 1000),
            "error": f"{type(e).__name__}: {e} | {last_frame_loc}",
            "agent_response": "",
        }


def _build_payload(row: dict, args: argparse.Namespace, run_dir: Path,
                   per_worker_rate_limit: int) -> dict:
    """Pack per-row args for `_run_one_in_worker` over IPC.

    Only picklable primitives — no live BackendQuery / SnapshotCache /
    QueryLLM clients (each worker constructs its own).
    """
    return {
        "row": row,
        "run_dir_str": str(run_dir),
        "user_id": args.user_id,
        "backend_dir": args.backend_dir,
        "mode": args.mode,
        "model_name": args.model,
        "claude_model": args.claude_model,
        "judge_model_name": args.judge_model,
        "enable_llm_judge": args.enable_llm_judge,
        "context_budget": args.context_budget,
        "dry_run": args.dry_run,
        "per_worker_rate_limit": per_worker_rate_limit,
    }


def _summarize_by_task(rows: list[dict]) -> dict:
    """Per-task: count + mean of each numeric metric field.

    Adds three derived fields per task on top of the raw per-metric means:

    - `non_substantive_response_rate`: fraction of rows where the runner
      flagged the agent's response as too short to be substantive (set
      by the substantive-engagement gate in `evaluation/metrics.py`).
      A high value here means the headline restraint score is being
      driven by silence, not by thoughtful restraint — the aggregator
      should flag that.

    - `mean_input_tokens / mean_output_tokens / mean_total_tokens /
      mean_cost_usd / cache_hit_rate`: per-row token spend means.
      `cache_hit_rate = cache_read_tokens / input_tokens`. These are
      already collected on every row (sub-agent or LLM API populates
      `subagent_stats`); _summarize_by_task surfaces them as
      first-class summary keys so the aggregator and reviewers don't
      have to grep `metrics_json` to see cost-vs-accuracy tradeoffs.

    - `error_rate`: fraction of rows whose status was not "ok".
    """
    from collections import defaultdict
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r.get("task_type", "")].append(r)

    _TOKEN_KEYS = ("input_tokens", "output_tokens",
                   "cache_read_tokens", "cost_usd")

    out: dict[str, dict] = {}
    for task, task_rows in by_task.items():
        m = {"n": len(task_rows)}
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        # Derived counters
        non_substantive = 0
        n_non_ok = 0
        tok_sums = {k: 0.0 for k in _TOKEN_KEYS}
        tok_counts = {k: 0 for k in _TOKEN_KEYS}
        for r in task_rows:
            status = r.get("status", "")
            if status != "ok":
                n_non_ok += 1
            mj = r.get("metrics_json") or ""
            if not mj:
                continue
            try:
                metrics = json.loads(mj)
            except Exception:
                continue
            # Substantive-gate signal: many runners emit either
            # `non_substantive_response` (over_personalization) or
            # `response_is_substantive` (e6) — handle both.
            if metrics.get("non_substantive_response"):
                non_substantive += 1
            elif (metrics.get("response_is_substantive") is not None
                  and not metrics.get("response_is_substantive")):
                non_substantive += 1
            for k, v in (metrics or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
                    metric_counts[k] = metric_counts.get(k, 0) + 1
                    if k in _TOKEN_KEYS:
                        tok_sums[k] += float(v)
                        tok_counts[k] += 1
        for k, s in metric_sums.items():
            m[k] = s / max(1, metric_counts[k])
        # Derived: error rate.
        m["error_rate"] = n_non_ok / max(1, len(task_rows))
        # Derived: non-substantive rate (silence-pass signal).
        m["non_substantive_response_rate"] = (
            non_substantive / max(1, len(task_rows))
        )
        # Derived: per-row token + cost means (the original metric
        # means already include these from the metric_sums loop above,
        # so these are duplicates — kept under explicit names so the
        # aggregator's cost-vs-accuracy table can pull them directly
        # without guessing field names).
        in_mean = tok_sums["input_tokens"] / max(1, tok_counts["input_tokens"])
        out_mean = tok_sums["output_tokens"] / max(1, tok_counts["output_tokens"])
        cache_mean = tok_sums["cache_read_tokens"] / max(1, tok_counts["cache_read_tokens"])
        cost_mean = tok_sums["cost_usd"] / max(1, tok_counts["cost_usd"])
        m["mean_input_tokens"] = in_mean
        m["mean_output_tokens"] = out_mean
        m["mean_cache_read_tokens"] = cache_mean
        m["mean_total_tokens"] = in_mean + out_mean
        m["mean_cost_usd"] = cost_mean
        # Anthropic + OpenAI report `input_tokens` as FRESH (non-cached)
        # tokens and `cache_read_tokens` separately. Cache-hit rate is
        # cached / (fresh + cached). Without this denominator fix, the
        # ratio reads ~200,000% for cached-heavy agentic runs.
        total_prompt = in_mean + cache_mean
        m["cache_hit_rate"] = (cache_mean / total_prompt) if total_prompt > 0 else 0.0
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

    # Partition rows. Agentic_* tasks must run in seq order in a single
    # worker (shared writes.jsonl overlay); everything else parallelizes.
    pending = [r for r in rows if r["query_id"] not in done]
    parallel_rows = [r for r in pending if not _is_sequential(r["task_type"], args.mode)]
    sequential_rows = [r for r in pending if _is_sequential(r["task_type"], args.mode)]
    print(f"[run_eval] partition: parallel={len(parallel_rows)} "
          f"sequential={len(sequential_rows)} workers={args.workers}")

    import threading
    import traceback
    write_lock = threading.Lock()

    def _emit(rec: dict) -> None:
        """Write one record + tqdm tick under the writer lock.
        Called from both the sequential thread and the parallel
        future-consumer (main thread)."""
        with write_lock:
            writer.writerow(rec)
            out_file.flush()
            written_results.append(rec)
            if bar is not None:
                bar.set_postfix_str(
                    f"{rec['task_type']}/{rec['status']} {rec['duration_ms']}ms",
                    refresh=False,
                )
                bar.update(1)

    def _run_seq_row(row: dict) -> dict:
        """Run one row inline in the current thread/process using the
        parent's shared `ctx`. Used by both the sequential queue and
        the --workers 1 / dry_run fallback."""
        qid = row["query_id"]
        try:
            inst = json.loads(row["instance_json"])
        except Exception as e:
            return {
                "query_id": qid, "seq": row["seq"],
                "user_id": args.user_id, "task_type": row["task_type"],
                "ts": row["ts"],
                "metrics_json": "", "status": "error", "duration_ms": 0,
                "error": f"instance_json parse: {type(e).__name__}: {e}",
                "agent_response": "",
            }
        _set_query_env(row, run_dir, args.user_id, args.backend_dir)
        t0 = time.time()
        try:
            result = dispatch_single(row["task_type"], inst, ctx)
            return _pack_rec(qid, row["seq"], args.user_id, row["task_type"],
                             row["ts"], result, int((time.time() - t0) * 1000))
        except Exception as e:
            tb = traceback.format_exc()
            print(
                f"[run_eval] ERROR on query_id={qid} task_type={row['task_type']}:\n{tb}",
                file=sys.stderr, flush=True,
            )
            last_frame_loc = ""
            for line in reversed(tb.splitlines()):
                if line.startswith("  File "):
                    last_frame_loc = line.strip()
                    break
            return {
                "query_id": qid, "seq": row["seq"],
                "user_id": args.user_id, "task_type": row["task_type"],
                "ts": row["ts"],
                "metrics_json": "", "status": "error",
                "duration_ms": int((time.time() - t0) * 1000),
                "error": f"{type(e).__name__}: {e} | {last_frame_loc}",
                "agent_response": "",
            }

    # Effective worker count — dry_run forces single-threaded so any
    # crashes in the runner surface directly with full tracebacks.
    use_pool = (args.workers > 1
                and not args.dry_run
                and len(parallel_rows) > 0)

    try:
        if not use_pool:
            # Sequential fallback — old behavior. Walks parallel_rows
            # then sequential_rows in seq order, all in the main thread.
            for row in parallel_rows + sequential_rows:
                _emit(_run_seq_row(row))
        else:
            import concurrent.futures
            from concurrent.futures import ProcessPoolExecutor

            # Sequential agentic queue runs in a parent-process thread
            # — shares `ctx` and parent os.environ (no race because
            # parallel workers never touch the overlay).
            seq_done_flag = threading.Event()

            def _drive_sequential():
                for row in sequential_rows:
                    _emit(_run_seq_row(row))
                seq_done_flag.set()

            seq_thread = threading.Thread(target=_drive_sequential, daemon=False)
            seq_thread.start()

            # Parallel pool drives the non-agentic queue. Each worker
            # gets its own per-rate-limit budget so the aggregate stays
            # under args.rate_limit.
            # Each worker gets at least as many RPM as there are workers,
            # so N concurrent workers can each fire at least 1 call at a time.
            # The old `rate_limit // workers` was too conservative (3 RPM per
            # worker at 16 workers / 50 RPM → artificially slow).
            per_worker = max(args.rate_limit, args.workers)
            payloads = [_build_payload(row, args, run_dir, per_worker)
                        for row in parallel_rows]
            # Per-future timeout safety net. The agent_tools / mcp_agent
            # subprocess call has its own timeout_seconds=600, so a
            # worker should always return within ~10 min. We give an
            # extra 5-min buffer here (judge call + LLM client setup +
            # IPC); anything past 900s is a true hang and we record an
            # error instead of blocking the whole eval forever.
            FUTURE_TIMEOUT_S = 900
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                fut_to_payload = {
                    pool.submit(_run_one_in_worker, p): p for p in payloads
                }
                pending = set(fut_to_payload.keys())
                while pending:
                    done, _ = concurrent.futures.wait(
                        pending,
                        timeout=FUTURE_TIMEOUT_S,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        # Nothing completed in 15 min → assume all
                        # remaining futures are wedged. Record them as
                        # errors so results.csv reaches its expected
                        # row count, then break out.
                        print(
                            f"[run_eval] WARN: {len(pending)} futures "
                            f"hung past {FUTURE_TIMEOUT_S}s — recording "
                            f"as errors and continuing.",
                            file=sys.stderr, flush=True,
                        )
                        for fut in pending:
                            payload = fut_to_payload[fut]
                            row = payload["row"]
                            _emit({
                                "query_id": row["query_id"],
                                "seq": row["seq"],
                                "user_id": args.user_id,
                                "task_type": row["task_type"],
                                "ts": row["ts"],
                                "metrics_json": "",
                                "status": "error",
                                "duration_ms": FUTURE_TIMEOUT_S * 1000,
                                "error": f"future_hung_no_completion_in_{FUTURE_TIMEOUT_S}s",
                                "agent_response": "",
                            })
                            fut.cancel()
                        break
                    for fut in done:
                        pending.discard(fut)
                        try:
                            _emit(fut.result(timeout=5))
                        except Exception as e:
                            payload = fut_to_payload[fut]
                            row = payload["row"]
                            _emit({
                                "query_id": row["query_id"],
                                "seq": row["seq"],
                                "user_id": args.user_id,
                                "task_type": row["task_type"],
                                "ts": row["ts"],
                                "metrics_json": "",
                                "status": "error",
                                "duration_ms": 0,
                                "error": f"worker_future_error: {type(e).__name__}: {e}",
                                "agent_response": "",
                            })

            # Wait for the sequential queue to drain (if it hasn't already).
            seq_thread.join()
    finally:
        if bar is not None:
            bar.close()
        out_file.close()

    # Per-persona summary + per-persona totals across all tasks.
    summary = _summarize_by_task(written_results)
    persona_totals = {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "cache_read_tokens": 0.0,
        "cost_usd": 0.0,
        "non_substantive_responses": 0,
        "errored_rows": 0,
    }
    for task, m in summary.items():
        n = int(m.get("n", 0))
        # Per-task means × n = totals.
        persona_totals["input_tokens"] += m.get("mean_input_tokens", 0.0) * n
        persona_totals["output_tokens"] += m.get("mean_output_tokens", 0.0) * n
        persona_totals["cache_read_tokens"] += m.get("mean_cache_read_tokens", 0.0) * n
        persona_totals["cost_usd"] += m.get("mean_cost_usd", 0.0) * n
        persona_totals["non_substantive_responses"] += int(
            m.get("non_substantive_response_rate", 0.0) * n
        )
        persona_totals["errored_rows"] += int(m.get("error_rate", 0.0) * n)
    persona_totals["total_tokens"] = (
        persona_totals["input_tokens"] + persona_totals["output_tokens"]
    )

    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "user_id": args.user_id,
                "mode": args.mode,
                "model": args.model,
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "n_queries_run": len(written_results),
                "persona_totals": persona_totals,
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
    print()
    print(f"[run_eval] persona totals: "
          f"input={persona_totals['input_tokens']:,.0f} tokens, "
          f"output={persona_totals['output_tokens']:,.0f}, "
          f"cache_read={persona_totals['cache_read_tokens']:,.0f}, "
          f"cost=${persona_totals['cost_usd']:.2f}, "
          f"non_substantive={persona_totals['non_substantive_responses']}/"
          f"{len(written_results)}, "
          f"errored={persona_totals['errored_rows']}/{len(written_results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
