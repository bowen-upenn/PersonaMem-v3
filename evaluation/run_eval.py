"""Per-persona sequential evaluation harness.

Reads `backend/{uid}/test.json` (a list of structured test instances written
by `scripts/prepare_eval_data.py`), iterates items in list order, dispatches
each query to its task-specific runner via `run_eval_dispatch.dispatch_single`,
and writes per-row results to `{run_dir}/results.csv` and a per-persona summary.

Strictly sequential within a persona — agentic writes accumulate across queries
via a single persistent MCP overlay file. Cross-persona parallelism happens at
the shell level (see `scripts/run_eval_all.sh`).

Legacy `benchmark/{uid}/queries.csv` is still loaded as a fallback if
present, but the benchmark/ folder is no longer produced by the build
pipeline. Each item's `instance_full` field becomes the runner's `inst`.

CLI:
    python -m evaluation.run_eval --user_id 115 --run_dir runs/<ts>
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
from evaluation.prompts import DEFAULT_MEMORY_TOKEN_CAP

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
                   choices=("llm_longctx", "llm_memory", "mem0", "mcp_agent", "agent_tools"),
                   default="llm_longctx")
    p.add_argument("--model", default=os.getenv("EVAL_MODEL", "gpt-5.5"),
                   help="Baseline LLM model for llm_longctx / llm_memory / mem0 modes")
    p.add_argument("--claude_model", default=os.getenv("EVAL_CLAUDE_MODEL", "sonnet"),
                   help="Claude Code subagent model (haiku/sonnet/opus)")
    p.add_argument("--judge_model", default=os.getenv("EVAL_JUDGE_MODEL", "gpt-5.5"))
    p.add_argument("--rate_limit", type=int, default=50)
    # Phase I.1: judge is ON by default — chatbot tasks need pr_held_out_score
    # which is judge-based; without it, chatbot_personalized_response
    # scored 5.4% in Phase F purely because the judge wasn't running.
    # Use --no_llm_judge to opt out (e.g., for cheap dry runs).
    p.add_argument("--enable_llm_judge", action=argparse.BooleanOptionalAction, default=True,
                   help="Run the LLM judge for pr_* dimensions (default: on). --no-enable_llm_judge to disable.")
    p.add_argument("--context_budget", type=int, default=None)
    # --- memory mode knobs (only used when --mode memory) ---
    p.add_argument("--memory_token_cap", type=int, default=DEFAULT_MEMORY_TOKEN_CAP,
                   help="Max tokens for the consolidated memory injected per query (com/mem0 modes)")
    p.add_argument("--memory_chunk_k", type=int, default=40,
                   help="Max events per memory-build LLM call (com/mem0 modes)")
    p.add_argument("--memory_builder_model", default=None,
                   help="Model that builds the memory (com/mem0 modes). Defaults to --model.")
    p.add_argument("--memory_builder_temperature", type=float, default=0.0,
                   help="Temperature for memory-build calls (com/mem0 modes; default 0.0 for determinism)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap total query rows (for quick smoke tests)")
    p.add_argument("--resume", action="store_true",
                   help="Skip queries already present in {run_dir}/results.csv")
    p.add_argument("--retry_failed", action="store_true",
                   help="Drop non-ok rows (error / failed_* / no_result) from "
                        "results.csv first, then re-run ONLY those failed/missing "
                        "query_ids (implies --resume). Use to complete a run that "
                        "hit transient API errors like 429 rate limits.")
    p.add_argument("--prune_invalid", action="store_true",
                   help="After the run, remove any rows still not status=='ok' "
                        "from results.csv so the aggregate contains only valid rows.")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--build_only", action="store_true",
                   help="Memory modes: build + persist the ledger to "
                        "{run_dir}/memory_states/, then exit before answering. "
                        "Lets the build run at high cross-persona concurrency "
                        "(decoupled from the rate-heavy answer phase); a later "
                        "--resume run reuses the cached ledger and only answers.")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel worker count for non-agentic rows (default 8; "
                        "12 was too aggressive for Azure gpt-5.5 and tripped 429s). "
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
    """Load the eval test set.

    Reads `backend/{uid}/test.json` (a list of structured instance dicts,
    written by `scripts/prepare_eval_data.py`). The downstream code path
    expects each row to look like the legacy `queries.csv` row shape —
    i.e. carry `query_id` / `seq` / `task_type` / `ts` columns plus an
    `instance_json` string the dispatcher will parse. Project each test
    item to that shape: index in the list becomes `seq`, and the dict's
    `instance_full` field (the original `inst` payload that ran through
    the postprocess) becomes `instance_json`.

    Legacy CSV path is also supported (file ending in `.csv`) so an
    existing queries.csv still loads if it's the file the user points at.
    """
    if queries_path.suffix == ".csv":
        with queries_path.open("r", encoding="utf-8") as f:
            first = f.readline().rstrip("\n")
            if not first.startswith("#"):
                f.seek(0)
            else:
                if f"queries_csv_version={QUERIES_CSV_VERSION}" not in first:
                    print(f"[run_eval] WARN: CSV version mismatch — header={first!r}, "
                          f"expected queries_csv_version={QUERIES_CSV_VERSION}")
            reader = csv.DictReader(f)
            rows = list(reader)
    else:
        # JSON path: test.json carries a list of dicts; project each into
        # the row shape downstream consumers expect.
        import json as _json
        with queries_path.open("r", encoding="utf-8") as f:
            items = _json.load(f)
        if not isinstance(items, list):
            raise ValueError(
                f"{queries_path} must be a JSON list of test instances "
                f"(got {type(items).__name__})"
            )
        rows = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            inst_full = item.get("instance_full") or item
            rows.append({
                "query_id": item.get("query_id", f"unknown:{i:04d}"),
                "seq": str(i),
                "user_id": item.get("user_id", ""),
                "task_family": item.get("task_family", ""),
                "task_type": item.get("task_type", ""),
                "instance_id": item.get("instance_id", ""),
                "ts": str(item.get("ts", 0)),
                "ts_iso": item.get("ts_iso", ""),
                "instance_json": _json.dumps(inst_full, ensure_ascii=False),
            })
    # Assert sort-by-seq
    seqs = [int(r["seq"]) for r in rows]
    if seqs != sorted(seqs):
        print("[run_eval] WARN: test set is not sorted by seq ascending — "
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


def _rewrite_keep_ok(results_csv: Path) -> tuple[int, int]:
    """Rewrite results.csv keeping ONLY status=='ok' rows; drop everything else
    (error / failed_writes / failed_quality / no_result). Returns (kept, dropped).

    Used by --retry_failed (pre-run: removing failed rows makes their query_ids
    re-run under resume) and by --prune_invalid (post-run: drop anything that
    still failed so the aggregate contains only valid rows). Writes via a temp
    file + atomic replace so a crash can't truncate results.csv mid-write."""
    if not results_csv.exists():
        return (0, 0)
    with results_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keep = [r for r in rows if (r.get("status") or "") == "ok"]
    dropped = len(rows) - len(keep)
    if dropped == 0:
        return (len(keep), 0)
    tmp = results_csv.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in keep:
            w.writerow(r)
    tmp.replace(results_csv)
    return (len(keep), dropped)


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
    if payload["mode"] in ("llm_longctx", "llm_memory", "mem0") and not payload["dry_run"]:
        from query_llm import QueryLLM
        baseline = QueryLLM(
            {"models": {"llm_model": payload["model_name"]}},
            rate_limit_per_min=payload["per_worker_rate_limit"],
        )

    bq = BackendQuery(payload["backend_dir"])
    # llm_memory mode: the per-user memory ledger was prebuilt in the parent and
    # shipped as a picklable {T_test: memory_str} dict — attach it (no rebuild,
    # no LLM calls in the worker). The `mem0` mode never reaches a worker — its
    # live qdrant store is neither picklable nor safe for concurrent processes,
    # so it is forced to run in-process (args.workers=1).
    worker_cache = SnapshotCache(mode=payload["mode"])
    if payload["mode"] == "llm_memory":
        worker_cache.attach_memory_checkpoints(payload.get("memory_checkpoints"))
    ctx = DispatchContext(
        user_id=user_id, bq=bq,
        llm_client=baseline, judge_client=judge,
        mode=payload["mode"], snapshot_cache=worker_cache,
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
                   per_worker_rate_limit: int,
                   memory_checkpoints: dict | None = None) -> dict:
    """Pack per-row args for `_run_one_in_worker` over IPC.

    Only picklable primitives — no live BackendQuery / SnapshotCache /
    QueryLLM clients (each worker constructs its own). `memory_checkpoints`
    (memory mode) is a plain {T_test: memory_string} dict, fully picklable.
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
        "memory_checkpoints": memory_checkpoints,
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

    # mem0's live qdrant store is a single-process embedded DB and is not
    # picklable, so this mode cannot use the ProcessPool — force in-process.
    # (Cross-persona parallelism still comes from running one process per uid.)
    if args.mode == "mem0" and args.workers != 1:
        print(f"[run_eval] mem0 mode runs in-process; overriding --workers "
              f"{args.workers} -> 1", flush=True)
        args.workers = 1

    # Source of truth: backend/{uid}/test.json (a list of test-instance
    # dicts written by scripts/prepare_eval_data.py). Legacy
    # benchmark/{uid}/queries.csv path still loads if present, but the
    # benchmark/ folder is no longer produced.
    queries_path = Path(args.backend_dir) / args.user_id / "test.json"
    if not queries_path.exists():
        legacy_csv = Path("benchmark") / args.user_id / "queries.csv"
        if legacy_csv.exists():
            queries_path = legacy_csv
        else:
            print(f"[run_eval] test.json missing at {queries_path} — "
                  f"build it first: python scripts/prepare_eval_data.py --user_id {args.user_id}",
                  file=sys.stderr)
            return 2

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_csv = run_dir / "results.csv"

    # --retry_failed: strip non-ok rows so their query_ids are no longer "done",
    # then resume — only the failed/missing rows re-run (now under 429 backoff).
    if args.retry_failed and results_csv.exists():
        kept, dropped = _rewrite_keep_ok(results_csv)
        print(f"[run_eval] retry_failed: removed {dropped} non-ok rows "
              f"(kept {kept} ok) — re-running the failed/missing query_ids.")
        args.resume = True

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
    snapshot_cache = SnapshotCache(mode=args.mode)

    # com / mem0 modes: build the per-user memory ledger ONCE here (in the
    # parent) with the faithful algorithm (args.mode), snapshotting consolidated
    # memory at every T_test boundary. Workers then serve from the prebuilt
    # {T_test: memory} dict — no per-worker rebuild, O(events) build cost, honest
    # token accounting in one place.
    memory_checkpoints = None
    memory_build_stats = None
    if args.mode == "mem0" and not args.dry_run:
        # Real mem0ai store (Azure LLM + text-embedding-3-large + local qdrant),
        # built ONCE over all events < max(T_test). Per-query top-k retrieval is
        # time-masked via a `ts < T_test` filter. Runs in-process (workers=1).
        from evaluation.mem0_backend import Mem0Backend
        boundaries = sorted({int(r["ts"]) for r in rows})
        t_max = (boundaries[-1] + 1) if boundaries else 0
        llm_dep = None if args.model in (None, "gpt-5.5") else args.model
        print(f"[run_eval] mem0 mode: building real-mem0 store up to T={t_max} "
              f"(LLM={llm_dep or 'AZURE_OPENAI_DEPLOYMENT_NAME'}, "
              f"embed=text-embedding-3-large, cap={args.memory_token_cap})...", flush=True)
        m0 = Mem0Backend(args.user_id, run_dir, llm_deployment=llm_dep,
                         token_cap=args.memory_token_cap)
        bstats = m0.build(bq, t_max, {"builder_model": args.model,
                                      "chunk_k": args.memory_chunk_k})
        snapshot_cache.attach_mem0_backend(m0)
        memory_build_stats = {
            # The real mem0ai library makes its own Azure calls (not via QueryLLM),
            # so build token usage is NOT tracked here — reported as 0 (the
            # n_chunks count is a proxy for build-call volume). Keys kept present
            # so the shared summary folding below never KeyErrors.
            "memory_build_input_tokens": 0,
            "memory_build_output_tokens": 0,
            "memory_build_calls": bstats.get("n_chunks", 0),
            "memory_build_chunks": bstats.get("n_chunks", 0),
            "memory_build_events": bstats.get("n_events", 0),
            "memory_build_memories": bstats.get("n_memories", 0),
        }
        print(f"[run_eval] mem0 store ready: {bstats}", flush=True)
    elif args.mode == "llm_memory" and not args.dry_run:
        from evaluation.memory_builder import (
            build_checkpoints, default_memory_config, load_existing_checkpoints,
        )
        # Independent builder client: the memory-build API calls must be SEPARATE
        # from the per-query answer calls — its OWN QueryLLM instance (+ own
        # rate-limit semaphore + usage accounting), never the shared answer
        # `llm_client`. The build is a distinct phase; isolating its client keeps
        # build vs answer traffic from contending or being conflated.
        from query_llm import QueryLLM
        _builder_model = args.memory_builder_model or args.model
        builder_client = QueryLLM(
            {"models": {"llm_model": _builder_model}},
            rate_limit_per_min=args.rate_limit,
        )
        mem_cfg = default_memory_config()
        mem_cfg.update({
            "token_cap": args.memory_token_cap,
            "chunk_k": args.memory_chunk_k,
            "builder_temperature": args.memory_builder_temperature,
            "builder_model": args.memory_builder_model or args.model,
        })
        # Build the memory at DAY boundaries (one consolidation per calendar day
        # of the user's activity), NOT at every query T_test. Per-T_test building
        # was ~1 gpt-5.5 call per boundary (~90/persona) and gpt-5.5's reasoning
        # endpoint caps build throughput at ~2.7 calls/min regardless of
        # concurrency → ~11h. Day boundaries (~8-10/persona) cut that ~10x. The
        # ledger's nearest-prior lookup maps each query to the memory consolidated
        # through the previous day (queries see same-day-earlier events only after
        # the next daily consolidation — accepted staleness, daily cadence).
        from evaluation.memory_builder import build_global_stream as _bgs
        _DAY = 86400
        _query_ts = [int(r["ts"]) for r in rows]
        _t_hi = max(_query_ts)
        _ev = _bgs(bq, args.user_id, _t_hi + 1)
        _ev_ts = [int(e.get("t") or 0) for e in _ev if e.get("t")]
        _lo = min(_ev_ts) if _ev_ts else min(_query_ts)
        # midnight (UTC) of the first activity day .. the day AFTER the last query
        boundaries = list(range((_lo // _DAY) * _DAY, ((_t_hi // _DAY) + 2) * _DAY, _DAY))
        existing = load_existing_checkpoints(run_dir, args.user_id, args.mode) if args.resume else None
        _usage_before = (builder_client.get_usage_totals()
                         if hasattr(builder_client, "get_usage_totals") else None)
        print(f"[run_eval] {args.mode} mode: building memory ledger over {len(boundaries)} "
              f"DAY boundaries (builder={mem_cfg['builder_model']}, "
              f"cap={mem_cfg['token_cap']}, chunk_k={mem_cfg['chunk_k']})...", flush=True)
        ledger = build_checkpoints(
            bq, args.user_id, boundaries, builder_client, mem_cfg,
            algo=args.mode, run_dir=run_dir, existing=existing,
        )
        memory_checkpoints = ledger.checkpoints
        snapshot_cache.attach_memory_checkpoints(memory_checkpoints)
        # Prefer provider-reported usage delta (includes cache reads); fall back
        # to the builder's local token estimates if the provider didn't report.
        memory_build_stats = {
            "memory_build_input_tokens": ledger.build_stats.get("input_tokens", 0),
            "memory_build_output_tokens": ledger.build_stats.get("output_tokens", 0),
            "memory_build_calls": ledger.build_stats.get("calls", 0),
        }
        if _usage_before is not None and hasattr(builder_client, "get_usage_totals"):
            _ua = builder_client.get_usage_totals()
            _dcalls = _ua.get("calls", 0) - _usage_before.get("calls", 0)
            if _dcalls > 0:
                memory_build_stats = {
                    "memory_build_input_tokens": _ua.get("input_tokens", 0) - _usage_before.get("input_tokens", 0),
                    "memory_build_output_tokens": _ua.get("output_tokens", 0) - _usage_before.get("output_tokens", 0),
                    "memory_build_calls": _dcalls,
                }
        print(f"[run_eval] memory ledger ready: {len(memory_checkpoints)} checkpoints, "
              f"{memory_build_stats['memory_build_calls']} build calls, "
              f"{memory_build_stats['memory_build_input_tokens']:,}+"
              f"{memory_build_stats['memory_build_output_tokens']:,} build tokens", flush=True)

    if args.build_only:
        # Decoupled build phase: the ledger is now persisted under
        # {run_dir}/memory_states/ (build_checkpoints dumped it). Exit before
        # answering so the build can run at high cross-persona concurrency
        # (builds barely touch the answer rate budget); a later --resume run
        # reloads these checkpoints via load_existing_checkpoints and only
        # answers. No-op for modes without a persisted ledger (llm_longctx).
        print(f"[run_eval] --build_only: ledger built for user {args.user_id}; "
              f"skipping answer phase.", flush=True)
        return 0

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
    # Cache locality for the single-call LLM modes: dispatch in ascending-T_test
    # order so each request's hoisted+chronological history is a PREFIX-EXTENSION
    # of the previous one — provider implicit caching then reuses the shared
    # leading history instead of re-billing it. Only meaningful when run
    # sequentially in one process (--workers 1) so requests hit the server-side
    # cache of the immediately-prior request. No effect on scoring (keyed by
    # query_id). Other modes keep their original order.
    if args.mode in ("llm_longctx", "llm_memory", "mem0"):
        parallel_rows.sort(key=lambda r: int(r.get("ts") or 0))
        if args.workers != 1:
            print(f"[run_eval] NOTE: mode={args.mode} caches best at --workers 1 "
                  f"(ascending-T, one process); got --workers {args.workers} — "
                  f"cache hit-rate will be lower.", flush=True)
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
        # mem0 per-query retrieval: stash this row's query so get_or_build can use
        # it without every task threading `query=` through (mem0 runs single-
        # threaded in-process, so this shared slot is race-free for that mode).
        if ctx.snapshot_cache is not None:
            ctx.snapshot_cache._pending_query = inst.get("user_query")
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
            payloads = [_build_payload(row, args, run_dir, per_worker, memory_checkpoints)
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

    # --prune_invalid: drop any rows that still failed after retries so the
    # aggregate (and per-persona summary below) reflect only completed rows.
    if args.prune_invalid:
        kept, dropped = _rewrite_keep_ok(results_csv)
        if dropped:
            print(f"[run_eval] prune_invalid: removed {dropped} still-invalid "
                  f"rows (kept {kept}).")
        written_results = [r for r in written_results
                           if (r.get("status") or "") == "ok"]

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
    # Memory mode: fold the one-time memory-build cost into the headline totals
    # so the cost report is honest (build is paid once per user, amortized over
    # all query rows; per-query answer tokens are already counted above).
    if memory_build_stats:
        persona_totals.update(memory_build_stats)
        persona_totals["total_tokens"] += (
            memory_build_stats.get("memory_build_input_tokens", 0)
            + memory_build_stats.get("memory_build_output_tokens", 0)
        )

    # Real provider-billed cost from the answer client's accumulated usage
    # (input/output/CACHED tokens as the API reported them). The per-row
    # metrics_json counts tokens locally and can't see cache hits, so this is
    # the honest cost number under the implicit-cache layout. Only populated
    # when the answer client ran IN-PROCESS (--workers 1) — with a ProcessPool
    # the parent client's counters stay empty (workers hold their own).
    api_cost = None
    try:
        if llm_client is not None and hasattr(llm_client, "get_usage_totals"):
            u = llm_client.get_usage_totals()
            if u.get("calls"):
                from evaluation.cost_model import RATES, gemini_cost
                in_tok = int(u.get("input_tokens", 0))
                out_tok = int(u.get("output_tokens", 0))
                cached = int(u.get("cached_input_tokens", 0))
                api_cost = {
                    "api_calls": int(u.get("calls", 0)),
                    "api_input_tokens": in_tok,
                    "api_output_tokens": out_tok,
                    "api_cached_input_tokens": cached,
                    "api_cache_hit_rate": round(cached / in_tok, 4) if in_tok else 0.0,
                }
                if args.model in RATES and "in_hi" not in RATES[args.model]:
                    # Gemini's prompt_token_count (recorded as input_tokens) is the
                    # TOTAL prompt incl. cached; cached_content_token_count is the
                    # cached subset. gemini_cost splits uncached = input - cached
                    # and bills cached at the cache rate.
                    api_cost["api_cost_usd"] = round(
                        gemini_cost(args.model, in_tok, out_tok,
                                    cached_input_tokens=cached, batch=False), 4)
                persona_totals.update(api_cost)
    except Exception as e:  # cost reporting must never break the run
        print(f"[run_eval] WARN: API cost report failed: {type(e).__name__}: {e}", file=sys.stderr)

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
    if api_cost is not None:
        print(f"[run_eval] API-billed: calls={api_cost['api_calls']}, "
              f"input={api_cost['api_input_tokens']:,} "
              f"(cached {api_cost['api_cached_input_tokens']:,} = "
              f"{api_cost['api_cache_hit_rate']*100:.1f}%), "
              f"output={api_cost['api_output_tokens']:,}, "
              f"cost=${api_cost.get('api_cost_usd', float('nan')):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
