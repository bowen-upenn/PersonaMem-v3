#!/usr/bin/env python3
"""Build the per-persona eval benchmark as ONE CSV per user.

This is the SINGLE entry point for benchmark construction — no separate
`benchmark.json` step. The CSV at `benchmark/{uid}/queries.csv` contains
both:

  - Narrow scannable columns (query_id, task_type, ts, query_text, etc.)
    for the HuggingFace-facing view of the benchmark.
  - An `instance_json` column with the full instance payload the runner
    needs for scoring. This makes the CSV the single source of truth
    for both eval ordering and eval execution — no JSON sidecar.

E6 (Active Mistake Prevention) is built INLINE here, same as every
other task family. An LLM client is built from env config automatically
for the discovery step; if no LLM is available, E6 yields zero instances.

Sort order:
    primary   : `ts` ascending (strict temporal order)
    secondary : `hash(instance_id) % 10**9` (interleaves task types
                within same-timestamp groups; deterministic)

CLI:
    python scripts/prepare_eval_data.py --user_id 115
    python scripts/prepare_eval_data.py --user_range 100-200 --parallel 8
    python scripts/prepare_eval_data.py --all --parallel 16

Missing `backend/{uid}` → user is skipped and logged to
`benchmark/_prepare_eval_data.skipped.txt`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Load .env BEFORE any os.getenv for LLM credentials (AZURE_OPENAI_*, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
except Exception:
    pass

from evaluation.audit_query_quality import (  # noqa: E402
    TOOL_CALL_VALIDITY_TASKS,
    check_tool_call_deterministic,
)
from evaluation.backend_query import BackendQuery  # noqa: E402
from evaluation.build_benchmark import build_benchmark  # noqa: E402
from evaluation.task_registry import (  # noqa: E402
    QUERIES_CSV_VERSION,
    TASK_TYPE_META,
    get_meta,
)


# CSV columns: 15 narrow scannable columns + 1 `instance_json` payload
# column for the runner. HF data viewers render the narrow columns
# front-and-center; instance_json is big and trailing.
COLUMNS: list[str] = [
    "query_id",
    "seq",
    "user_id",
    "task_family",
    "task_type",
    "instance_id",
    "ts",
    "ts_iso",
    "query_text",
    "app_context",
    "entry_point",
    "mcp_tools_allowed",
    "state_write_policy",
    "expected_response_kind",
    "rubric_tags",
    "instance_json",
]


def _build_llm_client() -> object | None:
    """Best-effort LLM client for the E6 discovery step. Returns None
    when no credentials are configured — E6 then yields zero instances.
    """
    model = os.getenv("EVAL_DISCOVERY_MODEL") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not model:
        print("[prepare_eval_data] no LLM model in env "
              "(EVAL_DISCOVERY_MODEL / AZURE_OPENAI_DEPLOYMENT_NAME) — E6 will be skipped")
        return None
    try:
        from query_llm import QueryLLM
        client = QueryLLM({"models": {"llm_model": model}}, rate_limit_per_min=50)
        print(f"[prepare_eval_data] E6 discovery client ready (model={model})")
        return client
    except Exception as exc:
        print(f"[prepare_eval_data] WARN: could not build LLM client for E6: {exc}")
        return None


def _skipped_log_path() -> Path:
    return Path("benchmark") / "_prepare_eval_data.skipped.txt"


def _append_skipped(user_id: str, reason: str) -> None:
    path = _skipped_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now(dt.timezone.utc).isoformat()}\t{user_id}\t{reason}\n")


def _extract_ts(inst: dict) -> int:
    """Unified timestamp extraction with the same fallback chain used by
    export_benchmark_csv. Returns 0 when no timestamp is recoverable.
    """
    for key in ("t_test", "test_timestamp", "source_timestamp",
                "t_probe", "t_late", "t_anchor"):
        v = inst.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    return 0


def _ts_iso(ts: int) -> str:
    if ts <= 0:
        return ""
    try:
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def _secondary_sort_key(instance_id: str) -> int:
    """Stable pseudo-random tie-break for within-timestamp interleaving."""
    h = hashlib.md5(str(instance_id).encode("utf-8")).hexdigest()
    return int(h[:12], 16) % 10**9


def _inst_field(inst: dict, *names: str, default: str = "") -> str:
    """Return the first string-coerced non-empty value among `names`."""
    for n in names:
        v = inst.get(n)
        if v not in (None, "", [], {}):
            return str(v)
    return default


def _synthesize_ts_for_no_ts_instance(
    inst: dict,
    task_type: str,
    benchmark: dict,
    user_id: str,
) -> int:
    """Fallback timestamp for instances that have none.

    c1b_sequences is the known offender — one instance, no per-instance
    timestamp. Place it at (latest observed ts in this benchmark - 1h)
    so it lands reasonably near the rest of the day-8 cluster rather than
    dropping out of the manifest.
    """
    latest = 0
    for bucket in benchmark.values():
        if not isinstance(bucket, list):
            continue
        for other in bucket:
            if isinstance(other, dict):
                t = _extract_ts(other)
                if t > latest:
                    latest = t
    if latest > 0:
        return latest - 3600
    return 0


def _project_row(
    seq: int,
    task_type: str,
    inst: dict,
    user_id: str,
    ts: int,
) -> dict:
    meta = get_meta(task_type)
    instance_id = _inst_field(
        inst, "instance_id", "pair_id", "scenario_id", "test_id",
        default=f"{task_type}_{seq}",
    )
    query_text = _inst_field(inst, "query", "user_message", "user_query")
    app_context = _inst_field(inst, "app", "target_app")
    entry_point = _inst_field(inst, "entry_point", "query_type")
    # instance_json — full payload for the runner. Separator compact to
    # keep the column narrow-ish in HF viewers (default=', ' → ',').
    instance_json = json.dumps(inst, ensure_ascii=False, separators=(",", ":"))
    return {
        "query_id": f"{user_id}:{seq:04d}:{instance_id}",
        "seq": seq,
        "user_id": user_id,
        "task_family": meta["task_family"],
        "task_type": task_type,
        "instance_id": instance_id,
        "ts": ts,
        "ts_iso": _ts_iso(ts),
        "query_text": query_text,
        "app_context": app_context,
        "entry_point": entry_point,
        "mcp_tools_allowed": meta["mcp_tools_allowed"],
        "state_write_policy": meta["state_write_policy"],
        "expected_response_kind": meta["expected_response_kind"],
        "rubric_tags": ";".join(meta["rubric_tags"]),
        "instance_json": instance_json,
    }


def _make_blind_check_llm(model: str):
    """Return a callable `llm(prompt) -> str` for Task B blind-check routing.

    Earlier this used `claude -p --model haiku` subagent calls (subscription
    auth, no API key) — but each spawn took ~10s and the build runs 250+
    blind checks sequentially → 30+ min wait. Switched to a direct API call
    via `QueryLLM` against `gpt-5.4-mini` (or whatever model name is passed),
    which is ~50x faster and parallelizable.

    The model arg is the deployment / model name (e.g. "gpt-5.4-mini").
    Falls back to None when no API credentials are configured, in which
    case Task B routing degrades to the fixed blind_score=2 default.
    """
    if not model:
        return None
    # Map historical Claude Code aliases ("haiku") to the gpt-5.4-mini
    # equivalent so older `--blind_check_model haiku` invocations still work.
    if model in ("haiku", "claude-haiku", "claude-haiku-4-5"):
        model = os.getenv("AZURE_OPENAI_BLIND_CHECK_DEPLOYMENT") or "gpt-5.4-mini"
    try:
        from query_llm import QueryLLM
        client = QueryLLM({"models": {"llm_model": model}}, rate_limit_per_min=200)
        print(f"[prepare_eval_data] blind-check client ready (model={model})")
    except Exception as exc:
        print(f"[prepare_eval_data] WARN: blind-check client init failed ({exc}); skipping")
        return None

    def _call(prompt: str) -> str:
        try:
            return client.query_llm(prompt) or ""
        except Exception:
            return ""
    return _call


def _build_benchmark_in_memory(
    user_id: str,
    backend_dir: Path,
    discovery_llm=None,
    blind_check_llm=None,
    blind_check_limit: int | None = None,
) -> dict | None:
    """Build the benchmark dict in memory. No disk persistence of JSON.

    The CSV written by prepare_one() is the sole on-disk artifact.
    """
    user_backend = backend_dir / user_id
    if not user_backend.exists():
        _append_skipped(user_id, f"backend/{user_id} missing — nothing to build from")
        return None

    try:
        return build_benchmark(
            backend_dir=str(backend_dir),
            user_id=user_id,
            discovery_llm=discovery_llm,
            blind_check_llm=blind_check_llm,
            blind_check_limit=blind_check_limit,
        )
    except SystemExit as e:
        _append_skipped(user_id, f"build_benchmark raised SystemExit: {e}")
        return None
    except Exception as e:
        _append_skipped(user_id, f"build_benchmark raised {type(e).__name__}: {e}")
        return None


def _drop_invalid_tool_call_instances(
    bm: dict,
    user_id: str,
    backend_dir: Path,
    verbose: bool = True,
) -> None:
    """Mutate `bm` in place: drop agentic / E3 / E6 instances whose tool-call
    layer fails deterministic validation.

    Failures are appended to
    `benchmark/{user_id}/build_benchmark.dropped.jsonl` so a reviewer can see
    exactly which instances were filtered and why. Format: one JSON object
    per line with `{instance_id, task_type, drop_reason}`.

    Only `TOOL_CALL_VALIDITY_TASKS` are checked. Other buckets are untouched.
    """
    try:
        bq = BackendQuery(str(backend_dir))
    except Exception as exc:
        if verbose:
            print(f"[{user_id}] WARN: tool-call gate skipped — couldn't init "
                  f"BackendQuery({backend_dir!r}): {exc}")
        return

    dropped: list[dict] = []
    for task_type in list(bm.keys()):
        if task_type not in TOOL_CALL_VALIDITY_TASKS:
            continue
        bucket = bm.get(task_type)
        if not isinstance(bucket, list) or not bucket:
            continue
        kept: list[dict] = []
        for inst in bucket:
            if not isinstance(inst, dict):
                kept.append(inst)
                continue
            check_inst = dict(inst)
            check_inst.setdefault("task_type", task_type)
            check_inst.setdefault("user_id", user_id)
            report = check_tool_call_deterministic(check_inst, bq)
            if not report["applicable"] or report["ok"]:
                kept.append(inst)
                continue
            dropped.append({
                "instance_id": inst.get("instance_id", "?"),
                "task_type": task_type,
                "drop_reason": "; ".join(report["errors"])[:400],
            })
        bm[task_type] = kept

    if dropped and verbose:
        print(f"[{user_id}] tool-call gate dropped {len(dropped)} instance(s) — "
              f"see benchmark/{user_id}/build_benchmark.dropped.jsonl")
    out_path = Path("benchmark") / user_id / "build_benchmark.dropped.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Always write (truncate) so the file reflects the LATEST run, not a
    # cumulative log. Empty file when nothing was dropped — useful as a
    # heartbeat that the gate ran.
    with out_path.open("w", encoding="utf-8") as f:
        for row in dropped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_one(
    user_id: str,
    backend_dir: Path,
    discovery_llm=None,
    blind_check_llm=None,
    blind_check_limit: int | None = None,
    postprocess_llm=None,
    enable_self_check: bool = True,
    enable_inferior: bool = True,
    verbose: bool = True,
) -> dict:
    """Build queries.csv for one user. Returns a small report dict.

    No benchmark.json is written. The CSV (with an `instance_json`
    column) is the sole on-disk artifact.

    Workstreams I + J: when ``postprocess_llm`` is provided, every
    personalization instance is post-processed to attach a self-check
    score and a paired inferior_response. ``enable_self_check`` /
    ``enable_inferior`` flag the two passes independently.
    """
    bm = _build_benchmark_in_memory(
        user_id, backend_dir,
        discovery_llm=discovery_llm,
        blind_check_llm=blind_check_llm,
        blind_check_limit=blind_check_limit,
    )
    if bm is None:
        return {"user_id": user_id, "rows": 0, "status": "skipped"}

    # Workstream I + J: post-build LLM passes.
    if postprocess_llm is not None and (enable_self_check or enable_inferior):
        try:
            from evaluation.backend_query import BackendQuery
            from evaluation.llm_postprocess import postprocess_benchmark
            bq = BackendQuery(backend_dir=str(backend_dir))
            postprocess_benchmark(
                bm, bq, user_id,
                self_check_llm=postprocess_llm if enable_self_check else None,
                inferior_llm=postprocess_llm if enable_inferior else None,
                verbose=verbose,
            )
        except Exception as exc:
            if verbose:
                print(f"[{user_id}] WARNING: postprocess failed: {exc}")

    # Build-time tool-call validation gate: drop any agentic / E3 / E6
    # instance whose declared tool calls reference unknown tool names OR
    # whose required read tools return zero data at the instance's t_test.
    # Schema-only (deterministic) — no LLM call. The full LLM judge sub-
    # check is reserved for the post-build audit pass.
    _drop_invalid_tool_call_instances(bm, user_id, backend_dir, verbose=verbose)

    pairs: list[tuple[str, dict, int]] = []
    unknown_task_types: set[str] = set()
    for task_type, bucket in bm.items():
        if not isinstance(bucket, list):
            continue
        if task_type not in TASK_TYPE_META:
            unknown_task_types.add(task_type)
        for inst in bucket:
            if not isinstance(inst, dict):
                continue
            ts = _extract_ts(inst)
            if ts <= 0:
                ts = _synthesize_ts_for_no_ts_instance(
                    inst, task_type, bm, user_id
                )
            if ts <= 0:
                if verbose:
                    print(f"[{user_id}] dropping {task_type} instance with no "
                          f"timestamp (id={inst.get('instance_id', '?')})")
                continue
            pairs.append((task_type, inst, ts))

    if not pairs:
        _append_skipped(user_id, "benchmark has zero instances with valid timestamps")
        return {"user_id": user_id, "rows": 0, "status": "empty"}

    # Sort: primary temporal, secondary hash-interleave. Using a tuple key
    # so ties on ts fall through to hash(instance_id).
    def _sort_key(item):
        task_type, inst, ts = item
        instance_id = _inst_field(
            inst, "instance_id", "pair_id", "scenario_id", "test_id",
            default=f"{task_type}_na",
        )
        return (ts, _secondary_sort_key(instance_id))
    pairs.sort(key=_sort_key)

    # Emit CSV
    csv_path = Path("benchmark") / user_id / "queries.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        # Version header comment; run_eval.py asserts this line on load.
        f.write(f"# queries_csv_version={QUERIES_CSV_VERSION}\n")
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for seq, (task_type, inst, ts) in enumerate(pairs):
            row = _project_row(seq, task_type, inst, user_id, ts)
            writer.writerow(row)

    if unknown_task_types and verbose:
        print(f"[{user_id}] WARNING: {len(unknown_task_types)} task_type(s) "
              f"not in TASK_TYPE_META — register them in "
              f"evaluation/task_registry.py: {sorted(unknown_task_types)}")

    # One-shot post-write artifacts: test.json dump, persona.html
    # re-render (so the timeline reflects the new buckets), and a
    # snapshot audit. All cheap, no LLM calls — kept inline so a
    # single ``prepare_eval_data.py`` invocation produces every artifact
    # a reviewer might want, no follow-up commands needed.
    test_json_path: str | None = None
    persona_html_path: str | None = None
    audit_md_path: str | None = None

    try:
        from data_preparation.visualize import dump_test_samples_json, generate_persona_html
        test_json_path = dump_test_samples_json(user_id)
        if verbose:
            print(f"[{user_id}] wrote {test_json_path}")
        persona_html_path = generate_persona_html(user_id)
        if verbose:
            print(f"[{user_id}] wrote {persona_html_path}")
    except Exception as exc:
        if verbose:
            print(f"[{user_id}] WARNING: post-write dump/render failed: {exc}")

    try:
        # Inline-call the audit driver — keeps the public CLI in
        # ``scripts/audit_test_queries.py`` for ad-hoc use without
        # forking a subprocess for every persona.
        from scripts.audit_test_queries import _run_audit
        audit_md_path, _ = _run_audit(str(user_id), "snapshot",
                                       benchmark_dir="benchmark",
                                       backend_dir="backend")
        if verbose:
            print(f"[{user_id}] wrote {audit_md_path}")
    except Exception as exc:
        if verbose:
            print(f"[{user_id}] WARNING: audit failed: {exc}")

    return {
        "user_id": user_id,
        "rows": len(pairs),
        "status": "ok",
        "csv_path": str(csv_path),
        "test_json_path": test_json_path,
        "persona_html_path": persona_html_path,
        "audit_md_path": audit_md_path,
        "unknown_task_types": sorted(unknown_task_types),
    }


def _prepare_one_worker(
    user_id: str,
    backend_dir_str: str,
    skip_e6: bool,
    blind_check_model: str | None,
    blind_check_limit: int | None,
    skip_self_check: bool = False,
    skip_inferior: bool = False,
) -> dict:
    """ProcessPool entry. Each worker rebuilds its own LLM clients from
    env (QueryLLM / claude-code subagent helpers don't pickle across processes)."""
    discovery = None if skip_e6 else _build_llm_client()
    blind = _make_blind_check_llm(blind_check_model) if blind_check_model else None
    postprocess = blind if (not skip_self_check or not skip_inferior) else None
    return prepare_one(
        user_id, Path(backend_dir_str),
        discovery_llm=discovery,
        blind_check_llm=blind,
        blind_check_limit=blind_check_limit,
        postprocess_llm=postprocess,
        enable_self_check=not skip_self_check,
        enable_inferior=not skip_inferior,
    )


def _resolve_user_ids(args: argparse.Namespace) -> list[str]:
    if args.user_id:
        return [args.user_id]
    if args.user_range:
        lo_s, hi_s = args.user_range.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        return [str(i) for i in range(lo, hi + 1)]
    if args.all:
        backend_dir = Path(args.backend_dir)
        uids = sorted(
            p.name for p in backend_dir.iterdir()
            if p.is_dir() and not p.name.startswith("_")
        )
        return uids
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build per-persona eval benchmark as a single CSV."
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--user_id", help="Single user id, e.g. 115")
    grp.add_argument("--user_range", help="Inclusive range A-B, e.g. 100-200")
    grp.add_argument("--all", action="store_true",
                     help="All users under --backend_dir")
    parser.add_argument("--backend_dir", default="backend")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Cross-user parallelism (ProcessPool)")
    parser.add_argument("--skip_e6", action="store_true",
                        help="Skip E6 discovery (saves the per-user LLM call "
                             "for paired warn/foil scenarios)")
    parser.add_argument("--skip_blind_check", action="store_true",
                        help="Skip Task B blind-check routing (proactive vs "
                             "over_personalization_chatbot_text). With this "
                             "flag, every chatbot candidate gets blind_score=2 "
                             "(default), which collapses control-arm coverage.")
    parser.add_argument("--blind_check_model", default="haiku",
                        help="Claude Code subagent model for Task B blind-check "
                             "(default: haiku — cheap; use sonnet for higher "
                             "routing fidelity)")
    parser.add_argument("--blind_check_limit", type=int, default=None,
                        help="Cap how many candidate queries get blind-checked "
                             "per user (for fast iteration)")
    parser.add_argument("--skip_self_check", action="store_true",
                        help="Workstream I: skip the per-instance self-check "
                             "of example_response against avoid_overpersonalization. "
                             "Saves ≈200 LLM calls per user.")
    parser.add_argument("--skip_inferior", action="store_true",
                        help="Workstream J: skip generation of paired "
                             "inferior_response foils. Saves ≈200 LLM calls per user.")
    args = parser.parse_args()

    user_ids = _resolve_user_ids(args)
    if not user_ids:
        print("No users resolved from args", file=sys.stderr)
        return 2

    backend_dir = Path(args.backend_dir)
    blind_check_model = None if args.skip_blind_check else args.blind_check_model

    # Two LLM hooks at benchmark-build time:
    #   - blind_check (Task B routing) — Claude Code subagent on `--blind_check_model`
    #   - E6 discovery (paired warn/foil) — QueryLLM via _build_llm_client()
    # Both are optional; flag a user out with --skip_blind_check / --skip_e6.
    print(f"Preparing queries.csv for {len(user_ids)} user(s) "
          f"(parallel={args.parallel}, "
          f"blind_check={'off' if args.skip_blind_check else args.blind_check_model}, "
          f"e6_discovery={'off' if args.skip_e6 else 'on'})")

    reports: list[dict] = []
    if args.parallel <= 1 or len(user_ids) == 1:
        discovery = None if args.skip_e6 else _build_llm_client()
        blind = _make_blind_check_llm(blind_check_model) if blind_check_model else None
        # Workstream I + J: reuse the blind-check client for the self-check
        # and inferior-generation passes — same gpt-5.4-mini deployment, no
        # need for a second client.
        postprocess = blind if (not args.skip_self_check or not args.skip_inferior) else None
        for uid in user_ids:
            try:
                reports.append(prepare_one(
                    uid, backend_dir,
                    discovery_llm=discovery,
                    blind_check_llm=blind,
                    blind_check_limit=args.blind_check_limit,
                    postprocess_llm=postprocess,
                    enable_self_check=not args.skip_self_check,
                    enable_inferior=not args.skip_inferior,
                ))
            except Exception as e:
                _append_skipped(uid, f"exception: {type(e).__name__}: {e}")
                reports.append({"user_id": uid, "rows": 0, "status": "error",
                                "error": str(e)})
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futs = {
                pool.submit(
                    _prepare_one_worker, uid, str(backend_dir),
                    args.skip_e6, blind_check_model, args.blind_check_limit,
                    args.skip_self_check, args.skip_inferior,
                ): uid
                for uid in user_ids
            }
            for fut in as_completed(futs):
                uid = futs[fut]
                try:
                    reports.append(fut.result())
                except Exception as e:
                    _append_skipped(uid, f"exception: {type(e).__name__}: {e}")
                    reports.append({"user_id": uid, "rows": 0, "status": "error",
                                    "error": str(e)})

    # Summary
    ok = [r for r in reports if r.get("status") == "ok"]
    empty = [r for r in reports if r.get("status") == "empty"]
    skipped = [r for r in reports if r.get("status") == "skipped"]
    errors = [r for r in reports if r.get("status") == "error"]
    total_rows = sum(r.get("rows", 0) for r in ok)
    print()
    print("=== prepare_eval_data summary ===")
    print(f"  ok      {len(ok):>5d}  ({total_rows} query rows total)")
    print(f"  empty   {len(empty):>5d}")
    print(f"  skipped {len(skipped):>5d}  (see {_skipped_log_path()})")
    print(f"  error   {len(errors):>5d}")
    if errors:
        for r in errors[:10]:
            print(f"    {r['user_id']}: {r.get('error', '?')}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
