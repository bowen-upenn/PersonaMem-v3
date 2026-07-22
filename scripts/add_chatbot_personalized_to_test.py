#!/usr/bin/env python
"""Surgically ADD more chatbot_personalized_response rows to backend/{uid}/test.json.

Why: the builder supplies ~40 proactive personalized-response instances per persona,
but the share-balancing routing verifier collapsed the shipped count to ~5 (≈4% of
queries) — far under its min-20 floor. This tops it up to ~target (≈20% share) using
the SAME pipeline machinery (build_task_b_arms → postprocess → audit → projection) so
the new rows are current-style and audit-quality.

KEEPS existing rows byte-identical (incl. their query_ids); new rows get fresh,
non-colliding ids appended at the end, so a re-eval with
`--task chatbot_personalized_response --resume` only evaluates the NEW rows.

Usage:
  python scripts/add_chatbot_personalized_to_test.py --users 1,2,... --target 26 [--dry_run] [--limit N]
"""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
try:
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env", override=True)
except Exception:
    pass

from evaluation import build_benchmark as bb
from evaluation.backend_query import BackendQuery
from evaluation.inference_utils import load_test_items
from evaluation.llm_postprocess import postprocess_benchmark
from evaluation.audit_query_quality import audit_buckets
from data_preparation.visualize import dump_test_samples_json
from prepare_eval_data import _project_row, _build_llm_client, _extract_ts

CPR = "chatbot_personalized_response"


def _cid(c: dict) -> str:
    return str(c.get("test_id") or c.get("instance_id") or c.get("source_object_id") or "")


def _existing_cpr_ids(rows: list[dict]) -> set[str]:
    ids = set()
    for r in rows:
        if r.get("task_type") != CPR:
            continue
        qid = r.get("query_id", "")
        if qid.count(":") == 2:
            ids.add(qid.split(":")[2])
        inf = r.get("instance_full") or {}
        if inf.get("test_id"):
            ids.add(str(inf["test_id"]))
    return ids


def _row_iid(r: dict) -> str:
    inf = r.get("instance_full") or {}
    return str(inf.get("test_id") or r.get("instance_id") or "")


def _audit_pass(res) -> bool:
    return all(d.passed for d in res.dimensions if not d.skipped)


def process(uid: str, target: int, backend: str, limit: int | None, dry: bool) -> tuple:
    discovery = _build_llm_client()                       # per-persona client (own rate budget)
    call = (lambda p: (discovery.query_llm(p) or ""))     # callable: blind-check + postprocess
    tpath = Path(backend) / uid / "test.json"
    existing = json.loads(tpath.read_text())
    have = sum(1 for r in existing if r.get("task_type") == CPR)
    need = max(0, target - have)
    if need == 0:
        return (uid, have, 0, 0, "already at target")

    bq = BackendQuery(backend)
    test_items = load_test_items(backend, uid)
    out = bb.build_task_b_arms(backend, bq, uid, test_items,
                               blind_check_llm=call, blind_check_limit=limit,
                               discovery_llm=discovery)
    cands = out.get(CPR) or []
    have_ids = _existing_cpr_ids(existing)
    fresh = [c for c in cands if _cid(c) and _cid(c) not in have_ids
             and (c.get("held_out_preference") or {}).get("persona_item")]
    fresh.sort(key=lambda c: -(c.get("blind_check_score") or 0))
    pick = fresh[: need + max(6, need // 3)]          # over-pick to survive audit drops
    if not pick:
        return (uid, have, 0, len(existing), "no fresh candidates")

    bm = {CPR: pick}
    postprocess_benchmark(bm, bq, uid, self_check_llm=call, inferior_llm=call, verbose=False)
    # audit ONLY the task bucket — postprocess_benchmark adds a non-bucket
    # 'postprocess_stats' key to bm that audit_buckets would choke on.
    results, _summ = audit_buckets({CPR: list(bm.get(CPR) or [])}, discovery, bq=bq)
    kept = [c for c, res in zip(bm[CPR], results) if _audit_pass(res)][:need]
    if not kept:
        return (uid, have, 0, len(existing), f"all {len(pick)} failed audit")

    pairs = [(CPR, inst, _extract_ts(inst)) for inst in kept]
    csv_rows = [_project_row(seq, CPR, inst, uid, ts) for seq, (_tt, inst, ts) in enumerate(pairs)]
    tmp = Path("/tmp/cpr_gen") / uid
    tmp.mkdir(parents=True, exist_ok=True)
    dump_test_samples_json(uid, output_path=str(tmp / "test.json"),
                           backend_dir=backend, precomputed_rows=csv_rows)
    new_final = json.loads((tmp / "test.json").read_text())

    base = len(existing)                              # fresh, non-colliding seqs
    for k, r in enumerate(new_final):
        iid = _row_iid(r) or f"cprgen{k}"
        r["query_id"] = f"{uid}:{base + k:04d}:{iid}"
    merged = existing + new_final
    if not dry:
        tpath.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    return (uid, have, len(new_final), len(merged), "ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", required=True, help="comma-separated persona ids")
    ap.add_argument("--target", type=int, default=26, help="desired chatbot_personalized_response rows per persona")
    ap.add_argument("--backend_dir", default="backend")
    ap.add_argument("--limit", type=int, default=None, help="cap builder candidates (blind_check_limit) for speed")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    users = [u.strip() for u in args.users.split(",") if u.strip()]

    summ = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, u, args.target, args.backend_dir, args.limit, args.dry_run): u
                for u in users}
        for fut in as_completed(futs):
            u = futs[fut]
            try:
                s = fut.result()
            except Exception as exc:
                import traceback; traceback.print_exc()
                print(f"[{u}] ERROR: {exc}", file=sys.stderr); continue
            summ.append(s)
            print(f"[{s[0]}] had={s[1]} added={s[2]} total_rows={s[3]} ({s[4]})", file=sys.stderr, flush=True)

    print("\n=== SUMMARY ===", file=sys.stderr)
    print(f"users={len(summ)}  CPR rows added={sum(s[2] for s in summ)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
