#!/usr/bin/env python3
"""Re-run the benchmark QA audit with Claude Opus as the auditor (Opus has no
Azure deployment, so it is driven through `claude -p`). Reuses the EXACT audit
logic (evaluation.audit_query_quality.audit_query) and the same per-dimension
prompts, so any difference vs the GPT-5.5 audit is the auditor's, not the code's.

Writes results/audit/qa_audit_p{uid}_opus/{audit_rows.jsonl,audit_summary.json}.
"""
import argparse, json, os, shutil, subprocess, sys, time, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/_scripts"))
from evaluation.audit_query_quality import audit_query
from evaluation.backend_query import BackendQuery
from run_qa_audit import _flatten   # identical instance flattening


class ClaudeLLM:
    """Auditor LLM backed by `claude -p` (Opus). Tools disallowed so it answers
    the JSON audit prompt directly in one turn. Satisfies both the `llm(prompt)`
    and `llm.query_llm(prompt)` call styles the audit dims use."""
    DENY = ("Bash", "Edit", "Write", "Read", "Grep", "Glob", "WebFetch",
            "WebSearch", "Task", "NotebookEdit")

    def __init__(self, model="opus"):
        self.model = model
        self.bin = shutil.which("claude") or str(Path.home() / ".local/bin/claude")

    def query_llm(self, prompt, *a, **k):
        cmd = [self.bin, "-p", prompt, "--model", self.model,
               "--output-format", "json", "--max-turns", "2",
               "--setting-sources", "", "--disable-slash-commands",
               "--disallowedTools", *self.DENY]
        for attempt in range(3):
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=150)
                if p.stdout:
                    try:
                        return json.loads(p.stdout).get("result", "") or ""
                    except json.JSONDecodeError:
                        return p.stdout
            except subprocess.TimeoutExpired:
                pass
            time.sleep(2 * (attempt + 1))
        return ""

    def __call__(self, prompt):
        return self.query_llm(prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_id", default="1")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--workers", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--backend_dir", default="backend")
    args = ap.parse_args()

    items = json.loads((Path(args.backend_dir) / args.user_id / "test.json").read_text())
    if args.limit:
        items = items[: args.limit]
    out_dir = Path(args.out_dir or f"results/audit/qa_audit_p{args.user_id}_opus")
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = ClaudeLLM(args.model)
    try:
        bq = BackendQuery(args.backend_dir)
    except Exception as exc:
        print(f"[qa_opus] WARN BackendQuery init failed ({exc})"); bq = None

    print(f"[qa_opus] auditing {len(items)} queries (persona {args.user_id}) "
          f"with claude {args.model}, {args.workers}-way parallel")
    by_dim = defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0})
    by_task_dim = defaultdict(lambda: defaultdict(lambda: {"passed": 0, "failed": 0, "skipped": 0}))
    fail_examples = defaultdict(list)
    fh = (out_dir / "audit_rows.jsonl").open("w")
    lock = threading.Lock(); done = [0]; t0 = time.time()

    def audit_one(item):
        inst = _flatten(item, args.user_id)
        return audit_query(inst, llm, query_id=inst.get("query_id", ""), bq=bq)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(audit_one, it) for it in items]):
            res = fut.result()
            with lock:
                fh.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n"); fh.flush()
                for d in res.dimensions:
                    slot = by_dim[d.name]; tslot = by_task_dim[res.task_type][d.name]
                    if d.skipped:
                        slot["skipped"] += 1; tslot["skipped"] += 1
                    elif d.passed:
                        slot["passed"] += 1; tslot["passed"] += 1
                    else:
                        slot["failed"] += 1; tslot["failed"] += 1
                        if len(fail_examples[d.name]) < 5:
                            fail_examples[d.name].append({"query_id": res.query_id,
                                "task_type": res.task_type, "reason": (d.reason or "")[:240]})
                done[0] += 1
                if done[0] % 10 == 0 or done[0] == len(items):
                    print(f"[qa_opus]   {done[0]}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    fh.close()

    def rate(s):
        ev = s["passed"] + s["failed"]; return (s["passed"] / ev) if ev else None
    dim_summary = {n: {**s, "evaluated": s["passed"] + s["failed"], "pass_rate": rate(s)}
                   for n, s in by_dim.items()}
    tot_pass = sum(s["passed"] for s in by_dim.values())
    tot_fail = sum(s["failed"] for s in by_dim.values())
    summary = {"user_id": args.user_id, "model": f"claude-{args.model}", "n_queries": len(items),
               "overall": {"evaluated_checks": tot_pass + tot_fail, "passed": tot_pass,
                           "failed": tot_fail, "pass_rate": (tot_pass / (tot_pass + tot_fail)) if (tot_pass + tot_fail) else None},
               "by_dimension": dim_summary,
               "by_task_dimension": {tt: {n: {**s, "pass_rate": rate(s)} for n, s in dims.items()}
                                     for tt, dims in by_task_dim.items()},
               "fail_examples": dict(fail_examples)}
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[qa_opus] DONE {time.time()-t0:.0f}s -> {out_dir}  "
          f"overall pass_rate={summary['overall']['pass_rate']}")


if __name__ == "__main__":
    main()
