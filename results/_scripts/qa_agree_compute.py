#!/usr/bin/env python3
"""Inter-model AGREEMENT on the benchmark quality-control checks.

Two checker models (gpt-5.5 = current, claude-opus-4.8) audit the SAME test.json
questions for a 3-persona sample. For each QC dimension we compare their pass/fail
verdicts on the items both actually evaluated (not skipped) and report per-model
pass rate, % agreement, and Cohen's kappa.

Reads audit_rows.jsonl from:
  gpt-5.5      : results/audit/qa_audit_p1/audit_rows.jsonl (p1) + qa_agree/gpt-5.5/p{2,3}
  opus         : results/audit/qa_agree/claude-opus-4.8/p{1,2,3}
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PERS = ["1", "2", "3"]
MODELS = {
    "gpt-5.5": lambda u: (ROOT / "results/audit/qa_audit_p1/audit_rows.jsonl") if u == "1"
        else (ROOT / f"results/audit/qa_agree/gpt-5.5/p{u}/audit_rows.jsonl"),
    "opus-4.8": lambda u: ROOT / f"results/audit/qa_agree/claude-opus-4.8/p{u}/audit_rows.jsonl",
}
NICE = {  # dim -> human label (matches the HTML QC section)
    "completeness": "Nothing missing", "schema_sanity": "Valid format",
    "naturalness": "Sounds like a real person", "no_giveaway": "Doesn't give itself away",
    "answerability": "Actually answers", "no_leakage": "No behind-the-scenes text",
    "context_required": "Truly needs this user", "context_restraint": "Fair don't-overshare",
    "gt_alignment": "Tests the right thing", "privacy_leak": "Keeps private things private",
    "inferior_axis_check": "Weaker answer fails right", "sensitive_probe_placement": "Clue before question",
    "telegraph_avoidance": "No telegraphing", "tool_call_validity": "Tool call valid",
}


def load(model):
    """(persona, query_id, dim) -> passed(bool), only non-skipped."""
    out = {}
    for u in PERS:
        f = MODELS[model](u)
        if not f.exists():
            return None  # incomplete
        for line in open(f):
            r = json.loads(line)
            qid = r["query_id"]
            for d in r.get("dimensions", []):
                if d.get("skipped"):
                    continue
                out[(u, qid, d["name"])] = bool(d["passed"])
    return out


data = {m: load(m) for m in MODELS}
missing = [m for m, v in data.items() if v is None]
if missing:
    print(f"INCOMPLETE — missing audit rows for: {missing}")
    import sys; sys.exit(0)

dims = sorted({k[2] for v in data.values() for k in v})


def cohen_kappa(labels):
    """labels: list of 2-tuples of bool. Cohen kappa, 2 categories, 2 raters."""
    n = len(labels)
    if n == 0:
        return None
    po = sum(a == b for a, b in labels) / n
    pa = sum(a for a, _ in labels) / n
    pb = sum(b for _, b in labels) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if (1 - pe) else 1.0


print(f"\n{'='*94}\nBENCHMARK QC — inter-model AGREEMENT on pass/fail (3-persona sample)\n{'='*94}")
print(f"{'check':26s} {'n*':>4s} {'GPT5.5':>6s} {'Opus':>6s} "
      f"{'agree%':>6s} {'kappa':>6s}")
print("-" * 70)
report = {}
all_pairs = []
for dim in dims:
    keys = [k for k in data["gpt-5.5"] if k[2] == dim]
    pairs = []
    for k in keys:
        if all(k in data[m] for m in MODELS):
            pairs.append(tuple(data[m][k] for m in MODELS))
    if not pairs:
        continue
    all_pairs += pairs
    n = len(pairs)
    pr = {m: sum(1 for t in pairs if t[i]) / n * 100 for i, m in enumerate(MODELS)}
    agr = sum(t[0] == t[1] for t in pairs) / n * 100
    kap = cohen_kappa(pairs)
    print(f"{NICE.get(dim,dim):26s} {n:>4d} {pr['gpt-5.5']:>6.0f} "
          f"{pr['opus-4.8']:>6.0f} {agr:>6.0f} {kap if kap is None else round(kap,2):>6}")
    report[dim] = {"label": NICE.get(dim, dim), "n": n,
                   "pass_rate": {m: round(pr[m], 1) for m in MODELS},
                   "agreement_pct": round(agr, 1), "cohen_kappa": None if kap is None else round(kap, 3)}

n = len(all_pairs)
agr = sum(t[0] == t[1] for t in all_pairs) / n * 100
print("-" * 70)
print(f"{'ALL CHECKS POOLED':26s} {n:>4d} {'':>6s} {'':>6s} {agr:>6.0f} "
      f"{round(cohen_kappa(all_pairs),2):>6}")
print("\n* n = items both models evaluated (none skipped). "
      "agree% = pass/fail agreement; kappa = Cohen (2 raters).")
out = ROOT / "results/audit/qa_agree/agreement.json"
out.write_text(json.dumps({"pooled": {"n": n, "agreement_pct": round(agr,1),
              "cohen_kappa": round(cohen_kappa(all_pairs),3)}, "by_dim": report}, indent=2))
print(f"wrote {out}")
