#!/usr/bin/env python3
"""Inter-judge AGREEMENT metrics on the per-item accuracy scores (personas 1/2/3).

Baseline = GPT-5.5 per-item pr_query_score_0_10 from each live results.csv.
Alt judges = the .items.jsonl sidecars from rejudge_existing (gpt-5.4-mini, opus).
Same saved responses scored by each judge, joined on (config, uid, qid).

Reports, per alt judge (pooled over all configs + items): n, Pearson r,
Spearman rho, Krippendorff alpha (interval), mean |Δ| (0-10), within-±1 rate,
and Cohen's kappa on the binarized pass decision (score ≥ 5).
"""
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0, str(ROOT))
from evaluation.personalization_rubric import APPLICABILITY

OUT = Path("/tmp/eval_regen/judge_sens/p3")
REPORT_OUT = ROOT / "results/audit/judge_agreement_p3/accuracy_agreement.json"
USERS = ["1", "2", "3"]
CFGS = ["llm_longctx_gpt5.5_judged", "llm_memory_gpt5.5", "mem0_gpt5.5",
        "codex_agent_gpt5.5", "llm_longctx_gemini3.5flash_judged",
        "llm_memory_gemini3.5flash_judged", "agent_tools_opus4.8", "agent_tools_sonnet4.6"]
ALT = ["gpt-5.4-mini", "claude-opus-4.8"]
SCOPE = set(APPLICABILITY) - {"over_personalization_repetition_recsys",
    "over_personalization_repetition_chatbot", "new_suggestions_recsys", "new_suggestions_chatbot"}


def baseline_items(cfg):
    """(uid,qid) -> GPT-5.5 score from live csv."""
    d = {}
    for u in USERS:
        rf = ROOT / "results" / cfg / u / "results.csv"
        if not rf.exists():
            continue
        for row in csv.DictReader(open(rf)):
            if (row.get("status") or "").strip() != "ok":
                continue
            if row.get("task_type") not in SCOPE:
                continue
            try:
                m = json.loads(row.get("metrics_json") or "{}")
            except Exception:
                continue
            s = m.get("pr_query_score_0_10")
            if isinstance(s, (int, float)):
                d[(u, row.get("query_id"))] = float(s)
    return d


def alt_items(judge, cfg):
    f = OUT / f"{judge}.{cfg}.json.items.jsonl"
    if not f.exists():
        return None
    d = {}
    for line in open(f):
        r = json.loads(line)
        d[(r["uid"], r["qid"])] = float(r["score"])
    return d


def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    def rank(a):
        a = np.asarray(a); order = a.argsort(); r = np.empty(len(a)); r[order] = np.arange(len(a))
        # average ties
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        avg = np.zeros(len(cnt)); pos = np.zeros(len(cnt))
        # simple tie-aware: recompute via argsort groups
        sr = np.empty(len(a)); tmp = a.argsort(kind="mergesort"); i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[tmp[j + 1]] == a[tmp[i]]:
                j += 1
            sr[tmp[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return sr
    return pearson(rank(x), rank(y))


def kripp_interval(x, y):
    """2-coder interval Krippendorff alpha."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    Do = np.mean((x - y) ** 2)
    vals = np.concatenate([x, y])
    De = 2 * vals.var()
    return float(1 - Do / De) if De > 0 else float("nan")


def cohen_kappa_binary(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    po = np.mean(a == b)
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return float((po - pe) / (1 - pe)) if (1 - pe) else float("nan")


print(f"\n{'='*78}\nACCURACY — inter-judge AGREEMENT vs GPT-5.5 (per-item, personas 1/2/3)\n{'='*78}")
report = {}
for judge in ALT:
    xs, ys = [], []
    for cfg in CFGS:
        b = baseline_items(cfg); a = alt_items(judge, cfg)
        if not a:
            continue
        for k in b:
            if k in a:
                xs.append(b[k]); ys.append(a[k])
    if not xs:
        print(f"\n## {judge}: no items yet (pending)"); continue
    xs, ys = np.array(xs), np.array(ys)
    pb = (xs >= 5).astype(int); pa = (ys >= 5).astype(int)
    m = {"n": len(xs), "pearson_r": round(pearson(xs, ys), 3),
         "spearman_rho": round(spearman(xs, ys), 3),
         "krippendorff_alpha": round(kripp_interval(xs, ys), 3),
         "mean_abs_delta": round(float(np.mean(np.abs(xs - ys))), 2),
         "within_1_pt_pct": round(float(np.mean(np.abs(xs - ys) <= 1) * 100), 1),
         "pass5_agreement_pct": round(float(np.mean(pb == pa) * 100), 1),
         "pass5_cohen_kappa": round(cohen_kappa_binary(pb, pa), 3),
         "mean_gpt5.5": round(float(xs.mean()), 2), "mean_alt": round(float(ys.mean()), 2)}
    report[judge] = m
    print(f"\n## GPT-5.5  vs  {judge}   (n={m['n']} items)")
    print(f"   Pearson r            {m['pearson_r']}")
    print(f"   Spearman rho         {m['spearman_rho']}")
    print(f"   Krippendorff alpha   {m['krippendorff_alpha']}  (interval)")
    print(f"   mean |Δ| (0-10)      {m['mean_abs_delta']}")
    print(f"   within ±1 pt         {m['within_1_pt_pct']}%")
    print(f"   pass@5 agreement     {m['pass5_agreement_pct']}%   Cohen κ {m['pass5_cohen_kappa']}")
    print(f"   mean score           GPT-5.5 {m['mean_gpt5.5']}  /  {judge} {m['mean_alt']}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "accuracy_agreement.json").write_text(json.dumps(report, indent=2))
REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
REPORT_OUT.write_text(json.dumps(report, indent=2))
print(f"\nwrote {OUT/'accuracy_agreement.json'}")
print(f"wrote {REPORT_OUT}")
