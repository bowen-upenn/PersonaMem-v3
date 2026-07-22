#!/usr/bin/env python3
"""Memory-size ablation: half (cap 2048) / baseline (4096) / double (8192) GPT-5.5
textual memory. Tests whether a bigger memory closes the lost-in-the-middle dip.
Reuses needle_depth_vs_accuracy for per-query accuracy + needle depth.
"""
import json, glob, statistics, collections, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_nd = importlib.util.spec_from_file_location("nd", ROOT / "results/_scripts/needle_depth_vs_accuracy.py")
nd = importlib.util.module_from_spec(_nd); _nd.loader.exec_module(nd)

CAPS = [
    ("Half · 2048", "llm_memory_gpt5.5_cap2048"),
    ("Baseline · 4096", "llm_memory_gpt5.5"),
    ("Double · 8192", "llm_memory_gpt5.5_cap8192"),
]


def mem_size(dirname):
    """Mean final-snapshot memory size (bullet lines, chars) over matched personas."""
    L, C = [], []
    for uid in nd.MATCHED:
        snaps = sorted((json.load(open(fp)) for fp in
                        glob.glob(f"{ROOT}/results/{dirname}/{uid}/memory_states/*.json")),
                       key=lambda d: d["t_test"])
        if not snaps:
            continue
        m = snaps[-1]["memory"]
        L.append(sum(1 for l in m.split("\n") if l.strip().startswith("-")))
        C.append(len(m))
    return (statistics.mean(L) if L else 0, statistics.mean(C) if C else 0)


def stats():
    out = []
    for label, d in CAPS:
        rows = nd.collect(d)
        allacc = [r["acc"] for r in rows]
        pr = [r for r in rows if r["depth"] is not None]   # all needle-bearing tasks

        def binned(edges):
            bk = collections.defaultdict(list)
            for r in pr:
                for i, (lo, hi) in enumerate(edges):
                    if lo <= r["depth"] < hi:
                        bk[i].append(r["acc"]); break
            vals = [statistics.mean(bk[i]) if bk.get(i) else None for i in range(len(edges))]
            ns = [len(bk.get(i, [])) for i in range(len(edges))]
            return vals, ns

        fine, fine_n = binned(nd.MIDBINS)        # fine-near / wide-middle / 10d+
        # dip = ends (first + last populated) minus the 3-5d & 5-7d middle (idx 3,4)
        midv = [fine[i] for i in (3, 4) if fine[i] is not None]
        endv = [fine[i] for i in (0, len(fine) - 1) if fine[i] is not None]
        mid = statistics.mean(midv) if midv else 0.0
        ends = statistics.mean(endv) if endv else 0.0
        lines, chars = mem_size(d)
        out.append({
            "label": label, "dir": d, "overall": statistics.mean(allacc), "n_all": len(allacc),
            "fine": fine, "fine_n": fine_n,
            "mid": mid, "ends": ends, "dip": ends - mid,
            "mem_lines": lines, "mem_chars": chars,
        })
    return out


if __name__ == "__main__":
    print(f"{'cap':18}{'mem lines':>10}{'mem kchars':>11}{'overall%':>10}"
          f"   {'/'.join(nd.MIDBIN_LABELS)}    dip")
    for s in stats():
        b = "/".join(f"{x:.0f}" if x is not None else "-" for x in s["fine"])
        print(f"{s['label']:18}{s['mem_lines']:>10.0f}{s['mem_chars']/1000:>11.1f}"
              f"{s['overall']:>10.1f}   {b}    {s['dip']:.0f}")
