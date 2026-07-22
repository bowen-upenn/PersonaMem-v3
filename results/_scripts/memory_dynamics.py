#!/usr/bin/env python3
"""Iterative-memory dynamics from the GPT-5.5 textual-memory snapshots (no LLM).

- forgetting_curve(): of memory lines first added on day d, fraction still present
  once the memory saturates (~day 11). Exact-line identity (after stripping the
  [×N] freq tag), so it is an UPPER BOUND on true forgetting (rewording counts as
  a drop). Option A (semantic locator) gives the confound-free version.
- capacity(): memory size (lines) vs cumulative events, by day index.
- churn(): retained / added / removed lines per consecutive update.
"""
import json, glob, re, statistics, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import os
MATCHED = set(os.environ.get("PERSONAS", "").split()) or \
          {u for u in os.listdir("results/agent_tools_opus4.8") if u.isdigit()}
MEM_DIR = "llm_memory_gpt5.5"


def _lines(m):
    return [re.sub(r"\[×\d+\]", "", l).lower().strip()
            for l in m.split("\n") if l.strip().startswith("-")]


def _snaps(uid, mem_dir=MEM_DIR):
    return sorted((json.load(open(fp)) for fp in
                   glob.glob(f"{ROOT}/results/{mem_dir}/{uid}/memory_states/*.json")),
                  key=lambda d: d["t_test"])


def forgetting_curve(mem_dir=MEM_DIR):
    """Survival to the FINAL memory state (the memory freezes once events end at
    ~day 9, so the last snapshot — day 20–34 — equals the day-10 state)."""
    add_day = collections.defaultdict(list)
    for uid in MATCHED:
        snaps = _snaps(uid, mem_dir)
        if len(snaps) < 3:
            continue
        final = set(_lines(snaps[-1]["memory"]))
        first = {}
        for i, d in enumerate(snaps):                       # scan ALL snapshots
            for ln in _lines(d["memory"]):
                first.setdefault(ln, i)
        for ln, day in first.items():
            add_day[day].append(ln in final)
    days = sorted(d for d in add_day if len(add_day[d]) >= 10)
    return [(d, 100 * sum(add_day[d]) / len(add_day[d]), len(add_day[d])) for d in days]


def capacity():
    sizes, nev = collections.defaultdict(list), collections.defaultdict(list)
    for uid in MATCHED:
        for i, d in enumerate(_snaps(uid)):
            sizes[i].append(len(_lines(d["memory"]))); nev[i].append(d["n_events"])
    return [(i, statistics.mean(sizes[i]), statistics.mean(nev[i])) for i in sorted(sizes)]


def churn():
    ret = add = rem = 0
    for uid in MATCHED:
        snaps = _snaps(uid)
        for a, b in zip(snaps, snaps[1:]):
            sa, sb = set(_lines(a["memory"])), set(_lines(b["memory"]))
            ret += len(sa & sb); add += len(sb - sa); rem += len(sa - sb)
    return {"retained": ret, "added": add, "removed": rem,
            "pct_dropped_per_update": 100 * rem / (ret + rem) if (ret + rem) else 0}


if __name__ == "__main__":
    print("churn:", churn())
    print("\nforgetting curve (added_day, survive%, n):")
    for d, s, n in forgetting_curve():
        print(f"  day {d:>2}: {s:>3.0f}%  (n={n})")
    print("\ncapacity (day, mem_lines, n_events):")
    for i, s, e in capacity()[:14]:
        print(f"  day {i:>2}: {s:>3.0f} lines, {e:>5.0f} events")
