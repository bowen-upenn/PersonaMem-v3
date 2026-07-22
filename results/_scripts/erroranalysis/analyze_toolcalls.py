#!/usr/bin/env python3
"""Parse the captured agent_tools stream-json transcripts (persona 1, agentic
tasks) into per-row tool-call sequences and classify tool-call-STEP failures:
did the agent search at all, did its searches come up empty, did it read what it
found, did it thrash? Cross-references the run's results.csv for the final answer."""
import json, os, re, csv, glob
from collections import Counter, defaultdict
csv.field_size_limit(10**8)

CAP = "/tmp/pm3_toolcap"
RUN = "/tmp/pm3_agentic_p1/results.csv"

SEARCH_BASH = re.compile(r"\b(grep|rg|find|ls|glob)\b", re.I)
READ_BASH = re.compile(r"\b(cat|sed|head|tail|less|awk|jq)\b", re.I)
EMPTY_RESULT = re.compile(r"no matches found|no files found|0 matches|^\s*$|found 0 |no such file|"
                          r"\bno results\b|did not match|nothing", re.I)


def _blocks(msg):
    c = msg.get("content")
    return c if isinstance(c, list) else []


def parse(path):
    """Return ordered list of steps: ('search'|'read'|'other', name, detail, result_text, empty)."""
    events = []
    results_by_id = {}
    raw_lines = [l for l in open(path) if l.strip()]
    objs = []
    for l in raw_lines:
        try:
            objs.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    # first pass: collect tool_results
    for o in objs:
        if o.get("type") == "user":
            for b in _blocks(o.get("message", {})):
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    txt = b.get("content")
                    if isinstance(txt, list):
                        txt = " ".join(x.get("text", "") for x in txt if isinstance(x, dict))
                    results_by_id[b.get("tool_use_id")] = (str(txt or ""), bool(b.get("is_error")))
    # second pass: tool_use steps in order
    steps = []
    num_turns = 0
    final = ""
    for o in objs:
        if o.get("type") == "result":
            num_turns = o.get("num_turns") or 0
            final = o.get("result") or ""
        if o.get("type") == "assistant":
            for b in _blocks(o.get("message", {})):
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    name = b.get("name", "")
                    inp = b.get("input", {}) or {}
                    if name in ("Grep", "Glob"):
                        kind = "search"; detail = inp.get("pattern") or inp.get("glob") or json.dumps(inp)[:80]
                    elif name == "Read":
                        kind = "read"; detail = os.path.basename(str(inp.get("file_path", "")))
                    elif name == "Bash":
                        cmd = str(inp.get("command", ""))
                        if SEARCH_BASH.search(cmd):
                            kind = "search"
                        elif READ_BASH.search(cmd):
                            kind = "read"
                        else:
                            kind = "other"
                        detail = cmd[:90]
                    else:
                        kind = "other"; detail = name
                    rtxt, rerr = results_by_id.get(b.get("id"), ("", False))
                    empty = kind == "search" and (rerr or len(rtxt.strip()) < 8 or
                                                  bool(EMPTY_RESULT.search(rtxt[:200])))
                    steps.append((kind, name, detail, empty))
    return steps, num_turns, final


def classify(steps):
    searches = [s for s in steps if s[0] == "search"]
    reads = [s for s in steps if s[0] == "read"]
    empties = [s for s in searches if s[3]]
    if not searches and not reads:
        return "answered_blind", "made no search/read calls — answered from the prompt only"
    if searches and len(empties) == len(searches):
        return "all_searches_empty", f"all {len(searches)} searches returned nothing, then answered anyway"
    if searches and not reads:
        return "searched_never_read", f"{len(searches)} searches but never Read a file — answered off snippets"
    if len(searches) >= 5 and len(empties) >= 0.6 * len(searches):
        return "thrashed", f"{len(empties)}/{len(searches)} searches empty — thrashed before answering"
    return "searched_ok", f"{len(searches)} searches ({len(empties)} empty), {len(reads)} reads"


def main():
    # map query_id -> task_type, final answer from results.csv
    meta = {}
    if os.path.exists(RUN):
        for r in csv.DictReader(open(RUN)):
            meta[r["query_id"]] = (r["task_type"], (r.get("agent_response") or "")[:200])
    files = sorted(glob.glob(f"{CAP}/*.jsonl"))
    print(f"transcripts: {len(files)}\n")
    by_label = Counter()
    by_task_label = defaultdict(Counter)
    rows = []
    for f in files:
        qid = os.path.basename(f)[:-6]
        steps, nturns, final = parse(f)
        label, why = classify(steps)
        task = meta.get(qid, ("?", ""))[0]
        ns = sum(1 for s in steps if s[0] == "search")
        ne = sum(1 for s in steps if s[0] == "search" and s[3])
        nr = sum(1 for s in steps if s[0] == "read")
        by_label[label] += 1
        by_task_label[task][label] += 1
        rows.append((task, qid, nturns, ns, ne, nr, label, why, steps))
    print("=== tool-call-step outcome distribution (28 agentic rows, persona 1) ===")
    for lab, n in by_label.most_common():
        print(f"  {lab:22s} {n:3d}  ({100*n/len(files):.0f}%)")
    print("\n=== by task type ===")
    for task in sorted(by_task_label):
        print(f"  {task:30s} {dict(by_task_label[task])}")
    print("\n=== per-row detail ===")
    for task, qid, nt, ns, ne, nr, lab, why, steps in sorted(rows):
        print(f"  [{task:26s}] turns={nt} search={ns}(empty {ne}) read={nr}  -> {lab}: {why}")
    print("\n=== sample failing search queries (empties) ===")
    shown = 0
    for task, qid, nt, ns, ne, nr, lab, why, steps in rows:
        if lab in ("answered_blind", "all_searches_empty", "thrashed", "searched_never_read"):
            qs = [d for k, n, d, e in steps if k == "search" and e][:4]
            print(f"  [{task}] {lab}: empty-searches={qs}")
            shown += 1
        if shown >= 14:
            break

    json.dump([{"task": t, "qid": q, "turns": nt, "n_search": ns, "n_empty": ne,
                "n_read": nr, "label": lab} for t, q, nt, ns, ne, nr, lab, _, _ in rows],
              open("results/_scripts/erroranalysis/toolcall_steps_p1.json", "w"), indent=1)


if __name__ == "__main__":
    main()
