#!/usr/bin/env python3
"""Surgical regen of ONLY the hidden_persona_recommendation slates in
backend/{uid}/test.json — replaces each existing entry's slate (candidates /
held_out_idx / hard_negative_idxs) with a freshly generated one, leaving every
other task + field untouched. 1:1 by persona label, reusing each entry's t_test.

Parallelism: ENTRY-level via a thread pool (default 50) sharing ONE gpt-5.5
discovery client — generates all ~117 slates concurrently (API allows ~50).

Usage:
  python scripts/regen_hidden_persona_slates.py --users 1,2,3 [--apply] [--max-parallel 50]
"""
import argparse, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, ".")
from evaluation.backend_query import BackendQuery
from scripts.prepare_eval_data import _build_llm_client
import evaluation.tasks.hidden_persona_recommendation as HP
from evaluation.llm_postprocess import (
    _compute_ranking_example, _compute_ranking_inferior,
)
from data_preparation.visualize import _gt_hidden_persona_recommendation

BACKUP_DIR = "backend/_hpregen_backup"
_GTP_RE = re.compile(r"Hidden persona \([^)]*\)(?:\s*[—-]\s*privacy-flagged)?:\s*(.+)")


def _as_obj(x):
    if isinstance(x, str):
        try: return json.loads(x)
        except Exception: return {}
    return x


def _gtp_text(inf):
    g = inf.get("groundtruth_preference")
    return g if isinstance(g, str) else (json.dumps(g) if g else "")


def _label_of(inf):
    m = _GTP_RE.match(_gtp_text(inf).split("\n", 1)[0])
    return m.group(1).strip() if m else ""


def _regen_slate(hp_persona, all_personas, canon, uctx, seed):
    is_pf = HP._hp_is_privacy_flagged(hp_persona)
    others = [o for o in all_personas if o is not hp_persona]
    parsed = HP._discover_slate(HP_DISCOVERY, hp_persona, is_pf, others, canon, user_context=uctx, verbose=False)
    if not parsed:
        return None
    items = parsed["items"]; ti = parsed["target_index"]
    decoys = parsed.get("surface_decoy_indices") or []
    cands = HP._items_to_candidates(items)
    order = list(range(HP.SLATE_SIZE)); random.Random(seed).shuffle(order)
    shuffled = [cands[j] for j in order]
    held = order.index(ti)
    hard = sorted(order.index(d) for d in decoys if 0 <= d < HP.SLATE_SIZE)
    return shuffled, held, hard


def build_tasks(users):
    """Return (tasks, state). tasks=[(uid,qid,args...)]. state[uid]=(data,insts,entries_by_qid)."""
    tasks = []; state = {}
    for uid in users:
        path = f"backend/{uid}/test.json"
        if not os.path.exists(path):
            print(f"[{uid}] no test.json — skip"); continue
        bq = BackendQuery(backend_dir="backend")
        prof = bq.get_full_profile(str(uid)) or {}
        allp = prof.get("hidden_personas") or []
        by_label = {(h.get("label") or "").strip(): h for h in allp}
        canon = [str(p) for p in (prof.get("preferences") or [])]
        uctx = HP._build_user_context(prof)
        data = json.load(open(path))
        insts = data if isinstance(data, list) else data.get("instances", [])
        ebyq = {}
        for e in insts:
            if (e.get("task_type") or e.get("task")) != "hidden_persona_recommendation":
                continue
            inf = _as_obj(e.get("instance_full")); label = _label_of(inf)
            hp = by_label.get(label)
            if hp is None:
                gtp = _gtp_text(inf)
                for k, v in by_label.items():
                    if k and k in gtp: hp, label = v, k; break
            qid = e.get("query_id")
            ebyq[qid] = e
            if hp is None:
                print(f"[{uid}] {qid}: persona {label!r} not in profile — keep old"); continue
            seed = abs(hash(qid or label)) % (2**31)
            tasks.append((str(uid), qid, hp, allp, canon, uctx, seed))
        state[str(uid)] = (data, insts, ebyq, path)
    return tasks, state


def main():
    global HP_DISCOVERY
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--max-parallel", type=int, default=50)
    a = ap.parse_args()
    users = [u.strip() for u in a.users.split(",") if u.strip()]
    HP_DISCOVERY = _build_llm_client()
    if HP_DISCOVERY is None:
        print("ERROR: discovery LLM unavailable", file=sys.stderr); return 2

    tasks, state = build_tasks(users)
    print(f"[regen] {len(tasks)} slates across {len(state)} users, max_parallel={a.max_parallel}", flush=True)
    results = {}; done = 0
    def run(t):
        uid, qid, hp, allp, canon, uctx, seed = t
        try: return (uid, qid, _regen_slate(hp, allp, canon, uctx, seed))
        except Exception as exc: return (uid, qid, ("ERR", str(exc)))
    with ThreadPoolExecutor(max_workers=a.max_parallel) as ex:
        futs = [ex.submit(run, t) for t in tasks]
        for f in as_completed(futs):
            uid, qid, res = f.result(); results[(uid, qid)] = res
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"[regen] {done}/{len(tasks)} done", flush=True)

    # apply per user
    for uid, (data, insts, ebyq, path) in state.items():
        n_ok = n_fail = 0
        for (u, qid), res in results.items():
            if u != uid: continue
            if not res or (isinstance(res, tuple) and res and res[0] == "ERR"):
                n_fail += 1; continue
            shuffled, held, hard = res
            e = ebyq[qid]; inf_raw = e.get("instance_full"); inf = _as_obj(inf_raw)
            inf["candidates"] = shuffled; inf["held_out_idx"] = held; inf["hard_negative_idxs"] = hard
            # Re-derive the golds from the NEW slate. Mutating candidates /
            # held_out_idx / hard_negative_idxs WITHOUT this leaves
            # example_response + inferior_response + the "Top item" GT stale
            # relative to the slate — the ranked gold then buries the held-out
            # target and the GT names an item absent from the slate
            # (audit 2026-07-16, T2-1). Update BOTH instance_full and top-level.
            _tt = "hidden_persona_recommendation"
            _ex = _compute_ranking_example(inf, _tt)
            _inf_rank = _compute_ranking_inferior(inf, _tt)
            _gt = _gt_hidden_persona_recommendation(inf) or {}
            if _ex:
                inf["example_response"] = _ex; e["example_response"] = _ex
            if _inf_rank:
                inf["inferior_response"] = _inf_rank
                if isinstance(e.get("inferior_response"), dict):
                    e["inferior_response"] = {**e["inferior_response"], "text": _inf_rank}
                else:
                    e["inferior_response"] = _inf_rank
            if _gt.get("groundtruth_preference"):
                e["groundtruth_preference"] = _gt["groundtruth_preference"]
                inf["groundtruth_preference"] = _gt["groundtruth_preference"]
            e["instance_full"] = json.dumps(inf) if isinstance(inf_raw, str) else inf
            n_ok += 1
        out = path if a.apply else path + ".hpregen"
        if a.apply:
            os.makedirs(f"{BACKUP_DIR}/{uid}", exist_ok=True)
            bp = f"{BACKUP_DIR}/{uid}/test.json"
            if not os.path.exists(bp):
                with open(path) as s, open(bp, "w") as d: d.write(s.read())
        json.dump(data, open(out, "w"))
        print(f"[{uid}] regenerated={n_ok} failed={n_fail} -> {out}{' (APPLIED)' if a.apply else ' (TEST)'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
