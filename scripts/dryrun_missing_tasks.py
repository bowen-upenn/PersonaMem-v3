"""Deterministic dry-run of the 3 zero-emit task builders over the selected
personas. Reports where each builder falls to zero WITHOUT any LLM calls
(the gold-content/answerability LLM steps are stubbed/skipped) so we can tell
data-sparsity from a builder bug."""
import os
import sys, traceback
sys.path.insert(0, ".")
from evaluation.backend_query import BackendQuery
from evaluation.build_benchmark import (
    _c1e_post_fatigue_anchors, _c1e_chatbot_ask_anchors, _c1e_at_ai_directive_anchors,
    _c1e_pick_flavor_b_event,
)
from evaluation.tasks.e5_horizon_lifecycle import (
    _collect_short_term_canonicals, build_e5_horizon_lifecycle,
)

MATCHED = [int(u) for u in os.environ.get("PERSONAS", "").split()] or \
          sorted(int(d) for d in os.listdir("backend")
                 if d.isdigit() and os.path.exists(f"backend/{d}/test.json"))
BASE = "backend"

def load_test_items(bq, uid):
    # build_c1e needs test_items only for the max-timestamp anchor + fatigue.
    # Re-derive a light proxy: every social event becomes a TestItem-like obj.
    from evaluation.build_benchmark import TestItem
    items = []
    for app in ("instagram", "facebook", "threads", "chatbot"):
        for e in bq._load_events(uid, app):
            ts = e.get("source_timestamp") or 0
            if ts:
                items.append(TestItem(source_timestamp=ts) if _ti_ok() else None)
    return [i for i in items if i]

def _ti_ok():
    return True

for uid in MATCHED:
    u = str(uid)
    print(f"\n===== persona {u} =====")
    try:
        bq = BackendQuery(BASE)
    except Exception as exc:
        print(f"  BackendQuery init failed: {exc}")
        continue

    # ---- E5 short-term lifecycle ----
    try:
        cans = _collect_short_term_canonicals(bq, u)
        print(f"  [e5] short_term canonicals w/ stop_condition: {len(cans)}")
        if cans:
            for c in cans[:5]:
                sc = c["stop_condition"] or {}
                print(f"        - {c['persona_item'][:50]!r} rows={len(c['row_timestamps'])} "
                      f"stop_ts={sc.get('expected_stop_ts')} first={c['row_timestamps'][0]}")
        insts = build_e5_horizon_lifecycle(bq, u, rng_seed=0)
        print(f"  [e5] EMITTED instances: {len(insts)}")
    except Exception:
        print("  [e5] EXCEPTION:"); traceback.print_exc()

    # ---- C1E new_suggestions anchors ----
    try:
        # test_items proxy
        from evaluation.build_benchmark import TestItem
        ti = []
        for app in ("instagram", "facebook", "threads", "chatbot"):
            for e in bq._load_events(u, app):
                ts = e.get("source_timestamp") or 0
                if ts:
                    try: ti.append(TestItem(source_timestamp=ts))
                    except Exception: pass
        import random as _r
        rng = _r.Random(f"0:c1e:{u}")
        fa = _c1e_post_fatigue_anchors(bq, u, ti, n_anchors=3)
        ca = _c1e_chatbot_ask_anchors(bq, u, n_anchors=3, rng=rng)
        aa = _c1e_at_ai_directive_anchors(bq, u, n_anchors=3)
        print(f"  [c1e] anchors: fatigue={len(fa)} chatbot_ask={len(ca)} at_ai={len(aa)}")
        # how many anchors have a flavor-B gold event (deterministic)?
        nb = 0
        for anc in (fa + ca + aa):
            try:
                if _c1e_pick_flavor_b_event(bq, u, int(anc["t_test"])) is not None:
                    nb += 1
            except Exception:
                pass
        print(f"  [c1e] anchors with a flavor-B future-truth gold: {nb}/{len(fa)+len(ca)+len(aa)}")
    except Exception:
        print("  [c1e] EXCEPTION:"); traceback.print_exc()
