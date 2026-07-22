"""Per-gate drop trace for build_e5_horizon_lifecycle across matched-10."""
import sys
sys.path.insert(0, ".")
from evaluation.backend_query import BackendQuery
from evaluation.tasks import e5_horizon_lifecycle as E5
from evaluation.tasks.e5_horizon_lifecycle import (
    _collect_short_term_canonicals, _gather_candidate_events, _jaccard, _project_candidate,
    E5_POOL_TARGET, E5_MIN_MATCHING_CANDIDATES,
)
import random as _random

MATCHED = [1, 2, 3, 5, 6, 8, 9, 10, 13, 14]
bq = BackendQuery("backend")

agg = {"canon": 0, "stop_le_first": 0, "t_active_ge_stop": 0, "no_dir_tags": 0,
       "pre_none": 0, "post_none": 0, "both_ok": 0}

for uid in MATCHED:
    u = str(uid)
    cans = _collect_short_term_canonicals(bq, u)
    max_ts = 0
    for app in ("instagram", "facebook", "threads", "chatbot"):
        for e in bq._load_events(u, app):
            max_ts = max(max_ts, e.get("source_timestamp") or 0)
    per = {k: 0 for k in agg}
    per["canon"] = len(cans)
    for rec in cans:
        row_tss = rec["row_timestamps"]; first_ts, last_ts = row_tss[0], row_tss[-1]
        sc = rec["stop_condition"] or {}
        stop = int(sc.get("expected_stop_ts") or 0)
        if stop <= first_ts:
            per["stop_le_first"] += 1; continue
        t_active = int(first_ts + 0.6 * (last_ts - first_ts))
        if t_active >= stop:
            per["t_active_ge_stop"] += 1; continue
        t_post = max(max_ts + 3600, stop + 2*3600)
        dir_tags = rec["row_hashtags"]
        if not dir_tags:
            per["no_dir_tags"] += 1; continue
        pre_events = _gather_candidate_events(bq, u, window_start=max(0, t_active-48*3600), window_end=t_active+24*3600)
        post_events = _gather_candidate_events(bq, u, window_start=max(0, t_post-48*3600), window_end=min(max_ts, t_post+24*3600))
        if len(post_events) < 3:
            post_events = pre_events
        rng = _random.Random(f"0:e5:{rec['persona_item'][:40]}")
        def matchcount(events_list):
            s = events_list[:]; rng.shuffle(s); s = s[:E5_POOL_TARGET]
            return sum(1 for e in s if _jaccard(e.get("source_hashtags") or [], dir_tags) >= 0.3)
        pre_m = matchcount(pre_events); post_m = matchcount(post_events)
        pre_ok = pre_m >= E5_MIN_MATCHING_CANDIDATES
        post_ok = post_m >= E5_MIN_MATCHING_CANDIDATES
        if not pre_ok: per["pre_none"] += 1
        if not post_ok: per["post_none"] += 1
        if pre_ok and post_ok: per["both_ok"] += 1
    print(f"persona {u:>3}: canon={per['canon']} stop<=first={per['stop_le_first']} "
          f"t_active>=stop={per['t_active_ge_stop']} no_tags={per['no_dir_tags']} "
          f"pre_fail={per['pre_none']} post_fail={per['post_none']} BOTH_OK={per['both_ok']}")
    for k in agg: agg[k] += per[k]

print("\nTOTALS:", agg)
