"""Cross-persona eval aggregator.

Scans `benchmark/*/runs/*/results.csv`, picks the latest run per persona
(unless `--run=all`), and emits:

  eval_aggregate/summary_by_task.csv       — per-task mean metric per task_type
  eval_aggregate/summary_by_persona.csv    — per-persona, per-task metric summary
  eval_aggregate/summary_overall.json      — grand totals + E6 paired-F1

E6 paired-F1 is computed here because it needs both polarities of each
pair — the per-persona summary can't compute it from per-instance rows
alone, but the aggregator can join warn + foil rows by `pair_id`.

Usage:
  python scripts/aggregate_eval.py                      # latest run per persona
  python scripts/aggregate_eval.py --run=all            # every run ever
  python scripts/aggregate_eval.py --out path/to/dir    # custom output dir
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# results.csv cells can be large (agent responses); match the runner limit.
csv.field_size_limit(10_000_000)


def _pick_runs(mode: str) -> list[Path]:
    runs = sorted(REPO_ROOT.glob("benchmark/*/runs/*/results.csv"))
    if not runs:
        return []
    if mode == "all":
        return runs

    # "latest": keep the newest run_dir per persona (name is timestamp-sortable)
    latest: dict[str, Path] = {}
    for r in runs:
        uid = r.parent.parent.parent.name  # benchmark/{uid}/runs/{ts}/results.csv
        ts = r.parent.name
        cur = latest.get(uid)
        if cur is None or cur.parent.name < ts:
            latest[uid] = r
    return list(latest.values())


def _load_rows(results_csv: Path) -> list[dict]:
    rows: list[dict] = []
    with results_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            mj = r.get("metrics_json") or ""
            try:
                metrics = json.loads(mj) if mj else {}
            except Exception:
                metrics = {}
            r["_metrics"] = metrics if isinstance(metrics, dict) else {}
            rows.append(r)
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate_numeric(rows: list[dict]) -> dict[str, float]:
    """Mean of every numeric metric key across the given rows."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        for k, v in (r.get("_metrics") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                sums[k] += float(v)
                counts[k] += 1
    return {k: sums[k] / counts[k] for k in sums}


def _e6_paired_f1(rows: list[dict]) -> dict[str, float]:
    """Pair warn+foil by pair_id (encoded into instance_id as `<pair>_warn|foil`).

    Per paired eval convention:
      - Paired-correct: both polarities correct on the same pair.
      - Warn-recall: warn-polarity accuracy.
      - Foil-precision: foil-polarity accuracy (true-silent rate).
      - Macro-F1: harmonic mean of the two.
    """
    e6 = [r for r in rows if r.get("task_type") == "active_mistake_prevention"]
    if not e6:
        return {}
    pairs: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in e6:
        qid = r.get("query_id") or ""
        inst_id = qid.split(":", 2)[-1] if ":" in qid else qid
        if inst_id.endswith("_warn"):
            pair_id, polarity = inst_id[: -len("_warn")], "warn"
        elif inst_id.endswith("_foil"):
            pair_id, polarity = inst_id[: -len("_foil")], "foil"
        else:
            continue
        pairs[pair_id][polarity] = r

    warn_correct: list[float] = []
    foil_correct: list[float] = []
    paired_correct: list[float] = []
    for pair in pairs.values():
        w = pair.get("warn")
        fo = pair.get("foil")
        wc = float((w or {}).get("_metrics", {}).get("correct_warn", 0)) if w else None
        fc = float((fo or {}).get("_metrics", {}).get("correct_foil", 0)) if fo else None
        if wc is not None:
            warn_correct.append(wc)
        if fc is not None:
            foil_correct.append(fc)
        if wc is not None and fc is not None:
            paired_correct.append(1.0 if (wc >= 1 and fc >= 1) else 0.0)

    warn_rate = _mean(warn_correct)
    foil_rate = _mean(foil_correct)
    denom = (warn_rate + foil_rate)
    macro_f1 = (2 * warn_rate * foil_rate / denom) if denom > 0 else 0.0
    return {
        "n_pairs": len(pairs),
        "warn_recall": round(warn_rate, 4),
        "foil_precision": round(foil_rate, 4),
        "paired_correct": round(_mean(paired_correct), 4),
        "macro_f1": round(macro_f1, 4),
    }


# ---------------------------------------------------------------------------
# Phase C: token-vs-accuracy table
# ---------------------------------------------------------------------------

def _accuracy_value(task_type: str, metrics: dict, status: str, e6_paired: dict | None = None) -> float | None:
    """Compute the headline accuracy_pct (0–100) for one row.

    Returns None when no PRIMARY_METRIC is registered for the task_type
    OR the metric value is missing — caller skips those rows.
    """
    from evaluation.task_registry import PRIMARY_METRIC, normalize_task_type
    spec = PRIMARY_METRIC.get(normalize_task_type(task_type))
    if not spec:
        return None
    key, kind = spec

    # Status gate: rows that flagged failed_writes / failed_quality count as 0.
    if status in ("failed_writes", "failed_quality", "error", "no_result"):
        return 0.0

    # Removal-regen rows where the model never personalized in turn 1 are
    # excluded from the denominator entirely — there's nothing to remove,
    # so a 0 here would slander the model. The build-time filter in
    # `build_c4_instances` should keep these rare; this is defence-in-depth.
    if metrics.get("removal_status") == "skipped_low_personalization":
        return None

    if kind == "agentic_pass_rate":
        passed = (
            (metrics.get("tool_call_rules_pass") or 0)
            + (metrics.get("final_state_rules_passed") or 0)
            + (metrics.get("output_quality_passed") or 0)
        )
        failed = (
            (metrics.get("tool_call_rules_fail") or 0)
            + (metrics.get("final_state_rules_failed") or 0)
            + (metrics.get("output_quality_failed") or 0)
        )
        denom = passed + failed
        return 100.0 * (passed / denom) if denom > 0 else 0.0

    if kind == "paired_correct":
        # For active_mistake_prevention rows we need the pair-level result;
        # the per-row metric "correct" is itself binary, so we approximate
        # as 100 * mean(correct). Pair-level macro-F1 is in summary_overall.
        v = metrics.get("correct")
        return 100.0 * float(v) if v is not None else None

    v = metrics.get(key)
    if v is None:
        return None
    if kind == "inverted_fraction":
        return 100.0 * (1.0 - float(v))
    return 100.0 * float(v)


def _build_token_accuracy_table(rows: list[dict], e6_paired: dict | None = None) -> list[dict]:
    """Group rows by task_type; produce one row of accuracy + token means.
    Append a final ALL row with n-weighted accuracy + token means.
    """
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r.get("task_type", "")].append(r)

    table: list[dict] = []
    weighted_sum_acc = 0.0
    weighted_n_acc = 0
    sum_in_tok = 0.0
    sum_out_tok = 0.0
    sum_cost = 0.0
    sum_dur = 0.0
    sum_n_tok = 0
    for task, task_rows in sorted(by_task.items()):
        accs: list[float] = []
        in_toks: list[float] = []
        out_toks: list[float] = []
        costs: list[float] = []
        durs: list[float] = []
        for r in task_rows:
            m = r.get("_metrics") or {}
            status = r.get("status", "ok")
            a = _accuracy_value(task, m, status, e6_paired)
            if a is not None:
                accs.append(a)
            try:
                in_toks.append(float(m.get("input_tokens") or 0))
                out_toks.append(float(m.get("output_tokens") or 0))
                costs.append(float(m.get("cost_usd") or 0))
            except Exception:
                pass
            try:
                durs.append(float(r.get("duration_ms") or 0))
            except Exception:
                pass
        n = len(task_rows)
        acc_mean = _mean(accs) if accs else None
        row = {
            "task_type": task,
            "n": n,
            "accuracy_pct": round(acc_mean, 2) if acc_mean is not None else "",
            "mean_input_tokens": round(_mean(in_toks), 1) if in_toks else 0,
            "mean_output_tokens": round(_mean(out_toks), 1) if out_toks else 0,
            "mean_cost_usd": round(_mean(costs), 4) if costs else 0,
            "mean_duration_ms": round(_mean(durs), 1) if durs else 0,
        }
        table.append(row)
        if accs:
            weighted_sum_acc += acc_mean * len(accs)
            weighted_n_acc += len(accs)
        if in_toks:
            sum_in_tok += sum(in_toks)
            sum_out_tok += sum(out_toks)
            sum_cost += sum(costs)
            sum_n_tok += len(in_toks)
        sum_dur += sum(durs)

    # ALL row (n-weighted, micro): every row contributes 1/N
    all_n = sum(r["n"] for r in table)
    all_row = {
        "task_type": "ALL (micro, row-weighted)",
        "n": all_n,
        "accuracy_pct": round(weighted_sum_acc / weighted_n_acc, 2) if weighted_n_acc else "",
        "mean_input_tokens": round(sum_in_tok / sum_n_tok, 1) if sum_n_tok else 0,
        "mean_output_tokens": round(sum_out_tok / sum_n_tok, 1) if sum_n_tok else 0,
        "mean_cost_usd": round(sum_cost / sum_n_tok, 4) if sum_n_tok else 0,
        "mean_duration_ms": round(sum_dur / max(1, all_n), 1),
    }
    table.append(all_row)

    # ALL row (macro, task-weighted): each task_type contributes one data
    # point in [0,1], averaged across task_types. This is the headline number.
    per_task_accs = [
        r["accuracy_pct"] for r in table
        if r["task_type"] not in ("ALL (micro, row-weighted)",)
        and isinstance(r.get("accuracy_pct"), (int, float))
    ]
    macro_row = {
        "task_type": "ALL (macro, task-weighted)",
        "n": len(per_task_accs),  # number of task_types contributing
        "accuracy_pct": round(sum(per_task_accs) / len(per_task_accs), 2) if per_task_accs else "",
        "mean_input_tokens": "",
        "mean_output_tokens": "",
        "mean_cost_usd": "",
        "mean_duration_ms": "",
    }
    table.append(macro_row)
    return table


def _print_token_accuracy_table(table: list[dict]) -> None:
    """Pretty-print to stdout — fixed-width columns, ALL row at the end."""
    cols = [
        ("task_type", 40),
        ("n", 5),
        ("accuracy_pct", 12),
        ("mean_input_tokens", 17),
        ("mean_output_tokens", 18),
        ("mean_cost_usd", 13),
        ("mean_duration_ms", 16),
    ]
    print()
    header = "  ".join(name.ljust(w) for name, w in cols)
    print(header)
    print("  ".join("-" * w for _, w in cols))
    for r in table:
        cells = []
        for name, w in cols:
            v = r.get(name, "")
            if isinstance(v, float):
                v = f"{v:.2f}" if name == "accuracy_pct" else str(v)
            cells.append(str(v).ljust(w))
        print("  ".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", choices=("latest", "all"), default="latest")
    ap.add_argument("--out", default="eval_aggregate")
    args = ap.parse_args()

    runs = _pick_runs(args.run)
    if not runs:
        print("[aggregate] no results.csv files found under benchmark/*/runs/",
              file=sys.stderr)
        return 2

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    per_persona: dict[str, list[dict]] = defaultdict(list)
    for rcsv in runs:
        uid = rcsv.parent.parent.parent.name
        rows = _load_rows(rcsv)
        for r in rows:
            r["_uid"] = uid
            r["_run_dir"] = str(rcsv.parent)
        all_rows.extend(rows)
        per_persona[uid].extend(rows)

    # --- by task ---
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_task[r.get("task_type", "")].append(r)

    by_task_rows: list[dict] = []
    metric_keys: set[str] = set()
    for task, task_rows in sorted(by_task.items()):
        agg = _aggregate_numeric(task_rows)
        metric_keys.update(agg.keys())
        by_task_rows.append({"task_type": task, "n": len(task_rows), **agg})

    metric_cols = sorted(metric_keys)
    with (out_dir / "summary_by_task.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_type", "n", *metric_cols])
        writer.writeheader()
        for row in by_task_rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})

    # --- by (persona, task) ---
    with (out_dir / "summary_by_persona.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "task_type", "n", *metric_cols])
        writer.writeheader()
        for uid in sorted(per_persona):
            by_task_uid: dict[str, list[dict]] = defaultdict(list)
            for r in per_persona[uid]:
                by_task_uid[r.get("task_type", "")].append(r)
            for task, task_rows in sorted(by_task_uid.items()):
                agg = _aggregate_numeric(task_rows)
                writer.writerow({
                    "user_id": uid, "task_type": task, "n": len(task_rows),
                    **{k: agg.get(k, "") for k in metric_cols},
                })

    # --- overall + E6 pair F1 ---
    overall = {
        "n_personas": len(per_persona),
        "n_runs": len(runs),
        "n_queries": len(all_rows),
        "by_task_summary_csv": str((out_dir / "summary_by_task.csv").relative_to(REPO_ROOT)),
        "by_persona_summary_csv": str((out_dir / "summary_by_persona.csv").relative_to(REPO_ROOT)),
        "e6_paired": _e6_paired_f1(all_rows),
    }
    (out_dir / "summary_overall.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print(f"[aggregate] {overall['n_queries']} queries across "
          f"{overall['n_personas']} persona(s), {overall['n_runs']} run(s)")
    print(f"[aggregate] wrote {out_dir}/summary_by_task.csv + summary_by_persona.csv + summary_overall.json")
    if overall["e6_paired"]:
        e6 = overall["e6_paired"]
        print(f"[aggregate] e6 paired: n={e6['n_pairs']}  "
              f"warn_recall={e6['warn_recall']:.3f}  "
              f"foil_precision={e6['foil_precision']:.3f}  "
              f"macro_f1={e6['macro_f1']:.3f}")

    # Phase C: token-vs-accuracy table — single artifact for the eval report.
    table = _build_token_accuracy_table(all_rows, overall["e6_paired"])
    # Lift the headline numbers into summary_overall.json so other tools can
    # read them without parsing the printed table.
    for r in table:
        if r["task_type"] == "ALL (macro, task-weighted)":
            overall["accuracy_pct_macro"] = r["accuracy_pct"]
            overall["n_task_types"] = r["n"]
        elif r["task_type"] == "ALL (micro, row-weighted)":
            overall["accuracy_pct_micro"] = r["accuracy_pct"]
    (out_dir / "summary_overall.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    table_path = out_dir / "token_accuracy_table.csv"
    cols = ["task_type", "n", "accuracy_pct",
            "mean_input_tokens", "mean_output_tokens", "mean_cost_usd", "mean_duration_ms"]
    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in table:
            writer.writerow({k: row.get(k, "") for k in cols})
    print(f"[aggregate] wrote {table_path.relative_to(REPO_ROOT)}")
    _print_token_accuracy_table(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
