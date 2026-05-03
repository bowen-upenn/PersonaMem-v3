"""Audit example_response / inferior_response pairs in a test set.

Usage:
    python evaluation/audit_example_inferior_pairs.py [--user-id 115]
                                                       [--path PATH]
                                                       [--examples-per-bucket 3]

Reports:
  - total samples
  - samples with example+inferior pair
  - pair counts in each failure bucket (prefix_overlap, jaccard>0.85,
    jaccard<0.20, length_ratio>0.5)
  - per-task_type breakdown
  - concrete pair samples per failure bucket

The script reuses `_validate_inferior` from llm_postprocess.py so the
audit's failure rules and the generation-time validator stay in sync.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evaluation.llm_postprocess import _validate_inferior, _token_jaccard


def _load_test_json(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of test samples")
    return data


def _classify_pair(example: str, inferior: str) -> tuple[bool, str, float]:
    """Returns (passed, failure_reason, jaccard).
    failure_reason is empty when passed."""
    ok, reason = _validate_inferior(example, inferior)
    j = _token_jaccard(example, inferior)
    return ok, reason, j


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", default="115",
                        help="user_id whose test.json to audit (default: 115)")
    parser.add_argument("--path",
                        help="explicit path to a test.json (overrides --user-id)")
    parser.add_argument("--examples-per-bucket", type=int, default=3,
                        help="how many concrete failing pairs to print per bucket")
    parser.add_argument("--repo-root", default=".",
                        help="repo root (default: cwd)")
    args = parser.parse_args()

    if args.path:
        path = Path(args.path)
    else:
        path = Path(args.repo_root) / "backend" / args.user_id / "test.json"
    if not path.exists():
        raise SystemExit(f"test.json not found at {path}")

    samples = _load_test_json(path)

    n_total = len(samples)
    n_pairs = 0
    n_passed = 0
    bucket_counts: dict[str, int] = defaultdict(int)
    bucket_examples: dict[str, list[dict]] = defaultdict(list)
    per_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for inst in samples:
        ex = inst.get("example_response") or ""
        inf_obj = inst.get("inferior_response") or {}
        inf = inf_obj.get("text") if isinstance(inf_obj, dict) else inf_obj
        inf = inf or ""
        if not ex or not inf:
            continue
        n_pairs += 1
        tt = inst.get("task_type") or "unknown"
        per_type[tt]["pairs"] += 1
        ok, reason, j = _classify_pair(ex, inf)
        if ok:
            n_passed += 1
            per_type[tt]["passed"] += 1
            continue
        # Map reason text to a short bucket key.
        if reason.startswith("identical"):
            bucket = "identical"
        elif reason.startswith("prefix_overlap"):
            bucket = "prefix_overlap"
        elif reason.startswith("substring_containment"):
            bucket = "substring_containment"
        elif reason.startswith("opening_overlap"):
            bucket = "opening_overlap"
        elif reason.startswith("too_similar"):
            bucket = "too_similar"
        elif reason.startswith("too_dissimilar"):
            bucket = "too_dissimilar"
        elif reason.startswith("length_mismatch"):
            bucket = "length_mismatch"
        elif reason == "empty_text":
            bucket = "empty_text"
        else:
            bucket = "other"
        bucket_counts[bucket] += 1
        per_type[tt][bucket] += 1
        if len(bucket_examples[bucket]) < args.examples_per_bucket:
            bucket_examples[bucket].append({
                "task_type": tt,
                "query_id": inst.get("query_id", ""),
                "example": ex[:200],
                "inferior": inf[:200],
                "jaccard": round(j, 2),
            })

    print(f"Test set: {path}")
    print(f"Total samples: {n_total}")
    print(f"Samples with example+inferior pair: {n_pairs}")
    print(f"  passed validator: {n_passed}")
    print(f"  failed validator: {n_pairs - n_passed}")
    print()
    print("Failure buckets:")
    for bucket in ("prefix_overlap", "substring_containment", "opening_overlap",
                   "too_similar", "too_dissimilar",
                   "length_mismatch", "identical", "empty_text", "other"):
        c = bucket_counts.get(bucket, 0)
        if c:
            print(f"  {bucket:<24} {c:>4}")
    print()

    rows = sorted(per_type.items(),
                  key=lambda kv: -(kv[1]["pairs"] - kv[1].get("passed", 0)))
    header = (f"{'task_type':<40}{'pairs':>6}{'pass':>6}"
              f"{'prefix':>8}{'subStr':>7}{'open':>6}"
              f"{'highJ':>7}{'lowJ':>6}{'lenSk':>7}{'ident':>6}")
    print(header)
    for tt, s in rows:
        print(
            f"{tt[:39]:<40}"
            f"{s.get('pairs', 0):>6}"
            f"{s.get('passed', 0):>6}"
            f"{s.get('prefix_overlap', 0):>8}"
            f"{s.get('substring_containment', 0):>7}"
            f"{s.get('opening_overlap', 0):>6}"
            f"{s.get('too_similar', 0):>7}"
            f"{s.get('too_dissimilar', 0):>6}"
            f"{s.get('length_mismatch', 0):>7}"
            f"{s.get('identical', 0):>6}"
        )
    print()

    print("Concrete failing examples:")
    for bucket, ex_list in bucket_examples.items():
        if not ex_list:
            continue
        print(f"\n  [{bucket}] ({bucket_counts[bucket]} total)")
        for e in ex_list:
            print(f"    task={e['task_type']} query_id={e['query_id']} jaccard={e['jaccard']}")
            print(f"      EX: {e['example']!r}")
            print(f"      IN: {e['inferior']!r}")


if __name__ == "__main__":
    main()
