#!/usr/bin/env python3
"""Dump every test query for a user into backend/{uid}/test.json.

Usage:
  python scripts/dump_test_json.py --user_id 115
  python scripts/dump_test_json.py --user_id 115 --output /tmp/test.json

Reads benchmark/{uid}/queries.csv and emits a single JSON list with one
record per test query. See the plan file for the record schema.
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the project root importable when running this file directly.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_preparation.visualize import dump_test_samples_json


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--user_id", required=True, help="user id, e.g. 115")
    p.add_argument("--benchmark_dir", default="benchmark")
    p.add_argument("--backend_dir", default="backend")
    p.add_argument(
        "--output",
        default=None,
        help="output path; default backend/{uid}/test.json",
    )
    args = p.parse_args()

    out = dump_test_samples_json(
        args.user_id,
        output_path=args.output,
        benchmark_dir=args.benchmark_dir,
        backend_dir=args.backend_dir,
    )
    size = os.path.getsize(out)
    print(f"wrote {out} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
