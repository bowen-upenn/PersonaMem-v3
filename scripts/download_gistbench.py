#!/usr/bin/env python3
"""
Download facebook/gistbench, inspect schema vs our expected CSV shape, and
convert it into the persona-pipeline input CSV (ALL users by default;
--sample N builds a small subset for a quick smoke run).

Uses ONLY pyarrow — numpy/pandas/datasets/huggingface_hub are intentionally
not required (PEP 668 blocks `pip install --user` in this environment).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


# Paths / constants
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PARQUET_DIR = os.path.join(REPO_ROOT, "data", "gistbench")
PARQUET_PATH = os.path.join(PARQUET_DIR, "uu_dataset.parquet")
PARQUET_URL = "https://huggingface.co/datasets/facebook/gistbench/resolve/main/uu_dataset.parquet"

# gistbench columns (after dropping dataset/ds which don't exist upstream)
EXPECTED_COLUMNS = [
    "interaction_type",
    "user_id",
    "object_id",
    "interaction_time",
    "object_text",
]

RANDOM_SEED = 42


def log(msg: str) -> None:
    print(msg, flush=True)


def download_parquet() -> None:
    """Download the parquet file if it doesn't already exist."""
    if os.path.exists(PARQUET_PATH):
        size_mb = os.path.getsize(PARQUET_PATH) / (1024 * 1024)
        log(f"[download] Using existing {PARQUET_PATH} ({size_mb:.1f} MB)")
        return

    os.makedirs(PARQUET_DIR, exist_ok=True)
    log(f"[download] Fetching {PARQUET_URL} -> {PARQUET_PATH}")
    result = subprocess.run(
        ["curl", "-L", "--fail", "-o", PARQUET_PATH, PARQUET_URL],
        check=False,
    )
    if result.returncode != 0:
        log(f"[download] FAILED with code {result.returncode}")
        sys.exit(1)
    size_mb = os.path.getsize(PARQUET_PATH) / (1024 * 1024)
    log(f"[download] Downloaded {size_mb:.1f} MB")


def inspect_schema() -> pa.Schema:
    """Print the parquet schema and a schema diff vs our expected CSV shape."""
    pf = pq.ParquetFile(PARQUET_PATH)
    schema = pf.schema_arrow
    total_rows = pf.metadata.num_rows

    log("\n[schema] Arrow schema of uu_dataset.parquet:")
    for field in schema:
        log(f"  - {field.name}: {field.type}")
    log(f"[schema] Total rows: {total_rows:,}")
    log(f"[schema] Row groups: {pf.num_row_groups}")

    actual_cols = {f.name for f in schema}
    expected_cols = set(EXPECTED_COLUMNS)

    log("\n[schema diff] vs our expected CSV shape:")
    for col in EXPECTED_COLUMNS:
        mark = "PRESENT" if col in actual_cols else "MISSING"
        arrow_type = str(schema.field(col).type) if col in actual_cols else "-"
        log(f"  {mark:<8} {col:<20} {arrow_type}")
    for col in actual_cols - expected_cols:
        log(f"  EXTRA    {col:<20} {schema.field(col).type} (not in our CSV)")

    # Peek at first 3 rows to see what real data looks like (esp. interaction_time format)
    log("\n[schema] First 3 rows (raw):")
    first_batch = next(pf.iter_batches(batch_size=3))
    first_df = first_batch.to_pydict()
    for i in range(min(3, len(first_df[EXPECTED_COLUMNS[0]]))):
        row = {c: first_df[c][i] for c in first_df}
        log(f"  row[{i}]: {row}")

    return schema


def load_user_ids() -> list[int]:
    """Load just the user_id column from the parquet and return unique sorted values."""
    log("\n[sample] Reading user_id column to collect unique users...")
    uid_table = pq.read_table(PARQUET_PATH, columns=["user_id"])
    uid_col = uid_table["user_id"].to_pylist()
    uids = sorted(set(uid_col))
    log(f"[sample] Found {len(uids)} unique users (range {uids[0]} - {uids[-1]})")
    return uids


def sample_users(uids: list[int], n: int) -> list[int]:
    if n > len(uids):
        log(f"[sample] Requested {n} users but only {len(uids)} available — using all.")
        n = len(uids)
    random.seed(RANDOM_SEED)
    selected = random.sample(uids, n)
    selected.sort()
    log(f"[sample] Seed {RANDOM_SEED} -> selected {n} users: {selected}")
    return selected


def extract_rows(selected: list[int], schema: pa.Schema) -> list[dict]:
    """Load all columns and filter down to rows where user_id is in the selected set."""
    log("[sample] Reading full parquet for filtering...")
    table = pq.read_table(PARQUET_PATH, columns=EXPECTED_COLUMNS)
    log(f"[sample] Loaded {table.num_rows:,} rows, filtering...")

    mask = pc.is_in(table["user_id"], value_set=pa.array(selected))
    filtered = table.filter(mask)
    log(f"[sample] After filter: {filtered.num_rows:,} rows for {len(selected)} users")

    # Determine how to handle interaction_time: if string, parse to unix int; if int, pass through
    it_type = schema.field("interaction_time").type
    it_is_string = pa.types.is_string(it_type) or pa.types.is_large_string(it_type)

    rows: list[dict] = []
    data = filtered.to_pydict()
    n = filtered.num_rows
    per_user_count: dict[int, int] = {}

    for i in range(n):
        raw_it = data["interaction_time"][i]
        if it_is_string:
            # Probe parsing on first row
            t = _parse_time_string(raw_it)
        else:
            t = int(raw_it) if raw_it is not None else 0

        uid = int(data["user_id"][i])
        per_user_count[uid] = per_user_count.get(uid, 0) + 1

        rows.append({
            "interaction_type": data["interaction_type"][i] or "",
            "user_id": uid,
            "object_id": int(data["object_id"][i]) if data["object_id"][i] is not None else 0,
            "interaction_time": t,
            "object_text": data["object_text"][i] or "",
        })

    log("[sample] Per-user row counts:")
    for uid in selected:
        log(f"  user {uid}: {per_user_count.get(uid, 0):,} rows")
    log(f"[sample] Total rows extracted: {len(rows):,}")

    return rows, per_user_count


def _parse_time_string(s: str) -> int:
    """Parse a date string from gistbench into a Unix timestamp (UTC).

    Tries several plausible formats; falls back to 0 on failure.
    """
    if s is None:
        return 0
    s = str(s).strip()
    if not s:
        return 0
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    # Might already be a numeric string
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def write_csv(rows: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # Sort globally by (user_id, interaction_time) for determinism
    rows.sort(key=lambda r: (r["user_id"], r["interaction_time"]))
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log(f"\n[write] Wrote {len(rows):,} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download facebook/gistbench and convert it to the persona-pipeline input CSV."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional: randomly sample only N users (smoke runs). Default: convert ALL users.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: data/gistbench_input.csv "
             "(or data/gistbench_sample_<N>users.csv with --sample).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    download_parquet()
    schema = inspect_schema()
    uids = load_user_ids()
    if args.sample is None:
        selected = uids
        log(f"[full] Converting all {len(uids)} users.")
        default_name = "gistbench_input.csv"
    else:
        selected = sample_users(uids, args.sample)
        default_name = f"gistbench_sample_{args.sample}users.csv"
    output_csv = args.output or os.path.join(REPO_ROOT, "data", default_name)
    rows, _per_user = extract_rows(selected, schema)
    write_csv(rows, output_csv)
    log("[done]")


if __name__ == "__main__":
    main()
