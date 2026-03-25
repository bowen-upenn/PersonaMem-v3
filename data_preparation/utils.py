from __future__ import annotations

import os
import re
import csv
import json
from typing import List, Dict, Optional, Union
from datetime import datetime, timezone


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def extract_json_from_response(response_text: str):
    """Extract JSON from an LLM response that may contain markdown fences or extra text."""
    if not response_text:
        return None

    # Try to find JSON inside ```json ... ``` fences
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', response_text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
    else:
        candidate = response_text.strip()

    # Try direct parse first
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fallback: try json_repair
    try:
        import json_repair
        return json_repair.loads(candidate)
    except Exception:
        pass

    # Last resort: find first [ ... ] or { ... } block
    for open_char, close_char in [('[', ']'), ('{', '}')]:
        start = candidate.find(open_char)
        end = candidate.rfind(close_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(candidate[start:end + 1])
            except json.JSONDecodeError:
                continue

    return None


def unix_to_formatted(unix_ts: int) -> str:
    """Convert a Unix timestamp to 'HH:MM, MM/DD/YYYY' format in UTC."""
    dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
    return dt.strftime("%H:%M, %m/%d/%Y")


def save_rows_to_csv(rows: list[dict], filepath: str) -> None:
    """Save a list of dicts to a CSV file."""
    if not rows:
        return
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rows_from_csv(filepath: str) -> list[dict]:
    """Load a CSV file into a list of dicts."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)
