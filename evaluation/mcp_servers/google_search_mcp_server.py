"""Google Search MCP server (opt-in, E4 only).

Exposes ONE tool: `search_google(query, num_results)`. Defaults to
cache-first replay — live Google Custom Search JSON API calls require
`PM3_E4_ALLOW_LIVE=1`. Without that flag, cache misses return a
`cache_miss_live_disabled` result and the instance is marked `skipped`.

The cache directory is `benchmark/{user_id}/google_search_cache/` with
files `{day}_{sha1(query)}.json`. Cross-day cache keys are NOT shared —
freshness matters for a news-ranking eval, so the day is part of the key.

Env vars:
- PM3_USER_ID        — required
- PM3_T_TEST         — required (used only for the cache-key day_label)
- PM3_BACKEND_DIR    — default "backend" (unused here; kept for uniformity)
- PM3_GOOGLE_CACHE_DIR — default "benchmark/{user_id}/google_search_cache"
- PM3_E4_ALLOW_LIVE  — "1" to enable live API calls on cache miss
- GOOGLE_API_KEY, GOOGLE_CSE_ID — required only for live calls
- PM3_E4_QUOTA_PER_DAY — default 20 (per-user daily live-call cap)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlencode

try:
    from fastmcp import FastMCP
except ImportError:
    FastMCP = None  # type: ignore


_DEFAULT_QUOTA: int = 20


def _build_env() -> dict:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--user_id", default=os.environ.get("PM3_USER_ID"))
    parser.add_argument("--t_test", type=int, default=int(os.environ.get("PM3_T_TEST", "0")))
    parser.add_argument("--backend_dir", default=os.environ.get("PM3_BACKEND_DIR", "backend"))
    parser.add_argument("--cache_dir", default=os.environ.get("PM3_GOOGLE_CACHE_DIR"))
    args, _ = parser.parse_known_args()
    if not args.user_id:
        print("[google_search_mcp] PM3_USER_ID is required", file=sys.stderr)
        sys.exit(2)
    if not args.cache_dir:
        args.cache_dir = f"benchmark/{args.user_id}/google_search_cache"
    return vars(args)


def _day_label_from_ts(ts: int) -> str:
    if ts and ts > 0:
        return _dt.datetime.fromtimestamp(int(ts), tz=_dt.timezone.utc).strftime("%Y-%m-%d")
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%d")


def _cache_filename(cache_dir: Path, day_label: str, query: str, num_results: int) -> Path:
    key = hashlib.sha1(f"{query}|{num_results}".encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{day_label}_{key}.json"


def _quota_file(cache_dir: Path, day_label: str) -> Path:
    return cache_dir / f".quota_{day_label}"


def _read_quota(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(path.read_text().strip() or "0")
    except (ValueError, OSError):
        return 0


def _bump_quota(path: Path) -> int:
    cur = _read_quota(path)
    path.write_text(str(cur + 1))
    return cur + 1


def _live_call(query: str, num_results: int) -> dict | None:
    """Call Google Custom Search JSON API. Returns None on any failure."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    cse_id = os.environ.get("GOOGLE_CSE_ID")
    if not (api_key and cse_id):
        return None
    import urllib.request
    import urllib.error
    num = max(1, min(10, int(num_results)))  # API caps num at 10 per page
    params = {"key": api_key, "cx": cse_id, "q": query, "num": num}
    url = "https://www.googleapis.com/customsearch/v1?" + urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError, TimeoutError, OSError):
        return None
    items = data.get("items") or []
    return {
        "query": query,
        "num_results": num,
        "results": [
            {
                "title": it.get("title", ""),
                "link": it.get("link", ""),
                "snippet": it.get("snippet", ""),
                "displayLink": it.get("displayLink", ""),
            }
            for it in items[:num]
        ],
    }


def make_server():
    if FastMCP is None:
        print("[google_search_mcp] fastmcp not installed; cannot start", file=sys.stderr)
        sys.exit(2)
    cfg = _build_env()
    user_id = cfg["user_id"]
    t_test = int(cfg["t_test"] or 0)
    day_label = _day_label_from_ts(t_test)
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    allow_live = os.environ.get("PM3_E4_ALLOW_LIVE") == "1"
    quota_cap = int(os.environ.get("PM3_E4_QUOTA_PER_DAY", _DEFAULT_QUOTA))

    mcp = FastMCP("pm3-google-search", version="1.0.0")

    @mcp.tool()
    def search_google(query: str, num_results: int = 10) -> dict:
        """Personalized Google Custom Search.

        Default: cache-first replay (benchmark/{user_id}/google_search_cache).
        Live calls require PM3_E4_ALLOW_LIVE=1 + GOOGLE_API_KEY + GOOGLE_CSE_ID.
        Per-user daily quota (default 20) applies to live calls only.
        """
        if not query or not isinstance(query, str):
            return {"error": "empty_query"}
        cache_path = _cache_filename(cache_dir, day_label, query, num_results)
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except (ValueError, OSError):
                pass  # fall through to live call path
        if not allow_live:
            return {
                "error": "cache_miss_live_disabled",
                "query": query,
                "day_label": day_label,
                "hint": "pass --e4_allow_live at run time to enable live API calls",
            }
        # Quota gate
        qfile = _quota_file(cache_dir, day_label)
        if _read_quota(qfile) >= quota_cap:
            return {"error": "quota_exceeded", "day_label": day_label, "cap": quota_cap}
        data = _live_call(query, num_results)
        if data is None:
            return {"error": "api_error_or_no_credentials", "query": query}
        _bump_quota(qfile)
        # Persist cache
        try:
            cache_path.write_text(json.dumps(data, indent=2))
        except OSError:
            pass
        return data

    return mcp


if __name__ == "__main__":
    make_server().run()
