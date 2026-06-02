"""Calendar mock MCP server — read-only, time-masked schedule view.

The calendar is a CRUD modification stream (`added` / `updated` / `removed`)
on the user's entries (`backend/{uid}/calendar.json`). This server folds the
modifications with `timestamp < t_test` and exposes the resulting current
schedule as a single read tool, so the `mcp_agent` mode sees the same calendar
state that `llm_longctx` / the memory modes (folded block) and `agent_tools`
(snapshot `calendar.json`) see. Read-only: the agent never mutates the calendar.
"""

from __future__ import annotations

import argparse
import os
import sys

from fastmcp import FastMCP

from evaluation.backend_query import BackendQuery


def _build_env() -> dict:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--user_id", default=os.environ.get("PM3_USER_ID"))
    parser.add_argument("--t_test", type=int, default=int(os.environ.get("PM3_T_TEST", "0")))
    parser.add_argument("--backend_dir", default=os.environ.get("PM3_BACKEND_DIR", "backend"))
    args, _ = parser.parse_known_args()
    if not (args.user_id and args.t_test):
        print(f"[calendar_mcp] missing config: {vars(args)}", file=sys.stderr)
        sys.exit(2)
    return vars(args)


def make_server() -> FastMCP:
    cfg = _build_env()
    user_id = cfg["user_id"]
    t_test = cfg["t_test"]
    bq = BackendQuery(cfg["backend_dir"])

    mcp = FastMCP("pm3-calendar", version="2.0.0")

    @mcp.tool()
    def get_calendar() -> dict:
        """The user's CURRENT calendar entries as of now (time-masked: only
        modifications before the present moment are folded in). Returns a list
        of entries with title, start/end, location, and attendees."""
        try:
            state = bq.get_calendar_state(user_id, t_test)
        except Exception as exc:  # never hard-fail the agent on a read
            return {"results": [], "error": str(exc)}
        entries = []
        for _eid, entry in sorted((state or {}).items(),
                                  key=lambda kv: (kv[1] or {}).get("start_ts", 0)):
            if not isinstance(entry, dict):
                continue
            loc = entry.get("location") or {}
            entries.append({
                "title": entry.get("title", ""),
                "start": entry.get("formatted_timestamp") or entry.get("start_ts"),
                "start_ts": entry.get("start_ts"),
                "end_ts": entry.get("end_ts"),
                "location": loc.get("city", "") if isinstance(loc, dict) else loc,
                "attendees": entry.get("attendees"),
                "notes": entry.get("notes") or entry.get("description"),
            })
        return {"results": entries}

    return mcp


if __name__ == "__main__":
    make_server().run()
