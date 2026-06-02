"""AI Studio mock MCP server — read-only companion-chat conversation history.

AI Studio is a Meta-AI-Studio-style companion-chat surface: each event is a
conversation (no feed, no posts), shaped like the chatbot's. This server is
READ-ONLY (the eval never posts to AI Studio) and exposes get_history /
search_history so the `mcp_agent` mode sees the same companion-chat history
that the other modes see (it is one of the five apps in `backend_query.APPS`).
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
        print(f"[ai_studio_mcp] missing config: {vars(args)}", file=sys.stderr)
        sys.exit(2)
    return vars(args)


def make_server() -> FastMCP:
    cfg = _build_env()
    user_id = cfg["user_id"]
    t_test = cfg["t_test"]
    bq = BackendQuery(cfg["backend_dir"])

    mcp = FastMCP("pm3-ai_studio", version="2.0.0")

    @mcp.tool()
    def get_history(cursor: str | None = None, limit: int = 20) -> dict:
        """Recent AI Studio companion-chat conversations (time-masked). Each
        event carries the full conversation turn list."""
        events = bq.get_events(user_id=user_id, app="ai_studio", since_timestamp=t_test)
        events.sort(key=lambda x: x.get("source_timestamp", 0), reverse=True)
        page = [{
            "conversation_id": str(e.get("source_object_id", "")),
            "timestamp": e.get("source_timestamp"),
            "conversation_type": e.get("conversation_type"),
            "conversation": e.get("conversation", []),
            "interaction_format": e.get("interaction_format", {}),
        } for e in events]
        start = int(cursor) if (cursor and cursor.isdigit()) else 0
        window = page[start:start + limit]
        out = {"results": window}
        if start + limit < len(page):
            out["nextCursor"] = str(start + limit)
        return out

    @mcp.tool()
    def search_history(query: str, limit: int = 10) -> dict:
        """Search past AI Studio turns (substring over conversation content)."""
        q = (query or "").lower()
        hits = []
        for e in bq.get_events(user_id=user_id, app="ai_studio", since_timestamp=t_test):
            for m in (e.get("conversation") or []):
                if q in (m.get("content", "") or "").lower():
                    hits.append({
                        "conversation_id": str(e.get("source_object_id", "")),
                        "timestamp": e.get("source_timestamp"),
                        "role": m.get("role"),
                        "snippet": (m.get("content") or "")[:200],
                    })
                    if len(hits) >= limit:
                        return {"results": hits}
        return {"results": hits}

    return mcp


if __name__ == "__main__":
    make_server().run()
