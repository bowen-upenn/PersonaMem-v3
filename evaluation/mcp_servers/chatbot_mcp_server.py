"""Chatbot mock MCP server — assistant-side conversation + cross-app dispatch.

Unlike the social-app servers, the chatbot has no "feed" — it has a
conversation history. The chatbot exposes:
- `get_history` / `search_history` (read past turns)
- `send_message` (not typically called by the agent being evaluated —
  this is the human-user tool; included for completeness of the simulation)
- `send_post_to_app` — cross-app dispatch; lets the chatbot request a post
  be made on Instagram/Facebook/Threads. Records to the overlay tagged
  with the target app so the grader can verify correct routing.
- `summarize_inbox` — raw-DMs helper for Task T8; returns the last N DMs
  from a target app for the agent to summarize.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from fastmcp import FastMCP

from evaluation.backend_query import BackendQuery
from evaluation.mcp_overlay import OverlayView, WriteOverlay


def _build_env() -> dict:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--user_id", default=os.environ.get("PM3_USER_ID"))
    parser.add_argument("--t_test", type=int, default=int(os.environ.get("PM3_T_TEST", "0")))
    parser.add_argument("--backend_dir", default=os.environ.get("PM3_BACKEND_DIR", "backend"))
    parser.add_argument("--overlay_path", default=os.environ.get("PM3_OVERLAY_PATH"))
    args, _ = parser.parse_known_args()
    if not (args.user_id and args.t_test and args.overlay_path):
        print(f"[chatbot_mcp] missing config: {vars(args)}", file=sys.stderr)
        sys.exit(2)
    return vars(args)


def make_server() -> FastMCP:
    cfg = _build_env()
    user_id = cfg["user_id"]
    t_test = cfg["t_test"]
    bq = BackendQuery(cfg["backend_dir"])
    overlay = WriteOverlay(Path(cfg["overlay_path"]))
    view = OverlayView(bq, overlay)

    mcp = FastMCP("pm3-chatbot", version="2.0.0")

    @mcp.tool()
    def get_history(cursor: str | None = None, limit: int = 20) -> dict:
        """Recent chatbot conversation events (time-masked). Each event
        contains the full conversation turn list.
        """
        events = view.get_events(user_id=user_id, app="chatbot", since_timestamp=t_test)
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
        """Search past chatbot turns (substring over user/assistant content)."""
        q = (query or "").lower()
        hits = []
        for e in view.get_events(user_id=user_id, app="chatbot", since_timestamp=t_test):
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

    @mcp.tool()
    def send_message(message: str) -> dict:
        """The user sends a turn to the chatbot. (Included for simulation
        completeness; typically NOT called by the agent being evaluated —
        the agent IS the chatbot.)
        """
        if not message:
            return {"error": "empty message"}
        rec = view.write(tool="chatbot_send_message", app="chatbot", event={
            "author_id": "self",
            "content": {"caption": message},
            "interaction_format": {"app": "Chatbot", "action": "user_message"},
            "is_self_authored": True,
        })
        return {"ok": True, "message_id": rec["synthetic_event_id"]}

    @mcp.tool()
    def send_post_to_app(target_app: str, caption: str, media_refs: list[str] | None = None) -> dict:
        """Cross-app dispatch: the chatbot posts something on a target social app.

        `target_app` must be one of instagram | facebook | threads.
        Records to the overlay tagged with the target app so the grader sees
        the routing decision.
        """
        target_app = (target_app or "").lower()
        if target_app not in ("instagram", "facebook", "threads"):
            return {"error": f"target_app must be instagram|facebook|threads, got '{target_app}'"}
        if not caption or not caption.strip():
            return {"error": "caption must be non-empty"}
        event = {
            "source_interaction_type": "explicit_positive",
            "source_hashtags": _extract_hashtags(caption),
            "content_type": "image" if media_refs else "text",
            "content": {"caption": caption, "media_refs": media_refs or []},
            "author_id": "self",
            "is_self_authored": True,
            "is_dm": False,
            "interaction_format": {"app": target_app.capitalize(), "action": "posted_via_chatbot"},
            "dispatched_from": "chatbot",
        }
        rec = view.write(tool=f"{target_app}_create_post", app=target_app, event=event)
        return {"ok": True, "post_id": rec["synthetic_event_id"], "target_app": target_app}

    @mcp.tool()
    def summarize_inbox(target_app: str, window_hours: int = 24) -> dict:
        """Fetch DMs from the last `window_hours` on the target app. Server
        does NOT summarize — the agent does. This tool just returns the raw
        list of message text + metadata.
        """
        target_app = (target_app or "").lower()
        if target_app not in ("instagram", "facebook", "threads"):
            return {"error": f"target_app must be a social app, got '{target_app}'"}
        since_ts = max(0, t_test - window_hours * 3600)
        page = view.list_dm_threads(user_id=user_id, app=target_app,
                                     since_timestamp=t_test, limit=50)
        out = []
        for thread_summary in page.get("results", []):
            tid = thread_summary.get("thread_id")
            if thread_summary.get("latest_ts", 0) < since_ts:
                continue
            thread = view.get_dm_thread(user_id=user_id, app=target_app, thread_id=tid,
                                         since_timestamp=t_test)
            if thread:
                recent = [m for m in thread.get("results", []) if (m.get("timestamp") or 0) >= since_ts]
                if recent:
                    out.append({
                        "thread_id": tid,
                        "participants": thread.get("participants"),
                        "is_group": thread.get("is_group"),
                        "messages": recent,
                    })
        return {"threads": out, "window_hours": window_hours, "target_app": target_app}

    return mcp


def _extract_hashtags(text: str) -> list[str]:
    import re
    return re.findall(r"#\w+", text or "")


def run_server():
    make_server().run()


if __name__ == "__main__":
    run_server()
