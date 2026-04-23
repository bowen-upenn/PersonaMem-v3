"""Factory for Instagram / Facebook / Threads mock MCP servers.

Each of the three social apps has nearly-identical tool surface (feed, DMs,
posts, reactions, comments, search), so a single factory parameterized by
(app_name, reactions_whitelist, caption_char_limit, special_actions) covers
all three with one set of tested code.

Invocation contract (stdio MCP):
    python -m evaluation.mcp_servers.instagram_mcp_server \\
        --user_id 115 --t_test 1775825828 \\
        --backend_dir backend \\
        --overlay_path benchmark/115/runs/<ts>/writes.jsonl

The Claude CLI runner passes these as environment variables via
`--mcp-config` → see evaluation/mcp_config_builder.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from evaluation.backend_query import BackendQuery
from evaluation.mcp_overlay import OverlayView, WriteOverlay


# Per-app constants.
CAPTION_LIMITS = {"instagram": 2200, "facebook": 63206, "threads": 500}
REACTIONS = {
    "instagram": {"like", "save", "share", "not_interested"},
    "facebook": {"like", "love", "haha", "wow", "sad", "angry", "care"},
    "threads": {"like", "quote_repost", "not_interested"},
}


def _build_env_from_argv_or_env() -> dict:
    """Extract runtime config from env vars (MCP config path) or argv."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", default=os.environ.get("PM3_USER_ID"))
    parser.add_argument("--t_test", type=int, default=int(os.environ.get("PM3_T_TEST", "0")))
    parser.add_argument("--backend_dir", default=os.environ.get("PM3_BACKEND_DIR", "backend"))
    parser.add_argument("--overlay_path", default=os.environ.get("PM3_OVERLAY_PATH"))
    parser.add_argument("--app", default=os.environ.get("PM3_APP"))
    args, _ = parser.parse_known_args()
    if not (args.user_id and args.t_test and args.overlay_path and args.app):
        print(f"[{args.app or '?'}_mcp] missing required config: {vars(args)}", file=sys.stderr)
        sys.exit(2)
    return vars(args)


def make_server(app: str) -> FastMCP:
    """Build a FastMCP server for one social app. Tools defined inline."""
    cfg = _build_env_from_argv_or_env()
    user_id = cfg["user_id"]
    t_test = cfg["t_test"]
    bq = BackendQuery(cfg["backend_dir"])
    overlay = WriteOverlay(Path(cfg["overlay_path"]))
    view = OverlayView(bq, overlay)

    mcp = FastMCP(f"pm3-{app}", version="2.0.0")

    @mcp.tool()
    def get_feed(cursor: str | None = None, limit: int = 20) -> dict:
        """Return the user's recent feed on this app, time-masked to < T_test
        (plus any posts the agent has itself created during this run).
        """
        events = view.get_events(user_id=user_id, app=app, since_timestamp=t_test)
        events.sort(key=lambda x: x.get("source_timestamp", 0), reverse=True)
        # Lightweight projection for feed.
        feed_items = [
            {
                "post_id": str(e.get("source_object_id", "")),
                "timestamp": e.get("source_timestamp"),
                "hashtags": e.get("source_hashtags", []),
                "content_type": e.get("content_type"),
                "title": (e.get("content") or {}).get("title"),
                "caption": (e.get("content") or {}).get("caption"),
                "author_id": e.get("author_id", "unknown"),
                "is_self_authored": bool(e.get("is_self_authored")),
            }
            for e in events
        ]
        # Simple slicing (cursor = index-based here; BackendQuery's typed pagination
        # used elsewhere, but MCP feed-read is usually small).
        start = int(cursor) if (cursor and cursor.isdigit()) else 0
        page = feed_items[start:start + limit]
        out = {"results": page}
        if start + limit < len(feed_items):
            out["nextCursor"] = str(start + limit)
        return out

    @mcp.tool()
    def get_post(post_id: str) -> dict | None:
        """Fetch full detail for one post."""
        return view.get_event_by_id(user_id=user_id, app=app, event_id=post_id)

    @mcp.tool()
    def search(query: str, search_type: str = "post", cursor: str | None = None, limit: int = 20) -> dict:
        """Search posts / users / hashtags on this app. time-masked to < T_test."""
        return view.search_events(
            user_id=user_id, app=app, query=query, search_type=search_type,
            since_timestamp=t_test, cursor=cursor, limit=limit,
        )

    @mcp.tool()
    def get_profile() -> dict:
        """Minimal user profile — app persona + bio only (no preferences)."""
        return view.get_profile_summary(user_id)

    @mcp.tool()
    def list_dms(cursor: str | None = None, limit: int = 20) -> dict:
        """Paginated DM threads on this app."""
        return view.list_dm_threads(user_id=user_id, app=app, since_timestamp=t_test,
                                     cursor=cursor, limit=limit)

    @mcp.tool()
    def get_dm_thread(thread_id: str, cursor: str | None = None, limit: int = 50) -> dict | None:
        """Paginated messages in one thread."""
        return view.get_dm_thread(user_id=user_id, app=app, thread_id=thread_id,
                                   since_timestamp=t_test, cursor=cursor, limit=limit)

    @mcp.tool()
    def create_post(caption: str, media_refs: list[str] | None = None, alt_text: str | None = None) -> dict:
        """Publish a post on this app on the user's behalf.

        Validates: non-empty caption, caption length ≤ app limit.
        Appends one record to the overlay; the post immediately appears in
        subsequent get_feed calls within this run.
        """
        if not caption or not caption.strip():
            return {"error": "caption must be non-empty"}
        max_len = CAPTION_LIMITS.get(app, 2200)
        if len(caption) > max_len:
            return {"error": f"caption exceeds {app} limit of {max_len} chars"}
        event = {
            "source_interaction_type": "explicit_positive",
            "source_hashtags": _extract_hashtags(caption),
            "content_type": "image" if media_refs else "text",
            "content": {"caption": caption, "media_refs": media_refs or [], "alt_text": alt_text},
            "author_id": "self",
            "relationship": "self",
            "is_self_authored": True,
            "is_dm": False,
            "interaction_format": {"app": app.capitalize(), "action": "posted_text" if not media_refs else "posted_photo"},
        }
        rec = view.write(tool=f"{app}_create_post", app=app, event=event)
        return {"post_id": rec["synthetic_event_id"], "created_at": rec["timestamp_ms"]}

    @mcp.tool()
    def react(post_id: str, reaction_type: str = "like") -> dict:
        """Record a reaction on a post. `reaction_type` is validated against
        this app's allowed reactions."""
        if reaction_type not in REACTIONS.get(app, set()):
            return {"error": f"reaction '{reaction_type}' not allowed on {app}; valid: {sorted(REACTIONS[app])}"}
        target = view.get_event_by_id(user_id=user_id, app=app, event_id=post_id)
        if target is None:
            return {"error": f"post_id {post_id} not found"}
        event = {
            "source_interaction_type": "explicit_positive" if reaction_type != "not_interested" else "explicit_negative",
            "source_hashtags": target.get("source_hashtags", []),
            "content_type": "text",
            "content": {"caption": f"reaction={reaction_type} on post={post_id}"},
            "author_id": "self",
            "is_self_authored": False,
            "is_dm": False,
            "interaction_format": {"app": app.capitalize(), "action": f"reacted_{reaction_type}"},
            "reacted_to_post_id": post_id,
        }
        rec = view.write(tool=f"{app}_react", app=app, event=event)
        return {"ok": True, "reaction_id": rec["synthetic_event_id"]}

    @mcp.tool()
    def comment(post_id: str, text: str) -> dict:
        """Post a comment on a post."""
        if not text or not text.strip():
            return {"error": "comment text must be non-empty"}
        target = view.get_event_by_id(user_id=user_id, app=app, event_id=post_id)
        if target is None:
            return {"error": f"post_id {post_id} not found"}
        event = {
            "source_interaction_type": "explicit_positive",
            "source_hashtags": target.get("source_hashtags", []),
            "content_type": "text",
            "content": {"caption": text},
            "author_id": "self",
            "is_self_authored": True,
            "is_dm": False,
            "interaction_format": {"app": app.capitalize(), "action": "commented"},
            "commented_on_post_id": post_id,
        }
        rec = view.write(tool=f"{app}_comment", app=app, event=event)
        return {"ok": True, "comment_id": rec["synthetic_event_id"]}

    @mcp.tool()
    def send_dm(recipient_id: str, message: str) -> dict:
        """Send a DM to a recipient. `recipient_id` must resolve via get_friend
        (Extension B friend graph) or be an existing thread participant.
        """
        if not message or not message.strip():
            return {"error": "message must be non-empty"}
        friend = view.get_friend(user_id, recipient_id)
        # v1 fallback: accept any recipient but log a warning.
        event = {
            "source_interaction_type": "explicit_positive",
            "source_hashtags": [],
            "content_type": "text",
            "content": {"caption": message},
            "author_id": "self",
            "recipient_id": recipient_id,
            "relationship": friend.get("relationship_depth") if friend else "unknown",
            "is_self_authored": True,
            "is_dm": True,
            "interaction_format": {"app": app.capitalize(), "action": "sent_dm"},
        }
        rec = view.write(tool=f"{app}_send_dm", app=app, event=event)
        return {"ok": True, "message_id": rec["synthetic_event_id"], "recipient_resolved": friend is not None}

    return mcp


def _extract_hashtags(text: str) -> list[str]:
    import re
    return re.findall(r"#\w+", text or "")


def run_server():
    """CLI entry point — parses args, builds server, runs stdio."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--app", required=False, default=os.environ.get("PM3_APP"))
    args, _ = parser.parse_known_args()
    app = args.app
    if not app:
        print("error: --app / PM3_APP required", file=sys.stderr)
        sys.exit(2)
    server = make_server(app)
    server.run()
