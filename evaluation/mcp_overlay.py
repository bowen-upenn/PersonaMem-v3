"""Write overlay for MCP mode.

When the agent under test calls an MCP write tool (create_post, react,
send_dm, etc.) the handler appends a record to the per-run
`writes.jsonl`. Subsequent read tools in the SAME run must union the
frozen backend events with the write overlay — so the agent sees its
own posts appear in its own feed, matching real-app semantics.

Writes never mutate `backend/{user_id}/`. Each run starts with an
empty overlay — initial state is reproducible.

The overlay file doubles as the `final_state_diff` rubric input; the
grader reads the same writes.jsonl to check the agent performed the
expected writes.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from evaluation.backend_query import APPS, BackendQuery, _paginate


class WriteOverlay:
    """Appendable write log for one MCP-mode run."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, tool: str, app: str, event: dict) -> dict:
        record = {
            "tool": tool,
            "app": app,
            "timestamp_ms": int(time.time() * 1000),
            "synthetic_event_id": f"{app}_write_{uuid.uuid4().hex[:8]}",
            "event": event,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return out

    def events_for_app(self, app: str) -> list[dict]:
        """Return write records that affected `app` as events (for feed-read union)."""
        out = []
        for rec in self.read_all():
            if rec.get("app") != app:
                continue
            ev = dict(rec.get("event") or {})
            ev.setdefault("source_object_id", rec["synthetic_event_id"])
            ev.setdefault("source_timestamp", int(rec["timestamp_ms"] / 1000))
            ev.setdefault("is_self_authored", rec["tool"].endswith("_create_post"))
            ev.setdefault("is_dm", rec["tool"].endswith("_send_dm"))
            out.append(ev)
        return out


class OverlayView:
    """Wraps BackendQuery to inject overlay writes into reads within a run.

    Reads delegate to BackendQuery, then append matching overlay events.
    Writes are logged to the overlay (never touches `backend/`).

    Time-mask: `since_timestamp` still applies, but overlay events have
    timestamps ≥ T_test (they were just written), so feeds include them.
    """

    def __init__(self, bq: BackendQuery, overlay: WriteOverlay):
        self.bq = bq
        self.overlay = overlay

    def _merge_feed(self, app: str, base_events: list[dict]) -> list[dict]:
        overlay_events = self.overlay.events_for_app(app)
        if not overlay_events:
            return base_events
        merged = list(base_events) + overlay_events
        merged.sort(key=lambda x: x.get("source_timestamp", 0))
        return merged

    # Pass-through accessors that inject overlay events where appropriate.

    def get_events(self, user_id: str, app, since_timestamp: int, **kwargs) -> list[dict]:
        events = self.bq.get_events(user_id=user_id, app=app, since_timestamp=since_timestamp, **kwargs)
        apps = [app] if isinstance(app, str) and app in APPS else (list(app) if not isinstance(app, str) else APPS)
        # Only merge overlay for single-app queries (avoid double-merging).
        if isinstance(app, str) and app in APPS:
            events = self._merge_feed(app, events)
        return events

    def get_event_by_id(self, user_id: str, app: str, event_id: str) -> dict | None:
        # Check overlay first — newly-written events live there.
        for ev in self.overlay.events_for_app(app):
            if ev.get("source_object_id") == event_id:
                return ev
        return self.bq.get_event_by_id(user_id, app, event_id)

    def search_events(self, user_id, app, query, search_type="post", since_timestamp=None, cursor=None, limit=20):
        return self.bq.search_events(user_id, app, query, search_type, since_timestamp, cursor, limit)

    def list_dm_threads(self, user_id, app, **kwargs):
        return self.bq.list_dm_threads(user_id, app, **kwargs)

    def get_dm_thread(self, user_id, app, thread_id, **kwargs):
        return self.bq.get_dm_thread(user_id, app, thread_id, **kwargs)

    def get_trending(self, user_id):
        return self.bq.get_trending(user_id)

    def get_friend(self, user_id, friend_id):
        return self.bq.get_friend(user_id, friend_id)

    def get_profile_summary(self, user_id):
        return self.bq.get_profile_summary(user_id)

    def hashtag_summary(self, user_id, since_timestamp, app=None):
        return self.bq.hashtag_summary(user_id=user_id, since_timestamp=since_timestamp, app=app)

    # Write paths — append to overlay.

    def write(self, tool: str, app: str, event: dict) -> dict:
        return self.overlay.append(tool, app, event)
