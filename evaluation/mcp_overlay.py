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
import os
import time
import uuid
from pathlib import Path
from typing import Any

from evaluation.backend_query import APPS, BackendQuery, _paginate


class WriteOverlay:
    """Appendable write log for one MCP-mode run.

    When `PM3_T_TEST` is set in the env, writes stamp their simulated
    `source_timestamp` at `PM3_T_TEST + 1 + k` (k = number of prior writes
    at this same t_test). That places overlay events right after the
    user's query moment in the simulated timeline — so subsequent queries
    at later `t_test` values see these writes via the standard time mask.

    Wall-clock (`timestamp_ms` on the record + `created_at_ms` on the
    event) is preserved for auditability but is NOT the timeline position.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        self._last_seen_t_test: int | None = None
        self._writes_this_query: int = 0

    def _compute_sim_timestamp(self) -> int | None:
        """`PM3_T_TEST + 1 + k` for the k-th write at this t_test, or None if unset."""
        raw = os.getenv("PM3_T_TEST")
        if not raw:
            return None
        try:
            t_test = int(raw)
        except (TypeError, ValueError):
            return None
        if t_test != self._last_seen_t_test:
            self._last_seen_t_test = t_test
            self._writes_this_query = 0
        sim_ts = t_test + 1 + self._writes_this_query
        self._writes_this_query += 1
        return sim_ts

    def append(self, tool: str, app: str, event: dict) -> dict:
        now_ms = int(time.time() * 1000)
        sim_ts = self._compute_sim_timestamp()
        ev = dict(event or {})
        if sim_ts is not None:
            # Force simulated-timeline position; keep the wall-clock for audit.
            ev["source_timestamp"] = sim_ts
            ev.setdefault("created_at_ms", now_ms)
        record = {
            "tool": tool,
            "app": app,
            "timestamp_ms": now_ms,
            "sim_timestamp": sim_ts,
            "synthetic_event_id": f"{app}_write_{uuid.uuid4().hex[:8]}",
            "event": ev,
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
            # Prefer simulated timestamp (placed right after the user's query
            # moment); fall back to wall-clock only for legacy records.
            if "source_timestamp" not in ev:
                ev["source_timestamp"] = (
                    rec.get("sim_timestamp")
                    if rec.get("sim_timestamp") is not None
                    else int(rec["timestamp_ms"] / 1000)
                )
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

    def _merge_feed(self, app: str, base_events: list[dict], since_timestamp: int | None) -> list[dict]:
        overlay_events = self.overlay.events_for_app(app)
        if since_timestamp is not None:
            # Apply the same upper-bound mask the backend uses — prevents
            # future-written overlay events from leaking into earlier queries.
            overlay_events = [
                ev for ev in overlay_events
                if (ev.get("source_timestamp") or 0) < since_timestamp
            ]
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
            events = self._merge_feed(app, events, since_timestamp)
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
