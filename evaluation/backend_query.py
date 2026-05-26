"""Time-masked view over the persona backend.

Two consumers:
- Long-context modes serialize into a big prompt via `serialize_history_for_context`
  (defined in `inference_utils.py`, which calls into this module).
- Mode 1a (real Claude Code subagent) materializes a **filtered filesystem snapshot**
  per test moment via `materialize_snapshot`. The subagent is spawned with `cwd` set
  to the snapshot, so its Read/Glob/Grep tools physically cannot reach outside it —
  no "please don't" prompt engineering, just filesystem scoping.

Scoring-side fields (`update_history`, cross-ref scores, stereotype mark,
hidden persona labels) are stripped from every materialized file. As of R8
the data-gen pipeline no longer emits `split` / `over_personalization_irrelevant`;
those entries in `_LEAK_FIELDS_*` remain for backward-compatibility with
pre-R8 backends.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Iterable

APPS = ("instagram", "facebook", "threads", "chatbot")

# Fields stripped from every event/preference before returning — these leak
# ground-truth labels or scoring internals the agent under test must not see.
_LEAK_FIELDS_EVENT = ("split",)
_LEAK_FIELDS_PREF = (
    "split",
    "over_personalization_irrelevant",
    "update_history",
    "confidence_score_init",
    "confidence_cross_referenced",
    "stereotype_mark",
    "hidden_persona_labels",
)


def _strip_pref(pref: dict) -> dict:
    return {k: v for k, v in pref.items() if k not in _LEAK_FIELDS_PREF}


def _strip_event(event: dict) -> dict:
    out = {k: v for k, v in event.items() if k not in _LEAK_FIELDS_EVENT}
    if "preferences" in out:
        out["preferences"] = [_strip_pref(p) for p in out["preferences"]]
    return out


def _parse_formatted_ts(formatted: str) -> int | None:
    """Parse a 'HH:MM, MM/DD/YYYY' timestamp into unix seconds, UTC.

    Best-effort — returns None on any parse failure. Used by the
    contradiction-aware GT filter to read `update_history` entries whose
    underlying `timestamp` was stripped at save time.
    """
    if not formatted or not isinstance(formatted, str):
        return None
    import datetime as _dt
    for fmt in ("%H:%M, %m/%d/%Y", "%Y-%m-%d %H:%M"):
        try:
            return int(_dt.datetime.strptime(formatted, fmt)
                       .replace(tzinfo=_dt.timezone.utc).timestamp())
        except ValueError:
            pass
    return None


def _is_superseded_at(pref: dict, as_of_timestamp: int) -> bool:
    """Return True if the preference was contradicted-and-superseded before T.

    Reads the preference's `update_history` for an entry with
    `update_type == "contradicted"` AND
    `resolution == "stance_shift_with_precedent"`. When such an entry's
    timestamp is ≤ `as_of_timestamp`, AND the preference's own source
    timestamp is earlier than that entry, the preference is superseded
    at `as_of_timestamp` — it should not count as current ground truth.

    Works against a preference dict that still has `update_history`
    populated (i.e. BEFORE `_strip_pref` is applied).
    """
    history = pref.get("update_history") or []
    src_ts = pref.get("source_timestamp") or 0
    for h in history:
        if not isinstance(h, dict):
            continue
        if h.get("update_type") != "contradicted":
            continue
        if h.get("resolution") != "stance_shift_with_precedent":
            continue
        h_ts = h.get("timestamp") or _parse_formatted_ts(h.get("formatted_timestamp", ""))
        if not h_ts:
            continue
        if h_ts <= as_of_timestamp and src_ts and src_ts < h_ts:
            return True
    return False


class BackendQuery:
    """Read-only access to `backend/{user_id}/*.json` with time masking.

    Every query requires a `since_timestamp`: events with
    `source_timestamp >= since_timestamp` are dropped. The eval harness sets
    this per test moment; subagent callers must not override it.
    """

    def __init__(self, backend_dir: str | Path):
        self.base = Path(backend_dir)
        self._event_cache: dict[tuple[str, str], list[dict]] = {}
        self._profile_cache: dict[str, dict] = {}

    def _load_events(self, user_id: str, app: str) -> list[dict]:
        key = (user_id, app)
        if key in self._event_cache:
            return self._event_cache[key]
        path = self.base / user_id / f"{app}.json"
        if not path.exists():
            self._event_cache[key] = []
            return []
        with path.open() as f:
            events = json.load(f)
        self._event_cache[key] = events
        return events

    def _load_profile(self, user_id: str) -> dict:
        if user_id in self._profile_cache:
            return self._profile_cache[user_id]
        path = self.base / user_id / "profile.json"
        if not path.exists():
            self._profile_cache[user_id] = {}
            return {}
        with path.open() as f:
            self._profile_cache[user_id] = json.load(f)
        return self._profile_cache[user_id]

    # -- Public queries ------------------------------------------------------

    def list_users(self) -> list[str]:
        if not self.base.exists():
            return []
        return sorted(p.name for p in self.base.iterdir() if p.is_dir())

    def get_events(
        self,
        user_id: str,
        app: str | Iterable[str],
        since_timestamp: int,
        hashtag: str | None = None,
        category: str | None = None,
        interaction_type: str | None = None,
        limit: int | None = None,
        include_dms: bool = False,
    ) -> list[dict]:
        """Time-masked event query. `app` may be a single app name or iterable.

        `hashtag` matches `source_hashtags` (case-insensitive substring).
        `category` matches any `preferences[].category` on the event.
        `interaction_type` matches `source_interaction_type` exactly.
        `include_dms` (default False): DM thread entries (is_dm=true) are
        excluded by default — DMs are private and feed/content queries
        should not leak them. Use `list_dm_threads` / `get_dm_thread` for
        DM-specific access, or pass `include_dms=True` to see both.
        """
        apps = (app,) if isinstance(app, str) else tuple(app)
        out: list[dict] = []
        for a in apps:
            if a not in APPS:
                continue
            for e in self._load_events(user_id, a):
                if e.get("is_dm") and not include_dms:
                    continue
                ts = e.get("source_timestamp", 0)
                if ts >= since_timestamp:
                    continue
                if hashtag:
                    hashtags = [h.lower() for h in e.get("source_hashtags", [])]
                    if not any(hashtag.lower() in h for h in hashtags):
                        continue
                if category:
                    cats = [p.get("category", "").lower() for p in e.get("preferences", [])]
                    if category.lower() not in cats:
                        continue
                if interaction_type and e.get("source_interaction_type") != interaction_type:
                    continue
                out.append(_strip_event(e))
        out.sort(key=lambda x: x.get("source_timestamp", 0))
        if limit is not None:
            out = out[-limit:]
        return out

    def get_preferences(
        self,
        user_id: str,
        since_timestamp: int,
        app: str | Iterable[str] | None = None,
        polarity: str | None = None,
        category: str | None = None,
        include_superseded: bool = False,
    ) -> list[dict]:
        """Flatten all preferences from events before `since_timestamp`.

        `polarity` ∈ {positive, negative} filters via `source_interaction_type`
        (explicit_positive / implicit_positive → positive, etc.).

        `include_superseded` (default False): when False, preferences whose
        canonical has been CONTRADICTED-AND-SUPERSEDED (Phase 3 cross-polarity
        gate, Case B) before `since_timestamp` are filtered out. The ground
        truth at any T_test is the LATER stance only, never the superseded
        earlier stance. Set to True to include both stances (e.g., for audit).

        Returns one dict per preference occurrence, annotated with
        `source_app`, `source_timestamp`, `source_interaction_type`,
        `source_hashtags`.
        """
        apps: Iterable[str] = APPS if app is None else (app,) if isinstance(app, str) else app
        out: list[dict] = []
        for a in apps:
            for e in self._load_events(user_id, a):
                ts = e.get("source_timestamp", 0)
                if ts >= since_timestamp:
                    continue
                it = e.get("source_interaction_type", "")
                if polarity == "positive" and "positive" not in it:
                    continue
                if polarity == "negative" and "negative" not in it:
                    continue
                for p in e.get("preferences", []):
                    if category and p.get("category", "").lower() != category.lower():
                        continue
                    # Contradiction-aware filter: skip canonicals superseded at T
                    if not include_superseded:
                        p_with_src_ts = dict(p)
                        p_with_src_ts.setdefault("source_timestamp", ts)
                        if _is_superseded_at(p_with_src_ts, since_timestamp):
                            continue
                    out.append({
                        **_strip_pref(p),
                        "source_app": a,
                        "source_timestamp": ts,
                        "source_interaction_type": it,
                        "source_hashtags": e.get("source_hashtags", []),
                    })
        out.sort(key=lambda x: x.get("source_timestamp", 0))
        return out

    def get_conversations(
        self,
        user_id: str,
        since_timestamp: int,
        limit: int | None = None,
    ) -> list[dict]:
        """Recent chatbot turns. Returns the `conversation` list per event."""
        out = []
        for e in self._load_events(user_id, "chatbot"):
            ts = e.get("source_timestamp", 0)
            if ts >= since_timestamp:
                continue
            out.append({
                "source_timestamp": ts,
                "formatted_timestamp": e.get("formatted_timestamp"),
                "conversation_type": e.get("conversation_type"),
                "interaction_format": e.get("interaction_format"),
                "conversation": e.get("conversation", []),
            })
        out.sort(key=lambda x: x["source_timestamp"])
        if limit is not None:
            out = out[-limit:]
        return out

    def hashtag_summary(
        self,
        user_id: str,
        since_timestamp: int,
        app: str | Iterable[str] | None = None,
    ) -> list[dict]:
        """Counts per hashtag, split by positive vs negative engagement."""
        apps: Iterable[str] = APPS if app is None else (app,) if isinstance(app, str) else app
        counts: dict[str, dict[str, int]] = {}
        for a in apps:
            for e in self._load_events(user_id, a):
                ts = e.get("source_timestamp", 0)
                if ts >= since_timestamp:
                    continue
                it = e.get("source_interaction_type", "")
                polarity = "positive" if "positive" in it else "negative" if "negative" in it else "other"
                for h in e.get("source_hashtags", []):
                    entry = counts.setdefault(h, {"hashtag": h, "positive": 0, "negative": 0, "other": 0})
                    entry[polarity] += 1
        rows = list(counts.values())
        rows.sort(key=lambda r: r["positive"] + r["negative"], reverse=True)
        return rows

    def get_profile_summary(self, user_id: str) -> dict:
        """Minimal profile slice safe to show the agent: name, career, app personas, bio.

        Excludes the flat `preferences` list and `hidden_personas` (both are
        ground truth for judging, not input to the agent under test).
        """
        prof = self._load_profile(user_id)
        return {
            "user_id": prof.get("user_id"),
            "name": prof.get("name"),
            "bio": prof.get("bio"),
            "career": prof.get("career"),
            "education": prof.get("education"),
            "gender": prof.get("gender"),
            "race_ethnicity": prof.get("race_ethnicity"),
            "big_five": prof.get("big_five"),
            "mbti": prof.get("mbti"),
            "app_personas": prof.get("app_personas"),
        }

    def get_full_profile(self, user_id: str) -> dict:
        """Full profile — for judge / eval-side use only, never pass to the agent."""
        return dict(self._load_profile(user_id))

    # -- Calendar (R5 calendar modification stream) --------------------------

    def _load_calendar_modifications(self, user_id: str) -> list[dict]:
        """Return the raw modification stream for `user_id` (no time mask).

        Internal — callers should use `get_calendar_modifications` (time-masked)
        or `get_calendar_state` (folded to a point in time).
        """
        path = self.base / user_id / "calendar.json"
        if not path.exists():
            return []
        try:
            with path.open() as f:
                doc = json.load(f)
        except (ValueError, OSError):
            return []
        if not isinstance(doc, dict):
            return []
        mods = doc.get("modifications", [])
        return list(mods) if isinstance(mods, list) else []

    def get_calendar_modifications(
        self,
        user_id: str,
        since_timestamp: int,
        limit: int | None = None,
    ) -> list[dict]:
        """Return CRUD modifications with `ts < since_timestamp`, sorted ascending.

        Time-masked the same way as `get_events` / `get_preferences` — the
        agent at T_test sees only the modifications the user had made by then.
        """
        out = [m for m in self._load_calendar_modifications(user_id)
               if isinstance(m, dict) and (m.get("ts") or 0) < since_timestamp]
        out.sort(key=lambda m: m.get("ts", 0))
        if limit is not None:
            out = out[-limit:]
        return out

    def get_calendar_state(self, user_id: str, as_of_timestamp: int) -> dict[str, dict]:
        """Fold the modification stream to produce the calendar state at T.

        Returns `{entry_id → entry_dict}` after applying all modifications
        with `ts <= as_of_timestamp`. Updates patch fields; removes drop
        the entry. Modifications with `ts > as_of_timestamp` are invisible.
        """
        state: dict[str, dict] = {}
        for m in self._load_calendar_modifications(user_id):
            if not isinstance(m, dict):
                continue
            ts = m.get("ts") or 0
            if ts > as_of_timestamp:
                continue
            action = m.get("action")
            if action == "added":
                entry = m.get("entry")
                if isinstance(entry, dict) and entry.get("entry_id"):
                    state[entry["entry_id"]] = dict(entry)
            elif action == "updated":
                entry_id = m.get("entry_id")
                if entry_id and entry_id in state:
                    diff = m.get("diff") or {}
                    if isinstance(diff, dict):
                        for field, change in diff.items():
                            if isinstance(change, dict) and "to" in change:
                                state[entry_id][field] = change["to"]
            elif action == "removed":
                entry_id = m.get("entry_id")
                if entry_id and entry_id in state:
                    state.pop(entry_id, None)
        return state

    # -- MCP-support queries (Extension A′) ---------------------------------
    # Cursor pagination: opaque base64(f"{ts}:{eid}").
    # Sorted descending by (timestamp, event_id) — newest first.

    def get_event_by_id(self, user_id: str, app: str, event_id: str) -> dict | None:
        """O(1) lookup after first call caches the index. Returns a stripped event."""
        for e in self._load_events(user_id, app):
            if str(e.get("source_object_id", "")) == event_id:
                return _strip_event(e)
        return None

    def search_events(
        self,
        user_id: str,
        app: str | Iterable[str],
        query: str,
        search_type: str = "post",
        since_timestamp: int | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> dict:
        """Paginated search. search_type ∈ {post, user, hashtag}.

        - `post`: substring search over title/caption/overall_description.
        - `hashtag`: exact hashtag match (normalized, #-stripped).
        - `user`: placeholder — returns matching author_ids (needs Ext B; for
          v1 backend returns empty list).
        """
        apps: list[str] = list(APPS) if app == "all" else ([app] if isinstance(app, str) else list(app))
        q_lower = (query or "").lower().lstrip("#")
        out: list[dict] = []
        for a in apps:
            if a not in APPS:
                continue
            for e in self._load_events(user_id, a):
                # Feed search never returns DM threads — those are accessed
                # via list_dm_threads / get_dm_thread, never via search.
                if e.get("is_dm"):
                    continue
                ts = e.get("source_timestamp", 0)
                if since_timestamp is not None and ts >= since_timestamp:
                    continue
                match = False
                if search_type == "hashtag":
                    tags = [h.lower().lstrip("#") for h in (e.get("source_hashtags") or [])]
                    match = q_lower in tags
                elif search_type == "user":
                    # Ext B: match on author_id. v1: no-op.
                    match = str(e.get("author_id", "")).lower() == q_lower if e.get("author_id") else False
                else:  # post
                    content = e.get("content") or {}
                    haystack = " ".join(str(v) for v in [
                        content.get("title"), content.get("caption"), content.get("overall_description"),
                    ] if v).lower()
                    match = q_lower in haystack
                if match:
                    stripped = _strip_event(e)
                    stripped["_app"] = a
                    out.append(stripped)
        out.sort(key=lambda x: (x.get("source_timestamp", 0), str(x.get("source_object_id", ""))), reverse=True)
        return _paginate(out, cursor, limit)

    def _iter_dm_thread_entries(self, user_id: str, app: str) -> list[dict]:
        """Return the full DM thread entries for this app, merged-format.

        Each DM thread now lives as ONE event in {app}.json with `is_dm:
        true` and the full `messages[]` embedded. This helper extracts
        those entries (raw, NOT stripped, since the harness itself needs
        thread metadata for time-masking). Legacy fallback: if an older
        backend still has {app}_dms.json on disk, read it.
        """
        legacy_path = self.base / user_id / f"{app}_dms.json"
        if legacy_path.exists():
            # Pre-merge backends (kept for backward compat).
            try:
                with legacy_path.open() as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
            except (ValueError, OSError):
                pass
        out: list[dict] = []
        for e in self._load_events(user_id, app):
            if e.get("is_dm") and e.get("messages"):
                out.append(e)
        return out

    def list_dm_threads(
        self,
        user_id: str,
        app: str,
        since_timestamp: int | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> dict:
        """List DM threads for an app.

        Reads `is_dm: true` entries from the merged `{app}.json` and
        applies the time mask at the message level — threads whose entire
        message history is past `since_timestamp` are dropped.
        """
        threads = self._iter_dm_thread_entries(user_id, app)
        filtered: list[dict] = []
        for t in threads:
            msgs = t.get("messages") or []
            if since_timestamp is not None:
                msgs = [m for m in msgs if (m.get("timestamp") or 0) < since_timestamp]
            if not msgs:
                continue
            latest_ts = max(m.get("timestamp") or 0 for m in msgs)
            filtered.append({
                "thread_id": t.get("thread_id") or str(t.get("source_object_id", "")),
                "participants": t.get("participants") or [],
                "is_group": bool(t.get("is_group") or t.get("is_group_dm")),
                "latest_ts": latest_ts,
                "last_message_preview": (msgs[-1].get("text") or "")[:80],
                # The thread is carrying a forwarded post — its hashtags
                # reflect the topic of the forward, even when the message
                # text itself is just "lol" / "saw this".
                "source_hashtags": t.get("source_hashtags") or [],
                "unread_count": 0,
            })
        filtered.sort(key=lambda x: x["latest_ts"], reverse=True)
        return _paginate(filtered, cursor, limit, key=lambda x: (x["latest_ts"], x["thread_id"]))

    def get_dm_thread(
        self,
        user_id: str,
        app: str,
        thread_id: str,
        since_timestamp: int | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> dict | None:
        """Paginated messages in one DM thread. Returns None if thread not found."""
        thread: dict | None = None
        for t in self._iter_dm_thread_entries(user_id, app):
            tid = t.get("thread_id") or str(t.get("source_object_id", ""))
            if tid == thread_id:
                thread = t
                break
        if thread is None:
            return None
        msgs = thread.get("messages") or []
        if since_timestamp is not None:
            msgs = [m for m in msgs if (m.get("timestamp") or 0) < since_timestamp]
        msgs.sort(key=lambda m: m.get("timestamp") or 0)
        page = _paginate(msgs, cursor, limit, key=lambda m: (m.get("timestamp", 0), str(m.get("msg_id", ""))))
        return {
            "thread_id": thread_id,
            "participants": thread.get("participants") or [],
            "is_group": bool(thread.get("is_group") or thread.get("is_group_dm")),
            # Source hashtags from the forwarded post — useful for
            # persona-relevance gating even when message text is brief.
            "source_hashtags": thread.get("source_hashtags") or [],
            **page,
        }

    def get_trending(self, user_id: str) -> list[dict]:
        """Return trending topics visible in this user's feed.

        Scans instagram/facebook/threads app JSONs for feed_visible events
        with ``is_trending=True``.  Falls back to legacy ``trending.json``
        then to a degraded top-20-hashtag synthesis.
        """
        trending_events: list[dict] = []
        for app in ("instagram", "facebook", "threads"):
            for e in self._load_events(user_id, app):
                if e.get("is_trending"):
                    content = e.get("content") or {}
                    trending_events.append({
                        "hashtag": (
                            e.get("trending_primary_hashtag")
                            or (e.get("source_hashtags") or [""])[0]
                        ),
                        "trending_topic": (
                            e.get("trending_topic")
                            or content.get("trending_topic", "")
                        ),
                        "relevance": e.get("trending_relevance", "relevant"),
                        "user_aligned": e.get("trending_relevance", "relevant") == "relevant",
                        "app": app,
                        "post_id": e.get("source_object_id", ""),
                    })
        if trending_events:
            relevant = [t for t in trending_events if t["user_aligned"]]
            irrelevant = [t for t in trending_events if not t["user_aligned"]]
            for i, t in enumerate(relevant + irrelevant):
                t["rank"] = i + 1
            return relevant + irrelevant

        # Legacy fallback: trending.json (pre-migration backends).
        path = self.base / user_id / "trending.json"
        if path.exists():
            with path.open() as f:
                data = json.load(f)
            if isinstance(data, dict) and "hashtags" in data:
                return data["hashtags"]
            if isinstance(data, list):
                return data

        # Degraded fallback.
        counts: dict[str, int] = {}
        for app in APPS:
            for e in self._load_events(user_id, app):
                for h in e.get("source_hashtags", []):
                    counts[h] = counts.get(h, 0) + 1
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:20]
        return [{"hashtag": h, "rank": i + 1, "degraded_fallback": True} for i, (h, _) in enumerate(top)]

    def get_friend(self, user_id: str, friend_id: str) -> dict | None:
        """Resolve a friend_id from profile.friends[] (Ext B addendum 1).

        Returns None if the friend graph isn't populated (v1 backend).
        """
        prof = self._load_profile(user_id)
        friends = prof.get("friends") or []
        for fr in friends:
            if fr.get("friend_id") == friend_id:
                return dict(fr)
        return None


# -- Cursor pagination helpers ----------------------------------------------

def _encode_cursor(ts: int, eid: str) -> str:
    return base64.urlsafe_b64encode(f"{ts}:{eid}".encode()).decode()


def _decode_cursor(c: str) -> tuple[int, str]:
    try:
        ts_s, eid = base64.urlsafe_b64decode(c.encode()).decode().split(":", 1)
        return int(ts_s), eid
    except Exception:
        return 0, ""


def _paginate(items: list[dict], cursor: str | None, limit: int, key=None) -> dict:
    """Slice `items` based on `cursor`. Returns {results, nextCursor?}.

    Default key: (source_timestamp desc, source_object_id).
    """
    if key is None:
        key = lambda x: (x.get("source_timestamp", 0), str(x.get("source_object_id", "")))
    start = 0
    if cursor:
        ts, eid = _decode_cursor(cursor)
        # Linear scan to find the cursor position (fine for per-user scale).
        for i, it in enumerate(items):
            k = key(it)
            if k[0] < ts or (k[0] == ts and k[1] > eid):
                start = i
                break
        else:
            start = len(items)
    page = items[start:start + limit]
    out = {"results": page}
    if start + limit < len(items):
        last = page[-1]
        k = key(last)
        out["nextCursor"] = _encode_cursor(int(k[0]), str(k[1]))
    return out


DEFAULT_SNAPSHOT_ROOT = Path("/tmp/pm3_eval_snapshots")

# -- Filesystem snapshot for real Claude Code subagents (Mode 1a) -----------

def materialize_snapshot(
    bq: "BackendQuery",
    user_id: str,
    t_test: int,
    out_dir: Path | None = None,
) -> Path:
    """Write a time-masked, leak-stripped copy of `backend/{user_id}/` to
    `<out_dir>/{user_id}/T_{t_test}` and return that path.

    Default `out_dir` is `/tmp/pm3_eval_snapshots/` — deliberately outside
    the project tree so Claude Code can't walk upward to the project's `.git`
    and leak the real `backend/` path via its dynamic system prompt.

    The subagent is spawned with `cwd` set to this path AND per-invocation
    settings that restrict Read/Glob/Grep to this absolute path. Every file
    inside has future events dropped and leak-sensitive fields removed.

    Cached: idempotent given the same (user_id, t_test). Delete the snapshot
    directory to force a rebuild.
    """
    base = Path(out_dir) if out_dir is not None else DEFAULT_SNAPSHOT_ROOT
    snap = base / user_id / f"T_{t_test}"
    # Cheap cache: if all app files exist, reuse (profile.json deliberately
    # excluded — see the firewall comment below).
    expected = [snap / f"{a}.json" for a in APPS]
    if snap.exists() and all(p.exists() for p in expected):
        return snap

    snap.mkdir(parents=True, exist_ok=True)
    # Profile is DELIBERATELY NOT written into the snapshot.
    # The eval-side firewall (Phase G) hides profile.json from the agent so that
    # personalization must be inferred from the event timeline alone — no
    # demographic / app-persona scaffolding that would shortcut the test.
    # Per-app event lists, already time-masked + leak-stripped by BackendQuery.
    for app in APPS:
        events = bq.get_events(user_id=user_id, app=app, since_timestamp=t_test)
        (snap / f"{app}.json").write_text(json.dumps(events, ensure_ascii=False, indent=2))
    # The agent only has Read (no Glob/Grep/Bash), so this README enumerates
    # every file it can open and describes their structure.
    instagram = json.loads((snap / "instagram.json").read_text())
    facebook = json.loads((snap / "facebook.json").read_text())
    threads = json.loads((snap / "threads.json").read_text())
    chatbot = json.loads((snap / "chatbot.json").read_text())
    (snap / "README.md").write_text(f"""# User {user_id} history snapshot (time-masked)

This directory contains user {user_id}'s interaction history as of a specific
test moment. Every event has `source_timestamp < T_test`; future events are
absent. Scoring fields (`update_history`, `confidence_*`, `stereotype_mark`,
`hidden_persona_labels`) are stripped — you are seeing the same view a
recommender would have at inference time. Newer backends no longer carry
`split` / `over_personalization_irrelevant`; the eval harness now picks test
moments dynamically from the full timeline.

## Files (use `Read` to open any of them)

You see only the per-app event timelines. The user's profile (demographics, app
personas, hidden persona summary, full preference list) is intentionally NOT
included — you must infer who this user is from their actual interactions
across the four apps below.

- `instagram.json` — {len(instagram)} events, time-sorted ascending.
- `facebook.json` — {len(facebook)} events.
- `threads.json` — {len(threads)} events.
- `chatbot.json` — {len(chatbot)} conversation events (each carries a full
  `conversation` turn list + `interaction_format.user_message`).

## Event schema

Each event is a JSON object with:
- `source_timestamp` (unix epoch), `formatted_timestamp`
- `source_hashtags` — array of `#tags` on the content
- `source_interaction_type` — one of:
  `explicit_positive`, `implicit_positive`, `explicit_negative`,
  `implicit_negative`
- `interaction_format.action_label` — human-readable action (e.g. "Liked",
  "Viewed more than 75% of the reel", "Asked the assistant not to personalize
  recommendations around a preference")
- `content.title`, `content.caption`, `content.overall_description`
- `preferences[]` — inferred `persona_item` + `category` for this engagement
  (may be empty for greyscale negative stubs)

## Reading tip

Events within each app JSON are sorted oldest → newest. Read the full file
once and scan; do not try to enumerate files by `ls` or `find` — those tools
are not available to you. Only the five filenames above exist in this
directory.
""")
    return snap
