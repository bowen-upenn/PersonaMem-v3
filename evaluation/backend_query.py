"""Time-masked view over the persona backend.

Two consumers:
- Long-context modes serialize into a big prompt via `serialize_history_for_context`
  (defined in `inference_utils.py`, which calls into this module).
- Mode 1a (real Claude Code subagent) materializes a **filtered filesystem snapshot**
  per test moment via `materialize_snapshot`. The subagent is spawned with `cwd` set
  to the snapshot, so its Read/Glob/Grep tools physically cannot reach outside it —
  no "please don't" prompt engineering, just filesystem scoping.

Fields that would leak train/test state (`split`, `over_personalization_irrelevant`,
`update_history`, cross-ref scores) are stripped from every materialized file.
"""

from __future__ import annotations

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
    ) -> list[dict]:
        """Time-masked event query. `app` may be a single app name or iterable.

        `hashtag` matches `source_hashtags` (case-insensitive substring).
        `category` matches any `preferences[].category` on the event.
        `interaction_type` matches `source_interaction_type` exactly.
        """
        apps = (app,) if isinstance(app, str) else tuple(app)
        out: list[dict] = []
        for a in apps:
            if a not in APPS:
                continue
            for e in self._load_events(user_id, a):
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
    ) -> list[dict]:
        """Flatten all preferences from events before `since_timestamp`.

        `polarity` ∈ {positive, negative} filters via `source_interaction_type`
        (explicit_positive / implicit_positive → positive, etc.).
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
    # Cheap cache: if all app files + profile.json exist, reuse.
    expected = [snap / "profile.json"] + [snap / f"{a}.json" for a in APPS]
    if snap.exists() and all(p.exists() for p in expected):
        return snap

    snap.mkdir(parents=True, exist_ok=True)
    # Profile: the safe slice only (no flat preferences, no hidden_personas).
    profile = bq.get_profile_summary(user_id)
    (snap / "profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2))
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
absent and the `split` / `over_personalization_irrelevant` labels have been
stripped — you are seeing the same view a recommender would have at inference
time.

## Files (use `Read` to open any of them)

- `profile.json` — demographic + app personas (Instagram, Facebook, Threads,
  Chatbot). Safe summary; no ground-truth preferences.
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
