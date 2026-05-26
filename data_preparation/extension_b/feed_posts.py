"""Generate feed-visible events for proactive feed-react tasks.

Two kinds of events are produced and appended to backend/{uid}/{app}.json:

  - Friend self-posts (`author_id = "friend_X"`) — generated synthetically
    via the LLM client. For each close friend, ~1 post on a hashtag the
    user is positive about (relevance="relevant") and ~1 post on a hashtag
    the user does not engage with (relevance="irrelevant").

  - Trending platform content (`author_id = "public_creator"`) — generated
    from real web-search results for "top {platform} trends {month year}"
    bucketed across the user's observation window. The LLM extracts trend
    topics from search results, then synthesizes 1 post per trend per app.
    Half are picked to match the user's positive hashtags
    (relevance="relevant"), half are off-topic (relevance="irrelevant").

All generated events carry `source_interaction_type = "feed_visible"` and
empty `preferences = []`. These are content the user *could* have seen
in their feed but did NOT engage with — the AI under test is supposed
to decide whether to surface them.

Caching: web-search results are cached at
`backend/{uid}/.trending_search_cache.json` keyed by
`(platform, year_month)`. Reruns are free.

Modes:
  - With both `llm_client` and `web_search_fn`: full generation.
  - With only `llm_client`: friend posts only; trending skipped with warn.
  - With neither: no-op (graceful skip; same pattern as Step 28).
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
import uuid
from pathlib import Path
from typing import Callable, Optional

from data_preparation.utils import extract_json_from_response
from data_preparation import prompts


SOCIAL_APPS = ("instagram", "facebook", "threads")
APP_PRETTY = {"instagram": "Instagram", "facebook": "Facebook", "threads": "Threads"}

# Per-platform feed-visible action sentinel — used so visualize.py and MCP
# servers do not crash when reading an event whose interaction_format is
# present but represents "no engagement, content was just in the feed".
FEED_VISIBLE_ACTION = {
    "instagram": {"action": "appeared_in_feed", "label": "Appeared in feed (no engagement)"},
    "facebook":  {"action": "appeared_in_feed", "label": "Appeared in feed (no engagement)"},
    "threads":   {"action": "appeared_in_feed", "label": "Appeared in feed (no engagement)"},
}

# Friend post counts per close friend, per app.
#
# Only "relevant" friend posts are generated — no irrelevant variant.
# Reason: close friends share interests with the user (that's why they
# are close), so any post in the friend's voice tends to land on a
# topic the user cares about even when we try to steer it off-topic
# via an off-user hashtag. The Stage-2 content-relevance check
# confirmed empirically that all "irrelevant" friend posts get flipped
# back to "relevant" because the friend writes about their actual
# shared interests. Restraint testing for proactive feed-react lives
# in `trending_feed_react` (irrelevant trending content) and
# `proactive_overactive_check` (idle moments) instead.
_FRIEND_POSTS_PER_FRIEND_PER_APP = {
    "instagram": 1,
    "facebook":  1,
    "threads":   1,
}

# Trending bucket plan: how many monthly buckets to sample, how many
# trends per bucket, how many posts per trend per app.
_TRENDING_BUCKETS = 4               # 4 monthly buckets across user's window
_TRENDS_PER_BUCKET = 3              # 3 distinct trend topics per bucket
_RELEVANT_FRACTION = 0.5            # ~50% of trends match user's positive hashtags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_ts(unix_ts: int) -> str:
    """HH:MM, MM/DD/YYYY format used by every other event in the backend."""
    d = dt.datetime.fromtimestamp(unix_ts, tz=dt.timezone.utc)
    return d.strftime("%H:%M, %m/%d/%Y")


def _new_oid() -> str:
    """Generate a fresh source_object_id distinguishable from existing ones."""
    return f"feedvis_{uuid.uuid4().hex[:10]}"


def _user_window(events: list[dict]) -> tuple[int, int]:
    """Min and max source_timestamp across a user's events."""
    ts = [int(e.get("source_timestamp", 0)) for e in events if e.get("source_timestamp")]
    if not ts:
        return (0, 0)
    return (min(ts), max(ts))


def _top_positive_hashtags(profile: dict, app_events: list[dict], k: int = 25) -> list[str]:
    """User's most-engaged-with hashtags from positive interactions."""
    counts: dict[str, int] = {}
    for e in app_events:
        itype = (e.get("source_interaction_type") or "").lower()
        if itype not in ("explicit_positive", "implicit_positive"):
            continue
        for h in (e.get("source_hashtags") or []):
            if not isinstance(h, str):
                continue
            counts[h] = counts.get(h, 0) + 1
    # Sort by count desc, return tag names.
    return [h for h, _ in sorted(counts.items(), key=lambda x: -x[1])[:k]]


def _negative_or_unseen_hashtags(profile: dict, app_events: list[dict],
                                  positive_set: set[str], k: int = 15) -> list[str]:
    """Hashtags the user has actively disliked OR never engaged with.

    First preference: explicit_negative engagement tags.
    Second preference: tags seen on others' posts but never engaged with positively.
    Fallback: a small set of generic off-topic hashtags so the irrelevant
    case still has data to draw from.
    """
    neg_counts: dict[str, int] = {}
    for e in app_events:
        itype = (e.get("source_interaction_type") or "").lower()
        if itype not in ("explicit_negative", "implicit_negative"):
            continue
        for h in (e.get("source_hashtags") or []):
            if isinstance(h, str) and h not in positive_set:
                neg_counts[h] = neg_counts.get(h, 0) + 1
    out = [h for h, _ in sorted(neg_counts.items(), key=lambda x: -x[1])[:k]]
    if len(out) >= 3:
        return out
    # Fallback: generic off-topic tags.
    generic = ["#celebritygossip", "#cryptotips", "#mlm", "#dietfad",
               "#dropshipping", "#luxurycars", "#celebritydrama",
               "#millionairemindset", "#hustleculture", "#fastfashion"]
    out.extend(h for h in generic if h not in positive_set and h not in out)
    return out[:k]


def _norm_tag(h: str) -> str:
    return (h or "").strip().lower().lstrip("#")


# ---------------------------------------------------------------------------
# Friend self-post generation
# ---------------------------------------------------------------------------

def generate_friend_self_posts(
    user_id: str,
    app: str,
    profile: dict,
    existing_events: list[dict],
    llm_client,
    rng_seed: int = 0,
    verbose: bool = False,
) -> list[dict]:
    """Generate friend-authored self-posts visible in user's feed.

    Returns a list of feed_visible event dicts ready to append to the app's
    JSON. Each carries `_feed_react_meta` with `relevance` and `friend_id`
    so the gather helper in Step 28 can read it back.
    """
    if llm_client is None:
        if verbose:
            print(f"[feed_posts/{app}] no llm_client — skipping friend posts.")
        return []

    friends = [
        f for f in (profile.get("friends") or [])
        if f.get("relationship_depth") == "close" and f.get("friend_id")
    ]
    if not friends:
        if verbose:
            print(f"[feed_posts/{app}] no close friends in graph — skipping.")
        return []

    pos_tags = _top_positive_hashtags(profile, existing_events, k=25)
    if not pos_tags:
        if verbose:
            print(f"[feed_posts/{app}] no positive hashtags — skipping friend posts.")
        return []
    neg_tags = _negative_or_unseen_hashtags(profile, existing_events,
                                              positive_set=set(pos_tags), k=15)

    lo_ts, hi_ts = _user_window(existing_events)
    if lo_ts == hi_ts == 0:
        return []

    rng = random.Random(rng_seed + 7777)
    out: list[dict] = []
    n_per_friend = _FRIEND_POSTS_PER_FRIEND_PER_APP.get(app, 1)
    app_pretty = APP_PRETTY[app]
    user_name = profile.get("name", "the user")

    for friend in friends:
        friend_id = friend["friend_id"]
        display_name = friend.get("display_name", friend_id)
        shared = [h for h in (friend.get("shared_interests") or []) if isinstance(h, str)]

        # Build picks: 1 relevant tag, 1 irrelevant tag.
        relevant_pool = [h for h in pos_tags if any(_norm_tag(s) in _norm_tag(h) or _norm_tag(h) in _norm_tag(s) for s in shared)]
        if not relevant_pool:
            relevant_pool = pos_tags[:8]

        # Only generate "relevant" friend posts — irrelevant ones get
        # flipped back to relevant by the Stage-2 content check anyway
        # (close friends write about their shared interests, which by
        # definition overlap with the user's). See note at
        # _FRIEND_POSTS_PER_FRIEND_PER_APP for rationale.
        picks: list[tuple[str, str]] = []
        for _ in range(max(0, n_per_friend)):
            picks.append((rng.choice(relevant_pool), "relevant"))

        for hashtag, polarity in picks:
            try:
                prompt = prompts.friend_feed_post_prompt(
                    friend_display_name=display_name,
                    friend_shared_interests=shared,
                    hashtag=hashtag,
                    polarity=polarity,
                    platform=app_pretty,
                    user_name=user_name,
                )
                resp = llm_client.query_llm(prompt)
                parsed = extract_json_from_response(resp) or {}
            except Exception as exc:
                if verbose:
                    print(f"[feed_posts/{app}] friend {friend_id} {polarity} failed: {exc}")
                continue
            if not isinstance(parsed, dict) or not parsed.get("caption"):
                continue

            ts = rng.randint(lo_ts, hi_ts)
            hashtags = parsed.get("hashtags") or [hashtag]
            if hashtag not in hashtags:
                hashtags = [hashtag] + list(hashtags)

            evt = {
                "source_object_id": _new_oid(),
                "source_timestamp": ts,
                "formatted_timestamp": _format_ts(ts),
                "source_hashtags": hashtags[:6],
                "source_interaction_type": "feed_visible",
                "author_id": friend_id,
                "recipient_id": "",
                "relationship": "close_friend",
                "is_self_authored": False,
                "is_dm": False,
                "is_trending": False,
                "interaction_format": {
                    "app": app_pretty,
                    "action": FEED_VISIBLE_ACTION[app]["action"],
                    "action_label": FEED_VISIBLE_ACTION[app]["label"],
                    "user_message": None,
                },
                "event_location": None,
                "content_type": parsed.get("content_type", "image" if app == "instagram" else "text"),
                "content": {
                    "title": parsed.get("title", ""),
                    "caption": parsed.get("caption", ""),
                    "overall_description": parsed.get("overall_description", ""),
                },
                "preferences": [],
                "_feed_react_meta": {
                    "kind": "friend_feed",
                    "friend_id": friend_id,
                    "friend_display_name": display_name,
                    "primary_hashtag": hashtag,
                    "relevance": polarity,
                },
            }
            out.append(evt)

    if verbose:
        print(f"[feed_posts/{app}] friend posts generated: {len(out)}")
    return out


# ---------------------------------------------------------------------------
# Trending post generation
# ---------------------------------------------------------------------------

def _month_buckets(lo_ts: int, hi_ts: int, n: int) -> list[tuple[int, str, str]]:
    """Pick n monthly buckets evenly spread between lo_ts and hi_ts.

    Returns list of (anchor_ts, year_str, month_str).
    """
    if n <= 0 or lo_ts >= hi_ts:
        return []
    out = []
    seen_months: set[str] = set()
    span = hi_ts - lo_ts
    for i in range(n):
        anchor_ts = lo_ts + (i + 1) * span // (n + 1)
        d = dt.datetime.fromtimestamp(anchor_ts, tz=dt.timezone.utc)
        key = d.strftime("%Y-%m")
        if key in seen_months:
            continue
        seen_months.add(key)
        out.append((anchor_ts, d.strftime("%Y"), d.strftime("%B")))
    return out


def _cache_path(backend_dir: str | Path, user_id: str) -> Path:
    return Path(backend_dir) / user_id / ".trending_search_cache.json"


def _load_cache(backend_dir: str | Path, user_id: str) -> dict:
    p = _cache_path(backend_dir, user_id)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return {}


def _save_cache(backend_dir: str | Path, user_id: str, cache: dict) -> None:
    p = _cache_path(backend_dir, user_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def generate_trending_posts(
    user_id: str,
    app: str,
    profile: dict,
    existing_events: list[dict],
    llm_client,
    web_search_fn: Optional[Callable[[str], str]],
    backend_dir: str | Path,
    rng_seed: int = 0,
    verbose: bool = False,
) -> list[dict]:
    """Generate platform-trending feed posts visible in user's feed.

    `web_search_fn(query)` should return a free-text blob of search results
    (titles + snippets concatenated). If None, trending generation is
    skipped gracefully — friend posts can still be generated.

    Returns a list of feed_visible event dicts.
    """
    if llm_client is None:
        if verbose:
            print(f"[feed_posts/{app}] trending skipped (no llm_client).")
        return []
    # web_search_fn is optional: if a bucket's trends aren't already cached
    # AND no web_search_fn is provided, we'll skip THAT bucket (not all
    # trending generation). This lets pre-populated caches work without
    # any web-search infrastructure.

    pos_tags = _top_positive_hashtags(profile, existing_events, k=20)
    neg_tags = _negative_or_unseen_hashtags(profile, existing_events,
                                              positive_set=set(pos_tags), k=12)
    if not pos_tags:
        if verbose:
            print(f"[feed_posts/{app}] no positive hashtags — skipping trending.")
        return []

    lo_ts, hi_ts = _user_window(existing_events)
    if lo_ts == hi_ts == 0:
        return []

    buckets = _month_buckets(lo_ts, hi_ts, _TRENDING_BUCKETS)
    if not buckets:
        return []

    app_pretty = APP_PRETTY[app]
    cache = _load_cache(backend_dir, user_id)
    rng = random.Random(rng_seed + 9999)
    out: list[dict] = []

    for anchor_ts, year, month in buckets:
        cache_key = f"{app}|{year}-{month}"
        if cache_key in cache and isinstance(cache[cache_key].get("trends"), list):
            trends = cache[cache_key]["trends"]
        else:
            # Cache miss — we need web_search_fn to fetch fresh trends.
            if web_search_fn is None:
                if verbose:
                    print(f"[feed_posts/{app}] cache miss for {cache_key} and no "
                          f"web_search_fn — skipping this bucket.")
                continue
            # Run a web search for trends in this period.
            try:
                query = f"top {app_pretty} trending topics {month} {year}"
                if verbose:
                    print(f"[feed_posts/{app}] web search: {query}")
                search_results = web_search_fn(query)
            except Exception as exc:
                if verbose:
                    print(f"[feed_posts/{app}] web search failed for {cache_key}: {exc}")
                continue

            # Extract trend topics via LLM.
            try:
                extract_prompt = prompts.trending_topic_extract_prompt(
                    platform=app_pretty,
                    month=month,
                    year=year,
                    search_results=search_results[:8000],
                    n_trends=_TRENDS_PER_BUCKET,
                )
                resp = llm_client.query_llm(extract_prompt)
                parsed = extract_json_from_response(resp)
                if not isinstance(parsed, list):
                    parsed = parsed.get("trends") if isinstance(parsed, dict) else None
                if not isinstance(parsed, list):
                    if verbose:
                        print(f"[feed_posts/{app}] trend extraction returned non-list for {cache_key}")
                    continue
                trends = [t for t in parsed if isinstance(t, dict) and t.get("label")][:_TRENDS_PER_BUCKET]
            except Exception as exc:
                if verbose:
                    print(f"[feed_posts/{app}] trend extraction failed for {cache_key}: {exc}")
                continue
            cache[cache_key] = {"trends": trends, "searched_at": dt.datetime.now(dt.timezone.utc).isoformat()}
            _save_cache(backend_dir, user_id, cache)

        # Synthesize 1 post per trend; mix relevant/irrelevant.
        n_relevant = max(1, int(round(len(trends) * _RELEVANT_FRACTION)))
        for i, trend in enumerate(trends):
            polarity = "relevant" if i < n_relevant else "irrelevant"
            tag_pool = pos_tags if polarity == "relevant" else neg_tags
            primary_tag = rng.choice(tag_pool) if tag_pool else trend.get("label", "")

            try:
                synth_prompt = prompts.trending_feed_post_prompt(
                    platform=app_pretty,
                    trend_label=trend.get("label", ""),
                    trend_description=trend.get("description", ""),
                    primary_hashtag=primary_tag,
                    polarity=polarity,
                )
                resp = llm_client.query_llm(synth_prompt)
                post = extract_json_from_response(resp) or {}
            except Exception as exc:
                if verbose:
                    print(f"[feed_posts/{app}] post synth failed for {trend.get('label')}: {exc}")
                continue
            if not isinstance(post, dict) or not post.get("caption"):
                continue

            # Anchor the post timestamp inside this bucket's month, but at
            # least a few hours away from any same-app existing event to
            # avoid duplicate-ts collisions when sorting.
            ts = int(anchor_ts) + rng.randint(-12 * 3600, 12 * 3600)
            ts = max(min(ts, hi_ts), lo_ts)

            hashtags = post.get("hashtags") or [primary_tag]
            if primary_tag and primary_tag not in hashtags:
                hashtags = [primary_tag] + list(hashtags)

            evt = {
                "source_object_id": _new_oid(),
                "source_timestamp": ts,
                "formatted_timestamp": _format_ts(ts),
                "source_hashtags": hashtags[:6],
                "source_interaction_type": "feed_visible",
                "author_id": "public_creator",
                "recipient_id": "",
                "relationship": "public",
                "is_self_authored": False,
                "is_dm": False,
                "is_trending": True,
                "trending_topic": trend.get("label", ""),
                "trending_relevance": polarity,
                "trending_primary_hashtag": primary_tag,
                "interaction_format": {
                    "app": app_pretty,
                    "action": FEED_VISIBLE_ACTION[app]["action"],
                    "action_label": FEED_VISIBLE_ACTION[app]["label"],
                    "user_message": None,
                },
                "event_location": None,
                "content_type": post.get("content_type", "image" if app == "instagram" else "text"),
                "content": {
                    "title": post.get("title", ""),
                    "caption": post.get("caption", ""),
                    "overall_description": post.get("overall_description", ""),
                    "trending_topic": trend.get("label", ""),
                },
                "preferences": [],
                "_feed_react_meta": {
                    "kind": "trending_feed",
                    "trending_topic": trend.get("label", ""),
                    "trend_description": trend.get("description", ""),
                    "primary_hashtag": primary_tag,
                    "relevance": polarity,
                    "search_query": f"top {app_pretty} trending topics {month} {year}",
                },
            }
            out.append(evt)

    if verbose:
        print(f"[feed_posts/{app}] trending posts generated: {len(out)}")
    return out


# ---------------------------------------------------------------------------
# Top-level orchestrator (called from Step 28 in persona_agent.py)
# ---------------------------------------------------------------------------

def generate_feed_posts(
    user_id: str,
    backend_dir: str | Path,
    llm_client,
    web_search_fn: Optional[Callable[[str], str]] = None,
    rng_seed: int = 0,
    verbose: bool = True,
) -> dict:
    """For each social app, generate friend self-posts + trending posts and
    append them to backend/{user_id}/{app}.json (re-sorted by ts).

    Returns a small report dict for logging.
    """
    base = Path(backend_dir) / user_id
    if not base.exists():
        raise SystemExit(f"backend directory does not exist: {base}")

    profile_path = base / "profile.json"
    if not profile_path.exists():
        raise SystemExit(f"profile.json missing at {profile_path}")
    profile = json.loads(profile_path.read_text())

    report: dict = {"user_id": user_id, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}

    for i, app in enumerate(SOCIAL_APPS):
        app_path = base / f"{app}.json"
        if not app_path.exists():
            if verbose:
                print(f"[feed_posts] {app}.json missing — skipping.")
            continue
        existing = json.loads(app_path.read_text())

        friend_posts = generate_friend_self_posts(
            user_id, app, profile, existing, llm_client,
            rng_seed=rng_seed + i, verbose=verbose,
        )
        trending_posts = generate_trending_posts(
            user_id, app, profile, existing, llm_client, web_search_fn,
            backend_dir=backend_dir, rng_seed=rng_seed + 100 + i, verbose=verbose,
        )

        new_events = friend_posts + trending_posts
        report[f"friend_posts_{app}"] = len(friend_posts)
        report[f"trending_posts_{app}"] = len(trending_posts)
        if not new_events:
            continue

        # Drop any previously-generated feed_visible events from this script
        # (so reruns are idempotent — no duplicate accumulation).
        kept = [e for e in existing if e.get("source_interaction_type") != "feed_visible"]
        merged = kept + new_events
        merged.sort(key=lambda x: int(x.get("source_timestamp", 0)))
        app_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
        if verbose:
            print(f"[feed_posts/{app}] wrote {len(new_events)} feed_visible events; "
                  f"app file now has {len(merged)} total events.")

    if verbose:
        print(f"[feed_posts] report: {report}")
    return report
