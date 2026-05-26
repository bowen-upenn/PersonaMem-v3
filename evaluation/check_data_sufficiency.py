"""Pre-build data-sufficiency gate.

Checks backend/{user_id}/ against the plan's minimum-volume assertions for
Tasks T6–T19. Green = benchmark can be built; red = block with an actionable
error message.

Assertions (from plan Extension D₀):
- ≥ 10 self-authored posts per social app (ceiling → floor 5 if posting_frequency=rarely)
- ≥ 25 inbound DMs across the four apps combined
- ≥ 15 outbound DMs (sent by the user)
- ≥ 3 group-DM threads (each ≥ 3 participants, ≥ 3 messages)
- ≥ 8 named friends
- ≥ 3 hidden_personas with privacy_ratio > 0.7
- ≥ 2 topics (hashtags) appearing on events from ≥ 2 apps
- ≥ 6 trending feed events (is_trending=True) across social app JSONs

e6 assertions (pass `--e6` to enable; class-adaptive):
- mobility_class present on profile
- per-class geo coverage + city-count + trip-arc presence
- calendar mods density + recent cancellation + multi-attendee meeting
- planted chatbot constraint turns present (audit hook, optional)

CLI:
    python -m evaluation.check_data_sufficiency --user_id 115
    python -m evaluation.check_data_sufficiency --user_id 115 --e6
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


APPS = ("instagram", "facebook", "threads", "chatbot")
SOCIAL_APPS = ("instagram", "facebook", "threads")


# e6 thresholds — keep in sync with data_preparation/persona_agent.py.
E6_MOBILITY_CLASS_MIN_GEO_COVERAGE: dict[str, float] = {
    "homebody":      0.20,
    "domestic":      0.30,
    "international": 0.30,
    "nomadic":       0.30,
}
E6_MIN_CALENDAR_MODIFICATIONS: int = 20
E6_RECENT_CANCELLATION_WINDOW_HOURS: int = 6
# Class-adaptive pass criterion: how many of the check rows must pass.
# Homebody users have fewer applicable checks (no trip-arc, no foreign
# geo) so their pass floor is lower.
E6_PASS_FLOOR: dict[str, int] = {
    "homebody":      5,   # of 7 non-geo-travel checks
    "domestic":      7,   # of 9
    "international": 7,   # of 9
    "nomadic":       7,   # of 9
}


def _load(path: Path) -> list | dict:
    if not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def check(user_id: str, backend_dir: str | Path) -> list[tuple[str, bool, str]]:
    """Return a list of (check_name, passed, note) tuples."""
    base = Path(backend_dir) / user_id
    profile = _load(base / "profile.json") or {}

    results: list[tuple[str, bool, str]] = []

    # Self-posts per social app.
    for app in SOCIAL_APPS:
        events = _load(base / f"{app}.json") or []
        self_posts = [e for e in events if e.get("is_self_authored")]
        posting_freq = ((profile.get("app_personas", {}) or {}).get(app.capitalize(), {}) or {}).get("posting_frequency", "weekly")
        floor = 5 if posting_freq == "rarely" else 10
        ok = len(self_posts) >= floor
        results.append((
            f"self_posts_{app}",
            ok,
            f"{len(self_posts)} self-posts ({posting_freq}, floor={floor})",
        ))

    # DM counts across apps. DM threads now live inline in {app}.json as
    # entries with is_dm=true + full messages[]; count messages per sender
    # (each thread carries 1–4 messages).
    inbound = 0
    outbound = 0
    group_threads = 0
    for app in SOCIAL_APPS:
        events = _load(base / f"{app}.json") or []
        for e in events:
            if not e.get("is_dm"):
                continue
            msgs = e.get("messages") or []
            for m in msgs:
                if m.get("sender") == "self":
                    outbound += 1
                else:
                    inbound += 1
            if (e.get("is_group") or e.get("is_group_dm")) \
                    and len(e.get("participants") or []) >= 3 \
                    and len(msgs) >= 3:
                group_threads += 1
    results.append(("inbound_dms_total", inbound >= 25, f"{inbound} inbound DMs (need ≥25)"))
    results.append(("outbound_dms_total", outbound >= 15, f"{outbound} outbound DMs (need ≥15)"))
    results.append(("group_dm_threads", group_threads >= 3, f"{group_threads} group threads (need ≥3)"))

    # Friends graph.
    friends = profile.get("friends") or []
    results.append(("named_friends", len(friends) >= 8, f"{len(friends)} friends (need ≥8)"))

    # Privacy-sensitive hidden personas.
    hidden = profile.get("hidden_personas") or []
    sensitive = [h for h in hidden if (h.get("privacy_ratio") or 0) > 0.7]
    results.append(("sensitive_hidden_personas", len(sensitive) >= 3, f"{len(sensitive)} with privacy_ratio>0.7 (need ≥3)"))

    # Multi-app topics.
    app_hashtag_union: dict[str, set[str]] = {}
    for app in APPS:
        events = _load(base / f"{app}.json") or []
        tags = set()
        for e in events:
            for h in (e.get("source_hashtags") or []):
                tags.add(h.lower())
        app_hashtag_union[app] = tags
    shared: Counter = Counter()
    for a in APPS:
        for b in APPS:
            if a >= b:
                continue
            shared.update(app_hashtag_union[a] & app_hashtag_union[b])
    multi_app_topics = len({h for h, _ in shared.most_common() if shared[h] >= 1})
    results.append(("multi_app_topics", multi_app_topics >= 2, f"{multi_app_topics} hashtags on ≥2 apps (need ≥2)"))

    # Trending events embedded in app JSONs.
    _trending_count = 0
    for _app in APPS[:3]:
        _app_data = _load(base / f"{_app}.json") or []
        if isinstance(_app_data, list):
            _trending_count += sum(1 for _e in _app_data if _e.get("is_trending"))
    results.append(("trending_events", _trending_count >= 6, f"{_trending_count} trending feed events (need ≥6)"))

    return results


def check_e6(user_id: str, backend_dir: str | Path) -> list[tuple[str, bool, str, bool]]:
    """Class-adaptive e6 sufficiency checks.

    Returns a list of (check_name, passed, note, applicable) tuples.
    `applicable=False` means this check is a no-op for the user's mobility
    class (e.g., trip-arc checks don't apply to homebodies) and should be
    excluded from the pass-floor count and not printed as a red ✗.
    """
    base = Path(backend_dir) / user_id
    profile = _load(base / "profile.json") or {}
    mobility_class = (profile.get("mobility_class") or "").strip()

    out: list[tuple[str, bool, str, bool]] = []

    # mobility_class present
    ok = mobility_class in {"homebody", "domestic", "international", "nomadic"}
    out.append(("mobility_class_present", ok,
                f"mobility_class={mobility_class!r}", True))
    if not ok:
        return out

    # Count geo-enriched events across all four apps
    total_events = 0
    geo_events = 0
    distinct_cities: set[str] = set()
    distinct_countries: set[str] = set()
    home_city: str | None = None
    for app in APPS:
        events = _load(base / f"{app}.json") or []
        for e in events:
            total_events += 1
            loc = (e.get("event_location") or {}) if isinstance(e, dict) else {}
            if loc.get("city"):
                geo_events += 1
                distinct_cities.add(loc.get("city"))
                if loc.get("country"):
                    distinct_countries.add(loc.get("country"))
    # Home city = most-frequent city (quick re-scan; fine for this volume).
    if distinct_cities:
        from collections import Counter as _Counter
        city_counts: _Counter = _Counter()
        for app in APPS:
            events = _load(base / f"{app}.json") or []
            for e in events:
                c = ((e.get("event_location") or {}) if isinstance(e, dict) else {}).get("city")
                if c:
                    city_counts[c] += 1
        home_city = city_counts.most_common(1)[0][0] if city_counts else None

    # Geo coverage floor (class-adaptive)
    floor = E6_MOBILITY_CLASS_MIN_GEO_COVERAGE.get(mobility_class, 0.30)
    coverage = geo_events / max(1, total_events)
    out.append((
        "e6_geo_coverage", coverage >= floor,
        f"{geo_events}/{total_events} ({coverage:.0%}, need ≥{int(floor*100)}% for {mobility_class})",
        True,
    ))

    # Geo city count — class-adaptive
    if mobility_class == "homebody":
        out.append((
            "e6_geo_homebody_single_city", len(distinct_cities) == 1,
            f"{len(distinct_cities)} cities (homebody must = 1)",
            True,
        ))
    elif mobility_class == "nomadic":
        out.append((
            "e6_geo_nomadic_multi_city", len(distinct_cities) >= 3,
            f"{len(distinct_cities)} cities (nomadic needs ≥3)",
            True,
        ))
    else:
        out.append((
            "e6_geo_cities_reasonable", 1 <= len(distinct_cities) <= 3,
            f"{len(distinct_cities)} cities",
            True,
        ))

    # Trip arc presence — applicable only to non-homebody
    trip_arcs = profile.get("geo_trip_arcs") or []
    if mobility_class == "homebody":
        out.append((
            "e6_trip_arc_absent", len(trip_arcs) == 0,
            f"{len(trip_arcs)} arcs (homebody expects 0)",
            True,
        ))
    else:
        out.append((
            "e6_trip_arc_present", len(trip_arcs) >= 1,
            f"{len(trip_arcs)} arcs (need ≥1 for {mobility_class})",
            True,
        ))

    # International class needs at least one foreign-locale arc
    if mobility_class == "international":
        has_intl = any(a.get("kind") == "international" for a in trip_arcs)
        out.append((
            "e6_trip_arc_international", has_intl,
            f"international arc present: {has_intl}",
            True,
        ))
    else:
        # Not applicable — still emit but marked so it doesn't count.
        out.append((
            "e6_trip_arc_international", True,
            "(n/a for this class)",
            False,
        ))

    # Calendar density
    calendar = _load(base / "calendar.json") or {}
    mods = calendar.get("modifications", []) if isinstance(calendar, dict) else []
    out.append((
        "e6_calendar_density", len(mods) >= E6_MIN_CALENDAR_MODIFICATIONS,
        f"{len(mods)} mods (need ≥{E6_MIN_CALENDAR_MODIFICATIONS})",
        True,
    ))

    # Required recent cancellation — at least one `removed` mod in the
    # last 6 hours before the latest mod timestamp (using the stream's max
    # ts as the proxy for obs_end_ts since we don't re-derive it here).
    recent_cancel_ok = False
    if mods:
        latest_ts = max((m.get("ts", 0) for m in mods), default=0)
        window_start = latest_ts - E6_RECENT_CANCELLATION_WINDOW_HOURS * 3600
        recent_cancel_ok = any(
            m.get("action") == "removed" and m.get("ts", 0) >= window_start
            for m in mods
        )
    out.append((
        "e6_calendar_recent_cancellation", recent_cancel_ok,
        f"removed-mod in last {E6_RECENT_CANCELLATION_WINDOW_HOURS}h: {recent_cancel_ok}",
        True,
    ))

    # Multi-attendee meeting — relaxed to ≥1 non-self attendee (was ≥2).
    # e6 archetype 4 (audience-shift in chat) only needs ONE named attendee
    # (the person the user is supposedly meeting with).
    multi_attendee_ok = False
    for m in mods:
        if m.get("action") != "added":
            continue
        entry = m.get("entry") or {}
        attendees = entry.get("attendees") or []
        others = [a for a in attendees if str(a).lower() != "self"]
        if len(others) >= 1:
            multi_attendee_ok = True
            break
    out.append((
        "e6_calendar_named_attendee", multi_attendee_ok,
        f"≥1 added entry with ≥1 non-self attendee: {multi_attendee_ok}",
        True,
    ))

    return out


def main():
    parser = argparse.ArgumentParser(description="Check Extension B data sufficiency for benchmark build.")
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--backend_dir", default="backend")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any check fails")
    parser.add_argument("--e6", action="store_true",
                        help="Also run class-adaptive e6 substrate checks "
                             "(mobility_class, geo coverage, calendar diversity)")
    args = parser.parse_args()

    results = check(args.user_id, args.backend_dir)
    green = "\033[92m"; red = "\033[91m"; end = "\033[0m"
    failed = []
    for name, ok, note in results:
        tag = f"{green}✓{end}" if ok else f"{red}✗{end}"
        print(f"  {tag} {name:30s}  {note}")
        if not ok:
            failed.append(name)
    print()
    if failed:
        print(f"{red}{len(failed)} check(s) failed:{end} {', '.join(failed)}")
        print("Re-run the main pipeline to close the gap (Extension B is")
        print("now Step 24 of the merged pipeline — no separate invocation):")
        print(f"  python scripts/run_persona_pipeline.py --user_id {args.user_id}")

    # e6 block — separate pass-count with class-adaptive floor.
    if args.e6:
        print()
        print("e6 substrate checks (class-adaptive):")
        e6_results = check_e6(args.user_id, args.backend_dir)
        applicable = [(n, ok, note) for (n, ok, note, ap) in e6_results if ap]
        non_applicable = [(n, note) for (n, ok, note, ap) in e6_results if not ap]
        e6_passed = 0
        for name, ok, note in applicable:
            tag = f"{green}✓{end}" if ok else f"{red}✗{end}"
            print(f"  {tag} {name:34s}  {note}")
            if ok:
                e6_passed += 1
        for name, note in non_applicable:
            print(f"  {green}-{end} {name:34s}  {note}")

        # Class-adaptive pass floor
        profile = _load(Path(args.backend_dir) / args.user_id / "profile.json") or {}
        mobility_class = (profile.get("mobility_class") or "").strip()
        pass_floor = E6_PASS_FLOOR.get(mobility_class, 7)
        e6_ok = e6_passed >= pass_floor
        status = f"{green}PASS{end}" if e6_ok else f"{red}FAIL{end}"
        print()
        print(f"  e6 pass floor: {e6_passed}/{len(applicable)} passed "
              f"(need ≥{pass_floor} for {mobility_class or 'unknown'}) — {status}")
        if not e6_ok and args.strict:
            sys.exit(1)

    if failed and args.strict:
        sys.exit(1)

    if not failed:
        print(f"{green}all data-sufficiency checks passed.{end}")


if __name__ == "__main__":
    main()
