"""Orchestrator for Extension B.

Runs AFTER the main persona pipeline on an existing `backend/{user_id}/`.

Idempotent: rerunning replaces the self-post events, DM files, friends list,
and trending list with fresh generations. Existing pipeline-authored events
are preserved.

CLI:
    python -m data_preparation.extension_b --user_id 115 [--backend_dir backend]
       [--model gpt-5-chat] [--rng_seed 0] [--dry_run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from data_preparation.extension_b.friend_graph import generate_friend_graph
from data_preparation.extension_b.self_posts import generate_self_posts
from data_preparation.extension_b.dm_threads import generate_dm_threads
from data_preparation.extension_b.trending import generate_trending


SOCIAL_APPS = ("instagram", "facebook", "threads")


def run_extension_b(
    user_id: str,
    backend_dir: str | Path,
    llm_client=None,
    rng_seed: int = 0,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """Runs all four Extension B generators on an existing backend. Returns a
    report dict with counts per app and file paths touched.
    """
    base = Path(backend_dir) / user_id
    if not base.exists():
        raise SystemExit(f"backend directory does not exist: {base}")

    # Load profile.
    profile_path = base / "profile.json"
    if not profile_path.exists():
        raise SystemExit(f"profile.json missing at {profile_path}")
    profile = json.loads(profile_path.read_text())

    if verbose:
        print(f"[ext_b] user_id={user_id}  backend={base}  dry_run={dry_run}")

    report: dict = {"user_id": user_id, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat()}

    # 1) Friend graph (one LLM call) — goes into profile.json.friends[].
    if llm_client is not None and not dry_run:
        if verbose: print("[ext_b] generating friend graph …")
        friends = generate_friend_graph(profile, llm_client)
    else:
        # Dry run / no LLM: use fallback graph.
        friends = [
            {"friend_id": "friend_1", "display_name": "Alex Morgan", "relationship_depth": "close", "shared_interests": []},
            {"friend_id": "friend_2", "display_name": "Alex Chen",   "relationship_depth": "acquaintance", "shared_interests": []},
            {"friend_id": "friend_3", "display_name": "Sam Rivera",  "relationship_depth": "distant", "shared_interests": []},
        ]
    report["friends"] = len(friends)

    # 2) Self-posts per social app.
    per_app_events: dict[str, list[dict]] = {}
    for i, app in enumerate(SOCIAL_APPS):
        app_path = base / f"{app}.json"
        if not app_path.exists():
            per_app_events[app] = []
            continue
        existing = json.loads(app_path.read_text())
        if llm_client is not None and not dry_run:
            if verbose: print(f"[ext_b] generating self-posts on {app} …")
            posts = generate_self_posts(user_id, app, profile, existing, llm_client, rng_seed=rng_seed + i)
        else:
            posts = []
        per_app_events[app] = posts
        report[f"self_posts_{app}"] = len(posts)

    # 3) DM threads per social app (needs friend graph).
    dm_threads_per_app: dict[str, list[dict]] = {}
    dm_mirrors_per_app: dict[str, list[dict]] = {}
    for i, app in enumerate(SOCIAL_APPS):
        app_path = base / f"{app}.json"
        if not app_path.exists():
            dm_threads_per_app[app] = []
            dm_mirrors_per_app[app] = []
            continue
        existing = json.loads(app_path.read_text())
        if llm_client is not None and not dry_run:
            if verbose: print(f"[ext_b] generating DM threads on {app} …")
            threads, mirrors = generate_dm_threads(
                user_id, app, profile, friends, existing, llm_client, rng_seed=rng_seed + 100 + i
            )
        else:
            threads, mirrors = [], []
        dm_threads_per_app[app] = threads
        dm_mirrors_per_app[app] = mirrors
        report[f"dm_threads_{app}"] = len(threads)
        report[f"dm_mirror_events_{app}"] = len(mirrors)

    # 4) Trending hashtags — deterministic, no LLM.
    if verbose: print("[ext_b] computing trending hashtags …")
    trending = generate_trending(user_id, backend_dir)
    report["trending_count"] = len(trending.get("hashtags", []))

    if dry_run:
        if verbose:
            print("\n[ext_b] DRY RUN — nothing written. Report:")
            print(json.dumps(report, indent=2))
        return report

    # -- Write-back -----------------------------------------------------------

    # (a) Add friends[] to profile.json.
    profile["friends"] = friends
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2))

    # (b) Append self-posts + DM mirror events to each app JSON (re-sort by ts).
    for app in SOCIAL_APPS:
        app_path = base / f"{app}.json"
        if not app_path.exists():
            continue
        existing = json.loads(app_path.read_text())
        new_events = per_app_events[app] + dm_mirrors_per_app[app]
        # Ensure existing events have the new Extension B fields default-populated
        # (so MCP servers see consistent schema).
        for e in existing:
            e.setdefault("author_id", "public_creator")
            e.setdefault("recipient_id", "")
            e.setdefault("relationship", "public")
            e.setdefault("is_self_authored", False)
            e.setdefault("is_dm", False)
        merged = existing + new_events
        merged.sort(key=lambda x: int(x.get("source_timestamp", 0)))
        app_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2))

    # (c) Write {app}_dms.json per app.
    for app in SOCIAL_APPS:
        dm_path = base / f"{app}_dms.json"
        dm_path.write_text(json.dumps(dm_threads_per_app[app], ensure_ascii=False, indent=2))

    # (d) Write trending.json.
    (base / "trending.json").write_text(json.dumps(trending, ensure_ascii=False, indent=2))

    if verbose:
        print("[ext_b] wrote: profile.json, {app}.json × 3 (appended), {app}_dms.json × 3, trending.json")
        print(f"[ext_b] report: {report}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run Pipeline Extension B post-processing.")
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--backend_dir", default="backend")
    parser.add_argument("--model", default=os.getenv("EXT_B_MODEL", "gpt-5-chat"))
    parser.add_argument("--rate_limit", type=int, default=30)
    parser.add_argument("--rng_seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", help="Don't call LLM, don't write files")
    args = parser.parse_args()

    llm_client = None
    if not args.dry_run:
        from query_llm import QueryLLM
        llm_client = QueryLLM({"models": {"llm_model": args.model}}, rate_limit_per_min=args.rate_limit)

    run_extension_b(
        user_id=args.user_id,
        backend_dir=args.backend_dir,
        llm_client=llm_client,
        rng_seed=args.rng_seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
