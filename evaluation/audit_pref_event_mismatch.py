"""Audit (event, preference) pairings in user's app JSONs for topical mismatches.

The pipeline can produce events whose attached preferences are topically
unrelated to the event's hashtags / content (e.g. a `#smokedhog` BBQ event
carrying a "sour and gummy candy" preference). Root cause: the per-row
LLM call occasionally hallucinates a persona_item that doesn't match the
input row, and downstream canonical-merge inflates the bogus atomic into
a high-confidence canonical that fans out to the wrong events.

This audit uses a two-stage detector that mirrors the cleanup script
(`scripts/clean_existing_personas.py`):

  Stage 1 — token overlap (cheap pre-filter): flag pairs where the
            persona_item's tokens have zero overlap with the event's
            hashtag tokens or content title/caption tokens.
  Stage 2 — canonical-cohort modal hashtag set: for each unique
            persona_item across the 4 app JSONs, collect its top-K most
            frequent hashtags (by row-frequency, excluding the current
            event's contribution). A pair fails Stage 2 if the event's
            hashtags share zero with the canonical's modal top-K.
            This is robust to single-row hallucinations.

Outputs a TSV at /tmp/persona_<uid>_mismatch_audit.tsv plus a per-app /
per-canonical summary on stdout.

Usage:
    python evaluation/audit_pref_event_mismatch.py --user-id 115
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

_APPS = ("instagram", "facebook", "threads", "chatbot")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to",
    "for", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "with", "from", "this",
    "that", "these", "those", "you", "your", "user", "users", "into",
    "about", "around", "over", "under", "via",
}
_MODAL_TOP_K = 5  # how many most-frequent hashtags define a canonical's modal set


def _load_event_lists(user_id: str, repo_root: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for app in _APPS:
        path = repo_root / "backend" / user_id / f"{app}.json"
        if not path.exists():
            continue
        with path.open() as f:
            data = json.load(f)
        if isinstance(data, list):
            out[app] = data
    return out


def _normalize_tag(t: str) -> str:
    return (t or "").lower().lstrip("#").strip()


def _persona_tokens(persona_item: str) -> set[str]:
    """Lowercase alphabetic tokens ≥4 chars from a persona_item, minus stopwords."""
    out: set[str] = set()
    for raw in (persona_item or "").lower().split():
        token = "".join(c for c in raw if c.isalpha())
        if len(token) >= 4 and token not in _STOPWORDS:
            out.add(token)
    return out


def _event_text_tokens(event: dict) -> set[str]:
    parts: list[str] = []
    content = event.get("content") or {}
    parts.append(content.get("title") or "")
    parts.append(content.get("caption") or "")
    parts.append(content.get("overall_description") or "")
    return _persona_tokens(" ".join(parts))


def _tag_set(event: dict) -> set[str]:
    return {_normalize_tag(h) for h in (event.get("source_hashtags") or [])
            if _normalize_tag(h)}


def _build_canonical_cohort(events_by_app: dict[str, list[dict]]) -> dict[str, list[set[str]]]:
    """For each unique persona_item text, list the hashtag sets of every
    event it's attached to (preserving duplicates so frequency counts
    reflect row-frequency)."""
    cohort: dict[str, list[set[str]]] = defaultdict(list)
    for app, events in events_by_app.items():
        for event in events:
            tags = _tag_set(event)
            for pref in (event.get("preferences") or []):
                pi = pref.get("persona_item") or ""
                if not pi:
                    continue
                cohort[pi].append(tags)
    return cohort


def _modal_tags(cohort_tagsets: list[set[str]],
                exclude_idx: int | None = None,
                top_k: int = _MODAL_TOP_K) -> set[str]:
    """Top-K most frequent hashtags across the cohort's tagsets, optionally
    excluding one entry (so a single bogus attachment can't self-justify)."""
    counter: Counter = Counter()
    for i, tagset in enumerate(cohort_tagsets):
        if exclude_idx is not None and i == exclude_idx:
            continue
        for t in tagset:
            counter[t] += 1
    return {t for t, _ in counter.most_common(top_k)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", default="115")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--top-k", type=int, default=_MODAL_TOP_K,
                        help="how many top hashtags define the canonical's modal set")
    parser.add_argument("--out-tsv",
                        help="output TSV path (default: /tmp/persona_<uid>_mismatch_audit.tsv)")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    events_by_app = _load_event_lists(args.user_id, repo_root)
    if not events_by_app:
        raise SystemExit(f"no app JSONs found under {repo_root}/backend/{args.user_id}")

    cohort = _build_canonical_cohort(events_by_app)

    out_tsv = Path(args.out_tsv) if args.out_tsv else \
        Path(f"/tmp/persona_{args.user_id}_mismatch_audit.tsv")

    n_events = sum(len(v) for v in events_by_app.values())
    n_pairs = 0
    n_stage1_flag = 0
    n_stage2_flag = 0
    per_app: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_itype: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_canonical_flagged: Counter = Counter()
    flag_examples: list[dict] = []

    with out_tsv.open("w") as out:
        out.write("app\tsource_object_id\tformatted_timestamp\tsource_interaction_type"
                  "\tsource_hashtags\tpersona_item\tstage1_token_overlap_zero"
                  "\tstage2_modal_overlap_zero\tcanonical_modal_top5\n")

        # Build a per-canonical occurrence index so we can pass exclude_idx
        # when computing modal sets.
        canonical_index: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for app, events in events_by_app.items():
            for event in events:
                oid = event.get("source_object_id", "")
                for pref in (event.get("preferences") or []):
                    pi = pref.get("persona_item") or ""
                    if pi:
                        canonical_index[pi].append((app, oid))

        for app, events in events_by_app.items():
            for event in events:
                event_tags = _tag_set(event)
                event_text = _event_text_tokens(event)
                itype = event.get("source_interaction_type", "")
                oid = event.get("source_object_id", "")
                ts = event.get("formatted_timestamp", "")
                tags_pretty = ",".join(sorted(event_tags))
                per_app[app]["events"] += 1
                per_itype[itype]["events"] += 1
                for pref in (event.get("preferences") or []):
                    pi = pref.get("persona_item") or ""
                    if not pi:
                        continue
                    n_pairs += 1
                    per_app[app]["pairs"] += 1
                    per_itype[itype]["pairs"] += 1

                    # Stage 1: token overlap
                    p_tokens = _persona_tokens(pi)
                    tag_token_set = {t for t in event_tags}  # already lowercase
                    if (p_tokens & tag_token_set) or (p_tokens & event_text):
                        stage1_flag = False
                    else:
                        stage1_flag = True
                        n_stage1_flag += 1
                        per_app[app]["stage1"] += 1
                        per_itype[itype]["stage1"] += 1

                    # Stage 2: modal overlap (only computed for cohort >= 2;
                    # singletons can't generate a meaningful modal set).
                    occurrences = canonical_index.get(pi, [])
                    cohort_tagsets = cohort.get(pi, [])
                    stage2_flag = False
                    modal_tags: set[str] = set()
                    if len(cohort_tagsets) >= 2:
                        # Find this attachment's index in the cohort to exclude it.
                        try:
                            self_idx = next(
                                i for i, (a, o) in enumerate(occurrences)
                                if a == app and o == oid
                            )
                        except StopIteration:
                            self_idx = None
                        modal_tags = _modal_tags(cohort_tagsets, exclude_idx=self_idx,
                                                 top_k=args.top_k)
                        if not (event_tags & modal_tags):
                            stage2_flag = True
                            n_stage2_flag += 1
                            per_app[app]["stage2"] += 1
                            per_itype[itype]["stage2"] += 1
                            per_canonical_flagged[pi] += 1

                    out.write(
                        f"{app}\t{oid}\t{ts}\t{itype}\t{tags_pretty}\t{pi}\t"
                        f"{int(stage1_flag)}\t{int(stage2_flag)}\t"
                        f"{','.join(sorted(modal_tags))}\n"
                    )

                    if stage2_flag and len(flag_examples) < 10:
                        flag_examples.append({
                            "app": app, "oid": oid, "ts": ts,
                            "tags": sorted(event_tags),
                            "persona_item": pi,
                            "modal_tags": sorted(modal_tags),
                        })

    print(f"User: {args.user_id}")
    print(f"Total events:  {n_events}")
    print(f"Total (event, preference) pairs: {n_pairs}")
    print(f"  Stage-1 token-overlap zero:   {n_stage1_flag} ({100*n_stage1_flag/max(n_pairs,1):.1f}%)")
    print(f"  Stage-2 modal-overlap zero:   {n_stage2_flag} ({100*n_stage2_flag/max(n_pairs,1):.1f}%)")
    print(f"\nWritten {out_tsv}")
    print()
    print("Per-app:")
    print(f"  {'app':<12}{'events':>8}{'pairs':>8}{'stage1':>8}{'stage2':>8}")
    for app in _APPS:
        s = per_app.get(app, {})
        print(f"  {app:<12}{s.get('events', 0):>8}{s.get('pairs', 0):>8}"
              f"{s.get('stage1', 0):>8}{s.get('stage2', 0):>8}")
    print()
    print("Per interaction-type:")
    print(f"  {'itype':<24}{'events':>8}{'pairs':>8}{'stage1':>8}{'stage2':>8}")
    for itype, s in sorted(per_itype.items(),
                           key=lambda kv: -kv[1].get("stage2", 0)):
        print(f"  {itype:<24}{s.get('events', 0):>8}{s.get('pairs', 0):>8}"
              f"{s.get('stage1', 0):>8}{s.get('stage2', 0):>8}")
    print()
    print("Top 10 leakiest canonicals (by stage-2 flag count):")
    for pi, c in per_canonical_flagged.most_common(10):
        print(f"  [{c:>3}] {pi[:90]}")
    print()
    print("Concrete stage-2 flagged examples:")
    for e in flag_examples:
        print(f"  [{e['app']}] {e['oid']} ({e['ts']})")
        print(f"    event hashtags : {e['tags']}")
        print(f"    canonical modal: {e['modal_tags']}")
        print(f"    persona_item   : {e['persona_item']}")


if __name__ == "__main__":
    main()
