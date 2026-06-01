"""Persona-free synthetic smoke test for the `memory` eval mode.

Verifies the implementation correctness of evaluation/memory_builder.py WITHOUT
any real persona data and WITHOUT any real LLM spend: a tiny hand-crafted event
stream + a deterministic FakeLLM stub. Covers the firewall/time-mask cut,
chronological merge, chunking, monotonic accumulation, profile-never-read,
polarity + pinned-negative survival under a forced tiny cap, text-only
consolidation (no embeddings), provider plumbing (temperature/builder model),
and debug-write isolation.

Run: `python tests/test_memory_builder.py`
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.backend_query import BackendQuery
from evaluation.memory_builder import (
    EMPTY_MEMORY,
    build_checkpoints,
    build_global_stream,
    consolidate_evict,
    default_memory_config,
)


# ---------------------------------------------------------------------------
# FakeLLM — deterministic, no network. Echoes each event's unique marker into
# the right section so checkpoints are inspectable. Records temperature/model.
# ---------------------------------------------------------------------------

def _num(marker: str) -> int:
    m = re.search(r"(\d+)", marker)
    return int(m.group(1)) if m else 0


class FakeLLM:
    def __init__(self, model="fake-model"):
        self.model = model
        self.calls: list[dict] = []

    def query_llm(self, prompt, temperature=None, **kwargs):
        self.calls.append({"temperature": temperature, "model": self.model})
        cur = ""
        if "## Current memory" in prompt and "## Rolling summary so far" in prompt:
            cur = prompt.split("## Current memory", 1)[1].split("## Rolling summary so far", 1)[0]
        ev = prompt.split("## New events", 1)[1] if "## New events" in prompt else ""

        pos = set(re.findall(r"EVENT_\d+", cur)) | set(re.findall(r"EVENT_\d+", ev))
        neg = set(re.findall(r"NEG_\d+", cur)) | set(re.findall(r"NEG_\d+", ev))
        stop = set(re.findall(r"STOP_\d+", cur)) | set(re.findall(r"STOP_\d+", ev))

        likes = "\n".join(f"- [topic] {m} liked content · last 04/05" for m in sorted(pos, key=_num)) or "(none yet)"
        dis = [f"- [dislike] {m} disliked content · last 04/05" for m in sorted(neg, key=_num)]
        dis += [f"- {m}: user asked to stop personalizing the feed · last 04/05" for m in sorted(stop, key=_num)]
        dislikes = "\n".join(dis) or "(none yet)"
        mem = (
            "# USER MEMORY (last event seen: fake)\n\n"
            "## Stable interests & likes\n" + likes + "\n\n"
            "## Dislikes & negative signals\n" + dislikes + "\n\n"
            "## Active / short-term threads\n(none yet)\n\n"
            "## Places\n(none yet)\n\n"
            "## People & social context\n(none yet)\n\n"
            "## Communication style\n(none yet)\n\n"
            "## Recent salient events\n(none yet)\n"
        )
        return f"<memory>\n{mem}\n</memory>\n<summary>fake summary</summary>"


# ---------------------------------------------------------------------------
# Synthetic backend fixture (no real personas)
# ---------------------------------------------------------------------------

def _event(oid, ts, app, marker, *, negative=False, stop=False):
    itype = "explicit_negative" if (negative or stop) else "implicit_positive"
    msg = "please stop personalizing my feed" if stop else None
    return {
        "source_object_id": str(oid),
        "source_timestamp": ts,
        "formatted_timestamp": f"12:00, 04/0{(ts % 9) + 1}/2026",
        "source_interaction_type": itype,
        "source_hashtags": [f"#{marker}"],
        "interaction_format": {
            "app": app, "action": "viewed", "action_label": "Viewed",
            "user_message": msg,
        },
        "content": {"title": f"{marker} sample title", "caption": f"caption for {marker}"},
        "content_type": "text",
    }


def _write_fixture(base: Path):
    """~16 events across 2 apps, controlled timestamps straddling 2 boundaries.

    Boundaries chosen in the test: T1=135, T2=1000.
    Events < 135: EVENT_0(100), EVENT_1(110, threads), NEG_2(120), EVENT_3(130), EVENT_9(134).
    Boundary-edge: EVENT_50 at ts EXACTLY 135 must be EXCLUDED at T1 (strict `<`).
    Events >=135: STOP_4(140), EVENT_5..8, NEG_10 — included only at T2.
    """
    uid = "synthU"
    udir = base / uid
    udir.mkdir(parents=True, exist_ok=True)

    ig = [
        _event("o0", 100, "instagram", "EVENT_0"),
        _event("o2", 120, "instagram", "NEG_2", negative=True),
        _event("o3", 130, "instagram", "EVENT_3"),
        _event("o9", 134, "instagram", "EVENT_9"),
        _event("oedge", 135, "instagram", "EVENT_50"),    # exactly at T1 → excluded at T1 (strict <)
        _event("o4", 140, "instagram", "STOP_4", stop=True),
        _event("o5", 200, "instagram", "EVENT_5"),
        _event("o6", 320, "instagram", "EVENT_6"),
        _event("o10", 480, "instagram", "NEG_10", negative=True),
    ]
    th = [
        _event("o1", 110, "threads", "EVENT_1"),
        _event("o7", 210, "threads", "EVENT_7"),
        _event("o8", 900, "threads", "EVENT_8"),
    ]
    (udir / "instagram.json").write_text(json.dumps(ig))
    (udir / "threads.json").write_text(json.dumps(th))
    # firewall sentinel — must NEVER appear in any checkpoint and must NEVER be read
    (udir / "profile.json").write_text(json.dumps({"secret": "SENTINEL_SECRET_XYZ", "name": "DoNotRead"}))
    return uid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_chronological_merge():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        uid = _write_fixture(base)
        bq = BackendQuery(str(base))
        rows = build_global_stream(bq, uid, t_max=10_000)
        ts = [r["t"] for r in rows]
        assert ts == sorted(ts), f"stream not sorted: {ts}"
        # cross-app interleave: threads EVENT_1@110 sits between ig 100 and 120
        markers = [re.search(r"#?(EVENT_\d+|NEG_\d+|STOP_\d+|EVENTEDGE)", str(r.get("hashtags"))).group(1)
                   for r in rows]
        assert markers.index("EVENT_1") == 1, f"interleave wrong: {markers}"
        print("  ✓ chronological_merge")


def test_firewall_time_mask_and_monotonic():
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as rd:
        base = Path(d)
        uid = _write_fixture(base)
        bq = BackendQuery(str(base))
        # Firewall: profile must never be read.
        def _boom(*a, **k):
            raise AssertionError("profile.json was read — firewall breach!")
        bq.get_full_profile = _boom       # type: ignore
        bq.get_profile_summary = _boom    # type: ignore

        fake = FakeLLM(model="builder-x")
        cfg = default_memory_config()
        cfg.update({"chunk_k": 3, "builder_model": "builder-x"})
        ledger = build_checkpoints(bq, uid, [135, 1000], fake, cfg, run_dir=Path(rd))

        t1 = ledger.checkpoints[135]
        t2 = ledger.checkpoints[1000]

        # events strictly < 135 present
        for present in ("EVENT_0", "EVENT_1", "NEG_2", "EVENT_3", "EVENT_9"):
            assert present in t1, f"{present} missing from T1 checkpoint"
        # boundary-edge (ts == 135) and everything >= 135 absent at T1
        for absent in ("EVENT_50", "STOP_4", "EVENT_5", "EVENT_6", "EVENT_7", "EVENT_8", "NEG_10"):
            assert absent not in t1, f"{absent} LEAKED into T1 checkpoint (>= boundary)"
        # T2 sees everything, and is a superset of T1 (monotonic accumulation)
        for present in ("EVENT_0", "EVENT_1", "NEG_2", "EVENT_3", "EVENT_5", "EVENT_8", "STOP_4", "NEG_10", "EVENT_50"):
            assert present in t2, f"{present} missing from T2 checkpoint"
        # sentinel never leaks
        assert "SENTINEL_SECRET_XYZ" not in t1 and "SENTINEL_SECRET_XYZ" not in t2
        # polarity: negatives/stop under Dislikes, not Likes
        likes_t2 = t2.split("## Dislikes")[0]
        assert "NEG_2" not in likes_t2 and "STOP_4" not in likes_t2, "negative leaked into Likes"
        print("  ✓ firewall_time_mask_and_monotonic")


def test_chunking_multiple_calls():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        uid = _write_fixture(base)
        bq = BackendQuery(str(base))
        fake = FakeLLM()
        cfg = default_memory_config()
        cfg.update({"chunk_k": 3})  # 12 events / 3 → several build calls
        build_checkpoints(bq, uid, [1000], fake, cfg)
        assert len(fake.calls) > 1, f"expected multiple chunked build calls, got {len(fake.calls)}"
        print(f"  ✓ chunking_multiple_calls ({len(fake.calls)} build calls)")


def test_provider_plumbing():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        uid = _write_fixture(base)
        bq = BackendQuery(str(base))
        fake = FakeLLM(model="builder-x")
        cfg = default_memory_config()
        cfg.update({"builder_temperature": 0.0, "builder_model": "builder-x"})
        build_checkpoints(bq, uid, [1000], fake, cfg)
        assert fake.calls, "no build calls made"
        assert all(c["temperature"] == 0.0 for c in fake.calls), "builder did not pass temperature=0.0"
        print("  ✓ provider_plumbing (temperature=0.0 honored)")


def test_pinned_negative_survives_tiny_cap():
    # Many evictable likes + one pinned 'stop personalizing' negative + one no-date like.
    likes = "\n".join(f"- [t] EVENT_{i} liked · last 04/0{(i % 9) + 1}" for i in range(12))
    md = (
        "# USER MEMORY\n\n"
        "## Stable interests & likes\n" + likes + "\n- [t] EVENT_NODATE liked\n\n"
        "## Dislikes & negative signals\n"
        "- HARDAVOID: user asked to stop personalizing the feed · last 04/05\n\n"
        "## Active / short-term threads\n(none yet)\n\n"
        "## Places\n(none yet)\n\n## People & social context\n(none yet)\n\n"
        "## Communication style\n(none yet)\n\n## Recent salient events\n(none yet)\n"
    )
    out = consolidate_evict(md, cap=40)  # force aggressive eviction
    assert "stop personalizing" in out, "pinned hard-avoid negative was evicted!"
    assert "EVENT_NODATE" not in out, "no-date (lowest-salience) like should be evicted first"
    # at least some dated likes dropped to fit the tiny cap
    kept_likes = len(re.findall(r"EVENT_\d+", out))
    assert kept_likes < 12, f"expected likes to be evicted under tiny cap, kept {kept_likes}"
    print(f"  ✓ pinned_negative_survives_tiny_cap (kept {kept_likes}/12 likes, negative pinned)")


def test_text_only_invariant():
    src = (Path(__file__).resolve().parents[1] / "evaluation" / "memory_builder.py").read_text()
    # strip comments/docstrings would be ideal; here we forbid actual import/call tokens
    forbidden = ["import faiss", "import chromadb", "import qdrant", "sentence_transformers",
                 "from sklearn", "cosine_similarity", "embeddings.create", ".embed("]
    for tok in forbidden:
        assert tok not in src, f"text-only invariant broken: found {tok!r} in memory_builder.py"
    print("  ✓ text_only_invariant (no embedding/vector imports)")


def test_debug_writes_isolated():
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as rd:
        base = Path(d)
        uid = _write_fixture(base)
        before = {p.name for p in (base / uid).iterdir()}
        bq = BackendQuery(str(base))
        fake = FakeLLM()
        build_checkpoints(bq, uid, [135, 1000], fake, default_memory_config(), run_dir=Path(rd))
        states = list((Path(rd) / "memory_states").glob(f"{uid}_T*.json"))
        assert len(states) == 2, f"expected 2 checkpoint dumps, got {len(states)}"
        after = {p.name for p in (base / uid).iterdir()}
        assert before == after, "backend dir was mutated by the builder!"
        print(f"  ✓ debug_writes_isolated ({len(states)} states under run_dir, backend untouched)")


def test_dry_run_graceful_empty():
    # No checkpoints attached → SnapshotCache memory path returns EMPTY_MEMORY (dry_run safety).
    from evaluation.inference_utils import SnapshotCache
    sc = SnapshotCache(mode="memory")
    text, stats = sc.get_or_build(None, "synthU", 999, None, None)
    assert text == EMPTY_MEMORY and stats.get("memory_mode") is True
    print("  ✓ dry_run_graceful_empty")


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} memory-builder smoke tests (no personas, no real LLM)...")
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {t.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\nFAILED {failed}/{len(tests)}")
        sys.exit(1)
    print(f"\nAll {len(tests)} passed ✓")


if __name__ == "__main__":
    _run_all()
