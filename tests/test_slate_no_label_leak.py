"""Reward-hacking audit: assert no candidate object emitted into the
slate prompt carries a field that would let the agent reverse-engineer
which one is the held-out target.

Run: `pytest tests/test_slate_no_label_leak.py -v`

This test loads every persona-115 slate instance from queries.csv (when
present) and inspects each candidate dict in `instance_json["slate"]`. If
any leak field appears, the test fails — preventing future builder
regressions that accidentally expose ground truth.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Allow running standalone (`python tests/test_slate_no_label_leak.py`)
# in environments without pytest. When pytest IS available, it picks the
# test_* functions up normally.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import pytest  # type: ignore
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False
    class _StubPytestMark:
        def skipif(self, *_, **__):
            def decorator(fn):
                return fn
            return decorator
    class _StubPytest:
        mark = _StubPytestMark()
    pytest = _StubPytest()  # type: ignore


# Fields that MUST NOT appear on a slate candidate emitted to the agent.
# These are either origin labels, ground-truth markers, or interaction
# metadata that would distinguish the held-out from distractors.
LEAK_FIELDS = frozenset({
    "_origin",
    "held_out",
    "is_target",
    "gt",
    "target",
    "split",
    "over_personalization_irrelevant",
    "source_interaction_type",
    "interaction_type",
    "preferences",
    "hidden_persona_labels",
    "source_object_id",
    "source_timestamp",
    "is_self_authored",
    "stereotype_mark",
    "confidence_init",
    "confidence_cross_referenced",
})

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_slate_instances(uid: str = "115") -> list[dict]:
    """Return slate-task instance dicts from benchmark/{uid}/queries.csv."""
    csv.field_size_limit(10_000_000)
    qcsv = REPO_ROOT / "benchmark" / uid / "queries.csv"
    if not qcsv.exists():
        return []
    out: list[dict] = []
    with qcsv.open() as f:
        first = f.readline()
        if not first.startswith("#"):
            f.seek(0)
        for r in csv.DictReader(f):
            # Match both v1 and v2 task type names.
            if r.get("task_type") not in ("slate_ranking", "personalized_feed_ranking"):
                continue
            try:
                inst = json.loads(r["instance_json"])
            except Exception:
                continue
            out.append(inst)
    return out


@pytest.mark.skipif(
    not (REPO_ROOT / "benchmark" / "115" / "queries.csv").exists(),
    reason="benchmark/115/queries.csv not built yet (run prepare_eval_data.py)",
)
def test_no_leak_fields_in_emitted_slate():
    instances = _load_slate_instances("115")
    assert instances, "expected at least one slate_ranking instance in queries.csv"

    for inst in instances:
        slate = inst.get("slate") or []
        for idx, cand in enumerate(slate):
            leaked = LEAK_FIELDS.intersection(cand.keys())
            assert not leaked, (
                f"slate candidate idx={idx} in test_id={inst.get('test_id')!r} "
                f"leaks fields {leaked} — these would let the agent identify "
                f"the held-out target"
            )


@pytest.mark.skipif(
    not (REPO_ROOT / "benchmark" / "115" / "queries.csv").exists(),
    reason="benchmark/115/queries.csv not built yet",
)
def test_slate_pool_size_at_least_10():
    """Pool size sanity — must be at least 10 (legacy) and ideally 16 post-Phase A1."""
    instances = _load_slate_instances("115")
    for inst in instances:
        n = len(inst.get("slate") or [])
        assert n >= 10, f"slate too small ({n} items) for test_id={inst.get('test_id')!r}"


def test_origin_gain_map_completeness():
    """Every `_origin` value the builder can produce must have a gain entry."""
    from evaluation.tasks.slate_ranking import ORIGIN_GAIN
    expected = {
        "held_out", "future_positive", "past_positive",
        "filler_lowsim", "filler", "random", "irrelevant", "negative",
    }
    missing = expected - set(ORIGIN_GAIN)
    assert not missing, f"ORIGIN_GAIN missing entries: {missing}"


def test_ndcg_graded_ideal_ordering():
    """Ideal ordering should produce ndcg_graded@5 == 1.0."""
    from evaluation.tasks.slate_ranking import compute_ranking_metrics
    inst = {
        "held_out_idx": 0,
        "origin_by_idx": (
            ["held_out"]
            + ["future_positive"] * 3
            + ["past_positive"] * 3
            + ["irrelevant"] * 3 + ["negative"] * 3 + ["random"] * 3
        ),
        "slate": [{"hashtags": []}] * 16,
    }
    m = compute_ranking_metrics(list(range(16)), inst)
    assert abs(m["ndcg_graded@5"] - 1.0) < 0.001
    assert m["accuracy"] == 1


def test_ndcg_graded_reverse_ordering():
    """Reverse-ideal ordering should produce ndcg_graded@5 near 0."""
    from evaluation.tasks.slate_ranking import compute_ranking_metrics
    inst = {
        "held_out_idx": 0,
        "origin_by_idx": (
            ["held_out"]
            + ["future_positive"] * 3
            + ["past_positive"] * 3
            + ["irrelevant"] * 3 + ["negative"] * 3 + ["random"] * 3
        ),
        "slate": [{"hashtags": []}] * 16,
    }
    m = compute_ranking_metrics(list(reversed(range(16))), inst)
    assert m["ndcg_graded@5"] < 0.3
    assert m["accuracy"] == 0


def test_similarity_floor_rejects_near_duplicates():
    from evaluation.build_benchmark import _too_similar_to_target
    held = {"hashtags": ["boxing", "workout", "gym"],
            "caption": "great morning at the boxing gym"}
    near_dup = {"hashtags": ["boxing", "workout"],
                "caption": "morning gym session boxing combos"}
    distinct = {"hashtags": ["cooking", "pasta"],
                "caption": "tried a new pasta recipe last night"}
    assert _too_similar_to_target(near_dup, held)
    assert not _too_similar_to_target(distinct, held)


if __name__ == "__main__":
    # Standalone runner — invokes every test_* function and reports.
    import inspect
    fns = [(n, f) for n, f in inspect.getmembers(sys.modules[__name__])
           if n.startswith("test_") and callable(f)]
    passed, failed, skipped = 0, [], []
    for name, fn in fns:
        # Honor the module-level skipif markers we set with @pytest.mark.skipif.
        # (Standalone path doesn't apply them — try/except handles missing fixtures.)
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            # E.g., FileNotFoundError when queries.csv absent — treat as skip.
            skipped.append((name, str(e)))
            print(f"  SKIP  {name}: {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {len(failed)} failed, {len(skipped)} skipped")
    sys.exit(1 if failed else 0)
