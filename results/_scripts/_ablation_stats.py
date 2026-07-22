#!/usr/bin/env python3
"""Lightweight agreement statistics for the judge-agreement ablation.

No scipy dependency (only numpy). Implements Pearson r, Spearman rho, and
Krippendorff's alpha for interval data via the coincidence-free pooled
form. Run `python _ablation_stats.py` to self-test alpha against
Krippendorff's canonical worked example (interval alpha = 0.849).
"""
from __future__ import annotations

import math

import numpy as np


def pearson(x, y) -> float | None:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return None
    x, y = x[m], y[m]
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a):
    """Average-rank (ties shared), like scipy.stats.rankdata."""
    a = np.asarray(a, float)
    order = a.argsort()
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average tied ranks
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return sums[inv] / counts[inv]


def spearman(x, y) -> float | None:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return None
    return pearson(_rankdata(x[m]), _rankdata(y[m]))


def krippendorff_alpha_interval(matrix) -> float | None:
    """Krippendorff's alpha (interval metric).

    `matrix`: array-like shape (n_coders, n_units); missing = NaN.
    Uses the pooled identity   Sum_{i<j}(a_i-a_j)^2 = n*Sum(a^2) - (Sum a)^2
    so no dense coincidence matrix is needed.
    """
    M = np.asarray(matrix, float)
    n_units = M.shape[1]

    # Observed disagreement.
    do_num = 0.0
    n_pairable = 0
    pooled = []
    for u in range(n_units):
        vals = M[:, u]
        vals = vals[np.isfinite(vals)]
        mu = len(vals)
        if mu < 2:
            continue
        within = mu * np.sum(vals ** 2) - np.sum(vals) ** 2  # = Sum_{i<j}(vi-vj)^2
        do_num += (2.0 / (mu - 1)) * within
        n_pairable += mu
        pooled.extend(vals.tolist())

    n = n_pairable
    if n < 2:
        return None
    Do = do_num / n

    p = np.asarray(pooled, float)
    total = n * np.sum(p ** 2) - np.sum(p) ** 2  # Sum_{i<j}(pi-pj)^2 over pooled
    De = (2.0 / (n * (n - 1))) * total
    if De == 0:
        return 1.0 if Do == 0 else None
    return float(1.0 - Do / De)


def mean_ci(vals, z: float = 1.96):
    """Mean + normal-approx half-width of the 95% CI."""
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if len(a) == 0:
        return None, None
    if len(a) == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(z * a.std(ddof=1) / math.sqrt(len(a)))


def _selftest():
    # Krippendorff's canonical reliability data (4 coders x 12 units), NaN = missing.
    n = float("nan")
    data = [
        [1, 2, 3, 3, 2, 1, 4, 1, 2, n, n, n],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, n, 3],
        [n, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, n],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, n],
    ]
    a = krippendorff_alpha_interval(data)
    print(f"interval alpha (canonical) = {a:.3f}  (expected ~0.849)")
    assert abs(a - 0.849) < 0.01, a
    # perfect agreement -> 1.0
    assert abs(krippendorff_alpha_interval([[1, 2, 3, 4], [1, 2, 3, 4]]) - 1.0) < 1e-9
    print("self-test OK")


if __name__ == "__main__":
    _selftest()
