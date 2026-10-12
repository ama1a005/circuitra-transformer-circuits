"""
Bootstrap confidence interval utilities.

Provides percentile-bootstrap CIs for arbitrary statistics and a paired
difference CI function for comparing two methods across seeds.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-sample bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute a percentile bootstrap confidence interval for ``stat_fn(data)``.

    Parameters
    ----------
    data : sequence of float
        Observed values (one per seed / replicate).
    stat_fn : callable
        Function mapping a 1-D array to a scalar statistic.  Defaults to
        ``np.mean``.
    n_resamples : int
        Number of bootstrap resamples.  Default: 10 000.
    ci : float
        Confidence level in (0, 1).  Default: 0.95.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    low : float
        Lower bound of the CI.
    high : float
        Upper bound of the CI.
    point_estimate : float
        Statistic applied to the original (unresampled) data.

    Raises
    ------
    ValueError
        If ``data`` is empty or ``ci`` is outside (0, 1).
    """
    if len(data) == 0:
        raise ValueError("data must not be empty.")
    if not (0 < ci < 1):
        raise ValueError(f"ci must be in (0, 1), got {ci}.")

    arr = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed=seed)

    point_estimate = float(stat_fn(arr))

    # Draw bootstrap samples.
    resampled_stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        resampled_stats[i] = stat_fn(sample)

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(resampled_stats, alpha * 100))
    high = float(np.percentile(resampled_stats, (1.0 - alpha) * 100))

    return low, high, point_estimate


# ---------------------------------------------------------------------------
# Paired difference bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_paired_difference_ci(
    a: Sequence[float],
    b: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Bootstrap CI for the paired difference ``stat_fn(a) - stat_fn(b)``.

    Pairing is by index — element ``a[i]`` is compared to ``b[i]``.  Resampling
    is performed over paired indices so that each resample respects the
    per-seed pairing structure.

    Parameters
    ----------
    a : sequence of float
        Results for method A (one per seed).
    b : sequence of float
        Results for method B (one per seed).
    stat_fn : callable
        Statistic to apply to each method's values.  Defaults to ``np.mean``.
    n_resamples : int
        Number of bootstrap resamples.
    ci : float
        Confidence level.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        ``"diff"`` — point estimate of ``stat_fn(a) - stat_fn(b)``.
        ``"ci_low"`` — lower CI bound.
        ``"ci_high"`` — upper CI bound.
        ``"ci_level"`` — the requested ``ci`` value.
        ``"n_pairs"`` — number of paired observations.
        ``"stat_fn"`` — name of ``stat_fn``.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different lengths or are empty.
    """
    if len(a) == 0 or len(b) == 0:
        raise ValueError("a and b must not be empty.")
    if len(a) != len(b):
        raise ValueError(
            f"a and b must have the same length, got {len(a)} vs {len(b)}."
        )
    if not (0 < ci < 1):
        raise ValueError(f"ci must be in (0, 1), got {ci}.")

    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    n = len(arr_a)
    rng = np.random.default_rng(seed=seed)

    point_diff = float(stat_fn(arr_a) - stat_fn(arr_b))

    boot_diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs[i] = stat_fn(arr_a[idx]) - stat_fn(arr_b[idx])

    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.percentile(boot_diffs, alpha * 100))
    ci_high = float(np.percentile(boot_diffs, (1.0 - alpha) * 100))

    return {
        "diff": point_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": ci,
        "n_pairs": n,
        "stat_fn": getattr(stat_fn, "__name__", str(stat_fn)),
    }
