"""
Permutation / randomisation test for paired comparisons.

Uses an *exact* permutation test when the number of pairs is small (≤ 20),
falling back to a Monte Carlo estimate otherwise.
"""

from __future__ import annotations

import itertools
import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

_EXACT_THRESHOLD = 20
"""Pairs at or below this count use exact enumeration."""


# ---------------------------------------------------------------------------
# Core test
# ---------------------------------------------------------------------------

def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    n_permutations: int = 10_000,
    seed: int = 42,
    direction: str = "two-sided",
) -> dict:
    """
    Paired permutation test for the difference in means between ``a`` and ``b``.

    For each pair ``(a[i], b[i])``, the permutation randomly swaps the two
    values (or not), creating a null distribution for the difference in means.

    Parameters
    ----------
    a : sequence of float
        Per-seed results for method A.
    b : sequence of float
        Per-seed results for method B.
    n_permutations : int
        Number of Monte Carlo permutations (used when n_pairs > 20).
    seed : int
        Random seed for Monte Carlo sampling.
    direction : str
        One of ``"two-sided"``, ``"greater"`` (A > B), ``"less"`` (A < B).

    Returns
    -------
    dict with keys:
        ``"observed_diff"`` — mean(a) − mean(b) on the original data.
        ``"p_value"`` — permutation p-value.
        ``"test_direction"`` — the requested direction.
        ``"n_permutations"`` — actual number of permutations evaluated.
        ``"unit_of_pairing"`` — always ``"seed"``.
        ``"method"`` — ``"exact"`` or ``"monte_carlo"``.
        ``"n_pairs"`` — number of paired observations.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` differ in length, are empty, or ``direction`` is
        invalid.
    """
    valid_directions = {"two-sided", "greater", "less"}
    if direction not in valid_directions:
        raise ValueError(
            f"direction must be one of {valid_directions}, got '{direction}'."
        )
    if len(a) == 0 or len(b) == 0:
        raise ValueError("a and b must not be empty.")
    if len(a) != len(b):
        raise ValueError(
            f"a and b must have the same length, got {len(a)} vs {len(b)}."
        )

    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    n_pairs = len(arr_a)

    observed_diff = float(np.mean(arr_a) - np.mean(arr_b))

    if n_pairs <= _EXACT_THRESHOLD:
        null_diffs, method = _exact_null(arr_a, arr_b)
    else:
        null_diffs, method = _monte_carlo_null(arr_a, arr_b, n_permutations, seed)

    p_value = _compute_p_value(observed_diff, null_diffs, direction)

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "test_direction": direction,
        "n_permutations": len(null_diffs),
        "unit_of_pairing": "seed",
        "method": method,
        "n_pairs": n_pairs,
    }


# ---------------------------------------------------------------------------
# Null distribution helpers
# ---------------------------------------------------------------------------

def _exact_null(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
) -> tuple[np.ndarray, str]:
    """
    Enumerate all 2^n sign-flip permutations and return the null differences.
    """
    n = len(arr_a)
    diffs = arr_a - arr_b  # signed pair differences

    null_diffs: list[float] = []
    for signs in itertools.product([-1, 1], repeat=n):
        signs_arr = np.array(signs, dtype=float)
        null_diffs.append(float(np.mean(signs_arr * diffs)))

    return np.array(null_diffs, dtype=float), "exact"


def _monte_carlo_null(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    n_permutations: int,
    seed: int,
) -> tuple[np.ndarray, str]:
    """
    Sample random sign-flip permutations and return the null differences.
    """
    rng = np.random.default_rng(seed=seed)
    n = len(arr_a)
    diffs = arr_a - arr_b

    null_diffs = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=n, replace=True)
        null_diffs[i] = float(np.mean(signs * diffs))

    return null_diffs, "monte_carlo"


def _compute_p_value(
    observed: float,
    null_diffs: np.ndarray,
    direction: str,
) -> float:
    """Compute the fraction of null diffs at least as extreme as observed."""
    n = len(null_diffs)
    if direction == "two-sided":
        count = int(np.sum(np.abs(null_diffs) >= abs(observed)))
    elif direction == "greater":
        count = int(np.sum(null_diffs >= observed))
    else:  # "less"
        count = int(np.sum(null_diffs <= observed))

    return count / n
