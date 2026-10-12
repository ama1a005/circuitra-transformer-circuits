"""
Trajectory-level metrics: quality-vs-cost curves, cost-at-target queries,
and retained-fraction helpers.

A *trajectory* is a sequence of episode steps stored as a JSONL file, where
each line is a JSON object with at least the fields:

    ``step`` : int          — step index within the episode
    ``cost`` : int          — cumulative forward-pass count at this step
    ``faithfulness`` : float — circuit faithfulness at this step

Additional fields (e.g. ``reward``, ``region_id``) are preserved and
accessible via the returned dicts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_trajectory(path: str) -> list[dict[str, Any]]:
    """
    Load a JSONL trajectory file and return a list of step dicts.

    Parameters
    ----------
    path : str
        Path to a ``.jsonl`` file where each line is a valid JSON object
        representing one trajectory step.

    Returns
    -------
    list[dict]
        List of step records, one per line.  Empty lines are skipped.
        The list preserves the original file order.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    json.JSONDecodeError
        If any non-empty line cannot be parsed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    steps: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"Line {lineno} in {path}: {exc.msg}", exc.doc, exc.pos
                ) from exc

    logger.debug("Loaded %d trajectory steps from %s", len(steps), path)
    return steps


# ---------------------------------------------------------------------------
# Curve construction
# ---------------------------------------------------------------------------

def compute_quality_vs_cost_curve(
    trajectory: list[dict[str, Any]],
) -> tuple[list[int], list[float]]:
    """
    Extract monotone-increasing cost vs. faithfulness pairs from a trajectory.

    Parameters
    ----------
    trajectory : list[dict]
        As returned by :func:`load_trajectory`.  Each dict must contain
        ``"cost"`` (int) and ``"faithfulness"`` (float).

    Returns
    -------
    costs : list[int]
        Sorted list of cumulative forward-pass costs.
    faithfulness : list[float]
        Faithfulness at each cost value.  If multiple steps share the same
        cost, the *last* faithfulness value is used.

    Notes
    -----
    *   Steps with missing ``"cost"`` or ``"faithfulness"`` keys are skipped
        with a warning.
    *   The output lists are always the same length.
    """
    cost_to_faith: dict[int, float] = {}

    for i, step in enumerate(trajectory):
        if "cost" not in step or "faithfulness" not in step:
            logger.warning(
                "Step %d missing 'cost' or 'faithfulness' — skipping.", i
            )
            continue
        cost = int(step["cost"])
        faith = float(step["faithfulness"])
        cost_to_faith[cost] = faith  # last value wins for duplicate costs

    if not cost_to_faith:
        return [], []

    sorted_items = sorted(cost_to_faith.items())
    costs, faithfulness = zip(*sorted_items)
    return list(costs), list(faithfulness)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def compute_cost_at_quality_target(
    trajectory: list[dict[str, Any]],
    target_faithfulness: float,
) -> int | None:
    """
    Find the *first* step at which faithfulness meets or exceeds
    ``target_faithfulness``.

    Parameters
    ----------
    trajectory : list[dict]
        As returned by :func:`load_trajectory`.
    target_faithfulness : float
        Minimum faithfulness threshold (inclusive).

    Returns
    -------
    int or None
        Cumulative forward-pass cost at the first qualifying step, or
        ``None`` if the target is never reached.
    """
    costs, faithfulness = compute_quality_vs_cost_curve(trajectory)
    for cost, faith in zip(costs, faithfulness):
        if faith >= target_faithfulness:
            return cost
    return None


def compute_retained_fraction_at_budget(
    trajectory: list[dict[str, Any]],
    budget: int,
) -> float:
    """
    Return the best faithfulness achieved within ``budget`` forward-pass
    evaluations.

    Parameters
    ----------
    trajectory : list[dict]
        As returned by :func:`load_trajectory`.
    budget : int
        Maximum allowed cumulative forward-pass count.

    Returns
    -------
    float
        Best faithfulness at or below ``budget``.  Returns ``0.0`` if no
        step falls within the budget.
    """
    costs, faithfulness = compute_quality_vs_cost_curve(trajectory)
    best = 0.0
    for cost, faith in zip(costs, faithfulness):
        if cost <= budget:
            best = max(best, faith)
        else:
            break
    return best


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_quality_cost_curve(
    curve: tuple[list[int], list[float]],
    output_path: str,
) -> None:
    """
    Persist a quality-vs-cost curve to a JSONL file.

    Each line contains ``{"cost": int, "faithfulness": float}``.

    Parameters
    ----------
    curve : tuple[list[int], list[float]]
        As returned by :func:`compute_quality_vs_cost_curve`.
    output_path : str
        Destination file path.  Parent directories are created automatically.
    """
    costs, faithfulness = curve
    if len(costs) != len(faithfulness):
        raise ValueError(
            f"costs and faithfulness must be the same length, "
            f"got {len(costs)} vs {len(faithfulness)}."
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as fh:
        for cost, faith in zip(costs, faithfulness):
            fh.write(json.dumps({"cost": cost, "faithfulness": faith}) + "\n")

    logger.info("Saved quality-cost curve (%d points) to %s", len(costs), out)
