"""
MIB metrics computation module.

Provides :class:`MIBMetrics` and helpers to compute and tabulate the
standard MIB benchmark metrics (CPR, CMD, task-specific score) from a
saved circuit artifact.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MIBMetrics:
    """
    Container for all MIB evaluation metrics for a single circuit/run.

    Attributes
    ----------
    cpr : float
        Circuit Precision–Recall — harmonic mean of edge precision and recall
        against the ground-truth circuit (where available).
    cmd : float
        Circuit Minimum Description length proxy — lower is sparser / better.
    task_specific : dict[str, float]
        Task-specific behavioural scores keyed by metric name
        (e.g. ``{"logit_diff": 0.82, "accuracy": 0.91}``).
    benchmark_revision : str
        Git commit hash / version tag of the MIB benchmark used for eval.
    circuit_artifact : str
        Absolute path to the evaluated circuit mask file.
    """

    cpr: float = float("nan")
    cmd: float = float("nan")
    task_specific: dict[str, float] = field(default_factory=dict)
    benchmark_revision: str = "unknown"
    circuit_artifact: str = ""

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a plain dict suitable for DataFrame construction."""
        row: dict[str, Any] = {
            "cpr": self.cpr,
            "cmd": self.cmd,
            "benchmark_revision": self.benchmark_revision,
            "circuit_artifact": self.circuit_artifact,
        }
        for k, v in self.task_specific.items():
            row[f"task_{k}"] = v
        return row


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_mib_metrics(
    circuit_path: str,
    benchmark: Any,
    split: str,
) -> MIBMetrics:
    """
    Evaluate a saved circuit against the MIB benchmark and return metrics.

    Parameters
    ----------
    circuit_path : str
        Path to the circuit mask file (as written by
        :func:`~hrl_circuits.benchmark.circuit_io.save_mask`).
    benchmark : CircuitBenchmark
        Adapter instance (e.g. ``MIBAdapter``).
    split : str
        Dataset split to evaluate on (``"train"``, ``"validation"``,
        ``"public_test"``).

    Returns
    -------
    MIBMetrics
        Populated metrics dataclass.

    Notes
    -----
    *   The function delegates the heavy lifting to
        ``benchmark.evaluate_exported_circuit``, which must return a
        ``dict[str, float]`` with at least the keys ``"cpr"`` and ``"cmd"``.
    *   If the benchmark cannot provide those keys, ``NaN`` is stored and a
        warning is emitted.
    """
    circuit_path_obj = Path(circuit_path)
    if not circuit_path_obj.exists():
        raise FileNotFoundError(f"Circuit file not found: {circuit_path}")

    logger.info("Evaluating circuit %s on split '%s'", circuit_path, split)

    # Delegate to the benchmark adapter.
    raw: dict[str, float] = benchmark.evaluate_exported_circuit(circuit_path, split)

    cpr = float(raw.get("cpr", float("nan")))
    cmd = float(raw.get("cmd", float("nan")))

    # Every key that is not 'cpr' or 'cmd' is treated as task-specific.
    task_specific = {
        k: float(v)
        for k, v in raw.items()
        if k not in {"cpr", "cmd", "benchmark_revision"}
    }

    benchmark_revision = str(raw.get("benchmark_revision", "unknown"))

    return MIBMetrics(
        cpr=cpr,
        cmd=cmd,
        task_specific=task_specific,
        benchmark_revision=benchmark_revision,
        circuit_artifact=str(circuit_path_obj.resolve()),
    )


# ---------------------------------------------------------------------------
# Tabulation
# ---------------------------------------------------------------------------

def format_metrics_table(results: list[MIBMetrics]) -> pd.DataFrame:
    """
    Convert a list of :class:`MIBMetrics` into a tidy ``pd.DataFrame``.

    Parameters
    ----------
    results : list[MIBMetrics]
        One entry per evaluated circuit / run.

    Returns
    -------
    pd.DataFrame
        Columns: ``cpr``, ``cmd``, ``benchmark_revision``,
        ``circuit_artifact``, plus one column per task-specific metric
        (prefixed with ``task_``).  Rows correspond to the input list order.
    """
    if not results:
        return pd.DataFrame()

    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)

    # Ensure standard columns appear first.
    priority_cols = ["cpr", "cmd", "benchmark_revision", "circuit_artifact"]
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[priority_cols + other_cols]

    return df


def aggregate_metrics_table(
    results: list[MIBMetrics],
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Aggregate a list of :class:`MIBMetrics` by computing mean ± std across
    runs.

    Parameters
    ----------
    results : list[MIBMetrics]
        Per-run metrics.
    group_col : str, optional
        If provided, the DataFrame is expected to have this column and will be
        grouped by it before aggregation.  Pass via pre-merged metadata.

    Returns
    -------
    pd.DataFrame with columns ``{metric}_mean`` and ``{metric}_std``.
    """
    df = format_metrics_table(results)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if group_col and group_col in df.columns:
        agg = (
            df.groupby(group_col)[numeric_cols]
            .agg(["mean", "std"])
        )
        agg.columns = ["_".join(c) for c in agg.columns]
        return agg.reset_index()

    agg_row = {f"{c}_mean": df[c].mean() for c in numeric_cols}
    agg_row.update({f"{c}_std": df[c].std() for c in numeric_cols})
    return pd.DataFrame([agg_row])
