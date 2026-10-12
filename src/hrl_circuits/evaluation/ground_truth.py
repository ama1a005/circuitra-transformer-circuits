"""
Ground-truth circuit evaluation: edge-level AUROC and InterpBench results.

This module computes how well a predicted circuit (represented as a mapping
from edge id to importance score) recovers the ground-truth circuit edges
provided by a benchmark.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InterpBenchResult:
    """
    Summary of a single circuit evaluation against an InterpBench ground truth.

    Attributes
    ----------
    edge_auroc : float
        Area under the ROC curve for edge-level circuit recovery.
        1.0 = perfect, 0.5 = random, < 0.5 = worse than random.
    behavioral_metric : float
        Task-specific behavioural metric (e.g. logit-diff faithfulness).
    n_gt_edges : int
        Number of ground-truth edges in the benchmark circuit.
    n_pred_edges : int
        Number of edges with a predicted score > 0 (i.e. considered part of
        the predicted circuit).
    extra : dict
        Any additional fields returned by the benchmark evaluator.
    """

    edge_auroc: float = float("nan")
    behavioral_metric: float = float("nan")
    n_gt_edges: int = 0
    n_pred_edges: int = 0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def compute_edge_auroc(
    predicted_scores: dict[int, float],
    ground_truth_edges: set[int],
    all_edge_ids: list[int],
) -> float:
    """
    Compute the edge-level AUROC for circuit recovery.

    Parameters
    ----------
    predicted_scores : dict[int, float]
        Mapping from edge id to a real-valued importance score.  Higher values
        indicate the edge is more likely to be in the circuit.  Edges absent
        from this dict are treated as having score 0.
    ground_truth_edges : set[int]
        Set of edge ids that form the ground-truth circuit.
    all_edge_ids : list[int]
        Complete list of candidate edge ids in the graph.  Defines the
        universe of edges considered (both positives and negatives).

    Returns
    -------
    float
        AUROC score in [0, 1].  Returns 0.5 if the binary classification
        problem is degenerate (all labels identical).

    Raises
    ------
    ValueError
        If ``all_edge_ids`` is empty.
    ValueError
        If ``ground_truth_edges`` is not a subset of ``all_edge_ids``.

    Notes
    -----
    *   Missing edges in ``predicted_scores`` receive a score of ``0.0``.
    *   The function handles the binary edge case: if every edge in
        ``all_edge_ids`` is a ground-truth edge (or none are), AUROC is
        undefined; 0.5 is returned with a warning.
    """
    if not all_edge_ids:
        raise ValueError("all_edge_ids must not be empty.")

    unknown = ground_truth_edges - set(all_edge_ids)
    if unknown:
        raise ValueError(
            f"{len(unknown)} ground-truth edge(s) not present in all_edge_ids: "
            f"{sorted(unknown)[:5]}{'...' if len(unknown) > 5 else ''}"
        )

    y_true = np.array(
        [1 if eid in ground_truth_edges else 0 for eid in all_edge_ids],
        dtype=np.int32,
    )
    y_score = np.array(
        [float(predicted_scores.get(eid, 0.0)) for eid in all_edge_ids],
        dtype=np.float64,
    )

    # Degenerate case: all labels identical → AUROC undefined.
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        logger.warning(
            "compute_edge_auroc: all labels are %d — AUROC is undefined, "
            "returning 0.5.",
            int(y_true[0]),
        )
        return 0.5

    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError as exc:
        logger.error("roc_auc_score raised: %s — returning 0.5.", exc)
        return 0.5

    return auroc


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_interp_bench_result(
    predicted_scores: dict[int, float],
    ground_truth_edges: set[int],
    all_edge_ids: list[int],
    behavioral_metric: float = float("nan"),
    binarize_threshold: float = 0.0,
    **extra: float,
) -> InterpBenchResult:
    """
    Compute :class:`InterpBenchResult` from raw scores and ground truth.

    Parameters
    ----------
    predicted_scores : dict[int, float]
        Edge id → score.
    ground_truth_edges : set[int]
        Ground-truth positive edge ids.
    all_edge_ids : list[int]
        Full universe of edges.
    behavioral_metric : float
        Pre-computed behavioural metric (e.g. faithfulness).
    binarize_threshold : float
        Scores strictly above this value count toward ``n_pred_edges``.
    **extra
        Additional fields stored in ``InterpBenchResult.extra``.

    Returns
    -------
    InterpBenchResult
    """
    auroc = compute_edge_auroc(predicted_scores, ground_truth_edges, all_edge_ids)

    n_pred = sum(
        1 for eid in all_edge_ids
        if predicted_scores.get(eid, 0.0) > binarize_threshold
    )

    return InterpBenchResult(
        edge_auroc=auroc,
        behavioral_metric=behavioral_metric,
        n_gt_edges=len(ground_truth_edges),
        n_pred_edges=n_pred,
        extra=dict(extra),
    )
