"""
Evaluation package for HRL Circuit Localization.

Provides:
- MIBMetrics / compute_mib_metrics  : MIB competition metrics (CPR, CMD)
- ground_truth                       : AUROC and InterpBench evaluation
- trajectory_metrics                 : quality-vs-cost curve utilities
- bootstrap                          : bootstrap confidence intervals
- permutation                        : paired permutation test
- aggregate                          : result aggregation across seeds / methods
"""

from .mib_metrics import MIBMetrics, compute_mib_metrics, format_metrics_table
from .ground_truth import InterpBenchResult, compute_edge_auroc
from .trajectory_metrics import (
    load_trajectory,
    compute_quality_vs_cost_curve,
    compute_cost_at_quality_target,
    compute_retained_fraction_at_budget,
    save_quality_cost_curve,
)
from .bootstrap import bootstrap_ci, bootstrap_paired_difference_ci
from .permutation import paired_permutation_test
from .aggregate import ResultAggregator

__all__ = [
    "MIBMetrics",
    "compute_mib_metrics",
    "format_metrics_table",
    "InterpBenchResult",
    "compute_edge_auroc",
    "load_trajectory",
    "compute_quality_vs_cost_curve",
    "compute_cost_at_quality_target",
    "compute_retained_fraction_at_budget",
    "save_quality_cost_curve",
    "bootstrap_ci",
    "bootstrap_paired_difference_ci",
    "paired_permutation_test",
    "ResultAggregator",
]
