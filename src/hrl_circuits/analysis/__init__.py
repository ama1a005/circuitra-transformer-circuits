"""
Analysis package for HRL Circuit Localization.

Provides:
- tables      : generate publication-ready pandas DataFrames from result files
- plots       : generate matplotlib figures from result files
- FinalReport : orchestrates all table / plot generation into a structured report
"""

from .tables import (
    generate_main_mib_table,
    generate_interpbench_table,
    generate_ablation_table,
    generate_per_seed_table,
    generate_compute_table,
)
from .plots import (
    plot_quality_vs_cost,
    plot_quality_vs_sparsity,
    plot_per_seed_comparison,
    plot_ablation_summary,
)
from .final_report import generate_final_report

__all__ = [
    "generate_main_mib_table",
    "generate_interpbench_table",
    "generate_ablation_table",
    "generate_per_seed_table",
    "generate_compute_table",
    "plot_quality_vs_cost",
    "plot_quality_vs_sparsity",
    "plot_per_seed_comparison",
    "plot_ablation_summary",
    "generate_final_report",
]
