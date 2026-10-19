"""analysis/__init__.py"""
from hrl_circuits.analysis.tables import (
    generate_main_table,
    generate_per_seed_table,
    generate_ablation_table,
    generate_compute_table,
    generate_statistical_tests_table,
    save_table,
)
from hrl_circuits.analysis.plots import (
    plot_faithfulness_by_method,
    plot_sparsity_by_method,
    plot_pareto_frontier,
    plot_per_seed_comparison,
    plot_eval_cost_comparison,
    generate_all_plots,
)
from hrl_circuits.analysis.final_report import FinalReport, generate_final_report, load_results_from_dir

__all__ = [
    "generate_main_table",
    "generate_per_seed_table",
    "generate_ablation_table",
    "generate_compute_table",
    "generate_statistical_tests_table",
    "save_table",
    "plot_faithfulness_by_method",
    "plot_sparsity_by_method",
    "plot_pareto_frontier",
    "plot_per_seed_comparison",
    "plot_eval_cost_comparison",
    "generate_all_plots",
    "FinalReport",
    "generate_final_report",
    "load_results_from_dir",
]
