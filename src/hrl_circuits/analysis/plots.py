"""analysis/plots.py
====================
Generate publication-quality plots from experiment results.
Uses matplotlib with the non-interactive Agg backend.
All functions accept DataFrames and save to file.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must be before importing pyplot
import matplotlib.pyplot as plt

try:
    import numpy as np
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_faithfulness_by_method(
    results_df: "pd.DataFrame",
    output_path: str,
    title: str = "Faithfulness by Method",
) -> None:
    """Bar chart of mean faithfulness per method with std error bars."""
    grouped = results_df.groupby("method")["faithfulness"].agg(["mean", "std"])
    methods = grouped.index.tolist()
    means = grouped["mean"].values
    stds = grouped["std"].fillna(0).values

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.2), 5))
    x = range(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color="steelblue", ecolor="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Faithfulness (higher is better)")
    ax.set_title(title)
    ax.set_ylim(0, min(1.15, max(means) * 1.25 + 0.05))
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.4, label="Perfect faithfulness")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved: {output_path}")


def plot_sparsity_by_method(
    results_df: "pd.DataFrame",
    output_path: str,
    title: str = "Sparsity by Method",
) -> None:
    """Bar chart of mean sparsity per method (higher = more edges pruned)."""
    grouped = results_df.groupby("method")["sparsity"].agg(["mean", "std"])
    methods = grouped.index.tolist()
    means = grouped["mean"].values
    stds = grouped["std"].fillna(0).values

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.2), 5))
    x = range(len(methods))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color="darkorange", ecolor="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Sparsity (fraction pruned)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved: {output_path}")


def plot_pareto_frontier(
    results_df: "pd.DataFrame",
    output_path: str,
    title: str = "Faithfulness vs Sparsity",
) -> None:
    """Scatter plot: faithfulness vs sparsity, colored by method."""
    import numpy as np
    methods = results_df["method"].unique()
    colors = plt.cm.tab10(range(len(methods)))

    fig, ax = plt.subplots(figsize=(7, 6))
    for method, color in zip(methods, colors):
        sub = results_df[results_df["method"] == method]
        ax.scatter(
            sub["sparsity"], sub["faithfulness"],
            label=method, color=color, alpha=0.75, s=60, edgecolors="k", linewidths=0.5,
        )
    ax.set_xlabel("Sparsity (fraction pruned, higher = smaller circuit)")
    ax.set_ylabel("Faithfulness (higher is better)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.1)
    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved: {output_path}")


def plot_per_seed_comparison(
    results_df: "pd.DataFrame",
    output_path: str,
    title: str = "Per-Seed Faithfulness",
) -> None:
    """Connected dot plot: faithfulness per seed, one line per method."""
    methods = results_df["method"].unique()
    colors = plt.cm.tab10(range(len(methods)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, color in zip(methods, colors):
        sub = results_df[results_df["method"] == method].sort_values("seed")
        if "seed" not in sub.columns:
            continue
        ax.plot(
            sub["seed"], sub["faithfulness"],
            "o-", label=method, color=color, alpha=0.85,
            markersize=7, linewidth=1.5,
        )
    ax.set_xlabel("Seed")
    ax.set_ylabel("Faithfulness")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved: {output_path}")


def plot_eval_cost_comparison(
    results_df: "pd.DataFrame",
    output_path: str,
    title: str = "Evaluation Cost by Method",
) -> None:
    """Bar chart of mean evaluation calls used per method."""
    if "eval_calls_used" not in results_df.columns:
        print("[Plots] Skipping eval cost plot: no eval_calls_used column")
        return
    grouped = results_df.groupby("method")["eval_calls_used"].agg(["mean", "std"])
    methods = grouped.index.tolist()
    means = grouped["mean"].values
    stds = grouped["std"].fillna(0).values

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.2), 5))
    x = range(len(methods))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color="mediumseagreen", ecolor="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Eval calls used (lower is cheaper)")
    ax.set_title(title)
    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plots] Saved: {output_path}")


def generate_all_plots(results_df: "pd.DataFrame", plots_dir: str) -> None:
    """Generate all standard plots into plots_dir."""
    p = Path(plots_dir)
    p.mkdir(parents=True, exist_ok=True)
    plot_faithfulness_by_method(results_df, str(p / "faithfulness_by_method.png"))
    plot_sparsity_by_method(results_df, str(p / "sparsity_by_method.png"))
    plot_pareto_frontier(results_df, str(p / "pareto_frontier.png"))
    if "seed" in results_df.columns:
        plot_per_seed_comparison(results_df, str(p / "per_seed_faithfulness.png"))
    plot_eval_cost_comparison(results_df, str(p / "eval_cost.png"))
    print(f"[Plots] All plots saved to {plots_dir}")
