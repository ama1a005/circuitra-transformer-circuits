"""analysis/tables.py
=====================
Generate comparison tables from experiment results.
All functions read from DataFrames -- no hardcoded numbers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import pandas as pd
    import numpy as np
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False
    raise ImportError("pandas and numpy required for analysis: pip install pandas numpy")


def _require_cols(df: "pd.DataFrame", cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")


def generate_main_table(results_df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Main comparison table: per-method faithfulness, sparsity, and eval cost.
    Returns a DataFrame with one row per method.
    """
    _require_cols(results_df, ["method", "faithfulness", "sparsity"])

    agg = results_df.groupby("method").agg(
        n=("faithfulness", "count"),
        faithfulness_mean=("faithfulness", "mean"),
        faithfulness_std=("faithfulness", "std"),
        sparsity_mean=("sparsity", "mean"),
        sparsity_std=("sparsity", "std"),
    )
    if "eval_calls_used" in results_df.columns:
        agg["eval_calls_mean"] = results_df.groupby("method")["eval_calls_used"].mean()
        agg["eval_calls_std"] = results_df.groupby("method")["eval_calls_used"].std()
    if "wall_clock_seconds" in results_df.columns:
        agg["wall_clock_mean"] = results_df.groupby("method")["wall_clock_seconds"].mean()

    # Round for readability
    for col in agg.columns:
        if agg[col].dtype == float:
            agg[col] = agg[col].round(4)
    return agg.reset_index()


def generate_per_seed_table(results_df: "pd.DataFrame") -> "pd.DataFrame":
    """Per-seed breakdown sorted by method and seed."""
    _require_cols(results_df, ["method", "faithfulness"])
    cols = ["method", "seed", "faithfulness", "sparsity", "eval_calls_used", "status"]
    cols = [c for c in cols if c in results_df.columns]
    sort_by = [c for c in ["method", "seed"] if c in results_df.columns]
    return results_df[cols].sort_values(sort_by).reset_index(drop=True)


def generate_ablation_table(results_df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Ablation study table. Assumes method names contain 'ablation' or
    results_df is pre-filtered to ablation runs.
    """
    if "method" not in results_df.columns:
        return pd.DataFrame()
    ablation_mask = results_df["method"].str.contains("ablation", case=False, na=False)
    ablation_df = results_df[ablation_mask]
    if ablation_df.empty:
        # If no "ablation" tag, return main table for all methods
        return generate_main_table(results_df)
    return generate_main_table(ablation_df)


def generate_compute_table(results_df: "pd.DataFrame") -> "pd.DataFrame":
    """Compute-cost breakdown: eval calls and wall-clock time per method."""
    _require_cols(results_df, ["method"])
    agg_cols = {}
    if "eval_calls_used" in results_df.columns:
        agg_cols["eval_calls_mean"] = ("eval_calls_used", "mean")
        agg_cols["eval_calls_std"] = ("eval_calls_used", "std")
        agg_cols["eval_calls_max"] = ("eval_calls_used", "max")
    if "wall_clock_seconds" in results_df.columns:
        agg_cols["wall_clock_mean"] = ("wall_clock_seconds", "mean")
        agg_cols["wall_clock_std"] = ("wall_clock_seconds", "std")
    if not agg_cols:
        return pd.DataFrame(columns=["method"])
    agg = results_df.groupby("method").agg(**agg_cols).round(3).reset_index()
    return agg


def generate_statistical_tests_table(results_df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Paired permutation test comparing each method vs baseline ('random').
    Returns DataFrame with p-values and effect sizes.
    """
    _require_cols(results_df, ["method", "faithfulness"])
    methods = [m for m in results_df["method"].unique() if m != "random"]
    baseline = results_df[results_df["method"] == "random"]["faithfulness"].values

    rows = []
    for method in methods:
        vals = results_df[results_df["method"] == method]["faithfulness"].values
        if len(vals) == 0 or len(baseline) == 0:
            continue
        # Simple permutation test
        observed_diff = vals.mean() - baseline.mean()
        combined = np.concatenate([vals, baseline])
        n_a = len(vals)
        rng = np.random.default_rng(42)
        count_extreme = 0
        n_perm = 5000
        for _ in range(n_perm):
            perm = rng.permutation(combined)
            diff = perm[:n_a].mean() - perm[n_a:].mean()
            if abs(diff) >= abs(observed_diff):
                count_extreme += 1
        p_value = count_extreme / n_perm
        rows.append({
            "method": method,
            "vs_baseline": "random",
            "observed_diff": round(observed_diff, 6),
            "p_value": round(p_value, 4),
            "significant_p05": p_value < 0.05,
        })
    return pd.DataFrame(rows)


def save_table(df: "pd.DataFrame", path: str, fmt: str = "csv") -> None:
    """Save a table to file. fmt can be 'csv' or 'latex'."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if fmt == "latex":
        df.to_latex(path, index=False, float_format="%.4f")
    else:
        df.to_csv(path, index=False)
    print(f"[Tables] Saved {fmt} table: {path}")
