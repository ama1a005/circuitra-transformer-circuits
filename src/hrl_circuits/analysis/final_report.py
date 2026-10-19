"""analysis/final_report.py
============================
Generates the complete final analysis report from a results directory.
Reads all run_result.json files, produces tables, plots, and a README.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
    import numpy as np
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def load_results_from_dir(results_root: str) -> "pd.DataFrame":
    """
    Glob for all run_result.json files under results_root.
    Returns a combined DataFrame with one row per run.
    """
    if not _HAS_PANDAS:
        raise ImportError("pandas required: pip install pandas")
    root = Path(results_root)
    result_files = list(root.rglob("run_result.json"))
    if not result_files:
        # Also try results.json (from train_flat.py output)
        result_files = list(root.rglob("results.json"))
    if not result_files:
        print(f"[FinalReport] No result files found under {results_root}")
        return pd.DataFrame()

    rows = []
    for rf in result_files:
        try:
            data = json.loads(rf.read_text(encoding="utf-8"))
            # Normalize field names
            row = {
                "run_id": data.get("run_id", rf.parent.name),
                "method": data.get("method", "unknown"),
                "seed": data.get("seed", 0),
                "faithfulness": float(data.get("faithfulness",
                                      data.get("final_faithfulness", 0.0))),
                "sparsity": float(data.get("sparsity",
                                  data.get("final_sparsity", 0.0))),
                "eval_calls_used": int(data.get("eval_calls_used", 0)),
                "wall_clock_seconds": float(data.get("wall_clock_seconds", 0.0)),
                "output_dir": str(rf.parent),
                "status": data.get("status", "completed"),
            }
            rows.append(row)
        except Exception as e:
            print(f"[FinalReport] Warning: could not load {rf}: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


class FinalReport:
    """
    Generates a complete analysis report from a results directory.

    Usage:
        report = FinalReport(results_root="results/final")
        report.generate()
    """

    def __init__(self, results_root: str):
        self.results_root = Path(results_root)
        self.results_root.mkdir(parents=True, exist_ok=True)

    def generate(self) -> None:
        """Load all results and generate tables, plots, and README."""
        print(f"[FinalReport] Loading results from: {self.results_root}")
        df = load_results_from_dir(str(self.results_root))

        if df.empty:
            print("[FinalReport] No results found. Creating placeholder report.")
            self._write_empty_readme()
            return

        print(f"[FinalReport] Loaded {len(df)} runs across {df['method'].nunique()} methods")

        # ── Tables ──────────────────────────────────────────────────────────
        from hrl_circuits.analysis.tables import (
            generate_main_table, generate_per_seed_table,
            generate_compute_table, generate_statistical_tests_table,
            save_table,
        )
        tables_dir = self.results_root / "tables"
        tables_dir.mkdir(exist_ok=True)

        main_tbl = generate_main_table(df)
        save_table(main_tbl, str(tables_dir / "main_comparison.csv"))

        per_seed_tbl = generate_per_seed_table(df)
        save_table(per_seed_tbl, str(tables_dir / "per_seed.csv"))

        compute_tbl = generate_compute_table(df)
        save_table(compute_tbl, str(tables_dir / "compute_cost.csv"))

        if "random" in df["method"].values:
            stat_tbl = generate_statistical_tests_table(df)
            save_table(stat_tbl, str(tables_dir / "statistical_tests.csv"))
            stat_tbl.to_json(
                str(self.results_root / "statistical_tests.json"),
                orient="records", indent=2,
            )

        # Also save summary CSV
        df.to_csv(str(self.results_root / "summary.csv"), index=False)

        # ── Plots ────────────────────────────────────────────────────────────
        from hrl_circuits.analysis.plots import generate_all_plots
        plots_dir = self.results_root / "plots"
        generate_all_plots(df, str(plots_dir))

        # ── README ───────────────────────────────────────────────────────────
        self._write_readme(df, main_tbl)
        print(f"[FinalReport] Complete. Results in: {self.results_root}")

    def _write_readme(self, df: "pd.DataFrame", main_tbl: "pd.DataFrame") -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# Final Results Report",
            "",
            f"Generated: {timestamp}",
            "",
            f"**Total runs:** {len(df)}",
            f"**Methods:** {', '.join(sorted(df['method'].unique()))}",
            f"**Seeds:** {sorted(df['seed'].unique()) if 'seed' in df.columns else 'N/A'}",
            "",
            "## Main Comparison Table",
            "",
            main_tbl.to_markdown(index=False) if hasattr(main_tbl, "to_markdown") else str(main_tbl),
            "",
            "## Files",
            "",
            "| File | Description |",
            "|------|-------------|",
            "| `summary.csv` | All runs combined |",
            "| `tables/main_comparison.csv` | Per-method mean faithfulness / sparsity |",
            "| `tables/per_seed.csv` | Per-seed breakdown |",
            "| `tables/compute_cost.csv` | Eval calls and wall-clock time |",
            "| `tables/statistical_tests.csv` | Permutation test p-values |",
            "| `plots/faithfulness_by_method.png` | Faithfulness bar chart |",
            "| `plots/pareto_frontier.png` | Faithfulness vs sparsity scatter |",
            "| `plots/per_seed_faithfulness.png` | Per-seed connected dots |",
            "",
        ]
        readme = self.results_root / "README.md"
        readme.write_text("\n".join(lines), encoding="utf-8")
        print(f"[FinalReport] README written: {readme}")

    def _write_empty_readme(self) -> None:
        readme = self.results_root / "README.md"
        readme.write_text(
            "# Final Results Report\n\nNo results found. Run experiments first.\n",
            encoding="utf-8",
        )


def generate_final_report(results_root: str) -> None:
    """Module-level convenience wrapper."""
    FinalReport(results_root).generate()
