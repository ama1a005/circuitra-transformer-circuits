"""
Result aggregation across seeds and methods.

:class:`ResultAggregator` reads per-run JSON result files from a directory
tree and produces tidy DataFrames with mean, std, and 95 % CI across seeds.
No values are ever hardcoded — everything is derived from files on disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Aggregate experiment results from a directory of JSON files.

    Directory layout expected::

        results_dir/
            run_A_seed0/result.json
            run_A_seed1/result.json
            run_B_seed0/result.json
            ...

    Each ``result.json`` must contain at least:

    .. code-block:: json

        {
            "method": "flat_ppo",
            "seed": 0,
            "metrics": { "cpr": 0.72, "cmd": 0.15, ... }
        }

    Additional top-level keys are passed through and become DataFrame columns.
    """

    # Keys that are lifted from the ``metrics`` sub-dict into top-level columns.
    METRIC_KEYS = ["cpr", "cmd", "faithfulness", "sparsity", "n_forward_calls",
                   "wall_time_seconds", "edge_auroc", "behavioral_metric"]

    def load_run_results(self, results_dir: str) -> pd.DataFrame:
        """
        Discover and load all ``result.json`` files under ``results_dir``.

        Parameters
        ----------
        results_dir : str
            Root directory to search recursively for ``result.json`` files.

        Returns
        -------
        pd.DataFrame
            One row per run.  Columns include ``method``, ``seed``, plus one
            column per metric in :attr:`METRIC_KEYS` (NaN if absent).

        Raises
        ------
        FileNotFoundError
            If ``results_dir`` does not exist.
        ValueError
            If no ``result.json`` files are found.
        """
        root = Path(results_dir)
        if not root.exists():
            raise FileNotFoundError(f"results_dir not found: {results_dir}")

        result_files = sorted(root.rglob("result.json"))
        if not result_files:
            raise ValueError(f"No result.json files found under {results_dir}")

        rows: list[dict] = []
        for fpath in result_files:
            row = self._load_single_result(fpath)
            if row is not None:
                rows.append(row)

        if not rows:
            raise ValueError("All result.json files failed to load.")

        df = pd.DataFrame(rows)
        logger.info(
            "Loaded %d run results from %s", len(df), results_dir
        )
        return df

    def _load_single_result(self, fpath: Path) -> dict | None:
        """Parse a single result.json; return None on error."""
        try:
            with fpath.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s — could not load: %s", fpath, exc)
            return None

        row: dict = {"result_path": str(fpath)}

        # Lift top-level scalar fields.
        for key in ("method", "seed", "task", "model_id", "run_id",
                    "graph_hash", "budget_forward_calls", "git_commit"):
            if key in data:
                row[key] = data[key]

        # Lift metrics sub-dict.
        metrics = data.get("metrics", {})
        for key in self.METRIC_KEYS:
            row[key] = metrics.get(key, np.nan)

        # Any extra metrics not in METRIC_KEYS.
        for k, v in metrics.items():
            if k not in self.METRIC_KEYS:
                row[f"metric_{k}"] = v

        return row

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_by_method(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group ``df`` by ``method`` and compute mean, std, and 95 % CI
        across seeds.

        Parameters
        ----------
        df : pd.DataFrame
            As returned by :meth:`load_run_results`.

        Returns
        -------
        pd.DataFrame
            One row per method.  For each numeric metric ``m``, columns:
            ``m_mean``, ``m_std``, ``m_ci95_low``, ``m_ci95_high``,
            ``m_n``.
        """
        if "method" not in df.columns:
            raise ValueError("DataFrame must have a 'method' column.")

        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != "seed"
        ]

        agg_rows: list[dict] = []
        for method, group in df.groupby("method"):
            row: dict = {"method": method, "n_seeds": len(group)}
            for col in numeric_cols:
                vals = group[col].dropna().values
                if len(vals) == 0:
                    row[f"{col}_mean"] = np.nan
                    row[f"{col}_std"] = np.nan
                    row[f"{col}_ci95_low"] = np.nan
                    row[f"{col}_ci95_high"] = np.nan
                    row[f"{col}_n"] = 0
                    continue

                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

                # 95 % t-interval (exact for small n, converges to z for large n).
                if len(vals) >= 2:
                    se = std / np.sqrt(len(vals))
                    t_crit = float(stats.t.ppf(0.975, df=len(vals) - 1))
                    ci_low = mean - t_crit * se
                    ci_high = mean + t_crit * se
                else:
                    ci_low = ci_high = mean

                row[f"{col}_mean"] = mean
                row[f"{col}_std"] = std
                row[f"{col}_ci95_low"] = ci_low
                row[f"{col}_ci95_high"] = ci_high
                row[f"{col}_n"] = len(vals)

            agg_rows.append(row)

        return pd.DataFrame(agg_rows)

    # ------------------------------------------------------------------
    # Per-task aggregation
    # ------------------------------------------------------------------

    def aggregate_by_method_and_task(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Like :meth:`aggregate_by_method` but grouped by ``(method, task)``.
        """
        if "task" not in df.columns:
            logger.warning("No 'task' column — falling back to aggregate_by_method.")
            return self.aggregate_by_method(df)

        # Temporarily combine keys, aggregate, then restore.
        df = df.copy()
        df["_group"] = df["method"] + "||" + df["task"].astype(str)
        agg = self.aggregate_by_method(df.rename(columns={"_group": "method"}))
        agg[["method", "task"]] = agg["method"].str.split("||", expand=True)
        return agg
