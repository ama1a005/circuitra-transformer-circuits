"""experiments/registry.py
==========================
CSV-based experiment registry that records all runs.
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


_COLUMNS = [
    "run_id", "timestamp", "method", "seed", "status",
    "faithfulness", "sparsity", "eval_calls_used",
    "wall_clock_seconds", "output_dir", "error_msg",
]


class ExperimentRegistry:
    """Append-only CSV registry of all experiment runs."""

    def __init__(self, registry_path: str = "results/registry.csv"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write_header()

    def _write_header(self) -> None:
        with open(self.registry_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_COLUMNS)
            writer.writeheader()

    def register(self, run_result) -> None:
        """Append a RunResult to the registry. Never overwrites completed runs."""
        # Check if already completed
        existing = self._find_run(run_result.run_id)
        if existing and existing.get("status") == "completed":
            print(f"[Registry] Skipping completed run: {run_result.run_id}")
            return
        row = {
            "run_id": run_result.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "method": run_result.method,
            "seed": run_result.seed,
            "status": run_result.status,
            "faithfulness": round(run_result.faithfulness, 6),
            "sparsity": round(run_result.sparsity, 6),
            "eval_calls_used": run_result.eval_calls_used,
            "wall_clock_seconds": round(run_result.wall_clock_seconds, 2),
            "output_dir": str(run_result.output_dir),
            "error_msg": run_result.error_msg or "",
        }
        with open(self.registry_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_COLUMNS)
            writer.writerow(row)
        print(f"[Registry] Registered: {run_result.run_id} ({run_result.status})")

    def update_status(self, run_id: str, status: str, **kwargs) -> None:
        """Append an updated row for an existing run_id with new status."""
        existing = self._find_run(run_id)
        if existing is None:
            print(f"[Registry] Warning: run_id {run_id} not found, cannot update.")
            return
        existing.update({"status": status, **{k: str(v) for k, v in kwargs.items()}})
        existing["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.registry_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_COLUMNS)
            writer.writerow({k: existing.get(k, "") for k in _COLUMNS})

    def _find_run(self, run_id: str) -> Optional[dict]:
        """Return the last row matching run_id, or None."""
        result = None
        if not self.registry_path.exists():
            return None
        with open(self.registry_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("run_id") == run_id:
                    result = row
        return result

    def get_run(self, run_id: str) -> Optional[dict]:
        return self._find_run(run_id)

    def load(self):
        """Load registry as a pandas DataFrame (or list of dicts if pandas unavailable)."""
        if not self.registry_path.exists():
            if _HAS_PANDAS:
                import pandas as pd
                return pd.DataFrame(columns=_COLUMNS)
            return []
        if _HAS_PANDAS:
            import pandas as pd
            df = pd.read_csv(self.registry_path)
            for col in ["faithfulness", "sparsity", "wall_clock_seconds"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            # Keep last status per run_id
            df = df.drop_duplicates(subset=["run_id"], keep="last")
            return df
        with open(self.registry_path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
