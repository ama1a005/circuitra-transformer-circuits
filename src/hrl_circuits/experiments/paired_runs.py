"""experiments/paired_runs.py
==============================
Manages paired flat-vs-HRL experiments to ensure fair comparison.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from hrl_circuits.experiments.runner import ExperimentRunner, RunResult


class PairedRunManager:
    """Creates and executes paired flat/HRL experiments with matching configs."""

    def __init__(self, runner: Optional[ExperimentRunner] = None):
        self.runner = runner or ExperimentRunner()

    def create_paired_manifest(
        self,
        flat_config: dict,
        hrl_config: dict,
        seeds: list[int],
    ) -> dict:
        """
        Create a manifest dict pairing flat vs HRL configs for each seed.
        Both configs must share the same graph/env settings for fair comparison.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        pairs = []
        for seed in seeds:
            pairs.append({
                "seed": seed,
                "flat": {"config": flat_config, "seed": seed},
                "hrl": {"config": hrl_config, "seed": seed},
            })
        manifest = {
            "timestamp": timestamp,
            "n_seeds": len(seeds),
            "seeds": seeds,
            "flat_config": flat_config,
            "hrl_config": hrl_config,
            "pairs": pairs,
        }
        # Validate configs share key env settings
        self._verify_configs_compatible(flat_config, hrl_config)
        return manifest

    def _verify_configs_compatible(self, flat_config: dict, hrl_config: dict) -> None:
        """Raise ValueError if configs are incompatible for pairing."""
        for key in ["num_edges", "num_regions", "max_causal_forward_calls"]:
            flat_val = flat_config.get(key)
            hrl_val = hrl_config.get(key)
            if flat_val is not None and hrl_val is not None and flat_val != hrl_val:
                raise ValueError(
                    f"Paired configs must have matching {key}: "
                    f"flat={flat_val} vs hrl={hrl_val}"
                )

    def save_manifest(self, manifest: dict, output_path: str) -> None:
        Path(output_path).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"[PairedRuns] Manifest saved to {output_path}")

    def load_manifest(self, manifest_path: str) -> dict:
        return json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    def run_paired(self, manifest: dict, output_dir: str) -> list[RunResult]:
        """Execute all pairs in the manifest and return results."""
        results = []
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, pair in enumerate(manifest["pairs"]):
            seed = pair["seed"]
            print(f"[PairedRuns] Pair {i+1}/{len(manifest['pairs'])} seed={seed}")
            flat_res = self.runner.run_flat_experiment(
                config=pair["flat"]["config"],
                seed=seed,
                output_dir=str(out / "flat"),
            )
            hrl_res = self.runner.run_hrl_experiment(
                config=pair["hrl"]["config"],
                seed=seed,
                output_dir=str(out / "hrl"),
            )
            results.extend([flat_res, hrl_res])
        return results

    def analyze_paired(self, results: list[RunResult]) -> dict:
        """Compute mean difference, win rate (HRL vs flat), and per-seed breakdown."""
        flat_results = [r for r in results if r.method == "flat_ppo"]
        hrl_results = [r for r in results if r.method == "hrl_ppo"]

        if not flat_results or not hrl_results:
            return {"error": "Need both flat and HRL results to analyze"}

        # Match by seed
        flat_by_seed = {r.seed: r for r in flat_results}
        hrl_by_seed = {r.seed: r for r in hrl_results}
        common_seeds = sorted(set(flat_by_seed) & set(hrl_by_seed))

        diffs = []
        per_seed = []
        hrl_wins = 0
        for seed in common_seeds:
            f = flat_by_seed[seed]
            h = hrl_by_seed[seed]
            diff = h.faithfulness - f.faithfulness
            diffs.append(diff)
            if diff > 0:
                hrl_wins += 1
            per_seed.append({
                "seed": seed,
                "flat_faithfulness": f.faithfulness,
                "hrl_faithfulness": h.faithfulness,
                "diff": diff,
            })

        n = len(diffs)
        mean_diff = sum(diffs) / n if n > 0 else 0.0
        return {
            "n_pairs": n,
            "mean_diff_hrl_minus_flat": round(mean_diff, 6),
            "hrl_win_rate": round(hrl_wins / n, 3) if n > 0 else 0.0,
            "per_seed": per_seed,
        }
