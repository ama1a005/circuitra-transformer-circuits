#!/usr/bin/env python
"""scripts/run_final_matrix.py
================================
Runs the full paired experiment matrix (flat PPO vs HRL PPO) across
multiple seeds. Requires a frozen config.

Usage
-----
    python scripts/run_final_matrix.py --config configs/final/main_mib.yaml \
        --seeds 0 1 2 3 4 --output-dir results/final [--allow-mock]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final paired experiment matrix")
    parser.add_argument("--config", type=str, default=None,
                        help="Frozen config YAML (optional, uses defaults if omitted)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Seeds to run (default: 0 1 2 3 4)")
    parser.add_argument("--output-dir", type=str, default="results/final",
                        help="Output directory for all run results")
    parser.add_argument("--allow-mock", action="store_true",
                        help="Allow running without a frozen config (uses mock env)")
    parser.add_argument("--num-edges", type=int, default=32,
                        help="Number of edges (used when --allow-mock)")
    parser.add_argument("--flat-steps", type=int, default=5000,
                        help="Total steps for flat PPO training per seed")
    parser.add_argument("--hrl-episodes", type=int, default=100,
                        help="Total episodes for HRL PPO training per seed")
    args = parser.parse_args()

    # Config handling
    config = {}
    if args.config:
        # Require frozen config for final runs
        from hrl_circuits.experiments.freeze import require_frozen_config, FrozenConfigError
        if not args.allow_mock:
            try:
                require_frozen_config(args.config)
                print(f"[OK] Config is frozen: {args.config}")
            except FrozenConfigError as e:
                print(f"[ERROR] {e}")
                print("  Freeze your config first: python -c \"from hrl_circuits.experiments.freeze import freeze_config; ...\"")
                print("  Or pass --allow-mock to skip this check.")
                return 1
        try:
            import yaml
            with open(args.config, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARN] Could not load config: {e}, using defaults")

    flat_config = {
        "num_edges": config.get("num_edges", args.num_edges),
        "num_regions": config.get("num_regions", 4),
        "max_causal_forward_calls": config.get("max_causal_forward_calls", 200),
        "max_episode_steps": config.get("max_episode_steps", 64),
        "total_steps": config.get("flat_total_steps", args.flat_steps),
    }
    hrl_config = {
        "num_edges": flat_config["num_edges"],
        "num_regions": flat_config["num_regions"],
        "max_causal_forward_calls": flat_config["max_causal_forward_calls"],
        "max_episode_steps": flat_config["max_episode_steps"],
        "option_horizon": config.get("option_horizon", 4),
        "total_episodes": config.get("hrl_total_episodes", args.hrl_episodes),
    }

    print(f"\n{'='*60}")
    print(f"  Final Experiment Matrix")
    print(f"  Seeds: {args.seeds}")
    print(f"  flat_config: num_edges={flat_config['num_edges']} steps={flat_config['total_steps']}")
    print(f"  hrl_config:  num_edges={hrl_config['num_edges']} episodes={hrl_config['total_episodes']}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    from hrl_circuits.experiments.runner import ExperimentRunner
    from hrl_circuits.experiments.registry import ExperimentRegistry
    from hrl_circuits.experiments.paired_runs import PairedRunManager

    registry = ExperimentRegistry(str(Path(args.output_dir) / "registry.csv"))
    runner = ExperimentRunner(registry=registry)
    paired = PairedRunManager(runner=runner)

    manifest = paired.create_paired_manifest(flat_config, hrl_config, args.seeds)
    manifest_path = str(Path(args.output_dir) / "manifest.json")
    paired.save_manifest(manifest, manifest_path)

    print(f"[INFO] Running {len(args.seeds) * 2} total experiments...")
    results = paired.run_paired(manifest, args.output_dir)

    # Analysis
    analysis = paired.analyze_paired(results)
    analysis_path = Path(args.output_dir) / "paired_analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  MATRIX COMPLETE")
    print(f"  Pairs: {analysis.get('n_pairs', 0)}")
    print(f"  HRL win rate: {analysis.get('hrl_win_rate', 0.0):.1%}")
    print(f"  Mean HRL-Flat diff: {analysis.get('mean_diff_hrl_minus_flat', 0.0):+.4f}")
    print(f"  Analysis: {analysis_path}")
    print(f"{'='*60}\n")

    # Generate final artifacts
    try:
        from hrl_circuits.analysis.final_report import FinalReport
        FinalReport(args.output_dir).generate()
    except Exception as e:
        print(f"[WARN] Could not generate report: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
