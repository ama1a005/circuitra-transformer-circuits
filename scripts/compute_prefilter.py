#!/usr/bin/env python
"""scripts/compute_prefilter.py
================================
Compute and save a prefilter artifact using EAP-IG-style attribution scoring
on the candidate graph edges, using the mock evaluator (or real benchmark).

Usage
-----
    python scripts/compute_prefilter.py --num-edges 32 --seed 0 [--output-dir artifacts/prefilters]
    python scripts/compute_prefilter.py --config configs/baselines/eap_ig.yaml --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np


def compute_mock_attribution_scores(env, method: str = "eap_ig") -> dict[int, float]:
    """
    Compute attribution scores for each edge using the mock env's static attributions.
    In real usage this would run EAP or EAP-IG via the model.
    Returns dict: edge_index -> score.
    """
    if hasattr(env, "_static_attributions"):
        raw = env._static_attributions
        scores = {}
        for i, s in enumerate(raw):
            scores[i] = float(s)
        return scores

    # Fallback: random scores
    rng = np.random.default_rng(42)
    num_edges = env.num_edges
    return {i: float(rng.random()) for i in range(num_edges)}


def build_prefilter_artifact(scores: dict[int, float], top_k: int, method: str,
                              graph_hash: str = "mock_graph") -> dict:
    """Build a prefilter artifact dict from edge scores."""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked_ids = [idx for idx, _ in ranked]
    top_k_ids = ranked_ids[:top_k]

    return {
        "graph_hash": graph_hash,
        "method": method,
        "split": "train",
        "top_k": top_k,
        "scores": {str(k): v for k, v in scores.items()},
        "ranked_edge_ids": ranked_ids,
        "top_k_edge_ids": top_k_ids,
        "benchmark_revision": "mock_v1",
        "config_hash": "none",
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute prefilter attribution scores")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--num-edges", type=int, default=32)
    parser.add_argument("--num-regions", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=20, help="Top-K edges to keep")
    parser.add_argument("--method", type=str, default="eap_ig",
                        choices=["eap", "eap_ig", "random"],
                        help="Attribution method")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="artifacts/prefilters")
    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config:
        try:
            import yaml
            with open(args.config) as f:
                config = yaml.safe_load(f) or {}
            print(f"[INFO] Loaded config from {args.config}")
        except Exception as e:
            print(f"[WARN] Could not load config: {e}, using defaults")

    num_edges = config.get("num_edges", args.num_edges)
    num_regions = config.get("num_regions", args.num_regions)
    top_k = config.get("candidate_top_k", args.top_k)
    method = config.get("prefilter_method", args.method)

    # Create mock env
    from hrl_circuits.env.mock_env import MockCircuitEnv
    env = MockCircuitEnv(
        num_edges=num_edges,
        num_regions=num_regions,
        max_causal_forward_calls=500,
        seed=args.seed,
    )
    env.reset()

    print(f"[INFO] Computing {method} scores for {num_edges} edges...")
    t0 = time.time()
    scores = compute_mock_attribution_scores(env, method=method)
    elapsed = time.time() - t0
    print(f"[INFO] Scoring done in {elapsed:.2f}s")

    # Build artifact
    artifact = build_prefilter_artifact(
        scores=scores,
        top_k=min(top_k, num_edges),
        method=method,
        graph_hash=f"mock_{num_edges}edges_seed{args.seed}",
    )

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"prefilter_{method}_{num_edges}edges_seed{args.seed}.json"
    out_path = out_dir / fname
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"[OK] Saved prefilter artifact: {out_path}")

    # Print top-10 edges
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n--- Top-10 edges by {method} score ---")
    for rank, (edge_idx, score) in enumerate(ranked[:10], 1):
        print(f"  #{rank:2d}  edge_idx={edge_idx:3d}  score={score:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
