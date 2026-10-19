"""
run_baseline.py -- Baseline policy evaluation for HRL Circuit Localization.

Usage:
    python scripts/run_baseline.py --baseline random [--num-edges 32]
        [--num-episodes 5] [--seed 0] [--output-dir runs/dev]

Available baselines:
    random          -- uniformly sample from valid actions each step
    stop_immediately -- take the STOP action on the very first step
    greedy          -- always prune the first valid live edge (action 0)
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hrl_circuits.env.mock_env import MockCircuitEnv
from hrl_circuits.seed import seed_everything


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Baseline evaluation for circuit localization")
    parser.add_argument(
        "--baseline",
        type=str,
        default="random",
        choices=["random", "stop_immediately", "greedy"],
        help="Baseline policy to evaluate",
    )
    parser.add_argument("--num-edges", type=int, default=32, help="Number of edges")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="runs/dev", help="Directory to save JSON results")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------

class RandomPolicy:
    """Uniformly sample a valid action at each step."""
    def __init__(self, rng):
        self.rng = rng

    def select_action(self, valid_mask):
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return len(valid_mask) - 1  # fallback: STOP
        return int(self.rng.choice(valid_indices))


class StopImmediatelyPolicy:
    """Always select STOP (last action) on the very first step."""
    def select_action(self, valid_mask):
        # STOP is the last action index
        return len(valid_mask) - 1


class GreedyPolicy:
    """Always prune the first valid live edge (lowest valid action index)."""
    def select_action(self, valid_mask):
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return len(valid_mask) - 1  # STOP fallback
        # Take the lowest-index valid action (first live edge)
        return int(valid_indices[0])


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, policy):
    """
    Run a single episode with the given policy.

    Returns:
        total_reward (float): sum of rewards over the episode
        eval_result: the EvalResult from env.finalize_episode()
    """
    env.reset()
    total_reward = 0.0

    while not env._episode_done:
        valid_mask = env.valid_action_mask()
        action = policy.select_action(valid_mask)
        obs, rew, term, trunc, info = env.step(action)
        total_reward += rew
        done = term or trunc
        if done:
            break

    eval_result = env.finalize_episode()
    return total_reward, eval_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    env = MockCircuitEnv(
        num_edges=args.num_edges,
        num_regions=4,
        max_causal_forward_calls=200,
        max_episode_steps=64,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Policy selection
    # ------------------------------------------------------------------
    if args.baseline == "random":
        policy = RandomPolicy(rng)
    elif args.baseline == "stop_immediately":
        policy = StopImmediatelyPolicy()
    elif args.baseline == "greedy":
        policy = GreedyPolicy()
    else:
        raise ValueError(f"Unknown baseline: {args.baseline}")

    print(f"[Baseline] policy={args.baseline}  num_episodes={args.num_episodes}  "
          f"num_edges={args.num_edges}  seed={args.seed}")

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    episode_rewards = []
    faithfulness_scores = []
    sparsity_scores = []
    edges_included_list = []
    edges_total_list = []
    eval_calls_list = []

    for ep in range(1, args.num_episodes + 1):
        total_reward, eval_result = run_episode(env, policy)

        faith = float(eval_result.faithfulness)
        spars = float(eval_result.sparsity)
        n_incl = int(eval_result.num_edges_included)
        n_total = int(eval_result.num_edges_total)
        n_calls = int(eval_result.eval_calls_used)

        episode_rewards.append(total_reward)
        faithfulness_scores.append(faith)
        sparsity_scores.append(spars)
        edges_included_list.append(n_incl)
        edges_total_list.append(n_total)
        eval_calls_list.append(n_calls)

        print(
            f"  Episode {ep:3d}: reward={total_reward:.4f}  "
            f"faithfulness={faith:.4f}  sparsity={spars:.4f}  "
            f"edges_included={n_incl}/{n_total}  eval_calls={n_calls}"
        )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_faith = float(np.mean(faithfulness_scores))
    std_faith = float(np.std(faithfulness_scores))
    mean_spars = float(np.mean(sparsity_scores))
    std_spars = float(np.std(sparsity_scores))

    print("")
    print(f"[Summary] baseline={args.baseline}")
    print(f"  mean_reward     : {mean_reward:.4f} +/- {std_reward:.4f}")
    print(f"  mean_faithfulness: {mean_faith:.4f} +/- {std_faith:.4f}")
    print(f"  mean_sparsity   : {mean_spars:.4f} +/- {std_spars:.4f}")

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    results = {
        "baseline": args.baseline,
        "num_episodes": args.num_episodes,
        "num_edges": args.num_edges,
        "seed": args.seed,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_faithfulness": mean_faith,
        "std_faithfulness": std_faith,
        "mean_sparsity": mean_spars,
        "std_sparsity": std_spars,
        "episode_rewards": [float(r) for r in episode_rewards],
        "faithfulness_scores": [float(f) for f in faithfulness_scores],
        "sparsity_scores": [float(s) for s in sparsity_scores],
        "edges_included": edges_included_list,
        "edges_total": edges_total_list,
        "eval_calls_used": eval_calls_list,
    }

    out_filename = f"baseline_{args.baseline}_results.json"
    results_path = os.path.join(args.output_dir, out_filename)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Done] Results saved to {results_path}")


if __name__ == "__main__":
    main()
