"""
evaluate_checkpoint.py -- Evaluate a saved PPO checkpoint.

Usage:
    python scripts/evaluate_checkpoint.py
        --checkpoint-dir runs/dev/checkpoint_2000
        [--num-episodes 5] [--seed 0] [--num-edges 32]

If actor_state_dict.pt / critic_state_dict.pt are not found in the checkpoint
directory, the script falls back to a random policy and logs a warning.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hrl_circuits.env.mock_env import MockCircuitEnv
from hrl_circuits.rl.networks import FlatActor, FlatCritic
from hrl_circuits.rl.masks import MaskedCategorical
from hrl_circuits.seed import seed_everything


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO checkpoint")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to checkpoint directory containing actor_state_dict.pt and critic_state_dict.pt",
    )
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-edges", type=int, default=32, help="Number of edges")
    parser.add_argument("--num-regions", type=int, default=4, help="Number of regions")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use greedy (argmax) policy instead of sampling (default: True)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Observation builder (must match train_flat.py)
# ---------------------------------------------------------------------------

def build_obs(env):
    """Flat obs: global features concatenated with mean-pooled edge features."""
    global_feats = env.get_global_features()
    edge_feats = env.get_edge_features()
    obs = np.concatenate([global_feats, edge_feats.mean(axis=0)], axis=0)
    return obs.astype(np.float32)


# ---------------------------------------------------------------------------
# Fallback random policy
# ---------------------------------------------------------------------------

def random_action(valid_mask, rng):
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return len(valid_mask) - 1
    return int(rng.choice(valid_indices))


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(env, actor, critic, device, deterministic, rng, use_random):
    """
    Run one evaluation episode.

    Returns:
        total_reward (float)
        eval_result: EvalResult from env.finalize_episode()
    """
    env.reset()
    total_reward = 0.0

    while not env._episode_done:
        obs_np = build_obs(env)
        valid_mask_np = env.valid_action_mask()

        if use_random:
            action_idx = random_action(valid_mask_np, rng)
        else:
            obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
            valid_mask_t = torch.tensor(valid_mask_np, dtype=torch.bool, device=device)

            with torch.no_grad():
                logits = actor(obs_t)

            if deterministic:
                # Mask out invalid actions by setting their logits to -inf
                masked_logits = logits.clone()
                masked_logits[~valid_mask_t] = float("-inf")
                action_idx = int(masked_logits.argmax().item())
            else:
                dist = MaskedCategorical(logits=logits, mask=valid_mask_t)
                action_idx = int(dist.sample().item())

        obs_new, rew, term, trunc, info = env.step(action_idx)
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
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    env = MockCircuitEnv(
        num_edges=args.num_edges,
        num_regions=args.num_regions,
        max_causal_forward_calls=200,
        max_episode_steps=64,
        seed=args.seed,
    )
    env.reset()

    # ------------------------------------------------------------------
    # Obs / action dims (must match train_flat.py)
    # ------------------------------------------------------------------
    global_feats = env.get_global_features()
    edge_feats = env.get_edge_features()
    global_dim = global_feats.shape[0]
    edge_feat_dim = edge_feats.shape[1]
    obs_dim = global_dim + edge_feat_dim
    action_dim = args.num_edges + 1

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    actor_path = os.path.join(args.checkpoint_dir, "actor_state_dict.pt")
    critic_path = os.path.join(args.checkpoint_dir, "critic_state_dict.pt")

    use_random = False

    if os.path.isfile(actor_path) and os.path.isfile(critic_path):
        print(f"[Eval] Loading checkpoint from {args.checkpoint_dir}")
        actor = FlatActor(obs_dim=obs_dim, action_dim=action_dim).to(device)
        critic = FlatCritic(obs_dim=obs_dim).to(device)
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        critic.load_state_dict(torch.load(critic_path, map_location=device))
        actor.eval()
        critic.eval()
        print(f"[Eval] obs_dim={obs_dim}  action_dim={action_dim}  deterministic={args.deterministic}")
    else:
        print(
            f"[Warning] Checkpoint files not found in {args.checkpoint_dir}. "
            f"Falling back to random policy."
        )
        actor = None
        critic = None
        use_random = True

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    episode_rewards = []
    faithfulness_scores = []
    sparsity_scores = []
    edges_included_list = []
    edges_total_list = []
    eval_calls_list = []

    print(f"[Eval] Running {args.num_episodes} episodes ...")

    for ep in range(1, args.num_episodes + 1):
        total_reward, eval_result = run_episode(
            env, actor, critic, device, args.deterministic, rng, use_random
        )

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
    # Aggregate statistics
    # ------------------------------------------------------------------
    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_faith = float(np.mean(faithfulness_scores))
    std_faith = float(np.std(faithfulness_scores))
    mean_spars = float(np.mean(sparsity_scores))
    std_spars = float(np.std(sparsity_scores))

    print("")
    print(f"[Summary] checkpoint={args.checkpoint_dir}")
    print(f"  mean_reward      : {mean_reward:.4f} +/- {std_reward:.4f}")
    print(f"  mean_faithfulness: {mean_faith:.4f} +/- {std_faith:.4f}")
    print(f"  mean_sparsity    : {mean_spars:.4f} +/- {std_spars:.4f}")

    # ------------------------------------------------------------------
    # Save eval_results.json alongside checkpoint (or in output-dir)
    # ------------------------------------------------------------------
    eval_results = {
        "checkpoint_dir": args.checkpoint_dir,
        "num_episodes": args.num_episodes,
        "num_edges": args.num_edges,
        "seed": args.seed,
        "policy": "random_fallback" if use_random else ("greedy" if args.deterministic else "stochastic"),
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

    # Write eval_results.json to the checkpoint dir if it exists, else cwd
    if os.path.isdir(args.checkpoint_dir):
        out_path = os.path.join(args.checkpoint_dir, "eval_results.json")
    else:
        out_path = "eval_results.json"

    with open(out_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"[Done] Eval results saved to {out_path}")


if __name__ == "__main__":
    main()
