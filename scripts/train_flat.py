"""
train_flat.py -- Flat PPO training for HRL Circuit Localization.

Usage:
    python scripts/train_flat.py --seed 0 [--output-dir runs/dev]
        [--num-edges 32] [--rollout-steps 128] [--update-epochs 4]
        [--total-steps 10000]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hrl_circuits.env.mock_env import MockCircuitEnv
from hrl_circuits.rl.networks import FlatActor, FlatCritic
from hrl_circuits.rl.advantages import compute_gae
from hrl_circuits.rl.ppo_core import ppo_clip_loss, value_loss
from hrl_circuits.rl.masks import MaskedCategorical
from hrl_circuits.seed import seed_everything


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Flat PPO training for circuit localization")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="runs/dev", help="Output directory for checkpoints and results")
    parser.add_argument("--num-edges", type=int, default=32, help="Number of edges in the circuit")
    parser.add_argument("--rollout-steps", type=int, default=128, help="Steps per rollout before PPO update")
    parser.add_argument("--update-epochs", type=int, default=4, help="Number of PPO update epochs per rollout")
    parser.add_argument("--total-steps", type=int, default=10000, help="Total environment steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--minibatch-size", type=int, default=32, help="Minibatch size for PPO updates")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def build_obs(env):
    """Build flat observation vector by concatenating global and mean-pooled edge features."""
    global_feats = env.get_global_features()          # shape: (global_dim,)
    edge_feats = env.get_edge_features()               # shape: (num_edges, feat_dim)
    obs = np.concatenate([global_feats, edge_feats.mean(axis=0)], axis=0)
    return obs.astype(np.float32)


def save_checkpoint(output_dir, step, actor, critic):
    """Save actor and critic state dicts to a checkpoint directory."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(actor.state_dict(), os.path.join(ckpt_dir, "actor_state_dict.pt"))
    torch.save(critic.state_dict(), os.path.join(ckpt_dir, "critic_state_dict.pt"))
    print(f"[Checkpoint] Saved to {ckpt_dir}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cpu")  # force CPU; no GPU needed for mock training
    print(f"[Config] device={device} seed={args.seed} num_edges={args.num_edges} "
          f"rollout_steps={args.rollout_steps} total_steps={args.total_steps}")

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
    env.reset()

    # ------------------------------------------------------------------
    # Obs dimension (global_dim + edge_feat_dim via mean pooling)
    # ------------------------------------------------------------------
    global_feats = env.get_global_features()
    edge_feats = env.get_edge_features()
    global_dim = global_feats.shape[0]
    edge_feat_dim = edge_feats.shape[1]
    obs_dim = global_dim + edge_feat_dim
    action_dim = args.num_edges + 1  # edges + STOP

    print(f"[Config] obs_dim={obs_dim}  action_dim={action_dim}")

    # ------------------------------------------------------------------
    # Networks and optimizers
    # ------------------------------------------------------------------
    actor = FlatActor(obs_dim=obs_dim, action_dim=action_dim).to(device)
    critic = FlatCritic(obs_dim=obs_dim).to(device)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # Tracking variables
    # ------------------------------------------------------------------
    global_step = 0
    episode_count = 0
    episode_rewards = []
    episode_faithfulness = []
    all_episode_rewards = []
    all_faithfulness = []

    # Metrics for progress reporting
    recent_rewards = []
    recent_faithfulness = []
    last_log_step = 0
    last_ckpt_step = 0

    # ------------------------------------------------------------------
    # Initial reset
    # ------------------------------------------------------------------
    env.reset()
    current_obs = build_obs(env)
    current_episode_reward = 0.0

    start_time = time.time()

    # ==================================================================
    # Main training loop
    # ==================================================================
    while global_step < args.total_steps:

        # --------------------------------------------------------------
        # Rollout collection
        # --------------------------------------------------------------
        rollout_obs = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_values = []
        rollout_rewards = []
        rollout_dones = []

        for _ in range(args.rollout_steps):
            obs_tensor = torch.tensor(current_obs, dtype=torch.float32, device=device)

            with torch.no_grad():
                logits = actor(obs_tensor)
                value = critic(obs_tensor)

            valid_mask_np = env.valid_action_mask()
            valid_mask = torch.tensor(valid_mask_np, dtype=torch.bool, device=device)

            dist = MaskedCategorical(logits=logits, mask=valid_mask)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Step environment
            obs_new, rew, term, trunc, info = env.step(action.item())
            done = term or trunc
            current_episode_reward += rew

            # Store transition
            rollout_obs.append(obs_tensor)
            rollout_actions.append(action)
            rollout_log_probs.append(log_prob)
            rollout_values.append(value.squeeze(-1))
            rollout_rewards.append(float(rew))
            rollout_dones.append(float(done))

            global_step += 1

            if done or env._episode_done:
                # Finalize episode and record stats
                try:
                    eval_result = env.finalize_episode()
                    ep_faith = float(eval_result.faithfulness)
                except Exception:
                    ep_faith = 0.0

                recent_rewards.append(current_episode_reward)
                recent_faithfulness.append(ep_faith)
                all_episode_rewards.append(current_episode_reward)
                all_faithfulness.append(ep_faith)
                episode_count += 1
                current_episode_reward = 0.0

                env.reset()
                current_obs = build_obs(env)
            else:
                current_obs = build_obs(env)

            # ----------------------------------------------------------
            # Logging every 1000 steps
            # ----------------------------------------------------------
            if global_step - last_log_step >= 1000:
                mean_rew = float(np.mean(recent_rewards)) if recent_rewards else 0.0
                mean_faith = float(np.mean(recent_faithfulness)) if recent_faithfulness else 0.0
                elapsed = time.time() - start_time
                print(
                    f"[Step {global_step:6d}] episodes={episode_count:4d}  "
                    f"mean_reward={mean_rew:.4f}  mean_faithfulness={mean_faith:.4f}  "
                    f"elapsed={elapsed:.1f}s"
                )
                recent_rewards = []
                recent_faithfulness = []
                last_log_step = global_step

            # ----------------------------------------------------------
            # Checkpoint every 2000 steps
            # ----------------------------------------------------------
            if global_step - last_ckpt_step >= 2000:
                save_checkpoint(args.output_dir, global_step, actor, critic)
                last_ckpt_step = global_step

            if global_step >= args.total_steps:
                break

        # --------------------------------------------------------------
        # Compute returns and advantages
        # --------------------------------------------------------------
        # Bootstrap value from final state
        final_obs_tensor = torch.tensor(current_obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            next_value = critic(final_obs_tensor).squeeze(-1)

        rewards_tensor = torch.tensor(rollout_rewards, dtype=torch.float32, device=device)
        values_tensor = torch.stack(rollout_values)                # (T,)
        dones_tensor = torch.tensor(rollout_dones, dtype=torch.float32, device=device)
        obs_tensor_batch = torch.stack(rollout_obs)                # (T, obs_dim)
        actions_tensor = torch.stack(rollout_actions)              # (T,)
        old_log_probs_tensor = torch.stack(rollout_log_probs)      # (T,)

        advantages = compute_gae(
            rewards_tensor,
            values_tensor,
            dones_tensor,
            next_value,
            gamma=args.gamma,
            lam=args.lam,
        )
        returns = advantages + values_tensor.detach()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --------------------------------------------------------------
        # PPO update epochs
        # --------------------------------------------------------------
        T = len(rollout_obs)
        for _epoch in range(args.update_epochs):
            perm = torch.randperm(T, device=device)

            for start in range(0, T, args.minibatch_size):
                mb_idx = perm[start: start + args.minibatch_size]
                if len(mb_idx) == 0:
                    continue

                mb_obs = obs_tensor_batch[mb_idx]
                mb_actions = actions_tensor[mb_idx]
                mb_old_lp = old_log_probs_tensor[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                # Forward pass
                new_logits = actor(mb_obs)

                # We don't re-mask during update (masks were applied at collection time).
                # Use an all-True mask so MaskedCategorical behaves like Categorical.
                all_true_mask = torch.ones(new_logits.shape[-1], dtype=torch.bool, device=device)
                new_dist = MaskedCategorical(
                    logits=new_logits,
                    mask=all_true_mask.unsqueeze(0).expand(mb_obs.shape[0], -1),
                )
                new_log_probs = new_dist.log_prob(mb_actions)
                entropy = new_dist.entropy().mean()

                new_values = critic(mb_obs).squeeze(-1)

                # Losses
                actor_loss = ppo_clip_loss(
                    log_prob_new=new_log_probs,
                    log_prob_old=mb_old_lp.detach(),
                    advantages=mb_adv.detach(),
                    clip_epsilon=args.clip_epsilon,
                )
                critic_loss = value_loss(new_values, mb_ret.detach())

                loss = actor_loss + args.value_coef * critic_loss - args.entropy_coef * entropy

                actor_opt.zero_grad()
                critic_opt.zero_grad()
                loss.backward()
                clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                clip_grad_norm_(critic.parameters(), args.max_grad_norm)
                actor_opt.step()
                critic_opt.step()

    # ==================================================================
    # Final checkpoint
    # ==================================================================
    save_checkpoint(args.output_dir, global_step, actor, critic)

    # ==================================================================
    # Save results JSON
    # ==================================================================
    results = {
        "total_steps": global_step,
        "total_episodes": episode_count,
        "mean_episode_reward": float(np.mean(all_episode_rewards)) if all_episode_rewards else 0.0,
        "mean_faithfulness": float(np.mean(all_faithfulness)) if all_faithfulness else 0.0,
        "std_faithfulness": float(np.std(all_faithfulness)) if all_faithfulness else 0.0,
        "all_episode_rewards": [float(r) for r in all_episode_rewards],
        "all_faithfulness": [float(f) for f in all_faithfulness],
        "config": vars(args),
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[Done] Results saved to {results_path}")
    print(f"[Done] total_steps={global_step} episodes={episode_count} "
          f"mean_faithfulness={results['mean_faithfulness']:.4f}")


if __name__ == "__main__":
    main()
