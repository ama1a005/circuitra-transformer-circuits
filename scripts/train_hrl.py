"""
train_hrl.py -- Hierarchical PPO training for HRL Circuit Localization.

Usage:
    python scripts/train_hrl.py --seed 0 [--output-dir runs/dev]
        [--num-edges 32] [--total-episodes 200]
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hrl_circuits.env.mock_env import MockCircuitEnv
from hrl_circuits.rl.advantages import compute_gae
from hrl_circuits.rl.ppo_core import ppo_clip_loss, value_loss
from hrl_circuits.rl.masks import MaskedCategorical
from hrl_circuits.seed import seed_everything


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="HRL PPO training for circuit localization")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="runs/dev", help="Output directory")
    parser.add_argument("--num-edges", type=int, default=32, help="Number of edges")
    parser.add_argument("--num-regions", type=int, default=4, help="Number of regions")
    parser.add_argument("--option-horizon", type=int, default=4, help="Option horizon (worker steps per option)")
    parser.add_argument("--total-episodes", type=int, default=200, help="Total training episodes")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--update-epochs", type=int, default=4, help="PPO update epochs")
    parser.add_argument("--minibatch-size", type=int, default=16, help="Minibatch size")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clip norm")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Observation builders
# ---------------------------------------------------------------------------

def build_manager_obs(env):
    """Concatenate global features with mean-pooled region features."""
    global_feats = env.get_global_features()           # (global_dim,)
    region_feats = env.get_region_features()           # (num_regions, region_feat_dim)
    obs = np.concatenate([global_feats, region_feats.mean(axis=0)], axis=0)
    return obs.astype(np.float32)


def build_worker_obs(env):
    """Concatenate global features with mean-pooled edge features."""
    global_feats = env.get_global_features()           # (global_dim,)
    edge_feats = env.get_edge_features()               # (num_edges, feat_dim)
    obs = np.concatenate([global_feats, edge_feats.mean(axis=0)], axis=0)
    return obs.astype(np.float32)


# ---------------------------------------------------------------------------
# PPO update helper (works for both manager and worker)
# ---------------------------------------------------------------------------

def ppo_update(
    actor_net,
    critic_net,
    optimizer,
    obs_list,
    actions_list,
    log_probs_list,
    values_list,
    returns_list,
    advantages_list,
    update_epochs,
    minibatch_size,
    clip_epsilon,
    entropy_coef,
    value_coef,
    max_grad_norm,
    device,
    action_dim,
):
    """Run PPO update epochs on collected experience."""
    if len(obs_list) == 0:
        return

    obs_t = torch.stack(obs_list).to(device)
    acts_t = torch.stack(actions_list).to(device)
    old_lp_t = torch.stack(log_probs_list).to(device).detach()
    rets_t = torch.stack(returns_list).to(device).detach()
    advs_t = torch.stack(advantages_list).to(device).detach()

    # Normalize advantages (skip if only 1 sample to avoid NaN from std=0)
    if advs_t.numel() > 1:
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

    T = obs_t.shape[0]
    all_true_mask = torch.ones(action_dim, dtype=torch.bool, device=device)

    for _epoch in range(update_epochs):
        perm = torch.randperm(T, device=device)
        for start in range(0, T, minibatch_size):
            idx = perm[start: start + minibatch_size]
            if len(idx) == 0:
                continue

            mb_obs = obs_t[idx]
            mb_acts = acts_t[idx]
            mb_old_lp = old_lp_t[idx]
            mb_adv = advs_t[idx]
            mb_ret = rets_t[idx]

            new_logits = actor_net(mb_obs)
            # Ensure logits are 2D: (batch, action_dim)
            if new_logits.dim() == 1:
                new_logits = new_logits.unsqueeze(0)
            expanded_mask = all_true_mask.unsqueeze(0).expand(new_logits.shape[0], -1)
            new_dist = MaskedCategorical(logits=new_logits, mask=expanded_mask)
            new_log_probs = new_dist.log_prob(mb_acts)
            entropy = new_dist.entropy().mean()

            new_values = critic_net(mb_obs).squeeze(-1)

            a_loss = ppo_clip_loss(
                log_prob_new=new_log_probs,
                log_prob_old=mb_old_lp,
                advantages=mb_adv,
                clip_epsilon=clip_epsilon,
            )
            c_loss = value_loss(new_values, mb_ret)
            loss = a_loss + value_coef * c_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(list(actor_net.parameters()) + list(critic_net.parameters()), max_grad_norm)
            optimizer.step()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(output_dir, episode, manager_actor, manager_critic, worker_actor, worker_critic):
    ckpt_dir = os.path.join(output_dir, f"checkpoint_{episode}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(manager_actor.state_dict(), os.path.join(ckpt_dir, "manager_actor_state_dict.pt"))
    torch.save(manager_critic.state_dict(), os.path.join(ckpt_dir, "manager_critic_state_dict.pt"))
    torch.save(worker_actor.state_dict(), os.path.join(ckpt_dir, "worker_actor_state_dict.pt"))
    torch.save(worker_critic.state_dict(), os.path.join(ckpt_dir, "worker_critic_state_dict.pt"))
    # Also save actor_state_dict.pt / critic_state_dict.pt for compatibility with evaluate_checkpoint.py
    torch.save(manager_actor.state_dict(), os.path.join(ckpt_dir, "actor_state_dict.pt"))
    torch.save(manager_critic.state_dict(), os.path.join(ckpt_dir, "critic_state_dict.pt"))
    print(f"[Checkpoint] Saved episode {episode} to {ckpt_dir}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cpu")  # force CPU; no GPU needed for mock training
    print(f"[Config] device={device} seed={args.seed} num_edges={args.num_edges} "
          f"num_regions={args.num_regions} total_episodes={args.total_episodes}")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    env = MockCircuitEnv(
        num_edges=args.num_edges,
        num_regions=args.num_regions,
        max_causal_forward_calls=200,
        option_horizon=args.option_horizon,
        seed=args.seed,
    )
    env.reset()

    # ------------------------------------------------------------------
    # Obs dimensions
    # ------------------------------------------------------------------
    global_feats = env.get_global_features()
    edge_feats = env.get_edge_features()
    region_feats = env.get_region_features()

    global_dim = global_feats.shape[0]
    edge_feat_dim = edge_feats.shape[1]
    region_feat_dim = region_feats.shape[1]

    manager_obs_dim = global_dim + region_feat_dim    # global + mean-pooled region
    worker_obs_dim = global_dim + edge_feat_dim       # global + mean-pooled edge

    manager_action_dim = args.num_regions + 1         # regions + STOP
    worker_action_dim = args.num_edges + 1            # edges + END_OPTION

    print(f"[Config] manager_obs_dim={manager_obs_dim} manager_action_dim={manager_action_dim}")
    print(f"[Config] worker_obs_dim={worker_obs_dim}  worker_action_dim={worker_action_dim}")

    # ------------------------------------------------------------------
    # Networks (simple linear as per spec)
    # ------------------------------------------------------------------
    manager_actor = nn.Linear(manager_obs_dim, manager_action_dim).to(device)
    manager_critic = nn.Linear(manager_obs_dim, 1).to(device)
    worker_actor = nn.Linear(worker_obs_dim, worker_action_dim).to(device)
    worker_critic = nn.Linear(worker_obs_dim, 1).to(device)

    manager_opt = torch.optim.Adam(
        list(manager_actor.parameters()) + list(manager_critic.parameters()), lr=args.lr
    )
    worker_opt = torch.optim.Adam(
        list(worker_actor.parameters()) + list(worker_critic.parameters()), lr=args.lr
    )

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------
    all_episode_rewards = []
    all_faithfulness = []
    recent_rewards = []
    recent_faithfulness = []
    start_time = time.time()

    # Accumulated experience for PPO updates (across episodes between updates)
    m_obs_buf, m_acts_buf, m_lp_buf, m_vals_buf, m_rets_buf, m_advs_buf = [], [], [], [], [], []
    w_obs_buf, w_acts_buf, w_lp_buf, w_vals_buf, w_rets_buf, w_advs_buf = [], [], [], [], [], []

    # ==================================================================
    # Episode loop
    # ==================================================================
    for episode in range(1, args.total_episodes + 1):

        env.reset()
        ep_reward = 0.0

        # Per-episode transition storage
        manager_obs_ep = []
        manager_acts_ep = []
        manager_lp_ep = []
        manager_vals_ep = []
        manager_rewards_ep = []
        manager_dones_ep = []

        worker_obs_ep = []
        worker_acts_ep = []
        worker_lp_ep = []
        worker_vals_ep = []
        worker_rewards_ep = []
        worker_dones_ep = []

        # ------------------------------------------------------------------
        # Episode rollout
        # ------------------------------------------------------------------
        while not env._episode_done:
            # ---- Manager step ----
            m_obs_np = build_manager_obs(env)
            m_obs_t = torch.tensor(m_obs_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                m_logits = manager_actor(m_obs_t)
                m_val = manager_critic(m_obs_t).squeeze(-1)

            m_valid_np = env.valid_manager_action_mask()
            m_valid_t = torch.tensor(m_valid_np, dtype=torch.bool, device=device)

            m_dist = MaskedCategorical(logits=m_logits, mask=m_valid_t)
            m_action = m_dist.sample()
            m_log_prob = m_dist.log_prob(m_action)

            region_idx = m_action.item()

            # select_region performs the manager action
            env.select_region(region_idx)

            # If the manager selected STOP (region == num_regions), end episode
            if region_idx == args.num_regions:
                manager_obs_ep.append(m_obs_t)
                manager_acts_ep.append(m_action)
                manager_lp_ep.append(m_log_prob)
                manager_vals_ep.append(m_val)
                manager_rewards_ep.append(0.0)
                manager_dones_ep.append(1.0)
                break

            # ---- Worker inner loop ----
            option_reward = 0.0
            w_option_obs = []
            w_option_acts = []
            w_option_lps = []
            w_option_vals = []
            w_option_rews = []
            w_option_dones = []

            while env.option_state.active:
                w_obs_np = build_worker_obs(env)
                w_obs_t = torch.tensor(w_obs_np, dtype=torch.float32, device=device)

                with torch.no_grad():
                    w_logits = worker_actor(w_obs_t)
                    w_val = worker_critic(w_obs_t).squeeze(-1)

                w_valid_np = env.valid_worker_action_mask()
                w_valid_t = torch.tensor(w_valid_np, dtype=torch.bool, device=device)
                # Worker logits have dim num_edges+1, but worker mask is per-region.
                # Pad the mask to match logit size (padded entries = False).
                logit_dim = w_logits.shape[-1]
                if w_valid_t.shape[0] < logit_dim:
                    pad = torch.zeros(logit_dim - w_valid_t.shape[0], dtype=torch.bool, device=device)
                    w_valid_t = torch.cat([w_valid_t, pad])
                elif w_valid_t.shape[0] > logit_dim:
                    w_valid_t = w_valid_t[:logit_dim]

                w_dist = MaskedCategorical(logits=w_logits, mask=w_valid_t)
                w_action = w_dist.sample()
                w_log_prob = w_dist.log_prob(w_action)

                # worker_step(last_in_worker_mask) -- pass the mask for END_OPTION signal
                obs_new, rew, term, trunc, info = env.worker_step(w_action.item())
                done = term or trunc
                option_reward += rew
                ep_reward += rew

                w_option_obs.append(w_obs_t)
                w_option_acts.append(w_action)
                w_option_lps.append(w_log_prob)
                w_option_vals.append(w_val)
                w_option_rews.append(float(rew))
                w_option_dones.append(float(done))

                if done or not env.option_state.active:
                    break

            # Accumulate worker transitions for this option
            worker_obs_ep.extend(w_option_obs)
            worker_acts_ep.extend(w_option_acts)
            worker_lp_ep.extend(w_option_lps)
            worker_vals_ep.extend(w_option_vals)
            worker_rewards_ep.extend(w_option_rews)
            worker_dones_ep.extend(w_option_dones)

            # Manager receives cumulative option reward
            manager_obs_ep.append(m_obs_t)
            manager_acts_ep.append(m_action)
            manager_lp_ep.append(m_log_prob)
            manager_vals_ep.append(m_val)
            manager_rewards_ep.append(option_reward)
            manager_dones_ep.append(float(env._episode_done))

        # ------------------------------------------------------------------
        # Finalize episode
        # ------------------------------------------------------------------
        try:
            eval_result = env.finalize_episode()
            ep_faith = float(eval_result.faithfulness)
        except Exception:
            ep_faith = 0.0

        all_episode_rewards.append(ep_reward)
        all_faithfulness.append(ep_faith)
        recent_rewards.append(ep_reward)
        recent_faithfulness.append(ep_faith)

        # ------------------------------------------------------------------
        # Compute returns and advantages
        # ------------------------------------------------------------------

        def compute_returns_and_advantages(obs_list, vals_list, rews_list, dones_list,
                                           critic_net, device, gamma, lam):
            """GAE for a list of episode transitions."""
            if len(obs_list) == 0:
                return [], []
            rews_t = torch.tensor(rews_list, dtype=torch.float32, device=device)
            vals_t = torch.stack(vals_list)
            dones_t = torch.tensor(dones_list, dtype=torch.float32, device=device)
            # Bootstrap with zero if episode ended
            next_val = torch.tensor(0.0, device=device)
            advs = compute_gae(rews_t, vals_t, dones_t, next_val, gamma=gamma, lam=lam)
            rets = advs + vals_t.detach()
            return advs, rets

        m_advs, m_rets = compute_returns_and_advantages(
            manager_obs_ep, manager_vals_ep, manager_rewards_ep, manager_dones_ep,
            manager_critic, device, args.gamma, args.lam,
        )
        w_advs, w_rets = compute_returns_and_advantages(
            worker_obs_ep, worker_vals_ep, worker_rewards_ep, worker_dones_ep,
            worker_critic, device, args.gamma, args.lam,
        )

        # Accumulate into buffers
        if len(manager_obs_ep) > 0 and len(m_advs) > 0:
            m_obs_buf.extend(manager_obs_ep)
            m_acts_buf.extend(manager_acts_ep)
            m_lp_buf.extend(manager_lp_ep)
            m_vals_buf.extend(manager_vals_ep)
            m_rets_buf.extend(list(m_rets))
            m_advs_buf.extend(list(m_advs))

        if len(worker_obs_ep) > 0 and len(w_advs) > 0:
            w_obs_buf.extend(worker_obs_ep)
            w_acts_buf.extend(worker_acts_ep)
            w_lp_buf.extend(worker_lp_ep)
            w_vals_buf.extend(worker_vals_ep)
            w_rets_buf.extend(list(w_rets))
            w_advs_buf.extend(list(w_advs))

        # ------------------------------------------------------------------
        # PPO update (every episode -- clear buffers after update)
        # ------------------------------------------------------------------
        ppo_update(
            manager_actor, manager_critic, manager_opt,
            m_obs_buf, m_acts_buf, m_lp_buf, m_vals_buf, m_rets_buf, m_advs_buf,
            args.update_epochs, args.minibatch_size, args.clip_epsilon,
            args.entropy_coef, args.value_coef, args.max_grad_norm,
            device, manager_action_dim,
        )
        ppo_update(
            worker_actor, worker_critic, worker_opt,
            w_obs_buf, w_acts_buf, w_lp_buf, w_vals_buf, w_rets_buf, w_advs_buf,
            args.update_epochs, args.minibatch_size, args.clip_epsilon,
            args.entropy_coef, args.value_coef, args.max_grad_norm,
            device, worker_action_dim,
        )

        # Clear buffers after update
        m_obs_buf.clear(); m_acts_buf.clear(); m_lp_buf.clear()
        m_vals_buf.clear(); m_rets_buf.clear(); m_advs_buf.clear()
        w_obs_buf.clear(); w_acts_buf.clear(); w_lp_buf.clear()
        w_vals_buf.clear(); w_rets_buf.clear(); w_advs_buf.clear()

        # ------------------------------------------------------------------
        # Logging every 50 episodes
        # ------------------------------------------------------------------
        if episode % 50 == 0:
            mean_rew = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            mean_faith = float(np.mean(recent_faithfulness)) if recent_faithfulness else 0.0
            elapsed = time.time() - start_time
            print(
                f"[Episode {episode:5d}/{args.total_episodes}] "
                f"mean_reward={mean_rew:.4f}  mean_faithfulness={mean_faith:.4f}  "
                f"elapsed={elapsed:.1f}s"
            )
            recent_rewards = []
            recent_faithfulness = []

        # ------------------------------------------------------------------
        # Checkpoint every 100 episodes
        # ------------------------------------------------------------------
        if episode % 100 == 0:
            save_checkpoint(
                args.output_dir, episode,
                manager_actor, manager_critic,
                worker_actor, worker_critic,
            )

    # ==================================================================
    # Final checkpoint
    # ==================================================================
    save_checkpoint(
        args.output_dir, args.total_episodes,
        manager_actor, manager_critic,
        worker_actor, worker_critic,
    )

    # ==================================================================
    # Save results JSON
    # ==================================================================
    results = {
        "total_episodes": args.total_episodes,
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
    print(f"[Done] episodes={args.total_episodes} "
          f"mean_faithfulness={results['mean_faithfulness']:.4f}")


if __name__ == "__main__":
    main()
