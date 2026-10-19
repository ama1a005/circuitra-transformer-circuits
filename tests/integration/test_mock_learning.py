"""
Integration test: minimal PPO learning sanity check.

Verifies that FlatActor / FlatCritic can be constructed, do forward passes,
and that shapes are correct. Also checks compute_gae and MaskedCategorical.
"""

import torch
import pytest

from hrl_circuits.env.mock_env import MockCircuitEnv
from hrl_circuits.rl.networks import FlatActor, FlatCritic
from hrl_circuits.rl.advantages import compute_gae
from hrl_circuits.rl.masks import MaskedCategorical


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_mock_flat_learns():
    """Minimal network forward-pass smoke test using MockCircuitEnv."""
    env = MockCircuitEnv(
        num_edges=8,
        num_regions=2,
        max_episode_steps=16,
        seed=42,
    )
    obs = env.reset()

    # Build tiny actor/critic in linear (simple) mode
    actor = FlatActor(obs_dim=16, action_dim=env.num_edges + 1)
    critic = FlatCritic(obs_dim=16)

    obs_tensor = torch.zeros(16)  # dummy obs vector
    logits = actor(obs_tensor)
    value = critic(obs_tensor)

    assert logits.shape == (env.num_edges + 1,), (
        f"Expected logits shape ({env.num_edges + 1},), got {logits.shape}"
    )
    assert value.shape == (1,) or value.numel() == 1, (
        f"Expected scalar value, got shape {value.shape}"
    )


def test_flat_actor_batch_forward():
    """FlatActor handles batched obs correctly in simple mode."""
    env = MockCircuitEnv(num_edges=8, num_regions=2, seed=0)
    env.reset()

    actor = FlatActor(obs_dim=16, action_dim=env.num_edges + 1)
    batch_obs = torch.zeros(4, 16)  # batch of 4
    logits = actor(batch_obs)
    assert logits.shape == (4, env.num_edges + 1)


def test_flat_critic_batch_forward():
    """FlatCritic handles batched obs correctly in simple mode."""
    critic = FlatCritic(obs_dim=16)
    batch_obs = torch.zeros(4, 16)
    values = critic(batch_obs)
    # Returns (batch, 1) in simple mode
    assert values.numel() == 4


def test_compute_gae_shapes():
    """compute_gae returns advantages of length T."""
    T = 10
    rewards = torch.zeros(T)
    values = torch.zeros(T)
    dones = torch.zeros(T)
    rewards[T - 1] = 1.0
    dones[T - 1] = 1.0

    advantages = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=0.0,
        gamma=0.99,
        lam=0.95,
    )
    assert advantages.shape == (T,)
    assert advantages.dtype == torch.float32


def test_compute_gae_terminal_done():
    """GAE advantages after a terminal step should not bleed past done."""
    T = 5
    rewards = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    values = torch.zeros(T)
    dones = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])  # terminal at t=2

    advantages = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=0.0,
    )
    # Advantage at t=2 should equal the reward (no bootstrap past done)
    assert abs(advantages[2].item() - 1.0) < 1e-5


def test_masked_categorical_sample():
    """MaskedCategorical samples only from valid actions."""
    logits = torch.tensor([10.0, -1e9, 10.0, -1e9])
    mask = torch.tensor([True, False, True, False])
    dist = MaskedCategorical(logits=logits, mask=mask)

    # Sample many times; should only get actions 0 or 2
    for _ in range(50):
        action = dist.sample().item()
        assert action in (0, 2), f"Got invalid action {action}"


def test_masked_categorical_log_prob():
    """MaskedCategorical log_prob is finite for valid actions."""
    logits = torch.zeros(5)
    mask = torch.ones(5, dtype=torch.bool)
    dist = MaskedCategorical(logits=logits, mask=mask)
    action = dist.sample()
    lp = dist.log_prob(action)
    assert torch.isfinite(lp)


def test_masked_categorical_entropy():
    """MaskedCategorical entropy is non-negative."""
    logits = torch.zeros(4)
    mask = torch.ones(4, dtype=torch.bool)
    dist = MaskedCategorical(logits=logits, mask=mask)
    ent = dist.entropy()
    assert ent.item() >= 0.0


def test_env_obs_compatible_with_network():
    """
    End-to-end: run a few env steps, build obs vector, pass through network.
    Uses global_features (dim=8) as a proxy flat observation.
    """
    env = MockCircuitEnv(num_edges=8, num_regions=2, seed=1)
    obs = env.reset()

    global_feat_dim = obs["global_features"].shape[0]  # 8
    actor = FlatActor(obs_dim=global_feat_dim, action_dim=env.num_edges + 1)

    for _ in range(3):
        import numpy as np
        obs_t = torch.from_numpy(obs["global_features"])
        logits = actor(obs_t)
        mask_np = env.valid_action_mask()
        mask_t = torch.from_numpy(mask_np)
        dist = MaskedCategorical(logits=logits, mask=mask_t)
        action = dist.sample().item()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    assert True  # No crash means pass


def test_ppo_loss_computation():
    """
    Minimal PPO surrogate loss computation: checks it runs without error
    and loss value is finite.
    """
    env = MockCircuitEnv(num_edges=8, num_regions=2, max_episode_steps=10, seed=3)
    env.reset()

    obs_dim = 8
    n_actions = env.num_edges + 1
    actor = FlatActor(obs_dim=obs_dim, action_dim=n_actions)
    critic = FlatCritic(obs_dim=obs_dim)

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=1e-3
    )

    # Fake a rollout buffer (T=8 steps)
    T = 8
    obs_batch = torch.zeros(T, obs_dim)
    actions = torch.randint(0, n_actions, (T,))
    old_log_probs = torch.full((T,), -2.0)
    advantages = torch.randn(T)
    returns = advantages + 0.5

    # PPO forward pass
    logits = actor(obs_batch)               # (T, n_actions)
    values = critic(obs_batch).squeeze(-1)  # (T,)

    log_probs = torch.distributions.Categorical(logits=logits).log_prob(actions)

    ratio = torch.exp(log_probs - old_log_probs)
    clip_eps = 0.2
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = (values - returns).pow(2).mean()
    loss = actor_loss + 0.5 * critic_loss

    assert torch.isfinite(loss), "Loss must be finite"

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
