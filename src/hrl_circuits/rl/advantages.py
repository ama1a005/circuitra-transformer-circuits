"""
Generalised Advantage Estimation (GAE) for flat PPO and semi-MDP HRL manager.

References
----------
* Schulman et al. (2015) – "High-Dimensional Continuous Control Using
  Generalised Advantage Estimation"
* Sutton et al. (1999) – "Between MDPs and Semi-MDPs"
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    next_value: "Tensor | float | None" = None,
    gamma: float = 0.99,
    lam: float = 0.95,
    *,
    gae_lambda: "float | None" = None,
) -> Tensor:

    """
    Compute GAE advantages and discounted returns for a flat MDP rollout.

    Parameters
    ----------
    rewards : Tensor, shape (T,)
        Per-step rewards.
    values : Tensor, shape (T,)
        Bootstrapped value estimates V(s_t).  The final entry is treated as
        the bootstrap value for the step *after* the rollout ends.
    dones : Tensor, shape (T,)
        Episode-terminal flags (1.0 = done, 0.0 = continuing).
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE λ trade-off between bias and variance.

    Returns
    -------
    advantages : Tensor, shape (T-1,)
    returns    : Tensor, shape (T-1,)

    Notes
    -----
    ``values`` must have length T; the last element ``values[-1]`` serves as
    the bootstrap value V(s_T).  ``rewards`` and ``dones`` have length T-1.
    """
    rewards = rewards.float()
    values = values.float()
    dones = dones.float()

    # Resolve lambda alias: gae_lambda kwarg takes precedence over lam positional
    _lam = gae_lambda if gae_lambda is not None else lam

    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=torch.float32)

    # Resolve bootstrap value for the step after the rollout
    if next_value is None:
        # Fall back to last element of values if values has T+1 elements
        _next_val = values[T] if values.shape[0] > T else torch.tensor(0.0)
    elif isinstance(next_value, (int, float)):
        _next_val = torch.tensor(float(next_value))
    else:
        _next_val = next_value.float()

    gae_acc: float = 0.0
    for t in reversed(range(T)):
        if t + 1 < T:
            v_next = values[t + 1]
        else:
            v_next = _next_val
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * v_next * next_non_terminal - values[t]
        gae_acc = float(delta) + gamma * _lam * next_non_terminal.item() * gae_acc
        advantages[t] = gae_acc

    return advantages



def compute_manager_gae(
    option_rewards: Tensor,
    manager_values: Tensor,
    dones: Tensor,
    option_durations: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """
    Duration-aware GAE for a semi-MDP manager policy.

    Each "step" k corresponds to one completed option of duration d_k.  The
    temporal-difference target uses gamma raised to the power d_k:

        δ_k = R_option_k + γ^{d_k} * (1 - done_k) * V(s_{k+1}) - V(s_k)

    The eligibility trace decays at rate (γ · λ)^{d_k} per option step.

    Parameters
    ----------
    option_rewards : Tensor, shape (K,)
        Discounted cumulative primitive reward for each option.
    manager_values : Tensor, shape (K+1,)
        V_manager at every option boundary (last = bootstrap).
    dones : Tensor, shape (K,)
        1.0 if the episode terminated at the end of option k.
    option_durations : Tensor, shape (K,)
        Number of primitive steps d_k in each option (int64 or float).
    gamma : float
        Base discount factor (applied per primitive step).
    gae_lambda : float
        GAE λ.

    Returns
    -------
    advantages : Tensor, shape (K,)
    returns    : Tensor, shape (K,)
    """
    option_rewards = option_rewards.float()
    manager_values = manager_values.float()
    dones = dones.float()
    durations = option_durations.float()

    K = option_rewards.shape[0]
    advantages = torch.zeros(K, dtype=torch.float32)

    gae: float = 0.0
    for k in reversed(range(K)):
        d_k = durations[k].item()
        gamma_d = gamma ** d_k
        gl_d = (gamma * gae_lambda) ** d_k

        next_value = (
            manager_values[k + 1] if k + 1 < manager_values.shape[0] else torch.tensor(0.0)
        )
        next_non_terminal = 1.0 - dones[k].item()

        delta = (
            option_rewards[k].item()
            + gamma_d * next_non_terminal * next_value.item()
            - manager_values[k].item()
        )
        gae = delta + gl_d * next_non_terminal * gae
        advantages[k] = gae

    returns = advantages + manager_values[:K]
    return advantages, returns


def normalize_advantages(advantages: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Standardise advantages to zero mean and unit variance.

    Parameters
    ----------
    advantages : Tensor, shape (N,)
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    Tensor, same shape as *advantages*.
    """
    mean = advantages.mean()
    std = advantages.std(unbiased=False)
    return (advantages - mean) / (std + eps)
