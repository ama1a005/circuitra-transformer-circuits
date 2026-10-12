"""
PPO core loss functions.

All functions accept plain :class:`torch.Tensor` inputs and return scalar
tensors suitable for backpropagation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def ppo_clip_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    clip_coef: float = 0.2,
) -> Tensor:
    """
    Clipped surrogate objective (PPO-Clip).

    L^{CLIP} = E[ min(r_t · Â_t,  clip(r_t, 1-ε, 1+ε) · Â_t) ]

    where r_t = exp(log π(a|s) - log π_old(a|s)).

    Parameters
    ----------
    log_probs : Tensor, shape (N,)
        Log-probabilities under the *current* policy.
    old_log_probs : Tensor, shape (N,)
        Log-probabilities recorded during data collection.
    advantages : Tensor, shape (N,)
        Normalised advantage estimates.
    clip_coef : float
        ε – clipping range (default 0.2).

    Returns
    -------
    Tensor
        Scalar surrogate loss (negated so gradient *ascent* becomes descent).
    """
    ratios = torch.exp(log_probs - old_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
    return -torch.mean(torch.min(surr1, surr2))


def value_loss(
    values: Tensor,
    returns: Tensor,
) -> Tensor:
    """
    Mean squared error between predicted values and empirical returns.

    Parameters
    ----------
    values : Tensor, shape (N,)
        Current value-function predictions.
    returns : Tensor, shape (N,)
        Target returns (advantages + old values, or plain MC returns).

    Returns
    -------
    Tensor
        Scalar MSE loss.
    """
    return F.mse_loss(values, returns)


def entropy_bonus(entropy: Tensor) -> Tensor:
    """
    Mean policy entropy across the minibatch.

    Maximising entropy encourages exploration; the caller negates this term
    when combining with the total loss (loss = ... - entropy_coef * bonus).

    Parameters
    ----------
    entropy : Tensor, shape (N,) or scalar
        Per-sample or batch-mean entropy.

    Returns
    -------
    Tensor
        Scalar mean entropy.
    """
    return entropy.mean()


def total_ppo_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    values: Tensor,
    returns: Tensor,
    entropy: Tensor,
    clip_coef: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> tuple[Tensor, dict[str, Any]]:
    """
    Combine PPO-Clip, value, and entropy losses into a single scalar.

    total_loss = L^{CLIP} + value_coef * L^{VF} - entropy_coef * H

    Parameters
    ----------
    log_probs : Tensor, shape (N,)
    old_log_probs : Tensor, shape (N,)
    advantages : Tensor, shape (N,)
    values : Tensor, shape (N,)
    returns : Tensor, shape (N,)
    entropy : Tensor, shape (N,) or scalar
    clip_coef : float
    value_coef : float
    entropy_coef : float

    Returns
    -------
    total_loss : Tensor  (scalar)
    info : dict
        Breakdown of individual losses and the approximate KL divergence
        for monitoring.
    """
    actor_loss = ppo_clip_loss(log_probs, old_log_probs, advantages, clip_coef)
    v_loss = value_loss(values, returns)
    e_bonus = entropy_bonus(entropy)

    total = actor_loss + value_coef * v_loss - entropy_coef * e_bonus

    # Approximate KL (Schulman 2017 blog) for early-stopping diagnostics
    with torch.no_grad():
        approx_kl = ((old_log_probs - log_probs).mean()).item()
        clip_fraction = (
            (torch.abs(torch.exp(log_probs - old_log_probs) - 1.0) > clip_coef)
            .float()
            .mean()
            .item()
        )

    info: dict[str, Any] = {
        "actor_loss": actor_loss.item(),
        "value_loss": v_loss.item(),
        "entropy_bonus": e_bonus.item(),
        "total_loss": total.item(),
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }

    return total, info
