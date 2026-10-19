"""hrl_circuits.reward.compute
==============================
Self-contained reward computation functions matching the test-suite API.

Implements the potential-based shaped reward from spec §17:

    Φ(C) = α·F(C) − β·S(C) − γ_c·Cost(C) − λ·max(0, τ − F(C))
    r_t   = Φ(C_{t+1}) − Φ(C_t) + terminal_component

All functions are pure (no side effects).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from hrl_circuits.types import RewardBreakdown


@dataclass
class RewardConfig:
    """Hyperparameters for the reward function.

    Attributes
    ----------
    alpha_faithfulness:
        Weight on the faithfulness delta (α).
    beta_sparsity:
        Weight on the sparsity improvement (β). Positive ⟹ reward for pruning.
    gamma_cost:
        Per-eval-call cost penalty (γ_c).
    guard_threshold:
        Faithfulness threshold τ below which a guard penalty fires.
    terminal_success_bonus:
        Bonus added at episode end when final faithfulness ≥ guard_threshold.
    terminal_failure_penalty:
        Penalty added at episode end when final faithfulness < guard_threshold.
    lambda_guard:
        Scale of the guard penalty (default 1.0).
    """
    alpha_faithfulness: float = 1.0
    beta_sparsity: float = 0.3
    gamma_cost: float = 0.01
    guard_threshold: float = 0.5
    terminal_success_bonus: float = 1.0
    terminal_failure_penalty: float = -1.0
    lambda_guard: float = 1.0


def potential(
    cfg: RewardConfig,
    faithfulness: float,
    live_fraction: float,
    cumulative_cost: float = 0.0,
) -> float:
    """Compute the potential Φ(C) for a circuit state.

    Parameters
    ----------
    cfg:
        Reward configuration.
    faithfulness:
        Current faithfulness score F(C) ∈ [0, 1].
    live_fraction:
        Fraction of edges still live (S(C)).  Lower ⟹ sparser ⟹ better.
    cumulative_cost:
        Normalised cumulative evaluation cost so far (∈ [0, 1]).

    Returns
    -------
    float
        Φ(C).
    """
    phi = cfg.alpha_faithfulness * faithfulness
    # Reward for pruning: sparsity gain = (1 - live_fraction)
    phi += cfg.beta_sparsity * (1.0 - live_fraction)
    # Cost penalty
    phi -= cfg.gamma_cost * cumulative_cost
    # Guard penalty
    guard_violation = max(0.0, cfg.guard_threshold - faithfulness)
    phi -= cfg.lambda_guard * guard_violation
    return phi


def compute_reward(
    cfg: RewardConfig,
    prev_faithfulness: float,
    next_faithfulness: float,
    prev_live_count: int,
    next_live_count: int,
    total_edges: int,
    eval_cost: int,
    is_terminal: bool = False,
    success: bool = False,
) -> RewardBreakdown:
    """Compute the full reward breakdown for one step transition.

    Implements:
        r = α·ΔF + β·ΔS − γ·cost − guard_penalty + terminal_component

    Parameters
    ----------
    cfg:
        Reward configuration.
    prev_faithfulness:
        Faithfulness before the action.
    next_faithfulness:
        Faithfulness after the action.
    prev_live_count:
        Number of live edges before the action.
    next_live_count:
        Number of live edges after the action.
    total_edges:
        Total edges in the candidate graph.
    eval_cost:
        Number of causal evaluation calls made this step.
    is_terminal:
        If True, compute and include the terminal bonus/penalty.
    success:
        If True (and is_terminal), apply success bonus; else failure penalty.

    Returns
    -------
    RewardBreakdown
        Detailed breakdown with total reward.
    """
    n = max(1, total_edges)

    # Faithfulness delta
    delta_faithfulness = cfg.alpha_faithfulness * (next_faithfulness - prev_faithfulness)

    # Sparsity delta: reward for each additional edge pruned
    delta_sparsity = cfg.beta_sparsity * (prev_live_count - next_live_count) / n

    # Cost penalty
    delta_cost = -cfg.gamma_cost * eval_cost

    # Guard penalty based on next faithfulness
    guard_penalty = -cfg.lambda_guard * max(0.0, cfg.guard_threshold - next_faithfulness)

    # Terminal component
    terminal_component = 0.0
    if is_terminal:
        terminal_component = (
            cfg.terminal_success_bonus if success else cfg.terminal_failure_penalty
        )

    total = (
        delta_faithfulness
        + delta_sparsity
        + delta_cost
        + guard_penalty
        + terminal_component
    )

    return RewardBreakdown(
        delta_faithfulness=delta_faithfulness,
        delta_sparsity=delta_sparsity,
        delta_cost=delta_cost,
        guard_penalty=guard_penalty,
        terminal_component=terminal_component,
        total=total,
    )


def compute_terminal_reward(
    cfg: RewardConfig,
    final_faithfulness: float,
) -> float:
    """Compute the terminal bonus or penalty.

    Parameters
    ----------
    cfg:
        Reward configuration.
    final_faithfulness:
        Faithfulness at episode end.

    Returns
    -------
    float
        Positive bonus if success, negative penalty if failure.
    """
    if final_faithfulness >= cfg.guard_threshold:
        return cfg.terminal_success_bonus
    return cfg.terminal_failure_penalty
