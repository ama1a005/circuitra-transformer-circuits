"""
tests.unit.test_reward
======================

Unit tests for the reward computation module.

Tests verify:
* Exact arithmetic for hand-computed reward values.
* Potential-based shaping: r = Phi(next) - Phi(current).
* RewardBreakdown components sum to total.
* Terminal success bonus applied correctly.
* Terminal failure penalty applied correctly.
"""

from __future__ import annotations

import math

import pytest

from hrl_circuits.reward.compute import (
    RewardConfig,
    compute_reward,
    compute_terminal_reward,
    potential,
)
from hrl_circuits.types import RewardBreakdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_cfg(**kwargs: float) -> RewardConfig:
    """Build a RewardConfig with sensible defaults, overriding with kwargs."""
    defaults = dict(
        alpha_faithfulness=1.0,
        beta_sparsity=0.3,
        gamma_cost=0.01,
        guard_threshold=0.5,
        terminal_success_bonus=1.0,
        terminal_failure_penalty=-1.0,
    )
    defaults.update(kwargs)
    return RewardConfig(**defaults)


# ---------------------------------------------------------------------------
# test_reward_arithmetic_exact
# ---------------------------------------------------------------------------

class TestRewardArithmeticExact:
    def test_reward_arithmetic_exact(self) -> None:
        """Verify reward against hand-computed values.

        Setup:
            alpha=1.0, beta=0.3, gamma=0.01
            prev_faithfulness=0.8, next_faithfulness=0.85  → delta_faith=+0.05
            prev_live=10,  next_live=8  (2 pruned), total=10  → delta_sparse=0.2
            eval_cost=1  → delta_cost = -gamma * 1 = -0.01
            guard_threshold=0.5, faithfulness=0.85 → guard_penalty=0.0

        Expected:
            total = 1.0*0.05 + 0.3*0.2 + (-0.01) + 0.0 = 0.05 + 0.06 - 0.01 = 0.10
        """
        cfg = _default_cfg()
        rb = compute_reward(
            cfg=cfg,
            prev_faithfulness=0.8,
            next_faithfulness=0.85,
            prev_live_count=10,
            next_live_count=8,
            total_edges=10,
            eval_cost=1,
            is_terminal=False,
            success=False,
        )
        assert rb.delta_faithfulness == pytest.approx(0.05, abs=1e-9)
        assert rb.delta_sparsity == pytest.approx(0.06, abs=1e-9)
        assert rb.delta_cost == pytest.approx(-0.01, abs=1e-9)
        assert rb.guard_penalty == pytest.approx(0.0, abs=1e-9)
        assert rb.total == pytest.approx(0.10, abs=1e-9)

    def test_reward_below_guard_threshold(self) -> None:
        """guard_penalty is applied when faithfulness < guard_threshold.

        Setup:
            threshold=0.5, next_faithfulness=0.3 → guard_penalty = -lambda
        """
        cfg = _default_cfg(guard_threshold=0.5)
        rb = compute_reward(
            cfg=cfg,
            prev_faithfulness=0.6,
            next_faithfulness=0.3,
            prev_live_count=10,
            next_live_count=9,
            total_edges=10,
            eval_cost=0,
            is_terminal=False,
            success=False,
        )
        assert rb.guard_penalty < 0.0, "Expected a negative guard penalty"

    def test_reward_no_prune_no_eval(self) -> None:
        """Reward is 0 for a STOP action with no change and no evaluation cost."""
        cfg = _default_cfg()
        rb = compute_reward(
            cfg=cfg,
            prev_faithfulness=0.9,
            next_faithfulness=0.9,
            prev_live_count=10,
            next_live_count=10,
            total_edges=10,
            eval_cost=0,
            is_terminal=False,
            success=False,
        )
        assert rb.delta_faithfulness == pytest.approx(0.0)
        assert rb.delta_sparsity == pytest.approx(0.0)
        assert rb.delta_cost == pytest.approx(0.0)
        assert rb.total == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# test_potential_difference
# ---------------------------------------------------------------------------

class TestPotentialDifference:
    def test_potential_difference(self) -> None:
        """Reward equals Phi(next) - Phi(current) for potential-based shaping.

        The potential function is Phi(s) = alpha * faithfulness + beta * sparsity.
        """
        cfg = _default_cfg(gamma_cost=0.0)  # No cost term for clarity
        alpha = cfg.alpha_faithfulness
        beta = cfg.beta_sparsity
        total = 10

        prev_faith = 0.7
        next_faith = 0.8
        prev_live = 10
        next_live = 7

        phi_prev = alpha * prev_faith + beta * (1 - prev_live / total)
        phi_next = alpha * next_faith + beta * (1 - next_live / total)
        expected_shaped = phi_next - phi_prev

        rb = compute_reward(
            cfg=cfg,
            prev_faithfulness=prev_faith,
            next_faithfulness=next_faith,
            prev_live_count=prev_live,
            next_live_count=next_live,
            total_edges=total,
            eval_cost=0,
            is_terminal=False,
            success=False,
        )
        # Total (ex guard) should match potential difference
        shaped = rb.delta_faithfulness + rb.delta_sparsity
        assert shaped == pytest.approx(expected_shaped, abs=1e-9)

    def test_potential_monotone_sparsity(self) -> None:
        """Removing edges always increases the sparsity component."""
        cfg = _default_cfg()
        rb = compute_reward(
            cfg=cfg,
            prev_faithfulness=0.9,
            next_faithfulness=0.9,
            prev_live_count=10,
            next_live_count=5,
            total_edges=10,
            eval_cost=0,
            is_terminal=False,
            success=False,
        )
        assert rb.delta_sparsity > 0.0


# ---------------------------------------------------------------------------
# test_reward_breakdown_sums
# ---------------------------------------------------------------------------

class TestRewardBreakdownSums:
    def test_reward_breakdown_sums(self) -> None:
        """total must equal the sum of all components in RewardBreakdown."""
        cfg = _default_cfg()
        rb = compute_reward(
            cfg=cfg,
            prev_faithfulness=0.8,
            next_faithfulness=0.7,
            prev_live_count=8,
            next_live_count=6,
            total_edges=10,
            eval_cost=2,
            is_terminal=False,
            success=False,
        )
        component_sum = (
            rb.delta_faithfulness
            + rb.delta_sparsity
            + rb.delta_cost
            + rb.guard_penalty
            + rb.terminal_component
        )
        assert rb.total == pytest.approx(component_sum, abs=1e-9)

    def test_reward_breakdown_sums_terminal(self) -> None:
        """Breakdown sums to total even when terminal bonus is non-zero."""
        cfg = _default_cfg()
        rb = compute_reward(
            cfg=cfg,
            prev_faithfulness=0.9,
            next_faithfulness=0.95,
            prev_live_count=6,
            next_live_count=4,
            total_edges=10,
            eval_cost=1,
            is_terminal=True,
            success=True,
        )
        component_sum = (
            rb.delta_faithfulness
            + rb.delta_sparsity
            + rb.delta_cost
            + rb.guard_penalty
            + rb.terminal_component
        )
        assert rb.total == pytest.approx(component_sum, abs=1e-9)


# ---------------------------------------------------------------------------
# test_terminal_success_bonus
# ---------------------------------------------------------------------------

class TestTerminalSuccessBonus:
    def test_terminal_success_bonus(self) -> None:
        """Terminal success bonus (>0) is added on successful termination."""
        cfg = _default_cfg(terminal_success_bonus=2.0)
        rb = compute_reward(
            cfg=cfg,
            prev_faithfulness=0.9,
            next_faithfulness=0.9,
            prev_live_count=5,
            next_live_count=5,
            total_edges=10,
            eval_cost=0,
            is_terminal=True,
            success=True,
        )
        assert rb.terminal_component > 0.0
        assert rb.terminal_component == pytest.approx(2.0, abs=1e-9)

    def test_no_terminal_bonus_on_nonterminal(self) -> None:
        """terminal_component is 0 for non-terminal steps."""
        cfg = _default_cfg(terminal_success_bonus=2.0)
        rb = compute_reward(
            cfg=cfg,
            prev_faithfulness=0.9,
            next_faithfulness=0.9,
            prev_live_count=5,
            next_live_count=5,
            total_edges=10,
            eval_cost=0,
            is_terminal=False,
            success=False,
        )
        assert rb.terminal_component == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# test_terminal_failure_penalty
# ---------------------------------------------------------------------------

class TestTerminalFailurePenalty:
    def test_terminal_failure_penalty(self) -> None:
        """Terminal failure penalty (<0) is applied on failed termination."""
        cfg = _default_cfg(terminal_failure_penalty=-1.5)
        rb = compute_reward(
            cfg=cfg,
            prev_faithfulness=0.9,
            next_faithfulness=0.3,
            prev_live_count=10,
            next_live_count=10,
            total_edges=10,
            eval_cost=0,
            is_terminal=True,
            success=False,
        )
        assert rb.terminal_component < 0.0
        assert rb.terminal_component == pytest.approx(-1.5, abs=1e-9)

    def test_failure_penalty_not_applied_on_success(self) -> None:
        """Failure penalty is not applied when success=True."""
        cfg = _default_cfg(terminal_failure_penalty=-5.0, terminal_success_bonus=1.0)
        rb = compute_reward(
            cfg=cfg,
            prev_faithfulness=0.9,
            next_faithfulness=0.9,
            prev_live_count=5,
            next_live_count=5,
            total_edges=10,
            eval_cost=0,
            is_terminal=True,
            success=True,
        )
        # terminal_component should be the success bonus, not the failure penalty
        assert rb.terminal_component > 0.0
