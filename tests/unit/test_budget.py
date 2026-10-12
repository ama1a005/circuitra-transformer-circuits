"""
tests.unit.test_budget
======================

Unit tests for the evaluation budget system.

Tests cover:
* Counter increments correctly on each evaluation call.
* BudgetExhaustedError is raised when budget is exceeded.
* Remaining budget decreases correctly with each call.
* can_evaluate returns False when budget is insufficient.
"""

from __future__ import annotations

import pytest

from hrl_circuits.evaluation.budget import (
    BudgetExhaustedError,
    EvaluationBudget,
)


# ---------------------------------------------------------------------------
# test_budget_counter
# ---------------------------------------------------------------------------

class TestBudgetCounter:
    def test_budget_counter(self) -> None:
        """Counter increments by the correct amount on each consume call."""
        budget = EvaluationBudget(max_evaluations=10)
        assert budget.consumed == 0
        budget.consume(1)
        assert budget.consumed == 1
        budget.consume(3)
        assert budget.consumed == 4

    def test_budget_counter_starts_at_zero(self) -> None:
        """Freshly constructed budget has consumed == 0."""
        budget = EvaluationBudget(max_evaluations=5)
        assert budget.consumed == 0

    def test_budget_counter_multiple_increments(self) -> None:
        """Multiple single-unit increments accumulate correctly."""
        budget = EvaluationBudget(max_evaluations=100)
        for i in range(1, 11):
            budget.consume(1)
            assert budget.consumed == i


# ---------------------------------------------------------------------------
# test_budget_exhaustion
# ---------------------------------------------------------------------------

class TestBudgetExhaustion:
    def test_budget_exhaustion(self) -> None:
        """BudgetExhaustedError raised when budget is exceeded."""
        budget = EvaluationBudget(max_evaluations=3)
        budget.consume(3)
        with pytest.raises(BudgetExhaustedError):
            budget.consume(1)

    def test_budget_exhaustion_exact_limit_ok(self) -> None:
        """Consuming exactly the max should succeed without error."""
        budget = EvaluationBudget(max_evaluations=5)
        budget.consume(5)  # Should not raise
        assert budget.consumed == 5

    def test_budget_exhaustion_one_over(self) -> None:
        """Consuming max+1 should raise BudgetExhaustedError."""
        budget = EvaluationBudget(max_evaluations=5)
        with pytest.raises(BudgetExhaustedError):
            budget.consume(6)

    def test_budget_exhaustion_message_informative(self) -> None:
        """BudgetExhaustedError message should mention the limit."""
        budget = EvaluationBudget(max_evaluations=5)
        budget.consume(5)
        with pytest.raises(BudgetExhaustedError) as exc_info:
            budget.consume(1)
        assert "5" in str(exc_info.value) or "budget" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# test_budget_remaining
# ---------------------------------------------------------------------------

class TestBudgetRemaining:
    def test_budget_remaining(self) -> None:
        """Remaining decreases correctly with each consume call."""
        budget = EvaluationBudget(max_evaluations=10)
        assert budget.remaining == 10
        budget.consume(3)
        assert budget.remaining == 7
        budget.consume(2)
        assert budget.remaining == 5

    def test_budget_remaining_starts_at_max(self) -> None:
        """Initial remaining equals max_evaluations."""
        budget = EvaluationBudget(max_evaluations=42)
        assert budget.remaining == 42

    def test_budget_remaining_zero_at_exhaustion(self) -> None:
        """remaining == 0 exactly when budget is exhausted."""
        budget = EvaluationBudget(max_evaluations=5)
        budget.consume(5)
        assert budget.remaining == 0

    def test_budget_remaining_plus_consumed_equals_max(self) -> None:
        """remaining + consumed == max_evaluations at all times."""
        budget = EvaluationBudget(max_evaluations=20)
        for step in range(1, 11):
            budget.consume(1)
            assert budget.remaining + budget.consumed == 20


# ---------------------------------------------------------------------------
# test_budget_can_evaluate
# ---------------------------------------------------------------------------

class TestBudgetCanEvaluate:
    def test_budget_can_evaluate(self) -> None:
        """can_evaluate returns True when budget > 0, False when exhausted."""
        budget = EvaluationBudget(max_evaluations=3)
        assert budget.can_evaluate(1) is True
        budget.consume(3)
        assert budget.can_evaluate(1) is False

    def test_budget_can_evaluate_amount(self) -> None:
        """can_evaluate(n) is False if remaining < n."""
        budget = EvaluationBudget(max_evaluations=5)
        budget.consume(3)  # remaining = 2
        assert budget.can_evaluate(2) is True
        assert budget.can_evaluate(3) is False

    def test_budget_can_evaluate_zero_amount(self) -> None:
        """can_evaluate(0) should always return True (no budget consumed)."""
        budget = EvaluationBudget(max_evaluations=0)
        assert budget.can_evaluate(0) is True

    def test_budget_reset(self) -> None:
        """After reset(), consumed returns to 0 and remaining to max."""
        budget = EvaluationBudget(max_evaluations=10)
        budget.consume(7)
        budget.reset()
        assert budget.consumed == 0
        assert budget.remaining == 10
