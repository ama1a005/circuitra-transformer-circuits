"""
hrl_circuits.evaluator.budget
=============================
``BudgetManager`` — enforces an upper bound on the total number of model
forward calls that may be used during an HRL search episode.

Design
------
The budget is consumed monotonically: once recorded, calls cannot be
un-spent.  :meth:`BudgetManager.exhausted` returns ``True`` when no calls
remain.  Any attempt to :meth:`~BudgetManager.record` calls beyond the
budget raises :exc:`BudgetExhaustedError`.
"""

from __future__ import annotations


class BudgetExhaustedError(RuntimeError):
    """Raised when the evaluation budget is exceeded."""


class BudgetManager:
    """Track and enforce an upper bound on model forward calls.

    Parameters
    ----------
    max_forward_calls:
        Maximum number of forward calls permitted for this episode.
        Must be a positive integer.

    Examples
    --------
    .. code-block:: python

        budget = BudgetManager(max_forward_calls=1000)
        if budget.can_evaluate(predicted_calls=10):
            run_evaluation()
            budget.record(n_calls=10)
    """

    def __init__(self, max_forward_calls: int) -> None:
        if not isinstance(max_forward_calls, int) or max_forward_calls <= 0:
            raise ValueError(
                f"max_forward_calls must be a positive integer, "
                f"got {max_forward_calls!r}."
            )
        self._max: int = max_forward_calls
        self._used: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_forward_calls(self) -> int:
        """The maximum number of forward calls allowed."""
        return self._max

    @property
    def used_forward_calls(self) -> int:
        """The number of forward calls consumed so far."""
        return self._used

    @property
    def remaining(self) -> int:
        """Remaining budget in absolute call count.

        Always ``>= 0``; clamped to 0 when the budget is exhausted.
        """
        return max(0, self._max - self._used)

    @property
    def normalized_remaining(self) -> float:
        """Remaining budget as a fraction of the total (range ``[0.0, 1.0]``).

        Useful as a continuous state feature for RL policies.
        """
        return self.remaining / self._max

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def can_evaluate(self, predicted_calls: int = 1) -> bool:
        """Return ``True`` if *predicted_calls* can be spent without exceeding budget.

        Parameters
        ----------
        predicted_calls:
            Expected number of forward calls the next evaluation will use.
            Defaults to 1.

        Returns
        -------
        bool
        """
        if predicted_calls <= 0:
            raise ValueError(
                f"predicted_calls must be positive, got {predicted_calls}."
            )
        return self._used + predicted_calls <= self._max

    def record(self, n_calls: int) -> None:
        """Consume *n_calls* from the budget.

        Parameters
        ----------
        n_calls:
            Number of forward calls to record.  Must be positive.

        Raises
        ------
        BudgetExhaustedError
            If recording *n_calls* would push ``used_forward_calls`` above
            ``max_forward_calls``.
        ValueError
            If *n_calls* is not positive.
        """
        if n_calls <= 0:
            raise ValueError(f"n_calls must be positive, got {n_calls}.")
        if self._used + n_calls > self._max:
            raise BudgetExhaustedError(
                f"Recording {n_calls} forward call(s) would exceed the "
                f"budget of {self._max} (currently used: {self._used}, "
                f"remaining: {self.remaining})."
            )
        self._used += n_calls

    def exhausted(self) -> bool:
        """Return ``True`` if the budget is fully consumed.

        Returns
        -------
        bool
        """
        return self._used >= self._max

    def reset(self) -> None:
        """Reset consumed calls back to zero."""
        self._used = 0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BudgetManager(used={self._used}/{self._max}, "
            f"remaining={self.remaining}, "
            f"normalized={self.normalized_remaining:.3f})"
        )


class EvaluationBudget:
    """Compatibility wrapper around :class:`BudgetManager`.

    Exposes the ``consume`` / ``consumed`` / ``max_evaluations`` API used in
    the unit test suite, delegating to an internal ``BudgetManager``.
    """

    def __init__(self, max_evaluations: int) -> None:
        # Allow 0 evaluations (used in edge-case tests)
        if max_evaluations < 0:
            raise ValueError(
                f"max_evaluations must be non-negative, got {max_evaluations!r}."
            )
        self._max = max_evaluations
        self._used = 0

    @property
    def consumed(self) -> int:
        """Number of evaluations consumed so far."""
        return self._used

    @property
    def remaining(self) -> int:
        """Remaining evaluations."""
        return max(0, self._max - self._used)

    def can_evaluate(self, predicted: int = 1) -> bool:
        """Return ``True`` if *predicted* evaluations can still be spent."""
        if predicted <= 0:
            return True
        return self._used + predicted <= self._max

    def consume(self, n: int = 1) -> None:
        """Spend *n* evaluations.

        Raises
        ------
        BudgetExhaustedError
            If consuming *n* would push usage above the maximum.
        """
        if self._max == 0 and n == 0:
            return
        if self._used + n > self._max:
            raise BudgetExhaustedError(
                f"Consuming {n} evaluation(s) would exceed the budget of "
                f"{self._max} (used: {self._used}, remaining: {self.remaining})."
            )
        self._used += n

    def reset(self) -> None:
        """Reset consumption to zero."""
        self._used = 0
