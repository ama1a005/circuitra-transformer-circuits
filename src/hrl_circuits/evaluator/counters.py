"""
hrl_circuits.evaluator.counters
===============================
``EvalCounter`` — tracks resource consumption during causal evaluation and
gradient-based attribution passes.

Spec reference: section 18

Usage
-----
.. code-block:: python

    counter = EvalCounter()
    counter.increment_causal(n_calls=2, n_examples=16)
    counter.increment_attribution(n_calls=1)
    print(counter.causal_eval_invocations)   # 1
    print(counter.model_forward_calls_causal) # 2
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class EvalCounter:
    """Accumulates evaluation resource-usage statistics.

    All fields start at 0.  Use the ``increment_*`` methods to update;
    never mutate fields directly.

    Attributes
    ----------
    causal_eval_invocations:
        Number of times :meth:`~hrl_circuits.evaluator.causal.CausalEvaluator.evaluate`
        has been called (each call may process many forward passes).
    model_forward_calls_causal:
        Total number of model forward passes in causal (patching) mode.
    examples_processed_causal:
        Total number of data examples fed through causal forward passes.
    model_backward_calls_attribution:
        Number of backward (gradient) passes used for attribution scoring.
    """

    causal_eval_invocations: int = 0
    model_forward_calls_causal: int = 0
    examples_processed_causal: int = 0
    model_backward_calls_attribution: int = 0

    # ------------------------------------------------------------------
    # Mutation helpers (return self for chaining)
    # ------------------------------------------------------------------

    def increment_causal(self, n_calls: int, n_examples: int) -> "EvalCounter":
        """Record one causal evaluation invocation.

        Parameters
        ----------
        n_calls:
            Number of model forward passes in this invocation.
        n_examples:
            Number of examples processed in this invocation.

        Returns
        -------
        EvalCounter
            ``self`` (mutated in place), for chaining.
        """
        if n_calls < 0:
            raise ValueError(f"n_calls must be non-negative, got {n_calls}.")
        if n_examples < 0:
            raise ValueError(f"n_examples must be non-negative, got {n_examples}.")

        self.causal_eval_invocations += 1
        self.model_forward_calls_causal += n_calls
        self.examples_processed_causal += n_examples
        return self

    def increment_attribution(self, n_calls: int) -> "EvalCounter":
        """Record attribution backward passes.

        Parameters
        ----------
        n_calls:
            Number of backward passes in this attribution call.

        Returns
        -------
        EvalCounter
            ``self`` (mutated in place), for chaining.
        """
        if n_calls < 0:
            raise ValueError(f"n_calls must be non-negative, got {n_calls}.")
        self.model_backward_calls_attribution += n_calls
        return self

    def reset(self) -> "EvalCounter":
        """Reset all counters to zero.

        Returns
        -------
        EvalCounter
            ``self`` (mutated in place), for chaining.
        """
        self.causal_eval_invocations = 0
        self.model_forward_calls_causal = 0
        self.examples_processed_causal = 0
        self.model_backward_calls_attribution = 0
        return self

    def copy(self) -> "EvalCounter":
        """Return an independent copy of this counter.

        Returns
        -------
        EvalCounter
            New instance with the same field values.
        """
        return EvalCounter(
            causal_eval_invocations=self.causal_eval_invocations,
            model_forward_calls_causal=self.model_forward_calls_causal,
            examples_processed_causal=self.examples_processed_causal,
            model_backward_calls_attribution=self.model_backward_calls_attribution,
        )

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "EvalCounter") -> "EvalCounter":
        """Sum two counters into a new :class:`EvalCounter`.

        Useful for aggregating across multiple evaluation episodes.

        Parameters
        ----------
        other:
            Another :class:`EvalCounter` to add to this one.

        Returns
        -------
        EvalCounter
            New instance with summed fields.

        Raises
        ------
        TypeError
            If *other* is not an :class:`EvalCounter`.
        """
        if not isinstance(other, EvalCounter):
            return NotImplemented
        return EvalCounter(
            causal_eval_invocations=(
                self.causal_eval_invocations + other.causal_eval_invocations
            ),
            model_forward_calls_causal=(
                self.model_forward_calls_causal + other.model_forward_calls_causal
            ),
            examples_processed_causal=(
                self.examples_processed_causal + other.examples_processed_causal
            ),
            model_backward_calls_attribution=(
                self.model_backward_calls_attribution
                + other.model_backward_calls_attribution
            ),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"EvalCounter("
            f"causal_invocations={self.causal_eval_invocations}, "
            f"fwd_causal={self.model_forward_calls_causal}, "
            f"examples={self.examples_processed_causal}, "
            f"bwd_attribution={self.model_backward_calls_attribution})"
        )
