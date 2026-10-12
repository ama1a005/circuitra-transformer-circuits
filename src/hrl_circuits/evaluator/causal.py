"""
hrl_circuits.evaluator.causal
=============================
Abstract base class and mock implementation for causal (activation-
patching) evaluation of circuit masks.

Architecture
------------
:class:`CausalEvaluator`
    Abstract base.  Concrete subclasses connect to a real model via
    hooks (e.g. TransformerLens, baukit).

:class:`MockCausalEvaluator`
    Deterministic stand-in for unit testing and development.  It uses
    ``ground_truth_edges`` — the set of edge IDs that form the true
    circuit — to compute a synthetic faithfulness score: the Jaccard
    similarity between the live mask and the ground truth set.

:class:`EvalResult`
    Structured return value: faithfulness score, normalised score, timing,
    and per-edge attribution (optional).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .counters import EvalCounter
from .faithfulness import normalize_faithfulness


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Return type for :meth:`CausalEvaluator.evaluate`.

    Attributes
    ----------
    raw_faithfulness:
        Raw faithfulness metric value (e.g. KL divergence, lower = more
        faithful in absolute sense).
    normalised_faithfulness:
        Faithfulness mapped to ``[0.0, 1.0]`` (higher = more faithful).
    wall_time_seconds:
        Elapsed wall-clock time for the evaluation call.
    num_forward_calls:
        Number of model forward passes used.
    num_examples:
        Number of data examples processed.
    per_edge_attribution:
        Optional dict mapping ``edge_id`` → attribution score.
    extra:
        Free-form metadata for debugging.
    """

    raw_faithfulness: float
    normalised_faithfulness: float
    wall_time_seconds: float
    num_forward_calls: int
    num_examples: int
    per_edge_attribution: dict[int, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class CausalEvaluator(ABC):
    """Abstract base for causal circuit evaluators.

    All concrete implementations must override :meth:`evaluate`.
    """

    @abstractmethod
    def evaluate(
        self,
        mask: np.ndarray,
        batch: Any,
        counter: EvalCounter,
    ) -> EvalResult:
        """Run a causal evaluation of the circuit defined by *mask*.

        Parameters
        ----------
        mask:
            Binary mask array (uint8, shape ``(num_edges,)``).  A value
            of 1 means the edge is live.
        batch:
            A batch of clean + counterfactual examples.  The exact type
            depends on the concrete implementation (typically a dict or
            a dataclass with ``input_ids``, ``counterfactual_ids``, etc.).
        counter:
            :class:`~hrl_circuits.evaluator.counters.EvalCounter` to update
            in-place with the resource usage of this call.

        Returns
        -------
        EvalResult
            Evaluation result with faithfulness scores and timing.
        """
        ...


# ---------------------------------------------------------------------------
# Mock implementation
# ---------------------------------------------------------------------------

class MockCausalEvaluator(CausalEvaluator):
    """Deterministic mock evaluator for testing and development.

    Faithfulness is computed as the Jaccard similarity between the set of
    live edges (``mask == 1``) and the provided *ground_truth_edges*::

        jaccard = |live ∩ truth| / |live ∪ truth|

    This is 1.0 when the mask exactly matches the ground truth, and 0.0
    when they are disjoint.

    The raw faithfulness is ``1 - jaccard`` (so lower divergence means
    higher similarity), and normalised faithfulness is ``jaccard`` itself.

    Parameters
    ----------
    ground_truth_edges:
        Set of edge IDs that constitute the true circuit.
    forward_calls_per_eval:
        Simulated number of forward passes per evaluation (for counter
        and budget tracking).  Defaults to 2 (one clean, one patched).
    """

    def __init__(
        self,
        ground_truth_edges: set[int],
        forward_calls_per_eval: int = 2,
    ) -> None:
        if not isinstance(ground_truth_edges, set):
            ground_truth_edges = set(ground_truth_edges)
        self._ground_truth: frozenset[int] = frozenset(ground_truth_edges)
        self._forward_calls_per_eval: int = forward_calls_per_eval

    @property
    def ground_truth_edges(self) -> frozenset[int]:
        """The ground-truth edge set (read-only)."""
        return self._ground_truth

    def evaluate(
        self,
        mask: np.ndarray,
        batch: Any,
        counter: EvalCounter,
    ) -> EvalResult:
        """Evaluate the mask against the ground truth deterministically.

        Parameters
        ----------
        mask:
            Binary mask array.
        batch:
            Ignored by the mock.  May be ``None`` or any value.
        counter:
            Updated in-place.

        Returns
        -------
        EvalResult
        """
        t0 = time.perf_counter()

        live_indices = set(int(i) for i in np.where(mask)[0])
        ground = set(self._ground_truth)

        intersection = live_indices & ground
        union = live_indices | ground

        if not union:
            # Both sets are empty → perfect (trivial) match
            jaccard = 1.0
        else:
            jaccard = len(intersection) / len(union)

        raw_faithfulness = 1.0 - jaccard          # 0 = perfect, 1 = disjoint
        normalised = float(np.clip(jaccard, 0.0, 1.0))

        # Simulate some examples equal to batch size or a default
        n_examples = _batch_size(batch)

        wall_time = time.perf_counter() - t0

        # Update the counter
        counter.increment_causal(
            n_calls=self._forward_calls_per_eval,
            n_examples=n_examples,
        )

        return EvalResult(
            raw_faithfulness=raw_faithfulness,
            normalised_faithfulness=normalised,
            wall_time_seconds=wall_time,
            num_forward_calls=self._forward_calls_per_eval,
            num_examples=n_examples,
            extra={
                "live_edges": sorted(live_indices),
                "ground_truth_edges": sorted(ground),
                "intersection_size": len(intersection),
                "union_size": len(union),
                "jaccard": jaccard,
                "mock": True,
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batch_size(batch: Any) -> int:
    """Infer the number of examples from *batch*.

    Returns 1 if the batch size cannot be determined.
    """
    if batch is None:
        return 1
    if isinstance(batch, dict):
        for v in batch.values():
            try:
                return len(v)
            except TypeError:
                continue
    try:
        return len(batch)
    except TypeError:
        return 1
