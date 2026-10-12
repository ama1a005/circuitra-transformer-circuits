"""
RandomMatchedBaseline
=====================
Samples random edge-subsets of size k (matched to the circuit budget) and
evaluates them through the same evaluator used by flat/HRL agents.  This gives
a fair lower-bound that is identically constrained in terms of sparsity.

Public API
----------
sample_random_circuits(graph, num_edges_k, n_samples, seed) -> list[np.ndarray]
evaluate_random_circuits(circuits, evaluator, batch, counter) -> list[EvalResult]
summarize_random_results(results) -> dict
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight result type (mirrors the EvalResult used by the rest of the
# codebase; import the real one if it is available, otherwise define locally).
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Faithfulness / quality score for a single circuit evaluation."""

    circuit_mask: np.ndarray
    """Boolean / float mask over graph edges (shape: [n_edges])."""

    faithfulness: float
    """Primary quality metric returned by the evaluator."""

    forward_calls: int = 0
    """Number of model forward passes consumed by this evaluation."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Any additional metric returned by the evaluator (task-specific)."""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def sample_random_circuits(
    graph: Any,
    num_edges_k: int,
    n_samples: int,
    seed: int = 0,
) -> list[np.ndarray]:
    """Sample *n_samples* random binary edge masks, each with exactly *num_edges_k* ones.

    Parameters
    ----------
    graph:
        Graph object that exposes ``.n_edges`` (int) or ``len(graph.edges)``.
        Any graph type used elsewhere in the project is accepted.
    num_edges_k:
        Number of edges to include in each random circuit (== budget k).
    n_samples:
        How many independent random circuits to draw.
    seed:
        Base random seed; each sample uses ``seed + sample_index`` internally
        to ensure reproducibility without correlation between samples.

    Returns
    -------
    list of np.ndarray
        Each array has shape ``(n_edges,)`` and dtype ``bool``.  Exactly
        ``num_edges_k`` entries are True.

    Raises
    ------
    ValueError
        If ``num_edges_k`` exceeds the total number of edges in *graph*.
    """
    # Resolve total edge count from graph object
    n_edges: int
    if hasattr(graph, "n_edges"):
        n_edges = int(graph.n_edges)
    elif hasattr(graph, "edges"):
        n_edges = len(graph.edges)
    else:
        raise TypeError(
            f"Cannot determine edge count from graph of type {type(graph).__name__}. "
            "Expected attribute 'n_edges' (int) or 'edges' (sequence)."
        )

    if num_edges_k > n_edges:
        raise ValueError(
            f"num_edges_k={num_edges_k} exceeds total graph edges={n_edges}."
        )
    if num_edges_k < 0:
        raise ValueError(f"num_edges_k must be non-negative, got {num_edges_k}.")

    circuits: list[np.ndarray] = []
    for i in range(n_samples):
        rng = np.random.default_rng(seed + i)
        mask = np.zeros(n_edges, dtype=bool)
        chosen = rng.choice(n_edges, size=num_edges_k, replace=False)
        mask[chosen] = True
        circuits.append(mask)

    logger.debug(
        "Sampled %d random circuits (k=%d, n_edges=%d, base_seed=%d)",
        n_samples,
        num_edges_k,
        n_edges,
        seed,
    )
    return circuits


def evaluate_random_circuits(
    circuits: list[np.ndarray],
    evaluator: Callable[[np.ndarray], tuple[float, int]],
    batch: Any,
    counter: Any | None = None,
) -> list[EvalResult]:
    """Evaluate each random circuit mask with *evaluator*.

    Parameters
    ----------
    circuits:
        List of boolean edge masks (output of :func:`sample_random_circuits`).
    evaluator:
        Callable that accepts ``(mask: np.ndarray, batch: Any)`` and returns
        ``(faithfulness: float, forward_calls: int)``.  The signature matches
        the evaluator interface used in the rest of the project.
    batch:
        Data batch passed through to *evaluator* unchanged.
    counter:
        Optional mutable counter object with a ``.increment(n)`` method used to
        track global forward-call budget.  Pass ``None`` to skip tracking.

    Returns
    -------
    list[EvalResult]
        One result per circuit mask, in the same order as *circuits*.
    """
    results: list[EvalResult] = []
    for idx, mask in enumerate(circuits):
        try:
            faithfulness, fwd_calls = evaluator(mask, batch)
        except Exception as exc:
            logger.warning(
                "Evaluator raised exception for random circuit %d: %s", idx, exc
            )
            faithfulness, fwd_calls = float("nan"), 0

        if counter is not None:
            try:
                counter.increment(fwd_calls)
            except Exception:
                pass  # counter interface mismatch — silently ignore

        results.append(
            EvalResult(
                circuit_mask=mask,
                faithfulness=faithfulness,
                forward_calls=fwd_calls,
            )
        )

    logger.debug("Evaluated %d random circuits.", len(results))
    return results


def summarize_random_results(results: list[EvalResult]) -> dict[str, float]:
    """Compute aggregate statistics over a list of random-circuit evaluations.

    Parameters
    ----------
    results:
        Output of :func:`evaluate_random_circuits`.

    Returns
    -------
    dict with keys:
        - ``mean_faithfulness``
        - ``std_faithfulness``
        - ``min_faithfulness``
        - ``max_faithfulness``
        - ``n_valid``   – number of non-NaN evaluations
        - ``n_total``   – total evaluations attempted
    """
    if not results:
        return {
            "mean_faithfulness": float("nan"),
            "std_faithfulness": float("nan"),
            "min_faithfulness": float("nan"),
            "max_faithfulness": float("nan"),
            "n_valid": 0,
            "n_total": 0,
        }

    faithfulness_values = np.array([r.faithfulness for r in results], dtype=float)
    valid_mask = np.isfinite(faithfulness_values)
    valid_values = faithfulness_values[valid_mask]

    summary: dict[str, float] = {
        "mean_faithfulness": float(np.mean(valid_values)) if valid_values.size else float("nan"),
        "std_faithfulness": float(np.std(valid_values, ddof=1)) if valid_values.size > 1 else float("nan"),
        "min_faithfulness": float(np.min(valid_values)) if valid_values.size else float("nan"),
        "max_faithfulness": float(np.max(valid_values)) if valid_values.size else float("nan"),
        "n_valid": int(valid_values.size),
        "n_total": len(results),
    }

    logger.info(
        "Random baseline summary: mean=%.4f, std=%.4f, n_valid=%d/%d",
        summary["mean_faithfulness"],
        summary["std_faithfulness"],
        summary["n_valid"],
        summary["n_total"],
    )
    return summary


# ---------------------------------------------------------------------------
# Class wrapper (thin, for registry / runner compatibility)
# ---------------------------------------------------------------------------

class RandomMatchedBaseline:
    """Facade class wrapping the module-level random baseline functions.

    Parameters
    ----------
    n_samples:
        Number of random circuits to sample per evaluation call.
    seed:
        Base random seed.
    """

    def __init__(self, n_samples: int = 100, seed: int = 0) -> None:
        self.n_samples = n_samples
        self.seed = seed

    # ------------------------------------------------------------------
    def sample(self, graph: Any, num_edges_k: int) -> list[np.ndarray]:
        """Delegate to :func:`sample_random_circuits`."""
        return sample_random_circuits(graph, num_edges_k, self.n_samples, self.seed)

    def evaluate(
        self,
        circuits: list[np.ndarray],
        evaluator: Callable,
        batch: Any,
        counter: Any | None = None,
    ) -> list[EvalResult]:
        """Delegate to :func:`evaluate_random_circuits`."""
        return evaluate_random_circuits(circuits, evaluator, batch, counter)

    def summarize(self, results: list[EvalResult]) -> dict[str, float]:
        """Delegate to :func:`summarize_random_results`."""
        return summarize_random_results(results)

    def run(
        self,
        graph: Any,
        num_edges_k: int,
        evaluator: Callable,
        batch: Any,
        counter: Any | None = None,
    ) -> dict[str, float]:
        """End-to-end: sample → evaluate → summarize.

        Returns the summary dict from :func:`summarize_random_results`.
        """
        circuits = self.sample(graph, num_edges_k)
        results = self.evaluate(circuits, evaluator, batch, counter)
        return self.summarize(results)
