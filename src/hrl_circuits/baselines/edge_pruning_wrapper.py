"""
EdgePruningWrapper — Interface wrapper for the Edge Pruning baseline.

Edge Pruning finds circuits by greedily removing edges whose removal has the
least effect on a behavioural metric, subject to a sparsity budget.

Reference: Bhaskar et al. (2024) — "Finding Transformer Circuits with Edge
Pruning"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EdgePruningConfig:
    """Configuration for the Edge Pruning baseline."""

    sparsity: float = 0.5
    """Target fraction of edges to *remove* (0 = keep all, 1 = remove all)."""

    lr: float = 0.01
    """Learning rate for the continuous relaxation optimiser."""

    n_steps: int = 1000
    """Number of optimisation steps."""

    regularisation: float = 1e-3
    """L1 regularisation coefficient on edge mask logits."""

    threshold: float = 0.5
    """Sigmoid threshold to binarise the continuous mask after optimisation."""

    metric: str = "logit_diff"
    """Behavioural metric used for gradient-based mask learning."""

    device: str = "cpu"
    """Torch device string."""


@dataclass
class EdgePruningLog:
    """Structured log object returned by Edge Pruning runs."""

    n_edges_start: int = 0
    n_edges_end: int = 0
    n_evaluations: int = 0
    steps_run: int = 0
    loss_curve: list[float] = field(default_factory=list)
    """Training loss at each optimisation step."""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class EdgePruningWrapper:
    """
    Thin wrapper around the Edge Pruning library that conforms to the project's
    baseline interface.

    Usage::

        wrapper = EdgePruningWrapper(config=EdgePruningConfig(sparsity=0.5))
        mask, log = wrapper.run_edge_pruning(model, benchmark)
        evals = wrapper.count_evaluations(log)

    .. note::
        Sections marked with ``TODO`` require the Edge Pruning package to be
        installed.  See ``_real_run`` for a step-by-step integration guide.
    """

    LABEL: str = "EdgePruning"

    def __init__(self, config: EdgePruningConfig | None = None) -> None:
        self.config = config or EdgePruningConfig()
        self._library_available: bool = self._try_import_edge_pruning()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_import_edge_pruning() -> bool:
        """Return True if the edge_pruning library can be imported."""
        try:
            import edge_pruning  # noqa: F401  # TODO: verify package name
            return True
        except ImportError:
            logger.warning(
                "edge_pruning library not found — EdgePruningWrapper will run "
                "in mock mode.  Install the Bhaskar et al. package to enable "
                "real runs."
            )
            return False

    # ------------------------------------------------------------------
    # Primary method
    # ------------------------------------------------------------------

    def run_edge_pruning(
        self,
        model: Any,
        benchmark: Any,
        config: EdgePruningConfig | None = None,
    ) -> tuple[np.ndarray, EdgePruningLog]:
        """
        Run Edge Pruning and return a binary edge mask.

        Parameters
        ----------
        model:
            Frozen ``nn.Module``.
        benchmark:
            ``CircuitBenchmark`` adapter.
        config:
            Optional override config; falls back to ``self.config``.

        Returns
        -------
        mask : np.ndarray, shape (n_edges,), dtype bool
            True for edges retained in the discovered circuit.
        log : EdgePruningLog
            Structured log with evaluation counts and loss curve.
        """
        cfg = config or self.config

        if not self._library_available:
            logger.warning("Running EdgePruning in mock mode — returning random mask.")
            return self._mock_run(model, benchmark, cfg)

        return self._real_run(model, benchmark, cfg)

    # ------------------------------------------------------------------
    # Real implementation (library-backed)
    # ------------------------------------------------------------------

    def _real_run(
        self,
        model: Any,
        benchmark: Any,
        config: EdgePruningConfig,
    ) -> tuple[np.ndarray, EdgePruningLog]:
        """
        Delegate to the edge_pruning library.

        TODO (integration):
            1.  ``import edge_pruning`` and its ``MaskedTransformer`` /
                ``EdgePruner`` class (verify API with the installed version).
            2.  Wrap ``model`` in the ``MaskedTransformer`` shim so that edge
                activations can be masked with continuous logits.
            3.  Build a dataset / dataloader from
                ``benchmark.get_clean_counterfactual_batch()``.
            4.  Instantiate the optimiser with ``config.lr`` and
                ``config.regularisation``.
            5.  Run the training loop for ``config.n_steps`` steps,
                recording loss at each step in ``log.loss_curve``.
            6.  Binarise the learned logits using ``config.threshold`` and
                ``torch.sigmoid``.
            7.  Map the pruner's internal edge ids back to the project's edge
                id space via ``benchmark.build_candidate_graph()``.
            8.  Return the boolean ``np.ndarray`` mask and populated log.
        """
        raise NotImplementedError(
            "TODO: integrate the edge_pruning library. "
            "See _real_run docstring for step-by-step instructions."
        )

    # ------------------------------------------------------------------
    # Mock / fallback implementation
    # ------------------------------------------------------------------

    def _mock_run(
        self,
        model: Any,
        benchmark: Any,
        config: EdgePruningConfig,
    ) -> tuple[np.ndarray, EdgePruningLog]:
        """
        Return a random mask sampled according to ``config.sparsity`` for
        smoke-testing purposes.
        """
        try:
            graph = benchmark.build_candidate_graph()
            n_edges = len(graph.all_edge_ids)
        except Exception:
            logger.warning("Could not query benchmark graph — defaulting to 100 edges.")
            n_edges = 100

        rng = np.random.default_rng(seed=0)
        mask = rng.random(n_edges) > config.sparsity
        log = EdgePruningLog(
            n_edges_start=n_edges,
            n_edges_end=int(mask.sum()),
            n_evaluations=config.n_steps,
            steps_run=config.n_steps,
        )
        return mask, log

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def count_evaluations(self, log: EdgePruningLog) -> int:
        """Return total model evaluations from ``log``."""
        return log.n_evaluations


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def run_edge_pruning(
    model: Any,
    benchmark: Any,
    config: EdgePruningConfig | None = None,
) -> tuple[np.ndarray, EdgePruningLog]:
    """
    Module-level convenience wrapper around :class:`EdgePruningWrapper`.
    """
    wrapper = EdgePruningWrapper(config=config)
    return wrapper.run_edge_pruning(model, benchmark, config)


def count_edge_pruning_evaluations(log: EdgePruningLog) -> int:
    """Return total model evaluations from ``log``."""
    if log.n_evaluations > 0:
        return log.n_evaluations
    return log.steps_run
