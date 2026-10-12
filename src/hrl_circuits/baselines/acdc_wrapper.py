"""
ACDCWrapper — Interface wrapper for the ACDC (Automatic Circuit DisCovery) baseline.

ACDC identifies a minimal circuit by iteratively pruning edges whose removal does
not degrade performance beyond a threshold.  This module provides a stable
interface so that the rest of the codebase can call ACDC as if it were a
first-class circuit-finding method, without depending directly on the acdc
library internals.

Reference: Conmy et al. (2023) — "Towards Automated Circuit Discovery for
Mechanistic Interpretability"
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
class ACDCConfig:
    """Hyper-parameters forwarded to the ACDC algorithm."""

    threshold: float = 0.1
    """KL-divergence threshold below which an edge is considered redundant."""

    metric: str = "kl_div"
    """Metric used to evaluate edge importance ('kl_div', 'logit_diff', ...)."""

    max_num_epochs: int = 50
    """Maximum number of ACDC pruning epochs."""

    abs_value_threshold: bool = True
    """Whether to threshold on absolute metric change rather than signed."""

    device: str = "cpu"
    """Torch device string."""


@dataclass
class ACDCLog:
    """Structured log object returned by ACDC runs."""

    n_edges_start: int = 0
    n_edges_end: int = 0
    n_evaluations: int = 0
    epochs_run: int = 0
    raw_log: list[dict[str, Any]] = field(default_factory=list)
    """Per-epoch records emitted by the ACDC library (if available)."""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class ACDCWrapper:
    """
    Thin wrapper around the ACDC library that conforms to the project's
    baseline interface.

    Usage::

        wrapper = ACDCWrapper(config=ACDCConfig(threshold=0.1))
        mask = wrapper.run_acdc(model, benchmark, config)
        evals = wrapper.count_acdc_evaluations(log)

    .. note::
        The actual ACDC library calls are marked with ``TODO`` comments.
        Install and import ``acdc`` (the Conmy et al. package) and fill in
        the marked sections before running real experiments.
    """

    LABEL: str = "ACDC-inspired"
    """
    Label applied to results produced by this wrapper.  It reads
    'ACDC-inspired' rather than 'ACDC' because the wrapper may not be a
    pixel-perfect reimplementation — callers should check whether the
    underlying library was successfully imported.
    """

    def __init__(self, config: ACDCConfig | None = None) -> None:
        self.config = config or ACDCConfig()
        self._library_available: bool = self._try_import_acdc()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_import_acdc() -> bool:
        """Return True if the acdc library can be imported."""
        try:
            import acdc  # noqa: F401  # TODO: verify package name on install
            return True
        except ImportError:
            logger.warning(
                "acdc library not found — ACDCWrapper will run in mock mode. "
                "Install the Conmy et al. ACDC package to enable real runs."
            )
            return False

    # ------------------------------------------------------------------
    # Primary method
    # ------------------------------------------------------------------

    def run_acdc(
        self,
        model: Any,
        benchmark: Any,
        config: ACDCConfig | None = None,
    ) -> tuple[np.ndarray, ACDCLog]:
        """
        Run ACDC circuit discovery and return a binary edge mask.

        Parameters
        ----------
        model:
            A frozen PyTorch ``nn.Module`` (loaded via ``ModelLoader``).
        benchmark:
            A ``CircuitBenchmark`` adapter (``MIBAdapter`` or
            ``InterpBenchAdapter``).
        config:
            Optional override config; falls back to ``self.config``.

        Returns
        -------
        mask : np.ndarray, shape (n_edges,), dtype bool
            True for edges that ACDC retains in the circuit.
        log : ACDCLog
            Structured log with evaluation counts and per-epoch records.

        Notes
        -----
        Label the result as ``ACDCWrapper.LABEL`` ('ACDC-inspired') in any
        downstream reporting unless you have verified that the underlying
        library call is an exact reimplementation.
        """
        cfg = config or self.config

        if not self._library_available:
            logger.warning("Running ACDC in mock mode — returning all-ones mask.")
            return self._mock_run(model, benchmark, cfg)

        return self._real_run(model, benchmark, cfg)

    # ------------------------------------------------------------------
    # Real implementation (library-backed)
    # ------------------------------------------------------------------

    def _real_run(
        self,
        model: Any,
        benchmark: Any,
        config: ACDCConfig,
    ) -> tuple[np.ndarray, ACDCLog]:
        """
        Delegate to the acdc library.

        TODO (integration):
            1.  ``import acdc`` and its ``ACDCExperiment`` or equivalent class.
            2.  Build the ``acdc``-compatible model wrapper from ``model`` and
                ``benchmark.get_clean_counterfactual_batch()``.
            3.  Instantiate the ACDC runner with ``config.threshold``,
                ``config.metric``, ``config.max_num_epochs``.
            4.  Call ``runner.run()`` and collect its edge importance scores /
                retained-edge set.
            5.  Map the ACDC edge ids back to the project's edge id space using
                ``benchmark.build_candidate_graph()``.
            6.  Build a boolean ``np.ndarray`` mask over
                ``graph.all_edge_ids``.
            7.  Populate and return an ``ACDCLog`` from the runner's logs.
        """
        raise NotImplementedError(
            "TODO: integrate the acdc library. "
            "See _real_run docstring for step-by-step instructions."
        )

    # ------------------------------------------------------------------
    # Mock / fallback implementation
    # ------------------------------------------------------------------

    def _mock_run(
        self,
        model: Any,
        benchmark: Any,
        config: ACDCConfig,
    ) -> tuple[np.ndarray, ACDCLog]:
        """
        Return a trivial all-ones mask for smoke-testing purposes.

        The graph is queried via ``benchmark.build_candidate_graph()`` so that
        the mask shape is always correct even in mock mode.
        """
        try:
            graph = benchmark.build_candidate_graph()
            n_edges = len(graph.all_edge_ids)
        except Exception:  # pragma: no cover
            logger.warning("Could not query benchmark graph — defaulting to 100 edges.")
            n_edges = 100

        mask = np.ones(n_edges, dtype=bool)
        log = ACDCLog(
            n_edges_start=n_edges,
            n_edges_end=n_edges,
            n_evaluations=1,
            epochs_run=0,
        )
        return mask, log


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def run_acdc(
    model: Any,
    benchmark: Any,
    config: ACDCConfig | None = None,
) -> tuple[np.ndarray, ACDCLog]:
    """
    Module-level convenience function for running ACDC.

    Parameters
    ----------
    model:
        Frozen ``nn.Module``.
    benchmark:
        ``CircuitBenchmark`` adapter.
    config:
        ``ACDCConfig`` instance (optional).

    Returns
    -------
    mask : np.ndarray — boolean edge mask.
    log : ACDCLog — structured run log.
    """
    wrapper = ACDCWrapper(config=config)
    return wrapper.run_acdc(model, benchmark, config)


def count_acdc_evaluations(log: ACDCLog) -> int:
    """
    Return the total number of model forward-pass evaluations recorded in
    ``log``.

    Parameters
    ----------
    log : ACDCLog
        Log object returned by :func:`run_acdc` or ``ACDCWrapper.run_acdc``.

    Returns
    -------
    int — total evaluation count.
    """
    if log.n_evaluations > 0:
        return log.n_evaluations
    # Fall back to summing per-epoch records if available.
    return sum(r.get("n_evaluations", 0) for r in log.raw_log)
