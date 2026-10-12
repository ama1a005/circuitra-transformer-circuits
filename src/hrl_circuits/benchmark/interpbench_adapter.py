"""
hrl_circuits.benchmark.interpbench_adapter
==========================================
Adapter stub for the **InterpBench** evaluation suite.

Status: **STUB — real implementation requires the ``interpbench`` package**

InterpBench provides mechanistic interpretability benchmarks with
standardised splits, task definitions, and faithfulness metrics.  This
stub follows the :class:`~hrl_circuits.benchmark.base.CircuitBenchmark`
protocol exactly and contains detailed ``TODO`` markers at every point
that needs the actual ``interpbench`` library.

Integration steps
-----------------
1. Install: ``pip install interpbench`` (or the relevant internal package).
2. Replace each ``TODO`` block with the corresponding InterpBench API.
3. Remove :exc:`InterpBenchNotInstalledError` guard from :meth:`load_model`
   once confirmed available.

References
----------
* InterpBench paper: https://arxiv.org/abs/2407.14494
* InterpBench GitHub: https://github.com/hannamw/eap-ig-faithfulness
  (TODO: update with official URL when released)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ..graph.schema import CandidateGraph
from ..graph.builder import CandidateGraphBuilder
from ..evaluator.faithfulness import (
    compute_faithfulness_from_outputs,
    normalize_faithfulness,
)
from .circuit_io import export_circuit_manifest
from .split_guard import SplitGuard


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class InterpBenchNotInstalledError(ImportError):
    """Raised when the ``interpbench`` package is not installed."""


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class InterpBenchAdapter:
    """Adapter wrapping InterpBench as a :class:`CircuitBenchmark`.

    Parameters
    ----------
    config:
        Configuration dict.  Required keys:

        * ``benchmark_revision`` (str)
        * ``model_id`` (str)
        * ``task`` (str)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        # TODO: validate required config keys
        self._config = config
        self._builder = CandidateGraphBuilder()

        # TODO: import interpbench here
        #   try:
        #       import interpbench as ib
        #       self._ib = ib
        #   except ImportError as exc:
        #       raise InterpBenchNotInstalledError(
        #           "The 'interpbench' package is required.  "
        #           "Install with: pip install interpbench"
        #       ) from exc

        warnings.warn(
            "InterpBenchAdapter is a stub. Real InterpBench integration "
            "is not yet implemented.",
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # CircuitBenchmark protocol
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_id: str,
        *,
        frozen: bool = True,
        device: str = "cpu",
    ) -> Any:
        """Load a model via InterpBench.

        TODO
        ----
        * Call ``self._ib.load_model(model_id, device=device)``
        * Optionally apply frozen-weight guard if *frozen* is True using
          :func:`~hrl_circuits.models.frozen_guard.assert_no_trainable_model_weights`.

        Returns
        -------
        Any
            Loaded model (stub raises).
        """
        # TODO: implement
        raise NotImplementedError(
            "InterpBenchAdapter.load_model is not yet implemented.  "
            "See module docstring for integration steps."
        )

    def load_split(
        self,
        split: str,
        task: str,
        *,
        max_examples: int | None = None,
    ) -> Any:
        """Load an InterpBench task split.

        TODO
        ----
        * Call ``self._ib.load_dataset(task, split=split)``
        * Apply max_examples truncation.
        * InterpBench tasks include: ``"ioi"``, ``"gt"``, ``"tracr"`` etc.

        Returns
        -------
        Any
            Dataset (stub raises).
        """
        SplitGuard.assert_can_train(split)
        # TODO: implement
        raise NotImplementedError(
            "InterpBenchAdapter.load_split is not yet implemented."
        )

    def build_candidate_graph(
        self,
        model: Any,
        task: str,
        benchmark_metadata: dict[str, Any],
    ) -> CandidateGraph:
        """Extract the candidate graph from the InterpBench model.

        TODO
        ----
        * Use InterpBench's ``get_circuit_graph(model, task)`` (or
          equivalent) to enumerate attention/MLP edges.
        * Convert each edge to the raw dict format expected by
          :class:`~hrl_circuits.graph.CandidateGraphBuilder`.
        * Return ``self._builder.build_from_edges(edges, benchmark_metadata)``.

        Returns
        -------
        CandidateGraph
            Stub raises.
        """
        # TODO: implement
        raise NotImplementedError(
            "InterpBenchAdapter.build_candidate_graph is not yet implemented."
        )

    def get_clean_counterfactual_batch(
        self,
        dataset: Any,
        batch_size: int,
        example_indices: list[int] | None = None,
    ) -> Any:
        """Collate a (clean, counterfactual) batch.

        TODO
        ----
        * Use InterpBench's collation utilities to produce paired inputs.
        * InterpBench typically uses ``(clean_tokens, patch_tokens)`` pairs.

        Returns
        -------
        Any
            Batch (stub raises).
        """
        # TODO: implement
        raise NotImplementedError(
            "InterpBenchAdapter.get_clean_counterfactual_batch is not yet "
            "implemented."
        )

    def evaluate_mask(
        self,
        mask: np.ndarray,
        model: Any,
        batch: Any,
        *,
        metric: str = "kl",
    ) -> dict[str, Any]:
        """Run activation patching for edges defined by *mask*.

        TODO
        ----
        * Identify live edges from the mask.
        * Use InterpBench's patching hooks to intervene on those edges.
        * Run clean and patched forward passes.
        * Compute faithfulness via
          :func:`~hrl_circuits.evaluator.faithfulness.compute_faithfulness_from_outputs`.

        Returns
        -------
        dict
            ``{raw_faithfulness, normalised_faithfulness,
            num_forward_calls, num_examples}`` (stub raises).
        """
        # TODO: implement
        raise NotImplementedError(
            "InterpBenchAdapter.evaluate_mask is not yet implemented."
        )

    def export_circuit(
        self,
        mask: np.ndarray,
        graph: CandidateGraph,
        output_dir: str | Path,
        run_id: str,
    ) -> Path:
        """Export the circuit manifest (fully implemented, no InterpBench needed).

        Returns
        -------
        Path
            Path to the written manifest file.
        """
        return export_circuit_manifest(
            mask=mask,
            graph=graph,
            output_dir=output_dir,
            run_id=run_id,
            extra_metadata={
                "adapter": "interpbench",
                "config": self._config,
            },
        )

    def evaluate_exported_circuit(
        self,
        manifest_path: str | Path,
        model: Any,
        split: str,
        task: str,
    ) -> dict[str, Any]:
        """Load a manifest and evaluate the circuit.

        TODO
        ----
        * Load the manifest JSON.
        * Reconstruct the mask as a numpy array.
        * Load the appropriate split and call :meth:`evaluate_mask`.

        Returns
        -------
        dict
            Faithfulness metrics (stub raises).
        """
        # TODO: implement
        raise NotImplementedError(
            "InterpBenchAdapter.evaluate_exported_circuit is not yet "
            "implemented."
        )
