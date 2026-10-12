"""
hrl_circuits.benchmark.mib_adapter
=====================================
Adapter stub for the **MIB** (Modular Interpretability Benchmark) library.

Status: **STUB — real implementation requires the ``mib`` package**

This stub provides the correct interface (matching
:class:`~hrl_circuits.benchmark.base.CircuitBenchmark`) with clearly
marked ``TODO`` comments at every integration point.  It can be imported
and type-checked without ``mib`` installed.

Integration steps
-----------------
1. Install the MIB library: ``pip install mib`` (or via the project's
   dependency group).
2. Replace each ``TODO`` block with the corresponding MIB API call.
3. Remove the :exc:`MIBNotInstalledError` guard from :meth:`load_model`
   once the dependency is confirmed available.

References
----------
* MIB paper: https://arxiv.org/abs/...  (TODO: add DOI once published)
* MIB GitHub: https://github.com/...   (TODO: add URL)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ..graph.schema import CandidateGraph
from ..graph.builder import CandidateGraphBuilder
from ..graph.masks import get_live_edge_indices
from ..evaluator.faithfulness import (
    compute_faithfulness_from_outputs,
    normalize_faithfulness,
)
from .circuit_io import export_circuit_manifest, load_mask, CircuitManifest
from .split_guard import SplitGuard


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MIBNotInstalledError(ImportError):
    """Raised when the ``mib`` package is not installed."""


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class MIBAdapter:
    """Adapter that wraps the MIB library as a :class:`CircuitBenchmark`.

    Parameters
    ----------
    config:
        Configuration dict or object.  Must include at least:

        * ``benchmark_revision`` (str)
        * ``model_id`` (str)
        * ``task`` (str)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        # TODO: validate that all required config keys are present
        self._config = config
        self._builder = CandidateGraphBuilder()

        # TODO: attempt to import mib here and store the module reference
        #   try:
        #       import mib
        #       self._mib = mib
        #   except ImportError as exc:
        #       raise MIBNotInstalledError(
        #           "The 'mib' package is required for MIBAdapter.  "
        #           "Install it with: pip install mib"
        #       ) from exc

        warnings.warn(
            "MIBAdapter is a stub. Real MIB integration is not yet implemented.",
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
        """Load a model via MIB.

        TODO
        ----
        * Call ``self._mib.load_model(model_id, device=device)``
        * Apply frozen-weight guard if *frozen* is True.

        Returns
        -------
        Any
            The loaded model (stub returns ``None``).
        """
        # TODO: implement
        raise NotImplementedError(
            "MIBAdapter.load_model is not yet implemented.  "
            "See module docstring for integration steps."
        )

    def load_split(
        self,
        split: str,
        task: str,
        *,
        max_examples: int | None = None,
    ) -> Any:
        """Load a MIB task split.

        TODO
        ----
        * Call ``self._mib.load_task(task, split=split)``
        * Apply ``max_examples`` truncation if provided.

        Returns
        -------
        Any
            Dataset object (stub raises).
        """
        SplitGuard.assert_can_train(split)  # minimal safety check
        # TODO: implement
        raise NotImplementedError(
            "MIBAdapter.load_split is not yet implemented."
        )

    def build_candidate_graph(
        self,
        model: Any,
        task: str,
        benchmark_metadata: dict[str, Any],
    ) -> CandidateGraph:
        """Extract the candidate graph from the MIB model.

        TODO
        ----
        * Use MIB's graph-extraction API to enumerate edges.
        * Convert each edge to the ``{src_layer, dst_layer, ...}`` dict
          format expected by :class:`~hrl_circuits.graph.CandidateGraphBuilder`.
        * Call ``self._builder.build_from_edges(edges_list, benchmark_metadata)``.

        Returns
        -------
        CandidateGraph
            Stub raises.
        """
        # TODO: implement
        raise NotImplementedError(
            "MIBAdapter.build_candidate_graph is not yet implemented."
        )

    def get_clean_counterfactual_batch(
        self,
        dataset: Any,
        batch_size: int,
        example_indices: list[int] | None = None,
    ) -> Any:
        """Collate a (clean, counterfactual) batch from *dataset*.

        TODO
        ----
        * Use MIB's collation utilities to build paired input tensors.

        Returns
        -------
        Any
            Batch dict (stub raises).
        """
        # TODO: implement
        raise NotImplementedError(
            "MIBAdapter.get_clean_counterfactual_batch is not yet implemented."
        )

    def evaluate_mask(
        self,
        mask: np.ndarray,
        model: Any,
        batch: Any,
        *,
        metric: str = "kl",
    ) -> dict[str, Any]:
        """Run activation patching for the edges in *mask*.

        TODO
        ----
        * For each live edge (``mask[i] == 1``), patch the corresponding
          activation from the counterfactual forward pass.
        * Run the patched forward pass to get *patched_out*.
        * Run the clean forward pass to get *clean_out*.
        * Call :func:`~hrl_circuits.evaluator.faithfulness.compute_faithfulness_from_outputs`.

        Returns
        -------
        dict
            Keys: ``raw_faithfulness``, ``normalised_faithfulness``,
            ``num_forward_calls``, ``num_examples`` (stub raises).
        """
        # TODO: implement
        raise NotImplementedError(
            "MIBAdapter.evaluate_mask is not yet implemented."
        )

    def export_circuit(
        self,
        mask: np.ndarray,
        graph: CandidateGraph,
        output_dir: str | Path,
        run_id: str,
    ) -> Path:
        """Export the circuit manifest.

        Delegates to :func:`~hrl_circuits.benchmark.circuit_io.export_circuit_manifest`.

        Returns
        -------
        Path
            Path to the written manifest file.
        """
        # This can be fully implemented without MIB
        return export_circuit_manifest(
            mask=mask,
            graph=graph,
            output_dir=output_dir,
            run_id=run_id,
            extra_metadata={"adapter": "mib", "config": self._config},
        )

    def evaluate_exported_circuit(
        self,
        manifest_path: str | Path,
        model: Any,
        split: str,
        task: str,
    ) -> dict[str, Any]:
        """Load and evaluate a previously exported circuit.

        TODO
        ----
        * Load the manifest and reconstruct the mask.
        * Call :meth:`evaluate_mask` with the reconstructed mask.

        Returns
        -------
        dict
            Faithfulness metrics (stub raises).
        """
        # TODO: implement
        raise NotImplementedError(
            "MIBAdapter.evaluate_exported_circuit is not yet implemented."
        )
