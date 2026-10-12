"""
hrl_circuits.benchmark.base
============================
``CircuitBenchmark`` — the protocol (structural interface) that every
benchmark adapter must satisfy.

Spec reference: section 9

Every method is declared here with full signatures, docstrings, and
``...`` bodies.  Concrete implementations (e.g. :mod:`.mib_adapter`,
:mod:`.interpbench_adapter`) inherit from this protocol or implement it
structurally (duck-typing via :class:`typing.Protocol`).

Why Protocol vs ABC?
--------------------
Using :class:`typing.Protocol` (with ``runtime_checkable=True``) allows
downstream code to check ``isinstance(obj, CircuitBenchmark)`` without
requiring adapters to explicitly inherit from this class.  This is useful
for third-party adapters that may not be able to modify their class
hierarchy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class CircuitBenchmark(Protocol):
    """Structural interface for HRL circuit localisation benchmarks.

    Implementing classes must provide all of the following methods.
    Type annotations use ``Any`` where the concrete type depends on the
    underlying model library (e.g. ``torch.nn.Module`` vs. ``jax`` module).
    """

    # ------------------------------------------------------------------
    # Model access
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_id: str,
        *,
        frozen: bool = True,
        device: str = "cpu",
    ) -> Any:
        """Load the model specified by *model_id*.

        Parameters
        ----------
        model_id:
            Hugging Face model ID or local path.
        frozen:
            If ``True`` (default), the model's weights must be frozen
            (requires_grad = False for PyTorch).  Implementations should
            call :func:`~hrl_circuits.models.frozen_guard.assert_no_trainable_model_weights`
            before returning.
        device:
            Target device string (``'cpu'``, ``'cuda'``, etc.).

        Returns
        -------
        Any
            The loaded model object.
        """
        ...

    # ------------------------------------------------------------------
    # Data splits
    # ------------------------------------------------------------------

    def load_split(
        self,
        split: str,
        task: str,
        *,
        max_examples: int | None = None,
    ) -> Any:
        """Load a data split for *task*.

        Parameters
        ----------
        split:
            One of ``'train'``, ``'validation'``, or ``'public_test'``.
        task:
            Task name (e.g. ``"ioi"``).
        max_examples:
            If provided, truncate the split to at most this many examples.

        Returns
        -------
        Any
            A dataset or dataloader object.
        """
        ...

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def build_candidate_graph(
        self,
        model: Any,
        task: str,
        benchmark_metadata: dict[str, Any],
    ):
        """Construct the candidate circuit graph for *model* and *task*.

        Parameters
        ----------
        model:
            The loaded model (from :meth:`load_model`).
        task:
            Task name.
        benchmark_metadata:
            Must include at least ``benchmark_revision``, ``model_id``,
            and ``task``.

        Returns
        -------
        CandidateGraph
            Fully built, hashed, and validated graph.
        """
        ...

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def get_clean_counterfactual_batch(
        self,
        dataset: Any,
        batch_size: int,
        example_indices: list[int] | None = None,
    ) -> Any:
        """Return a batch of (clean, counterfactual) example pairs.

        Parameters
        ----------
        dataset:
            Dataset returned by :meth:`load_split`.
        batch_size:
            Number of pairs to include.
        example_indices:
            If provided, return only the specified examples (by position).
            Otherwise, the first *batch_size* examples are returned.

        Returns
        -------
        Any
            A batch object containing paired examples suitable for
            activation patching.  The exact type is implementation-defined
            (typically a dict with ``'clean'`` and ``'counterfactual'``
            keys, each holding tokenised input tensors).
        """
        ...

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_mask(
        self,
        mask: np.ndarray,
        model: Any,
        batch: Any,
        *,
        metric: str = "kl",
    ) -> dict[str, Any]:
        """Evaluate the circuit defined by *mask* on *batch*.

        Parameters
        ----------
        mask:
            Binary edge mask (uint8, shape ``(num_edges,)``).
        model:
            The loaded model.
        batch:
            A batch from :meth:`get_clean_counterfactual_batch`.
        metric:
            Faithfulness metric: ``'kl'`` or ``'mse'``.

        Returns
        -------
        dict
            At minimum contains:

            * ``'raw_faithfulness'`` (float)
            * ``'normalised_faithfulness'`` (float)
            * ``'num_forward_calls'`` (int)
            * ``'num_examples'`` (int)
        """
        ...

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_circuit(
        self,
        mask: np.ndarray,
        graph: Any,
        output_dir: str | Path,
        run_id: str,
    ) -> Path:
        """Export the circuit defined by *mask* to *output_dir*.

        Parameters
        ----------
        mask:
            Binary edge mask defining the circuit.
        graph:
            The :class:`~hrl_circuits.graph.CandidateGraph` the mask
            refers to.
        output_dir:
            Directory in which to write the circuit manifest.
        run_id:
            Unique identifier for this run (used in file naming).

        Returns
        -------
        Path
            Path to the written manifest file.
        """
        ...

    def evaluate_exported_circuit(
        self,
        manifest_path: str | Path,
        model: Any,
        split: str,
        task: str,
    ) -> dict[str, Any]:
        """Evaluate a previously exported circuit manifest.

        Parameters
        ----------
        manifest_path:
            Path to the manifest file from :meth:`export_circuit`.
        model:
            The loaded model.
        split:
            Data split to evaluate on.
        task:
            Task name.

        Returns
        -------
        dict
            Faithfulness metrics (same format as :meth:`evaluate_mask`).
        """
        ...
