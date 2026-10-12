"""
hrl_circuits.graph.builder
==========================
``CandidateGraphBuilder`` — the single authoritative place where raw edge
data is turned into a validated, hashed :class:`CandidateGraph`.

Design rules enforced here
--------------------------
* **Canonical sort**: edges are ordered by
  ``(src_layer, dst_layer, src_node_id, dst_node_id, edge_type)`` before
  IDs are assigned, so that ``edge_id == position-in-canonical-order``.
* **Stable monotonic IDs**: ``edge_id`` is the 0-based position of the edge
  in the canonical ordering.  Re-building from the same logical edges always
  produces the same IDs.
* **No duplicate IDs / keys**: enforced via
  :func:`~hrl_circuits.graph.schema.validate_edge_records`.
* **Native mapping existence**: every ``native_src_id`` / ``native_dst_id``
  must be non-empty (validated in schema).
* **Graph hash**: computed with
  :func:`~hrl_circuits.graph.hash.compute_graph_hash` *after* sorting.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .hash import compute_graph_hash
from .schema import (
    CandidateGraph,
    EdgeRecord,
    validate_benchmark_metadata,
    validate_edge_records,
)


class CandidateGraphBuilder:
    """Build and validate :class:`CandidateGraph` instances from raw edge data.

    Usage
    -----
    .. code-block:: python

        builder = CandidateGraphBuilder()
        graph = builder.build_from_edges(edges_list, benchmark_metadata)

    The builder is stateless — calling :meth:`build_from_edges` multiple
    times on the same instance is safe.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_from_edges(
        self,
        edges_list: list[dict[str, Any]],
        benchmark_metadata: dict[str, Any],
    ) -> CandidateGraph:
        """Construct a validated :class:`CandidateGraph` from raw edge dicts.

        Parameters
        ----------
        edges_list:
            A list of dicts, each describing one edge.  Required keys:

            * ``src_layer`` (int)
            * ``dst_layer`` (int)
            * ``src_node_id`` (str)
            * ``dst_node_id`` (str)
            * ``edge_type`` (str)
            * ``native_src_id`` (str)
            * ``native_dst_id`` (str)

            Optional keys:

            * ``metadata`` (dict) — passed through unchanged
            * ``edge_id`` — *ignored*; IDs are always reassigned

        benchmark_metadata:
            Must contain at least ``benchmark_revision``, ``model_id``,
            and ``task``.

        Returns
        -------
        CandidateGraph
            Fully validated graph with stable edge IDs and a graph hash.

        Raises
        ------
        ValueError
            If required fields are missing, native mappings are empty, or
            duplicate canonical sort keys are found.
        TypeError
            If *edges_list* is not a list.
        """
        if not isinstance(edges_list, list):
            raise TypeError(
                f"edges_list must be a list, got {type(edges_list).__name__!r}"
            )

        # Validate benchmark metadata first so failures are reported early
        validate_benchmark_metadata(benchmark_metadata)

        # Parse raw dicts into preliminary EdgeRecord objects (edge_id=0 placeholder)
        raw_edges: list[EdgeRecord] = [
            self._parse_raw_edge(idx, raw) for idx, raw in enumerate(edges_list)
        ]

        # Canonical sort
        sorted_edges = sorted(raw_edges, key=lambda e: e.canonical_sort_key)

        # Assign stable monotonic IDs (position in canonical order)
        final_edges: list[EdgeRecord] = [
            replace(edge, edge_id=idx) for idx, edge in enumerate(sorted_edges)
        ]

        # Structural validation (duplicate IDs, duplicate keys, native ID checks)
        validate_edge_records(final_edges)

        # Compute the graph hash
        graph_hash = compute_graph_hash(final_edges)

        return CandidateGraph(
            edges=tuple(final_edges),
            graph_hash=graph_hash,
            benchmark_metadata=dict(benchmark_metadata),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_raw_edge(index: int, raw: dict[str, Any]) -> EdgeRecord:
        """Parse one raw edge dict into an :class:`EdgeRecord`.

        ``edge_id`` is set to 0 as a placeholder; it will be overwritten
        after canonical sorting.

        Raises
        ------
        ValueError
            If required keys are missing or have wrong types.
        """
        required_str_keys = (
            "src_node_id",
            "dst_node_id",
            "edge_type",
            "native_src_id",
            "native_dst_id",
        )
        required_int_keys = ("src_layer", "dst_layer")

        for key in required_str_keys:
            if key not in raw:
                raise ValueError(
                    f"Edge at index {index} is missing required key {key!r}."
                )
            if not isinstance(raw[key], str):
                raise ValueError(
                    f"Edge at index {index}: {key!r} must be str, "
                    f"got {type(raw[key]).__name__!r}."
                )

        for key in required_int_keys:
            if key not in raw:
                raise ValueError(
                    f"Edge at index {index} is missing required key {key!r}."
                )
            if not isinstance(raw[key], int):
                raise ValueError(
                    f"Edge at index {index}: {key!r} must be int, "
                    f"got {type(raw[key]).__name__!r}."
                )

        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Edge at index {index}: 'metadata' must be a dict, "
                f"got {type(metadata).__name__!r}."
            )

        return EdgeRecord(
            edge_id=0,  # placeholder; reassigned after sort
            src_layer=raw["src_layer"],
            dst_layer=raw["dst_layer"],
            src_node_id=raw["src_node_id"],
            dst_node_id=raw["dst_node_id"],
            edge_type=raw["edge_type"],
            native_src_id=raw["native_src_id"],
            native_dst_id=raw["native_dst_id"],
            metadata=dict(metadata),
        )
