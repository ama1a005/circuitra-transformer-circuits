"""
hrl_circuits.graph.schema
=========================
Dataclasses and type definitions that form the shared schema for all graph
operations in the HRL Circuit Localization pipeline.

All dataclasses are frozen (immutable) to enable safe hashing and to
prevent accidental mutation after construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Core edge record
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class EdgeRecord:
    """A single directed edge in the candidate circuit graph.

    Attributes
    ----------
    edge_id:
        Stable, monotonic integer assigned at graph-construction time.
        Determines the position of this edge in mask arrays.
    src_layer:
        Integer index of the source transformer layer (or -1 for the
        embedding layer).
    dst_layer:
        Integer index of the destination transformer layer.
    src_node_id:
        String identifier for the source node within *src_layer*
        (e.g. ``"h3.q"`` for query of head 3).
    dst_node_id:
        String identifier for the destination node within *dst_layer*.
    edge_type:
        Semantic type of the connection, e.g. ``"attention"`` or ``"mlp"``.
    native_src_id:
        Opaque string mapping back to the model-native source component
        (required for running actual causal interventions).
    native_dst_id:
        Opaque string mapping back to the model-native destination component.
    metadata:
        Arbitrary extra key/value data preserved transparently.  Not included
        in hash computation or canonical sort key.
    """

    edge_id: int
    src_layer: int
    dst_layer: int
    src_node_id: str
    dst_node_id: str
    edge_type: str
    native_src_id: str
    native_dst_id: str
    metadata: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def canonical_sort_key(self) -> tuple:
        """Return the tuple used for canonical ordering.

        Ordering is by ``(src_layer, dst_layer, src_node_id, dst_node_id,
        edge_type)`` — deliberately excluding ``edge_id`` and ``metadata``
        so that the sort is stable across re-builds with different ID
        assignment strategies.
        """
        return (
            self.src_layer,
            self.dst_layer,
            self.src_node_id,
            self.dst_node_id,
            self.edge_type,
        )

    def to_hashable_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict for hash computation.

        ``edge_id`` and ``metadata`` are *excluded* so that the hash is
        purely structural and independent of assignment order.
        """
        return {
            "src_layer": self.src_layer,
            "dst_layer": self.dst_layer,
            "src_node_id": self.src_node_id,
            "dst_node_id": self.dst_node_id,
            "edge_type": self.edge_type,
            "native_src_id": self.native_src_id,
            "native_dst_id": self.native_dst_id,
        }


# ---------------------------------------------------------------------------
# Candidate graph
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateGraph:
    """The full candidate circuit graph for one model/task combination.

    Attributes
    ----------
    edges:
        Tuple of :class:`EdgeRecord` objects in canonical order.
        Position ``i`` corresponds to mask index ``i`` (``edge_id == i``).
    graph_hash:
        SHA-256 hex digest computed over the canonically sorted edges.
        Deterministic across processes and machines.
    benchmark_metadata:
        Free-form dict with at least the keys ``benchmark_revision``,
        ``model_id``, and ``task``.
    """

    edges: tuple[EdgeRecord, ...]
    graph_hash: str
    benchmark_metadata: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def num_edges(self) -> int:
        """Total number of edges in the graph."""
        return len(self.edges)

    def edge_by_id(self, edge_id: int) -> EdgeRecord:
        """Return the :class:`EdgeRecord` with the given *edge_id*.

        Raises
        ------
        KeyError
            If no edge with that ID exists.
        """
        if 0 <= edge_id < len(self.edges):
            edge = self.edges[edge_id]
            if edge.edge_id == edge_id:
                return edge
        # Fallback linear scan (shouldn't be needed if IDs are monotonic)
        for edge in self.edges:
            if edge.edge_id == edge_id:
                return edge
        raise KeyError(f"No edge with edge_id={edge_id}")

    def edge_id_to_index(self) -> dict[int, int]:
        """Return a mapping from ``edge_id`` to position in ``self.edges``."""
        return {edge.edge_id: idx for idx, edge in enumerate(self.edges)}


# ---------------------------------------------------------------------------
# Region record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegionRecord:
    """A named partition of the edge set used for hierarchical search.

    Attributes
    ----------
    region_id:
        Stable, deterministic string identifier for this region
        (e.g. ``"layer_3"`` or ``"layer_2.attn"``).
    edge_ids:
        Frozenset of edge IDs belonging to this region.  Every edge must
        appear in *exactly one* region within a given region list.
    metadata:
        Free-form dict for debugging / visualisation.
    """

    region_id: str
    edge_ids: frozenset[int]
    metadata: dict[str, Any] = field(
        default_factory=dict,
        compare=False,
        hash=False,
    )

    @property
    def size(self) -> int:
        """Number of edges in this region."""
        return len(self.edge_ids)

    def contains(self, edge_id: int) -> bool:
        """Return ``True`` if *edge_id* belongs to this region."""
        return edge_id in self.edge_ids


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

REQUIRED_BENCHMARK_KEYS: frozenset[str] = frozenset(
    {"benchmark_revision", "model_id", "task"}
)


def validate_benchmark_metadata(metadata: dict[str, Any]) -> None:
    """Raise :exc:`ValueError` if *metadata* is missing required keys."""
    missing = REQUIRED_BENCHMARK_KEYS - metadata.keys()
    if missing:
        raise ValueError(
            f"benchmark_metadata is missing required keys: {sorted(missing)}"
        )


def validate_edge_records(edges: list[EdgeRecord]) -> None:
    """Perform basic structural validation on a list of :class:`EdgeRecord`.

    Checks:
    * No duplicate ``edge_id`` values.
    * No duplicate canonical sort keys (which would indicate structurally
      identical edges).
    * ``native_src_id`` and ``native_dst_id`` are non-empty strings.

    Raises
    ------
    ValueError
        On any validation failure, with a descriptive message.
    """
    seen_ids: set[int] = set()
    seen_keys: set[tuple] = set()

    for edge in edges:
        if edge.edge_id in seen_ids:
            raise ValueError(
                f"Duplicate edge_id={edge.edge_id} detected in edge list."
            )
        seen_ids.add(edge.edge_id)

        key = edge.canonical_sort_key
        if key in seen_keys:
            raise ValueError(
                f"Duplicate canonical sort key {key!r} detected — two "
                "structurally identical edges."
            )
        seen_keys.add(key)

        if not edge.native_src_id:
            raise ValueError(
                f"Edge {edge.edge_id}: native_src_id must be a non-empty string."
            )
        if not edge.native_dst_id:
            raise ValueError(
                f"Edge {edge.edge_id}: native_dst_id must be a non-empty string."
            )
