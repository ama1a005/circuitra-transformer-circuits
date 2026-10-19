"""
tests.unit.test_graph
=====================

Unit tests for ``hrl_circuits.graph.schema`` covering edge ID stability,
hash determinism, native mapping round-trips, and canonical sort ordering.

Spec reference: Section A.4
"""

from __future__ import annotations

import hashlib
import json

import pytest

from hrl_circuits.graph.schema import (
    CandidateGraph,
    EdgeRecord,
    validate_edge_records,
)
from tests.fixtures.mock_data import make_tiny_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_edges(n: int) -> list[EdgeRecord]:
    """Return n deterministic EdgeRecord objects with unique canonical sort keys."""
    edges = []
    for i in range(n):
        src = i          # unique src_layer per edge → unique canonical key
        dst = i + 1
        edges.append(EdgeRecord(
            edge_id=i,
            src_layer=src,
            dst_layer=dst,
            src_node_id=f"L{src}.h0",
            dst_node_id=f"L{dst}.h0",
            edge_type="attention" if i % 2 == 0 else "mlp",
            native_src_id=f"blocks.{src}.attn.hook_q",
            native_dst_id=f"blocks.{dst}.attn.hook_k",
            metadata={"idx": i},
        ))
    return edges


def _graph_hash_from_edges(edges: list[EdgeRecord]) -> str:
    canonical = sorted(
        [e.to_hashable_dict() for e in edges],
        key=lambda d: (d["src_layer"], d["dst_layer"],
                       d["src_node_id"], d["dst_node_id"], d["edge_type"]),
    )
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# test_graph_ids_stable
# ---------------------------------------------------------------------------

class TestGraphIdsStable:
    """Same edge contents must produce identical edge_ids across separate builds."""

    def test_graph_ids_stable(self) -> None:
        """Reconstructing the same edges always yields the same edge_id sequence."""
        edges_a = _build_edges(16)
        edges_b = _build_edges(16)
        for a, b in zip(edges_a, edges_b):
            assert a.edge_id == b.edge_id, (
                f"edge_id mismatch at position {a.edge_id}: {a.edge_id} != {b.edge_id}"
            )

    def test_graph_ids_unique(self) -> None:
        """Every edge_id in the list must be distinct."""
        edges = _build_edges(16)
        ids = [e.edge_id for e in edges]
        assert len(ids) == len(set(ids)), "Duplicate edge_ids found"

    def test_graph_ids_monotonic(self) -> None:
        """edge_ids built by make_tiny_graph are monotonically increasing from 0."""
        graph = make_tiny_graph(16)
        ids = [e.edge_id for e in graph.edges]
        assert ids == list(range(len(ids)))


# ---------------------------------------------------------------------------
# test_graph_hash_stable
# ---------------------------------------------------------------------------

class TestGraphHashStable:
    """SHA-256 graph hash must be deterministic across calls and processes."""

    def test_graph_hash_stable(self) -> None:
        """Same edge contents → same graph hash."""
        edges_a = _build_edges(8)
        edges_b = _build_edges(8)
        hash_a = _graph_hash_from_edges(edges_a)
        hash_b = _graph_hash_from_edges(edges_b)
        assert hash_a == hash_b

    def test_graph_hash_changes_on_mutation(self) -> None:
        """Changing a single edge field must change the graph hash."""
        edges = _build_edges(8)
        hash_original = _graph_hash_from_edges(edges)

        # Build a different edge list with one edge altered
        edges2 = list(edges)
        old = edges2[0]
        edges2[0] = EdgeRecord(
            edge_id=old.edge_id,
            src_layer=old.src_layer,
            dst_layer=old.dst_layer,
            src_node_id="DIFFERENT_SRC",  # ← changed
            dst_node_id=old.dst_node_id,
            edge_type=old.edge_type,
            native_src_id=old.native_src_id,
            native_dst_id=old.native_dst_id,
        )
        hash_changed = _graph_hash_from_edges(edges2)
        assert hash_original != hash_changed

    def test_hash_excludes_metadata(self) -> None:
        """Metadata changes must NOT affect the graph hash (metadata is excluded)."""
        edges_a = _build_edges(4)
        edges_b = [
            EdgeRecord(
                edge_id=e.edge_id,
                src_layer=e.src_layer,
                dst_layer=e.dst_layer,
                src_node_id=e.src_node_id,
                dst_node_id=e.dst_node_id,
                edge_type=e.edge_type,
                native_src_id=e.native_src_id,
                native_dst_id=e.native_dst_id,
                metadata={"completely_different": True},
            )
            for e in edges_a
        ]
        assert _graph_hash_from_edges(edges_a) == _graph_hash_from_edges(edges_b)

    def test_make_tiny_graph_hash_stable(self) -> None:
        """make_tiny_graph always returns the same graph_hash."""
        g1 = make_tiny_graph(16)
        g2 = make_tiny_graph(16)
        assert g1.graph_hash == g2.graph_hash


# ---------------------------------------------------------------------------
# test_graph_native_mapping_roundtrip
# ---------------------------------------------------------------------------

class TestGraphNativeMappingRoundtrip:
    """edge_id → native_src_id/native_dst_id → edge_id must preserve info."""

    def test_graph_native_mapping_roundtrip(self) -> None:
        """Look up an edge by edge_id, read native IDs, find back by native IDs."""
        graph = make_tiny_graph(16)
        for edge in graph.edges:
            # Recover by edge_by_id
            recovered = graph.edge_by_id(edge.edge_id)
            assert recovered.native_src_id == edge.native_src_id
            assert recovered.native_dst_id == edge.native_dst_id

    def test_edge_id_to_index_bijective(self) -> None:
        """edge_id_to_index maps every position to a unique edge_id."""
        graph = make_tiny_graph(16)
        mapping = graph.edge_id_to_index()
        assert len(mapping) == graph.num_edges
        # Values (positions) should also be unique
        assert len(set(mapping.values())) == graph.num_edges

    def test_edge_by_id_raises_on_missing(self) -> None:
        """Requesting a non-existent edge_id must raise KeyError."""
        graph = make_tiny_graph(4)
        with pytest.raises(KeyError):
            graph.edge_by_id(9999)


# ---------------------------------------------------------------------------
# test_no_duplicate_edge_ids
# ---------------------------------------------------------------------------

class TestNoDuplicateEdgeIds:
    def test_no_duplicate_edge_ids(self) -> None:
        """validate_edge_records must raise when duplicate edge_ids exist."""
        edges = _build_edges(4)
        # Inject a duplicate
        dup = EdgeRecord(
            edge_id=0,  # ← same as first edge
            src_layer=99,
            dst_layer=100,
            src_node_id="X.dup_src",
            dst_node_id="X.dup_dst",
            edge_type="mlp",
            native_src_id="blocks.99.hook_q",
            native_dst_id="blocks.100.hook_k",
        )
        with pytest.raises(ValueError, match="Duplicate edge_id"):
            validate_edge_records(edges + [dup])

    def test_no_duplicate_canonical_keys(self) -> None:
        """validate_edge_records must raise on structural duplicates (same canonical key)."""
        from hrl_circuits.graph.schema import EdgeRecord
        # Two edges with different edge_ids but identical structural content
        edge_a = EdgeRecord(
            edge_id=0,
            src_layer=1,
            dst_layer=2,
            src_node_id="L1.h0",
            dst_node_id="L2.h0",
            edge_type="attention",
            native_src_id="blocks.1.hook_q",
            native_dst_id="blocks.2.hook_k",
        )
        edge_b = EdgeRecord(
            edge_id=1,           # different edge_id!
            src_layer=1,         # same structural content → duplicate canonical key
            dst_layer=2,
            src_node_id="L1.h0",
            dst_node_id="L2.h0",
            edge_type="attention",
            native_src_id="blocks.1.hook_q",
            native_dst_id="blocks.2.hook_k",
        )
        with pytest.raises(ValueError, match="Duplicate canonical sort key"):
            validate_edge_records([edge_a, edge_b])


    def test_valid_edge_list_passes(self) -> None:
        """validate_edge_records should not raise for a valid edge list."""
        edges = _build_edges(8)
        validate_edge_records(edges)  # Should not raise

    def test_empty_native_ids_rejected(self) -> None:
        """validate_edge_records should reject empty native IDs."""
        edge = EdgeRecord(
            edge_id=0,
            src_layer=0,
            dst_layer=1,
            src_node_id="L0.h0",
            dst_node_id="L1.h0",
            edge_type="attention",
            native_src_id="",  # ← empty: invalid
            native_dst_id="blocks.1.hook_k",
        )
        with pytest.raises(ValueError, match="native_src_id"):
            validate_edge_records([edge])


# ---------------------------------------------------------------------------
# test_canonical_sort_deterministic
# ---------------------------------------------------------------------------

class TestCanonicalSortDeterministic:
    def test_canonical_sort_deterministic(self) -> None:
        """Sorting edges by canonical_sort_key is order-independent."""
        import random

        edges = _build_edges(16)
        expected_order = sorted(edges, key=lambda e: e.canonical_sort_key)

        for seed in range(5):
            rng = random.Random(seed)
            shuffled = list(edges)
            rng.shuffle(shuffled)
            sorted_again = sorted(shuffled, key=lambda e: e.canonical_sort_key)
            assert sorted_again == expected_order, (
                f"Sort not deterministic with seed {seed}"
            )

    def test_canonical_sort_key_excludes_edge_id(self) -> None:
        """canonical_sort_key does not include edge_id."""
        # Use a distinctive edge_id (99) that won't match layer indices
        edge = EdgeRecord(
            edge_id=99,
            src_layer=0,
            dst_layer=1,
            src_node_id="L0.h0",
            dst_node_id="L1.h0",
            edge_type="attention",
            native_src_id="blocks.0.hook_q",
            native_dst_id="blocks.1.hook_k",
        )
        key = edge.canonical_sort_key
        assert 99 not in key, f"edge_id=99 should not appear in sort key {key}"


    def test_canonical_sort_key_structure(self) -> None:
        """canonical_sort_key is a tuple of (src_layer, dst_layer, src_node, dst_node, type)."""
        edge = _build_edges(1)[0]
        key = edge.canonical_sort_key
        assert isinstance(key, tuple)
        assert len(key) == 5
        assert key[0] == edge.src_layer
        assert key[1] == edge.dst_layer
        assert key[2] == edge.src_node_id
        assert key[3] == edge.dst_node_id
        assert key[4] == edge.edge_type
