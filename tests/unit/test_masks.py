"""
tests.unit.test_masks
=====================

Unit tests for the edge-mask utilities in ``hrl_circuits.graph``.

Tests cover:
* make_full_mask / make_empty_mask construction
* apply_prune monotonicity guarantee
* validate_monotonic correctness
* get_live_edges / get_live_edge_indices consistency
"""

from __future__ import annotations

import pytest

from hrl_circuits.graph import (
    apply_prune,
    get_live_edge_indices,
    get_live_edges,
    make_empty_mask,
    make_full_mask,
    mask_for_region,
    validate_monotonic,
)
from hrl_circuits.graph.schema import RegionRecord
from tests.fixtures.mock_data import make_tiny_graph, make_tiny_regions


# ---------------------------------------------------------------------------
# test_full_mask_all_ones
# ---------------------------------------------------------------------------

class TestFullMask:
    def test_full_mask_all_ones(self) -> None:
        """make_full_mask returns a mask where every entry is 1."""
        graph = make_tiny_graph(16)
        mask = make_full_mask(graph)
        assert len(mask) == graph.num_edges
        assert all(m == 1 for m in mask), "Expected all-ones mask"

    def test_full_mask_length(self) -> None:
        for n in (1, 8, 32):
            g = make_tiny_graph(n)
            m = make_full_mask(g)
            assert len(m) == n

    def test_full_mask_type(self) -> None:
        graph = make_tiny_graph(8)
        mask = make_full_mask(graph)
        assert all(isinstance(v, int) for v in mask)


# ---------------------------------------------------------------------------
# test_empty_mask_all_zeros
# ---------------------------------------------------------------------------

class TestEmptyMask:
    def test_empty_mask_all_zeros(self) -> None:
        """make_empty_mask returns a mask where every entry is 0."""
        graph = make_tiny_graph(16)
        mask = make_empty_mask(graph)
        assert len(mask) == graph.num_edges
        assert all(m == 0 for m in mask), "Expected all-zeros mask"

    def test_empty_mask_length(self) -> None:
        for n in (1, 8, 32):
            g = make_tiny_graph(n)
            m = make_empty_mask(g)
            assert len(m) == n


# ---------------------------------------------------------------------------
# test_apply_prune_monotonic
# ---------------------------------------------------------------------------

class TestApplyPruneMonotonic:
    def test_apply_prune_removes_edge(self) -> None:
        """apply_prune sets the specified edge to 0."""
        graph = make_tiny_graph(8)
        mask = make_full_mask(graph)
        new_mask = apply_prune(mask, edge_index=3)
        assert new_mask[3] == 0
        # Other edges unchanged
        for i, v in enumerate(new_mask):
            if i != 3:
                assert v == 1

    def test_apply_prune_monotonic(self) -> None:
        """Applying prune repeatedly: mask[i] can go 1->0 but never 0->1."""
        graph = make_tiny_graph(16)
        mask = make_full_mask(graph)
        history = [list(mask)]
        for i in range(0, 8):
            mask = apply_prune(mask, edge_index=i)
            history.append(list(mask))
        # Verify monotonicity: no position ever goes from 0 back to 1
        for t in range(1, len(history)):
            for j in range(len(history[0])):
                if history[t - 1][j] == 0:
                    assert history[t][j] == 0, (
                        f"Edge {j} went 0→1 at step {t}: monotonicity violated"
                    )

    def test_apply_prune_idempotent_on_zero(self) -> None:
        """Pruning an already-pruned edge leaves the mask unchanged."""
        graph = make_tiny_graph(4)
        mask = make_empty_mask(graph)
        new_mask = apply_prune(mask, edge_index=2)
        assert new_mask[2] == 0
        assert new_mask == mask

    def test_apply_prune_does_not_mutate(self) -> None:
        """apply_prune should return a new mask without mutating the original."""
        graph = make_tiny_graph(8)
        mask = make_full_mask(graph)
        original = list(mask)
        _ = apply_prune(mask, edge_index=0)
        assert list(mask) == original, "apply_prune must not mutate the input mask"


# ---------------------------------------------------------------------------
# test_mask_monotonic
# ---------------------------------------------------------------------------

class TestMaskMonotonic:
    def test_mask_monotonic_valid(self) -> None:
        """validate_monotonic returns True for a valid 1→0 transition."""
        prev = [1, 1, 1, 1]
        curr = [1, 0, 1, 0]
        assert validate_monotonic(prev, curr) is True

    def test_mask_monotonic_invalid(self) -> None:
        """validate_monotonic returns False when a 0→1 flip is detected."""
        prev = [1, 0, 1, 1]
        curr = [1, 1, 1, 1]  # index 1 went 0→1: INVALID
        assert validate_monotonic(prev, curr) is False

    def test_mask_monotonic_same(self) -> None:
        """No change between masks is trivially monotonic."""
        mask = [1, 0, 1, 0]
        assert validate_monotonic(mask, mask) is True

    def test_mask_monotonic_full_prune(self) -> None:
        """Going from full mask to empty mask is monotonic."""
        n = 8
        full = [1] * n
        empty = [0] * n
        assert validate_monotonic(full, empty) is True

    def test_mask_monotonic_length_mismatch_raises(self) -> None:
        """Mismatched mask lengths should raise an error."""
        with pytest.raises((ValueError, AssertionError, IndexError)):
            validate_monotonic([1, 1], [1, 1, 1])


# ---------------------------------------------------------------------------
# test_live_edges_match_mask
# ---------------------------------------------------------------------------

class TestLiveEdgesMatchMask:
    def test_live_edges_match_mask(self) -> None:
        """get_live_edges returns exactly the edges whose mask entry is 1."""
        graph = make_tiny_graph(8)
        mask = [1, 0, 1, 0, 1, 1, 0, 1]
        live = get_live_edges(graph, mask)
        live_ids = {e.edge_id for e in live}

        expected_ids = {i for i, m in enumerate(mask) if m == 1}
        assert live_ids == expected_ids

    def test_live_edge_indices_match_mask(self) -> None:
        """get_live_edge_indices returns exactly the indices where mask==1."""
        graph = make_tiny_graph(8)
        mask = [1, 0, 1, 0, 1, 1, 0, 1]
        indices = get_live_edge_indices(mask)
        expected = [i for i, m in enumerate(mask) if m == 1]
        assert list(indices) == expected

    def test_full_mask_all_live(self) -> None:
        """Full mask → all edges returned as live."""
        graph = make_tiny_graph(8)
        mask = make_full_mask(graph)
        live = get_live_edges(graph, mask)
        assert len(live) == graph.num_edges

    def test_empty_mask_no_live(self) -> None:
        """Empty mask → no live edges."""
        graph = make_tiny_graph(8)
        mask = make_empty_mask(graph)
        live = get_live_edges(graph, mask)
        assert len(live) == 0

    def test_mask_for_region_correct(self) -> None:
        """mask_for_region sets 1 for edges in region, 0 for others."""
        graph = make_tiny_graph(8)
        regions = make_tiny_regions(graph, mode="layer")
        region = regions[0]
        region_mask = mask_for_region(graph, region)
        assert len(region_mask) == graph.num_edges
        for i, v in enumerate(region_mask):
            if graph.edges[i].edge_id in region.edge_ids:
                assert v == 1, f"Edge {i} in region but mask=0"
            else:
                assert v == 0, f"Edge {i} not in region but mask=1"
