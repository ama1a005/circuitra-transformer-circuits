"""
tests.unit.test_regions
=======================

Unit tests for the region-partition utilities in ``hrl_circuits.graph``.

Tests cover:
* Every edge appears in exactly one region (partition correctness).
* Region IDs are stable across identical graph inputs.
* No region is empty after construction.
* Assignment is deterministic (multiple calls → identical output).
"""

from __future__ import annotations

import pytest

from hrl_circuits.graph import RegionBuilder
from hrl_circuits.graph.schema import CandidateGraph, RegionRecord
from tests.fixtures.mock_data import make_tiny_graph, make_tiny_regions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_edge_ids(regions: list[RegionRecord]) -> list[int]:
    """Collect all edge IDs across all regions (may contain duplicates if bug)."""
    ids: list[int] = []
    for r in regions:
        ids.extend(r.edge_ids)
    return ids


# ---------------------------------------------------------------------------
# test_region_partition
# ---------------------------------------------------------------------------

class TestRegionPartition:
    def test_region_partition(self) -> None:
        """Every edge in the graph appears in exactly one region."""
        graph = make_tiny_graph(16)
        regions = make_tiny_regions(graph, mode="component")
        all_ids = _all_edge_ids(regions)
        expected = {e.edge_id for e in graph.edges}

        # No missing edges
        assert set(all_ids) == expected, (
            f"Missing edges: {expected - set(all_ids)}"
        )
        # No duplicates
        assert len(all_ids) == len(expected), (
            f"Duplicate edge IDs across regions: {len(all_ids)} vs {len(expected)}"
        )

    def test_region_partition_layer_mode(self) -> None:
        """Layer-mode partition also covers every edge exactly once."""
        graph = make_tiny_graph(16)
        regions = make_tiny_regions(graph, mode="layer")
        all_ids = _all_edge_ids(regions)
        expected = {e.edge_id for e in graph.edges}
        assert set(all_ids) == expected
        assert len(all_ids) == len(expected)

    def test_region_partition_covers_all_sizes(self) -> None:
        """Partition is valid for graphs of various sizes."""
        for n in (4, 8, 16, 32):
            graph = make_tiny_graph(n)
            regions = make_tiny_regions(graph)
            all_ids = _all_edge_ids(regions)
            assert len(all_ids) == n, f"Failed for n={n}"

    def test_region_builder_partition(self) -> None:
        """RegionBuilder.build produces a valid partition via the production API."""
        graph = make_tiny_graph(16)
        builder = RegionBuilder()
        regions = builder.build(graph)
        all_ids = _all_edge_ids(regions)
        expected = {e.edge_id for e in graph.edges}
        assert set(all_ids) == expected
        assert len(all_ids) == len(expected)


# ---------------------------------------------------------------------------
# test_region_ids_stable
# ---------------------------------------------------------------------------

class TestRegionIdsStable:
    def test_region_ids_stable(self) -> None:
        """Same graph always produces the same region IDs in the same order."""
        graph = make_tiny_graph(16)
        regions_a = make_tiny_regions(graph, mode="component")
        regions_b = make_tiny_regions(graph, mode="component")
        assert [r.region_id for r in regions_a] == [r.region_id for r in regions_b]

    def test_region_ids_stable_layer(self) -> None:
        graph = make_tiny_graph(16)
        ra = make_tiny_regions(graph, mode="layer")
        rb = make_tiny_regions(graph, mode="layer")
        assert [r.region_id for r in ra] == [r.region_id for r in rb]

    def test_region_builder_ids_stable(self) -> None:
        """RegionBuilder.build IDs are stable across calls."""
        graph = make_tiny_graph(16)
        builder = RegionBuilder()
        ra = builder.build(graph)
        rb = builder.build(graph)
        assert [r.region_id for r in ra] == [r.region_id for r in rb]


# ---------------------------------------------------------------------------
# test_no_empty_regions
# ---------------------------------------------------------------------------

class TestNoEmptyRegions:
    def test_no_empty_regions(self) -> None:
        """No region produced by make_tiny_regions has zero edges."""
        graph = make_tiny_graph(16)
        for mode in ("component", "layer"):
            regions = make_tiny_regions(graph, mode=mode)
            for r in regions:
                assert r.size > 0, (
                    f"Region {r.region_id!r} in mode={mode!r} is empty"
                )

    def test_region_builder_no_empty(self) -> None:
        """RegionBuilder never produces empty regions."""
        graph = make_tiny_graph(16)
        builder = RegionBuilder()
        regions = builder.build(graph)
        for r in regions:
            assert r.size > 0, f"RegionBuilder produced empty region: {r.region_id!r}"

    def test_single_edge_graph_one_region(self) -> None:
        """A one-edge graph must produce exactly one non-empty region."""
        graph = make_tiny_graph(1)
        regions = make_tiny_regions(graph)
        assert len(regions) >= 1
        assert all(r.size > 0 for r in regions)
        total = sum(r.size for r in regions)
        assert total == 1


# ---------------------------------------------------------------------------
# test_region_assignment_deterministic
# ---------------------------------------------------------------------------

class TestRegionAssignmentDeterministic:
    def test_region_assignment_deterministic(self) -> None:
        """Region assignment (edge → region_id) is identical across multiple calls."""
        graph = make_tiny_graph(16)

        def assignment(regions: list[RegionRecord]) -> dict[int, str]:
            return {eid: r.region_id for r in regions for eid in r.edge_ids}

        regions_a = make_tiny_regions(graph, mode="component")
        regions_b = make_tiny_regions(graph, mode="component")
        assert assignment(regions_a) == assignment(regions_b)

    def test_region_assignment_deterministic_different_instances(self) -> None:
        """Regions built from independently constructed graphs are identical."""
        graph_a = make_tiny_graph(16)
        graph_b = make_tiny_graph(16)

        def assignment(regions: list[RegionRecord]) -> dict[int, str]:
            return {eid: r.region_id for r in regions for eid in r.edge_ids}

        assert assignment(make_tiny_regions(graph_a)) == assignment(
            make_tiny_regions(graph_b)
        )

    def test_region_frozenset_immutable(self) -> None:
        """edge_ids in a RegionRecord must be a frozenset (immutable)."""
        graph = make_tiny_graph(8)
        regions = make_tiny_regions(graph)
        for r in regions:
            assert isinstance(r.edge_ids, frozenset)
            with pytest.raises((AttributeError, TypeError)):
                r.edge_ids.add(9999)  # type: ignore[attr-defined]
