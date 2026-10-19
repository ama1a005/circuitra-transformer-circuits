"""
hrl_circuits.graph.regions
==========================
``RegionBuilder`` — partitions the edge set of a :class:`CandidateGraph`
into non-overlapping, exhaustive :class:`RegionRecord` groups.

Two built-in strategies are provided:

layer_regions
    One region per unique ``dst_layer`` value.  Region IDs are of the
    form ``"layer_{dst_layer}"``.

component_regions
    One region per ``(dst_layer, edge_type)`` combination (the *component
    type* within a layer, e.g. attention vs. MLP).  Region IDs are of the
    form ``"layer_{dst_layer}.{edge_type}"``.

Invariants
----------
* Empty regions are **never** emitted.
* Every edge in the graph appears in **exactly one** region.
* ``region_id`` strings are deterministic and human-readable.
* Regions within the returned list are sorted by ``region_id`` for
  reproducibility.
"""

from __future__ import annotations

from collections import defaultdict

from .schema import CandidateGraph, EdgeRecord, RegionRecord


class RegionBuilder:
    """Build region partitions from a :class:`CandidateGraph`.

    All methods are pure — they do not mutate the graph.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_layer_regions(self, graph: CandidateGraph) -> list[RegionRecord]:
        """Partition edges into one region per unique ``dst_layer``.

        Parameters
        ----------
        graph:
            The fully built :class:`CandidateGraph`.

        Returns
        -------
        list[RegionRecord]
            Sorted by ``region_id``; empty regions are excluded.

        Raises
        ------
        ValueError
            If the resulting partition does not cover all edges (sanity
            check — should not happen if the graph is valid).
        """
        buckets: dict[int, set[int]] = defaultdict(set)
        for edge in graph.edges:
            buckets[edge.dst_layer].add(edge.edge_id)

        regions = [
            RegionRecord(
                region_id=f"layer_{layer}",
                edge_ids=frozenset(edge_ids),
                metadata={"dst_layer": layer},
            )
            for layer, edge_ids in buckets.items()
            if edge_ids  # exclude empty (shouldn't happen, but defensive)
        ]

        regions.sort(key=lambda r: r.region_id)
        self._validate_partition(graph, regions)
        return regions

    def build_component_regions(
        self, graph: CandidateGraph
    ) -> list[RegionRecord]:
        """Partition edges by ``(dst_layer, edge_type)`` (component type).

        Parameters
        ----------
        graph:
            The fully built :class:`CandidateGraph`.

        Returns
        -------
        list[RegionRecord]
            Sorted by ``region_id``; empty regions are excluded.
        """
        buckets: dict[tuple[int, str], set[int]] = defaultdict(set)
        for edge in graph.edges:
            key = (edge.dst_layer, edge.edge_type)
            buckets[key].add(edge.edge_id)

        regions = [
            RegionRecord(
                region_id=f"layer_{layer}.{edge_type}",
                edge_ids=frozenset(edge_ids),
                metadata={"dst_layer": layer, "edge_type": edge_type},
            )
            for (layer, edge_type), edge_ids in buckets.items()
            if edge_ids
        ]

        regions.sort(key=lambda r: r.region_id)
        self._validate_partition(graph, regions)
        return regions

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_region_for_edge(
        edge_id: int,
        regions: list[RegionRecord],
    ) -> RegionRecord:
        """Return the :class:`RegionRecord` that contains *edge_id*.

        Parameters
        ----------
        edge_id:
            The ID of the edge to look up.
        regions:
            A list of :class:`RegionRecord` (from either build method).

        Returns
        -------
        RegionRecord
            The region that contains the edge.

        Raises
        ------
        KeyError
            If no region contains *edge_id*.
        """
        for region in regions:
            if region.contains(edge_id):
                return region
        raise KeyError(
            f"edge_id={edge_id} does not belong to any region in the "
            f"provided list of {len(regions)} regions."
        )

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    def build(
        self,
        graph: "CandidateGraph",
        mode: str = "component",
    ) -> list["RegionRecord"]:
        """Convenience entry-point delegating to the chosen strategy.

        Parameters
        ----------
        graph:
            The fully built :class:`CandidateGraph`.
        mode:
            ``"component"`` (default) or ``"layer"``.

        Returns
        -------
        list[RegionRecord]
        """
        if mode == "layer":
            return self.build_layer_regions(graph)
        return self.build_component_regions(graph)

    @staticmethod
    def _validate_partition(
        graph: CandidateGraph,
        regions: list[RegionRecord],
    ) -> None:
        """Assert that *regions* form a valid partition of *graph.edges*.

        Raises
        ------
        ValueError
            If any edge is missing from all regions, or appears in more
            than one region.
        """
        all_edge_ids: set[int] = {e.edge_id for e in graph.edges}
        covered: set[int] = set()

        for region in regions:
            overlap = covered & region.edge_ids
            if overlap:
                raise ValueError(
                    f"Edges {sorted(overlap)} appear in more than one "
                    f"region (duplicate coverage detected in "
                    f"'{region.region_id}')."
                )
            covered |= region.edge_ids

        missing = all_edge_ids - covered
        if missing:
            raise ValueError(
                f"Edges {sorted(missing)} are not covered by any region."
            )
