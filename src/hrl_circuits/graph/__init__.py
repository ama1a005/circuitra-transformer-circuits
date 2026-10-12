"""
hrl_circuits.graph
==================
Graph construction, hashing, region decomposition, mask utilities,
and pre-filter artifact management for HRL Circuit Localization.
"""

from .schema import EdgeRecord, CandidateGraph, RegionRecord
from .builder import CandidateGraphBuilder
from .hash import compute_graph_hash
from .regions import RegionBuilder
from .masks import (
    make_full_mask,
    make_empty_mask,
    apply_prune,
    get_live_edges,
    get_live_edge_indices,
    mask_for_region,
    validate_monotonic,
)
from .prefilter import (
    PrefilterArtifact,
    save_prefilter,
    load_prefilter,
    validate_prefilter_against_graph,
    select_top_k,
)

__all__ = [
    # schema
    "EdgeRecord",
    "CandidateGraph",
    "RegionRecord",
    # builder
    "CandidateGraphBuilder",
    # hash
    "compute_graph_hash",
    # regions
    "RegionBuilder",
    # masks
    "make_full_mask",
    "make_empty_mask",
    "apply_prune",
    "get_live_edges",
    "get_live_edge_indices",
    "mask_for_region",
    "validate_monotonic",
    # prefilter
    "PrefilterArtifact",
    "save_prefilter",
    "load_prefilter",
    "validate_prefilter_against_graph",
    "select_top_k",
]
