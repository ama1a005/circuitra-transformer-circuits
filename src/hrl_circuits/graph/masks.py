"""
hrl_circuits.graph.masks
========================
Binary mask utilities for progressive edge pruning.

A *mask* is a ``numpy.ndarray`` of dtype ``np.uint8`` with shape
``(num_edges,)``.  A value of ``1`` means the edge is *live* (retained);
``0`` means the edge has been pruned.

Key design invariants
---------------------
* **Monotonicity**: pruning is irreversible.  :func:`apply_prune` returns a
  *new* mask that is a subset of the old one.  :func:`validate_monotonic`
  checks this.
* **Index-stable**: position ``i`` in the mask always corresponds to
  ``edge_id == i`` (as guaranteed by :class:`CandidateGraphBuilder`).
* All functions are pure — they never modify their input arrays.
"""

from __future__ import annotations

import numpy as np

from .schema import EdgeRecord, RegionRecord


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_num_edges(num_edges) -> int:
    """Accept either an int or a CandidateGraph and return the edge count."""
    if isinstance(num_edges, int):
        return num_edges
    # Duck-type for CandidateGraph (avoids circular imports)
    if hasattr(num_edges, "num_edges"):
        return num_edges.num_edges
    raise TypeError(
        f"Expected int or CandidateGraph, got {type(num_edges).__name__!r}"
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def make_full_mask(num_edges) -> np.ndarray:
    """Return an all-ones mask (all edges live).

    Parameters
    ----------
    num_edges:
        Number of edges in the graph, or a ``CandidateGraph`` instance.

    Returns
    -------
    np.ndarray
        Shape ``(num_edges,)``, dtype ``uint8``, all values 1.
    """
    n = _resolve_num_edges(num_edges)
    if n < 0:
        raise ValueError(f"num_edges must be non-negative, got {n}.")
    return np.ones(n, dtype=np.uint8)


def make_empty_mask(num_edges) -> np.ndarray:
    """Return an all-zeros mask (all edges pruned).

    Parameters
    ----------
    num_edges:
        Number of edges in the graph, or a ``CandidateGraph`` instance.

    Returns
    -------
    np.ndarray
        Shape ``(num_edges,)``, dtype ``uint8``, all values 0.
    """
    n = _resolve_num_edges(num_edges)
    if n < 0:
        raise ValueError(f"num_edges must be non-negative, got {n}.")
    return np.zeros(n, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def apply_prune(mask: np.ndarray, edge_index: int) -> np.ndarray:
    """Return a new mask with position *edge_index* set to 0 (pruned).

    The operation is monotonic: once pruned, an edge cannot be revived.
    If the edge is already pruned, a copy of the mask is returned
    unchanged (no-op, no error).

    Parameters
    ----------
    mask:
        Current mask array.
    edge_index:
        Index (== ``edge_id``) of the edge to prune.

    Returns
    -------
    np.ndarray
        New mask with the specified edge pruned.

    Raises
    ------
    IndexError
        If *edge_index* is out of bounds for *mask*.
    """
    if edge_index < 0 or edge_index >= len(mask):
        raise IndexError(
            f"edge_index={edge_index} is out of bounds for mask of "
            f"length {len(mask)}."
        )
    new_mask = mask.copy()
    new_mask[edge_index] = 0
    return new_mask


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def get_live_edges(mask, edges) -> list:
    """Return the subset of *edges* whose mask position is live (1).

    Accepts two calling conventions:

    1. ``get_live_edges(mask, edges)`` — standard (mask first, edges second)
    2. ``get_live_edges(graph, mask)`` — convenience (CandidateGraph first, mask second)

    Parameters
    ----------
    mask:
        Current mask array (ndarray or list), OR a CandidateGraph.
    edges:
        List of EdgeRecord objects, a CandidateGraph, or the mask array
        when *mask* is a CandidateGraph.

    Returns
    -------
    list[EdgeRecord]
        Live edges in canonical order.
    """
    # Detect CandidateGraph-first calling convention
    if hasattr(mask, "edges"):
        # Called as get_live_edges(graph, mask_array)
        graph = mask
        mask_arr = np.asarray(edges, dtype=np.uint8)
        edge_list = graph.edges
    else:
        mask_arr = np.asarray(mask, dtype=np.uint8)
        if hasattr(edges, "edges"):
            edge_list = edges.edges
        else:
            edge_list = edges

    if len(mask_arr) != len(edge_list):
        raise ValueError(
            f"mask length ({len(mask_arr)}) does not match number of edges "
            f"({len(edge_list)})."
        )
    return [edge for edge, live in zip(edge_list, mask_arr) if live]


def get_live_edge_indices(mask) -> list[int]:
    """Return the list of indices where ``mask[i] == 1``.

    Parameters
    ----------
    mask:
        Current mask array (ndarray or list).

    Returns
    -------
    list[int]
        Sorted list of live indices.
    """
    return list(np.where(np.asarray(mask))[0].astype(int))


def mask_for_region(
    mask_or_graph,
    region,
    edge_id_to_index=None,
) -> np.ndarray:
    """Return a sub-mask restricted to the edges in *region*.

    Accepts two calling conventions:

    1. ``mask_for_region(mask, region, edge_id_to_index)`` — standard
    2. ``mask_for_region(graph, region)`` — convenience (auto full mask)

    Parameters
    ----------
    mask_or_graph:
        Global mask array (ndarray or list), OR a CandidateGraph.
    region:
        The RegionRecord to restrict to.
    edge_id_to_index:
        Mapping from edge_id to mask position. If None and mask_or_graph
        is a CandidateGraph, built automatically from graph.
        If None and mask_or_graph is an array, edge_id == position.

    Returns
    -------
    np.ndarray
        New mask where only region edges can be live.
    """
    # Detect CandidateGraph-first calling convention
    if hasattr(mask_or_graph, "edges"):
        graph = mask_or_graph
        mask_arr = make_full_mask(graph.num_edges)
        id_map = graph.edge_id_to_index()
    else:
        mask_arr = np.asarray(mask_or_graph, dtype=np.uint8)
        id_map = edge_id_to_index  # may be None

    region_mask = np.zeros_like(mask_arr)
    for edge_id in region.edge_ids:
        if id_map is not None:
            idx = id_map.get(edge_id)
        else:
            idx = edge_id  # assume edge_id == position
        if idx is not None and 0 <= idx < len(mask_arr):
            region_mask[idx] = mask_arr[idx]
    return region_mask


# ---------------------------------------------------------------------------
# Monotonicity validation
# ---------------------------------------------------------------------------

def validate_monotonic(old_mask, new_mask) -> bool:
    """Return ``True`` if *new_mask* is a valid monotonic update of *old_mask*.

    A monotonic update satisfies:  for every index ``i``,
    ``new_mask[i] <= old_mask[i]``.  In other words, no edge that was
    previously pruned (0) can become live (1).

    Parameters
    ----------
    old_mask:
        The mask before the update (ndarray or list).
    new_mask:
        The mask after the update (ndarray or list).

    Returns
    -------
    bool
        ``True`` if the update is monotonic, ``False`` otherwise.

    Raises
    ------
    ValueError
        If the masks have different lengths.
    """
    old = np.asarray(old_mask, dtype=np.uint8)
    new = np.asarray(new_mask, dtype=np.uint8)
    if old.shape != new.shape:
        raise ValueError(
            f"Mask shape mismatch: old={old.shape}, new={new.shape}."
        )
    # new must be a subset of old
    return bool(np.all(new <= old))
