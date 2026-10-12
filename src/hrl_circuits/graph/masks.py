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
# Construction
# ---------------------------------------------------------------------------

def make_full_mask(num_edges: int) -> np.ndarray:
    """Return an all-ones mask (all edges live).

    Parameters
    ----------
    num_edges:
        Number of edges in the graph.

    Returns
    -------
    np.ndarray
        Shape ``(num_edges,)``, dtype ``uint8``, all values 1.
    """
    if num_edges < 0:
        raise ValueError(f"num_edges must be non-negative, got {num_edges}.")
    return np.ones(num_edges, dtype=np.uint8)


def make_empty_mask(num_edges: int) -> np.ndarray:
    """Return an all-zeros mask (all edges pruned).

    Parameters
    ----------
    num_edges:
        Number of edges in the graph.

    Returns
    -------
    np.ndarray
        Shape ``(num_edges,)``, dtype ``uint8``, all values 0.
    """
    if num_edges < 0:
        raise ValueError(f"num_edges must be non-negative, got {num_edges}.")
    return np.zeros(num_edges, dtype=np.uint8)


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

def get_live_edges(mask: np.ndarray, edges: list[EdgeRecord]) -> list[EdgeRecord]:
    """Return the subset of *edges* whose mask position is live (1).

    Parameters
    ----------
    mask:
        Current mask array.
    edges:
        List of :class:`~hrl_circuits.graph.schema.EdgeRecord` objects
        (from ``graph.edges``).  Assumed to be in canonical order so that
        ``edges[i].edge_id == i``.

    Returns
    -------
    list[EdgeRecord]
        Live edges in canonical order.

    Raises
    ------
    ValueError
        If ``len(mask) != len(edges)``.
    """
    if len(mask) != len(edges):
        raise ValueError(
            f"mask length ({len(mask)}) does not match number of edges "
            f"({len(edges)})."
        )
    return [edge for edge, live in zip(edges, mask) if live]


def get_live_edge_indices(mask: np.ndarray) -> list[int]:
    """Return the list of indices where ``mask[i] == 1``.

    Parameters
    ----------
    mask:
        Current mask array.

    Returns
    -------
    list[int]
        Sorted list of live indices.
    """
    return list(np.where(mask)[0].astype(int))


def mask_for_region(
    mask: np.ndarray,
    region: RegionRecord,
    edge_id_to_index: dict[int, int],
) -> np.ndarray:
    """Return a sub-mask restricted to the edges in *region*.

    The returned array has the same length as *mask*; positions outside
    the region are set to 0.

    Parameters
    ----------
    mask:
        Global mask array.
    region:
        The :class:`~hrl_circuits.graph.schema.RegionRecord` to restrict to.
    edge_id_to_index:
        Mapping from ``edge_id`` to position in *mask* (from
        :meth:`~hrl_circuits.graph.schema.CandidateGraph.edge_id_to_index`).

    Returns
    -------
    np.ndarray
        New mask where only region edges can be live.
    """
    region_mask = np.zeros_like(mask)
    for edge_id in region.edge_ids:
        idx = edge_id_to_index.get(edge_id)
        if idx is not None and idx < len(mask):
            region_mask[idx] = mask[idx]
    return region_mask


# ---------------------------------------------------------------------------
# Monotonicity validation
# ---------------------------------------------------------------------------

def validate_monotonic(old_mask: np.ndarray, new_mask: np.ndarray) -> bool:
    """Return ``True`` if *new_mask* is a valid monotonic update of *old_mask*.

    A monotonic update satisfies:  for every index ``i``,
    ``new_mask[i] <= old_mask[i]``.  In other words, no edge that was
    previously pruned (0) can become live (1).

    Parameters
    ----------
    old_mask:
        The mask before the update.
    new_mask:
        The mask after the update.

    Returns
    -------
    bool
        ``True`` if the update is monotonic, ``False`` otherwise.

    Raises
    ------
    ValueError
        If the masks have different shapes.
    """
    if old_mask.shape != new_mask.shape:
        raise ValueError(
            f"Mask shape mismatch: old={old_mask.shape}, new={new_mask.shape}."
        )
    # new_mask must be a subset of old_mask
    return bool(np.all(new_mask <= old_mask))
