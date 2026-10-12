"""
hrl_circuits.graph.hash
=======================
Deterministic SHA-256 graph hash.

The hash is computed over the *canonical* representation of the edge set:
edges are sorted by their ``canonical_sort_key``, serialised to a
newline-separated JSON string (keys alphabetically ordered within each
record), and then SHA-256 digested.

The hash is *independent* of:
* ``edge_id`` assignments
* ``metadata`` fields
* Python process, platform, or hash seed

This makes it safe to use as a cache / artifact invalidation key.
"""

from __future__ import annotations

import hashlib
import json

from .schema import EdgeRecord


def compute_graph_hash(edges: list[EdgeRecord]) -> str:
    """Return a deterministic SHA-256 hex digest for *edges*.

    Parameters
    ----------
    edges:
        List of :class:`~hrl_circuits.graph.schema.EdgeRecord` objects.
        The list need not be pre-sorted — the function sorts internally
        to guarantee determinism regardless of insertion order.

    Returns
    -------
    str
        64-character lowercase hex string (SHA-256 digest).

    Examples
    --------
    >>> from hrl_circuits.graph.schema import EdgeRecord
    >>> e = EdgeRecord(0, 0, 1, "h0.q", "h1.v", "attention", "blk0.attn.q", "blk1.attn.v")
    >>> h = compute_graph_hash([e])
    >>> len(h)
    64
    """
    # Sort by canonical key for determinism
    sorted_edges = sorted(edges, key=lambda e: e.canonical_sort_key)

    # Serialise each edge as a JSON object with alphabetically ordered keys
    lines = [
        json.dumps(edge.to_hashable_dict(), sort_keys=True)
        for edge in sorted_edges
    ]

    # Join with newlines and encode to bytes
    payload = "\n".join(lines).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()
