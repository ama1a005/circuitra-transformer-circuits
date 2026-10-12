"""
hrl_circuits.benchmark.circuit_io
==================================
Utilities for persisting and loading circuit masks and full circuit
manifests.

A *circuit manifest* bundles the binary mask, the graph hash it refers to,
run provenance (run_id, timestamp), and the live-edge list into a single
directory so that circuits can be reliably reproduced and audited.

Directory structure written by :func:`export_circuit_manifest`
-------------------------------------------------------------
::

    <output_dir>/<run_id>/
        mask.npy           – uint8 binary mask array
        mask_metadata.json – metadata sidecar for the mask
        manifest.json      – full CircuitManifest (mask inline as list)

"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..graph.schema import CandidateGraph


# ---------------------------------------------------------------------------
# CircuitManifest dataclass
# ---------------------------------------------------------------------------

@dataclass
class CircuitManifest:
    """Self-contained record of a circuit produced by HRL search.

    Attributes
    ----------
    run_id:
        Unique identifier for the search run (e.g. UUID or timestamp-slug).
    graph_hash:
        SHA-256 of the :class:`~hrl_circuits.graph.CandidateGraph` used.
    mask:
        Binary mask as a Python list of ints (0/1), in canonical edge order.
        Stored as a list for JSON serialisability.
    live_edge_ids:
        Edge IDs where ``mask[i] == 1``.
    num_edges_total:
        Total number of edges in the candidate graph.
    num_edges_live:
        Number of live edges (sum of mask).
    benchmark_metadata:
        Free-form provenance dict (model_id, task, split, etc.).
    timestamp_utc:
        ISO-8601 UTC timestamp of when the manifest was created.
    extra_metadata:
        Optional free-form dict for experiment tracking.
    """

    run_id: str
    graph_hash: str
    mask: list[int]
    live_edge_ids: list[int]
    num_edges_total: int
    num_edges_live: int
    benchmark_metadata: dict[str, Any] = field(default_factory=dict)
    timestamp_utc: str = field(default_factory=lambda: _utc_now())
    extra_metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mask I/O
# ---------------------------------------------------------------------------

def save_mask(
    mask: np.ndarray,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a binary mask array to *path* with an optional JSON sidecar.

    Parameters
    ----------
    mask:
        Binary uint8 mask array.
    path:
        Destination ``.npy`` file path.  Parent directories created as
        needed.
    metadata:
        Optional dict written to ``<path>.json`` alongside the array.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(mask, dtype=np.uint8))

    meta = metadata or {}
    meta.setdefault("dtype", "uint8")
    meta.setdefault("shape", list(mask.shape))
    meta.setdefault("num_live", int(np.sum(mask)))
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_mask(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a mask array and its metadata sidecar.

    Parameters
    ----------
    path:
        Path to the ``.npy`` file.  The JSON sidecar ``<path>.json`` is
        loaded if it exists; otherwise an empty dict is returned.

    Returns
    -------
    tuple[np.ndarray, dict]
        ``(mask_array, metadata_dict)``

    Raises
    ------
    FileNotFoundError
        If the ``.npy`` file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    mask = np.load(path).astype(np.uint8)

    sidecar = path.with_suffix(".json")
    metadata: dict[str, Any] = {}
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))

    return mask, metadata


# ---------------------------------------------------------------------------
# Circuit manifest export
# ---------------------------------------------------------------------------

def export_circuit_manifest(
    mask: np.ndarray,
    graph: CandidateGraph,
    output_dir: str | Path,
    run_id: str,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Export the full circuit manifest for *mask* to *output_dir*.

    Creates the directory ``<output_dir>/<run_id>/`` and writes:

    * ``mask.npy`` — the raw mask array
    * ``mask_metadata.json`` — sidecar for the mask
    * ``manifest.json`` — full :class:`CircuitManifest`

    Parameters
    ----------
    mask:
        Binary uint8 mask array (shape ``(graph.num_edges,)``).
    graph:
        The :class:`~hrl_circuits.graph.CandidateGraph` this mask refers to.
    output_dir:
        Root output directory.
    run_id:
        Unique identifier for this run.
    extra_metadata:
        Optional additional provenance information.

    Returns
    -------
    Path
        Path to the written ``manifest.json`` file.

    Raises
    ------
    ValueError
        If ``len(mask) != graph.num_edges``.
    """
    mask = np.asarray(mask, dtype=np.uint8)

    if len(mask) != graph.num_edges:
        raise ValueError(
            f"Mask length ({len(mask)}) does not match graph.num_edges "
            f"({graph.num_edges})."
        )

    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Identify live edges
    live_edge_ids: list[int] = [
        int(i) for i in np.where(mask)[0]
    ]

    # Build manifest
    manifest = CircuitManifest(
        run_id=run_id,
        graph_hash=graph.graph_hash,
        mask=mask.tolist(),
        live_edge_ids=live_edge_ids,
        num_edges_total=graph.num_edges,
        num_edges_live=len(live_edge_ids),
        benchmark_metadata=dict(graph.benchmark_metadata),
        extra_metadata=extra_metadata or {},
    )

    # Write mask array + sidecar
    mask_path = run_dir / "mask.npy"
    save_mask(
        mask,
        mask_path,
        metadata={
            "run_id": run_id,
            "graph_hash": graph.graph_hash,
            "num_live": len(live_edge_ids),
            "num_total": graph.num_edges,
        },
    )

    # Write manifest JSON
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return manifest_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"
