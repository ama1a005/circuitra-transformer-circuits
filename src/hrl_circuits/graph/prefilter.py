"""
hrl_circuits.graph.prefilter
============================
Pre-filter artifact: stores edge importance scores produced on the *train*
split by any scoring method (e.g. gradient attribution, activation
patching).  The artifact is the authoritative source for the initial
candidate ranking fed into the HRL search.

Spec reference: section 8.4

Safety invariants
-----------------
* A :class:`PrefilterArtifact` with ``split != 'train'`` **cannot** be
  used for training comparisons — :func:`validate_prefilter_against_graph`
  enforces this.
* The ``graph_hash`` in the artifact must match the graph's own
  ``graph_hash``; otherwise the ranking is stale.
* ``ranked_edge_ids`` must be a permutation of the keys in ``scores``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .schema import CandidateGraph


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PrefilterArtifact:
    """Serialisable record of edge importance scores for one (model, task, split).

    Fields
    ------
    graph_hash:
        SHA-256 hex digest of the :class:`CandidateGraph` these scores
        were computed on.  Used for staleness detection.
    method:
        Name of the scoring algorithm, e.g. ``"gradient_attribution"`` or
        ``"activation_patching"``.
    split:
        The data split used to compute scores.  Must be ``"train"`` for
        artifacts that will drive the HRL search.
    task:
        Name of the circuit localisation task, e.g. ``"ioi"``.
    top_k:
        The ``k`` value used when this artifact was originally created
        (informational; :func:`select_top_k` accepts an arbitrary ``k``).
    scores:
        Mapping of ``edge_id`` (as *str* key for JSON compat) to
        importance score (float).  Higher scores → more important.
    ranked_edge_ids:
        Edge IDs sorted by descending score (pre-computed).
        ``ranked_edge_ids[0]`` is the most important edge.
    benchmark_revision:
        Git tag / commit SHA of the benchmark that produced these scores.
    config_hash:
        SHA-256 hex digest of the scoring configuration, for
        reproducibility tracking.
    extra_metadata:
        Optional free-form dict for provenance / experiment tracking.
    """

    graph_hash: str
    method: str
    split: str
    task: str
    top_k: int
    scores: dict[str, float]          # str keys for JSON serialisation
    ranked_edge_ids: list[int]
    benchmark_revision: str
    config_hash: str
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def score_for_edge(self, edge_id: int) -> float:
        """Return the importance score for *edge_id*.

        Raises
        ------
        KeyError
            If *edge_id* is not in ``scores``.
        """
        return self.scores[str(edge_id)]

    @property
    def num_edges_scored(self) -> int:
        """Number of edges with a recorded score."""
        return len(self.scores)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_prefilter(artifact: PrefilterArtifact, path: str | Path) -> None:
    """Persist *artifact* to *path* as a UTF-8 JSON file.

    Parameters
    ----------
    artifact:
        The :class:`PrefilterArtifact` to save.
    path:
        Destination file path.  Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(artifact)
    # ranked_edge_ids stored as a list of ints — JSON handles this natively
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def load_prefilter(path: str | Path) -> PrefilterArtifact:
    """Load a :class:`PrefilterArtifact` from a JSON file.

    Parameters
    ----------
    path:
        Path to the JSON file previously created by :func:`save_prefilter`.

    Returns
    -------
    PrefilterArtifact

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file is malformed or missing required fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PrefilterArtifact file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    required_keys = {
        "graph_hash",
        "method",
        "split",
        "task",
        "top_k",
        "scores",
        "ranked_edge_ids",
        "benchmark_revision",
        "config_hash",
    }
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(
            f"PrefilterArtifact JSON at {path} is missing keys: "
            f"{sorted(missing)}"
        )

    # ranked_edge_ids should be list[int]
    ranked_edge_ids = [int(x) for x in data["ranked_edge_ids"]]

    return PrefilterArtifact(
        graph_hash=str(data["graph_hash"]),
        method=str(data["method"]),
        split=str(data["split"]),
        task=str(data["task"]),
        top_k=int(data["top_k"]),
        scores={str(k): float(v) for k, v in data["scores"].items()},
        ranked_edge_ids=ranked_edge_ids,
        benchmark_revision=str(data["benchmark_revision"]),
        config_hash=str(data["config_hash"]),
        extra_metadata=data.get("extra_metadata", {}),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_prefilter_against_graph(
    artifact: PrefilterArtifact,
    graph: CandidateGraph,
) -> None:
    """Validate that *artifact* is compatible with *graph*.

    Checks performed
    ----------------
    1. ``artifact.graph_hash == graph.graph_hash`` — the artifact was
       produced on the same graph structure.
    2. ``artifact.split == 'train'`` — only train-split artifacts may be
       used for training-time comparisons.
    3. Every ``edge_id`` in ``artifact.scores`` exists in the graph.
    4. ``ranked_edge_ids`` is a permutation of the scored edge IDs.

    Raises
    ------
    PrefilterValidationError
        On any validation failure, with a descriptive message.
    """
    # 1. Graph hash match
    if artifact.graph_hash != graph.graph_hash:
        raise PrefilterValidationError(
            f"graph_hash mismatch: artifact has {artifact.graph_hash!r} but "
            f"current graph has {graph.graph_hash!r}.  "
            "Regenerate the pre-filter artifact."
        )

    # 2. Split guard
    if artifact.split != "train":
        raise PrefilterValidationError(
            f"PrefilterArtifact.split must be 'train' for training "
            f"comparisons, got {artifact.split!r}."
        )

    # 3. All scored edge IDs exist in the graph
    graph_edge_ids = {e.edge_id for e in graph.edges}
    scored_ids = {int(k) for k in artifact.scores.keys()}
    unknown = scored_ids - graph_edge_ids
    if unknown:
        raise PrefilterValidationError(
            f"PrefilterArtifact contains scores for edge IDs not in the "
            f"graph: {sorted(unknown)}"
        )

    # 4. ranked_edge_ids is a permutation of scored IDs
    ranked_set = set(artifact.ranked_edge_ids)
    if ranked_set != scored_ids:
        extra = ranked_set - scored_ids
        missing = scored_ids - ranked_set
        raise PrefilterValidationError(
            f"ranked_edge_ids is not a permutation of scored edge IDs.  "
            f"Extra in ranked: {sorted(extra)}.  "
            f"Missing from ranked: {sorted(missing)}."
        )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_top_k(artifact: PrefilterArtifact, k: int) -> list[int]:
    """Return the top-*k* edge IDs by importance score.

    Parameters
    ----------
    artifact:
        The loaded :class:`PrefilterArtifact`.
    k:
        Number of top edges to return.  Clamped to the total number of
        scored edges.

    Returns
    -------
    list[int]
        Edge IDs in descending order of importance (most important first).

    Raises
    ------
    ValueError
        If ``k <= 0``.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")

    k = min(k, len(artifact.ranked_edge_ids))
    return artifact.ranked_edge_ids[:k]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PrefilterValidationError(ValueError):
    """Raised when a :class:`PrefilterArtifact` fails validation."""
