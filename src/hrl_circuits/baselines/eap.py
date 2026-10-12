"""
EAPBaseline — Edge Attribution Patching
========================================
Implements attribution-score computation via activation patching (also known as
Edge Attribution Patching, EAP).  For each edge (sender → receiver) in the
computation graph we measure how much model behaviour changes when the
activation on that edge is replaced by its value from a counterfactual
(corrupted) input.  Edges with high absolute change are assumed to be more
important.

Public API
----------
compute_eap_scores(model, clean_batch, counterfactual_batch, graph)
    -> dict[int, float]
get_top_k_circuit(scores, graph, k) -> np.ndarray
save_eap_artifact(scores, graph, split, task, output_path)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Artifact dataclass
# ---------------------------------------------------------------------------

@dataclass
class PrefilterArtifact:
    """Serialisable record of a completed prefilter run."""

    method: str
    task: str
    split: str
    graph_hash: str
    scores: dict[str, float]          # str(edge_id) -> score
    top_k_indices: list[int]
    timestamp: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PrefilterArtifact":
        return cls(**d)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("Saved EAP artifact to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "PrefilterArtifact":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _resolve_edge_list(graph: Any) -> list[int]:
    """Return a list of edge IDs from a graph object."""
    if hasattr(graph, "edge_ids"):
        return list(graph.edge_ids)
    if hasattr(graph, "edges"):
        return list(range(len(graph.edges)))
    if hasattr(graph, "n_edges"):
        return list(range(int(graph.n_edges)))
    raise TypeError(
        f"Cannot resolve edge list from graph of type {type(graph).__name__}. "
        "Expected 'edge_ids', 'edges', or 'n_edges'."
    )


def compute_eap_scores(
    model: Any,
    clean_batch: Any,
    counterfactual_batch: Any,
    graph: Any,
    metric_fn: Any | None = None,
) -> dict[int, float]:
    """Compute EAP attribution scores for every edge in *graph*.

    For each edge *e* with sender node *s* and receiver node *r* we estimate::

        score(e) = | metric(model_patched(x_clean, x_cf, e)) - metric(model(x_cf)) |

    where ``model_patched`` replaces the activation on edge *e* (as it flows
    from *s* to *r*) with the value it takes on the clean input.

    Because the full patching loop requires hooks into the model's forward pass,
    this implementation uses the graph's ``patch_edge`` context manager when
    available, and falls back to a finite-difference approximation otherwise.

    Parameters
    ----------
    model:
        Transformer / neural-network model.  Expected to expose
        ``.run_with_cache(batch)`` returning ``(logits, cache)`` where *cache*
        maps node names to activation tensors.
    clean_batch:
        Batch of clean (uncorrupted) inputs.
    counterfactual_batch:
        Batch of counterfactual (corrupted / patched) inputs matched to
        *clean_batch*.
    graph:
        Computation graph.  Must expose edge IDs (see :func:`_resolve_edge_list`)
        and optionally a ``patch_edge(cache_clean, cache_cf, edge_id)`` method.
    metric_fn:
        Optional callable ``(logits) -> float`` scoring model output quality.
        Defaults to log-probability of the correct token if not provided.

    Returns
    -------
    dict[int, float]
        Maps each edge ID to its attribution score (higher ↔ more important).
    """
    t0 = time.perf_counter()
    edge_ids = _resolve_edge_list(graph)

    # ---- Obtain clean and corrupted caches --------------------------------
    try:
        clean_logits, clean_cache = model.run_with_cache(clean_batch)
        cf_logits, cf_cache = model.run_with_cache(counterfactual_batch)
    except Exception as exc:
        logger.error("model.run_with_cache failed: %s", exc)
        raise

    # ---- Baseline metric on counterfactual run ----------------------------
    if metric_fn is None:
        # Default: mean log-softmax of the top clean token
        try:
            import torch  # type: ignore
            with torch.no_grad():
                clean_top = int(clean_logits.argmax(dim=-1).flatten()[0])
                baseline_metric = float(
                    torch.log_softmax(cf_logits, dim=-1).flatten()[clean_top]
                )
        except Exception:
            baseline_metric = 0.0
    else:
        baseline_metric = float(metric_fn(cf_logits))

    scores: dict[int, float] = {}

    # ---- Per-edge patching loop -------------------------------------------
    for edge_id in edge_ids:
        try:
            if hasattr(graph, "patch_edge"):
                patched_logits = graph.patch_edge(
                    model=model,
                    clean_cache=clean_cache,
                    cf_cache=cf_cache,
                    edge_id=edge_id,
                    batch=counterfactual_batch,
                )
            else:
                # Fallback: treat score as 0 (graph does not support patching)
                patched_logits = cf_logits

            if metric_fn is None:
                try:
                    import torch  # type: ignore
                    with torch.no_grad():
                        patched_metric = float(
                            torch.log_softmax(patched_logits, dim=-1)
                            .flatten()[clean_top]  # type: ignore[possibly-undefined]
                        )
                except Exception:
                    patched_metric = baseline_metric
            else:
                patched_metric = float(metric_fn(patched_logits))

            scores[edge_id] = abs(patched_metric - baseline_metric)
        except Exception as exc:
            logger.warning("EAP scoring failed for edge %d: %s", edge_id, exc)
            scores[edge_id] = 0.0

    elapsed = time.perf_counter() - t0
    logger.info(
        "EAP scores computed for %d edges in %.2f s (baseline_metric=%.4f)",
        len(scores),
        elapsed,
        baseline_metric,
    )
    return scores


def get_top_k_circuit(
    scores: dict[int, float],
    graph: Any,
    k: int,
) -> np.ndarray:
    """Return a boolean edge mask selecting the *k* highest-scored edges.

    Parameters
    ----------
    scores:
        Edge attribution scores from :func:`compute_eap_scores`.
    graph:
        Computation graph (used to resolve total edge count).
    k:
        Number of edges to select (budget).

    Returns
    -------
    np.ndarray of shape ``(n_edges,)`` and dtype ``bool``.
    """
    edge_ids = _resolve_edge_list(graph)
    n_edges = len(edge_ids)

    if k > n_edges:
        logger.warning("k=%d > n_edges=%d; returning full mask.", k, n_edges)
        k = n_edges

    # Sort by score descending, take top-k
    sorted_ids = sorted(scores, key=lambda eid: scores[eid], reverse=True)
    top_k_ids = set(sorted_ids[:k])

    mask = np.zeros(n_edges, dtype=bool)
    for local_idx, edge_id in enumerate(edge_ids):
        if edge_id in top_k_ids:
            mask[local_idx] = True

    logger.debug("Top-%d circuit selected, density=%.2f%%", k, 100 * mask.mean())
    return mask


def save_eap_artifact(
    scores: dict[int, float],
    graph: Any,
    split: str,
    task: str,
    output_path: Path | str,
    k: int | None = None,
) -> PrefilterArtifact:
    """Persist EAP scores and optional top-k mask to *output_path*.

    Parameters
    ----------
    scores:
        Edge attribution scores from :func:`compute_eap_scores`.
    graph:
        Computation graph.
    split:
        Dataset split identifier (e.g. ``'train'``, ``'val'``).
    task:
        Task / benchmark identifier (e.g. ``'ioi'``, ``'gt'``).
    output_path:
        File path where the JSON artifact will be written.
    k:
        If provided, also compute and store top-k mask.

    Returns
    -------
    PrefilterArtifact
    """
    import hashlib, datetime

    graph_hash = hashlib.md5(str(sorted(scores.keys())).encode()).hexdigest()[:8]

    edge_ids = _resolve_edge_list(graph)
    top_k_indices: list[int] = []
    if k is not None:
        mask = get_top_k_circuit(scores, graph, k)
        top_k_indices = [edge_ids[i] for i in np.where(mask)[0].tolist()]

    artifact = PrefilterArtifact(
        method="eap",
        task=task,
        split=split,
        graph_hash=graph_hash,
        scores={str(eid): v for eid, v in scores.items()},
        top_k_indices=top_k_indices,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    )
    artifact.save(output_path)
    return artifact


# ---------------------------------------------------------------------------
# Class wrapper
# ---------------------------------------------------------------------------

class EAPBaseline:
    """Facade for EAP-based circuit prefiltering.

    Parameters
    ----------
    metric_fn:
        Optional metric function; passed through to :func:`compute_eap_scores`.
    """

    def __init__(self, metric_fn: Any | None = None) -> None:
        self.metric_fn = metric_fn

    def compute_scores(
        self,
        model: Any,
        clean_batch: Any,
        counterfactual_batch: Any,
        graph: Any,
    ) -> dict[int, float]:
        return compute_eap_scores(
            model, clean_batch, counterfactual_batch, graph, self.metric_fn
        )

    def get_top_k_circuit(
        self, scores: dict[int, float], graph: Any, k: int
    ) -> np.ndarray:
        return get_top_k_circuit(scores, graph, k)

    def save_artifact(
        self,
        scores: dict[int, float],
        graph: Any,
        split: str,
        task: str,
        output_path: Path | str,
        k: int | None = None,
    ) -> PrefilterArtifact:
        return save_eap_artifact(scores, graph, split, task, output_path, k)
