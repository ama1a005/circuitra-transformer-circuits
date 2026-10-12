"""
hrl_circuits.evaluator.cache
============================
``ActivationCache`` — keyed storage for clean and counterfactual
activations from model forward passes.

Design
------
A single *cache key* encodes all factors that determine uniqueness:
benchmark revision, model ID, task, split, example IDs, graph hash, and
config hash.  Any mismatch between the key used to *store* activations
and the key used to *retrieve* them results in a cache miss (or raises
a :exc:`CacheKeyMismatchError` when reading an existing slot with a
different key).

Storage is in-memory by default (``backend='memory'``).  Persistence to
disk is available (``backend='disk'``, ``cache_dir`` required) and stores
each activation array as a ``.npy`` file with a JSON sidecar for key
metadata.

Invariants
----------
* The same *cache_key* always refers to the same activation tensors.
* Storing to an existing slot with a *different* key raises
  :exc:`CacheKeyMismatchError`.
* Storing to an existing slot with the *same* key is a silent no-op
  (first-write-wins).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CacheKeyMismatchError(ValueError):
    """Raised when a slot already exists under a different key."""


# ---------------------------------------------------------------------------
# Key computation
# ---------------------------------------------------------------------------

def cache_key(
    benchmark_revision: str,
    model_id: str,
    task: str,
    split: str,
    example_ids: list[str | int],
    graph_hash: str,
    config_hash: str,
) -> str:
    """Compute a deterministic cache key as a SHA-256 hex digest.

    Parameters
    ----------
    benchmark_revision:
        Git tag / commit of the benchmark.
    model_id:
        Hugging Face model ID or local path.
    task:
        Task name (e.g. ``"ioi"``).
    split:
        Data split (e.g. ``"train"``).
    example_ids:
        Ordered list of example identifiers in the batch.  Order matters
        (same examples in different order → different key).
    graph_hash:
        SHA-256 of the candidate graph.
    config_hash:
        SHA-256 of the evaluation config.

    Returns
    -------
    str
        64-character lowercase hex string (SHA-256).
    """
    payload = json.dumps(
        {
            "benchmark_revision": benchmark_revision,
            "model_id": model_id,
            "task": task,
            "split": split,
            "example_ids": [str(eid) for eid in example_ids],
            "graph_hash": graph_hash,
            "config_hash": config_hash,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Cache implementation
# ---------------------------------------------------------------------------

class ActivationCache:
    """Keyed storage for clean and counterfactual activations.

    Parameters
    ----------
    backend:
        ``'memory'`` (default) — stores everything in process memory.
        ``'disk'`` — persists ``.npy`` arrays; requires *cache_dir*.
    cache_dir:
        Root directory for disk-backed storage.  Ignored when
        ``backend='memory'``.

    Examples
    --------
    .. code-block:: python

        key = cache_key(benchmark_revision, model_id, task, split,
                        example_ids, graph_hash, config_hash)
        c = ActivationCache()
        c.set_clean_activations(key, acts)
        acts2 = c.get_clean_activations(key)  # returns acts
    """

    def __init__(
        self,
        backend: str = "memory",
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        if backend not in ("memory", "disk"):
            raise ValueError(
                f"backend must be 'memory' or 'disk', got {backend!r}."
            )
        if backend == "disk" and cache_dir is None:
            raise ValueError("cache_dir must be provided when backend='disk'.")

        self._backend = backend
        self._cache_dir: Optional[Path] = (
            Path(cache_dir) if cache_dir is not None else None
        )

        # In-memory stores: key → array
        self._clean_store: dict[str, np.ndarray] = {}
        self._cf_store: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Clean activations
    # ------------------------------------------------------------------

    def get_clean_activations(self, key: str) -> Optional[np.ndarray]:
        """Return clean activations for *key*, or ``None`` on miss.

        Parameters
        ----------
        key:
            Cache key from :func:`cache_key`.

        Returns
        -------
        np.ndarray or None
        """
        if self._backend == "memory":
            return self._clean_store.get(key)
        return self._disk_load(key, "clean")

    def set_clean_activations(self, key: str, activations: np.ndarray) -> None:
        """Store *activations* under *key* (clean pass).

        A second write with the same key is silently ignored.  A write
        with a different array but the same key raises nothing — first
        write wins.

        Parameters
        ----------
        key:
            Cache key from :func:`cache_key`.
        activations:
            Numpy array of activations.
        """
        activations = np.asarray(activations)
        if self._backend == "memory":
            if key not in self._clean_store:
                self._clean_store[key] = activations
        else:
            self._disk_save(key, "clean", activations)

    # ------------------------------------------------------------------
    # Counterfactual activations
    # ------------------------------------------------------------------

    def get_counterfactual_activations(self, key: str) -> Optional[np.ndarray]:
        """Return counterfactual activations for *key*, or ``None`` on miss.

        Parameters
        ----------
        key:
            Cache key from :func:`cache_key`.

        Returns
        -------
        np.ndarray or None
        """
        if self._backend == "memory":
            return self._cf_store.get(key)
        return self._disk_load(key, "counterfactual")

    def set_counterfactual_activations(
        self, key: str, activations: np.ndarray
    ) -> None:
        """Store *activations* under *key* (counterfactual pass).

        Parameters
        ----------
        key:
            Cache key from :func:`cache_key`.
        activations:
            Numpy array of counterfactual activations.
        """
        activations = np.asarray(activations)
        if self._backend == "memory":
            if key not in self._cf_store:
                self._cf_store[key] = activations
        else:
            self._disk_save(key, "counterfactual", activations)

    def invalidate(self, key: str) -> None:
        """Remove *key* from both clean and counterfactual stores.

        No-op if *key* is not present.

        Parameters
        ----------
        key:
            Cache key to invalidate.
        """
        self._clean_store.pop(key, None)
        self._cf_store.pop(key, None)
        if self._backend == "disk" and self._cache_dir is not None:
            for suffix in ("clean", "counterfactual"):
                npy = self._cache_dir / f"{key}_{suffix}.npy"
                meta = self._cache_dir / f"{key}_{suffix}.json"
                npy.unlink(missing_ok=True)
                meta.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Disk helpers
    # ------------------------------------------------------------------

    def _disk_save(
        self, key: str, kind: str, activations: np.ndarray
    ) -> None:
        assert self._cache_dir is not None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        npy_path = self._cache_dir / f"{key}_{kind}.npy"
        if not npy_path.exists():
            np.save(npy_path, activations)
            meta_path = self._cache_dir / f"{key}_{kind}.json"
            meta_path.write_text(
                json.dumps({"key": key, "kind": kind}), encoding="utf-8"
            )

    def _disk_load(
        self, key: str, kind: str
    ) -> Optional[np.ndarray]:
        assert self._cache_dir is not None
        npy_path = self._cache_dir / f"{key}_{kind}.npy"
        if not npy_path.exists():
            return None
        # Validate key from sidecar
        meta_path = self._cache_dir / f"{key}_{kind}.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("key") != key:
                raise CacheKeyMismatchError(
                    f"Cache file {npy_path} was written under key "
                    f"{meta.get('key')!r} but requested with key {key!r}."
                )
        return np.load(npy_path)
