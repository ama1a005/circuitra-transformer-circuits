"""
hrl_circuits.seed
=================

Deterministic seeding utilities for reproducible experiments.

A single call to :func:`seed_everything` sets the random seed for Python's
``random`` module, NumPy, PyTorch (both CPU and all CUDA devices), and
optionally enables deterministic CUDA algorithms.

Usage
-----
::

    from hrl_circuits.seed import seed_everything

    seed_everything(42)                        # deterministic CUDA on
    seed_everything(42, deterministic=False)   # faster, non-deterministic CUDA
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set all random seeds for fully reproducible experiment execution.

    Covers:
    * Python built-in ``random`` module
    * NumPy global RNG
    * PyTorch CPU RNG
    * PyTorch CUDA RNG (all visible devices)
    * ``PYTHONHASHSEED`` environment variable
    * ``torch.backends.cudnn.deterministic`` / ``benchmark`` flags
    * ``torch.use_deterministic_algorithms`` (when ``deterministic=True``)

    Parameters
    ----------
    seed : int
        The integer seed value.  Must be in the range ``[0, 2**32 - 1]``
        for compatibility with NumPy.
    deterministic : bool, optional
        If ``True`` (default), configure PyTorch for fully deterministic
        operation.  This may reduce performance on some GPU operations.
        Set to ``False`` when speed is preferred over strict bit-exact
        reproducibility.

    Raises
    ------
    ValueError
        If ``seed`` is outside the valid range ``[0, 2**32 - 1]``.

    Examples
    --------
    >>> seed_everything(42)
    >>> seed_everything(0, deterministic=False)

    Notes
    -----
    Even with all seeds fixed and ``deterministic=True``, some operations
    (e.g., ``atomicAdd`` in CUDA) may produce non-deterministic results on
    certain hardware/driver combinations. See PyTorch reproducibility docs
    for details.
    """
    if not (0 <= seed <= 2**32 - 1):
        raise ValueError(
            f"seed must be in [0, 2**32 - 1], got {seed}."
        )

    # ── Python hash seed ──────────────────────────────────────────────────────
    os.environ["PYTHONHASHSEED"] = str(seed)

    # ── Python random ─────────────────────────────────────────────────────────
    random.seed(seed)

    # ── NumPy ─────────────────────────────────────────────────────────────────
    np.random.seed(seed)

    # ── PyTorch CPU ───────────────────────────────────────────────────────────
    torch.manual_seed(seed)

    # ── PyTorch CUDA (all devices) ────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ── cuDNN / determinism flags ─────────────────────────────────────────────
    if deterministic:
        # Disable cuDNN auto-tuner — ensures deterministic convolutions.
        torch.backends.cudnn.benchmark = False   # type: ignore[attr-defined]
        # Force deterministic cuDNN algorithms (raises error if unavailable).
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        # Raise an error if a non-deterministic PyTorch op is called.
        # ``warn_only=True`` turns errors into warnings if strict mode breaks
        # too many ops (useful during prototyping).
        torch.use_deterministic_algorithms(True, warn_only=False)
        # Required for some deterministic scatter/gather operations on CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True    # type: ignore[attr-defined]
        torch.backends.cudnn.deterministic = False  # type: ignore[attr-defined]
        torch.use_deterministic_algorithms(False)

    logger.info(
        "seed_everything: seed=%d  deterministic=%s  "
        "CUDA available=%s  CUDA devices=%d",
        seed,
        deterministic,
        torch.cuda.is_available(),
        torch.cuda.device_count() if torch.cuda.is_available() else 0,
    )


__all__: list[str] = ["seed_everything"]
