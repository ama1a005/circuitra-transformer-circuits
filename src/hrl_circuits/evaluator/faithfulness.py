"""
hrl_circuits.evaluator.faithfulness
====================================
Faithfulness computation utilities.

*Faithfulness* measures how well the sub-circuit identified by a mask
reproduces the full model's behaviour on a task.  Two metrics are
supported:

``kl``
    KL-divergence between the full-model output distribution and the
    patched output distribution.  Lower → more faithful.  This raw value
    is then mapped to ``[0, 1]`` by :func:`normalize_faithfulness`.

``mse``
    Mean-squared error between clean and patched logits (or any numeric
    vector outputs).
"""

from __future__ import annotations

import math
import numpy as np
from typing import Literal


MetricName = Literal["kl", "mse"]


def compute_faithfulness_from_outputs(
    clean_out: np.ndarray,
    patched_out: np.ndarray,
    metric: MetricName = "kl",
) -> float:
    """Compute a raw faithfulness score from model outputs.

    For ``metric='kl'``, both arrays are interpreted as **logits** and
    converted to probability distributions via softmax before computing
    the forward KL divergence ``KL(clean || patched)``.

    For ``metric='mse'``, the mean-squared error over all elements is
    returned.

    Parameters
    ----------
    clean_out:
        Output of the full (unpatched) model.  Shape ``(vocab_size,)`` or
        ``(batch, vocab_size)`` — batched inputs are averaged.
    patched_out:
        Output of the patched (masked) model.  Must have the same shape as
        *clean_out*.
    metric:
        ``'kl'`` (default) or ``'mse'``.

    Returns
    -------
    float
        Raw divergence / error value.  For ``'kl'`` this is in nats (≥ 0).
        For ``'mse'`` this is ≥ 0.

    Raises
    ------
    ValueError
        If shapes are incompatible or *metric* is unknown.
    """
    clean_out = np.asarray(clean_out, dtype=np.float64)
    patched_out = np.asarray(patched_out, dtype=np.float64)

    if clean_out.shape != patched_out.shape:
        raise ValueError(
            f"clean_out shape {clean_out.shape} != patched_out shape "
            f"{patched_out.shape}."
        )

    if metric == "kl":
        return _kl_divergence(clean_out, patched_out)
    elif metric == "mse":
        return float(np.mean((clean_out - patched_out) ** 2))
    else:
        raise ValueError(
            f"Unknown metric {metric!r}.  Supported: 'kl', 'mse'."
        )


def normalize_faithfulness(raw: float) -> float:
    """Map a raw faithfulness score to ``[0.0, 1.0]``.

    For KL-divergence the mapping is::

        faithfulness = exp(-raw)

    This gives 1.0 when ``raw == 0`` (perfect faithfulness) and
    approaches 0.0 as divergence grows.  The result is then clipped to
    ``[0, 1]`` to handle any floating-point edge cases.

    For a pre-normalised score (e.g. already in ``[0, 1]``), the same
    formula is applied but the clip ensures the contract is always met.

    Parameters
    ----------
    raw:
        Non-negative raw divergence / error value.

    Returns
    -------
    float
        Normalised faithfulness in ``[0.0, 1.0]``.
    """
    if raw < 0.0:
        # Negative divergence is physically impossible but can arise from
        # numerical errors; clamp to 0 before mapping.
        raw = 0.0
    normalised = math.exp(-raw)
    return float(np.clip(normalised, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    x = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def _kl_divergence(clean: np.ndarray, patched: np.ndarray) -> float:
    """Forward KL divergence KL(P_clean || P_patched) in nats.

    Inputs are logits; softmax is applied internally.  Batched inputs
    (shape ``(batch, vocab)``) are averaged over the batch dimension.
    """
    # Flatten to 2-D (batch, vocab) if needed
    if clean.ndim == 1:
        clean = clean[np.newaxis, :]
        patched = patched[np.newaxis, :]

    p = _softmax(clean)    # shape (batch, vocab)
    q = _softmax(patched)  # shape (batch, vocab)

    # KL per example: sum_i p_i * log(p_i / q_i)
    # Add epsilon for numerical stability
    eps = 1e-10
    kl_per_example = np.sum(p * np.log((p + eps) / (q + eps)), axis=-1)  # (batch,)
    return float(np.mean(kl_per_example))
