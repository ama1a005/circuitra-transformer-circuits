"""
Action masking utilities for discrete action spaces.

Masking prevents the policy from selecting infeasible actions by setting
their logits to a large negative value before the softmax / Categorical
distribution is constructed.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Categorical


class AllActionsMaskedError(ValueError):
    """Raised when an action mask has no valid (True) entries."""


def validate_action_mask(mask: Tensor) -> None:
    """
    Assert that *mask* contains at least one valid (True) action.

    Parameters
    ----------
    mask : Tensor
        Boolean tensor of shape (n_actions,) or (batch, n_actions).

    Raises
    ------
    AllActionsMaskedError
        If every action in any sample in the batch is masked out.
    """
    if mask.dim() == 1:
        if not mask.any():
            raise AllActionsMaskedError(
                "All actions are masked – at least one valid action is required."
            )
    else:
        # batch dimension
        any_valid = mask.any(dim=-1)  # (batch,)
        if not any_valid.all():
            bad_indices = (~any_valid).nonzero(as_tuple=False).flatten().tolist()
            raise AllActionsMaskedError(
                f"All actions masked for batch indices: {bad_indices}"
            )


def apply_action_mask(
    logits: Tensor,
    valid_mask: Tensor,
    fill_value: float = -1e9,
) -> Tensor:
    """
    Zero-out (fill) logits for invalid actions.

    Parameters
    ----------
    logits : Tensor
        Raw un-normalised logits, shape (n_actions,) or (batch, n_actions).
    valid_mask : Tensor
        Boolean tensor, same shape as *logits*.  True = action is valid.
    fill_value : float
        Value assigned to masked-out positions (default -1e9 ≈ -∞).

    Returns
    -------
    Tensor
        Masked logits, same shape as *logits*.

    Raises
    ------
    AllActionsMaskedError
        If every action is masked for any sample.
    """
    validate_action_mask(valid_mask)
    masked = logits.clone()
    masked[~valid_mask] = fill_value
    return masked


def masked_categorical(
    logits: Tensor,
    valid_mask: Tensor,
    fill_value: float = -1e9,
) -> Categorical:
    """
    Build a :class:`torch.distributions.Categorical` over only valid actions.

    Invalid actions receive logit ``fill_value`` so their probability is
    effectively zero after softmax.

    Parameters
    ----------
    logits : Tensor
        Shape (n_actions,) or (batch, n_actions).
    valid_mask : Tensor
        Boolean tensor, same shape as *logits*.
    fill_value : float
        Fill value for masked-out logits (default -1e9).

    Returns
    -------
    Categorical
        Distribution ready for ``.sample()``, ``.log_prob()``, ``.entropy()``.

    Raises
    ------
    AllActionsMaskedError
        If every action is masked for any sample.
    """
    masked_logits = apply_action_mask(logits, valid_mask, fill_value)
    return Categorical(logits=masked_logits)
