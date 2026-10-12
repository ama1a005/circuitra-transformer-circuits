"""
hrl_circuits.models.frozen_guard
==================================
Safety utilities to ensure that model weights are not accidentally made
trainable during HRL circuit-localisation experiments.

The HRL agent learns to *select* edges — it must never fine-tune the
underlying circuit model.  :func:`assert_no_trainable_model_weights`
is the enforcement point for this invariant and should be called
immediately after loading any model.

Supported frameworks
--------------------
* **PyTorch** (``torch.nn.Module``): checks ``param.requires_grad``
* **JAX / Flax** (no-op — JAX is functional and parameters are plain
  arrays without gradient state; pass ``framework='jax'`` to suppress
  the PyTorch import attempt)
* **Unknown** frameworks: raises a :exc:`FrozenModelViolationError` unless
  ``strict=False`` is passed.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class FrozenModelViolationError(RuntimeError):
    """Raised when a model has trainable (unfrozen) weights."""


# ---------------------------------------------------------------------------
# Main guard
# ---------------------------------------------------------------------------

def assert_no_trainable_model_weights(
    model: Any,
    *,
    framework: str = "auto",
    strict: bool = True,
) -> None:
    """Assert that *model* has no trainable weights.

    Parameters
    ----------
    model:
        The model object to inspect.
    framework:
        One of ``'auto'``, ``'pytorch'``, or ``'jax'``.
        In ``'auto'`` mode the function attempts to detect the framework
        from the model's type.
    strict:
        If ``True`` (default), raise :exc:`FrozenModelViolationError`
        when the framework is unrecognised.  If ``False``, emit a
        ``UserWarning`` instead.

    Raises
    ------
    FrozenModelViolationError
        If any parameter has ``requires_grad = True`` (PyTorch), or if
        the framework is unknown and ``strict=True``.
    """
    resolved = _resolve_framework(model, framework)

    if resolved == "pytorch":
        _assert_frozen_pytorch(model)
    elif resolved == "jax":
        # JAX is functional — no per-parameter gradient state to check.
        # The caller is responsible for not passing parameters to an
        # optimiser.
        return
    else:
        _handle_unknown_framework(model, strict)


# ---------------------------------------------------------------------------
# Framework-specific helpers
# ---------------------------------------------------------------------------

def _resolve_framework(model: Any, hint: str) -> str:
    """Resolve the framework from *hint* or by inspecting *model*."""
    if hint in ("pytorch", "jax"):
        return hint
    if hint != "auto":
        return "unknown"

    # Auto-detect from type hierarchy
    try:
        import torch.nn as nn  # type: ignore
        if isinstance(model, nn.Module):
            return "pytorch"
    except ImportError:
        pass

    # Flax / Haiku / Equinox — no standard base class; check for .params
    type_name = type(model).__module__ or ""
    if type_name.startswith("flax") or type_name.startswith("haiku"):
        return "jax"

    return "unknown"


def _assert_frozen_pytorch(model: Any) -> None:
    """Check that no PyTorch parameter has ``requires_grad == True``.

    Raises
    ------
    FrozenModelViolationError
        With a list of the first up to 10 offending parameter names.
    """
    try:
        import torch.nn as nn  # type: ignore
    except ImportError:
        raise FrozenModelViolationError(
            "PyTorch framework was specified but 'torch' is not installed."
        )

    if not isinstance(model, nn.Module):
        raise FrozenModelViolationError(
            f"Expected a torch.nn.Module, got {type(model).__name__!r}."
        )

    trainable: list[str] = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
    ]

    if trainable:
        examples = trainable[:10]
        suffix = f" ... and {len(trainable) - 10} more" if len(trainable) > 10 else ""
        raise FrozenModelViolationError(
            f"Model has {len(trainable)} trainable parameter(s).  "
            f"Call model.eval() and model.requires_grad_(False) before use.  "
            f"First offenders: {examples}{suffix}"
        )


def _handle_unknown_framework(model: Any, strict: bool) -> None:
    """Warn or raise for unrecognised frameworks."""
    message = (
        f"Cannot verify frozen weights for model of type "
        f"{type(model).__name__!r}: framework not recognised.  "
        "Pass framework='pytorch' or framework='jax' explicitly."
    )
    if strict:
        raise FrozenModelViolationError(message)
    else:
        import warnings
        warnings.warn(message, stacklevel=4)
