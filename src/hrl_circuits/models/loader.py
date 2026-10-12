"""
hrl_circuits.models.loader
============================
``ModelLoader`` — the single entry point for loading circuit-analysis
models in the HRL pipeline.

All loaded models are frozen by default: the HRL agent searches over the
*edge mask*, it never updates model weights.
:func:`~hrl_circuits.models.frozen_guard.assert_no_trainable_model_weights`
is called unconditionally when ``frozen=True`` (the default).

Supported backends
------------------
* **HuggingFace Transformers** — any ``AutoModelForCausalLM``-compatible
  model.
* **TransformerLens** — wraps the HuggingFace model for easier hook-based
  activation access.
* **Local path** — any model ID that is a valid local directory path is
  loaded with ``from_pretrained``.

If neither ``transformers`` nor ``transformer_lens`` is installed, the
loader raises a helpful error rather than an opaque ``ImportError``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .frozen_guard import assert_no_trainable_model_weights, FrozenModelViolationError

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load transformer models in a frozen-weight-safe manner.

    Parameters
    ----------
    default_device:
        Default device for all ``load_*`` calls (``'cpu'``, ``'cuda'``,
        etc.).  Can be overridden per call.

    Examples
    --------
    .. code-block:: python

        loader = ModelLoader(default_device="cuda")
        model = loader.load_frozen_model("gpt2")
    """

    def __init__(self, default_device: str = "cpu") -> None:
        self._default_device = default_device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_frozen_model(
        self,
        model_id: str,
        *,
        device: str | None = None,
        backend: str = "transformers",
        dtype: str = "float32",
        extra_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Load *model_id* with all weights frozen.

        Parameters
        ----------
        model_id:
            HuggingFace model ID (e.g. ``"gpt2"``) or a local directory
            path.
        device:
            Target device.  Defaults to ``self.default_device``.
        backend:
            ``'transformers'`` (default) or ``'transformer_lens'``.
        dtype:
            Floating-point precision: ``'float32'``, ``'float16'``, or
            ``'bfloat16'``.  Only used when *backend* is ``'transformers'``.
        extra_kwargs:
            Additional keyword arguments forwarded to the underlying
            ``from_pretrained`` call.

        Returns
        -------
        Any
            The loaded model.

        Raises
        ------
        FrozenModelViolationError
            If the model has trainable weights after loading (should not
            happen if ``frozen=True`` and the backend is correctly
            implemented; raised as a safety net).
        RuntimeError
            If the requested backend is not installed.
        ValueError
            If *backend* or *dtype* is unrecognised.
        """
        device = device or self._default_device
        extra_kwargs = extra_kwargs or {}

        if backend == "transformers":
            model = self._load_transformers(
                model_id, device=device, dtype=dtype, extra_kwargs=extra_kwargs
            )
        elif backend == "transformer_lens":
            model = self._load_transformer_lens(
                model_id, device=device, extra_kwargs=extra_kwargs
            )
        else:
            raise ValueError(
                f"Unknown backend {backend!r}.  "
                "Supported: 'transformers', 'transformer_lens'."
            )

        # Freeze weights and verify
        model = _freeze(model, backend=backend)
        assert_no_trainable_model_weights(model)
        logger.info("Loaded frozen model %r on device=%s via %s.", model_id, device, backend)
        return model

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _load_transformers(
        model_id: str,
        device: str,
        dtype: str,
        extra_kwargs: dict[str, Any],
    ) -> Any:
        """Load via HuggingFace Transformers ``AutoModelForCausalLM``."""
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' and 'torch' packages are required "
                "for backend='transformers'.  Install with: "
                "pip install transformers torch"
            ) from exc

        torch_dtype = _resolve_torch_dtype(torch, dtype)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            **extra_kwargs,
        )
        model = model.to(device)
        model.eval()
        return model

    @staticmethod
    def _load_transformer_lens(
        model_id: str,
        device: str,
        extra_kwargs: dict[str, Any],
    ) -> Any:
        """Load via TransformerLens ``HookedTransformer``."""
        try:
            from transformer_lens import HookedTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformer_lens' package is required for "
                "backend='transformer_lens'.  Install with: "
                "pip install transformer_lens"
            ) from exc

        model = HookedTransformer.from_pretrained(
            model_id,
            **extra_kwargs,
        )
        model = model.to(device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def default_device(self) -> str:
        """The default device used when *device* is not specified."""
        return self._default_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _freeze(model: Any, backend: str) -> Any:
    """Freeze all parameters of *model* in-place.

    Returns the model (may be the same object or a new wrapper).
    """
    # PyTorch-style freeze
    try:
        model.requires_grad_(False)
    except AttributeError:
        # Not a PyTorch module — no-op for other frameworks
        pass
    return model


def _resolve_torch_dtype(torch: Any, dtype: str) -> Any:
    """Map a dtype string to a ``torch.dtype``."""
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(
            f"Unknown dtype {dtype!r}.  Supported: {sorted(mapping)}."
        )
    return mapping[dtype]
