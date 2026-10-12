"""
hrl_circuits.models
===================
Model loading utilities and frozen-weight safety checks.
"""

from .loader import ModelLoader
from .frozen_guard import assert_no_trainable_model_weights, FrozenModelViolationError

__all__ = [
    "ModelLoader",
    "assert_no_trainable_model_weights",
    "FrozenModelViolationError",
]
