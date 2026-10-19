"""
hrl_circuits.benchmark.split_guard
====================================
Safety guards that prevent data leakage between splits during tuning,
training, and evaluation.

Design
------
Three principal protection rules:

1. **No tuning on public_test** — hyperparameter search must not touch
   the held-out test split.
2. **No training on validation or public_test** — the RL agent may only
   be trained on the ``'train'`` split.
3. **Frozen config required** — before running a public-test evaluation
   the configuration must be declared frozen (to prevent silent
   post-hoc changes).

All violations raise :exc:`TuningSplitViolationError`, a
:exc:`RuntimeError` subclass, to fail loudly and early.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class TuningSplitViolationError(RuntimeError):
    """Raised when a forbidden split operation is attempted."""

# Alias used by tests
SplitGuardError = TuningSplitViolationError


# ---------------------------------------------------------------------------
# Module-level convenience functions (delegates to SplitGuard statics)
# ---------------------------------------------------------------------------

def _is_test_split(split: str) -> bool:
    """Return True if *split* is any test/holdout variant."""
    s = split.lower()
    return (
        s == "public_test"
        or s == "test"
        or s == "private_test"
        or s == "holdout"
        or s.endswith("_test")
        or s.startswith("test")
    )


def _is_validation_like(split: str) -> bool:
    """Return True if *split* is any validation/dev variant (not train)."""
    s = split.lower()
    return s.startswith("val") or s.startswith("dev")


def assert_can_tune(split: str) -> None:
    """Raise SplitGuardError if *split* is any test-like split."""
    if _is_test_split(split):
        raise TuningSplitViolationError(
            f"Hyperparameter tuning on split={split!r} is forbidden. "
            "Use split='validation' or split='train' for tuning."
        )


def assert_can_train(split: str) -> None:
    """Raise SplitGuardError if *split* is not a training split."""
    if _is_test_split(split) or _is_validation_like(split):
        raise TuningSplitViolationError(
            f"Agent training on split={split!r} is forbidden. "
            "Training must use split='train' (or 'train_fold_*') only."
        )



# ---------------------------------------------------------------------------
# Guard class
# ---------------------------------------------------------------------------

class SplitGuard:
    """Namespace of static split-safety assertions.

    All methods are static — the class is never instantiated.  Import and
    call directly::

        SplitGuard.assert_can_tune("train")        # OK
        SplitGuard.assert_can_tune("public_test")  # raises
    """

    # ------------------------------------------------------------------
    # Tuning guard
    # ------------------------------------------------------------------

    @staticmethod
    def assert_can_tune(split: str) -> None:
        """Assert that hyperparameter tuning is permissible on *split*.

        Tuning is forbidden on the ``'public_test'`` split to prevent
        indirect data leakage via metric-guided search.

        Parameters
        ----------
        split:
            The split name to validate.

        Raises
        ------
        TuningSplitViolationError
            If ``split == 'public_test'``.
        """
        if split == "public_test":
            raise TuningSplitViolationError(
                "Hyperparameter tuning on split='public_test' is forbidden. "
                "Use split='validation' or split='train' for tuning. "
                "The public_test split is reserved for final, frozen "
                "evaluation only."
            )

    # ------------------------------------------------------------------
    # Training guard
    # ------------------------------------------------------------------

    @staticmethod
    def assert_can_train(split: str) -> None:
        """Assert that agent training is permissible on *split*.

        Training the RL agent is permitted only on ``'train'``.  Both
        ``'validation'`` and ``'public_test'`` are off-limits.

        Parameters
        ----------
        split:
            The split name to validate.

        Raises
        ------
        TuningSplitViolationError
            If ``split`` is ``'public_test'`` or ``'validation'``.
        """
        forbidden = {"public_test", "validation"}
        if split in forbidden:
            raise TuningSplitViolationError(
                f"Agent training on split={split!r} is forbidden. "
                "Training must use split='train' only. "
                f"Forbidden splits: {sorted(forbidden)}."
            )

    # ------------------------------------------------------------------
    # Config freeze guard
    # ------------------------------------------------------------------

    @staticmethod
    def require_frozen_config(config: Any) -> None:
        """Assert that *config* is marked as frozen.

        A config is considered frozen if it has a truthy ``frozen``
        attribute **or** a truthy ``is_frozen`` attribute.  If neither
        exists, a :exc:`TuningSplitViolationError` is raised because we
        cannot confirm immutability.

        Parameters
        ----------
        config:
            Any configuration object (dataclass, dict, SimpleNamespace, …).

        Raises
        ------
        TuningSplitViolationError
            If the config is not marked as frozen.

        Notes
        -----
        For dict-like configs, the ``'frozen'`` key is checked first.
        For dataclass / namespace configs, ``config.frozen`` is checked
        first, then ``config.is_frozen``.
        """
        # dict-like
        if isinstance(config, dict):
            if config.get("frozen") or config.get("is_frozen"):
                return
            raise TuningSplitViolationError(
                "Configuration must be frozen before running public_test "
                "evaluation.  Set config['frozen'] = True."
            )

        # attribute-based (dataclass, SimpleNamespace, etc.)
        for attr in ("frozen", "is_frozen"):
            if hasattr(config, attr) and getattr(config, attr):
                return

        raise TuningSplitViolationError(
            "Configuration must be frozen before running public_test "
            "evaluation.  Set config.frozen = True (or config.is_frozen = True)."
        )
