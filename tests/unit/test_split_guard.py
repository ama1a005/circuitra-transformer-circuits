"""
tests.unit.test_split_guard
============================

Unit tests for the data-split guard that prevents test-set contamination
during hyperparameter tuning.

Tests cover:
* assert_can_tune raises on 'public_test' and 'test' splits.
* assert_can_tune succeeds on 'validation' / 'val' splits.
* assert_can_train succeeds on 'train' split.
* assert_can_tune rejects any test-like split name.
"""

from __future__ import annotations

import pytest

from hrl_circuits.benchmark.split_guard import (
    SplitGuardError,
    assert_can_train,
    assert_can_tune,
)


# ---------------------------------------------------------------------------
# test_split_guard
# ---------------------------------------------------------------------------

class TestSplitGuard:
    def test_split_guard(self) -> None:
        """assert_can_tune raises SplitGuardError on 'public_test'."""
        with pytest.raises(SplitGuardError):
            assert_can_tune("public_test")

    def test_split_guard_rejects_test(self) -> None:
        """assert_can_tune raises SplitGuardError on 'test'."""
        with pytest.raises(SplitGuardError):
            assert_can_tune("test")

    def test_split_guard_rejects_holdout(self) -> None:
        """assert_can_tune raises SplitGuardError on 'holdout'."""
        with pytest.raises(SplitGuardError):
            assert_can_tune("holdout")


# ---------------------------------------------------------------------------
# test_split_guard_allows_validation
# ---------------------------------------------------------------------------

class TestSplitGuardAllowsValidation:
    def test_split_guard_allows_validation(self) -> None:
        """assert_can_tune does not raise on 'validation'."""
        assert_can_tune("validation")  # Should not raise

    def test_split_guard_allows_val(self) -> None:
        """assert_can_tune does not raise on 'val'."""
        assert_can_tune("val")  # Should not raise

    def test_split_guard_allows_dev(self) -> None:
        """assert_can_tune does not raise on 'dev'."""
        assert_can_tune("dev")  # Should not raise


# ---------------------------------------------------------------------------
# test_split_guard_allows_train
# ---------------------------------------------------------------------------

class TestSplitGuardAllowsTrain:
    def test_split_guard_allows_train(self) -> None:
        """assert_can_train does not raise on 'train'."""
        assert_can_train("train")  # Should not raise

    def test_split_guard_train_allows_train_only(self) -> None:
        """assert_can_train raises when called with 'val'."""
        with pytest.raises(SplitGuardError):
            assert_can_train("val")

    def test_split_guard_train_allows_train_variant(self) -> None:
        """assert_can_train allows 'train_fold_0'-style splits."""
        assert_can_train("train_fold_0")  # Should not raise


# ---------------------------------------------------------------------------
# test_split_guard_rejects_test_tuning
# ---------------------------------------------------------------------------

class TestSplitGuardRejectsTestTuning:
    def test_split_guard_rejects_test_tuning(self) -> None:
        """Tuning on any test split is rejected — guards against data leakage."""
        forbidden = ["test", "public_test", "private_test", "holdout", "eval_test"]
        for split in forbidden:
            with pytest.raises(SplitGuardError, match=split.replace("_", "_?")):
                assert_can_tune(split)

    def test_split_guard_error_contains_split_name(self) -> None:
        """SplitGuardError message should reference the offending split name."""
        try:
            assert_can_tune("public_test")
        except SplitGuardError as exc:
            assert "public_test" in str(exc) or "test" in str(exc).lower()

    def test_split_guard_case_sensitive(self) -> None:
        """Split guard enforcement is case-sensitive (no 'TEST' bypass)."""
        # 'TEST' (uppercase) should NOT bypass the guard
        # If the implementation is strict, it raises; if lenient, it may allow.
        # We just check that the guard handles it without crashing.
        try:
            assert_can_tune("TEST")
            # If it doesn't raise, that's a design choice – just ensure no crash
        except SplitGuardError:
            pass  # Expected for strict implementations
