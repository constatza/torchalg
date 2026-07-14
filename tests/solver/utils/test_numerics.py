"""Tests for ``torchalg.utils.numerics``.

Focused translation-correctness checks for the dependency floor of
``torchalg`` — everything else in the migration builds on these
functions being right.
"""

from __future__ import annotations

import math

import torch

from torchalg.utils.numerics import (
    check_breakdown,
    compute_curvature,
    stable_dot_product,
)


class TestStableDotProduct:
    """Tests for ``stable_dot_product``."""

    def test_matches_plain_dot_for_ordinary_vectors(
        self,
        orthogonal_dot_vectors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Ordinary-magnitude vectors match ``torch.dot`` exactly."""
        a, b = orthogonal_dot_vectors
        assert stable_dot_product(a, b) == float(torch.dot(a, b))

    def test_prevents_overflow_for_extreme_magnitudes(
        self,
        overflow_cancelling_vectors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Balanced scaling recovers the exact answer where naive dot overflows to nan."""
        a, b = overflow_cancelling_vectors

        naive = torch.dot(a, b)
        assert math.isnan(float(naive)), "fixture should overflow under naive computation"

        result = stable_dot_product(a, b)
        assert math.isfinite(result)
        assert result == 0.0

    def test_returns_python_float(
        self,
        orthogonal_dot_vectors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Return type is a plain float, matching the reference's interface."""
        a, b = orthogonal_dot_vectors
        assert isinstance(stable_dot_product(a, b), float)


class TestComputeCurvature:
    """Tests for ``compute_curvature``."""

    def test_positive_curvature_no_breakdown(
        self,
        positive_curvature_pair: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Well-conditioned SPD-consistent pair reports no breakdown."""
        p, q = positive_curvature_pair
        curvature, breakdown = compute_curvature(p, q)
        assert curvature == float(torch.dot(p, q))
        assert breakdown is False

    def test_negative_curvature_triggers_breakdown(
        self,
        negative_curvature_pair: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Negative curvature (non-SPD signal) triggers breakdown."""
        _, breakdown = compute_curvature(*negative_curvature_pair)
        assert breakdown is True

    def test_nonfinite_dot_triggers_breakdown(
        self,
        nonfinite_curvature_pair: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """A ``nan`` in the matvec product triggers breakdown."""
        curvature, breakdown = compute_curvature(*nonfinite_curvature_pair)
        assert math.isnan(curvature)
        assert breakdown is True

    def test_tiny_curvature_below_relative_threshold_triggers_breakdown(
        self,
        torch_dtype: torch.dtype,
    ) -> None:
        """Curvature far below ``breakdown_tol * ||p||^2`` triggers breakdown."""
        p = torch.tensor([1.0, 2.0, 3.0], dtype=torch_dtype)
        # p . q chosen tiny relative to ||p||^2 = 14, but not below the
        # absolute eps**2 floor, to isolate the *relative* threshold branch.
        q = torch.tensor([1e-13, 0.0, 0.0], dtype=torch_dtype)
        _, breakdown = compute_curvature(p, q, breakdown_tol=1e-2)
        assert breakdown is True


class TestCheckBreakdown:
    """Tests for ``check_breakdown``."""

    def test_normal_value_is_not_breakdown(self) -> None:
        """An ordinary-magnitude value is not flagged as breakdown."""
        assert check_breakdown(1.0) is False

    def test_tiny_value_is_breakdown(self) -> None:
        """A value below the default threshold is flagged as breakdown."""
        assert check_breakdown(1e-20) is True

    def test_nan_is_breakdown(self) -> None:
        """``nan`` is always flagged as breakdown."""
        assert check_breakdown(float("nan")) is True

    def test_inf_is_breakdown(self) -> None:
        """``inf`` is always flagged as breakdown."""
        assert check_breakdown(float("inf")) is True

    def test_custom_threshold_is_respected(self) -> None:
        """A custom threshold changes the breakdown boundary."""
        assert check_breakdown(0.5, threshold=1.0) is True
        assert check_breakdown(0.5, threshold=0.1) is False
