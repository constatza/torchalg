"""Tests for ``torchalg.utils.numerics``.

Focused translation-correctness checks for the dependency floor of
``torchalg`` — everything else in the migration builds on these
functions being right.
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch

from torchalg.utils.numerics import stable_dot_product


class TestStableDotProduct:
    """Tests for ``stable_dot_product``."""

    def test_matches_plain_dot_for_ordinary_vectors(
        self,
        orthogonal_dot_vectors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Ordinary-magnitude vectors match ``torch.dot`` exactly."""
        a, b = orthogonal_dot_vectors
        assert stable_dot_product(a, b).item() == float(torch.dot(a, b))

    def test_prevents_overflow_for_extreme_magnitudes(
        self,
        overflow_cancelling_vectors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Balanced scaling recovers the exact answer where naive dot overflows to nan."""
        a, b = overflow_cancelling_vectors

        naive = torch.dot(a, b)
        assert math.isnan(float(naive)), "fixture should overflow under naive computation"

        result = stable_dot_product(a, b)
        assert math.isfinite(result.item())
        assert result.item() == 0.0

    def test_returns_zero_dim_tensor(
        self,
        orthogonal_dot_vectors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Return type is a 0-d tensor."""
        a, b = orthogonal_dot_vectors
        result = stable_dot_product(a, b)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0

    def test_never_calls_tensor_bool(
        self,
        orthogonal_dot_vectors: tuple[torch.Tensor, torch.Tensor],
        overflow_cancelling_vectors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """torch.where implementation never forces bool() conversions."""
        bool_call_count = 0
        original_bool = torch.Tensor.__bool__

        def counting_bool(self: Any) -> bool:
            nonlocal bool_call_count
            bool_call_count += 1
            return original_bool(self)

        # Test with ordinary magnitude vectors
        bool_call_count = 0
        torch.Tensor.__bool__ = cast(Any, counting_bool)
        try:
            a, b = orthogonal_dot_vectors
            stable_dot_product(a, b)
            assert bool_call_count == 0, f"Expected 0 bool() calls, got {bool_call_count}"

            # Test with overflow-triggering vectors
            bool_call_count = 0
            a, b = overflow_cancelling_vectors
            stable_dot_product(a, b)
            assert bool_call_count == 0, f"Expected 0 bool() calls, got {bool_call_count}"
        finally:
            torch.Tensor.__bool__ = original_bool

    def test_gradient_finite_when_not_scaling(
        self,
        torch_dtype: torch.dtype,
    ) -> None:
        """torch.where backward pass produces finite gradients without forcing sync."""
        a = torch.randn(5, dtype=torch_dtype, requires_grad=True)
        b = torch.randn(5, dtype=torch_dtype)
        result = stable_dot_product(a, b)
        result.backward()
        grad = a.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
