"""Tests for ``torchalg.utils.numerics``.

Focused translation-correctness checks for the dependency floor of
``torchalg`` — everything else in the migration builds on these
functions being right.
"""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from torchalg.utils.numerics import _overflow_scale_threshold, stable_dot_product


def _branch_free_stable_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Previous ``stable_dot_product``: always-on ``torch.where`` scaling (oracle)."""
    max_a = a.abs().max()
    max_b = b.abs().max()
    sqrt_threshold = _overflow_scale_threshold(torch.result_type(a, b))
    target = math.sqrt(sqrt_threshold)
    max_val = torch.maximum(max_a, max_b)
    needs_scaling = max_val > sqrt_threshold
    ones = torch.ones_like(max_val)
    safe_max_val = torch.where(needs_scaling, max_val, ones)
    scale = torch.where(needs_scaling, target / safe_max_val, ones)
    return torch.dot(a * scale, b * scale) / (scale * scale)


class _OpCounter(TorchDispatchMode):
    """Counts dispatched ATen ops and records the largest output element count."""

    def __init__(self) -> None:
        super().__init__()
        self.ops = 0
        self.largest_output = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops += 1
        out = func(*args, **(kwargs or {}))
        if isinstance(out, torch.Tensor):
            self.largest_output = max(self.largest_output, out.numel())
        return out


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


class TestStableDotProductFastPath:
    """The common path is cheap and matches the previous implementation exactly."""

    def test_ordinary_vectors_are_bit_identical_to_previous_implementation(
        self, seeded_dot_vectors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """No rescaling means the plain dot, bit for bit equal to the old formula."""
        a, b = seeded_dot_vectors
        assert torch.equal(stable_dot_product(a, b), _branch_free_stable_dot(a, b))

    def test_overflow_vectors_match_previous_implementation(
        self, overflow_cancelling_vectors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Vectors past the overflow scale take the rescaled path with the old value."""
        a, b = overflow_cancelling_vectors
        assert torch.equal(stable_dot_product(a, b), _branch_free_stable_dot(a, b))

    def test_finite_plain_dot_past_threshold_matches_previous_to_rounding(
        self, huge_times_tiny_vectors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """A finite plain result is the answer even past the threshold; it agrees to rounding."""
        a, b = huge_times_tiny_vectors
        assert math.isfinite(float(torch.dot(a, b)))
        torch.testing.assert_close(
            stable_dot_product(a, b), _branch_free_stable_dot(a, b), rtol=1e-14, atol=0.0
        )

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
    def test_non_finite_inputs_match_previous_implementation(self, bad_value: float) -> None:
        """``nan``/``inf`` entries propagate exactly as before."""
        a = torch.tensor([1.0, bad_value, 2.0], dtype=torch.float64)
        b = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64)
        new, old = stable_dot_product(a, b), _branch_free_stable_dot(a, b)
        assert torch.equal(new.isnan(), old.isnan())
        assert torch.equal(new.isinf(), old.isinf())

    def test_common_path_launches_few_ops_and_no_vector_temporaries(
        self, seeded_dot_vectors: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Plain dot plus a finiteness test: few ops, and none produces an ``n``-sized tensor."""
        a, b = seeded_dot_vectors
        with _OpCounter() as counter:
            stable_dot_product(a, b)
        assert counter.ops <= 8  # previous branch-free formulation launched about 15
        assert counter.largest_output <= 1
