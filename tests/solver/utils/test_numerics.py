"""Tests for ``torchalg.utils.numerics``.

Focused translation-correctness checks for the dependency floor of
``torchalg`` — everything else in the migration builds on these
functions being right.
"""

from __future__ import annotations

import math

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
