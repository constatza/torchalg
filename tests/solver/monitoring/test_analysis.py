"""Tests for ``torchalg.monitoring.analysis``."""

from __future__ import annotations

import math

import pytest
import torch

from torchalg.monitoring.analysis import energy_norm_history, golub_meurant_error_bound
from torchalg.utils.energy import energy_dot


class TestGolubMeurantErrorBound:
    """Tests for ``golub_meurant_error_bound``."""

    def test_matches_hand_computed_windowed_sum(self) -> None:
        """Each entry is sqrt of the forward-windowed sum of decrements."""
        decrements = [1.0, 0.5, 0.25, 0.125]

        bound = golub_meurant_error_bound(decrements, delay=2)

        expected = (
            math.sqrt(1.0 + 0.5),
            math.sqrt(0.5 + 0.25),
            math.sqrt(0.25 + 0.125),
            math.sqrt(0.125),
        )
        assert bound == pytest.approx(expected)

    def test_full_delay_sums_all_remaining_decrements(self) -> None:
        """A delay covering the whole tail reduces to a single cumulative sum per index."""
        decrements = [1.0, 2.0, 3.0]

        bound = golub_meurant_error_bound(decrements, delay=10)

        assert bound == pytest.approx((math.sqrt(6.0), math.sqrt(5.0), math.sqrt(3.0)))

    def test_empty_decrements_returns_empty_tuple(self) -> None:
        """No recorded decrements means no bound values."""
        assert golub_meurant_error_bound([], delay=5) == ()


class TestEnergyNormHistory:
    """Tests for ``energy_norm_history``."""

    def test_matches_direct_energy_dot_computation(self, torch_dtype: torch.dtype) -> None:
        """Chunked streaming gives the same answer as one direct ``energy_dot`` call."""
        A = torch.diag(torch.tensor([2.0, 8.0], dtype=torch_dtype))
        x_exact = torch.zeros(2, dtype=torch_dtype)
        solution_vectors = torch.tensor(
            [[1.0, 1.0], [0.5, 0.5], [0.0, 0.0], [0.25, -0.25]], dtype=torch_dtype
        )

        result = energy_norm_history(A, x_exact, solution_vectors, chunk_size=2)

        error = solution_vectors - x_exact
        reference = energy_dot(error, error, A).clamp_min(0.0).sqrt()
        assert torch.allclose(result, reference)
        assert result.device.type == "cpu"

    def test_chunk_size_larger_than_history_matches_single_chunk(
        self, torch_dtype: torch.dtype
    ) -> None:
        """A chunk size covering the whole history still matches the direct computation."""
        A = torch.eye(3, dtype=torch_dtype)
        x_exact = torch.ones(3, dtype=torch_dtype)
        solution_vectors = torch.zeros((5, 3), dtype=torch_dtype)

        result = energy_norm_history(A, x_exact, solution_vectors, chunk_size=1000)

        assert torch.allclose(result, torch.full((5,), math.sqrt(3.0), dtype=torch_dtype))
