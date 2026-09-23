"""Tests for ``torchalg.strategies.norms``."""

from __future__ import annotations

import math

import pytest
import torch

from torchalg.strategies.norms import energy_norm, euclidean_norm


class TestEuclideanNorm:
    """Tests for ``euclidean_norm``."""

    def test_matches_hand_computed_value(self, three_four_vector: torch.Tensor) -> None:
        """``||[3, 4]||_2 == 5``, the textbook 3-4-5 triangle."""
        assert euclidean_norm(three_four_vector).item() == pytest.approx(5.0)

    def test_returns_tensor(self, three_four_vector: torch.Tensor) -> None:
        """Return type is a tensor, matching the updated ``Norm`` type alias."""
        assert isinstance(euclidean_norm(three_four_vector), torch.Tensor)

    def test_zero_vector_has_zero_norm(self, torch_dtype: torch.dtype) -> None:
        """The zero vector has norm 0."""
        assert euclidean_norm(torch.zeros(4, dtype=torch_dtype)).item() == 0.0


class TestEnergyNorm:
    """Tests for ``energy_norm``."""

    def test_diagonal_case_matches_hand_computed_value(
        self,
        energy_norm_diag_entries: torch.Tensor,
        ones_vector_2: torch.Tensor,
    ) -> None:
        """``||[1, 1]||_A == sqrt(2 + 8) == sqrt(10)`` for ``A = diag(2, 8)``."""
        anorm = energy_norm(energy_norm_diag_entries)
        assert anorm(ones_vector_2).item() == pytest.approx(math.sqrt(10.0))

    def test_full_matrix_case_matches_hand_computed_value(
        self,
        energy_norm_full_matrix: torch.Tensor,
        ones_vector_2: torch.Tensor,
    ) -> None:
        """The dense-matrix branch gives the same result as the diagonal branch."""
        anorm = energy_norm(energy_norm_full_matrix)
        assert anorm(ones_vector_2).item() == pytest.approx(math.sqrt(10.0))

    def test_diagonal_and_full_branches_agree(
        self,
        energy_norm_diag_entries: torch.Tensor,
        energy_norm_full_matrix: torch.Tensor,
        ones_vector_2: torch.Tensor,
    ) -> None:
        """The O(n) diagonal path and the O(n^2) full-matrix path agree exactly."""
        diag_norm = energy_norm(energy_norm_diag_entries)(ones_vector_2)
        full_norm = energy_norm(energy_norm_full_matrix)(ones_vector_2)
        assert diag_norm.item() == pytest.approx(full_norm.item())

    def test_returns_tensor(
        self,
        energy_norm_diag_entries: torch.Tensor,
        ones_vector_2: torch.Tensor,
    ) -> None:
        """Return type is a tensor, matching the updated ``Norm`` type alias."""
        anorm = energy_norm(energy_norm_diag_entries)
        assert isinstance(anorm(ones_vector_2), torch.Tensor)

    def test_tiny_negative_quadratic_form_clamps_to_zero(self, torch_dtype: torch.dtype) -> None:
        """A quadratic form pushed slightly negative by float noise returns 0, not NaN."""
        anorm = energy_norm(torch.tensor([-1e-20, 1.0], dtype=torch_dtype))
        assert anorm(torch.tensor([1.0, 0.0], dtype=torch_dtype)).item() == 0.0
