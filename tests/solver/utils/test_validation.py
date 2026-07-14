"""Tests for ``torchalg.utils.validation``.

Focused translation-correctness checks for the dependency floor of
``torchalg``.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.utils.validation import (
    check_solution_validity,
    describe_breakdown_reason,
    validate_ax_equals_b,
    validate_matrix,
    validate_rhs_vector,
)


class TestValidateMatrix:
    """Tests for ``validate_matrix``."""

    def test_valid_square_finite_matrix_passes(
        self,
        valid_spd_system: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """A square, finite matrix raises nothing."""
        a, _, _ = valid_spd_system
        validate_matrix(a)

    def test_non_square_matrix_raises(self, non_square_matrix: torch.Tensor) -> None:
        """A non-square matrix raises ``ValueError``."""
        with pytest.raises(ValueError, match="square"):
            validate_matrix(non_square_matrix)

    def test_non_2d_matrix_raises(self, torch_dtype: torch.dtype) -> None:
        """A 1D tensor raises ``ValueError``."""
        with pytest.raises(ValueError, match="2D"):
            validate_matrix(torch.ones(3, dtype=torch_dtype))

    def test_non_finite_matrix_raises(self, non_finite_matrix: torch.Tensor) -> None:
        """A matrix containing ``nan``/``inf`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="non-finite"):
            validate_matrix(non_finite_matrix)


class TestValidateRhsVector:
    """Tests for ``validate_rhs_vector``."""

    def test_none_is_a_noop(self) -> None:
        """``None`` RHS is accepted (nothing to validate)."""
        validate_rhs_vector(None)

    def test_valid_vector_passes(
        self,
        valid_spd_system: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """A finite vector matching the matrix size raises nothing."""
        a, b, _ = valid_spd_system
        validate_rhs_vector(b, a)

    def test_non_finite_vector_raises(self, non_finite_vector: torch.Tensor) -> None:
        """A vector containing ``nan``/``inf`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="non-finite"):
            validate_rhs_vector(non_finite_vector)

    def test_mismatched_size_raises(
        self,
        valid_spd_system: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch_dtype: torch.dtype,
    ) -> None:
        """A vector whose length doesn't match the matrix size raises ``ValueError``."""
        a, _, _ = valid_spd_system
        wrong_size_b = torch.ones(a.shape[0] + 1, dtype=torch_dtype)
        with pytest.raises(ValueError, match="doesn't match matrix size"):
            validate_rhs_vector(wrong_size_b, a)

    def test_non_column_2d_vector_raises(self, torch_dtype: torch.dtype) -> None:
        """A 2D RHS that isn't a column vector raises ``ValueError``."""
        with pytest.raises(ValueError, match="column vector"):
            validate_rhs_vector(torch.ones((3, 2), dtype=torch_dtype))

    def test_higher_than_2d_raises(self, torch_dtype: torch.dtype) -> None:
        """A 3D RHS raises ``ValueError``."""
        with pytest.raises(ValueError, match="1D or 2D"):
            validate_rhs_vector(torch.ones((2, 2, 2), dtype=torch_dtype))


class TestValidateAxEqualsB:
    """Tests for ``validate_ax_equals_b``."""

    def test_consistent_system_passes(
        self,
        valid_spd_system: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """``matrix @ lhs`` parallel to ``rhs`` raises nothing."""
        a, b, x_exact = valid_spd_system
        validate_ax_equals_b(a, b, x_exact)

    def test_scaled_rhs_still_passes(
        self,
        valid_spd_system: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """The check is scale-agnostic: a positive multiple of ``b`` still passes."""
        a, b, x_exact = valid_spd_system
        validate_ax_equals_b(a, 3.0 * b, x_exact)

    def test_inconsistent_system_raises(
        self,
        valid_spd_system: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch_dtype: torch.dtype,
    ) -> None:
        """An ``rhs`` unrelated to ``matrix @ lhs`` raises ``ValueError``."""
        a, _, x_exact = valid_spd_system
        unrelated_rhs = torch.arange(1, a.shape[0] + 1, dtype=torch_dtype)
        # Flip sign so the (near-random) unrelated RHS is very unlikely to be
        # accidentally parallel with matrix @ lhs.
        unrelated_rhs = unrelated_rhs.flip(0) * -1.0
        with pytest.raises(ValueError, match="Ax=b invariant violated"):
            validate_ax_equals_b(a, unrelated_rhs, x_exact)


class TestCheckSolutionValidity:
    """Tests for ``check_solution_validity``."""

    def test_finite_solution_is_valid(self, finite_solution_vector: torch.Tensor) -> None:
        """A fully-finite vector is valid."""
        assert check_solution_validity(finite_solution_vector) is True

    def test_nan_solution_is_invalid(self, nan_solution_vector: torch.Tensor) -> None:
        """A vector containing ``nan`` is invalid."""
        assert check_solution_validity(nan_solution_vector) is False

    def test_inf_solution_is_invalid(self, inf_solution_vector: torch.Tensor) -> None:
        """A vector containing ``inf`` is invalid."""
        assert check_solution_validity(inf_solution_vector) is False


class TestDescribeBreakdownReason:
    """Tests for ``describe_breakdown_reason``."""

    def test_finite_solution_has_empty_reason(self, finite_solution_vector: torch.Tensor) -> None:
        """A fully-finite vector has no breakdown reason."""
        assert describe_breakdown_reason(finite_solution_vector) == ""

    def test_nan_only_reports_nan_reason(self, nan_solution_vector: torch.Tensor) -> None:
        """A ``nan``-only vector reports exactly the nan reason."""
        assert describe_breakdown_reason(nan_solution_vector) == "nan_in_solution"

    def test_inf_only_reports_inf_reason(self, inf_solution_vector: torch.Tensor) -> None:
        """An ``inf``-only vector reports exactly the inf reason."""
        assert describe_breakdown_reason(inf_solution_vector) == "inf_in_solution"

    def test_nan_and_inf_reports_both_reasons(
        self,
        nan_and_inf_solution_vector: torch.Tensor,
    ) -> None:
        """A vector with both ``nan`` and ``inf`` reports both reasons."""
        assert describe_breakdown_reason(nan_and_inf_solution_vector) == (
            "nan_in_solution, inf_in_solution"
        )
