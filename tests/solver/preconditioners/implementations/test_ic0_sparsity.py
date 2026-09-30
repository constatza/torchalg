"""Tests for IC(0)'s sparsity mask (the non-filling guarantee).

Checks ``ic0_sparsity_mask`` directly rather than running a full
``IC0Preconditioner``/``dense_ic0`` factorization: the mask is what decides
IC(0)'s sparsity pattern (``lower_triangle & (matrix.abs() > threshold)``),
computed before any elimination happens. Testing it directly is O(n^2) with
no loop, and is independent of numerical convergence or breakdown - a
question about pattern shouldn't require running (and surviving) the O(n^3)
elimination to answer.

Verifies that:
- IC(0)'s sparsity mask respects drop thresholds correctly.
- Zero-threshold preserves the original sparsity pattern exactly.
- nnz(mask) ≤ nnz(A) in all cases (non-filling).
- Normalized matrices maintain the same sparsity properties.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations._masked_factorization import ic0_sparsity_mask


def _nnz(matrix: torch.Tensor) -> int:
    """Count non-zero (or true) entries in a dense tensor."""
    return int(
        (matrix != 0).sum().item()
        if matrix.dtype == torch.bool
        else (matrix.abs() > 0).sum().item()
    )


def _equilibrate_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Apply diagonal equilibration (row and column scaling) to make matrix unit diagonal."""
    diag_sqrt = matrix.diagonal().sqrt()
    scaling = 1.0 / diag_sqrt
    scaled = matrix * scaling.unsqueeze(1) * scaling.unsqueeze(0)
    return scaled


def _get_threshold_values(matrix: torch.Tensor) -> tuple[float, float, float]:
    """Return three threshold values: 0, single-precision eps, and half of that.

    Args:
        matrix: Input matrix to compute scale from.

    Returns:
        Tuple of (0, single_precision_eps * scale, 0.5 * single_precision_eps * scale).
    """
    # Single precision epsilon for float32
    single_prec_eps = torch.finfo(torch.float32).eps
    # Scale relative to matrix magnitude (Frobenius norm)
    scale = torch.norm(matrix).item() if torch.norm(matrix) > 0 else 1.0
    return (0.0, single_prec_eps * scale, 0.5 * single_prec_eps * scale)


@pytest.fixture
def sparse_mixed_spd_matrix(torch_dtype: torch.dtype) -> torch.Tensor:
    """8x8 SPD matrix with mixed magnitude values (including negatives and ~machine epsilon).

    Designed to test threshold effects: contains entries spanning multiple
    orders of magnitude (1e1 down to 1e-14 range) with both positive and
    negative values, plus structural zeros. Different thresholds drop
    different subsets of small entries.

    Returns:
        torch.Tensor: 8x8 SPD matrix, sparse (~25% density).
    """
    n = 8
    diag_values = torch.tensor([20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0], dtype=torch_dtype)
    matrix = torch.diag(diag_values)

    # Off-diagonals: mix of large (O(1)), medium (O(1e-2)), tiny (O(1e-8)), and machine-epsilon (O(1e-14))
    # Large negative values (O(1))
    for i in range(n - 1):
        matrix[i + 1, i] = matrix[i, i + 1] = -(2.0 + 0.5 * i)

    # Medium values (O(1e-2)): positive
    matrix[0, 2] = matrix[2, 0] = 0.05
    matrix[1, 3] = matrix[3, 1] = 0.03
    matrix[4, 6] = matrix[6, 4] = 0.02
    matrix[5, 7] = matrix[7, 5] = 0.01

    # Small values (O(1e-8)): negative
    matrix[0, 4] = matrix[4, 0] = -1e-8
    matrix[2, 6] = matrix[6, 2] = -5e-9
    matrix[1, 5] = matrix[5, 1] = -2e-8

    # Machine-epsilon scale (O(1e-14)): mix of positive and negative
    matrix[3, 5] = matrix[5, 3] = 1.5e-14
    matrix[0, 7] = matrix[7, 0] = -2.5e-14
    matrix[2, 4] = matrix[4, 2] = 8e-15

    # Symmetrize to ensure SPD (matrix + matrix.T) / 2
    matrix = (matrix + matrix.T) / 2
    # Boost diagonal to ensure positive definiteness
    matrix.diagonal().add_(1.0)

    return matrix


class TestIC0SparsityMask:
    """Tests for IC(0)'s sparsity mask across thresholds and normalizations."""

    def test_raw_matrix_zero_threshold_preserves_sparsity(
        self,
        sparse_mixed_spd_matrix: torch.Tensor,
    ) -> None:
        """Verify that zero threshold produces exactly the same sparsity pattern as input.

        At threshold=0, the mask condition ``|A_ij| > 0`` keeps every
        structurally non-zero entry - no numerical dropping. The mask's
        sparsity should be identical to the lower triangle of the input.
        """
        matrix = sparse_mixed_spd_matrix
        mask = ic0_sparsity_mask(matrix, threshold=0.0)

        lower_triangle = torch.tril(matrix)
        nnz_original = _nnz(lower_triangle)
        nnz_mask = _nnz(mask)

        assert nnz_mask == nnz_original, (
            f"Zero-threshold IC(0) mask must match input's sparsity pattern: "
            f"nnz(A_lower)={nnz_original}, nnz(mask)={nnz_mask}"
        )

    def test_raw_matrix_mask_smaller_than_original(
        self,
        sparse_mixed_spd_matrix: torch.Tensor,
    ) -> None:
        """Verify that nnz(mask) ≤ nnz(A) for all threshold values.

        The mask is defined as ``lower_triangle & (|A| > threshold)`` - an
        AND against the input's own support - so it can only maintain or
        shrink the sparsity pattern, never add entries outside it.
        """
        matrix = sparse_mixed_spd_matrix
        threshold_0, threshold_1, threshold_2 = _get_threshold_values(matrix)

        nnz_original = _nnz(torch.tril(matrix))

        for threshold in [threshold_0, threshold_1, threshold_2]:
            mask = ic0_sparsity_mask(matrix, threshold=threshold)
            nnz_mask = _nnz(mask)

            assert nnz_mask <= nnz_original, (
                f"nnz(mask(threshold={threshold}))={nnz_mask} > "
                f"nnz(A_lower)={nnz_original} - mask has more entries than input"
            )

    def test_normalized_matrix_zero_threshold_preserves_sparsity(
        self,
        sparse_mixed_spd_matrix: torch.Tensor,
    ) -> None:
        """Verify zero threshold on equilibrated matrix preserves its sparsity pattern.

        After diagonal equilibration (row/column scaling to unit diagonal),
        the sparsity pattern should remain unchanged, and the mask at zero
        threshold should match it exactly.
        """
        matrix = sparse_mixed_spd_matrix
        normalized = _equilibrate_matrix(matrix)
        mask = ic0_sparsity_mask(normalized, threshold=0.0)

        lower_triangle = torch.tril(normalized)
        nnz_normalized = _nnz(lower_triangle)
        nnz_mask = _nnz(mask)

        assert nnz_mask == nnz_normalized, (
            f"Zero-threshold IC(0) mask on normalized matrix must match pattern: "
            f"nnz(A_lower_norm)={nnz_normalized}, nnz(mask)={nnz_mask}"
        )

    def test_normalized_matrix_mask_smaller_than_original(
        self,
        sparse_mixed_spd_matrix: torch.Tensor,
    ) -> None:
        """Verify that nnz(mask) ≤ nnz(A_normalized) for all threshold values."""
        matrix = sparse_mixed_spd_matrix
        normalized = _equilibrate_matrix(matrix)
        threshold_0, threshold_1, threshold_2 = _get_threshold_values(normalized)

        nnz_normalized = _nnz(torch.tril(normalized))

        for threshold in [threshold_0, threshold_1, threshold_2]:
            mask = ic0_sparsity_mask(normalized, threshold=threshold)
            nnz_mask = _nnz(mask)

            assert nnz_mask <= nnz_normalized, (
                f"nnz(mask_norm(threshold={threshold}))={nnz_mask} > "
                f"nnz(A_lower_norm)={nnz_normalized}"
            )

    def test_threshold_monotonicity_raw_matrix(
        self,
        sparse_mixed_spd_matrix: torch.Tensor,
    ) -> None:
        """Verify that increasing threshold monotonically decreases nnz(mask).

        If threshold_1 < threshold_2, then nnz(mask_1) ≥ nnz(mask_2), since
        more entries fail ``|A_ij| > threshold`` at the larger threshold.
        """
        matrix = sparse_mixed_spd_matrix
        threshold_0, threshold_1, threshold_2 = _get_threshold_values(matrix)

        nnz_0 = _nnz(ic0_sparsity_mask(matrix, threshold=threshold_0))
        nnz_1 = _nnz(ic0_sparsity_mask(matrix, threshold=threshold_1))
        nnz_2 = _nnz(ic0_sparsity_mask(matrix, threshold=threshold_2))

        assert nnz_0 >= nnz_1 >= nnz_2, (
            f"Mask sparsity should decrease as threshold increases: "
            f"nnz(t=0)={nnz_0}, nnz(t={threshold_1})={nnz_1}, nnz(t={threshold_2})={nnz_2}"
        )

    def test_threshold_monotonicity_normalized_matrix(
        self,
        sparse_mixed_spd_matrix: torch.Tensor,
    ) -> None:
        """Verify that increasing threshold monotonically decreases nnz(mask) for normalized matrix."""
        matrix = sparse_mixed_spd_matrix
        normalized = _equilibrate_matrix(matrix)
        threshold_0, threshold_1, threshold_2 = _get_threshold_values(normalized)

        nnz_0 = _nnz(ic0_sparsity_mask(normalized, threshold=threshold_0))
        nnz_1 = _nnz(ic0_sparsity_mask(normalized, threshold=threshold_1))
        nnz_2 = _nnz(ic0_sparsity_mask(normalized, threshold=threshold_2))

        assert nnz_0 >= nnz_1 >= nnz_2, (
            f"Normalized matrix: mask nnz should decrease as threshold increases: "
            f"nnz(t=0)={nnz_0}, nnz(t={threshold_1})={nnz_1}, nnz(t={threshold_2})={nnz_2}"
        )
