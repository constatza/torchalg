"""Tests for ``torchalg.sparse.kernels.prolongation``."""

from __future__ import annotations

import torch

from torchalg.sparse.kernels.prolongation import (
    sparse_interpolation_prolongation,
    sparse_piecewise_constant_prolongation,
)


class TestSparsePiecewiseConstantProlongation:
    def test_one_nonzero_per_assigned_row(self, torch_dtype: torch.dtype) -> None:
        aggregate = torch.tensor([0, 0, 1, -1, 1], dtype=torch.long)
        result = sparse_piecewise_constant_prolongation(aggregate, torch_dtype).to_dense()
        expected = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch_dtype,
        )
        torch.testing.assert_close(result, expected)


class TestSparseInterpolationProlongation:
    def test_builds_multi_nonzero_rows_from_triples(self, torch_dtype: torch.dtype) -> None:
        """BAMG's P shape: multiple (caliber-bounded) nonzeros per fine row, LS-fitted values."""
        row_indices = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long)
        col_indices = torch.tensor([1, 0, 0, 2, 0, 1], dtype=torch.long)
        values = torch.tensor([0.5, 0.25, 1.0, 1.0, 0.3, 0.2], dtype=torch_dtype)

        result = sparse_interpolation_prolongation(row_indices, col_indices, values, (3, 3))

        expected = torch.zeros(3, 3, dtype=torch_dtype)
        expected[0, 1] = 0.5
        expected[0, 0] = 0.25
        expected[1, 0] = 1.0
        expected[2, 2] = 1.0
        expected[2, 0] = 0.3
        expected[2, 1] = 0.2
        torch.testing.assert_close(result.to_dense(), expected)

    def test_returns_sparse_csr(self, torch_dtype: torch.dtype) -> None:
        row_indices = torch.tensor([0], dtype=torch.long)
        col_indices = torch.tensor([0], dtype=torch.long)
        values = torch.tensor([1.0], dtype=torch_dtype)
        result = sparse_interpolation_prolongation(row_indices, col_indices, values, (2, 2))
        assert result.layout == torch.sparse_csr

    def test_handles_empty_triples(self, torch_dtype: torch.dtype) -> None:
        """A row with no interpolatory entries at all (e.g. a coarse point's own row)."""
        row_indices = torch.empty(0, dtype=torch.long)
        col_indices = torch.empty(0, dtype=torch.long)
        values = torch.empty(0, dtype=torch_dtype)
        result = sparse_interpolation_prolongation(row_indices, col_indices, values, (2, 2))
        torch.testing.assert_close(result.to_dense(), torch.zeros(2, 2, dtype=torch_dtype))
