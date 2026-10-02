"""Tests for ``torchalg.sparse.kernels.lsr_correction.sparse_lsr_correction``.

Dense-as-oracle, this project's standing rule: checked against
``preconditioners.implementations.amg._least_squares.lsr_correction`` on a
shared fixture.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations.amg._least_squares import (
    lsr_correction as dense_lsr_correction,
)
from torchalg.sparse.kernels.lsr_correction import sparse_lsr_correction


class TestSparseLsrCorrection:
    def test_matches_dense_on_poisson_1d_all_rows(self, poisson_1d_dense: torch.Tensor) -> None:
        torch.manual_seed(0)
        n = poisson_1d_dense.shape[0]
        test_vectors = torch.randn(n, 3, dtype=poisson_1d_dense.dtype)
        target_rows = torch.arange(n, dtype=torch.long)

        expected = dense_lsr_correction(test_vectors, poisson_1d_dense, target_rows)
        actual = sparse_lsr_correction(test_vectors, poisson_1d_dense.to_sparse_csr(), target_rows)
        torch.testing.assert_close(actual, expected)

    def test_matches_dense_on_poisson_2d_subset_of_rows(
        self, poisson_2d_dense: torch.Tensor
    ) -> None:
        torch.manual_seed(1)
        n = poisson_2d_dense.shape[0]
        test_vectors = torch.randn(n, 2, dtype=poisson_2d_dense.dtype)
        target_rows = torch.tensor([0, 2, 5], dtype=torch.long)

        expected = dense_lsr_correction(test_vectors, poisson_2d_dense, target_rows)
        actual = sparse_lsr_correction(test_vectors, poisson_2d_dense.to_sparse_csr(), target_rows)
        torch.testing.assert_close(actual, expected)

    def test_leaves_non_target_rows_unchanged(self, poisson_1d_dense: torch.Tensor) -> None:
        torch.manual_seed(2)
        n = poisson_1d_dense.shape[0]
        test_vectors = torch.randn(n, 2, dtype=poisson_1d_dense.dtype)
        target_rows = torch.tensor([3], dtype=torch.long)

        actual = sparse_lsr_correction(test_vectors, poisson_1d_dense.to_sparse_csr(), target_rows)
        mask = torch.ones(n, dtype=torch.bool)
        mask[target_rows] = False
        torch.testing.assert_close(actual[mask], test_vectors[mask])

    def test_does_not_mutate_input(self, poisson_1d_dense: torch.Tensor) -> None:
        torch.manual_seed(3)
        n = poisson_1d_dense.shape[0]
        test_vectors = torch.randn(n, 2, dtype=poisson_1d_dense.dtype)
        original = test_vectors.clone()
        target_rows = torch.tensor([0, 1], dtype=torch.long)

        sparse_lsr_correction(test_vectors, poisson_1d_dense.to_sparse_csr(), target_rows)
        torch.testing.assert_close(test_vectors, original)

    def test_rejects_non_csr_input(self, poisson_1d_dense: torch.Tensor) -> None:
        test_vectors = torch.randn(poisson_1d_dense.shape[0], 2, dtype=poisson_1d_dense.dtype)
        target_rows = torch.tensor([0], dtype=torch.long)
        with pytest.raises(ValueError, match="sparse CSR"):
            sparse_lsr_correction(test_vectors, poisson_1d_dense, target_rows)
