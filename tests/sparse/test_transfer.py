"""Tests for ``torchalg.sparse.transfer``."""

from __future__ import annotations

import torch

from torchalg.sparse.transfer import SparseTransferOperator

_RANK = 4


class TestSparseTransferOperator:
    """``SparseTransferOperator`` matches ``DenseTransferOperator`` on a dense twin."""

    def test_prolongate_matches_dense_matmul(self, poisson_1d_dense: torch.Tensor) -> None:
        """``prolongate(coarse) == P @ coarse`` against the dense prolongation matrix."""
        n = poisson_1d_dense.shape[0]
        prolongation_dense = torch.eye(n, dtype=poisson_1d_dense.dtype)[:, :_RANK]
        operator = SparseTransferOperator(prolongation_dense.to_sparse_csr())
        coarse = torch.arange(_RANK, dtype=poisson_1d_dense.dtype) + 1.0
        torch.testing.assert_close(operator.prolongate(coarse), prolongation_dense @ coarse)

    def test_restrict_matches_dense_transpose_matmul(self, poisson_1d_dense: torch.Tensor) -> None:
        """``restrict(fine) == P.T @ fine`` against the dense prolongation matrix."""
        n = poisson_1d_dense.shape[0]
        prolongation_dense = torch.eye(n, dtype=poisson_1d_dense.dtype)[:, :_RANK]
        operator = SparseTransferOperator(prolongation_dense.to_sparse_csr())
        fine = torch.arange(n, dtype=poisson_1d_dense.dtype)
        torch.testing.assert_close(operator.restrict(fine), prolongation_dense.T @ fine)

    def test_prolongate_output_is_dense(self, poisson_1d_dense: torch.Tensor) -> None:
        """Applying the transfer operator to a dense vector returns a dense vector."""
        n = poisson_1d_dense.shape[0]
        prolongation_dense = torch.eye(n, dtype=poisson_1d_dense.dtype)[:, :_RANK]
        operator = SparseTransferOperator(prolongation_dense.to_sparse_csr())
        coarse = torch.ones(_RANK, dtype=poisson_1d_dense.dtype)
        assert operator.prolongate(coarse).layout == torch.strided
