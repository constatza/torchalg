"""Tests for ``torchalg.sparse.transfer``."""

from __future__ import annotations

import pytest
import torch

from torchalg.sparse.preconditioners.amg.transfer import (
    SparseTransferOperator,
    TransferOperatorCache,
)

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


@pytest.fixture
def prolongation(poisson_1d_dense: torch.Tensor) -> torch.Tensor:
    """Sparse CSR prolongation selecting the first ``_RANK`` columns of the identity."""
    n = poisson_1d_dense.shape[0]
    return torch.eye(n, dtype=poisson_1d_dense.dtype)[:, :_RANK].to_sparse_csr()


class TestTransferOperatorCache:
    """One operator per prolongation tensor within a stage."""

    def test_same_prolongation_returns_the_same_operator(self, prolongation: torch.Tensor) -> None:
        cache = TransferOperatorCache()
        assert cache(prolongation) is cache(prolongation)

    def test_distinct_prolongations_get_distinct_operators(
        self, prolongation: torch.Tensor
    ) -> None:
        cache = TransferOperatorCache()
        assert cache(prolongation) is not cache(prolongation.clone())

    def test_cached_operator_matches_direct_construction(self, prolongation: torch.Tensor) -> None:
        coarse = torch.arange(_RANK, dtype=prolongation.dtype) + 1.0
        fine = torch.arange(prolongation.shape[0], dtype=prolongation.dtype)
        cached = TransferOperatorCache()(prolongation)
        direct = SparseTransferOperator(prolongation)
        torch.testing.assert_close(cached.prolongate(coarse), direct.prolongate(coarse))
        torch.testing.assert_close(cached.restrict(fine), direct.restrict(fine))

    def test_inference_mode_prolongation_yields_a_persistent_operator(
        self, poisson_1d_dense: torch.Tensor
    ) -> None:
        """An operator built from an inference tensor still works outside inference mode."""
        n = poisson_1d_dense.shape[0]
        with torch.inference_mode():
            inference_p = torch.eye(n, dtype=poisson_1d_dense.dtype)[:, :_RANK].to_sparse_csr()
            operator = TransferOperatorCache()(inference_p)
        fine = torch.arange(n, dtype=poisson_1d_dense.dtype)
        assert operator.restrict(fine).shape == (_RANK,)
