"""Tests for ``torchalg.sparse.galerkin``."""

from __future__ import annotations

import pytest
import torch

from torchalg.sparse.kernels.galerkin import form_sparse_sparse

_RANK = 4


@pytest.fixture
def aggregation_prolongation_dense(poisson_1d_dense: torch.Tensor) -> torch.Tensor:
    """Piecewise-constant (n, r) indicator matrix grouping rows into ``_RANK`` aggregates."""
    n = poisson_1d_dense.shape[0]
    dtype = poisson_1d_dense.dtype
    prolongation = torch.zeros(n, _RANK, dtype=dtype)
    aggregate_size = n // _RANK
    for row in range(n):
        prolongation[row, min(row // aggregate_size, _RANK - 1)] = 1.0
    return prolongation


class TestFormSparseSparse:
    """``form_sparse_sparse`` matches the dense Galerkin product ``P.T @ A @ P``."""

    def test_matches_dense_reference(
        self,
        poisson_1d_dense: torch.Tensor,
        poisson_1d_csr: torch.Tensor,
        aggregation_prolongation_dense: torch.Tensor,
    ) -> None:
        """Sparse-sparse formation equals the dense triple product within tolerance."""
        expected = (
            aggregation_prolongation_dense.T @ poisson_1d_dense @ aggregation_prolongation_dense
        )
        prolongation_sparse = aggregation_prolongation_dense.to_sparse_csr()
        actual = form_sparse_sparse(prolongation_sparse, poisson_1d_csr)
        torch.testing.assert_close(actual.to_dense(), expected)

    def test_returns_sparse_csr_tensor(
        self, poisson_1d_csr: torch.Tensor, aggregation_prolongation_dense: torch.Tensor
    ) -> None:
        """The coarse operator stays sparse CSR, never densified - so a multi-level
        hierarchy keeps every non-coarsest level sparse."""
        actual = form_sparse_sparse(aggregation_prolongation_dense.to_sparse_csr(), poisson_1d_csr)
        assert actual.layout == torch.sparse_csr
        assert actual.shape == (_RANK, _RANK)
