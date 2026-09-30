"""Tests for ``torchalg.sparse.galerkin``."""

from __future__ import annotations

import pytest
import torch

from torchalg.sparse.galerkin import form_sparse_dense, form_sparse_sparse

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
        torch.testing.assert_close(actual, expected)

    def test_returns_dense_tensor(
        self, poisson_1d_csr: torch.Tensor, aggregation_prolongation_dense: torch.Tensor
    ) -> None:
        """The (small) coarse operator is returned dense, not sparse."""
        actual = form_sparse_sparse(aggregation_prolongation_dense.to_sparse_csr(), poisson_1d_csr)
        assert actual.layout == torch.strided
        assert actual.shape == (_RANK, _RANK)


class TestFormSparseDense:
    """``form_sparse_dense`` matches the dense Galerkin product ``Phi.T @ A @ Phi``."""

    def test_matches_dense_reference(
        self,
        poisson_1d_dense: torch.Tensor,
        poisson_1d_csr: torch.Tensor,
        aggregation_prolongation_dense: torch.Tensor,
    ) -> None:
        """Sparse-dense formation equals the dense triple product within tolerance."""
        expected = (
            aggregation_prolongation_dense.T @ poisson_1d_dense @ aggregation_prolongation_dense
        )
        actual = form_sparse_dense(poisson_1d_csr, aggregation_prolongation_dense)
        torch.testing.assert_close(actual, expected)

    def test_returns_dense_tensor(
        self, poisson_1d_csr: torch.Tensor, aggregation_prolongation_dense: torch.Tensor
    ) -> None:
        """The (small) coarse operator is returned dense."""
        actual = form_sparse_dense(poisson_1d_csr, aggregation_prolongation_dense)
        assert actual.layout == torch.strided
        assert actual.shape == (_RANK, _RANK)
