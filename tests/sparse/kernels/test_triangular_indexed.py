"""Parity tests for the indexed (``rows=``) variant of sparse Gauss-Seidel.

Mirrors PyAMG's ``gauss_seidel_indexed`` against a dense ``solve_triangular``
oracle (``tests/sparse/conftest.py``): only the given rows are updated,
everything else freezes at its current value for the full call (used by
adaptive-SA's candidate-refinement loop).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchalg.sparse.preconditioners.amg._gauss_seidel import PreparedSymmetricGaussSeidel

if TYPE_CHECKING:
    from tests.sparse.conftest import DenseSymmetricGaussSeidel


class TestIndexedGaussSeidelMatchesDense:
    """Prepared sparse Gauss-Seidel with ``rows=`` matches the dense oracle."""

    def test_matches_dense_poisson_1d_subset_rows(
        self,
        poisson_1d_dense: torch.Tensor,
        poisson_1d_csr: torch.Tensor,
        dense_symmetric_gauss_seidel: DenseSymmetricGaussSeidel,
    ) -> None:
        torch.manual_seed(0)
        n = poisson_1d_dense.shape[0]
        x = torch.randn(n, dtype=poisson_1d_dense.dtype)
        rhs = torch.randn(n, dtype=poisson_1d_dense.dtype)
        rows = torch.nonzero(torch.arange(n) % 2 == 0).flatten()

        dense_result = dense_symmetric_gauss_seidel(
            poisson_1d_dense, x.clone(), rhs, iterations=2, rows=rows
        )
        sparse_result = PreparedSymmetricGaussSeidel.build(poisson_1d_csr).smooth(
            rhs, x.clone(), 2, rows=rows
        )
        torch.testing.assert_close(sparse_result, dense_result)

    def test_matches_dense_poisson_2d_subset_rows(
        self,
        poisson_2d_dense: torch.Tensor,
        poisson_2d_csr: torch.Tensor,
        dense_symmetric_gauss_seidel: DenseSymmetricGaussSeidel,
    ) -> None:
        torch.manual_seed(1)
        n = poisson_2d_dense.shape[0]
        x = torch.randn(n, dtype=poisson_2d_dense.dtype)
        rhs = torch.randn(n, dtype=poisson_2d_dense.dtype)
        rows = torch.nonzero(torch.arange(n) % 3 != 0).flatten()

        dense_result = dense_symmetric_gauss_seidel(
            poisson_2d_dense, x.clone(), rhs, iterations=3, rows=rows
        )
        sparse_result = PreparedSymmetricGaussSeidel.build(poisson_2d_csr).smooth(
            rhs, x.clone(), 3, rows=rows
        )
        torch.testing.assert_close(sparse_result, dense_result)

    def test_no_rows_matches_full_update(
        self,
        poisson_1d_dense: torch.Tensor,
        poisson_1d_csr: torch.Tensor,
    ) -> None:
        """Omitting ``rows`` keeps the default behavior: every row updates."""
        torch.manual_seed(2)
        n = poisson_1d_dense.shape[0]
        x = torch.randn(n, dtype=poisson_1d_dense.dtype)
        rhs = torch.randn(n, dtype=poisson_1d_dense.dtype)
        prepared = PreparedSymmetricGaussSeidel.build(poisson_1d_csr)

        full_rows_result = prepared.smooth(rhs, x.clone(), 2, rows=torch.arange(n))
        no_rows_result = prepared.smooth(rhs, x.clone(), 2)
        torch.testing.assert_close(full_rows_result, no_rows_result)

    def test_frozen_rows_keep_their_value(
        self,
        poisson_1d_csr: torch.Tensor,
        poisson_1d_dense: torch.Tensor,
    ) -> None:
        torch.manual_seed(3)
        n = poisson_1d_dense.shape[0]
        x = torch.randn(n, dtype=poisson_1d_dense.dtype)
        rhs = torch.randn(n, dtype=poisson_1d_dense.dtype)
        rows = torch.tensor([1, 4, 7])
        result = PreparedSymmetricGaussSeidel.build(poisson_1d_csr).smooth(rhs, x, 2, rows=rows)
        frozen = torch.ones(n, dtype=torch.bool)
        frozen[rows] = False
        torch.testing.assert_close(result[frozen], x[frozen], rtol=0, atol=0)
