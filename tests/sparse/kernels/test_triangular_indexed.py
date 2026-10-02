"""Parity tests for the indexed (``rows=``) variant of sparse Gauss-Seidel.

Mirrors PyAMG's ``gauss_seidel_indexed`` / the dense
``preconditioners.implementations.amg._relaxation.symmetric_gauss_seidel``'s
``rows=`` parameter: only the given rows are updated, everything else
freezes at its current value for the full call (used by adaptive-SA's
candidate-refinement loop).
"""

from __future__ import annotations

import torch

from torchalg.preconditioners.implementations.amg._relaxation import (
    symmetric_gauss_seidel as dense_symmetric_gauss_seidel,
)
from torchalg.sparse.preconditioners.amg._gauss_seidel import sparse_symmetric_gauss_seidel


class TestIndexedGaussSeidelMatchesDense:
    """Sparse ``sparse_symmetric_gauss_seidel(..., rows=...)`` matches dense."""

    def test_matches_dense_poisson_1d_subset_rows(
        self,
        poisson_1d_dense: torch.Tensor,
        poisson_1d_csr: torch.Tensor,
    ) -> None:
        torch.manual_seed(0)
        n = poisson_1d_dense.shape[0]
        x = torch.randn(n, dtype=poisson_1d_dense.dtype)
        rhs = torch.randn(n, dtype=poisson_1d_dense.dtype)
        rows = torch.nonzero(torch.arange(n) % 2 == 0).flatten()

        dense_result = dense_symmetric_gauss_seidel(
            poisson_1d_dense, x.clone(), rhs, iterations=2, rows=rows
        )
        sparse_result = sparse_symmetric_gauss_seidel(
            poisson_1d_csr, x.clone(), rhs, iterations=2, rows=rows
        )
        torch.testing.assert_close(sparse_result, dense_result)

    def test_matches_dense_poisson_2d_subset_rows(
        self,
        poisson_2d_dense: torch.Tensor,
        poisson_2d_csr: torch.Tensor,
    ) -> None:
        torch.manual_seed(1)
        n = poisson_2d_dense.shape[0]
        x = torch.randn(n, dtype=poisson_2d_dense.dtype)
        rhs = torch.randn(n, dtype=poisson_2d_dense.dtype)
        rows = torch.nonzero(torch.arange(n) % 3 != 0).flatten()

        dense_result = dense_symmetric_gauss_seidel(
            poisson_2d_dense, x.clone(), rhs, iterations=3, rows=rows
        )
        sparse_result = sparse_symmetric_gauss_seidel(
            poisson_2d_csr, x.clone(), rhs, iterations=3, rows=rows
        )
        torch.testing.assert_close(sparse_result, dense_result)

    def test_no_rows_matches_full_update(
        self,
        poisson_1d_dense: torch.Tensor,
        poisson_1d_csr: torch.Tensor,
    ) -> None:
        """Omitting ``rows`` keeps today's behavior: every row updates."""
        torch.manual_seed(2)
        n = poisson_1d_dense.shape[0]
        x = torch.randn(n, dtype=poisson_1d_dense.dtype)
        rhs = torch.randn(n, dtype=poisson_1d_dense.dtype)

        full_rows_result = sparse_symmetric_gauss_seidel(
            poisson_1d_csr, x.clone(), rhs, iterations=2, rows=torch.arange(n)
        )
        no_rows_result = sparse_symmetric_gauss_seidel(poisson_1d_csr, x.clone(), rhs, iterations=2)
        torch.testing.assert_close(full_rows_result, no_rows_result)
