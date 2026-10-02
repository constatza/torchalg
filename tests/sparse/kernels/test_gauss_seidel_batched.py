"""Batched-column (n, k) Gauss-Seidel matches looped (n,) column-by-column calls.

Covers both the dense ``_relaxation.symmetric_gauss_seidel`` and the sparse
``_gauss_seidel.sparse_symmetric_gauss_seidel`` - each must produce, for
every column ``j`` of a batched ``(n, k)`` call, exactly the same result as
calling the same function independently with ``x[:, j]``/``rhs[:, j]``.
"""

from __future__ import annotations

import torch

from torchalg.preconditioners.implementations.amg._relaxation import (
    symmetric_gauss_seidel as dense_symmetric_gauss_seidel,
)
from torchalg.sparse.preconditioners.amg._gauss_seidel import sparse_symmetric_gauss_seidel

_NUM_COLUMNS = 3
_ITERATIONS = 2


class TestDenseSymmetricGaussSeidelBatchedColumns:
    """Dense ``symmetric_gauss_seidel`` batched call matches looped unbatched calls."""

    def test_matches_looped_unbatched_columns(self, poisson_1d_dense: torch.Tensor) -> None:
        torch.manual_seed(0)
        n = poisson_1d_dense.shape[0]
        x = torch.randn(n, _NUM_COLUMNS, dtype=poisson_1d_dense.dtype)
        rhs = torch.randn(n, _NUM_COLUMNS, dtype=poisson_1d_dense.dtype)

        batched = dense_symmetric_gauss_seidel(poisson_1d_dense, x, rhs, _ITERATIONS)

        for col in range(_NUM_COLUMNS):
            expected = dense_symmetric_gauss_seidel(
                poisson_1d_dense, x[:, col], rhs[:, col], _ITERATIONS
            )
            torch.testing.assert_close(batched[:, col], expected)


class TestSparseSymmetricGaussSeidelBatchedColumns:
    """Sparse ``sparse_symmetric_gauss_seidel`` batched call matches looped unbatched calls."""

    def test_matches_looped_unbatched_columns(self, poisson_1d_csr: torch.Tensor) -> None:
        torch.manual_seed(0)
        n = poisson_1d_csr.shape[0]
        x = torch.randn(n, _NUM_COLUMNS, dtype=poisson_1d_csr.dtype)
        rhs = torch.randn(n, _NUM_COLUMNS, dtype=poisson_1d_csr.dtype)

        batched = sparse_symmetric_gauss_seidel(poisson_1d_csr, x, rhs, _ITERATIONS)

        for col in range(_NUM_COLUMNS):
            expected = sparse_symmetric_gauss_seidel(
                poisson_1d_csr, x[:, col], rhs[:, col], _ITERATIONS
            )
            torch.testing.assert_close(batched[:, col], expected)
