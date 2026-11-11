"""Batched-column (n, k) Gauss-Seidel matches looped (n,) column-by-column calls.

Covers the prepared sparse symmetric Gauss-Seidel - for every column ``j`` of
a batched ``(n, k)`` call, exactly the result of calling it independently with
``x[:, j]``/``rhs[:, j]`` - and, as the dense oracle for the same property, the
``solve_triangular`` reference of ``tests/sparse/conftest.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchalg.sparse.preconditioners.amg._gauss_seidel import PreparedSymmetricGaussSeidel

if TYPE_CHECKING:
    from tests.sparse.conftest import DenseSymmetricGaussSeidel

_NUM_COLUMNS = 3
_ITERATIONS = 2


class TestSparseSymmetricGaussSeidelBatchedColumns:
    """Prepared sparse Gauss-Seidel batched call matches looped unbatched calls."""

    def test_matches_looped_unbatched_columns(self, poisson_1d_csr: torch.Tensor) -> None:
        torch.manual_seed(0)
        n = poisson_1d_csr.shape[0]
        x = torch.randn(n, _NUM_COLUMNS, dtype=poisson_1d_csr.dtype)
        rhs = torch.randn(n, _NUM_COLUMNS, dtype=poisson_1d_csr.dtype)
        prepared = PreparedSymmetricGaussSeidel.build(poisson_1d_csr)

        batched = prepared.smooth(rhs, x, _ITERATIONS)

        for col in range(_NUM_COLUMNS):
            expected = prepared.smooth(rhs[:, col], x[:, col], _ITERATIONS)
            torch.testing.assert_close(batched[:, col], expected)

    def test_matches_dense_oracle_on_the_block(
        self,
        poisson_1d_dense: torch.Tensor,
        poisson_1d_csr: torch.Tensor,
        dense_symmetric_gauss_seidel: DenseSymmetricGaussSeidel,
    ) -> None:
        torch.manual_seed(1)
        n = poisson_1d_dense.shape[0]
        x = torch.randn(n, _NUM_COLUMNS, dtype=poisson_1d_dense.dtype)
        rhs = torch.randn(n, _NUM_COLUMNS, dtype=poisson_1d_dense.dtype)

        expected = dense_symmetric_gauss_seidel(poisson_1d_dense, x, rhs, _ITERATIONS)
        actual = PreparedSymmetricGaussSeidel.build(poisson_1d_csr).smooth(rhs, x, _ITERATIONS)
        torch.testing.assert_close(actual, expected)
