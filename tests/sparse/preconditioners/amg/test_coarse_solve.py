"""``torchalg.sparse.preconditioners.amg.coarse_solve`` tests."""

from __future__ import annotations

import torch

from torchalg.sparse.preconditioners.amg.coarse_solve import (
    dense_coarse_solve,
    dense_pseudo_inverse_solve,
)


class TestDenseCoarseSolve:
    def test_matches_torch_linalg_solve_on_densified_sparse_matrix(
        self, poisson_1d_dense: torch.Tensor, poisson_1d_csr: torch.Tensor
    ) -> None:
        """Sparse CSR input must produce the same result as solving the densified matrix."""
        rhs = torch.ones(poisson_1d_dense.shape[0], dtype=poisson_1d_dense.dtype)
        expected = torch.linalg.solve(poisson_1d_dense, rhs)
        actual = dense_coarse_solve(poisson_1d_csr, rhs)
        torch.testing.assert_close(actual, expected)

    def test_accepts_already_dense_matrix(self, poisson_1d_dense: torch.Tensor) -> None:
        """A dense input must be solved directly, matching ``torch.linalg.solve``."""
        rhs = torch.ones(poisson_1d_dense.shape[0], dtype=poisson_1d_dense.dtype)
        expected = torch.linalg.solve(poisson_1d_dense, rhs)
        actual = dense_coarse_solve(poisson_1d_dense, rhs)
        torch.testing.assert_close(actual, expected)


class TestDensePseudoInverseSolve:
    def test_matches_torch_linalg_pinv_on_densified_sparse_matrix(
        self, poisson_1d_dense: torch.Tensor, poisson_1d_csr: torch.Tensor
    ) -> None:
        """Sparse CSR input must produce the same result as pinv-solving the densified matrix."""
        rhs = torch.ones(poisson_1d_dense.shape[0], dtype=poisson_1d_dense.dtype)
        expected = torch.linalg.pinv(poisson_1d_dense) @ rhs
        actual = dense_pseudo_inverse_solve(poisson_1d_csr, rhs)
        torch.testing.assert_close(actual, expected)

    def test_accepts_already_dense_matrix(self, poisson_1d_dense: torch.Tensor) -> None:
        """A dense input must be solved directly, matching ``torch.linalg.pinv``."""
        rhs = torch.ones(poisson_1d_dense.shape[0], dtype=poisson_1d_dense.dtype)
        expected = torch.linalg.pinv(poisson_1d_dense) @ rhs
        actual = dense_pseudo_inverse_solve(poisson_1d_dense, rhs)
        torch.testing.assert_close(actual, expected)

    def test_handles_singular_coarse_matrix(self, torch_dtype: torch.dtype) -> None:
        """A singular matrix (dropped candidates leaving a zero row/column) needs pinv, not solve."""
        singular = torch.zeros(3, 3, dtype=torch_dtype)
        singular[0, 0] = 2.0
        singular[1, 1] = 3.0
        # row/col 2 is all-zero - singular, torch.linalg.solve would raise.
        sparse_singular = singular.to_sparse_csr()
        rhs = torch.ones(3, dtype=torch_dtype)

        expected = torch.linalg.pinv(singular) @ rhs
        actual = dense_pseudo_inverse_solve(sparse_singular, rhs)
        torch.testing.assert_close(actual, expected)
