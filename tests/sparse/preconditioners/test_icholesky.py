"""Tests for ``torchalg.sparse.preconditioners.icholesky`` (sparse-CSR ICholesky sibling).

Dense-as-oracle, this project's standing testing rule: both preconditioners
are handed the *same* factor ``L`` (one sparse CSR, one its dense twin) and
must solve ``(L @ L.T) z = r`` to the same answer, regardless of which
triangular-solve machinery (``torch.cholesky_solve`` vs. the level-scheduled
sparse solve) computed it.
"""

from __future__ import annotations

import torch

from torchalg.preconditioners.implementations import (
    ICholeskyPreconditioner as DenseICholeskyPreconditioner,
)
from torchalg.sparse.preconditioners.ic0 import sparse_ic0
from torchalg.sparse.preconditioners.icholesky import ICholeskyPreconditioner


class TestICholeskyPreconditionerSparseDenseParity:
    """The sparse ``ICholeskyPreconditioner``'s ``apply()`` matches the dense sibling's."""

    def test_matches_dense_on_poisson_1d(self, poisson_1d_csr: torch.Tensor) -> None:
        torch.manual_seed(42)
        factor = sparse_ic0(poisson_1d_csr)
        residual = torch.randn(poisson_1d_csr.shape[0], dtype=poisson_1d_csr.values().dtype)

        dense_result = DenseICholeskyPreconditioner(factor.to_dense()).apply(residual)
        sparse_result = ICholeskyPreconditioner(factor).apply(residual)

        torch.testing.assert_close(sparse_result, dense_result)

    def test_matches_dense_on_poisson_2d(self, poisson_2d_csr: torch.Tensor) -> None:
        torch.manual_seed(42)
        factor = sparse_ic0(poisson_2d_csr)
        residual = torch.randn(poisson_2d_csr.shape[0], dtype=poisson_2d_csr.values().dtype)

        dense_result = DenseICholeskyPreconditioner(factor.to_dense()).apply(residual)
        sparse_result = ICholeskyPreconditioner(factor).apply(residual)

        torch.testing.assert_close(sparse_result, dense_result)


class TestICholeskyPreconditionerShapes:
    """Verify shape handling for batched residual input (mirrors the dense
    sibling's ``test_icholesky_preconditioner_shapes``)."""

    def test_batched_residual(self, poisson_1d_csr: torch.Tensor) -> None:
        factor = sparse_ic0(poisson_1d_csr)
        residual = torch.ones(poisson_1d_csr.shape[0], 2, dtype=poisson_1d_csr.values().dtype)

        precond = ICholeskyPreconditioner(factor)
        z = precond.apply(residual)

        assert z.shape == residual.shape
