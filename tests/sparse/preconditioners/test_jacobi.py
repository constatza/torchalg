"""Tests for ``torchalg.sparse.preconditioners.jacobi`` (sparse-CSR Jacobi sibling).

Dense-as-oracle, this project's standing testing rule (see ``docs/plan.md``'s
"Next steps" item 1 and ``tests/sparse/preconditioners/test_ic0.py``'s own
parity tests for the established pattern).
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations import (
    JacobiPreconditioner as DenseJacobiPreconditioner,
)
from torchalg.sparse.preconditioners.jacobi import JacobiPreconditioner


class TestJacobiPreconditionerSparseDenseParity:
    """The sparse ``JacobiPreconditioner``'s ``apply()`` matches the dense sibling's."""

    def test_matches_dense_on_poisson_1d(self, poisson_1d_dense: torch.Tensor) -> None:
        torch.manual_seed(42)
        residual = torch.randn(poisson_1d_dense.shape[0], dtype=poisson_1d_dense.dtype)

        dense_result = DenseJacobiPreconditioner().setup(poisson_1d_dense).apply(residual)
        sparse_result = (
            JacobiPreconditioner().setup(poisson_1d_dense.to_sparse_csr()).apply(residual)
        )

        torch.testing.assert_close(sparse_result, dense_result)

    def test_matches_dense_on_poisson_2d(self, poisson_2d_dense: torch.Tensor) -> None:
        torch.manual_seed(42)
        residual = torch.randn(poisson_2d_dense.shape[0], dtype=poisson_2d_dense.dtype)

        dense_result = DenseJacobiPreconditioner().setup(poisson_2d_dense).apply(residual)
        sparse_result = (
            JacobiPreconditioner().setup(poisson_2d_dense.to_sparse_csr()).apply(residual)
        )

        torch.testing.assert_close(sparse_result, dense_result)


@pytest.fixture
def diagonal_matrix(torch_dtype: torch.dtype) -> torch.Tensor:
    """4x4 diagonal SPD matrix matching ``residual_block``'s row count."""
    return torch.diag(torch.tensor([4.0, 3.0, 2.0, 2.0], dtype=torch_dtype))


class TestJacobiPreconditionerResidualBlock:
    """An ``(n, k)`` residual block matches the dense sibling column by column."""

    def test_matches_dense_on_block(
        self, diagonal_matrix: torch.Tensor, residual_block: torch.Tensor
    ) -> None:
        dense_result = DenseJacobiPreconditioner().setup(diagonal_matrix).apply(residual_block)
        sparse_result = (
            JacobiPreconditioner().setup(diagonal_matrix.to_sparse_csr()).apply(residual_block)
        )

        expected = residual_block / torch.diagonal(diagonal_matrix).unsqueeze(-1)
        torch.testing.assert_close(dense_result, expected, rtol=1e-14, atol=0.0)
        torch.testing.assert_close(sparse_result, dense_result, rtol=0.0, atol=0.0)


class TestJacobiPreconditionerIsNnModule:
    """Mirrors the dense sibling's buffer-registration regression tests
    (``tests/solver/preconditioners/implementations/test_jacobi.py::TestJacobiIsNnModule``) -
    the payoff (``.to(dtype=...)`` propagation) is identical regardless of
    which kernel built ``inv_diag``.
    """

    def test_inv_diag_is_registered_buffer(self, poisson_1d_csr: torch.Tensor) -> None:
        precond = JacobiPreconditioner()
        precond.setup(poisson_1d_csr)
        buffer_names = dict(precond.named_buffers())
        assert "inv_diag" in buffer_names
        assert buffer_names["inv_diag"] is precond.inv_diag

    def test_to_dtype_moves_the_buffer(self, poisson_1d_csr: torch.Tensor) -> None:
        precond = JacobiPreconditioner()
        precond.setup(poisson_1d_csr)
        assert precond.inv_diag.dtype == torch.float64

        precond = precond.to(dtype=torch.float32)

        assert precond.inv_diag.dtype == torch.float32
        r = torch.ones(poisson_1d_csr.shape[0], dtype=torch.float32)
        z = precond.apply(r)
        assert z.dtype == torch.float32
