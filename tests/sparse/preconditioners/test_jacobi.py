"""Tests for ``torchalg.sparse.preconditioners.jacobi`` (sparse-CSR Jacobi sibling).

Dense-as-oracle, this project's standing testing rule (see ``docs/plan.md``'s
"Next steps" item 1 and ``tests/sparse/preconditioners/test_ic0.py``'s own
parity tests for the established pattern).
"""

from __future__ import annotations

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

        dense_result = DenseJacobiPreconditioner(poisson_1d_dense).apply(residual)
        sparse_result = JacobiPreconditioner(poisson_1d_dense.to_sparse_csr()).apply(residual)

        torch.testing.assert_close(sparse_result, dense_result)

    def test_matches_dense_on_poisson_2d(self, poisson_2d_dense: torch.Tensor) -> None:
        torch.manual_seed(42)
        residual = torch.randn(poisson_2d_dense.shape[0], dtype=poisson_2d_dense.dtype)

        dense_result = DenseJacobiPreconditioner(poisson_2d_dense).apply(residual)
        sparse_result = JacobiPreconditioner(poisson_2d_dense.to_sparse_csr()).apply(residual)

        torch.testing.assert_close(sparse_result, dense_result)


class TestJacobiPreconditionerIsNnModule:
    """Mirrors the dense sibling's buffer-registration regression tests
    (``tests/solver/preconditioners/implementations/test_jacobi.py::TestJacobiIsNnModule``) -
    the payoff (``.to(dtype=...)`` propagation) is identical regardless of
    which kernel built ``inv_diag``.
    """

    def test_inv_diag_is_registered_buffer(self, poisson_1d_csr: torch.Tensor) -> None:
        precond = JacobiPreconditioner(poisson_1d_csr)
        buffer_names = dict(precond.named_buffers())
        assert "inv_diag" in buffer_names
        assert buffer_names["inv_diag"] is precond.inv_diag

    def test_to_dtype_moves_the_buffer(self, poisson_1d_csr: torch.Tensor) -> None:
        precond = JacobiPreconditioner(poisson_1d_csr)
        assert precond.inv_diag.dtype == torch.float64

        precond = precond.to(dtype=torch.float32)

        assert precond.inv_diag.dtype == torch.float32
        r = torch.ones(poisson_1d_csr.shape[0], dtype=torch.float32)
        z = precond.apply(r)
        assert z.dtype == torch.float32
