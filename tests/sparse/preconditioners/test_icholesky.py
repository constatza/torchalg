"""Tests for ``torchalg.sparse.preconditioners.icholesky`` (sparse-CSR ICholesky sibling).

Dense-as-oracle, this project's standing testing rule: both preconditioners
are handed the *same* factor ``L`` (one sparse CSR, one its dense twin) and
must solve ``(L @ L.T) z = r`` to the same answer, regardless of which
triangular-solve machinery (``torch.cholesky_solve`` vs. the level-scheduled
sparse solve) computed it.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from torchalg.preconditioners.implementations import (
    ICholeskyPreconditioner as DenseICholeskyPreconditioner,
)
from torchalg.sparse.preconditioners.ic0 import sparse_ic0
from torchalg.sparse.preconditioners.icholesky import ICholeskyPreconditioner


@pytest.fixture
def differentiable_factor(torch_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a sparse lower factor and its leaf values for gradient assertions."""
    dense = torch.tensor([[2.0, 0.0], [1.0, 3.0]], dtype=torch_dtype)
    base = dense.to_sparse_csr()
    values = base.values().clone().requires_grad_(True)
    factor = torch.sparse_csr_tensor(
        base.crow_indices(),
        base.col_indices(),
        values,
        size=base.shape,
        check_invariants=False,
    )
    return factor, values


class TestICholeskyPreconditionerSparseDenseParity:
    """The sparse ``ICholeskyPreconditioner``'s ``apply()`` matches the dense sibling's."""

    def test_matches_dense_on_poisson_1d(self, poisson_1d_csr: torch.Tensor) -> None:
        torch.manual_seed(42)
        factor = sparse_ic0(poisson_1d_csr)
        residual = torch.randn(poisson_1d_csr.shape[0], dtype=poisson_1d_csr.values().dtype)

        dense_precond = DenseICholeskyPreconditioner()
        dense_precond.setup(factor.to_dense())
        sparse_precond = ICholeskyPreconditioner()
        sparse_precond.setup(factor)

        torch.testing.assert_close(sparse_precond.apply(residual), dense_precond.apply(residual))

    def test_matches_dense_on_poisson_2d(self, poisson_2d_csr: torch.Tensor) -> None:
        torch.manual_seed(42)
        factor = sparse_ic0(poisson_2d_csr)
        residual = torch.randn(poisson_2d_csr.shape[0], dtype=poisson_2d_csr.values().dtype)

        dense_precond = DenseICholeskyPreconditioner()
        dense_precond.setup(factor.to_dense())
        sparse_precond = ICholeskyPreconditioner()
        sparse_precond.setup(factor)

        torch.testing.assert_close(sparse_precond.apply(residual), dense_precond.apply(residual))


class TestICholeskyPreconditionerShapes:
    """Verify shape handling for batched residual input (mirrors the dense
    sibling's ``test_icholesky_preconditioner_shapes``)."""

    def test_batched_residual(self, poisson_1d_csr: torch.Tensor) -> None:
        factor = sparse_ic0(poisson_1d_csr)
        residual = torch.ones(poisson_1d_csr.shape[0], 2, dtype=poisson_1d_csr.values().dtype)

        precond = ICholeskyPreconditioner()
        precond.setup(factor)
        z = precond.apply(residual)

        assert z.shape == residual.shape

    def test_batched_residual_uses_two_triangular_solves(
        self,
        poisson_1d_csr: torch.Tensor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A matrix RHS is solved as one batch, not recursively by column."""
        import torchalg.sparse.preconditioners._triangular as triangular_module

        factor = sparse_ic0(poisson_1d_csr)
        residual = torch.ones(poisson_1d_csr.shape[0], 3, dtype=factor.dtype)
        precond = ICholeskyPreconditioner().setup(factor)
        counting_triangular_solve = Mock(wraps=triangular_module.triangular_solve)
        monkeypatch.setattr(triangular_module, "triangular_solve", counting_triangular_solve)

        precond.apply(residual)

        assert counting_triangular_solve.call_count == 2

    def test_apply_reuses_setup_schedule(
        self,
        poisson_1d_csr: torch.Tensor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Repeated application performs no schedule construction."""
        import torchalg.sparse.preconditioners._triangular as triangular_module

        factor = sparse_ic0(poisson_1d_csr)
        residual = torch.ones(poisson_1d_csr.shape[0], dtype=factor.dtype)
        precond = ICholeskyPreconditioner().setup(factor)

        def unexpected_schedule(*args: object, **kwargs: object) -> None:
            raise AssertionError("level_schedule must only run during setup")

        monkeypatch.setattr(triangular_module, "level_schedule", unexpected_schedule)

        precond.apply(residual)
        precond.apply(residual)

    def test_cached_state_follows_dtype_conversion(self, poisson_1d_csr: torch.Tensor) -> None:
        """The factor and all floating-point cached solve state convert together."""
        factor = sparse_ic0(poisson_1d_csr)
        precond = ICholeskyPreconditioner().setup(factor).to(dtype=torch.float32)
        residual = torch.ones(poisson_1d_csr.shape[0], dtype=torch.float32)

        actual = precond.apply(residual)
        expected = torch.cholesky_solve(
            residual.unsqueeze(1), precond._operator.to_dense()
        ).squeeze(1)

        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected)

    def test_setup_replaces_complete_cached_state(
        self,
        poisson_1d_csr: torch.Tensor,
        poisson_2d_csr: torch.Tensor,
    ) -> None:
        """A second setup atomically binds factors and schedules for the new shape."""
        first_factor = sparse_ic0(poisson_1d_csr)
        second_factor = sparse_ic0(poisson_2d_csr)
        precond = ICholeskyPreconditioner().setup(first_factor)
        first_cache = precond._solve_cache

        precond.setup(second_factor)
        residual = torch.ones(second_factor.shape[0], dtype=second_factor.dtype)

        assert precond._solve_cache is not first_cache
        assert precond._solve_cache.upper.shape == second_factor.shape
        assert precond.apply(residual).shape == residual.shape

    def test_failed_setup_preserves_previous_complete_state(
        self,
        poisson_1d_csr: torch.Tensor,
        poisson_1d_dense: torch.Tensor,
    ) -> None:
        """A failed rebind cannot expose a factor/cache mixture."""
        factor = sparse_ic0(poisson_1d_csr)
        residual = torch.ones(factor.shape[0], dtype=factor.dtype)
        precond = ICholeskyPreconditioner().setup(factor)
        expected = precond.apply(residual)
        previous_operator = precond._operator
        previous_cache = precond._solve_cache

        with pytest.raises(ValueError, match="sparse CSR"):
            precond.setup(poisson_1d_dense)

        assert precond._operator is previous_operator
        assert precond._solve_cache is previous_cache
        torch.testing.assert_close(precond.apply(residual), expected)

    def test_cached_state_preserves_autograd(
        self,
        differentiable_factor: tuple[torch.Tensor, torch.Tensor],
        torch_dtype: torch.dtype,
    ) -> None:
        """Setup-time transpose materialization does not detach factor values."""
        factor, values = differentiable_factor
        residual = torch.ones(factor.shape[0], dtype=torch_dtype, requires_grad=True)
        precond = ICholeskyPreconditioner().setup(factor)

        precond.apply(residual).sum().backward()

        assert values.grad is not None
        assert residual.grad is not None
