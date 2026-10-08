"""Tests for ``torchalg.sparse.preconditioners.ilu.sparse_ilu0``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

from torchalg.preconditioners.implementations import ILUPreconditioner as DenseILUPreconditioner
from torchalg.preconditioners.implementations._masked_factorization import dense_ilu0
from torchalg.sparse.preconditioners.ilu import ILUPreconditioner, sparse_ilu0


class TestSparseILU0:
    """``sparse_ilu0`` matches ``dense_ilu0`` (the dense algorithm as the oracle)."""

    def test_matches_dense_ilu0_poisson_1d(
        self, poisson_1d_dense: torch.Tensor, poisson_1d_csr: torch.Tensor
    ) -> None:
        expected = dense_ilu0(poisson_1d_dense)
        actual = sparse_ilu0(poisson_1d_csr).to_dense()
        torch.testing.assert_close(actual, expected)

    def test_matches_dense_ilu0_poisson_2d(
        self, poisson_2d_dense: torch.Tensor, poisson_2d_csr: torch.Tensor
    ) -> None:
        expected = dense_ilu0(poisson_2d_dense)
        actual = sparse_ilu0(poisson_2d_csr).to_dense()
        torch.testing.assert_close(actual, expected)

    def test_matches_dense_ilu0_larger_anisotropic_grid(self, torch_dtype: torch.dtype) -> None:
        nx = 5
        stencil = 2.0 * torch.eye(nx, dtype=torch_dtype)
        stencil -= torch.diag(torch.ones(nx - 1, dtype=torch_dtype), 1)
        stencil -= torch.diag(torch.ones(nx - 1, dtype=torch_dtype), -1)
        eye = torch.eye(nx, dtype=torch_dtype)
        dense = torch.kron(eye, stencil) + 0.3 * torch.kron(stencil, eye)
        sparse = dense.to_sparse_csr()
        expected = dense_ilu0(dense)
        actual = sparse_ilu0(sparse).to_dense()
        torch.testing.assert_close(actual, expected)

    def test_matches_dense_ilu0_large_anisotropic_grid(
        self, anisotropic_2d_factory: Callable[[int, float], torch.Tensor]
    ) -> None:
        """64x64 grid: impractically slow under the old per-entry Python loop.

        Same rationale as ``sparse_ic0``'s equivalent test - concrete
        evidence the level/degree-position batching fix actually worked.
        """
        dense = anisotropic_2d_factory(8, 1.0)
        sparse = dense.to_sparse_csr()
        expected = dense_ilu0(dense)
        actual = sparse_ilu0(sparse).to_dense()
        torch.testing.assert_close(actual, expected)

    def test_matches_dense_ilu0_asymmetric_matrix(self, torch_dtype: torch.dtype) -> None:
        """A non-symmetric perturbation exercises the row/column lookup asymmetry.

        A symmetric test matrix never distinguishes the ``k`` lookup
        happening against a *fixed column, varying row* (ILU0's real
        asymmetric access pattern) from IC0's simpler same-row lookup - so
        this perturbs only the upper off-diagonal of a tridiagonal matrix.
        """
        n = 8
        dense = 2.0 * torch.eye(n, dtype=torch_dtype)
        dense -= torch.diag(torch.ones(n - 1, dtype=torch_dtype), 1)
        dense -= torch.diag(torch.ones(n - 1, dtype=torch_dtype), -1)
        dense += 0.3 * torch.diag(torch.ones(n - 1, dtype=torch_dtype), 1)
        sparse = dense.to_sparse_csr()

        expected = dense_ilu0(dense)
        actual = sparse_ilu0(sparse).to_dense()
        torch.testing.assert_close(actual, expected)

    def test_rejects_non_csr_layout(self, poisson_1d_dense: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="sparse CSR"):
            sparse_ilu0(poisson_1d_dense)

    def test_differentiable_through_values(self, poisson_1d_dense: torch.Tensor) -> None:
        matrix = poisson_1d_dense.to_sparse_csr()
        values = matrix.values().clone().requires_grad_(True)
        matrix = torch.sparse_csr_tensor(
            matrix.crow_indices(),
            matrix.col_indices(),
            values,
            size=matrix.shape,
            check_invariants=False,
        )
        factor = sparse_ilu0(matrix)
        factor.values().sum().backward()
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()


class TestILUPreconditionerSparseDenseParity:
    """The sparse ``ILUPreconditioner``'s ``apply()`` matches the dense sibling's.

    Dense-as-oracle, this project's standing testing rule - replaces the
    deleted ``test_ilu_sparse_dispatch.py`` (which asserted one class's two
    branches agreed; now there are two classes).
    """

    def test_matches_dense_on_poisson_1d(self, poisson_1d_dense: torch.Tensor) -> None:
        torch.manual_seed(42)
        residual = torch.randn(poisson_1d_dense.shape[0], dtype=poisson_1d_dense.dtype)

        dense_precond = DenseILUPreconditioner()
        dense_precond.setup(poisson_1d_dense)
        sparse_precond = ILUPreconditioner()
        sparse_precond.setup(poisson_1d_dense.to_sparse_csr())

        torch.testing.assert_close(sparse_precond.apply(residual), dense_precond.apply(residual))

    def test_matches_dense_on_poisson_2d(self, poisson_2d_dense: torch.Tensor) -> None:
        torch.manual_seed(42)
        residual = torch.randn(poisson_2d_dense.shape[0], dtype=poisson_2d_dense.dtype)

        dense_precond = DenseILUPreconditioner()
        dense_precond.setup(poisson_2d_dense)
        sparse_precond = ILUPreconditioner()
        sparse_precond.setup(poisson_2d_dense.to_sparse_csr())

        torch.testing.assert_close(sparse_precond.apply(residual), dense_precond.apply(residual))

    def test_batched_apply_matches_dense(self, poisson_2d_dense: torch.Tensor) -> None:
        """A matrix RHS uses the native batched triangular-solve path."""
        torch.manual_seed(42)
        residual = torch.randn(poisson_2d_dense.shape[0], 3, dtype=poisson_2d_dense.dtype)
        dense_precond = DenseILUPreconditioner().setup(poisson_2d_dense)
        sparse_precond = ILUPreconditioner().setup(poisson_2d_dense.to_sparse_csr())

        torch.testing.assert_close(sparse_precond.apply(residual), dense_precond.apply(residual))

    def test_apply_reuses_setup_schedule(
        self,
        poisson_1d_csr: torch.Tensor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Repeated application performs no factor splitting or scheduling."""
        import torchalg.sparse.preconditioners.ilu as ilu_module

        residual = torch.ones(poisson_1d_csr.shape[0], dtype=poisson_1d_csr.dtype)
        precond = ILUPreconditioner().setup(poisson_1d_csr)

        def unexpected_schedule(*args: object, **kwargs: object) -> None:
            raise AssertionError("level_schedule must only run during setup")

        monkeypatch.setattr(ilu_module, "level_schedule", unexpected_schedule)

        precond.apply(residual)
        precond.apply(residual)
