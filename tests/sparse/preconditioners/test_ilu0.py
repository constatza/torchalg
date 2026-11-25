"""Tests for ``torchalg.sparse.preconditioners.ilu.sparse_ilu0``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

from torchalg.preconditioners.implementations import ILUPreconditioner as DenseILUPreconditioner
from torchalg.preconditioners.implementations._masked_factorization import dense_ilu0
from torchalg.sparse.kernels import triangular as triangular_module
from torchalg.sparse.preconditioners import _ilu0_pattern
from torchalg.sparse.preconditioners import ilu as ilu_module
from torchalg.sparse.preconditioners._ilu0_pattern import ILU0Pattern, analyze_ilu0
from torchalg.sparse.preconditioners.ilu import ILUPreconditioner, sparse_ilu0


def _assert_same_factor(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual.crow_indices().long(), expected.crow_indices().long())
    assert torch.equal(actual.col_indices().long(), expected.col_indices().long())
    assert torch.equal(actual.values(), expected.values())


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

    def test_unfound_gather_does_not_leak_an_infinite_value_as_nan(
        self, infinite_corner_csr: torch.Tensor, torch_dtype: torch.dtype
    ) -> None:
        """The unfound ``(1, 2)``, ``(1, 4)``, ``(2, 4)`` positions index the infinite ``a00``."""
        inf = float("inf")
        third = torch.tensor(3.0, dtype=torch_dtype) / torch.tensor(9.0, dtype=torch_dtype)
        expected = torch.tensor([inf, 4.0, 9.0, 16.0, 0.5, third.item(), 27.0], dtype=torch_dtype)

        assert torch.equal(sparse_ilu0(infinite_corner_csr).values(), expected)

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
        residual = torch.ones(poisson_1d_csr.shape[0], dtype=poisson_1d_csr.dtype)
        precond = ILUPreconditioner().setup(poisson_1d_csr)

        def unexpected_schedule(*args: object, **kwargs: object) -> None:
            raise AssertionError("level_schedule must only run during setup")

        monkeypatch.setattr(_ilu0_pattern, "level_schedule", unexpected_schedule)
        monkeypatch.setattr(triangular_module, "level_schedule", unexpected_schedule)

        precond.apply(residual)
        precond.apply(residual)


class TestILUPreconditionerPatternReuse:
    """The instance keeps its pattern and re-analyzes only when the stored structure changes."""

    @pytest.fixture
    def analysis_calls(self, monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
        """Matrices handed to ``analyze_ilu0`` through ``ilu``'s own namespace."""
        calls: list[torch.Tensor] = []

        def spy(matrix: torch.Tensor) -> ILU0Pattern:
            calls.append(matrix)
            return analyze_ilu0(matrix)

        monkeypatch.setattr(ilu_module, "analyze_ilu0", spy)
        return calls

    def test_second_setup_with_same_structure_skips_analysis(
        self,
        poisson_2d_csr: torch.Tensor,
        poisson_2d_rescaled_csr: torch.Tensor,
        analysis_calls: list[torch.Tensor],
    ) -> None:
        precond = ILUPreconditioner().setup(poisson_2d_csr)
        precond.setup(poisson_2d_rescaled_csr)

        assert len(analysis_calls) == 1

    def test_changed_structure_re_analyzes(
        self,
        poisson_2d_csr: torch.Tensor,
        poisson_2d_extra_entry_csr: torch.Tensor,
        analysis_calls: list[torch.Tensor],
    ) -> None:
        precond = ILUPreconditioner().setup(poisson_2d_csr)
        precond.setup(poisson_2d_extra_entry_csr)

        assert len(analysis_calls) == 2

    def test_reused_pattern_apply_equals_fresh_preconditioner(
        self,
        poisson_2d_csr: torch.Tensor,
        poisson_2d_rescaled_csr: torch.Tensor,
        torch_dtype: torch.dtype,
        test_seed: int,
    ) -> None:
        generator = torch.Generator().manual_seed(test_seed)
        residual = torch.randn(poisson_2d_csr.shape[0], dtype=torch_dtype, generator=generator)
        reused = ILUPreconditioner().setup(poisson_2d_csr).setup(poisson_2d_rescaled_csr)
        fresh = ILUPreconditioner().setup(poisson_2d_rescaled_csr)

        assert torch.equal(reused.apply(residual), fresh.apply(residual))

    def test_in_place_index_mutation_re_analyzes_and_equals_a_fresh_factor(
        self,
        tridiagonal_csr: torch.Tensor,
        redirect_last_row_entry: Callable[[torch.Tensor], None],
        analysis_calls: list[torch.Tensor],
    ) -> None:
        precond = ILUPreconditioner().setup(tridiagonal_csr)
        redirect_last_row_entry(tridiagonal_csr)
        precond.setup(tridiagonal_csr)

        fresh = ILUPreconditioner().setup(tridiagonal_csr)
        assert len(analysis_calls) == 3
        _assert_same_factor(precond._operator, fresh._operator)

    def test_int32_matrix_setup_reuses_the_pattern(
        self,
        poisson_2d_int32_csr: torch.Tensor,
        poisson_2d_rescaled_int32_csr: torch.Tensor,
        analysis_calls: list[torch.Tensor],
    ) -> None:
        precond = ILUPreconditioner().setup(poisson_2d_int32_csr)
        precond.setup(poisson_2d_rescaled_int32_csr)

        assert len(analysis_calls) == 1
        _assert_same_factor(
            precond._operator, ILUPreconditioner().setup(poisson_2d_rescaled_int32_csr)._operator
        )

    def test_alternating_index_dtypes_re_analyze_each_time_and_stay_correct(
        self,
        poisson_2d_csr: torch.Tensor,
        poisson_2d_int32_csr: torch.Tensor,
        analysis_calls: list[torch.Tensor],
    ) -> None:
        precond = ILUPreconditioner().setup(poisson_2d_csr)
        precond.setup(poisson_2d_int32_csr)
        precond.setup(poisson_2d_csr)

        assert len(analysis_calls) == 3
        torch.testing.assert_close(
            precond._operator.to_dense(),
            ILUPreconditioner().setup(poisson_2d_csr)._operator.to_dense(),
        )

    def test_single_entry_matrix_setup(self, single_entry_csr: torch.Tensor) -> None:
        applied = (
            ILUPreconditioner()
            .setup(single_entry_csr)
            .apply(torch.ones(1, dtype=single_entry_csr.dtype))
        )

        torch.testing.assert_close(applied, torch.tensor([0.25], dtype=single_entry_csr.dtype))

    def test_diagonal_only_matrix_setup_inverts_the_diagonal(
        self, diagonal_csr: torch.Tensor
    ) -> None:
        residual = torch.ones(3, dtype=diagonal_csr.dtype)

        applied = ILUPreconditioner().setup(diagonal_csr).apply(residual)

        torch.testing.assert_close(applied, residual / diagonal_csr.values())

    def test_non_csr_setup_preserves_previous_complete_state(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_dense: torch.Tensor
    ) -> None:
        residual = torch.ones(poisson_2d_csr.shape[0], dtype=poisson_2d_csr.dtype)
        precond = ILUPreconditioner().setup(poisson_2d_csr)
        expected = precond.apply(residual)
        previous = (precond._operator, precond._solve_cache, precond._pattern)

        with pytest.raises(ValueError, match="sparse CSR"):
            precond.setup(poisson_2d_dense)

        assert all(
            now is before
            for now, before in zip(
                (precond._operator, precond._solve_cache, precond._pattern), previous, strict=True
            )
        )
        assert torch.equal(precond.apply(residual), expected)

    def test_failure_after_factorization_preserves_previous_complete_state(
        self,
        poisson_2d_csr: torch.Tensor,
        poisson_2d_extra_entry_csr: torch.Tensor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """State is assigned only after the solve pair is built, so a late failure leaves none."""
        precond = ILUPreconditioner().setup(poisson_2d_csr)
        previous = (precond._operator, precond._solve_cache, precond._pattern)

        def failing_pair(*args: object, **kwargs: object) -> None:
            raise RuntimeError("late failure")

        monkeypatch.setattr(ilu_module, "build_sparse_lu_solve_cache", failing_pair)
        with pytest.raises(RuntimeError, match="late failure"):
            precond.setup(poisson_2d_extra_entry_csr)

        assert all(
            now is before
            for now, before in zip(
                (precond._operator, precond._solve_cache, precond._pattern), previous, strict=True
            )
        )
