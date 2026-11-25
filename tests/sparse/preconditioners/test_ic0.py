"""Tests for ``torchalg.sparse.ic0`` (``_lookup``/``_build_lookup``, ``sparse_ic0``, ``IC0Preconditioner``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from torchalg.preconditioners.implementations import IC0Preconditioner as DenseIC0Preconditioner
from torchalg.preconditioners.implementations._masked_factorization import dense_ic0
from torchalg.sparse.kernels.triangular import prepare_triangular_solve
from torchalg.sparse.preconditioners import ic0 as ic0_module
from torchalg.sparse.preconditioners._ic0_pattern import IC0Pattern, analyze_ic0
from torchalg.sparse.preconditioners.ic0 import IC0Preconditioner, sparse_ic0

if TYPE_CHECKING:
    from collections.abc import Callable


def _assert_same_factor(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual.crow_indices().long(), expected.crow_indices().long())
    assert torch.equal(actual.col_indices().long(), expected.col_indices().long())
    assert torch.equal(actual.values(), expected.values())


class TestSparseIC0:
    """``sparse_ic0`` matches ``dense_ic0`` (the dense algorithm as the oracle)."""

    def test_matches_dense_ic0_poisson_1d(
        self, poisson_1d_dense: torch.Tensor, poisson_1d_csr: torch.Tensor
    ) -> None:
        expected = dense_ic0(poisson_1d_dense)
        actual = sparse_ic0(poisson_1d_csr).to_dense()
        torch.testing.assert_close(actual, expected)

    def test_matches_dense_ic0_poisson_2d(
        self, poisson_2d_dense: torch.Tensor, poisson_2d_csr: torch.Tensor
    ) -> None:
        expected = dense_ic0(poisson_2d_dense)
        actual = sparse_ic0(poisson_2d_csr).to_dense()
        torch.testing.assert_close(actual, expected)

    def test_matches_dense_ic0_larger_anisotropic_grid(
        self, anisotropic_2d_factory: Callable[[int, float], torch.Tensor]
    ) -> None:
        dense = anisotropic_2d_factory(5, 0.3)
        sparse = dense.to_sparse_csr()
        expected = dense_ic0(dense)
        actual = sparse_ic0(sparse).to_dense()
        torch.testing.assert_close(actual, expected)

    def test_matches_dense_ic0_large_anisotropic_grid(
        self, anisotropic_2d_factory: Callable[[int, float], torch.Tensor]
    ) -> None:
        """64x64 grid: impractically slow under the old per-entry Python loop.

        Concrete evidence the level/degree-position batching fix actually
        worked - correctness on a matrix large enough that a per-entry loop
        (``nnz``-bounded, thousands of scalar tensor ops) would be far
        slower than this now-batched (``num_levels * max_degree``-bounded)
        version.
        """
        dense = anisotropic_2d_factory(8, 1.0)
        sparse = dense.to_sparse_csr()
        expected = dense_ic0(dense)
        actual = sparse_ic0(sparse).to_dense()
        torch.testing.assert_close(actual, expected)

    def test_unfound_gather_does_not_leak_an_infinite_value_as_nan(
        self, infinite_corner_csr: torch.Tensor, torch_dtype: torch.dtype
    ) -> None:
        """The unfound ``(2, 1)`` position indexes the infinite ``a00``; masking keeps it out of the dot."""
        inf = float("inf")
        expected = torch.tensor([inf, 2.0, 3.0, 4.0, 1.0, 1.0, 5.0], dtype=torch_dtype)

        assert torch.equal(sparse_ic0(infinite_corner_csr).values(), expected)

    def test_raises_on_non_spd_breakdown(self, torch_dtype: torch.dtype) -> None:
        dense = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch_dtype)
        with pytest.raises(ValueError, match="breakdown"):
            dense_ic0(dense)
        with pytest.raises(ValueError, match="breakdown"):
            sparse_ic0(dense.to_sparse_csr())

    def test_breakdown_names_the_first_failing_pivot(
        self, two_breakdown_tridiagonal_csr: torch.Tensor
    ) -> None:
        """Later pivots are nan or negative too; the error must still report row 2."""
        with pytest.raises(ValueError, match=r"breakdown at pivot 2: diagonal value -"):
            sparse_ic0(two_breakdown_tridiagonal_csr)

    def test_failed_setup_after_a_later_level_breakdown_keeps_previous_state(
        self,
        poisson_2d_csr: torch.Tensor,
        two_breakdown_tridiagonal_csr: torch.Tensor,
    ) -> None:
        precond = IC0Preconditioner().setup(poisson_2d_csr)
        previous = (precond._operator, precond._solve_cache, precond._pattern)

        with pytest.raises(ValueError, match="breakdown at pivot 2"):
            precond.setup(two_breakdown_tridiagonal_csr)

        assert all(
            now is before
            for now, before in zip(
                (precond._operator, precond._solve_cache, precond._pattern), previous, strict=True
            )
        )

    def test_factorization_leaves_the_input_values_untouched(
        self, poisson_2d_csr: torch.Tensor
    ) -> None:
        before = poisson_2d_csr.values().clone()

        sparse_ic0(poisson_2d_csr)

        assert torch.equal(poisson_2d_csr.values(), before)

    def test_rejects_non_csr_layout(self, poisson_1d_dense: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="sparse CSR"):
            sparse_ic0(poisson_1d_dense)

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
        factor = sparse_ic0(matrix)
        factor.values().sum().backward()
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()


class TestIC0PreconditionerSparseDenseParity:
    """The sparse ``IC0Preconditioner``'s ``apply()`` matches the dense sibling's.

    Dense-as-oracle, this project's standing testing rule - replaces the
    deleted ``test_ic0_sparse_dispatch.py`` (which asserted one class's two
    branches agreed; now there are two classes).
    """

    def test_matches_dense_on_poisson_1d(
        self, poisson_1d_dense: torch.Tensor, test_seed: int
    ) -> None:
        generator = torch.Generator().manual_seed(test_seed)
        residual = torch.randn(
            poisson_1d_dense.shape[0], dtype=poisson_1d_dense.dtype, generator=generator
        )

        dense_precond = DenseIC0Preconditioner()
        dense_precond.setup(poisson_1d_dense)
        sparse_precond = IC0Preconditioner()
        sparse_precond.setup(poisson_1d_dense.to_sparse_csr())

        torch.testing.assert_close(sparse_precond.apply(residual), dense_precond.apply(residual))

    def test_matches_dense_on_poisson_2d(
        self, poisson_2d_dense: torch.Tensor, test_seed: int
    ) -> None:
        generator = torch.Generator().manual_seed(test_seed)
        residual = torch.randn(
            poisson_2d_dense.shape[0], dtype=poisson_2d_dense.dtype, generator=generator
        )

        dense_precond = DenseIC0Preconditioner()
        dense_precond.setup(poisson_2d_dense)
        sparse_precond = IC0Preconditioner()
        sparse_precond.setup(poisson_2d_dense.to_sparse_csr())

        torch.testing.assert_close(sparse_precond.apply(residual), dense_precond.apply(residual))

    def test_batched_apply_matches_dense(
        self, poisson_2d_dense: torch.Tensor, test_seed: int
    ) -> None:
        """A matrix RHS is handled by the native batched triangular solve."""
        generator = torch.Generator().manual_seed(test_seed)
        residual = torch.randn(
            poisson_2d_dense.shape[0], 3, dtype=poisson_2d_dense.dtype, generator=generator
        )
        dense_precond = DenseIC0Preconditioner().setup(poisson_2d_dense)
        sparse_precond = IC0Preconditioner().setup(poisson_2d_dense.to_sparse_csr())

        torch.testing.assert_close(sparse_precond.apply(residual), dense_precond.apply(residual))


class TestIC0PreconditionerPatternReuse:
    """The instance keeps its pattern and re-analyzes only when the key changes."""

    @pytest.fixture
    def analysis_calls(self, monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
        """Matrices handed to ``analyze_ic0`` through ``ic0``'s own namespace."""
        calls: list[torch.Tensor] = []

        def spy(matrix: torch.Tensor) -> IC0Pattern:
            calls.append(matrix)
            return analyze_ic0(matrix)

        monkeypatch.setattr(ic0_module, "analyze_ic0", spy)
        return calls

    def test_second_setup_with_same_structure_skips_analysis(
        self,
        poisson_2d_csr: torch.Tensor,
        poisson_2d_rescaled_csr: torch.Tensor,
        analysis_calls: list[torch.Tensor],
    ) -> None:
        precond = IC0Preconditioner().setup(poisson_2d_csr)
        precond.setup(poisson_2d_rescaled_csr)

        assert len(analysis_calls) == 1

    @pytest.mark.parametrize(
        "changed", ["poisson_2d_extra_entry_csr", "poisson_2d_explicit_zero_csr"]
    )
    def test_changed_structure_or_mask_re_analyzes(
        self,
        changed: str,
        poisson_2d_csr: torch.Tensor,
        analysis_calls: list[torch.Tensor],
        request: pytest.FixtureRequest,
    ) -> None:
        precond = IC0Preconditioner().setup(poisson_2d_csr)
        precond.setup(request.getfixturevalue(changed))

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
        reused = IC0Preconditioner().setup(poisson_2d_csr).setup(poisson_2d_rescaled_csr)
        fresh = IC0Preconditioner().setup(poisson_2d_rescaled_csr)

        assert torch.equal(reused.apply(residual), fresh.apply(residual))

    def test_in_place_index_mutation_re_analyzes_and_equals_a_fresh_factor(
        self,
        tridiagonal_csr: torch.Tensor,
        redirect_last_row_entry: Callable[[torch.Tensor], None],
        analysis_calls: list[torch.Tensor],
    ) -> None:
        precond = IC0Preconditioner().setup(tridiagonal_csr)
        redirect_last_row_entry(tridiagonal_csr)
        precond.setup(tridiagonal_csr)

        fresh = IC0Preconditioner().setup(tridiagonal_csr)
        assert len(analysis_calls) == 3
        _assert_same_factor(precond._operator, fresh._operator)

    def test_int32_matrix_setup_reuses_the_pattern(
        self,
        poisson_2d_int32_csr: torch.Tensor,
        poisson_2d_rescaled_int32_csr: torch.Tensor,
        analysis_calls: list[torch.Tensor],
    ) -> None:
        precond = IC0Preconditioner().setup(poisson_2d_int32_csr)
        precond.setup(poisson_2d_rescaled_int32_csr)

        assert len(analysis_calls) == 1
        _assert_same_factor(
            precond._operator, IC0Preconditioner().setup(poisson_2d_rescaled_int32_csr)._operator
        )

    def test_alternating_index_dtypes_re_analyze_each_time_and_stay_correct(
        self,
        poisson_2d_csr: torch.Tensor,
        poisson_2d_int32_csr: torch.Tensor,
        analysis_calls: list[torch.Tensor],
    ) -> None:
        precond = IC0Preconditioner().setup(poisson_2d_csr)
        precond.setup(poisson_2d_int32_csr)
        precond.setup(poisson_2d_csr)

        assert len(analysis_calls) == 3
        torch.testing.assert_close(
            precond._operator.to_dense(),
            IC0Preconditioner().setup(poisson_2d_csr)._operator.to_dense(),
        )

    def test_single_entry_matrix_setup(self, single_entry_csr: torch.Tensor) -> None:
        precond = IC0Preconditioner().setup(single_entry_csr)

        torch.testing.assert_close(
            precond.apply(torch.ones(1, dtype=single_entry_csr.dtype)),
            torch.tensor([0.25], dtype=single_entry_csr.dtype),
        )

    def test_diagonal_only_matrix_setup_inverts_the_diagonal(
        self, diagonal_csr: torch.Tensor
    ) -> None:
        residual = torch.ones(3, dtype=diagonal_csr.dtype)

        applied = IC0Preconditioner().setup(diagonal_csr).apply(residual)

        torch.testing.assert_close(applied, residual / diagonal_csr.values())

    def test_explicit_zero_diagonal_in_the_first_matrix_raises_breakdown(
        self, explicit_zero_diagonal_csr: torch.Tensor
    ) -> None:
        with pytest.raises(ValueError, match="breakdown"):
            IC0Preconditioner().setup(explicit_zero_diagonal_csr)

    @pytest.mark.parametrize("failing", ["indefinite_csr", "poisson_2d_dense"])
    def test_failed_setup_preserves_previous_complete_state(
        self,
        failing: str,
        poisson_2d_csr: torch.Tensor,
        request: pytest.FixtureRequest,
    ) -> None:
        residual = torch.ones(poisson_2d_csr.shape[0], dtype=poisson_2d_csr.dtype)
        precond = IC0Preconditioner().setup(poisson_2d_csr)
        expected = precond.apply(residual)
        previous = (precond._operator, precond._solve_cache, precond._pattern)

        with pytest.raises(ValueError, match="breakdown|sparse CSR"):
            precond.setup(request.getfixturevalue(failing))

        assert all(
            now is before
            for now, before in zip(
                (precond._operator, precond._solve_cache, precond._pattern), previous, strict=True
            )
        )
        assert torch.equal(precond.apply(residual), expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="needs a CUDA GPU (ROCm's CSR triangular solve faults)",
)
def test_moving_to_gpu_re_resolves_the_triangular_backend(poisson_2d_csr: torch.Tensor) -> None:
    """A CPU-built cache moved to the GPU uses the GPU's registered solver, not the CPU's."""
    residual = torch.linspace(1.0, 2.0, poisson_2d_csr.shape[0], dtype=poisson_2d_csr.dtype)
    precond = IC0Preconditioner().setup(poisson_2d_csr)
    expected = precond.apply(residual)

    moved = precond.to("cuda")
    first = moved.apply(residual.cuda())
    second = moved.apply(residual.cuda())

    expected_solver = type(prepare_triangular_solve(poisson_2d_csr.cuda(), "forward"))
    assert isinstance(moved._solve_cache.forward_solve, expected_solver)
    torch.testing.assert_close(first.cpu(), expected)
    torch.testing.assert_close(second.cpu(), expected)
