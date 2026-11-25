"""Tests for ``ILU0Pattern``/``analyze_ilu0`` and pattern reuse in ``sparse_ilu0``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from torchalg.preconditioners.implementations._masked_factorization import dense_ilu0
from torchalg.sparse.preconditioners import _ilu0_pattern
from torchalg.sparse.preconditioners._ilu0_pattern import ILU0Pattern, analyze_ilu0
from torchalg.sparse.preconditioners._triangular import build_sparse_lu_solve_cache
from torchalg.sparse.preconditioners.ilu import ILUPreconditioner, sparse_ilu0

if TYPE_CHECKING:
    from collections.abc import Callable


def _assert_same_csr(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual.crow_indices().long(), expected.crow_indices().long())
    assert torch.equal(actual.col_indices().long(), expected.col_indices().long())
    assert torch.equal(actual.values(), expected.values())


class TestPatternReuse:
    """A pattern changes only where the work happens, never the factor."""

    def test_explicit_pattern_is_bit_identical_to_internal_analysis(
        self, poisson_2d_csr: torch.Tensor
    ) -> None:
        _assert_same_csr(
            sparse_ilu0(poisson_2d_csr, analyze_ilu0(poisson_2d_csr)), sparse_ilu0(poisson_2d_csr)
        )

    def test_pattern_from_old_values_factors_new_values_bit_identically(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_rescaled_csr: torch.Tensor
    ) -> None:
        pattern = analyze_ilu0(poisson_2d_csr)
        _assert_same_csr(
            sparse_ilu0(poisson_2d_rescaled_csr, pattern), sparse_ilu0(poisson_2d_rescaled_csr)
        )

    def test_asymmetric_matrix_with_reused_pattern_matches_dense_oracle(
        self, asymmetric_csr: torch.Tensor, asymmetric_rescaled_csr: torch.Tensor
    ) -> None:
        pattern = analyze_ilu0(asymmetric_csr)
        expected = dense_ilu0(asymmetric_rescaled_csr.to_dense())
        actual = sparse_ilu0(asymmetric_rescaled_csr, pattern).to_dense()
        torch.testing.assert_close(actual, expected)

    def test_mismatched_pattern_is_rejected(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_extra_entry_csr: torch.Tensor
    ) -> None:
        with pytest.raises(ValueError, match="pattern"):
            sparse_ilu0(poisson_2d_extra_entry_csr, analyze_ilu0(poisson_2d_csr))

    def test_analysis_rejects_non_csr_layout(self, poisson_2d_dense: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="sparse CSR"):
            analyze_ilu0(poisson_2d_dense)

    @pytest.fixture
    def lookup_calls(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        """Batch size of every ``lookup`` call made through the pattern module."""
        calls: list[int] = []
        real_lookup = _ilu0_pattern.lookup

        def counting_lookup(
            keys: torch.Tensor,
            values: torch.Tensor,
            query_rows: torch.Tensor,
            query_cols: torch.Tensor,
            n: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append(query_rows.numel())
            return real_lookup(keys, values, query_rows, query_cols, n)

        monkeypatch.setattr(_ilu0_pattern, "lookup", counting_lookup)
        return calls

    def test_numeric_phase_makes_no_lookup_calls(
        self, poisson_2d_csr: torch.Tensor, lookup_calls: list[int]
    ) -> None:
        """Pattern positions are resolved once in analysis; factorizing only gathers."""
        pattern = analyze_ilu0(poisson_2d_csr)
        analysis_calls = len(lookup_calls)

        sparse_ilu0(poisson_2d_csr, pattern)

        assert analysis_calls > 0
        assert len(lookup_calls) == analysis_calls

    def test_analysis_lookups_are_one_per_step_with_a_gather(
        self, poisson_2d_csr: torch.Tensor, lookup_calls: list[int]
    ) -> None:
        pattern = analyze_ilu0(poisson_2d_csr)

        gathers = sum(step.gather is not None for step in pattern.steps)
        assert len(lookup_calls) == gathers
        assert gathers < poisson_2d_csr._nnz()

    def test_differentiable_through_values_with_a_reused_pattern(
        self, poisson_2d_csr: torch.Tensor
    ) -> None:
        pattern = analyze_ilu0(poisson_2d_csr)
        values = poisson_2d_csr.values().clone().requires_grad_(True)
        matrix = torch.sparse_csr_tensor(
            poisson_2d_csr.crow_indices(),
            poisson_2d_csr.col_indices(),
            values,
            size=poisson_2d_csr.shape,
            check_invariants=False,
        )
        factor = sparse_ilu0(matrix, pattern)
        pair = build_sparse_lu_solve_cache(factor, pattern)

        (pair.lower.values().sum() + pair.upper.values().sum()).backward()

        assert values.grad is not None
        assert torch.isfinite(values.grad).all()


class TestMatches:
    """The key is the stored structure: ILU(0) has no value mask."""

    def test_true_for_same_structure_and_new_values(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_rescaled_csr: torch.Tensor
    ) -> None:
        pattern = analyze_ilu0(poisson_2d_csr)
        assert pattern.matches(poisson_2d_csr)
        assert pattern.matches(poisson_2d_rescaled_csr)

    def test_false_after_in_place_index_mutation(
        self,
        tridiagonal_csr: torch.Tensor,
        redirect_last_row_entry: Callable[[torch.Tensor], None],
    ) -> None:
        """The key is a copy, so an in-place edit of the matrix's own indices is visible."""
        pattern = analyze_ilu0(tridiagonal_csr)
        redirect_last_row_entry(tridiagonal_csr)

        assert not pattern.matches(tridiagonal_csr)

    def test_single_entry_matrix(self, single_entry_csr: torch.Tensor) -> None:
        pattern = analyze_ilu0(single_entry_csr)

        assert pattern.matches(single_entry_csr)
        assert sparse_ilu0(single_entry_csr, pattern).values().item() == 4.0

    def test_diagonal_only_matrix_factor_is_the_matrix(self, diagonal_csr: torch.Tensor) -> None:
        factor = sparse_ilu0(diagonal_csr, analyze_ilu0(diagonal_csr))

        torch.testing.assert_close(factor.to_dense(), diagonal_csr.to_dense())

    def test_true_when_an_entry_is_stored_as_zero(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_explicit_zero_csr: torch.Tensor
    ) -> None:
        """A stored zero stays in ILU(0)'s pattern, so it is not a structure change."""
        assert analyze_ilu0(poisson_2d_csr).matches(poisson_2d_explicit_zero_csr)

    def test_false_for_different_structure_same_size(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_extra_entry_csr: torch.Tensor
    ) -> None:
        assert not analyze_ilu0(poisson_2d_csr).matches(poisson_2d_extra_entry_csr)

    def test_false_for_different_size(
        self, poisson_2d_csr: torch.Tensor, poisson_1d_csr: torch.Tensor
    ) -> None:
        assert not analyze_ilu0(poisson_2d_csr).matches(poisson_1d_csr)

    def test_false_for_non_csr_layout(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_dense: torch.Tensor
    ) -> None:
        assert not analyze_ilu0(poisson_2d_csr).matches(poisson_2d_dense)

    def test_false_for_other_index_dtypes(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_int32_csr: torch.Tensor
    ) -> None:
        assert not analyze_ilu0(poisson_2d_csr).matches(poisson_2d_int32_csr)
        assert not analyze_ilu0(poisson_2d_int32_csr).matches(poisson_2d_csr)

    def test_true_for_the_same_int32_matrix(self, poisson_2d_int32_csr: torch.Tensor) -> None:
        assert analyze_ilu0(poisson_2d_int32_csr).matches(poisson_2d_int32_csr)


class TestSolveCache:
    """``build_sparse_lu_solve_cache`` splits the combined factor into the explicit triangles."""

    def test_triangles_equal_the_value_based_split(self, poisson_2d_csr: torch.Tensor) -> None:
        pattern = analyze_ilu0(poisson_2d_csr)
        factor = sparse_ilu0(poisson_2d_csr, pattern)
        pair = build_sparse_lu_solve_cache(factor, pattern)
        lower, upper = pair.lower, pair.upper

        dense = factor.to_dense()
        expected_lower = torch.tril(dense, -1) + torch.eye(dense.shape[0], dtype=dense.dtype)
        assert torch.equal(lower.to_dense(), expected_lower)
        assert torch.equal(upper.to_dense(), torch.triu(dense))

    def test_lower_has_an_explicit_unit_diagonal_in_every_row(
        self, poisson_2d_csr: torch.Tensor
    ) -> None:
        pattern = analyze_ilu0(poisson_2d_csr)
        lower = build_sparse_lu_solve_cache(sparse_ilu0(poisson_2d_csr, pattern), pattern).lower
        row = torch.repeat_interleave(
            torch.arange(lower.shape[0]), lower.crow_indices().diff().long()
        )
        assert int((lower.col_indices().long() == row).sum()) == lower.shape[0]


STRUCTURAL_INDEX_FIELDS = (
    "factor_crow",
    "factor_col",
    "lower_crow",
    "lower_col",
    "lower_source",
    "upper_crow",
    "upper_col",
    "upper_source",
)
"""Pattern arrays that only index storage; stored at the narrowest width that fits."""


class TestStructuralIndexWidth:
    """Structural index arrays take 4 bytes per entry when they fit, 8 otherwise."""

    @pytest.mark.parametrize("field", STRUCTURAL_INDEX_FIELDS)
    def test_int32_when_indices_fit(self, narrow_ilu0_pattern: ILU0Pattern, field: str) -> None:
        assert getattr(narrow_ilu0_pattern, field).dtype == torch.int32

    @pytest.mark.parametrize("field", STRUCTURAL_INDEX_FIELDS)
    def test_int64_when_indices_exceed_the_limit(
        self, wide_ilu0_pattern: ILU0Pattern, field: str
    ) -> None:
        assert getattr(wide_ilu0_pattern, field).dtype == torch.int64

    def test_structural_bytes_are_halved(
        self, narrow_ilu0_pattern: ILU0Pattern, wide_ilu0_pattern: ILU0Pattern
    ) -> None:
        def nbytes(pattern: ILU0Pattern) -> int:
            return sum(
                getattr(pattern, field).numel() * getattr(pattern, field).element_size()
                for field in STRUCTURAL_INDEX_FIELDS
            )

        assert 2 * nbytes(narrow_ilu0_pattern) == nbytes(wide_ilu0_pattern)

    def test_per_step_gather_indices_stay_int64(self, narrow_ilu0_pattern: ILU0Pattern) -> None:
        """Advanced-index gathers are slower with int32 indices, so the hot arrays keep int64."""
        for step in narrow_ilu0_pattern.steps:
            assert step.flat_pos.dtype == step.diag_pos_of_j.dtype == torch.int64
            if step.gather is not None:
                assert step.gather.pos_ik.dtype == step.gather.pos_kj.dtype == torch.int64

    def test_reuse_key_owns_copies_of_the_matrix_index_tensors(
        self, poisson_2d_csr: torch.Tensor, narrow_ilu0_pattern: ILU0Pattern
    ) -> None:
        assert narrow_ilu0_pattern.crow.data_ptr() != poisson_2d_csr.crow_indices().data_ptr()
        assert narrow_ilu0_pattern.col.data_ptr() != poisson_2d_csr.col_indices().data_ptr()
        assert narrow_ilu0_pattern.crow.dtype == poisson_2d_csr.crow_indices().dtype
        assert torch.equal(narrow_ilu0_pattern.crow, poisson_2d_csr.crow_indices())
        assert torch.equal(narrow_ilu0_pattern.col, poisson_2d_csr.col_indices())

    def test_factor_takes_the_pattern_index_width(
        self,
        poisson_2d_csr: torch.Tensor,
        narrow_ilu0_pattern: ILU0Pattern,
        wide_ilu0_pattern: ILU0Pattern,
    ) -> None:
        narrow = sparse_ilu0(poisson_2d_csr, narrow_ilu0_pattern)
        wide = sparse_ilu0(poisson_2d_csr, wide_ilu0_pattern)
        assert narrow.col_indices().dtype == narrow.crow_indices().dtype == torch.int32
        assert wide.col_indices().dtype == wide.crow_indices().dtype == torch.int64

    def test_factor_is_bit_identical_to_the_int64_plan(
        self,
        poisson_2d_csr: torch.Tensor,
        narrow_ilu0_pattern: ILU0Pattern,
        wide_ilu0_pattern: ILU0Pattern,
    ) -> None:
        _assert_same_csr(
            sparse_ilu0(poisson_2d_csr, narrow_ilu0_pattern),
            sparse_ilu0(poisson_2d_csr, wide_ilu0_pattern),
        )

    def test_application_is_bit_identical_to_the_int64_plan(
        self,
        poisson_2d_csr: torch.Tensor,
        ramp_rhs: torch.Tensor,
        narrow_ilu0_pattern: ILU0Pattern,
        wide_ilu0_pattern: ILU0Pattern,
    ) -> None:
        def apply(pattern: ILU0Pattern) -> torch.Tensor:
            factor = sparse_ilu0(poisson_2d_csr, pattern)
            return build_sparse_lu_solve_cache(factor, pattern).solve(ramp_rhs)

        assert torch.equal(apply(narrow_ilu0_pattern), apply(wide_ilu0_pattern))

    def test_level_scheduled_solve_on_the_narrow_pair_matches_the_int64_pair(
        self,
        poisson_2d_csr: torch.Tensor,
        ramp_rhs: torch.Tensor,
        narrow_ilu0_pattern: ILU0Pattern,
        wide_ilu0_pattern: ILU0Pattern,
    ) -> None:
        """The differentiable path (``requires_grad`` target) reads the pair's index arrays."""
        target = ramp_rhs.clone().requires_grad_(True)

        def apply(pattern: ILU0Pattern) -> torch.Tensor:
            factor = sparse_ilu0(poisson_2d_csr, pattern)
            return build_sparse_lu_solve_cache(factor, pattern).solve(target)

        assert torch.equal(apply(narrow_ilu0_pattern), apply(wide_ilu0_pattern))


def test_preconditioner_state_uses_the_pattern_factors(poisson_2d_csr: torch.Tensor) -> None:
    """``ILUPreconditioner`` stores the factor and the solve pair the pattern split."""
    precond = ILUPreconditioner().setup(poisson_2d_csr)
    pattern = analyze_ilu0(poisson_2d_csr)
    _assert_same_csr(precond._operator, sparse_ilu0(poisson_2d_csr, pattern))
    assert precond._solve_cache.lower.col_indices().dtype == torch.int32
