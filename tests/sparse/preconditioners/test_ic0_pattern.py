"""Tests for ``IC0Pattern``/``analyze_ic0`` and pattern reuse in ``sparse_ic0``."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest
import torch

from torchalg.sparse.preconditioners import _ic0_pattern
from torchalg.sparse.preconditioners._ic0_pattern import IC0Pattern, analyze_ic0
from torchalg.sparse.preconditioners._triangular import build_sparse_cholesky_solve_cache
from torchalg.sparse.preconditioners.ic0 import _factorize, sparse_ic0

if TYPE_CHECKING:
    from collections.abc import Callable


def _assert_same_csr(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual.crow_indices(), expected.crow_indices())
    assert torch.equal(actual.col_indices(), expected.col_indices())
    assert torch.equal(actual.values(), expected.values())


class TestPatternReuse:
    """A pattern changes only where the work happens, never the factor."""

    def test_explicit_pattern_is_bit_identical_to_internal_analysis(
        self, poisson_2d_csr: torch.Tensor
    ) -> None:
        _assert_same_csr(
            sparse_ic0(poisson_2d_csr, analyze_ic0(poisson_2d_csr)), sparse_ic0(poisson_2d_csr)
        )

    def test_pattern_from_old_values_factors_new_values_bit_identically(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_rescaled_csr: torch.Tensor
    ) -> None:
        pattern = analyze_ic0(poisson_2d_csr)
        _assert_same_csr(
            sparse_ic0(poisson_2d_rescaled_csr, pattern), sparse_ic0(poisson_2d_rescaled_csr)
        )

    def test_mismatched_pattern_is_rejected(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_extra_entry_csr: torch.Tensor
    ) -> None:
        with pytest.raises(ValueError, match="pattern"):
            sparse_ic0(poisson_2d_extra_entry_csr, analyze_ic0(poisson_2d_csr))

    def test_numeric_phase_makes_no_lookup_calls(
        self, poisson_2d_csr: torch.Tensor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pattern positions are resolved once in analysis; factorizing only gathers."""
        calls = 0
        real_lookup = _ic0_pattern.lookup

        def counting_lookup(
            keys: torch.Tensor,
            values: torch.Tensor,
            query_rows: torch.Tensor,
            query_cols: torch.Tensor,
            n: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal calls
            calls += 1
            return real_lookup(keys, values, query_rows, query_cols, n)

        monkeypatch.setattr(_ic0_pattern, "lookup", counting_lookup)
        pattern = analyze_ic0(poisson_2d_csr)
        analysis_calls = calls

        sparse_ic0(poisson_2d_csr, pattern)

        assert analysis_calls > 0
        assert calls == analysis_calls

    def test_analysis_lookups_bounded_by_levels_times_max_degree(
        self, poisson_2d_csr: torch.Tensor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One lookup per (level, degree position) with ``d > 0``, never one per entry."""
        calls = 0
        real_lookup = _ic0_pattern.lookup

        def counting_lookup(
            keys: torch.Tensor,
            values: torch.Tensor,
            query_rows: torch.Tensor,
            query_cols: torch.Tensor,
            n: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal calls
            calls += 1
            return real_lookup(keys, values, query_rows, query_cols, n)

        monkeypatch.setattr(_ic0_pattern, "lookup", counting_lookup)
        pattern = analyze_ic0(poisson_2d_csr)

        bound = sum(max(len(level.steps) - 1, 0) for level in pattern.levels)
        assert calls == bound
        assert calls < poisson_2d_csr._nnz()


class TestMatches:
    def test_true_for_same_structure_and_new_values(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_rescaled_csr: torch.Tensor
    ) -> None:
        pattern = analyze_ic0(poisson_2d_csr)
        assert pattern.matches(poisson_2d_csr)
        assert pattern.matches(poisson_2d_rescaled_csr)

    def test_false_for_different_structure_same_size(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_extra_entry_csr: torch.Tensor
    ) -> None:
        assert not analyze_ic0(poisson_2d_csr).matches(poisson_2d_extra_entry_csr)

    def test_false_for_different_size(
        self, poisson_2d_csr: torch.Tensor, poisson_1d_csr: torch.Tensor
    ) -> None:
        assert not analyze_ic0(poisson_2d_csr).matches(poisson_1d_csr)

    def test_false_when_an_entry_turns_zero(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_explicit_zero_csr: torch.Tensor
    ) -> None:
        assert not analyze_ic0(poisson_2d_csr).matches(poisson_2d_explicit_zero_csr)

    def test_false_when_an_entry_turns_non_zero(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_explicit_zero_csr: torch.Tensor
    ) -> None:
        assert not analyze_ic0(poisson_2d_explicit_zero_csr).matches(poisson_2d_csr)

    def test_false_for_non_csr_layout(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_dense: torch.Tensor
    ) -> None:
        assert not analyze_ic0(poisson_2d_csr).matches(poisson_2d_dense)

    def test_false_after_in_place_index_mutation(
        self,
        tridiagonal_csr: torch.Tensor,
        redirect_last_row_entry: Callable[[torch.Tensor], None],
    ) -> None:
        """The key is a copy, so an in-place edit of the matrix's own indices is visible."""
        pattern = analyze_ic0(tridiagonal_csr)
        redirect_last_row_entry(tridiagonal_csr)

        assert not pattern.matches(tridiagonal_csr)

    def test_single_entry_matrix(self, single_entry_csr: torch.Tensor) -> None:
        pattern = analyze_ic0(single_entry_csr)

        assert pattern.matches(single_entry_csr)
        assert sparse_ic0(single_entry_csr, pattern).values().item() == 2.0

    def test_diagonal_only_matrix_factor_is_the_square_root(
        self, diagonal_csr: torch.Tensor
    ) -> None:
        factor = sparse_ic0(diagonal_csr, analyze_ic0(diagonal_csr))

        torch.testing.assert_close(factor.values(), diagonal_csr.values().sqrt())

    def test_pattern_of_an_explicit_zero_diagonal_breaks_down_like_a_fresh_factorization(
        self, explicit_zero_diagonal_csr: torch.Tensor
    ) -> None:
        with pytest.raises(ValueError, match="breakdown"):
            sparse_ic0(explicit_zero_diagonal_csr)
        with pytest.raises(ValueError, match="breakdown"):
            sparse_ic0(explicit_zero_diagonal_csr, analyze_ic0(explicit_zero_diagonal_csr))

    def test_explicit_zero_is_not_pattern(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_explicit_zero_csr: torch.Tensor
    ) -> None:
        """A stored zero is dropped from the factor, like the dense mask ``A != 0``."""
        factor = sparse_ic0(poisson_2d_explicit_zero_csr)
        assert factor._nnz() == sparse_ic0(poisson_2d_csr)._nnz() - 1

    def test_nan_entry_is_not_pattern(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_nan_entry_csr: torch.Tensor
    ) -> None:
        """A NaN is dropped like a zero (``|value| > 0`` is false for NaN), never kept as pattern."""
        pattern = analyze_ic0(poisson_2d_nan_entry_csr)
        base = analyze_ic0(poisson_2d_csr)

        assert int(pattern.lower_mask.sum()) == int(base.lower_mask.sum()) - 1
        assert pattern.matches(poisson_2d_nan_entry_csr)
        assert not base.matches(poisson_2d_nan_entry_csr)


class TestTransposePermutation:
    def test_planned_transpose_equals_coo_transpose(self, poisson_2d_csr: torch.Tensor) -> None:
        pattern = analyze_ic0(poisson_2d_csr)
        factor = sparse_ic0(poisson_2d_csr, pattern)

        planned = build_sparse_cholesky_solve_cache(factor, pattern).upper
        expected = factor.to_sparse_coo().t().to_sparse_csr()

        _assert_same_csr(planned, expected)

    def test_planned_transpose_is_differentiable_through_values(
        self, poisson_2d_csr: torch.Tensor
    ) -> None:
        pattern = analyze_ic0(poisson_2d_csr)
        values = poisson_2d_csr.values().clone().requires_grad_(True)
        matrix = torch.sparse_csr_tensor(
            poisson_2d_csr.crow_indices(),
            poisson_2d_csr.col_indices(),
            values,
            size=poisson_2d_csr.shape,
            check_invariants=False,
        )
        factor = sparse_ic0(matrix, pattern)

        build_sparse_cholesky_solve_cache(factor, pattern).upper.values().sum().backward()

        assert values.grad is not None
        assert torch.isfinite(values.grad).all()


STRUCTURAL_INDEX_FIELDS = (
    "row",
    "lower_crow",
    "lower_col",
    "upper_crow",
    "upper_col",
    "transpose_perm",
)
"""Pattern arrays that only index storage; stored at the narrowest width that fits."""


class TestStructuralIndexWidth:
    """Structural index arrays take 4 bytes per entry when they fit, 8 otherwise."""

    @pytest.mark.parametrize("field", STRUCTURAL_INDEX_FIELDS)
    def test_int32_when_indices_fit(self, narrow_pattern: IC0Pattern, field: str) -> None:
        assert getattr(narrow_pattern, field).dtype == torch.int32

    @pytest.mark.parametrize("field", STRUCTURAL_INDEX_FIELDS)
    def test_int64_when_indices_exceed_the_limit(
        self, wide_pattern: IC0Pattern, field: str
    ) -> None:
        assert getattr(wide_pattern, field).dtype == torch.int64

    def test_structural_bytes_are_halved(
        self, narrow_pattern: IC0Pattern, wide_pattern: IC0Pattern
    ) -> None:
        def nbytes(pattern: IC0Pattern) -> int:
            return sum(
                getattr(pattern, field).numel() * getattr(pattern, field).element_size()
                for field in STRUCTURAL_INDEX_FIELDS
            )

        assert 2 * nbytes(narrow_pattern) == nbytes(wide_pattern)

    def test_per_step_gather_indices_stay_int64(self, narrow_pattern: IC0Pattern) -> None:
        """Advanced-index gathers are slower with int32 indices, so the hot arrays keep int64."""
        for level in narrow_pattern.levels:
            assert level.rows.dtype == level.diag_pos.dtype == torch.int64
            for step in level.steps:
                assert step.flat_pos.dtype == torch.int64
                assert step.diag_pos_of_j.dtype == torch.int64
                assert step.level_local_idx.dtype == torch.int64
                if step.gather is not None:
                    assert step.gather.pos_ik.dtype == torch.int64
                    assert step.gather.pos_jk.dtype == torch.int64

    def test_reuse_key_owns_copies_of_the_matrix_index_tensors(
        self, poisson_2d_csr: torch.Tensor, narrow_pattern: IC0Pattern
    ) -> None:
        assert narrow_pattern.crow.data_ptr() != poisson_2d_csr.crow_indices().data_ptr()
        assert narrow_pattern.col.data_ptr() != poisson_2d_csr.col_indices().data_ptr()
        assert narrow_pattern.crow.dtype == poisson_2d_csr.crow_indices().dtype
        assert torch.equal(narrow_pattern.crow, poisson_2d_csr.crow_indices())
        assert torch.equal(narrow_pattern.col, poisson_2d_csr.col_indices())

    def test_factor_is_bit_identical_to_the_int64_plan(
        self, poisson_2d_csr: torch.Tensor, narrow_pattern: IC0Pattern, wide_pattern: IC0Pattern
    ) -> None:
        _assert_same_csr(
            sparse_ic0(poisson_2d_csr, narrow_pattern), sparse_ic0(poisson_2d_csr, wide_pattern)
        )

    def test_application_is_bit_identical_to_the_int64_plan(
        self,
        poisson_2d_csr: torch.Tensor,
        ramp_rhs: torch.Tensor,
        narrow_pattern: IC0Pattern,
        wide_pattern: IC0Pattern,
    ) -> None:
        def apply(pattern: IC0Pattern) -> torch.Tensor:
            factor = sparse_ic0(poisson_2d_csr, pattern)
            return build_sparse_cholesky_solve_cache(factor, pattern).solve(ramp_rhs)

        assert torch.equal(apply(narrow_pattern), apply(wide_pattern))

    def test_level_scheduled_solve_on_the_narrow_pair_matches_the_int64_pair(
        self,
        poisson_2d_csr: torch.Tensor,
        ramp_rhs: torch.Tensor,
        narrow_pattern: IC0Pattern,
        wide_pattern: IC0Pattern,
    ) -> None:
        """The differentiable path (``requires_grad`` target) reads the pair's index arrays."""
        target = ramp_rhs.clone().requires_grad_(True)

        def apply(pattern: IC0Pattern) -> torch.Tensor:
            factor = sparse_ic0(poisson_2d_csr, pattern)
            return build_sparse_cholesky_solve_cache(factor, pattern).solve(target)

        assert torch.equal(apply(narrow_pattern), apply(wide_pattern))

    def test_matches_same_matrix_with_int32_indices(
        self, poisson_2d_int32_csr: torch.Tensor
    ) -> None:
        assert analyze_ic0(poisson_2d_int32_csr).matches(poisson_2d_int32_csr)

    def test_int32_matrix_factor_equals_int64_matrix_factor(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_int32_csr: torch.Tensor
    ) -> None:
        assert torch.equal(
            sparse_ic0(poisson_2d_int32_csr).values(), sparse_ic0(poisson_2d_csr).values()
        )

    def test_matches_rejects_a_matrix_with_other_index_dtypes(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_int32_csr: torch.Tensor
    ) -> None:
        assert not analyze_ic0(poisson_2d_csr).matches(poisson_2d_int32_csr)
        assert not analyze_ic0(poisson_2d_int32_csr).matches(poisson_2d_csr)


class TestLowerPositions:
    """Integer gather positions select ``L``'s input values without a boolean-mask sync."""

    @pytest.mark.parametrize("fixture", ["poisson_2d_csr", "poisson_2d_explicit_zero_csr"])
    def test_select_the_same_values_as_the_lower_mask(
        self, fixture: str, request: pytest.FixtureRequest
    ) -> None:
        matrix = request.getfixturevalue(fixture)
        pattern = analyze_ic0(matrix)

        assert pattern.lower_positions.dtype == torch.int64
        assert torch.equal(
            matrix.values()[pattern.lower_positions], matrix.values()[pattern.lower_mask]
        )

    def test_factor_is_bit_identical_to_the_mask_gather(
        self, poisson_2d_rescaled_csr: torch.Tensor
    ) -> None:
        pattern = analyze_ic0(poisson_2d_rescaled_csr)
        mask_gather = dataclasses.replace(pattern, lower_positions=pattern.lower_mask)

        _assert_same_csr(
            _factorize(poisson_2d_rescaled_csr, pattern),
            _factorize(poisson_2d_rescaled_csr, mask_gather),
        )
