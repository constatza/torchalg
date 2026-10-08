"""Tests for ``torchalg.sparse.ic0`` (``_lookup``/``_build_lookup``, ``sparse_ic0``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from torchalg.preconditioners.implementations import IC0Preconditioner as DenseIC0Preconditioner
from torchalg.preconditioners.implementations._masked_factorization import dense_ic0
from torchalg.sparse.preconditioners.ic0 import (
    IC0Preconditioner,
    _build_lookup,
    _lookup,
    sparse_ic0,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class TestLookup:
    """Direct unit tests for the shared global sparse-element lookup primitive."""

    @pytest.fixture
    def pattern(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """A 3-row pattern spanning multiple rows: row 0 cols [0, 2], row 1
        cols [1], row 2 cols [0, 2]. Values: [10.0, 20.0, 30.0, 40.0, 50.0].
        """
        n = 3
        row_index = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int64)
        col_index = torch.tensor([0, 2, 1, 0, 2], dtype=torch.int64)
        values = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.float64)
        return row_index, col_index, values, n

    def test_finds_present_entries_spanning_multiple_rows(
        self, pattern: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
    ) -> None:
        row_index, col_index, values, n = pattern
        keys = _build_lookup(row_index, col_index, n)
        found, gathered = _lookup(
            keys,
            values,
            query_rows=torch.tensor([0, 1, 2], dtype=torch.int64),
            query_cols=torch.tensor([2, 1, 0], dtype=torch.int64),
            n=n,
        )
        torch.testing.assert_close(found, torch.tensor([True, True, True]))
        torch.testing.assert_close(gathered, torch.tensor([20.0, 30.0, 40.0], dtype=torch.float64))

    def test_absent_entries_report_not_found_and_zero(
        self, pattern: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
    ) -> None:
        row_index, col_index, values, n = pattern
        keys = _build_lookup(row_index, col_index, n)
        found, gathered = _lookup(
            keys,
            values,
            query_rows=torch.tensor([0, 1, 1], dtype=torch.int64),
            query_cols=torch.tensor([1, 0, 2], dtype=torch.int64),
            n=n,
        )
        torch.testing.assert_close(found, torch.tensor([False, False, False]))
        torch.testing.assert_close(gathered, torch.zeros(3, dtype=torch.float64))

    def test_mixed_present_and_absent_in_one_batch(
        self, pattern: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
    ) -> None:
        row_index, col_index, values, n = pattern
        keys = _build_lookup(row_index, col_index, n)
        found, gathered = _lookup(
            keys,
            values,
            query_rows=torch.tensor([0, 0, 1, 2, 2], dtype=torch.int64),
            query_cols=torch.tensor([0, 1, 1, 0, 1], dtype=torch.int64),
            n=n,
        )
        torch.testing.assert_close(found, torch.tensor([True, False, True, True, False]))
        torch.testing.assert_close(
            gathered, torch.tensor([10.0, 0.0, 30.0, 40.0, 0.0], dtype=torch.float64)
        )

    def test_build_lookup_matches_storage_order(
        self, pattern: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]
    ) -> None:
        row_index, col_index, _, n = pattern
        keys = _build_lookup(row_index, col_index, n)
        assert bool((keys[1:] > keys[:-1]).all())


class TestSparseIC0:
    """``sparse_ic0`` matches ``dense_ic0`` (the dense algorithm as the oracle)."""

    def test_matches_dense_ic0_poisson_1d(
        self, poisson_1d_dense: torch.Tensor, poisson_1d_csr: torch.Tensor
    ) -> None:
        expected = dense_ic0(poisson_1d_dense, threshold=0.0)
        actual = sparse_ic0(poisson_1d_csr, threshold=0.0).to_dense()
        torch.testing.assert_close(actual, expected)

    def test_matches_dense_ic0_poisson_2d(
        self, poisson_2d_dense: torch.Tensor, poisson_2d_csr: torch.Tensor
    ) -> None:
        expected = dense_ic0(poisson_2d_dense, threshold=0.0)
        actual = sparse_ic0(poisson_2d_csr, threshold=0.0).to_dense()
        torch.testing.assert_close(actual, expected)

    def test_matches_dense_ic0_larger_anisotropic_grid(
        self, anisotropic_2d_factory: Callable[[int, float], torch.Tensor]
    ) -> None:
        dense = anisotropic_2d_factory(5, 0.3)
        sparse = dense.to_sparse_csr()
        expected = dense_ic0(dense, threshold=0.0)
        actual = sparse_ic0(sparse, threshold=0.0).to_dense()
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
        expected = dense_ic0(dense, threshold=0.0)
        actual = sparse_ic0(sparse, threshold=0.0).to_dense()
        torch.testing.assert_close(actual, expected)

    def test_lookup_call_count_bounded_by_levels_times_max_degree(
        self, poisson_2d_csr: torch.Tensor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The inner Python loop iterates ``num_levels * max_degree`` times,
        not ``nnz`` - instrumented via a call counter on ``_lookup``, which
        is invoked at most once per ``d`` within a level (never once per
        matrix entry).
        """
        import torchalg.sparse.preconditioners.ic0 as ic0_module
        from torchalg.sparse.kernels.triangular import level_schedule

        call_count = 0
        real_lookup = ic0_module._lookup

        def counting_lookup(
            keys: torch.Tensor,
            values: torch.Tensor,
            query_rows: torch.Tensor,
            query_cols: torch.Tensor,
            n: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal call_count
            call_count += 1
            return real_lookup(keys, values, query_rows, query_cols, n)

        monkeypatch.setattr(ic0_module, "_lookup", counting_lookup)
        sparse_ic0(poisson_2d_csr, threshold=0.0)

        row_index = ic0_module._expand_row_index(poisson_2d_csr)
        col = poisson_2d_csr.col_indices()
        keep = (col <= row_index) & (poisson_2d_csr.values().abs() > 0.0)
        pattern = ic0_module._csr_from_boolean_submask(poisson_2d_csr, keep)
        schedule = level_schedule(pattern, direction="forward")
        row_offdiag_count = pattern.crow_indices()[1:] - pattern.crow_indices()[:-1] - 1

        offset = 0
        max_lookup_calls = 0
        for level_size in schedule.level_sizes.tolist():
            level_rows = schedule.row_order[offset : offset + level_size]
            offset += level_size
            degrees = row_offdiag_count[level_rows]
            max_degree = int(degrees.max()) if level_size > 0 else 0
            # ``d == 0`` never calls ``_lookup`` (no earlier columns yet).
            max_lookup_calls += max(max_degree - 1, 0)

        assert call_count <= max_lookup_calls
        assert call_count < pattern.values().numel()

    def test_threshold_matches_dense_sparsity_pattern(
        self, poisson_2d_dense: torch.Tensor, poisson_2d_csr: torch.Tensor
    ) -> None:
        threshold = 0.5
        expected = dense_ic0(poisson_2d_dense, threshold=threshold)
        actual = sparse_ic0(poisson_2d_csr, threshold=threshold).to_dense()
        torch.testing.assert_close(actual, expected)

    def test_raises_on_non_spd_breakdown(self, torch_dtype: torch.dtype) -> None:
        dense = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch_dtype)
        with pytest.raises(ValueError, match="breakdown"):
            dense_ic0(dense, threshold=0.0)
        with pytest.raises(ValueError, match="breakdown"):
            sparse_ic0(dense.to_sparse_csr(), threshold=0.0)

    def test_rejects_non_csr_layout(self, poisson_1d_dense: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="sparse CSR"):
            sparse_ic0(poisson_1d_dense, threshold=0.0)

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
        factor = sparse_ic0(matrix, threshold=0.0)
        factor.values().sum().backward()
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()


class TestIC0PreconditionerSparseDenseParity:
    """The sparse ``IC0Preconditioner``'s ``apply()`` matches the dense sibling's.

    Dense-as-oracle, this project's standing testing rule - replaces the
    deleted ``test_ic0_sparse_dispatch.py`` (which asserted one class's two
    branches agreed; now there are two classes).
    """

    def test_matches_dense_on_poisson_1d(self, poisson_1d_dense: torch.Tensor) -> None:
        torch.manual_seed(42)
        residual = torch.randn(poisson_1d_dense.shape[0], dtype=poisson_1d_dense.dtype)

        dense_precond = DenseIC0Preconditioner()
        dense_precond.setup(poisson_1d_dense)
        sparse_precond = IC0Preconditioner()
        sparse_precond.setup(poisson_1d_dense.to_sparse_csr())

        torch.testing.assert_close(sparse_precond.apply(residual), dense_precond.apply(residual))

    def test_matches_dense_on_poisson_2d(self, poisson_2d_dense: torch.Tensor) -> None:
        torch.manual_seed(42)
        residual = torch.randn(poisson_2d_dense.shape[0], dtype=poisson_2d_dense.dtype)

        dense_precond = DenseIC0Preconditioner()
        dense_precond.setup(poisson_2d_dense)
        sparse_precond = IC0Preconditioner()
        sparse_precond.setup(poisson_2d_dense.to_sparse_csr())

        torch.testing.assert_close(sparse_precond.apply(residual), dense_precond.apply(residual))

    def test_batched_apply_matches_dense(self, poisson_2d_dense: torch.Tensor) -> None:
        """A matrix RHS is handled by the native batched triangular solve."""
        torch.manual_seed(42)
        residual = torch.randn(poisson_2d_dense.shape[0], 3, dtype=poisson_2d_dense.dtype)
        dense_precond = DenseIC0Preconditioner().setup(poisson_2d_dense)
        sparse_precond = IC0Preconditioner().setup(poisson_2d_dense.to_sparse_csr())

        torch.testing.assert_close(sparse_precond.apply(residual), dense_precond.apply(residual))
