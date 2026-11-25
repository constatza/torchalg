"""Tests for ``torchalg.sparse.preconditioners._pattern_index`` (``build_lookup``, ``lookup``)."""

from __future__ import annotations

import pytest
import torch

from torchalg.sparse.preconditioners._pattern_index import build_lookup, lookup


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
        keys = build_lookup(row_index, col_index, n)
        found, gathered = lookup(
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
        keys = build_lookup(row_index, col_index, n)
        found, gathered = lookup(
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
        keys = build_lookup(row_index, col_index, n)
        found, gathered = lookup(
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
        keys = build_lookup(row_index, col_index, n)
        assert bool((keys[1:] > keys[:-1]).all())
