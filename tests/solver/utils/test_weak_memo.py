"""Tests for ``torchalg.utils.weak_cache.weak_memo``."""

from __future__ import annotations

import gc
import weakref

import torch

from torchalg.utils.weak_cache import weak_memo


class TestWeakMemo:
    def test_same_object_and_key_computes_once(self) -> None:
        calls: list[int] = []

        @weak_memo
        def scaled(matrix: torch.Tensor, factor: float) -> torch.Tensor:
            calls.append(1)
            return matrix * factor

        matrix = torch.ones(3)
        assert scaled(matrix, 2.0) is scaled(matrix, 2.0)
        assert len(calls) == 1

    def test_distinct_keys_and_distinct_objects_compute_separately(self) -> None:
        calls: list[int] = []

        @weak_memo
        def scaled(matrix: torch.Tensor, factor: float) -> torch.Tensor:
            calls.append(1)
            return matrix * factor

        first, second = torch.ones(3), torch.ones(3)  # equal values, distinct objects
        scaled(first, 1.0)
        scaled(first, 2.0)
        scaled(second, 1.0)
        assert len(calls) == 3

    def test_cache_does_not_keep_the_matrix_alive(self) -> None:
        @weak_memo
        def scaled(matrix: torch.Tensor, factor: float) -> torch.Tensor:
            return matrix * factor

        matrix = torch.ones(3)
        reference = weakref.ref(matrix)
        scaled(matrix, 2.0)
        del matrix
        gc.collect()
        assert reference() is None
