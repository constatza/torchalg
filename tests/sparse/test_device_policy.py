"""Tests for ``torchalg.sparse.device_policy.recommend_device``.

Pure function over measured thresholds (``docs/plan.md``'s "The generic
CPU-vs-CUDA rule"/"At-scale evidence") - no actual CUDA device required to
test it, since it only ever returns a ``torch.device`` object, never moves a
tensor.
"""

from __future__ import annotations

import torch

from torchalg.sparse.device_policy import (
    ELEMENTWISE_CUDA_CROSSOVER_N,
    recommend_device,
)


class TestRecommendDeviceParallelKernel:
    def test_below_crossover_recommends_cpu(self, parallel_kernel_below_crossover_n: int) -> None:
        device = recommend_device("parallel_kernel", parallel_kernel_below_crossover_n)
        assert device == torch.device("cpu")

    def test_at_or_above_crossover_recommends_cuda(
        self, parallel_kernel_at_crossover_n: int
    ) -> None:
        device = recommend_device("parallel_kernel", parallel_kernel_at_crossover_n)
        assert device == torch.device("cuda")


class TestRecommendDeviceElementwise:
    def test_never_crosses_in_tested_range(self, largest_measured_elementwise_n: int) -> None:
        device = recommend_device("elementwise", largest_measured_elementwise_n)
        assert device == torch.device("cpu")

    def test_crossover_constant_is_unreachable(self) -> None:
        assert ELEMENTWISE_CUDA_CROSSOVER_N == float("inf")


class TestRecommendDeviceIterative:
    def test_below_crossover_recommends_cpu(self, iterative_below_crossover_n: int) -> None:
        device = recommend_device("iterative", iterative_below_crossover_n)
        assert device == torch.device("cpu")

    def test_at_or_above_crossover_recommends_cuda(self, iterative_at_crossover_n: int) -> None:
        device = recommend_device("iterative", iterative_at_crossover_n)
        assert device == torch.device("cuda")
