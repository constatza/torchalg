"""Tests for the pure ``build_hierarchy`` function and the relocated test-vector helpers."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torchalg.preconditioners.implementations.amg import AggregationCoarsening
from torchalg.preconditioners.implementations.amg._test_vectors import (
    apply_jacobi_damping,
    apply_jacobi_damping_trajectory,
)
from torchalg.preconditioners.implementations.amg.hierarchy import build_hierarchy
from torchalg.preconditioners.implementations.pod import weighting


@pytest.fixture
def poisson_matrix(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    """64-node 1D Poisson matrix."""
    return poisson_1d_factory(64)


@pytest.fixture
def coarsening() -> AggregationCoarsening:
    """Default smoothed-aggregation coarsening."""
    return AggregationCoarsening()


class TestBuildHierarchy:
    @pytest.mark.parametrize("n_levels", [2, 3])
    def test_level_count_and_coarsest_has_no_transfer(
        self, poisson_matrix: torch.Tensor, coarsening: AggregationCoarsening, n_levels: int
    ) -> None:
        hierarchy = build_hierarchy(poisson_matrix, coarsening, n_levels)
        assert len(hierarchy.levels) == n_levels
        assert hierarchy.levels[-1].transfer is None
        assert all(level.transfer is not None for level in hierarchy.levels[:-1])

    def test_finest_level_is_input_matrix_and_sizes_shrink(
        self, poisson_matrix: torch.Tensor, coarsening: AggregationCoarsening
    ) -> None:
        hierarchy = build_hierarchy(poisson_matrix, coarsening, 3)
        sizes = [level.matrix.shape[0] for level in hierarchy.levels]
        assert hierarchy.levels[0].matrix is poisson_matrix
        assert sizes == sorted(sizes, reverse=True)
        assert len(set(sizes)) == len(sizes)


class TestTestVectorHelpersRelocation:
    def test_pod_weighting_reexports_the_same_functions(self) -> None:
        assert weighting.apply_jacobi_damping is apply_jacobi_damping
        assert weighting.apply_jacobi_damping_trajectory is apply_jacobi_damping_trajectory
