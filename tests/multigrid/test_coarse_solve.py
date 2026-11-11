"""Prepared coarse solvers: factor once per hierarchy, apply per cycle.

Covers ``torchalg.multigrid.coarse_solve`` (factories), ``VCycle``/``WCycle.prepare``
(attaching the prepared solve to the coarsest level) and the guarantee that applying a
prepared hierarchy never re-factorizes.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torchalg.multigrid.coarse_solve import (
    CoarseSolve,
    HierarchyNotPreparedError,
    lu_coarse_solver,
    pseudo_inverse_coarse_solver,
)
from torchalg.multigrid.cycle import VCycle, WCycle
from torchalg.multigrid.hierarchy import MultigridHierarchy, build_hierarchy
from torchalg.preconditioners.implementations.amg import AggregationCoarsening, JacobiSmoother

SEED = 7
COARSE_SIZE = 12


@pytest.fixture
def spd_matrix() -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED)
    factor = torch.randn(COARSE_SIZE, COARSE_SIZE, dtype=torch.float64, generator=generator)
    return factor @ factor.T + COARSE_SIZE * torch.eye(COARSE_SIZE, dtype=torch.float64)


@pytest.fixture
def singular_matrix() -> torch.Tensor:
    """Symmetric rank-deficient matrix: last row/column zero."""
    matrix = torch.eye(COARSE_SIZE, dtype=torch.float64) * 2.0
    matrix[-1, -1] = 0.0
    return matrix


@pytest.fixture
def rhs() -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED + 1)
    return torch.randn(COARSE_SIZE, dtype=torch.float64, generator=generator)


@pytest.fixture
def hierarchy(poisson_1d_factory: Callable[[int], torch.Tensor]) -> MultigridHierarchy:
    return build_hierarchy(poisson_1d_factory(20), AggregationCoarsening(), n_levels=2)


class CountingFactory:
    """Coarse-solver factory that records how many times it factorized."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, matrix: torch.Tensor) -> CoarseSolve:
        self.calls += 1
        return lu_coarse_solver(matrix)


class TestFactories:
    def test_lu_solver_matches_linalg_solve(
        self, spd_matrix: torch.Tensor, rhs: torch.Tensor
    ) -> None:
        solve = lu_coarse_solver(spd_matrix)
        torch.testing.assert_close(solve(rhs), torch.linalg.solve(spd_matrix, rhs))

    def test_lu_solver_handles_matrix_rhs(self, spd_matrix: torch.Tensor) -> None:
        block = torch.eye(COARSE_SIZE, dtype=torch.float64)[:, :3]
        torch.testing.assert_close(
            lu_coarse_solver(spd_matrix)(block), torch.linalg.solve(spd_matrix, block)
        )

    def test_lu_solver_rejects_singular_matrix(self, singular_matrix: torch.Tensor) -> None:
        with pytest.raises(torch.linalg.LinAlgError):
            lu_coarse_solver(singular_matrix)

    def test_pinv_solver_is_exactly_the_per_call_pinv(
        self, spd_matrix: torch.Tensor, rhs: torch.Tensor
    ) -> None:
        solve = pseudo_inverse_coarse_solver(spd_matrix)
        assert torch.equal(solve(rhs), torch.linalg.pinv(spd_matrix) @ rhs)

    def test_pinv_solver_returns_minimum_norm_solution_when_singular(
        self, singular_matrix: torch.Tensor, rhs: torch.Tensor
    ) -> None:
        solution = pseudo_inverse_coarse_solver(singular_matrix)(rhs)
        assert solution[-1] == 0.0
        torch.testing.assert_close(solution[:-1], rhs[:-1] / 2.0)


class TestPreparedHierarchy:
    @pytest.mark.parametrize("cycle_class", [VCycle, WCycle])
    def test_factory_runs_once_across_many_applies(
        self,
        cycle_class: type[VCycle | WCycle],
        hierarchy: MultigridHierarchy,
    ) -> None:
        factory = CountingFactory()
        cycle = cycle_class(JacobiSmoother(), coarse_solver_factory=factory)
        prepared = cycle.prepare(hierarchy)
        rhs = torch.ones(hierarchy.levels[0].matrix.shape[0], dtype=torch.float64)

        for _ in range(3):
            cycle.apply(prepared, rhs)

        assert factory.calls == 1

    def test_prepare_is_idempotent(self, hierarchy: MultigridHierarchy) -> None:
        factory = CountingFactory()
        cycle = VCycle(JacobiSmoother(), coarse_solver_factory=factory)
        prepared = cycle.prepare(hierarchy)

        assert cycle.prepare(prepared) is prepared
        assert factory.calls == 1

    def test_prepare_leaves_input_hierarchy_and_cycle_untouched(
        self, hierarchy: MultigridHierarchy
    ) -> None:
        cycle = VCycle(JacobiSmoother())
        state_before = dict(vars(cycle))

        cycle.prepare(hierarchy)

        assert hierarchy.levels[-1].coarse_solve is None
        assert vars(cycle) == state_before

    def test_apply_rejects_unprepared_hierarchy(self, hierarchy: MultigridHierarchy) -> None:
        cycle = VCycle(JacobiSmoother())
        rhs = torch.ones(hierarchy.levels[0].matrix.shape[0], dtype=torch.float64)

        with pytest.raises(HierarchyNotPreparedError):
            cycle.apply(hierarchy, rhs)

    @pytest.mark.parametrize("cycle_class", [VCycle, WCycle])
    def test_prepared_apply_matches_per_call_solve(
        self, cycle_class: type[VCycle | WCycle], hierarchy: MultigridHierarchy
    ) -> None:
        cycle = cycle_class(JacobiSmoother())
        rhs = torch.linspace(1.0, 2.0, hierarchy.levels[0].matrix.shape[0], dtype=torch.float64)
        coarse_matrix = hierarchy.levels[-1].matrix

        result = cycle.apply(cycle.prepare(hierarchy), rhs)

        # Reference: same cycle but its coarse solve is the raw per-call torch.linalg.solve.
        calls: list[torch.Tensor] = []

        def reference_factory(matrix: torch.Tensor) -> CoarseSolve:
            def solve(vector: torch.Tensor) -> torch.Tensor:
                calls.append(vector)
                return torch.linalg.solve(matrix, vector)

            return solve

        reference = cycle_class(JacobiSmoother(), coarse_solver_factory=reference_factory)
        expected = reference.apply(reference.prepare(hierarchy), rhs)
        torch.testing.assert_close(result, expected)
        assert calls
        assert coarse_matrix.shape[0] < rhs.shape[0]
