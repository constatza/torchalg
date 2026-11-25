"""Prepared coarse solvers: factor once per hierarchy, apply per cycle.

Covers ``torchalg.multigrid.coarse_solve`` (factories), ``VCycle``/``WCycle.prepare``
(attaching the prepared solve to the coarsest level) and the guarantee that applying a
prepared hierarchy never re-factorizes.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch

from torchalg.multigrid.coarse_solve import (
    CHOLESKY_MIN_DIMENSION,
    CoarseSolve,
    HierarchyNotPreparedError,
    lu_coarse_solver,
    pseudo_inverse_coarse_solver,
)
from torchalg.multigrid.cycle import VCycle, WCycle
from torchalg.multigrid.hierarchy import MultigridHierarchy, PreparedCoarsestLevel, build_hierarchy
from torchalg.preconditioners.base import PreconditionerNotReadyError
from torchalg.preconditioners.implementations.amg import AggregationCoarsening, JacobiSmoother

SEED = 7
COARSE_SIZE = 12
MODERATE_CONDITION = 1e11


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


def _spd(size: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED)
    factor = torch.randn(size, size, dtype=torch.float64, generator=generator)
    return factor @ factor.T + size * torch.eye(size, dtype=torch.float64)


@pytest.fixture
def large_spd_matrix() -> torch.Tensor:
    """SPD matrix wide enough to take the Cholesky route."""
    return _spd(CHOLESKY_MIN_DIMENSION)


@pytest.fixture
def large_singular_matrix() -> torch.Tensor:
    """Wide symmetric matrix with a zero last row/column, whose Cholesky fails on a zero pivot."""
    matrix = torch.eye(CHOLESKY_MIN_DIMENSION, dtype=torch.float64) * 2.0
    matrix[-1, -1] = 0.0
    return matrix


@pytest.fixture
def large_rhs() -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED + 1)
    return torch.randn(CHOLESKY_MIN_DIMENSION, dtype=torch.float64, generator=generator)


@pytest.fixture
def semidefinite_matrix() -> torch.Tensor:
    """Symmetric positive semidefinite of rank ``n - 1`` whose Cholesky still succeeds.

    Its last pivot is rounding noise rather than an exact zero (asserted here), so only the
    condition bound keeps it away from a garbage inverse.
    """
    generator = torch.Generator().manual_seed(0)
    size = CHOLESKY_MIN_DIMENSION
    factor = torch.randn(size, size - 1, dtype=torch.float64, generator=generator)
    matrix = factor @ factor.T
    matrix = (matrix + matrix.T) / 2
    assert int(torch.linalg.cholesky_ex(matrix).info) == 0
    return matrix


@pytest.fixture
def spd_with_condition() -> Callable[[float], torch.Tensor]:
    """Build a ``CHOLESKY_MIN_DIMENSION``-wide SPD matrix with a known 2-norm condition number.

    The eigenvalues are log-spaced from ``1 / condition`` to ``1`` under a seeded random
    orthogonal basis.
    """

    def build(condition: float) -> torch.Tensor:
        generator = torch.Generator().manual_seed(SEED)
        size = CHOLESKY_MIN_DIMENSION
        basis, _ = torch.linalg.qr(
            torch.randn(size, size, dtype=torch.float64, generator=generator)
        )
        spectrum = torch.logspace(-math.log10(condition), 0.0, size, dtype=torch.float64)
        matrix = basis @ torch.diag(spectrum) @ basis.T
        return (matrix + matrix.T) / 2

    return build


@pytest.fixture
def pinv_truncation_condition() -> float:
    """``1 / (c eps)``: ``pinv``'s default ``rtol`` truncates singular values below this ratio."""
    return 1.0 / (CHOLESKY_MIN_DIMENSION * torch.finfo(torch.float64).eps)


@pytest.fixture
def forbid_pinv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any call to ``torch.linalg.pinv`` fail the test."""

    def forbidden(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("this coarse matrix must not need the SVD-based pinv")

    monkeypatch.setattr(torch.linalg, "pinv", forbidden)


@pytest.fixture
def nonsymmetric_matrix(large_spd_matrix: torch.Tensor) -> torch.Tensor:
    """Well-conditioned matrix whose lower triangle alone is a different SPD matrix."""
    skew = torch.triu(torch.ones_like(large_spd_matrix), diagonal=1) * 0.5
    return large_spd_matrix + skew


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

    def test_small_spd_matrix_keeps_the_exact_pinv(
        self, spd_matrix: torch.Tensor, rhs: torch.Tensor
    ) -> None:
        solve = pseudo_inverse_coarse_solver(spd_matrix)
        assert torch.equal(solve(rhs), torch.linalg.pinv(spd_matrix) @ rhs)

    def test_large_spd_matrix_matches_the_pseudo_inverse(
        self, large_spd_matrix: torch.Tensor, large_rhs: torch.Tensor
    ) -> None:
        solve = pseudo_inverse_coarse_solver(large_spd_matrix)
        torch.testing.assert_close(
            solve(large_rhs),
            torch.linalg.pinv(large_spd_matrix) @ large_rhs,
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.usefixtures("forbid_pinv")
    def test_large_spd_matrix_is_inverted_without_an_svd(
        self, large_spd_matrix: torch.Tensor
    ) -> None:
        pseudo_inverse_coarse_solver(large_spd_matrix)

    @pytest.mark.usefixtures("forbid_pinv")
    def test_moderately_conditioned_spd_matrix_takes_cholesky(
        self, spd_with_condition: Callable[[float], torch.Tensor]
    ) -> None:
        """``kappa = 1e11`` is far enough from ``pinv``'s truncation for the bound to prove it."""
        pseudo_inverse_coarse_solver(spd_with_condition(MODERATE_CONDITION))

    def test_cholesky_inverse_is_at_least_as_accurate_as_pinv(
        self, spd_with_condition: Callable[[float], torch.Tensor], large_rhs: torch.Tensor
    ) -> None:
        """Their difference at a moderate ``kappa`` is rounding, not a different solution."""
        matrix = spd_with_condition(MODERATE_CONDITION)
        rhs = matrix @ large_rhs

        def forward_error(solution: torch.Tensor) -> float:
            return float((solution - large_rhs).norm() / large_rhs.norm())

        cholesky_error = forward_error(pseudo_inverse_coarse_solver(matrix)(rhs))
        assert cholesky_error <= forward_error(torch.linalg.pinv(matrix) @ rhs)

    def test_spd_matrix_near_the_pinv_truncation_keeps_the_exact_pinv(
        self,
        spd_with_condition: Callable[[float], torch.Tensor],
        pinv_truncation_condition: float,
        large_rhs: torch.Tensor,
    ) -> None:
        """Near ``1 / (c eps)`` the bound cannot prove ``pinv`` truncates nothing."""
        matrix = spd_with_condition(0.5 * pinv_truncation_condition)

        solve = pseudo_inverse_coarse_solver(matrix)

        assert torch.equal(solve(large_rhs), torch.linalg.pinv(matrix) @ large_rhs)

    def test_pinv_solver_is_exactly_the_pinv_when_nonsymmetric(
        self, nonsymmetric_matrix: torch.Tensor, large_rhs: torch.Tensor
    ) -> None:
        """Cholesky reads one triangle only, so a nonsymmetric matrix must take the SVD route."""
        solve = pseudo_inverse_coarse_solver(nonsymmetric_matrix)
        assert torch.equal(solve(large_rhs), torch.linalg.pinv(nonsymmetric_matrix) @ large_rhs)

    def test_pinv_solver_is_exactly_the_pinv_when_semidefinite(
        self, semidefinite_matrix: torch.Tensor, large_rhs: torch.Tensor
    ) -> None:
        solve = pseudo_inverse_coarse_solver(semidefinite_matrix)
        assert torch.equal(solve(large_rhs), torch.linalg.pinv(semidefinite_matrix) @ large_rhs)

    def test_pinv_solver_is_exactly_the_pinv_when_a_wide_matrix_has_a_zero_row(
        self, large_singular_matrix: torch.Tensor, large_rhs: torch.Tensor
    ) -> None:
        solve = pseudo_inverse_coarse_solver(large_singular_matrix)
        assert torch.equal(solve(large_rhs), torch.linalg.pinv(large_singular_matrix) @ large_rhs)

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

        assert isinstance(hierarchy, MultigridHierarchy)
        assert not isinstance(hierarchy.levels[-1], PreparedCoarsestLevel)
        assert vars(cycle) == state_before

    def test_apply_rejects_unprepared_hierarchy(self, hierarchy: MultigridHierarchy) -> None:
        cycle = VCycle(JacobiSmoother())
        rhs = torch.ones(hierarchy.levels[0].matrix.shape[0], dtype=torch.float64)

        with pytest.raises(HierarchyNotPreparedError):
            cycle.apply(hierarchy, rhs)  # ty: ignore[invalid-argument-type]

    def test_unprepared_hierarchy_error_is_a_not_ready_error(self) -> None:
        assert issubclass(HierarchyNotPreparedError, PreconditionerNotReadyError)

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
