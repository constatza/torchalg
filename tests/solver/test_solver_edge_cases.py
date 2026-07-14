"""Edge-case tests for torch-native iterative solvers."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from numpy.typing import NDArray

from torchalg import flexible_cg, pcg
from torchalg.monitoring import TraceMode


@pytest.mark.parametrize("solver_name", ["fcg", "pcg"])
def test_zero_rhs_converges_immediately(
    solver_name: str,
    tridiagonal_spd_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
    integration_tolerances: tuple[float, float],
) -> None:
    """Zero RHS converges without a CG update."""
    matrix = to_torch(tridiagonal_spd_small)
    rhs = torch.zeros(matrix.shape[0], dtype=matrix.dtype)
    rtol, atol = integration_tolerances
    solver = flexible_cg if solver_name == "fcg" else pcg

    solution, result = solver(matrix, rhs, rtol=rtol, atol=atol, trace_mode=TraceMode.MINIMAL)

    assert result.converged is True
    assert result.iterations == 0
    assert torch.equal(solution, torch.zeros_like(rhs))


@pytest.mark.parametrize("solver_name", ["fcg", "pcg"])
def test_exact_initial_guess_converges_immediately(
    solver_name: str,
    tridiagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    integration_tolerances: tuple[float, float],
) -> None:
    """An exact initial guess is accepted as already converged."""
    matrix, rhs, expected = tridiagonal_system_known_solution_torch
    rtol, atol = integration_tolerances
    solver = flexible_cg if solver_name == "fcg" else pcg

    solution, result = solver(
        matrix,
        rhs,
        x0=expected,
        rtol=rtol,
        atol=atol,
        trace_mode=TraceMode.MINIMAL,
    )

    assert result.converged is True
    assert result.iterations == 0
    assert torch.allclose(solution, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("solver_name", ["fcg", "pcg"])
def test_identity_matrix_converges_in_one_iteration(
    solver_name: str,
    identity_matrix_small: NDArray,
    random_rhs_10_torch: torch.Tensor,
    to_torch: Callable[[NDArray], torch.Tensor],
    integration_tolerances: tuple[float, float],
) -> None:
    """Identity systems converge in one Krylov update."""
    matrix = to_torch(identity_matrix_small)
    rtol, atol = integration_tolerances
    solver = flexible_cg if solver_name == "fcg" else pcg

    solution, result = solver(
        matrix,
        random_rhs_10_torch,
        rtol=rtol,
        atol=atol,
        trace_mode=TraceMode.MINIMAL,
    )

    assert result.converged is True
    assert result.iterations == 1
    assert torch.allclose(solution, random_rhs_10_torch, rtol=rtol, atol=atol)


@pytest.mark.parametrize("solver_name", ["fcg", "pcg"])
def test_respects_maxiter_limit(
    solver_name: str,
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """Solvers stop at maxiter when tolerance is too strict."""
    matrix = to_torch(tridiagonal_spd_small)
    rhs = to_torch(rhs_ones_small)
    solver = flexible_cg if solver_name == "fcg" else pcg

    _, result = solver(matrix, rhs, rtol=1e-15, atol=1e-20, maxiter=3)

    assert result.iterations == 3
    assert result.converged is False
    assert result.stopping_criterion == "max_iter"


@pytest.mark.parametrize("solver_name", ["fcg", "pcg"])
def test_tight_tolerance_requires_at_least_as_many_iterations(
    solver_name: str,
    tridiagonal_spd_small: NDArray,
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> None:
    """Tight tolerances require no fewer iterations than loose tolerances."""
    matrix = to_torch(tridiagonal_spd_small)
    rhs = to_torch(rhs_ones_small)
    solver = flexible_cg if solver_name == "fcg" else pcg

    _, loose = solver(matrix, rhs, rtol=1e-2, atol=1e-14)
    _, tight = solver(matrix, rhs, rtol=1e-12, atol=1e-14)

    assert tight.iterations >= loose.iterations


@pytest.mark.parametrize("solver_name", ["fcg", "pcg"])
def test_good_initial_guess_reduces_iterations(
    solver_name: str,
    tridiagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    integration_tolerances: tuple[float, float],
) -> None:
    """A closer initial guess does not increase the iteration count."""
    matrix, rhs, expected = tridiagonal_system_known_solution_torch
    rtol, atol = integration_tolerances
    solver = flexible_cg if solver_name == "fcg" else pcg

    _, zero_result = solver(
        matrix,
        rhs,
        x0=torch.zeros_like(rhs),
        rtol=rtol,
        atol=atol,
    )
    _, good_result = solver(
        matrix,
        rhs,
        x0=0.8 * expected,
        rtol=rtol,
        atol=atol,
    )

    assert good_result.iterations <= zero_result.iterations
