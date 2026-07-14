"""High-precision convergence tests for torch-native CG solvers."""

from __future__ import annotations

import pytest
import torch

from torchalg import flexible_cg, pcg
from torchalg.preconditioners.implementations.jacobi import JacobiPreconditioner


@pytest.mark.parametrize("solver_name", ["fcg", "pcg"])
@pytest.mark.parametrize(
    "system_fixture",
    ["tridiagonal_system_known_solution_torch", "diagonal_system_known_solution_torch"],
)
def test_solver_matches_direct_solution_high_precision(
    solver_name: str,
    system_fixture: str,
    convergence_tolerances: tuple[float, float],
    request: pytest.FixtureRequest,
) -> None:
    """Torch-native CG solvers match known direct solutions at strict tolerances."""
    matrix, rhs, expected = request.getfixturevalue(system_fixture)
    rtol, atol = convergence_tolerances
    solver = flexible_cg if solver_name == "fcg" else pcg

    solution, result = solver(matrix, rhs, rtol=rtol, atol=atol, maxiter=200)

    assert result.converged is True
    assert torch.allclose(solution, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("solver_name", ["fcg", "pcg"])
def test_solver_residual_equation_matches_reported_residual(
    solver_name: str,
    tridiagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    convergence_tolerances: tuple[float, float],
) -> None:
    """Reported absolute residual matches explicit ``b - A @ x`` recomputation."""
    matrix, rhs, _ = tridiagonal_system_known_solution_torch
    rtol, atol = convergence_tolerances
    solver = flexible_cg if solver_name == "fcg" else pcg

    solution, result = solver(matrix, rhs, rtol=rtol, atol=atol, maxiter=200)

    explicit_residual = rhs - matrix @ solution
    assert result.converged is True
    assert torch.isclose(
        torch.tensor(result.residual_abs, dtype=rhs.dtype),
        torch.linalg.norm(explicit_residual),
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(explicit_residual, torch.zeros_like(rhs), atol=atol)


@pytest.mark.parametrize("solver_name", ["fcg", "pcg"])
def test_solver_iteration_count_reasonable_with_jacobi(
    solver_name: str,
    tridiagonal_system_known_solution_torch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    convergence_tolerances: tuple[float, float],
) -> None:
    """Well-conditioned small systems converge within a conservative bound."""
    matrix, rhs, _ = tridiagonal_system_known_solution_torch
    rtol, atol = convergence_tolerances
    preconditioner = JacobiPreconditioner(matrix)
    solver = flexible_cg if solver_name == "fcg" else pcg

    _, result = solver(
        matrix,
        rhs,
        preconditioner=preconditioner,
        rtol=rtol,
        atol=atol,
        maxiter=200,
    )

    assert result.converged is True
    assert result.iterations <= 50
