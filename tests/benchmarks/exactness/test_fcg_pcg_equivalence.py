"""Benchmark: torchalg FCG vs PCG equivalence for fixed preconditioners."""

from __future__ import annotations

import pytest
import torch

from tests.benchmarks.exactness.conftest import (
    ITER_DIFF_THRESHOLD_FCG_PCG,
    SOLUTION_COMPARISON_ATOL,
    SOLUTION_COMPARISON_RTOL,
    assert_solutions_match,
    check_residual_norm,
    generate_spd_system,
)
from torchalg import flexible_cg, pcg
from torchalg.monitoring import TraceMode
from torchalg.preconditioners.implementations.ilu import ILUPreconditioner

TEST_CASES = [
    (100, "tridiagonal", None),
    (100, "tridiagonal", "ilu"),
    (200, "tridiagonal", None),
    (200, "diagonal", None),
    (500, "tridiagonal", None),
    (500, "diagonal", None),
]


def _is_slow_case(size: int) -> bool:
    """Return whether the exactness case should be marked slow."""
    return size >= 500


@pytest.mark.benchmark
@pytest.mark.parametrize("size,matrix_type,precond_type", TEST_CASES)
def test_fcg_pcg_equivalence(
    size: int,
    matrix_type: str,
    precond_type: str | None,
    convergence_tolerances: tuple[float, float],
    request: pytest.FixtureRequest,
) -> None:
    """FCG and PCG remain near-equivalent with fixed preconditioners."""
    if _is_slow_case(size):
        request.node.add_marker(pytest.mark.slow)

    matrix_np, rhs_np, _, _ = generate_spd_system(size, matrix_type)
    matrix = torch.from_numpy(matrix_np).clone()
    rhs = torch.from_numpy(rhs_np).clone()
    x0 = torch.zeros_like(rhs)
    rtol, atol = convergence_tolerances

    preconditioner_pcg = None
    preconditioner_fcg = None
    if precond_type == "ilu":
        preconditioner_pcg = ILUPreconditioner(matrix)
        preconditioner_fcg = ILUPreconditioner(matrix)

    x_pcg, result_pcg = pcg(
        matrix,
        rhs,
        x0=x0,
        preconditioner=preconditioner_pcg,
        trace_mode=TraceMode.MINIMAL,
        rtol=rtol,
        atol=atol,
        maxiter=size * 2,
    )
    x_fcg, result_fcg = flexible_cg(
        matrix,
        rhs,
        x0=x0,
        preconditioner=preconditioner_fcg,
        m_max=-1,
        trace_mode=TraceMode.MINIMAL,
        rtol=rtol,
        atol=atol,
        maxiter=size * 2,
    )

    assert result_pcg.converged is True
    assert result_fcg.converged is True
    check_residual_norm(
        matrix_np,
        rhs_np,
        x_pcg.detach().cpu().numpy(),
        rtol,
        atol,
        label="PCG solution",
    )
    check_residual_norm(
        matrix_np,
        rhs_np,
        x_fcg.detach().cpu().numpy(),
        rtol,
        atol,
        label="FCG solution",
    )

    diff = result_fcg.iterations - result_pcg.iterations
    assert abs(diff) <= ITER_DIFF_THRESHOLD_FCG_PCG
    assert_solutions_match(
        matrix_np,
        rhs_np,
        x_pcg.detach().cpu().numpy(),
        x_fcg.detach().cpu().numpy(),
        rtol=SOLUTION_COMPARISON_RTOL,
        atol=SOLUTION_COMPARISON_ATOL,
        label1="PCG",
        label2="FCG",
    )
