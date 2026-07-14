"""Benchmark: torchalg PCG vs scipy CG exactness."""

from __future__ import annotations

import pytest
import torch

from tests.benchmarks.exactness.conftest import (
    BENCHMARK_SIZES,
    MATRIX_TYPES,
    SOLUTION_COMPARISON_ATOL,
    SOLUTION_COMPARISON_RTOL,
    assert_solutions_match,
    generate_spd_system,
)
from tests.support.scipy_reference.preconditioners.implementations.identity import (
    Identity as SciPyIdentity,
)
from tests.support.scipy_reference.scipy_wrapper import SciPyCGSolver
from torchalg import pcg
from torchalg.monitoring import TraceMode


def _is_slow_case(size: int) -> bool:
    """Return whether the exactness case should be marked slow."""
    return size >= 500


@pytest.mark.benchmark
@pytest.mark.parametrize("size", BENCHMARK_SIZES)
@pytest.mark.parametrize("matrix_type", MATRIX_TYPES)
def test_pcg_scipy_exact_match(
    size: int,
    matrix_type: str,
    convergence_tolerances: tuple[float, float],
    request: pytest.FixtureRequest,
) -> None:
    """Torch PCG and scipy CG match iteration counts on deterministic SPD systems."""
    if _is_slow_case(size):
        request.node.add_marker(pytest.mark.slow)

    matrix_np, rhs_np, _, _ = generate_spd_system(size, matrix_type)
    matrix = torch.from_numpy(matrix_np).clone()
    rhs = torch.from_numpy(rhs_np).clone()
    rtol, atol = convergence_tolerances

    scipy_solver = SciPyCGSolver(preconditioner=SciPyIdentity())
    x_scipy, result_scipy = scipy_solver.solve(
        matrix_np,
        rhs_np,
        rtol=rtol,
        atol=atol,
        maxiter=1000,
        trace_mode=TraceMode.MINIMAL.value,
    )
    x_pcg, result_pcg = pcg(
        matrix,
        rhs,
        rtol=rtol,
        atol=atol,
        maxiter=1000,
        trace_mode=TraceMode.MINIMAL,
    )

    assert result_scipy.converged is True
    assert result_pcg.converged is True
    assert result_pcg.iterations == result_scipy.iterations
    assert_solutions_match(
        matrix_np,
        rhs_np,
        x_pcg.detach().cpu().numpy(),
        x_scipy,
        rtol=SOLUTION_COMPARISON_RTOL,
        atol=SOLUTION_COMPARISON_ATOL,
        label1="PCG",
        label2="SciPy",
    )
