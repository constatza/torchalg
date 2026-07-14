"""Fixtures and utilities for CG iteration count exactness benchmarks.

Ported from the reference (``dl-experiments``'s
``tests/benchmarks/exactness/conftest.py``). Tests using these fixtures
should be marked ``@pytest.mark.benchmark`` so they stay opt-in (excluded by
the default ``addopts = ["-m", "not benchmark"]`` in ``pyproject.toml``),
matching the reference's convention for this scipy-equivalence /
Notay-2000 paper-reproduction suite.

``assert_solutions_match``/``check_residuals_match`` reuse
``CombinedToleranceCriterion`` from the frozen scipy oracle snapshot
(``tests/support/scipy_reference/strategies/convergence.py``) exactly as the
reference's own benchmark suite does, to avoid duplicating convergence logic.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import diags

from tests.support.scipy_reference.strategies.convergence import CombinedToleranceCriterion

# System sizes for benchmarks
BENCHMARK_SIZES = [100, 200, 500, 1000]

# Matrix types
MATRIX_TYPES = ["tridiagonal", "diagonal"]

# Single source of truth for all tolerances
EXACTNESS_RTOL = 1e-12
EXACTNESS_ATOL = 1e-14

# Solution comparison tolerances
# Worst case: if both solvers converge within rtol of exact solution,
# they can differ by up to 2*rtol from each other
SOLUTION_COMPARISON_RTOL = 2 * EXACTNESS_RTOL
SOLUTION_COMPARISON_ATOL = 2 * EXACTNESS_ATOL

# Iteration count difference thresholds
ITER_DIFF_THRESHOLD_FCG_PCG = 2  # Allow up to 2 iterations difference for FCG vs PCG


def _create_tridiagonal_spd(n: int) -> NDArray:
    """Create tridiagonal SPD matrix.

    Args:
        n: Matrix size.

    Returns:
        Tridiagonal SPD matrix with diag=4, off-diag=-1.
    """
    sparse_mat = diags(
        [-1.0, 4.0, -1.0],
        [-1, 0, 1],
        shape=(n, n),
        dtype=np.float64,
    )
    return sparse_mat.toarray()


def _estimate_tridiagonal_kappa(n: int) -> float:
    """Estimate condition number for tridiagonal SPD matrix.

    Args:
        n: Matrix size.

    Returns:
        Estimated condition number (kappa ~= 4n^2/pi^2).
    """
    return 4 * n**2 / (np.pi**2)


def generate_spd_system(
    size: int,
    matrix_type: str,
) -> tuple[NDArray, NDArray, NDArray, float]:
    """Generate a deterministic SPD system with known solution.

    Completely deterministic - no randomness. Running with the same
    parameters always produces identical A, b, x_exact.

    Determinism details:
        - Tridiagonal matrix: hardcoded structure (diag=4, off-diag=-1)
        - Diagonal matrix: hardcoded entries (1, 2, ..., n)
        - x_exact: vector of ones (no randomness)
        - b: computed as A @ x_exact (fully deterministic)

    Args:
        size: Matrix size (n x n).
        matrix_type: One of "tridiagonal", "diagonal".

    Returns:
        Tuple of (A, b, x_exact, kappa), all deterministic.

    Raises:
        ValueError: If matrix_type is not recognized.
    """
    match matrix_type:
        case "tridiagonal":
            a = _create_tridiagonal_spd(size)
            kappa = _estimate_tridiagonal_kappa(size)
        case "diagonal":
            a = np.diag(np.arange(1, size + 1, dtype=np.float64))
            kappa = float(size)
        case _:
            raise ValueError(f"Unknown matrix_type: {matrix_type}")

    x_exact = np.ones(size, dtype=np.float64)
    b = a @ x_exact

    return a, b, x_exact, kappa


@pytest.fixture
def convergence_tolerances() -> tuple[float, float]:
    """High-precision convergence tolerances for exactness benchmarks.

    Returns:
        Tuple of (rtol, atol) for solver convergence.
    """
    return EXACTNESS_RTOL, EXACTNESS_ATOL


def check_residual_norm(
    A: NDArray,
    b: NDArray,
    x: NDArray,
    rtol: float,
    atol: float,
    label: str = "solution",
) -> None:
    """Check that solution satisfies convergence criterion.

    Args:
        A: System matrix.
        b: RHS vector.
        x: Solution vector.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        label: Label for error message.

    Raises:
        AssertionError: If solution doesn't satisfy convergence criterion.
    """
    residual = b - A @ x
    residual_norm = np.linalg.norm(residual)
    b_norm = np.linalg.norm(b)
    threshold = max(rtol * b_norm, atol)

    if residual_norm > threshold:
        raise AssertionError(
            f"{label} doesn't satisfy convergence criterion: "
            f"||r||={residual_norm:.2e} > threshold={threshold:.2e} "
            f"(max({rtol:.1e} * ||b||, {atol:.1e}))"
        )


def check_residuals_match(
    A: NDArray,
    b: NDArray,
    x1: NDArray,
    x2: NDArray,
    rtol: float,
    atol: float,
) -> tuple[bool, dict[str, float]]:
    """Check if residuals of two solutions match using convergence criterion.

    Compares residual norms using the combined tolerance criterion:
    ``||r1 - r2|| <= max(rtol * ||b||, atol)``. Reuses
    ``CombinedToleranceCriterion`` to avoid duplicate logic.

    Args:
        A: System matrix.
        b: RHS vector.
        x1: First solution vector.
        x2: Second solution vector.
        rtol: Relative tolerance (already doubled at call site).
        atol: Absolute tolerance (already doubled at call site).

    Returns:
        Tuple of (matches, diagnostics) where matches is True if residuals
        match within tolerance and diagnostics holds residual norms/differences.
    """
    criterion = CombinedToleranceCriterion(rtol=rtol, atol=atol)

    r_diff = A @ (x1 - x2)

    b_norm = float(np.linalg.norm(b))
    matches = criterion.has_converged(r_diff, b_norm)

    r1 = b - A @ x1
    r2 = b - A @ x2
    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))
    diff_norm = float(np.linalg.norm(r_diff))

    diagnostics = {
        "r1_norm": r1_norm,
        "r2_norm": r2_norm,
        "r1_rel": r1_norm / b_norm,
        "r2_rel": r2_norm / b_norm,
        "diff_norm": diff_norm,
        "b_norm": b_norm,
        "threshold": criterion.threshold(b_norm),
    }

    return matches, diagnostics


def assert_solutions_match(
    A: NDArray,
    b: NDArray,
    x1: NDArray,
    x2: NDArray,
    rtol: float,
    atol: float,
    label1: str = "x1",
    label2: str = "x2",
) -> None:
    """Assert residuals of two solutions match using convergence criterion.

    Args:
        A: System matrix.
        b: RHS vector.
        x1: First solution vector.
        x2: Second solution vector.
        rtol: Relative tolerance (already doubled at call site).
        atol: Absolute tolerance (already doubled at call site).
        label1: Name/description of first solution (for error message).
        label2: Name/description of second solution (for error message).

    Raises:
        AssertionError: If residuals don't match within tolerance.
    """
    matches, diag = check_residuals_match(A, b, x1, x2, rtol, atol)

    if not matches:
        raise AssertionError(
            f"Residuals for {label1} and {label2} don't match within tolerance.\n"
            f"  ||r_{label1}|| = {diag['r1_norm']:.3e} "
            f"(relative: {diag['r1_rel']:.3e})\n"
            f"  ||r_{label2}|| = {diag['r2_norm']:.3e} "
            f"(relative: {diag['r2_rel']:.3e})\n"
            f"  ||r_{label1} - r_{label2}|| = {diag['diff_norm']:.3e}\n"
            f"  ||b|| = {diag['b_norm']:.3e}\n"
            f"  Threshold = max({rtol:.1e} * ||b||, {atol:.1e}) = "
            f"{diag['threshold']:.3e}\n"
            f"  Tolerances: rtol={rtol:.1e}, atol={atol:.1e}"
        )
