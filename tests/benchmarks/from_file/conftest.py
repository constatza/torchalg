"""Fixtures for from-file benchmarks.

Ported from the reference (``dl-experiments``'s
``tests/benchmarks/from_file/conftest.py``), adapted from numpy to torch
tensors. The reference loaded matrices via
``neuralls.platform.storage.base.load_matrix`` and resolved user-supplied
paths via ``neuralls.platform.config.resolution.resolve_user_path`` -
neuralls-specific config/storage infrastructure this repo has no reason to
depend on. Loading here is plain ``numpy.loadtxt`` plus ``pathlib``/
``os.environ``, since the only thing that ever mattered for this benchmark
is the CSV-delimited matrix data itself.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import linalg

DATA_DIR = Path(__file__).with_name("data")
DEFAULT_MATRIX_PATHS = (DATA_DIR / "Ktilde.txt",)
DEFAULT_RHS_PATHS = (DATA_DIR / "ftilde.txt",)
DEFAULT_L_PATH = DATA_DIR / "L.txt"

# Sparsity threshold for determining non-zero entries
SPARSITY_THRESHOLD = 1e-19  # Entries with |value| <= this are considered zero

# Convergence tolerances for CG solvers
CONVERGENCE_RTOL = 1e-12  # Relative tolerance for convergence
CONVERGENCE_ATOL = 1e-14  # Absolute tolerance for convergence

# IC0 comparison tolerances
IC0_MATRIX_RTOL = 1e-12  # Relative tolerance for L matrix comparison
IC0_MATRIX_ATOL = 1e-14  # Absolute tolerance for L matrix comparison
IC0_SPARSITY_COUNT_TOLERANCE = (
    5  # Allow up to 5 non-zero count difference due to numerical precision
)
IC0_SOLUTION_RTOL = 1e-10  # Relative tolerance for solution comparison
IC0_SOLUTION_ATOL = 1e-12  # Absolute tolerance for solution comparison
IC0_THRESHOLD = 0.0  # Use no thresholding to match reference
# Note: Integer metrics (sparsity counts, iteration counts) use exact equality (zero tolerance)


def _paths_from_env(var_name: str, default_paths: tuple[Path, ...]) -> tuple[Path, ...]:
    value = os.environ.get(var_name)
    if value is None:
        return default_paths
    return tuple(Path(item).expanduser() for item in value.split(",") if item)


MATRIX_PATHS = _paths_from_env("BENCHMARK_MATRIX_PATHS", DEFAULT_MATRIX_PATHS)
RHS_PATHS = _paths_from_env("BENCHMARK_RHS_PATHS", DEFAULT_RHS_PATHS)
L_PATH = Path(os.environ.get("BENCHMARK_L_PATH", str(DEFAULT_L_PATH))).expanduser()


def check_existence(path: Path | None) -> None:
    """Skip the test if the given path is missing."""
    if path is None or not path.exists():
        pytest.skip(f"Matrix file not found: {path}")


# Use [None] if empty to ensure at least one test is collected and then skipped with a message
@pytest.fixture(params=MATRIX_PATHS if MATRIX_PATHS else [None])
def matrix_path(request: pytest.FixtureRequest) -> Path | None:
    """Matrix file path, skipping the test if missing."""
    path = request.param
    check_existence(path)
    return path


@pytest.fixture(params=RHS_PATHS if RHS_PATHS else [None])
def rhs_path(request: pytest.FixtureRequest) -> Path | None:
    """Right-hand side file path, skipping the test if missing."""
    path = request.param
    check_existence(path)
    return path


@pytest.fixture
def system(
    matrix_path: Path, rhs_path: Path
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load matrix and construct exact solution.

    Returns:
        Tuple ``(A, b, x_exact, L)`` where ``x_exact`` is computed via
        ``scipy.linalg.solve`` on the numpy-loaded arrays before conversion.
    """
    a = np.loadtxt(matrix_path, delimiter=",")
    l_ref = np.loadtxt(L_PATH, delimiter=",")
    a[np.absolute(a) <= SPARSITY_THRESHOLD] = 0
    b = np.loadtxt(rhs_path, delimiter=",")
    x_exact = linalg.solve(a, b)

    return (
        torch.as_tensor(a, dtype=torch.float64),
        torch.as_tensor(b, dtype=torch.float64),
        torch.as_tensor(x_exact, dtype=torch.float64),
        torch.as_tensor(l_ref, dtype=torch.float64),
    )


@pytest.fixture
def convergence_tolerances() -> tuple[float, float]:
    """Tolerances for solver convergence check."""
    return CONVERGENCE_RTOL, CONVERGENCE_ATOL
