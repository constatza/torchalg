"""Shared fixtures for ``torchalg.sparse`` tests.

All test data is created via fixtures - never inline in test functions -
per project convention. Reuses ``tests/conftest.py``'s ``poisson_1d_factory``
as the canonical sparse-capable SPD matrix source.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import pytest
import torch

from torchalg.sparse.device_policy import (
    ITERATIVE_CUDA_CROSSOVER_N,
    PARALLEL_KERNEL_CUDA_CROSSOVER_N,
)
from torchalg.sparse.preconditioners.amg.smoothers import GaussSeidelStage
from torchalg.utils.spectral import approximate_spectral_radius


@pytest.fixture
def parallel_kernel_below_crossover_n() -> int:
    """Problem size immediately below the parallel-kernel CUDA threshold."""
    return PARALLEL_KERNEL_CUDA_CROSSOVER_N - 1


@pytest.fixture
def parallel_kernel_at_crossover_n() -> int:
    """Problem size at the parallel-kernel CUDA threshold."""
    return PARALLEL_KERNEL_CUDA_CROSSOVER_N


@pytest.fixture
def iterative_below_crossover_n() -> int:
    """Problem size immediately below the iterative CUDA threshold."""
    return ITERATIVE_CUDA_CROSSOVER_N - 1


@pytest.fixture
def iterative_at_crossover_n() -> int:
    """Problem size at the iterative CUDA threshold."""
    return ITERATIVE_CUDA_CROSSOVER_N


@pytest.fixture
def largest_measured_elementwise_n() -> int:
    """Largest tested elementwise problem size from the advisory study."""
    return 32_041


@pytest.fixture
def poisson_1d_dense(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    """Dense 12x12 1D Poisson matrix (tridiagonal [-1, 2, -1]).

    Args:
        poisson_1d_factory: Size-parametrized Poisson matrix factory
            (``tests/conftest.py``).

    Returns:
        torch.Tensor: Dense 12x12 SPD tridiagonal matrix.
    """
    return poisson_1d_factory(12)


@pytest.fixture
def poisson_1d_csr(poisson_1d_dense: torch.Tensor) -> torch.Tensor:
    """``poisson_1d_dense`` converted to sparse CSR.

    Args:
        poisson_1d_dense: Dense 12x12 Poisson matrix fixture.

    Returns:
        torch.Tensor: Sparse CSR twin of ``poisson_1d_dense``.
    """
    return poisson_1d_dense.to_sparse_csr()


@pytest.fixture
def poisson_1d_large_dense(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    """Dense 64x64 1D Poisson matrix (tridiagonal [-1, 2, -1]).

    Large enough to coarsen meaningfully across more than two AMG levels
    (``poisson_1d_dense``'s 12 dofs collapse too fast for an ``n_levels=3+``
    hierarchy to stay interesting).

    Args:
        poisson_1d_factory: Size-parametrized Poisson matrix factory
            (``tests/conftest.py``).

    Returns:
        torch.Tensor: Dense 64x64 SPD tridiagonal matrix.
    """
    return poisson_1d_factory(64)


@pytest.fixture
def poisson_1d_large_csr(poisson_1d_large_dense: torch.Tensor) -> torch.Tensor:
    """``poisson_1d_large_dense`` converted to sparse CSR.

    Args:
        poisson_1d_large_dense: Dense 64x64 Poisson matrix fixture.

    Returns:
        torch.Tensor: Sparse CSR twin of ``poisson_1d_large_dense``.
    """
    return poisson_1d_large_dense.to_sparse_csr()


@pytest.fixture
def poisson_1d_xlarge_dense(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    """Dense 300x300 1D Poisson matrix (tridiagonal [-1, 2, -1]).

    Larger than ``poisson_1d_large_dense`` specifically to exercise
    ``BAMGCoarsening._prolongation``'s vectorized candidate-gathering at a
    scale where fine rows plausibly land with several distinct
    coarse-neighbor candidate-set sizes (exercising the padded/masked
    layout's ragged-to-padded packing, not just the uniform case a small
    fixture could pass by accident).

    Args:
        poisson_1d_factory: Size-parametrized Poisson matrix factory
            (``tests/conftest.py``).

    Returns:
        torch.Tensor: Dense 300x300 SPD tridiagonal matrix.
    """
    return poisson_1d_factory(300)


@pytest.fixture
def bamg_default_test_vectors(
    poisson_1d_large_dense: torch.Tensor, torch_dtype: torch.dtype, test_seed: int
) -> torch.Tensor:
    """Seeded default-width BAMG vectors for dense/CSR value-parity checks."""
    generator = torch.Generator().manual_seed(test_seed)
    return torch.randn((poisson_1d_large_dense.shape[0], 8), dtype=torch_dtype, generator=generator)


@pytest.fixture
def poisson_1d_large_rhs(poisson_1d_large_dense: torch.Tensor) -> torch.Tensor:
    """Deterministic right-hand side for large sparse/dense solver parity tests."""
    return torch.ones(poisson_1d_large_dense.shape[0], dtype=poisson_1d_large_dense.dtype)


@pytest.fixture
def poisson_2d_dense(anisotropic_2d_factory: Callable[[int, float], torch.Tensor]) -> torch.Tensor:
    """Dense 9x9 2D isotropic Poisson matrix (5-point stencil on a 3x3 grid).

    Unlike ``poisson_1d_dense`` (tridiagonal, exactly one row per dependency
    level), this gives a genuine multi-row-per-level schedule, exercising
    wavefront grouping in ``triangular.level_schedule`` and vectorized row
    updates in ``triangular_solve``.

    Args:
        anisotropic_2d_factory: 2D anisotropic Poisson factory
            (``tests/conftest.py``).

    Returns:
        torch.Tensor: Dense 9x9 SPD 5-point-stencil matrix.
    """
    return anisotropic_2d_factory(3, 1.0)


@pytest.fixture
def poisson_2d_csr(poisson_2d_dense: torch.Tensor) -> torch.Tensor:
    """``poisson_2d_dense`` converted to sparse CSR.

    Args:
        poisson_2d_dense: Dense 9x9 Poisson matrix fixture.

    Returns:
        torch.Tensor: Sparse CSR twin of ``poisson_2d_dense``.
    """
    return poisson_2d_dense.to_sparse_csr()


class DenseJacobiSweeps(Protocol):
    """Dense weighted-Jacobi oracle: ``steps`` sweeps of ``x + omega D^-1 (rhs - A x)``."""

    def __call__(
        self,
        matrix: torch.Tensor,
        rhs: torch.Tensor,
        x: torch.Tensor,
        steps: int,
        omega: float | None = None,
    ) -> torch.Tensor: ...


class DenseSymmetricGaussSeidel(Protocol):
    """Dense symmetric Gauss-Seidel oracle built from ``solve_triangular`` directly."""

    def __call__(
        self,
        matrix: torch.Tensor,
        x: torch.Tensor,
        rhs: torch.Tensor,
        iterations: int,
        rows: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


_ORACLE_SEED = 0
_DIAGONAL_TOL = 1e-14


@pytest.fixture
def dense_jacobi_sweeps() -> DenseJacobiSweeps:
    """Dense weighted-Jacobi reference from first principles.

    ``omega=None`` is the relaxation rule ``1 / rho(D^-1 A)`` with ``rho``
    from the shared Arnoldi estimate on the dense scaled matrix and the
    fixed-seed start vector the sparse tree uses. Independent of any dense
    smoother class so the oracle stays valid while that API changes.
    """

    def _sweeps(
        matrix: torch.Tensor,
        rhs: torch.Tensor,
        x: torch.Tensor,
        steps: int,
        omega: float | None = None,
    ) -> torch.Tensor:
        diagonal = torch.diagonal(matrix)
        if omega is None:
            generator = torch.Generator().manual_seed(_ORACLE_SEED)

            def draw(size: int) -> torch.Tensor:
                return torch.rand(size, generator=generator, dtype=torch.float64)

            rho = approximate_spectral_radius(matrix / diagonal.unsqueeze(1), draw)
            omega = float(1.0 / rho)
        inverse = omega / diagonal
        column = (slice(None),) + (None,) * (rhs.ndim - 1)
        for _ in range(steps):
            x = x + inverse[column] * (rhs - matrix @ x)
        return x

    return _sweeps


@pytest.fixture
def dense_symmetric_gauss_seidel() -> DenseSymmetricGaussSeidel:
    """Dense symmetric Gauss-Seidel reference using ``solve_triangular`` directly.

    A forward sweep solves ``(D + L) x = rhs - U x_old`` and a backward sweep
    ``(D + U) x = rhs - L x_old``. Rows outside ``rows`` (or with a zero
    diagonal) become identity equations frozen to their current (respectively
    target) value, matching the indexed Gauss-Seidel semantics.
    """

    def _sweeps(
        matrix: torch.Tensor,
        x: torch.Tensor,
        rhs: torch.Tensor,
        iterations: int,
        rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = matrix.shape[0]
        inactive = torch.zeros(n, dtype=torch.bool)
        if rows is not None:
            inactive = torch.ones(n, dtype=torch.bool)
            inactive[rows] = False
        frozen = inactive | (torch.diagonal(matrix).abs() <= _DIAGONAL_TOL)
        identity = torch.eye(n, dtype=matrix.dtype)
        column = (slice(None),) + (None,) * (rhs.ndim - 1)

        def sweep(triangle: torch.Tensor, strict: torch.Tensor, upper: bool) -> torch.Tensor:
            target = rhs - strict @ x
            freeze = torch.where(inactive[column], x, target)
            system = torch.where(frozen.unsqueeze(1), identity, triangle)
            rhs_frozen = torch.where(frozen[column], freeze, target)
            vector = rhs_frozen if rhs.ndim == 2 else rhs_frozen.unsqueeze(1)
            solved = torch.linalg.solve_triangular(system, vector, upper=upper)
            return solved if rhs.ndim == 2 else solved.squeeze(1)

        for _ in range(iterations):
            x = sweep(torch.tril(matrix), torch.triu(matrix, 1), upper=False)
            x = sweep(torch.triu(matrix), torch.tril(matrix, -1), upper=True)
        return x

    return _sweeps


class _PreparedDenseOracle:
    """Dense Gauss-Seidel oracle bound to one matrix (the ``PreparedSmoother`` shape)."""

    def __init__(self, matrix: torch.Tensor, sweeps: DenseSymmetricGaussSeidel) -> None:
        self._matrix = matrix
        self._sweeps = sweeps

    def smooth(self, rhs: torch.Tensor, x: torch.Tensor, steps: int) -> torch.Tensor:
        return self._sweeps(self._matrix, x, rhs, steps)

    def smooth_from_zero(self, rhs: torch.Tensor, steps: int) -> torch.Tensor:
        return self._sweeps(self._matrix, torch.zeros_like(rhs), rhs, steps)


class DenseRelaxation:
    """Dense ``MultigridSmoother`` backed by the ``solve_triangular`` oracle."""

    def __init__(self, sweeps: DenseSymmetricGaussSeidel) -> None:
        self._sweeps = sweeps

    def prepare(self, A: torch.Tensor) -> _PreparedDenseOracle:
        return _PreparedDenseOracle(A, self._sweeps)


@pytest.fixture
def dense_relaxation(dense_symmetric_gauss_seidel: DenseSymmetricGaussSeidel) -> DenseRelaxation:
    """Dense Gauss-Seidel smoother (``prepare`` -> prepared sweeps) from the oracle."""
    return DenseRelaxation(dense_symmetric_gauss_seidel)


@pytest.fixture
def sparse_relaxation() -> GaussSeidelStage:
    """Fresh stage-scoped sparse Gauss-Seidel relaxation callable."""
    return GaussSeidelStage()


@pytest.fixture
def zero_diagonal_csr(poisson_2d_dense: torch.Tensor) -> torch.Tensor:
    """``poisson_2d_dense`` with one structurally zero diagonal entry, as sparse CSR.

    Exercises the frozen-pivot row of the Gauss-Seidel sweeps.
    """
    matrix = poisson_2d_dense.clone()
    matrix[4, 4] = 0.0
    return matrix.to_sparse_csr()


@pytest.fixture
def gauss_seidel_block(poisson_2d_dense: torch.Tensor, test_seed: int) -> tuple[torch.Tensor, ...]:
    """Seeded ``(rhs, x, rows)``: a 2-column block, an iterate, and an indexed row subset."""
    generator = torch.Generator().manual_seed(test_seed)
    n = poisson_2d_dense.shape[0]
    rhs = torch.randn(n, 2, generator=generator, dtype=poisson_2d_dense.dtype)
    x = torch.randn(n, 2, generator=generator, dtype=poisson_2d_dense.dtype)
    rows = torch.tensor([0, 2, 3, 7])
    return rhs, x, rows


@pytest.fixture
def csr_gauss_seidel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the level-scheduled CSR Gauss-Seidel, whatever the matrix size.

    For tests that assert properties of the CSR path (schedules built once per matrix).
    """
    from torchalg.sparse.preconditioners.amg import _gauss_seidel_format

    monkeypatch.setattr(_gauss_seidel_format, "DENSE_ALWAYS_MAX_N", 0)
    monkeypatch.setattr(_gauss_seidel_format, "CSR_ALWAYS_MIN_N", 0)
