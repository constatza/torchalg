"""Shared fixtures for preconditioner tests.

Ported from the reference (``dl-experiments``'s
``tests/solver/preconditioners/conftest.py``) fixture set, adapted from
numpy to torch tensors. All test data is created via fixtures - never
inline in test functions - per project convention.

Stage 7 adds the ``minimal_predictor``/``capturing_predictor`` fixtures
(backed by ``_MinimalPredictor``/``_CapturingPredictor`` test doubles),
mirroring the reference's ``PredictorPort``/``ExtraInputPredictorPort``
fixtures - now these ports exist (``preconditioners/ports.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from torchalg.preconditioners.base import PreconditionerContext
from torchalg.preconditioners.implementations.jacobi import JacobiPreconditioner
from torchalg.preconditioners.implementations.pod import compute_pod_basis
from torchalg.preconditioners.ports import ExtraInputPredictorPort, PredictorPort

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@pytest.fixture
def small_diagonal_matrix(torch_dtype: torch.dtype) -> torch.Tensor:
    """Small 3x3 diagonal matrix for base/bind_inputs tests.

    Returns:
        torch.Tensor: 3x3 diagonal matrix with entries ``[2, 4, 8]``.
    """
    return torch.diag(torch.tensor([2.0, 4.0, 8.0], dtype=torch_dtype))


@pytest.fixture
def jacobi(small_diagonal_matrix: torch.Tensor) -> JacobiPreconditioner:
    """``JacobiPreconditioner`` built from ``small_diagonal_matrix``.

    Args:
        small_diagonal_matrix: 3x3 diagonal matrix fixture.

    Returns:
        JacobiPreconditioner: Fresh instance.
    """
    return JacobiPreconditioner(small_diagonal_matrix)


@pytest.fixture
def well_conditioned_matrix(torch_dtype: torch.dtype) -> torch.Tensor:
    """Well-conditioned 4x4 SPD matrix for testing.

    Returns:
        torch.Tensor: 4x4 diagonal matrix with condition number ~2.
    """
    return torch.diag(torch.tensor([4.0, 3.0, 2.0, 2.0], dtype=torch_dtype))


@pytest.fixture
def residual_vector(torch_dtype: torch.dtype) -> torch.Tensor:
    """Test residual vector.

    Returns:
        torch.Tensor: 4D residual vector.
    """
    return torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch_dtype)


@pytest.fixture
def residual(torch_dtype: torch.dtype) -> torch.Tensor:
    """4-element residual vector for port contract tests.

    Returns:
        torch.Tensor: 4D vector of ones.
    """
    return torch.ones(4, dtype=torch_dtype)


class _MinimalPredictor(PredictorPort):
    """Minimal implementation of base ``PredictorPort`` - ignores extra inputs."""

    def apply(self, residual: torch.Tensor, **extra_inputs: torch.Tensor) -> torch.Tensor:
        """Return a clone of the residual unchanged.

        Args:
            residual (torch.Tensor): Input residual vector.
            **extra_inputs (torch.Tensor): Accepted but ignored.

        Returns:
            torch.Tensor: Clone of the residual.
        """
        return residual.clone()

    def cleanup(self) -> None:
        """No-op cleanup."""


class _CapturingPredictor(ExtraInputPredictorPort):
    """``ExtraInputPredictorPort`` that records extra inputs for assertion."""

    def __init__(self) -> None:
        """Initialize with an empty extra-inputs record."""
        self.last_extra: dict[str, torch.Tensor] = {}

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Return an empty tuple - capturing predictor accepts any inputs.

        Returns:
            tuple[str, ...]: Empty tuple of required input names.
        """
        return ()

    def apply(self, residual: torch.Tensor, **extra_inputs: torch.Tensor) -> torch.Tensor:
        """Record extra inputs and return a clone of the residual.

        Args:
            residual (torch.Tensor): Input residual vector.
            **extra_inputs (torch.Tensor): Named extra tensors to capture.

        Returns:
            torch.Tensor: Clone of the residual.
        """
        self.last_extra = extra_inputs
        return residual.clone()

    def cleanup(self) -> None:
        """No-op cleanup."""


@pytest.fixture
def capturing_predictor() -> _CapturingPredictor:
    """``ExtraInputPredictorPort`` that captures extra inputs passed to ``apply()``.

    Returns:
        _CapturingPredictor: Fresh instance.
    """
    return _CapturingPredictor()


@pytest.fixture
def minimal_predictor() -> _MinimalPredictor:
    """Minimal ``PredictorPort`` implementation - passes residual through unchanged.

    Returns:
        _MinimalPredictor: Fresh instance.
    """
    return _MinimalPredictor()


@pytest.fixture
def small_diagonal_matrix_residual(small_diagonal_matrix: torch.Tensor) -> torch.Tensor:
    """Residual vector matching ``small_diagonal_matrix``'s diagonal entries.

    Chosen so that Jacobi scaling (``z = D^{-1}r``) reduces every component
    to exactly ``1.0``, giving a hand-computable expected result.

    Args:
        small_diagonal_matrix: 3x3 diagonal matrix fixture.

    Returns:
        torch.Tensor: ``[2.0, 4.0, 8.0]``, cloned from
            ``small_diagonal_matrix``'s diagonal.
    """
    return torch.diagonal(small_diagonal_matrix).clone()


@pytest.fixture
def tridiagonal_spd_small_torch(
    tridiagonal_spd_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> torch.Tensor:
    """Torch twin of ``tridiagonal_spd_small`` (``tests/solver/conftest.py``).

    Args:
        tridiagonal_spd_small: 10x10 numpy tridiagonal SPD matrix.
        to_torch: Numpy -> torch adapter fixture.

    Returns:
        torch.Tensor: Cloned ``torch_dtype`` tensor twin.
    """
    return to_torch(tridiagonal_spd_small)


@pytest.fixture
def rhs_ones_small_torch(
    rhs_ones_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> torch.Tensor:
    """Torch twin of ``rhs_ones_small`` (``tests/solver/conftest.py``).

    Args:
        rhs_ones_small: 10D numpy vector of ones.
        to_torch: Numpy -> torch adapter fixture.

    Returns:
        torch.Tensor: Cloned ``torch_dtype`` tensor twin.
    """
    return to_torch(rhs_ones_small)


@pytest.fixture
def mixed_sign_diagonal_matrix(torch_dtype: torch.dtype) -> torch.Tensor:
    """5x5 diagonal matrix with alternating-sign entries.

    Returns:
        torch.Tensor: ``diag([2.0, -3.0, 4.0, -5.0, 1.0])``.
    """
    return torch.diag(torch.tensor([2.0, -3.0, 4.0, -5.0, 1.0], dtype=torch_dtype))


@pytest.fixture
def mixed_sign_residual(torch_dtype: torch.dtype) -> torch.Tensor:
    """5-element residual vector with alternating signs.

    Returns:
        torch.Tensor: ``[1.0, -1.0, 2.0, -2.0, 0.5]``.
    """
    return torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5], dtype=torch_dtype)


# =============================================================================
# ILU / IC(0) / ICholesky fixtures (Stage 4)
# =============================================================================


@pytest.fixture
def dense_spd_matrix(torch_dtype: torch.dtype) -> torch.Tensor:
    """Dense 5x5 SPD tridiagonal matrix for ILU testing.

    Returns:
        torch.Tensor: 5x5 tridiagonal matrix, diag=2, off-diag=-1.
    """
    n = 5
    return (
        2 * torch.eye(n, dtype=torch_dtype)
        - torch.diag(torch.ones(n - 1, dtype=torch_dtype), 1)
        - torch.diag(torch.ones(n - 1, dtype=torch_dtype), -1)
    )


@pytest.fixture
def dense_spd_matrix_residual(torch_dtype: torch.dtype) -> torch.Tensor:
    """5-element residual vector of ones, matching ``dense_spd_matrix``'s size.

    Returns:
        torch.Tensor: ``[1.0, 1.0, 1.0, 1.0, 1.0]``.
    """
    return torch.ones(5, dtype=torch_dtype)


@pytest.fixture
def random_spd_small_torch(
    random_spd_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> torch.Tensor:
    """Torch twin of ``random_spd_small`` (``tests/solver/conftest.py``).

    Args:
        random_spd_small: 10x10 numpy random SPD matrix.
        to_torch: Numpy -> torch adapter fixture.

    Returns:
        torch.Tensor: Cloned ``torch_dtype`` tensor twin.
    """
    return to_torch(random_spd_small)


@pytest.fixture
def rhs_random_small_torch(
    rhs_random_small: NDArray,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> torch.Tensor:
    """Torch twin of ``rhs_random_small`` (``tests/solver/conftest.py``).

    Args:
        rhs_random_small: 10D numpy random vector.
        to_torch: Numpy -> torch adapter fixture.

    Returns:
        torch.Tensor: Cloned ``torch_dtype`` tensor twin.
    """
    return to_torch(rhs_random_small)


@pytest.fixture
def cholesky_factor_random_spd_small_torch(random_spd_small_torch: torch.Tensor) -> torch.Tensor:
    """Exact Cholesky factor ``L`` of ``random_spd_small_torch``.

    Args:
        random_spd_small_torch: 10x10 random SPD matrix.

    Returns:
        torch.Tensor: Lower triangular ``L`` such that ``L @ L.T`` exactly
            reconstructs ``random_spd_small_torch``.
    """
    return torch.linalg.cholesky(random_spd_small_torch)


@pytest.fixture
def identity_matrix_5(torch_dtype: torch.dtype) -> torch.Tensor:
    """5x5 identity matrix, for ICholesky shape-handling tests.

    Returns:
        torch.Tensor: ``torch.eye(5)``.
    """
    return torch.eye(5, dtype=torch_dtype)


@pytest.fixture
def cholesky_factor_identity_5(identity_matrix_5: torch.Tensor) -> torch.Tensor:
    """Cholesky factor of ``identity_matrix_5`` (trivially the identity itself).

    Args:
        identity_matrix_5: 5x5 identity matrix.

    Returns:
        torch.Tensor: ``L = I``.
    """
    return torch.linalg.cholesky(identity_matrix_5)


@pytest.fixture
def ones_residual_5(torch_dtype: torch.dtype) -> torch.Tensor:
    """5-element residual vector of ones.

    Returns:
        torch.Tensor: ``torch.ones(5)``.
    """
    return torch.ones(5, dtype=torch_dtype)


@pytest.fixture
def ones_residual_5x2(torch_dtype: torch.dtype) -> torch.Tensor:
    """5x2 batch of residual columns, all ones - for batched-apply tests.

    Returns:
        torch.Tensor: ``torch.ones(5, 2)``.
    """
    return torch.ones(5, 2, dtype=torch_dtype)


@pytest.fixture
def tridiagonal_spd_small_perturbed_torch(
    tridiagonal_spd_small: NDArray,
    test_seed: int,
    to_torch: Callable[[NDArray], torch.Tensor],
) -> torch.Tensor:
    """``tridiagonal_spd_small`` with tiny (~1e-12) seeded lower-triangular noise added.

    Mirrors the reference's perturbation used to exercise IC(0)'s
    ``threshold`` parameter: small enough to be dropped by an aggressive
    threshold but not by a strict one.

    Args:
        tridiagonal_spd_small: 10x10 numpy tridiagonal SPD matrix.
        test_seed: Fixed random seed for reproducibility.
        to_torch: Numpy -> torch adapter fixture.

    Returns:
        torch.Tensor: Perturbed, re-symmetrized, diagonally-boosted SPD
            matrix.
    """
    rng = np.random.default_rng(test_seed)
    perturbed = tridiagonal_spd_small.copy()
    perturbed += np.tril(rng.standard_normal((10, 10)) * 1e-12)
    perturbed = (perturbed + perturbed.T) / 2
    perturbed += 0.1 * np.eye(10)
    return to_torch(perturbed)


@pytest.fixture
def near_singular_matrix_torch(torch_dtype: torch.dtype) -> torch.Tensor:
    """10x10 identity matrix with one near-zero diagonal entry.

    Returns:
        torch.Tensor: ``I`` with ``A[5, 5] = 1e-15``.
    """
    matrix = torch.eye(10, dtype=torch_dtype)
    matrix[5, 5] = 1e-15
    return matrix


@pytest.fixture
def non_spd_matrix_2x2_torch(torch_dtype: torch.dtype) -> torch.Tensor:
    """2x2 non-symmetric, non-SPD matrix - IC(0) must raise ValueError on construction.

    Returns:
        torch.Tensor: ``[[1, 2], [3, 4]]``.
    """
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch_dtype)


# =============================================================================
# AMG fixtures (Stage 5)
# =============================================================================


@pytest.fixture
def poisson_1d(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    """20x20 1D Poisson matrix (tridiagonal [-1, 2, -1]) for AMG unit tests.

    Args:
        poisson_1d_factory: Size-parametrized Poisson matrix factory
            (``tests/conftest.py``).

    Returns:
        torch.Tensor: Dense 20x20 SPD tridiagonal matrix.
    """
    return poisson_1d_factory(20)


@pytest.fixture
def poisson_rhs(poisson_1d: torch.Tensor) -> torch.Tensor:
    """Right-hand side vector of ones matching ``poisson_1d``'s size.

    Args:
        poisson_1d: The 20x20 Poisson matrix fixture.

    Returns:
        torch.Tensor: Vector of ones with length 20.
    """
    return torch.ones(poisson_1d.shape[0], dtype=poisson_1d.dtype)


@pytest.fixture
def small_spd_amg(torch_dtype: torch.dtype) -> torch.Tensor:
    """Small 6x6 SPD diagonal matrix for AMG smoother unit tests.

    Returns:
        torch.Tensor: 6x6 diagonal matrix (well-conditioned, trivial
            aggregation): ``diag([4, 3, 4, 3, 4, 3])``.
    """
    return torch.diag(torch.tensor([4.0, 3.0, 4.0, 3.0, 4.0, 3.0], dtype=torch_dtype))


@pytest.fixture
def soft_inclusion_2d_stiffness(torch_dtype: torch.dtype) -> torch.Tensor:
    """25-node 2D 5-point-stencil stiffness matrix with a soft circular inclusion.

    Assembles a 5x5 grid Laplacian-style stiffness matrix (row-sum-zero
    off-diagonals, positive diagonal), then scales every stencil edge
    touching a central disk-shaped patch (Euclidean radius 2 around the
    grid center, 13 of the 25 nodes) down by a factor of 1000x - a
    "soft sphere inside a stiffer medium" heterogeneity, the realistic
    case `TargetDimensionCoarsening` targets rather than a flat/uniform
    stencil.

    Empirically (see
    `TestTargetDimensionCoarsening.test_realized_dimension_is_not_monotonic_in_theta`),
    this matrix's `standard_aggregation` coarse dimension jumps from 6 to 8
    as `theta` increases from 0.01 to 0.02 - the non-monotonicity
    `TargetDimensionCoarsening`'s exhaustive grid search exists to guard
    against.

    Args:
        torch_dtype: Default dtype for the returned matrix.

    Returns:
        torch.Tensor: Dense 25x25 SPD stiffness matrix.
    """
    grid_size = 5
    inclusion_radius = 2
    softness = 1000.0
    center = grid_size // 2

    def node_index(row: int, col: int) -> int:
        return row * grid_size + col

    def is_soft(row: int, col: int) -> bool:
        return (row - center) ** 2 + (col - center) ** 2 <= inclusion_radius**2

    matrix = torch.zeros(grid_size**2, grid_size**2, dtype=torch_dtype)
    for row in range(grid_size):
        for col in range(grid_size):
            neighbors = [
                (row + delta_row, col + delta_col)
                for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= row + delta_row < grid_size and 0 <= col + delta_col < grid_size
            ]
            for neighbor_row, neighbor_col in neighbors:
                weight = (
                    1.0 / softness
                    if is_soft(row, col) or is_soft(neighbor_row, neighbor_col)
                    else 1.0
                )
                matrix[node_index(row, col), node_index(neighbor_row, neighbor_col)] = -weight
    matrix.diagonal().copy_(-matrix.sum(dim=1))
    return matrix


@pytest.fixture
def zero_iteration_context() -> PreconditionerContext:
    """Dummy ``PreconditionerContext`` at iteration 0.

    Returns:
        PreconditionerContext: ``iteration=0, residual_norm=1.0, rhs_norm=1.0``.
    """
    return PreconditionerContext(iteration=0, residual_norm=1.0, rhs_norm=1.0)


# =============================================================================
# POD fixtures (Stage 6)
# =============================================================================


@pytest.fixture
def poisson_snapshots(poisson_1d: torch.Tensor, test_seed: int) -> torch.Tensor:
    """Snapshot ensemble spanning ``poisson_1d``'s solution space.

    Solves ``poisson_1d @ x = b_i`` for several random RHS vectors, so a
    full-rank POD basis recovers the solution space exactly and a truncated
    basis approximates it.

    Args:
        poisson_1d: The 20x20 Poisson matrix fixture.
        test_seed: Fixed random seed for reproducibility.

    Returns:
        torch.Tensor: Snapshot ensemble, shape (15, 20) - one solution
            vector per row.
    """
    generator = torch.Generator().manual_seed(test_seed)
    n = poisson_1d.shape[0]
    rhs_batch = torch.randn(15, n, dtype=poisson_1d.dtype, generator=generator)
    return torch.linalg.solve(poisson_1d, rhs_batch.T).T


@pytest.fixture
def pod_basis(poisson_snapshots: torch.Tensor) -> torch.Tensor:
    """POD basis (rank 10) built from ``poisson_snapshots``.

    Args:
        poisson_snapshots: Snapshot ensemble fixture.

    Returns:
        torch.Tensor: Phi_r, shape (20, 10).
    """
    return compute_pod_basis(poisson_snapshots, rank=10)


@pytest.fixture
def snapshot_row_scales(poisson_snapshots: torch.Tensor) -> torch.Tensor:
    """Distinct, strictly positive per-snapshot row scales for weighted-POD tests.

    Args:
        poisson_snapshots: Snapshot ensemble fixture.

    Returns:
        torch.Tensor: Shape (15,), values spread over [0.5, 2.0] so no two
            snapshots carry the same weight.
    """
    return torch.linspace(0.5, 2.0, poisson_snapshots.shape[0], dtype=poisson_snapshots.dtype)


@pytest.fixture
def single_dominant_row_scales(poisson_snapshots: torch.Tensor) -> torch.Tensor:
    """Row scales that leave only the first snapshot with meaningful weight.

    Makes the weighted covariance effectively rank-1, so the leading POD mode
    must align with the first snapshot's direction - the sharpest observable
    consequence of row weighting reaching the SVD.

    Args:
        poisson_snapshots: Snapshot ensemble fixture.

    Returns:
        torch.Tensor: Shape (15,), ``[1.0, 1e-8, 1e-8, ...]``.
    """
    scales = torch.full((poisson_snapshots.shape[0],), 1e-8, dtype=poisson_snapshots.dtype)
    scales[0] = 1.0
    return scales


@pytest.fixture
def snapshots_with_zero_row(poisson_snapshots: torch.Tensor) -> torch.Tensor:
    """``poisson_snapshots`` with its first row zeroed out.

    A zero snapshot has no direction and an undefined normalization, so it
    exercises the near-zero guards in the weighting helpers.

    Args:
        poisson_snapshots: Snapshot ensemble fixture.

    Returns:
        torch.Tensor: Shape (15, 20) with row 0 all zeros.
    """
    snapshots = poisson_snapshots.clone()
    snapshots[0] = 0.0
    return snapshots
