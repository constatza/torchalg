"""Multigrid eigensolver (MGE) tests (Bootstrap AMG, docs/bootstrap-amg.md Sec. 4.2, Algorithm 1).

Fixtures live in this module, matching ``test_algebraic_distance.py``'s,
``test_compatible_relaxation.py``'s and ``test_least_squares.py``'s
precedent: each ``implementations/amg/`` test module keeps its fixtures
local rather than pre-emptively factoring them into a shared conftest before
a second consumer exists.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torchalg.preconditioners.implementations.amg._mge import (
    coarsest_eigenpairs,
    composite_transfer_metrics,
    multigrid_eigensolver,
    refine_eigenpair,
)
from torchalg.preconditioners.implementations.amg.smoothers import GaussSeidelSmoother


@pytest.fixture
def diagonal_matrix_4(torch_dtype: torch.dtype) -> torch.Tensor:
    """Diagonal SPD matrix, shape ``(4, 4)``, eigenvalues ``[1, 2, 3, 4]`` by construction."""
    return torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch_dtype))


@pytest.fixture
def prolongation_4_to_2(torch_dtype: torch.dtype) -> torch.Tensor:
    """Full-rank prolongation, shape ``(4, 2)``, mapping a 2-node coarse grid up to 4 fine nodes."""
    return torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.0, 0.5]], dtype=torch_dtype)


def test_composite_transfer_metrics_finest_level_is_identity(
    torch_dtype: torch.dtype,
) -> None:
    """[BAMG11]/[STATUS14]: ``T = I`` on the finest level (docs/bootstrap-amg.md line 234)."""
    metrics = composite_transfer_metrics([], n0=5, dtype=torch_dtype, device=torch.device("cpu"))

    assert len(metrics) == 1
    assert torch.equal(metrics[0], torch.eye(5, dtype=torch_dtype))


def test_composite_transfer_metrics_matches_direct_gram_product(
    prolongation_4_to_2: torch.Tensor, torch_dtype: torch.dtype
) -> None:
    """``T_l = P_l^H P_l`` for the composite interpolation ``P_l`` ([BAMG11] eq. 3.2)."""
    metrics = composite_transfer_metrics(
        [prolongation_4_to_2], n0=4, dtype=torch_dtype, device=torch.device("cpu")
    )

    assert len(metrics) == 2
    assert torch.equal(metrics[0], torch.eye(4, dtype=torch_dtype))
    expected_T1 = prolongation_4_to_2.T @ prolongation_4_to_2
    assert torch.allclose(metrics[1], expected_T1)


def test_coarsest_eigenpairs_reduces_to_plain_eigh_when_t_is_identity(
    diagonal_matrix_4: torch.Tensor, torch_dtype: torch.dtype
) -> None:
    """With ``T = I`` the generalized problem is the standard eigenproblem: smallest eigenvalues first."""
    identity = torch.eye(4, dtype=torch_dtype)

    values, vectors = coarsest_eigenpairs(diagonal_matrix_4, identity, k_e=2)

    assert values.shape == (2,)
    assert vectors.shape == (4, 2)
    assert torch.allclose(values, torch.tensor([1.0, 2.0], dtype=torch_dtype), atol=1e-8)
    for k in range(2):
        residual = diagonal_matrix_4 @ vectors[:, k] - values[k] * vectors[:, k]
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-8)


def test_coarsest_eigenpairs_satisfies_generalized_eigenproblem_with_nontrivial_t(
    torch_dtype: torch.dtype,
) -> None:
    """``A w = lambda T w`` holds for a non-identity SPD ``T`` too ([BAMG11] eq. in Algorithm 1)."""
    A = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch_dtype)
    T = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch_dtype)

    values, vectors = coarsest_eigenpairs(A, T, k_e=2)

    assert values.shape == (2,)
    assert values[0] <= values[1], "ascending eigenvalues - k_e smallest kept first"
    for k in range(2):
        residual = A @ vectors[:, k] - values[k] * (T @ vectors[:, k])
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)


def test_coarsest_eigenpairs_clamps_k_e_to_system_size(
    diagonal_matrix_4: torch.Tensor, torch_dtype: torch.dtype
) -> None:
    """Requesting more eigenpairs than the system has returns only what exists, not an error."""
    identity = torch.eye(4, dtype=torch_dtype)

    values, vectors = coarsest_eigenpairs(diagonal_matrix_4, identity, k_e=10)

    assert values.shape == (4,)
    assert vectors.shape == (4, 4)


def test_refine_eigenpair_leaves_an_exact_eigenpair_unchanged(
    torch_dtype: torch.dtype,
) -> None:
    """Relaxing ``(A - lambda T) w = 0`` from an exact solution is a zero-residual no-op for GS."""
    A = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch_dtype)
    T = torch.eye(2, dtype=torch_dtype)
    eigenvalue = 2.0
    eigenvector = torch.tensor([1.0, 0.0], dtype=torch_dtype)  # exact eigenpair of (A, T)

    relaxation = GaussSeidelSmoother().smooth
    refreshed_value, refreshed_vector = refine_eigenpair(
        A, T, eigenvalue, eigenvector, relaxation, sweeps=3
    )

    assert torch.allclose(refreshed_vector, eigenvector, atol=1e-10)
    assert refreshed_value == pytest.approx(eigenvalue, abs=1e-10)


def test_refine_eigenpair_stays_bounded_when_the_shifted_matrix_is_indefinite(
    torch_dtype: torch.dtype,
) -> None:
    """A large ``eigenvalue`` makes ``A - lambda T`` indefinite - GS on an unnormalized
    homogeneous system then has no fixed point and diverges geometrically (measured
    without renormalization, on this same matrix: column norms reaching ``1e99``
    within a handful of sweeps). Renormalizing to unit ``T``-norm after *every*
    sweep, as ordinary Rayleigh-quotient iteration does, keeps the result bounded.
    A small tridiagonal (non-diagonal) matrix, representative of the dense,
    non-diagonal Galerkin operators this module actually runs on - unlike a
    diagonal matrix, where Gauss-Seidel solves the homogeneous system exactly
    in one sweep (to the unique solution ``0``), which is not representative.
    """
    A = torch.tensor([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]], dtype=torch_dtype)
    T = torch.eye(3, dtype=torch_dtype)
    eigenvector = torch.tensor([1.0, 1.0, 1.0], dtype=torch_dtype)

    relaxation = GaussSeidelSmoother().smooth
    _, refreshed_vector = refine_eigenpair(
        A, T, eigenvalue=100.0, eigenvector=eigenvector, relaxation=relaxation, sweeps=8
    )

    assert torch.isfinite(refreshed_vector).all()
    assert torch.allclose(
        refreshed_vector @ (T @ refreshed_vector), torch.tensor(1.0, dtype=torch_dtype)
    )


def test_multigrid_eigensolver_enriches_every_level_except_coarsest(
    poisson_1d_factory: Callable[[int], torch.Tensor],
) -> None:
    """[BAMG11] Algorithm 1 (``l = L,...,1`` in the paper's 1-indexed, finest-last convention -
    every level including the finest in torchalg's 0-indexed, finest-first ``levels`` list)
    produces ``k_e``-column eigenvector approximations for every level except the coarsest,
    which is only the algorithm's starting point, never a *consumer* of enrichment.
    """
    from torchalg.multigrid.bootstrap_setup import BootstrapSetup, _random_test_vectors
    from torchalg.preconditioners.implementations.amg._presets import (
        GS_SETUP_CYCLE,
        seeded_draw,
    )
    from torchalg.preconditioners.implementations.amg.bootstrap import BAMGCoarsening
    from torchalg.preconditioners.implementations.amg.transfer import DenseTransferOperator

    matrix = poisson_1d_factory(16)
    relaxation = GaussSeidelSmoother().smooth
    setup = BootstrapSetup(
        coarsening_factory=lambda vectors, relaxation, draw: BAMGCoarsening(
            vectors, relaxation, caliber=3, draw=draw
        ),
        transfer_operator_factory=DenseTransferOperator,
        relaxation=relaxation,
        setup_cycle=GS_SETUP_CYCLE,
        eta=2,
        k_r=4,
        max_levels=4,
        max_coarse=4,
    )
    draw = seeded_draw(7)
    vectors = _random_test_vectors(matrix, setup.k_r, draw)
    coarsening = setup._coarsening_from(vectors, draw)
    levels, prolongations = setup._build_levels(matrix, coarsening, relaxation)
    assert len(levels) >= 3, "fixture must build >= 3 levels for a meaningful multi-level check"

    enriched = multigrid_eigensolver(levels, prolongations, k_e=2, relaxation=relaxation, sweeps=2)

    assert set(enriched) == {level.shape[0] for level in levels[:-1]}
    for level in levels[:-1]:
        assert enriched[level.shape[0]].shape == (level.shape[0], 2)
    metrics = composite_transfer_metrics(
        prolongations, n0=matrix.shape[0], dtype=matrix.dtype, device=matrix.device
    )
    for index, level in enumerate(levels[:-1]):
        vectors_at_level = enriched[level.shape[0]]
        T = metrics[index]
        for k in range(vectors_at_level.shape[1]):
            w = vectors_at_level[:, k]
            rayleigh = float((w @ (level @ w)) / (w @ (T @ w)))
            assert rayleigh > 0.0, "SPD generalized Rayleigh quotient must stay positive"
