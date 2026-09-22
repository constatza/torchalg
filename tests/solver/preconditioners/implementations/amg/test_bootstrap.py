"""Bootstrap AMG tests (BAMGCoarsening, BootstrapSetup, BootstrapAMGPreconditioner).

Fixtures live in this module, matching ``test_compatible_relaxation.py``'s,
``test_algebraic_distance.py``'s and ``test_least_squares.py``'s precedent:
each ``implementations/amg/`` test module keeps its fixtures local rather
than pre-emptively factoring them into a shared conftest before a second
consumer exists.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torchalg import pcg
from torchalg.preconditioners.implementations.amg.bootstrap import (
    BAMGCoarsening,
    BootstrapAMGPreconditioner,
    BootstrapAMGResult,
    BootstrapSetup,
)
from torchalg.preconditioners.implementations.amg.smoothers import GaussSeidelSmoother


@pytest.fixture
def poisson_16(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    """16x16 1D Poisson matrix, the brief's own ``BAMGCoarsening`` unit-test system."""
    return poisson_1d_factory(16)


@pytest.fixture
def seeded_test_vectors_16_3(torch_dtype: torch.dtype) -> torch.Tensor:
    """Seeded random test vectors, shape ``(16, 3)``, matching ``poisson_16``."""
    generator = torch.Generator().manual_seed(4)
    return torch.randn(16, 3, generator=generator, dtype=torch_dtype)


@pytest.fixture
def bamg_coarsening(seeded_test_vectors_16_3: torch.Tensor) -> BAMGCoarsening:
    """``BAMGCoarsening`` seeded with ``seeded_test_vectors_16_3``, caliber 3, LSR on."""
    return BAMGCoarsening(
        seeded_test_vectors_16_3,
        relaxation=GaussSeidelSmoother().smooth,
        nu=5,
        delta=0.7,
        theta_ad=0.5,
        caliber=3,
        gamma=1.5,
        use_lsr=True,
    )


@pytest.fixture
def ones_vector_factory(torch_dtype: torch.dtype) -> Callable[[int], torch.Tensor]:
    """Factory building a length-``n`` vector of ones - ``n`` is only known once a coarse level is built."""

    def _factory(n: int) -> torch.Tensor:
        return torch.ones(n, dtype=torch_dtype)

    return _factory


def test_bamg_coarsening_produces_spd_galerkin_operator(
    poisson_16: torch.Tensor,
    bamg_coarsening: BAMGCoarsening,
    ones_vector_factory: Callable[[int], torch.Tensor],
) -> None:
    """One ``build_transfer`` call must coarsen strictly and return a PSD Galerkin operator."""
    coarse_matrix, transfer = bamg_coarsening.build_transfer(poisson_16)

    eigenvalues = torch.linalg.eigvalsh(coarse_matrix)
    assert coarse_matrix.shape[0] < 16
    assert torch.all(eigenvalues > -1e-8)
    assert transfer.prolongate(ones_vector_factory(coarse_matrix.shape[0])).shape == (16,)


def test_bamg_coarsening_without_lsr_also_produces_psd_galerkin_operator(
    poisson_16: torch.Tensor, seeded_test_vectors_16_3: torch.Tensor
) -> None:
    """Plain LS (``use_lsr=False``) must also produce a valid, strictly smaller Galerkin operator."""
    coarsening = BAMGCoarsening(
        seeded_test_vectors_16_3,
        relaxation=GaussSeidelSmoother().smooth,
        nu=5,
        delta=0.7,
        theta_ad=0.5,
        caliber=3,
        gamma=1.5,
        use_lsr=False,
    )
    coarse_matrix, _ = coarsening.build_transfer(poisson_16)

    eigenvalues = torch.linalg.eigvalsh(coarse_matrix)
    assert coarse_matrix.shape[0] < 16
    assert torch.all(eigenvalues > -1e-8)


def test_bamg_coarsening_stores_restricted_test_vectors_for_next_level(
    poisson_16: torch.Tensor,
    bamg_coarsening: BAMGCoarsening,
    seeded_test_vectors_16_3: torch.Tensor,
) -> None:
    """After ``build_transfer``, the coarse-dimension test vectors are ``P^T @ V`` (dimension-keyed dict, amg-integration-architecture.md Sec. 3.2)."""
    coarse_matrix, transfer = bamg_coarsening.build_transfer(poisson_16)

    stored = bamg_coarsening.test_vectors_for(coarse_matrix)
    expected = transfer.restrict(seeded_test_vectors_16_3)
    assert torch.allclose(stored, expected)


def test_bamg_coarsening_last_prolongation_raises_before_build_transfer(
    seeded_test_vectors_16_3: torch.Tensor,
) -> None:
    """Reading ``last_prolongation`` before any ``build_transfer`` call raises, not returns stale/None data."""
    coarsening = BAMGCoarsening(seeded_test_vectors_16_3, relaxation=GaussSeidelSmoother().smooth)
    with pytest.raises(RuntimeError, match="build_transfer has not been called"):
        _ = coarsening.last_prolongation


@pytest.fixture
def bootstrap_scale_test_vectors(seeded_test_vectors_16_3: torch.Tensor) -> torch.Tensor:
    """``seeded_test_vectors_16_3`` shrunk to the magnitude bootstrap cycles actually produce.

    Relaxing/cycling on ``A x = 0`` (exact solution ``0``) drives the test
    vectors toward zero - measured around ``1e-08`` at N=31 after the
    default two bootstrap cycles. Interpolation must not degrade when that
    happens.
    """
    return seeded_test_vectors_16_3 * 1e-8


def test_bamg_coarsening_prolongation_has_no_all_zero_rows(
    poisson_16: torch.Tensor, bamg_coarsening: BAMGCoarsening
) -> None:
    """Every row of ``P`` must carry at least one nonzero: an all-zero F-row is invisible to the coarse grid."""
    bamg_coarsening.build_transfer(poisson_16)

    prolongation = bamg_coarsening.last_prolongation
    zero_rows = torch.nonzero(prolongation.abs().sum(dim=1) == 0).flatten()
    assert zero_rows.numel() == 0, f"all-zero prolongation rows at {zero_rows.tolist()}"


def test_bamg_coarsening_prolongation_has_no_all_zero_rows_for_tiny_test_vectors(
    poisson_16: torch.Tensor, bootstrap_scale_test_vectors: torch.Tensor
) -> None:
    """The same invariant must hold for bootstrap-shrunk test vectors (~1e-08), not only unit-scale ones."""
    coarsening = BAMGCoarsening(
        bootstrap_scale_test_vectors, relaxation=GaussSeidelSmoother().smooth, caliber=3
    )
    coarsening.build_transfer(poisson_16)

    prolongation = coarsening.last_prolongation
    zero_rows = torch.nonzero(prolongation.abs().sum(dim=1) == 0).flatten()
    assert zero_rows.numel() == 0, f"all-zero prolongation rows at {zero_rows.tolist()}"


@pytest.fixture
def bootstrap_setup_small() -> BootstrapSetup:
    """``BootstrapSetup`` with small counts, fast enough for a unit test."""
    return BootstrapSetup(
        nu=5,
        delta=0.7,
        theta_ad=0.5,
        caliber=3,
        gamma=1.5,
        eta=2,
        k_r=4,
        use_lsr=True,
        n_bootstrap_cycles=1,
        max_levels=4,
        max_coarse=4,
    )


@pytest.fixture
def bootstrap_draw() -> Callable[[int], torch.Tensor]:
    """Seeded uniform ``[0, 1)`` draw source for ``BootstrapSetup.run``."""
    generator = torch.Generator().manual_seed(7)

    def _draw(n: int) -> torch.Tensor:
        return torch.rand(n, generator=generator, dtype=torch.float64)

    return _draw


def test_bootstrap_setup_run_produces_a_multilevel_result(
    poisson_16: torch.Tensor,
    bootstrap_setup_small: BootstrapSetup,
    bootstrap_draw: Callable[[int], torch.Tensor],
) -> None:
    """``run`` builds at least two levels, one prolongation per coarsening step, and finest-level candidates."""
    result = bootstrap_setup_small.run(poisson_16, bootstrap_draw)

    assert isinstance(result, BootstrapAMGResult)
    assert len(result.matrices) >= 2
    assert len(result.prolongations) == len(result.matrices) - 1
    assert result.candidates.shape == (16, 4)
    for level, prolongation in zip(result.matrices[1:], result.prolongations, strict=True):
        assert level.shape[0] == prolongation.shape[1]


def test_bootstrap_setup_run_hierarchy_matches_matrices(
    poisson_16: torch.Tensor,
    bootstrap_setup_small: BootstrapSetup,
    bootstrap_draw: Callable[[int], torch.Tensor],
) -> None:
    """``result.hierarchy`` carries the same level matrices, finest first, coarsest with no transfer."""
    result = bootstrap_setup_small.run(poisson_16, bootstrap_draw)

    hierarchy = result.hierarchy
    assert len(hierarchy.levels) == len(result.matrices)
    assert hierarchy.levels[-1].transfer is None
    for level, matrix in zip(hierarchy.levels, result.matrices, strict=True):
        assert torch.equal(level.matrix, matrix)


@pytest.fixture
def poisson_31(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    """31x31 1D Poisson matrix - Task 5's own reproduction case for the degenerate-empty-coarse-level bug."""
    return poisson_1d_factory(31)


@pytest.fixture
def bootstrap_draw_seed0() -> Callable[[int], torch.Tensor]:
    """Seeded (seed 0) uniform ``[0, 1)`` draw source - deterministically reproduces a CR pass converging with ``C = empty set`` on an already-coarsened level."""
    generator = torch.Generator().manual_seed(0)

    def _draw(n: int) -> torch.Tensor:
        return torch.rand(n, generator=generator, dtype=torch.float64)

    return _draw


@pytest.fixture
def bootstrap_setup_default() -> BootstrapSetup:
    """``BootstrapSetup`` with the class's own defaults - matches the degenerate-level repro's parameters."""
    return BootstrapSetup()


def test_bootstrap_setup_run_never_appends_a_degenerate_empty_coarse_level(
    poisson_31: torch.Tensor,
    bootstrap_setup_default: BootstrapSetup,
    bootstrap_draw_seed0: Callable[[int], torch.Tensor],
) -> None:
    """CR can legitimately converge with ``C = empty set`` on an already-coarsened level; ``_build_levels``
    must stop the hierarchy there rather than appending the resulting zero-column-prolongation,
    zero-size "coarsest" level (Task 5's N=31/seed=0 repro: without the fix this produces level
    shapes ``[31, 15, 0]``, a degenerate final level contributing exactly zero coarse-grid
    correction)."""
    result = bootstrap_setup_default.run(poisson_31, bootstrap_draw_seed0)

    assert all(matrix.shape[0] > 0 for matrix in result.matrices)
    assert all(prolongation.shape[1] > 0 for prolongation in result.prolongations)


def test_bootstrap_setup_run_prolongations_have_no_all_zero_rows(
    poisson_31: torch.Tensor,
    bootstrap_setup_default: BootstrapSetup,
    bootstrap_draw_seed0: Callable[[int], torch.Tensor],
) -> None:
    """End-to-end invariant: no level's ``P`` may contain an all-zero row after the full bootstrap setup.

    The setup's own bootstrap cycles shrink the test vectors by many orders
    of magnitude (they relax on ``A x = 0``), which is exactly the regime
    where a scale-dependent interpolatory-set rule collapses to the empty
    set and leaves F-rows of ``P`` entirely zero.
    """
    result = bootstrap_setup_default.run(poisson_31, bootstrap_draw_seed0)

    for level, prolongation in enumerate(result.prolongations):
        zero_rows = torch.nonzero(prolongation.abs().sum(dim=1) == 0).flatten()
        assert zero_rows.numel() == 0, f"level {level}: all-zero rows at {zero_rows.tolist()}"


@pytest.fixture
def anisotropic_2d_matrix_64(
    anisotropic_2d_factory: Callable[[int, float], torch.Tensor],
) -> torch.Tensor:
    """64-node (8x8) strongly anisotropic 2D matrix, BAMG's payoff case (docs/bootstrap-amg.md Sec. 6)."""
    return anisotropic_2d_factory(8, 0.05)


@pytest.fixture
def pcg_rhs_64(torch_dtype: torch.dtype) -> torch.Tensor:
    """Seeded random right-hand side, length 64, matching ``anisotropic_2d_matrix_64``."""
    generator = torch.Generator().manual_seed(5)
    return torch.randn(64, dtype=torch_dtype, generator=generator)


@pytest.fixture
def bootstrap_amg_preconditioner_64(
    anisotropic_2d_matrix_64: torch.Tensor,
) -> BootstrapAMGPreconditioner:
    """``BootstrapAMGPreconditioner`` on the anisotropic 64-node system, the brief's own worked example."""
    return BootstrapAMGPreconditioner(
        anisotropic_2d_matrix_64, k_r=8, eta=4, n_bootstrap_cycles=2, seed=5
    )


def test_bootstrap_amg_preconditioner_reduces_pcg_iterations(
    anisotropic_2d_matrix_64: torch.Tensor,
    pcg_rhs_64: torch.Tensor,
    bootstrap_amg_preconditioner_64: BootstrapAMGPreconditioner,
) -> None:
    """BAMG-preconditioned PCG must need fewer iterations than unpreconditioned PCG."""
    _, result = pcg(
        anisotropic_2d_matrix_64,
        pcg_rhs_64,
        preconditioner=bootstrap_amg_preconditioner_64,
        tol=1e-8,
        maxiter=200,
    )
    _, baseline = pcg(anisotropic_2d_matrix_64, pcg_rhs_64, tol=1e-8, maxiter=200)

    assert result.iterations < baseline.iterations


def test_bootstrap_amg_preconditioner_is_linear_and_does_not_require_flexible_cg(
    bootstrap_amg_preconditioner_64: BootstrapAMGPreconditioner,
) -> None:
    """A fixed hierarchy + fixed symmetric-GS ``VCycle`` is a linear operator: plain PCG is valid."""
    assert bootstrap_amg_preconditioner_64.requires_flexible_cg is False


def test_bootstrap_amg_preconditioner_raises_when_setup_yields_a_single_level(
    poisson_1d_factory: Callable[[int], torch.Tensor],
) -> None:
    """A ``max_coarse`` at or above the system size leaves nothing to coarsen; must raise, not silently no-op."""
    matrix = poisson_1d_factory(8)
    with pytest.raises(ValueError, match="single level"):
        BootstrapAMGPreconditioner(matrix, k_r=4, eta=2, n_bootstrap_cycles=1, max_coarse=8, seed=1)
