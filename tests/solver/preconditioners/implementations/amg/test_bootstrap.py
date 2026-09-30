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
from torchalg.preconditioners.implementations.amg.protocols import MultigridSmoother
from torchalg.preconditioners.implementations.amg.smoothers import (
    GaussSeidelSmoother,
    JacobiSmoother,
)


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


def test_prolongation_batches_neighbor_lookup_not_one_nonzero_per_fine_row(
    poisson_16: torch.Tensor,
    bamg_coarsening: BAMGCoarsening,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_prolongation`` must not call `torch.nonzero` once per fine row.

    Regression guard: the previous implementation looked up each fine
    row's coarse-neighbor candidates via `torch.nonzero(neighborhood[i] &
    coarse_mask)` inside a `for i in fine_points.tolist():` loop.
    `torch.nonzero` forces a device sync to learn its data-dependent
    output size, so that was one blocking sync per fine row - the same
    O(n) regression `standard_aggregation` had. `torch.nonzero` calls
    inside `_prolongation` must stay bounded independent of how many fine
    points there are, not scale with that count.
    """
    from torchalg.preconditioners.implementations.amg._algebraic_distance import (
        algebraic_distance,
    )

    test_vectors = bamg_coarsening.test_vectors_for(poisson_16)
    distance = algebraic_distance(test_vectors, poisson_16, depth=bamg_coarsening._depth)
    coarse_mask = bamg_coarsening._coarse_mask(poisson_16, distance)
    fine_count = int((~coarse_mask).sum())
    assert fine_count >= 3, "fixture must have enough fine points to make this test meaningful"

    call_count = 0
    original_nonzero = torch.nonzero

    def counting_nonzero(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        return original_nonzero(*args, **kwargs)  # ty: ignore[no-matching-overload]

    monkeypatch.setattr(torch, "nonzero", counting_nonzero)
    bamg_coarsening._prolongation(poisson_16, test_vectors, coarse_mask, distance)

    assert call_count < fine_count, (
        f"expected torch.nonzero() calls bounded independent of fine-point count, got "
        f"{call_count} calls for {fine_count} fine points - one-per-row indicates the "
        "O(n) sync regression"
    )


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


def test_bamg_coarsening_fit_vectors_corrects_every_vector_at_its_own_top_residual_points(
    torch_dtype: torch.dtype,
) -> None:
    """[STATUS14] Sec. 3 (Table 1 caption, verified directly against arXiv:1406.1819): "the
    update ... is applied only to 20% of the entries of the TVs for which the associated
    values of the residual r_i^(kappa) are largest in absolute value" - every test vector
    is corrected (no restriction to a single vector), each at *its own* highest-residual
    points, not a residual combined/shared across vectors.

    Construction: two locally-coupled zones (rows 1-3 and rows 6-8) on an otherwise-identity
    matrix. Vector 0 has a huge spike in zone 1 only; vector 1 has a spike in zone 2 only,
    plus a small secondary presence in zone 1 (small enough that a wrong, cross-vector
    selection there would still be observable rather than a no-op). If row selection used a
    residual combined across vectors, vector 0's huge zone-1 spike would dominate and could
    pull vector 1's own correction into zone 1 too; using each vector's own residual keeps
    vector 1's correction confined to its own zone 2.
    """
    n = 10
    matrix = torch.eye(n, dtype=torch_dtype)
    for i, j in ((1, 2), (2, 3), (6, 7), (7, 8)):
        matrix[i, j] = matrix[j, i] = 0.3
    vector_a = torch.zeros(n, dtype=torch_dtype)
    vector_a[2] = 100.0  # huge spike in zone 1 (rows 1-3), vector_a's own peak
    vector_b = torch.zeros(n, dtype=torch_dtype)
    vector_b[7] = 5.0  # its own peak, zone 2 (rows 6-8)
    vector_b[2] = 0.5  # small secondary presence in zone 1
    test_vectors = torch.stack([vector_a, vector_b], dim=1)
    fine_points = torch.arange(n, dtype=torch.long)

    coarsening = BAMGCoarsening(test_vectors, relaxation=GaussSeidelSmoother().smooth, use_lsr=True)
    corrected = coarsening._fit_vectors(matrix, test_vectors, fine_points)

    assert not torch.equal(corrected[:, 0], test_vectors[:, 0]), (
        "vector_a is corrected at its own zone 1 peak"
    )
    assert torch.equal(corrected[2, 1], test_vectors[2, 1]), (
        "vector_b's own residual in zone 1 is small - it must not be selected just because "
        "vector_a (a DIFFERENT vector) has a huge residual there"
    )
    assert not torch.equal(corrected[7, 1], test_vectors[7, 1]), (
        "zone 2 is vector_b's own dominant residual - it must be selected"
    )


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


def test_bootstrap_setup_improve_levels_improves_every_level_except_coarsest(
    poisson_16: torch.Tensor,
    bootstrap_setup_small: BootstrapSetup,
    bootstrap_draw: Callable[[int], torch.Tensor],
) -> None:
    """[BAMG11] Sec. 5 / [STATUS14]: every level's test vectors are improved via its own
    sub-hierarchy cycle each bootstrap cycle - not just the finest level's, which was a
    real simplification of Sec. 5 this module used to make (see ``run``'s previous
    ``TODO(bamg-fidelity, mandatory)`` marker).
    """
    from torchalg.preconditioners.implementations.amg.bootstrap import (
        _hierarchy_of,
        _improve_test_vectors,
        _random_test_vectors,
    )

    relaxation = GaussSeidelSmoother().smooth
    vectors = _random_test_vectors(poisson_16, bootstrap_setup_small.k_r, bootstrap_draw)
    coarsening = bootstrap_setup_small._coarsening_from(vectors, relaxation, bootstrap_draw)
    levels, prolongations = bootstrap_setup_small._build_levels(poisson_16, coarsening, relaxation)
    assert len(levels) >= 3, "fixture must build >= 3 levels to distinguish per-level improvement"
    before = {level.shape[0]: coarsening.test_vectors_for(level) for level in levels[:-1]}

    improved = bootstrap_setup_small._improve_levels(levels, prolongations, before)

    assert set(improved) == {level.shape[0] for level in levels[:-1]}
    for level in levels[:-1]:
        assert improved[level.shape[0]].shape == before[level.shape[0]].shape

    coarse_level = levels[1]
    assert not torch.equal(improved[coarse_level.shape[0]], before[coarse_level.shape[0]]), (
        "a coarser level's vectors must actually change - the previous finest-only "
        "implementation left every coarser level's improvement step a no-op"
    )

    full_hierarchy = _hierarchy_of(levels, prolongations)
    expected_finest = _improve_test_vectors(
        poisson_16, before[poisson_16.shape[0]], full_hierarchy, bootstrap_setup_small.eta
    )
    assert torch.equal(improved[poisson_16.shape[0]], expected_finest)


def test_seeded_test_vectors_concatenates_seed_columns_after_random_draws(
    poisson_16: torch.Tensor,
) -> None:
    """[STATUS14] Table 3 seeds a known near-null vector alongside ``k_r`` random draws, not instead of them."""
    from torchalg.preconditioners.implementations.amg._presets import seeded_draw
    from torchalg.preconditioners.implementations.amg.bootstrap import (
        _random_test_vectors,
        _seeded_test_vectors,
    )

    seed_vectors = torch.ones(poisson_16.shape[0], 1, dtype=poisson_16.dtype)

    seeded = _seeded_test_vectors(poisson_16, k_r=3, draw=seeded_draw(7), seed_vectors=seed_vectors)
    random_only = _seeded_test_vectors(poisson_16, k_r=3, draw=seeded_draw(7), seed_vectors=None)

    assert seeded.shape == (16, 4)
    assert torch.equal(seeded[:, 3:4], seed_vectors)
    assert torch.equal(random_only, _random_test_vectors(poisson_16, 3, seeded_draw(7)))


def test_bootstrap_setup_run_includes_seed_vector_columns(
    poisson_16: torch.Tensor,
    bootstrap_setup_small: BootstrapSetup,
    bootstrap_draw: Callable[[int], torch.Tensor],
) -> None:
    """``run(seed_vectors=...)`` grows the finest-level test-vector count by the seed columns."""
    seed_vectors = torch.ones(poisson_16.shape[0], 1, dtype=poisson_16.dtype)

    result = bootstrap_setup_small.run(poisson_16, bootstrap_draw, seed_vectors=seed_vectors)

    assert result.candidates.shape[1] == 4 + 1


def test_bootstrap_setup_run_test_vector_draw_is_independent_of_draw(
    poisson_16: torch.Tensor,
    bootstrap_setup_small: BootstrapSetup,
) -> None:
    """``draw`` is used for two unrelated things: generating the initial ``k_r``
    test vectors, and seeding compatible relaxation's own internal
    convergence-rate probe (``cr_rate``, documented to expect a uniform
    ``[0, 1)`` start for its short, ``nu``-sweep power-iteration-style
    estimate). Reusing one ``draw`` for both means changing the test-vector
    distribution silently changes CR's behavior too. ``test_vector_draw``
    (optional, defaults to ``draw`` - unchanged behavior) decouples them:
    the finest-level test vectors before any bootstrap-cycle mixing depend
    only on ``test_vector_draw`` and the (deterministic) relaxation, not on
    ``draw`` - confirmed here by holding ``test_vector_draw`` fixed and
    varying ``draw``, which must not change the result.
    """
    from torchalg.preconditioners.implementations.amg._presets import seeded_draw

    fixed_test_vector_draw = seeded_draw(3)
    result_a = bootstrap_setup_small.run(
        poisson_16, seeded_draw(1), test_vector_draw=seeded_draw(3)
    )
    result_b = bootstrap_setup_small.run(
        poisson_16, seeded_draw(2), test_vector_draw=fixed_test_vector_draw
    )

    assert torch.equal(result_a.candidates, result_b.candidates)


def test_bootstrap_setup_run_normal_test_vectors_do_not_break_cr_coarsening(
    poisson_16: torch.Tensor,
    bootstrap_setup_small: BootstrapSetup,
) -> None:
    """Regression: before ``test_vector_draw`` existed, generating test vectors
    from a signed distribution (e.g. N(0,1), matching [STATUS14] Sec. 3's
    "generated randomly with a normal distribution ... N(0,1)") reused the
    same ``draw`` for CR's internal uniform-``[0,1)``-expecting probe too,
    which collapsed compatible-relaxation coarsening to a single level -
    reproduced directly (not merely inferred) on a 256-node 2D fixture during
    investigation. With the distributions separated, CR must coarsen normally
    regardless of what distribution generates the test vectors.
    """
    from torchalg.preconditioners.implementations.amg._presets import seeded_draw

    def normal_draw(seed: int) -> Callable[[int], torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        return lambda n: torch.randn(n, generator=generator, dtype=torch.float64)

    result = bootstrap_setup_small.run(poisson_16, seeded_draw(7), test_vector_draw=normal_draw(0))

    assert len(result.matrices) >= 2, "CR coarsening must not collapse to a single level"


def test_bootstrap_setup_run_includes_mge_eigenvector_columns(
    poisson_16: torch.Tensor,
    bootstrap_setup_small: BootstrapSetup,
    bootstrap_draw: Callable[[int], torch.Tensor],
) -> None:
    """[BAMG11] Algorithm 1: MGE eigenvectors are appended to the relaxation-derived
    TVs (``k = k_r + k_e``), not built in v1 but opt-in via ``k_e``."""
    from dataclasses import replace

    setup = replace(bootstrap_setup_small, k_e=2)

    result = setup.run(poisson_16, bootstrap_draw)

    assert result.candidates.shape[1] == 4 + 2


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
    """A fixed hierarchy + fixed symmetric ``VCycle`` is a linear operator: plain PCG is valid."""
    assert bootstrap_amg_preconditioner_64.requires_flexible_cg is False


def test_bootstrap_amg_preconditioner_defaults_to_jacobi_at_solve_time(
    bootstrap_amg_preconditioner_64: BootstrapAMGPreconditioner,
) -> None:
    """The quick solve path defaults to Jacobi without changing GS-based CR/setup."""
    assert isinstance(
        bootstrap_amg_preconditioner_64._cycle._smoother,  # ty: ignore[unresolved-attribute]
        JacobiSmoother,
    )


def test_bootstrap_amg_preconditioner_defaults_to_v11_solve_cycle(
    bootstrap_amg_preconditioner_64: BootstrapAMGPreconditioner,
) -> None:
    """Solve-time sweep counts default to V(1,1), matching the historical hardcoded cycle."""
    assert bootstrap_amg_preconditioner_64._cycle._n_pre == 1  # ty: ignore[unresolved-attribute]
    assert bootstrap_amg_preconditioner_64._cycle._n_post == 1  # ty: ignore[unresolved-attribute]


def test_bootstrap_amg_preconditioner_uses_configured_cycle_sweep_counts(
    anisotropic_2d_matrix_64: torch.Tensor,
) -> None:
    """[STATUS14] treats sweep counts as experiment-specific (eta = 2, 4, 6, 8); expose them."""
    preconditioner = BootstrapAMGPreconditioner(
        anisotropic_2d_matrix_64, k_r=8, eta=4, n_bootstrap_cycles=2, seed=5, n_pre=3, n_post=2
    )
    assert preconditioner._cycle._n_pre == 3  # ty: ignore[unresolved-attribute]
    assert preconditioner._cycle._n_post == 2  # ty: ignore[unresolved-attribute]


def test_bootstrap_amg_preconditioner_accepts_seed_vectors(
    anisotropic_2d_matrix_64: torch.Tensor,
) -> None:
    """[STATUS14] Table 3's near-null seeding is reachable through the public preconditioner too."""
    seed_vectors = torch.ones(
        anisotropic_2d_matrix_64.shape[0], 1, dtype=anisotropic_2d_matrix_64.dtype
    )

    preconditioner = BootstrapAMGPreconditioner(
        anisotropic_2d_matrix_64,
        k_r=8,
        eta=4,
        n_bootstrap_cycles=2,
        seed=5,
        seed_vectors=seed_vectors,
    )

    assert preconditioner.result.candidates.shape[1] == 8 + 1


def test_bootstrap_amg_preconditioner_accepts_k_e(
    anisotropic_2d_matrix_64: torch.Tensor,
) -> None:
    """[BAMG11] Algorithm 1's MGE enrichment is reachable through the public preconditioner too."""
    preconditioner = BootstrapAMGPreconditioner(
        anisotropic_2d_matrix_64,
        k_r=8,
        eta=4,
        n_bootstrap_cycles=2,
        seed=5,
        k_e=2,
    )

    assert preconditioner.result.candidates.shape[1] == 8 + 2


def test_bootstrap_amg_preconditioner_accepts_test_vector_draw(
    anisotropic_2d_matrix_64: torch.Tensor,
) -> None:
    """``test_vector_draw`` is reachable through the public preconditioner too,
    letting a caller match [STATUS14] Sec. 3's N(0,1) test vectors without
    also feeding a signed distribution into CR's own uniform-expecting probe."""

    def normal_draw(n: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(0)
        return torch.randn(n, generator=generator, dtype=anisotropic_2d_matrix_64.dtype)

    preconditioner = BootstrapAMGPreconditioner(
        anisotropic_2d_matrix_64,
        k_r=8,
        eta=4,
        n_bootstrap_cycles=2,
        seed=5,
        test_vector_draw=normal_draw,
    )

    assert len(preconditioner.result.matrices) >= 2


def test_bootstrap_amg_preconditioner_uses_configured_solve_smoother(
    anisotropic_2d_matrix_64: torch.Tensor,
    pcg_rhs_64: torch.Tensor,
    raising_smoother: MultigridSmoother,
) -> None:
    """BAMG setup stays internal, while its solve-time cycle uses the injected smoother."""
    preconditioner = BootstrapAMGPreconditioner(
        anisotropic_2d_matrix_64,
        k_r=4,
        eta=2,
        n_bootstrap_cycles=1,
        seed=5,
        smoother=raising_smoother,
    )
    with pytest.raises(RuntimeError, match="configured smoother used"):
        preconditioner.apply(pcg_rhs_64)


def test_bootstrap_amg_preconditioner_raises_when_setup_yields_a_single_level(
    poisson_1d_factory: Callable[[int], torch.Tensor],
) -> None:
    """A ``max_coarse`` at or above the system size leaves nothing to coarsen; must raise, not silently no-op."""
    matrix = poisson_1d_factory(8)
    with pytest.raises(ValueError, match="single level"):
        BootstrapAMGPreconditioner(matrix, k_r=4, eta=2, n_bootstrap_cycles=1, max_coarse=8, seed=1)


class TestBootstrapAMGResultAndStr:
    def test_result_property_matches_private_attribute(
        self, bootstrap_amg_preconditioner_64: BootstrapAMGPreconditioner
    ) -> None:
        """`result` property exposes exactly the setup result stored at construction."""
        assert bootstrap_amg_preconditioner_64.result is bootstrap_amg_preconditioner_64._result

    def test_str_reports_levels_k_r_and_coarse_dim(
        self, bootstrap_amg_preconditioner_64: BootstrapAMGPreconditioner
    ) -> None:
        """`str()` reports the realized hierarchy shape, not a generic repr."""
        text = str(bootstrap_amg_preconditioner_64)
        assert "BAMG" in text
        assert f"n_levels={len(bootstrap_amg_preconditioner_64.result.matrices)}" in text
        assert f"k_r={bootstrap_amg_preconditioner_64.result.candidates.shape[1]}" in text
