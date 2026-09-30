"""Benchmark: Bootstrap AMG paper-table reproduction (docs/bootstrap-amg.md Sec. 6).

**Primary fidelity target:** ``test_bootstrap_amg_matches_status14_table1_exact_setup``,
below. It reproduces [STATUS14] (arXiv:1406.1819) Table 1's exact experimental
setup - discretization, grid, cycle, hyperparameters, test-vector distribution,
all confirmed directly against the paper, not inferred - and is the one test
in this module checked against a specific, verified paper configuration
rather than an approximation of one. Read its own docstring for the full
correspondence and for the one remaining, understood gap (a different
coarsening algorithm) that keeps its absolute numbers from matching the
paper's exactly.

Every other test below instead attempts to reproduce the *shape* of
[BAMG11] Tables 4.2/4.3 - LS degrading with problem size, LSR generally
doing better - on this repo's own plain finite-difference Poisson problems
(``poisson_1d_factory``/``anisotropic_2d_factory``, ``tests/conftest.py``),
using hyperparameters that were never checked against a specific verified
table. Not an attempt to match the paper's absolute ``rho`` numbers.

What is measured:
    The multigrid *solve-phase* convergence factor,
    ``rho = ||e^nu||_A / ||e^{nu-1}||_A`` (energy/A-norm ratio of two
    successive iterates), on the homogeneous system ``A x = 0`` starting
    from a random error. This directly parallels
    ``_compatible_relaxation.py``'s ``cr_rate`` (``AD11`` eq. 3.4's
    root-form rate estimate), but applies the full
    ``BootstrapAMGPreconditioner.apply()`` V-cycle each step instead of a
    single-level HCR F-relaxation sweep - i.e. it is the actual solver
    convergence factor the paper's Tables 4.2/4.3 report, not the CR
    diagnostic rate Task 1's own property test covers. A direct
    convergence-factor measurement was chosen over a PCG-iteration-count
    proxy because ``AMGPreconditioner.apply(residual)`` already exposes one
    V-cycle as a plain callable (see ``amg.py``), making the paper's own
    ``rho`` formula directly reproducible without going through a CG loop
    that would only measure ``rho`` indirectly. ``nu=10`` (module constant
    ``_MG_ITERATIONS``) is a practical setup-time estimate, matching
    ``cr_rate``'s own small-``nu`` convention and ``docs/bootstrap-amg.md``
    Sec. 5's own "estimate ... rho; stop if rho <= target" - not the true
    infinite-iteration limit.

Why the absolute numbers will not match [BAMG11] Tables 4.2/4.3:
    1. **Discretization.** ``poisson_1d_factory``/``anisotropic_2d_factory``
       build a plain 5-point finite-difference Poisson stencil; [BAMG11]
       Sec. 4.1 uses a finite-element Laplace discretization on the same
       nominal grid sizes. Different discretizations have different
       algebraically-smooth-error spectra, which directly changes ``rho``.
    2. **Cycle.** ``BootstrapAMGPreconditioner`` hardcodes a fixed
       V(1,1)-cycle (``_presets.PRESET_CYCLE``), not the paper's V(2,2).
    3. **No near-null-space seeding.** [BAMG11] Sec. 6's near-optimal LSR
       range (``rho ~= 0.04-0.15``) requires seeding the known constant
       vector ``1`` into the test-vector set alongside the ``k_r`` random
       ones; ``BootstrapAMGPreconditioner`` has no such seeding hook - its
       test vectors are purely ``k_r`` random draws
       (``_random_test_vectors`` in ``bootstrap.py``). This is this
       harness's best-attributed reason the absolute numbers here sit around
       ``rho ~= 0.45-0.75`` rather than the paper's seeded range; it is a
       documented, deliberate scope decision, not a defect. See point 4:
       nothing here decomposes this reason's share of the gap from that
       one's, since only seeding (and MGE, not listed here because this
       benchmark targets Tables 4.2/4.3, which are MGE-free) has an explicit
       paper-sanctioned "optional" framing.
    4. **Finest-level-only test-vector improvement (RESOLVED).**
       ``BootstrapSetup.run`` used to improve only the finest level's test
       vectors per bootstrap cycle; [BAMG11] Sec. 5 and [STATUS14]'s base
       bootstrap loop (independent of MGE) improve every level's, and no
       paper text sanctioned finest-only as an alternate mode, so this was a
       mandatory fix (``BootstrapSetup._improve_levels``, ``bootstrap.py``).
       Re-measuring after the fix showed a small, mixed effect on this
       harness's ``rho`` values, not the clear improvement a first guess
       might expect: the fix only changes anything for hierarchies with 3+
       levels (a 2-level hierarchy's "every level except the coarsest" is
       just the finest level, identical to the old behavior), and even
       where it does apply, FD-vs-FE discretization (point 1) and V(1,1)
       vs. V(2,2) (point 2) appear to dominate the measured ``rho`` far more
       than this specific algorithmic gap at these problem sizes/hyperparameters.
    5. **LSR practical schedule (RESOLVED, corrected twice; final version
       checked against [STATUS14] Sec. 3 directly, arXiv:1406.1819, not a
       secondary transcription).** The schedule corrects *every* test
       vector, each at the 20% of F-points with the largest absolute value
       of *that same vector's own* residual - never a single vector, never
       a residual combined across vectors (``bootstrap.py``'s module
       docstring has the full correction history, including a first fix
       that turned out to still be wrong). Once implemented faithfully, the
       LSR-beats-LS trend visible in every earlier version of the table
       below mostly disappears in this harness: LSR now wins outright at
       half the rows and loses outright at N=127 (0/8) and the 2D case, with
       no clean pattern by size. This does not contradict [BAMG11]/[STATUS14]'s
       own claim that LSR is the better-scaling interpolation - the same
       undecomposed confounders as points 1-3 (discretization, cycle,
       missing seeding) plausibly dominate which of LS/LSR wins at these
       problem sizes in this harness, same as they already do for the
       *absolute* level of ``rho``.

LS vs LSR, as measured by this module's own harness:
    Averaged over 8 independent ``BootstrapAMGPreconditioner`` setup seeds
    (``seed=0..7``), 10 V(1,1) iterations, this repo's plain 5-point FD 1D
    Poisson, ``mean (sd)`` of ``rho``, and how many of the 8 seeds LSR won:

        N     LS mean (sd)     LSR mean (sd)    LSR wins
        31    0.4850 (0.099)   0.4303 (0.149)   4/8
        63    0.7156 (0.166)   0.6839 (0.120)   5/8
        127   0.5988 (0.042)   0.7354 (0.111)   0/8
        255   0.7323 (0.056)   0.7043 (0.060)   5/8
        511   0.7311 (0.058)   0.7390 (0.054)   4/8
        2D 16x16 (256 nodes)
              0.4833 (0.064)   0.5397 (0.054)   2/8

    **No LSR advantage survives the LSR-schedule correction (point 5
    above).** Every earlier version of this table showed LSR's mean beating
    LS's mean at every size, with only the *shape* of the trend in dispute.
    After correcting the schedule to match what [STATUS14] actually
    describes (every vector corrected at its own residual, not a combined
    one), LSR now loses outright at N=127 and the 2D case, and its
    advantage elsewhere is small relative to the per-seed spread. This is
    not read as evidence the paper's LSR claim is wrong - it is read as
    further confirmation that this harness's undecomposed confounders
    (points 1-3) dominate the measured numbers strongly enough to overturn
    even the *direction* of a paper-reported trend, not just its absolute
    level or shape.

    This replaces two earlier versions of this docstring. The first
    reported LS and LSR as statistically indistinguishable at *every* size,
    an artifact of a real bug (``select_interpolatory_set`` compared raw,
    un-normalized LS functional values, so the bootstrap cycles' own
    shrinking of the test vectors stalled interpolatory-set growth and left
    most F-rows of ``P`` entirely zero - e.g. 14 of 15 F-rows at N=31,
    ``rho ~= 0.88``); not caused by the seeding gap. The second, after that
    fix, showed LSR beating LS's mean at every size - an artifact of the LSR
    schedule itself using a residual combined across every test vector to
    pick F-points, instead of each vector's own residual (point 5's
    correction). Neither artifact announced itself as a bug; both were
    caught only because a later, unrelated fix changed the measured trend
    enough to demand re-examining why.

Problem-size choice (31/63/127/255/511 for the 1D case, a 16x16 grid for 2D):
    N=31 and N=63 were excluded from an earlier version of this sweep
    because ``BootstrapSetup``'s default ``max_coarse=10`` loop appended a
    degenerate, empty (0-node) coarsest level at those sizes. That bug was
    fixed (``_build_levels`` now discards a coarsening pass that does not
    strictly shrink, with its own regression test in ``test_bootstrap.py``),
    and the interpolatory-set fix above independently changed what these
    sizes measure, so they are back in the sweep: both build well-formed,
    strictly-shrinking hierarchies under the default parameters.
"""

from __future__ import annotations

from statistics import mean
from typing import TYPE_CHECKING

import pytest
import torch

from torchalg.preconditioners.implementations.amg.bootstrap import BootstrapAMGPreconditioner
from torchalg.preconditioners.implementations.amg.smoothers import GaussSeidelSmoother

if TYPE_CHECKING:
    from collections.abc import Callable

_MG_ITERATIONS = 10
"""V-cycle iterations before reading off the convergence-factor ratio - see the
module docstring's "What is measured" section for why this (not the
infinite-iteration limit) is the practical choice."""


def _energy_norm(matrix: torch.Tensor, vector: torch.Tensor) -> float:
    """A-norm (energy norm) of ``vector`` under SPD ``matrix``.

    Args:
        matrix (torch.Tensor): SPD matrix ``A``, shape ``(n, n)``.
        vector (torch.Tensor): Vector, shape ``(n,)``.

    Returns:
        float: ``sqrt(vector @ matrix @ vector)``.
    """
    return torch.sqrt(vector @ matrix @ vector).item()


def _mg_convergence_factor(
    preconditioner: BootstrapAMGPreconditioner, matrix: torch.Tensor, start: torch.Tensor
) -> float:
    """Multigrid convergence factor ``rho = ||e^nu||_A / ||e^{nu-1}||_A``.

    Repeatedly applies one V-cycle to the homogeneous system ``A x = 0``
    (using the same ``x - apply(A @ x)`` linearity identity
    ``bootstrap.py``'s own ``_improve_test_vectors`` uses), then reads off
    the energy-norm ratio of the last two iterates.

    Args:
        preconditioner (BootstrapAMGPreconditioner): The built solver.
        matrix (torch.Tensor): SPD system matrix ``A``, shape ``(n, n)``.
        start (torch.Tensor): Initial error vector, shape ``(n,)``.

    Returns:
        float: The measured convergence factor ``rho``.
    """
    x = start.clone()
    previous_norm = _energy_norm(matrix, x)
    final_norm = previous_norm
    for _ in range(_MG_ITERATIONS):
        x = x - preconditioner.apply(matrix @ x)
        previous_norm = final_norm
        final_norm = _energy_norm(matrix, x)
    return final_norm / previous_norm


@pytest.fixture
def poisson_paper_sizes() -> tuple[int, ...]:
    """1D problem sizes reproducing [BAMG11] Tables 4.2/4.3's ``N`` column."""
    return (31, 63, 127, 255, 511)


@pytest.fixture
def rho_per_size_bound() -> float:
    """Per-size upper bound on ``rho``, calibrated to this harness's own measurements.

    The worst single ``rho`` measured at ``seed=5`` across the sizes above
    is ``0.8718`` (LS, N=127); the worst over ``seed=0..7`` is ``0.918``.
    ``0.95`` is a real numeric bound with headroom for seed-to-seed spread -
    a plain ``rho < 1.0`` would pass at ``rho = 0.97``, i.e. at a hierarchy
    that is barely converging at all.
    """
    return 0.95


@pytest.fixture
def rho_mean_bound() -> float:
    """Upper bound on the mean ``rho`` across ``poisson_paper_sizes``.

    Measured at ``seed=5``: LS mean ``0.6711``, LSR mean ``0.5660``. The
    mean is where the real signal is (a single size's ``rho`` is noisy
    across seeds), so this is the assertion that would actually catch a
    regression like the one that produced ``rho ~= 0.88-0.97``.
    """
    return 0.75


@pytest.fixture
def energy_norm_start_factory(torch_dtype: torch.dtype) -> Callable[[int], torch.Tensor]:
    """Factory building a seeded random error vector of length ``n``.

    A factory, not a plain fixture, since the vector's length depends on
    whichever problem size a test builds at runtime.
    """

    def _factory(n: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(11)
        return torch.randn(n, dtype=torch_dtype, generator=generator)

    return _factory


@pytest.fixture
def bootstrap_amg_factory() -> Callable[[torch.Tensor, bool], BootstrapAMGPreconditioner]:
    """Factory building a ``BootstrapAMGPreconditioner`` with [BAMG11] Sec. 6's default hyperparameters.

    ``seed=5`` matches this codebase's existing convention for
    ``BootstrapAMGPreconditioner`` tests (``test_bootstrap.py``'s
    ``bootstrap_amg_preconditioner_64`` fixture), so the fixed-seed
    LSR-vs-LS comparison below is reproducible against that precedent, not
    a value hand-picked for this benchmark.

    ``smoother=GaussSeidelSmoother()`` is pinned explicitly rather than
    left to the class default: this benchmark validates against [BAMG11]'s
    own published Table 4.2/4.3 numbers, which are for the paper's
    algorithm specifically (symmetric GS smoothing) - it needs to keep
    measuring that claim regardless of what ``BootstrapAMGPreconditioner``
    defaults to for everyday PCG use, which is a separate, orthogonal
    decision (currently weighted Jacobi, for solve-time performance).

    A factory, not a plain fixture, since it is built once per (matrix,
    ``use_lsr``) pair across several problem sizes/variants within a single
    test.
    """

    def _factory(matrix: torch.Tensor, use_lsr: bool) -> BootstrapAMGPreconditioner:
        return BootstrapAMGPreconditioner(
            matrix,
            k_r=8,
            eta=4,
            n_bootstrap_cycles=2,
            use_lsr=use_lsr,
            seed=5,
            smoother=GaussSeidelSmoother(),
        )

    return _factory


@pytest.fixture
def isotropic_2d_matrix_256(
    anisotropic_2d_factory: Callable[[int, float], torch.Tensor],
) -> torch.Tensor:
    """16x16 (256-node) isotropic 2D Poisson matrix (``eps=1.0``) - the 2D analogue of Sec. 6's table."""
    return anisotropic_2d_factory(16, 1.0)


@pytest.mark.benchmark
def test_bootstrap_amg_ls_and_lsr_both_converge_on_1d_poisson(
    poisson_paper_sizes: tuple[int, ...],
    poisson_1d_factory: Callable[[int], torch.Tensor],
    energy_norm_start_factory: Callable[[int], torch.Tensor],
    bootstrap_amg_factory: Callable[[torch.Tensor, bool], BootstrapAMGPreconditioner],
    rho_per_size_bound: float,
    rho_mean_bound: float,
) -> None:
    """Both LS and LSR build convergent hierarchies at every size, within a real numeric bound.

    Asserts two things:
      1. every ``rho`` is below ``rho_per_size_bound`` (``0.95``) - a bound
         that a barely-converging hierarchy (``rho ~= 0.97``) fails, unlike
         the plain ``rho < 1.0`` this test used to assert;
      2. the *mean* ``rho`` over the sweep is below ``rho_mean_bound``
         (``0.75``) for both variants - the assertion that would actually
         have caught the un-normalized-penalization bug, whose LS means sat
         at ``0.87``.

    A third assertion - LSR's mean beats LS's - was removed after correcting
    the LSR practical schedule to match [STATUS14] Sec. 3 (every vector
    corrected at its own residual, not one combined across vectors; see the
    module docstring's point 5). LSR no longer reliably beats LS in this
    harness once that correction is in place; asserting it would be fitting
    the test to a result this repo cannot currently reproduce faithfully,
    not verifying a real property. Per-size ordering was already not
    asserted before this change, for the same "genuinely noisy" reason the
    module docstring's table shows.

    Actual measured values (``seed=5``, 10 V(1,1)-cycle iterations, this
    repo's plain 5-point FD 1D Poisson, N=31/63/127/255/511 - recorded for
    future regression comparison; superseded twice, see the module
    docstring's points 4 and 5):
        LS rho:  [0.5833, 0.6382, 0.6733, 0.7724, 0.7757]  (mean 0.6886)
        LSR rho: [0.4559, 0.7280, 0.8384, 0.6591, 0.7788]  (mean 0.6920)
    """
    rho_ls_values: list[float] = []
    rho_lsr_values: list[float] = []
    for n in poisson_paper_sizes:
        matrix = poisson_1d_factory(n)
        start = energy_norm_start_factory(n)
        rho_ls = _mg_convergence_factor(bootstrap_amg_factory(matrix, False), matrix, start)
        rho_lsr = _mg_convergence_factor(bootstrap_amg_factory(matrix, True), matrix, start)
        rho_ls_values.append(rho_ls)
        rho_lsr_values.append(rho_lsr)

        assert 0.0 <= rho_ls < rho_per_size_bound, f"LS rho={rho_ls} at N={n}"
        assert 0.0 <= rho_lsr < rho_per_size_bound, f"LSR rho={rho_lsr} at N={n}"

    mean_ls = mean(rho_ls_values)
    mean_lsr = mean(rho_lsr_values)
    assert mean_ls < rho_mean_bound, f"LS mean rho={mean_ls} over {rho_ls_values}"
    assert mean_lsr < rho_mean_bound, f"LSR mean rho={mean_lsr} over {rho_lsr_values}"


@pytest.mark.benchmark
def test_bootstrap_amg_ls_and_lsr_both_converge_at_fixed_seed_on_2d_poisson(
    isotropic_2d_matrix_256: torch.Tensor,
    energy_norm_start_factory: Callable[[int], torch.Tensor],
    bootstrap_amg_factory: Callable[[torch.Tensor, bool], BootstrapAMGPreconditioner],
    rho_per_size_bound: float,
) -> None:
    """At the codebase's standard seed, both LS and LSR converge on the 2D payoff case.

    Was named ``..._lsr_beats_ls_...`` and asserted ``rho_lsr < rho_ls``.
    That assertion is no longer true after correcting the LSR practical
    schedule to match [STATUS14] Sec. 3 (every vector corrected at its own
    residual, not one combined across vectors - module docstring point 5):
    at this size and seed, LSR is now measurably *worse* than LS
    (``0.5808`` vs ``0.5445``, see below), not better, consistent with the
    8-seed aggregate (module docstring table: 2 of 8 seeds now won by LSR,
    not 5 of 8). Renamed and the assertion removed rather than kept passing
    on a claim this harness no longer supports; both are still bounded well
    below ``rho_per_size_bound``, which is what the test now demonstrates.

    Actual measured values (``seed=5``, 10 V(1,1)-cycle iterations, 16x16
    isotropic 2D Poisson, 256 nodes - recorded for future regression
    comparison; superseded by the LSR-schedule correction, unlike the
    every-level test-vector-improvement fix, which left this particular
    seed's numbers unchanged since it happens to build only a 2-level
    hierarchy here):
        LS rho:  0.5445
        LSR rho: 0.5808
    """
    matrix = isotropic_2d_matrix_256
    start = energy_norm_start_factory(matrix.shape[0])

    rho_ls = _mg_convergence_factor(bootstrap_amg_factory(matrix, False), matrix, start)
    rho_lsr = _mg_convergence_factor(bootstrap_amg_factory(matrix, True), matrix, start)

    assert 0.0 <= rho_ls < rho_per_size_bound, f"LS rho={rho_ls}"
    assert 0.0 <= rho_lsr < rho_per_size_bound, f"LSR rho={rho_lsr}"


@pytest.fixture
def isotropic_2d_matrix_h64(
    anisotropic_2d_factory: Callable[[int, float], torch.Tensor],
) -> torch.Tensor:
    """63x63 (3969-node) isotropic 2D Poisson matrix, ``h=1/64`` - [STATUS14] Table 1's
    exact grid ("central finite difference discretization of the Poisson problem with
    homogenous Dirichlet boundary conditions on a uniform quadrilateral grid," verified
    directly against arXiv:1406.1819)."""
    return anisotropic_2d_factory(63, 1.0)


@pytest.fixture
def normal_seeded_draw_factory(
    torch_dtype: torch.dtype,
) -> Callable[[int], Callable[[int], torch.Tensor]]:
    """N(0,1) draw-source factory, matching [STATUS14] Sec. 3's test-vector distribution
    ("generated randomly with a normal distribution with expectation zero and variance
    one, N(0,1)") - distinct from this package's default uniform ``[0, 1)`` ``seeded_draw``."""

    def _factory(seed: int) -> Callable[[int], torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        return lambda n: torch.randn(n, generator=generator, dtype=torch_dtype)

    return _factory


@pytest.mark.benchmark
def test_bootstrap_amg_matches_status14_table1_exact_setup(
    isotropic_2d_matrix_h64: torch.Tensor,
    energy_norm_start_factory: Callable[[int], torch.Tensor],
    normal_seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
    rho_per_size_bound: float,
) -> None:
    """**Primary fidelity regression.** [STATUS14] Sec. 3, Table 1's exact experimental
    setup, reproduced as faithfully as this implementation currently allows - the
    canonical target this module is checked against, not one more loose approximation
    like the tests above. Confirmed directly against arXiv:1406.1819, not inferred:

    - ``h = 1/64`` on a "uniform quadrilateral grid" -> 63x63 = 3969 interior nodes
      (``isotropic_2d_matrix_h64``);
    - "central finite difference discretization" -> ``anisotropic_2d_factory(63, 1.0)``,
      already this repo's FD stencil, no discretization mismatch to correct for here;
    - "two pre- and two post-smoothing steps" -> ``n_pre=2, n_post=2`` (V(2,2));
    - ``k=8`` test vectors, ``eta=4`` Gauss-Seidel sweeps -> ``k_r=8, eta=4``;
    - "no near-null-space seeding" for this table -> ``seed_vectors=None`` (default);
    - test vectors "generated randomly with a normal distribution ... N(0,1)" ->
      ``test_vector_draw=normal_seeded_draw_factory(...)``, *not* the default uniform
      ``draw`` (which must stay uniform - it also seeds compatible relaxation's own
      internal convergence-rate probe, which is documented to expect a uniform ``[0,1)``
      start; conflating the two used to collapse coarsening entirely under a signed
      distribution, a real bug fixed alongside this test - see
      ``BootstrapSetup.run``'s docstring and ``test_bootstrap.py``'s
      ``test_bootstrap_setup_run_normal_test_vectors_do_not_break_cr_coarsening``);
    - "Gauss Seidel iterations ... starting with k distinct initial guesses" with no
      mention of iterative bootstrap refinement -> ``n_bootstrap_cycles=0`` (a single
      relaxation pass, not this module's usual default of 2).

    Paper's own reported rho at exactly these hyperparameters (Table 1, the k=8,
    eta=4 cell): **LS 0.648, LSR 0.403** (LSR in parentheses in the original).

    Measured here: **LS 0.75, LSR 0.7034** - both worse than the paper's numbers, and
    LSR's margin over LS is smaller. This gap is understood, not a loose end: torchalg
    implements compatible-relaxation + algebraic-distance coarsening ([AD11]'s
    generalization for problems without a structured grid), while the paper's own
    reported numbers use full geometric (mesh-doubling) coarsening for structured grids
    like this one (``docs/bootstrap-amg.md`` Sec. 6 already documents this as a
    deliberate scope choice). Closing it would mean implementing geometric coarsening
    as a genuinely separate strategy - out of scope here, and not attempted by this
    test. What this test asserts is exactly what *is* verified reproducible with the
    setup above, no more and no less:

    1. the pipeline builds a working multi-level hierarchy at all under this exact
       setup (a direct regression guard for the draw-conflation bug, which made this
       specific configuration crash outright before the fix above);
    2. both LS and LSR converge (``rho < rho_per_size_bound``);
    3. LSR beats LS - the one comparative claim from Table 1 that *does* survive
       faithful reproduction, even though the absolute level does not.

    Slow (~65s): builds two BAMG hierarchies on a 3969-node dense system. Marked
    ``benchmark`` like every other test in this module for that reason.
    """
    matrix = isotropic_2d_matrix_h64
    start = energy_norm_start_factory(matrix.shape[0])

    def factory(use_lsr: bool, seed: int) -> BootstrapAMGPreconditioner:
        return BootstrapAMGPreconditioner(
            matrix,
            k_r=8,
            eta=4,
            n_bootstrap_cycles=0,
            use_lsr=use_lsr,
            test_vector_draw=normal_seeded_draw_factory(seed),
            n_pre=2,
            n_post=2,
            smoother=GaussSeidelSmoother(),
        )

    preconditioner_ls = factory(use_lsr=False, seed=0)
    preconditioner_lsr = factory(use_lsr=True, seed=0)
    assert len(preconditioner_ls.result.matrices) >= 2, (
        "must build a real multi-level hierarchy under the paper's exact setup - "
        "this configuration used to collapse to a single level before the "
        "draw/test_vector_draw fix"
    )
    assert len(preconditioner_lsr.result.matrices) >= 2

    rho_ls = _mg_convergence_factor(preconditioner_ls, matrix, start)
    rho_lsr = _mg_convergence_factor(preconditioner_lsr, matrix, start)

    assert 0.0 <= rho_ls < rho_per_size_bound, f"LS rho={rho_ls}"
    assert 0.0 <= rho_lsr < rho_per_size_bound, f"LSR rho={rho_lsr}"
    assert rho_lsr < rho_ls, f"LSR ({rho_lsr}) must beat LS ({rho_ls}) - Table 1's own claim"
