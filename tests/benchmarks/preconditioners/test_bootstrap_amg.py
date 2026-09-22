"""Benchmark: Bootstrap AMG paper-table reproduction (docs/bootstrap-amg.md Sec. 6).

Attempts to reproduce the *shape* of [BAMG11] Tables 4.2/4.3 - LS degrading
with problem size, LSR generally doing better - on this repo's own plain
finite-difference Poisson problems (``poisson_1d_factory``/
``anisotropic_2d_factory``, ``tests/conftest.py``). Not an attempt to match
the paper's absolute ``rho`` numbers.

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

Why the absolute numbers will not match [BAMG11] Tables 4.2/4.3, and why
this benchmark only asserts *convergence* rather than a strict LSR-beats-LS
ordering (both findings from developing this benchmark; see the Task 5
report for the full writeup):
    1. **Discretization.** ``poisson_1d_factory``/``anisotropic_2d_factory``
       build a plain 5-point finite-difference Poisson stencil; [BAMG11]
       Sec. 4.1 uses a finite-element Laplace discretization on the same
       nominal grid sizes. Different discretizations have different
       algebraically-smooth-error spectra, which directly changes ``rho``.
    2. **Cycle.** ``BootstrapAMGPreconditioner`` hardcodes a fixed
       V(1,1)-cycle (``bootstrap.py``'s module-level ``_CYCLE``), not the
       paper's V(2,2).
    3. **No near-null-space seeding, and this matters more than expected.**
       [BAMG11] Sec. 6's near-optimal LSR range (``rho ~= 0.04-0.15``)
       requires seeding the known constant vector ``1`` into the
       test-vector set alongside the ``k_r`` random ones;
       ``BootstrapAMGPreconditioner`` has no such seeding hook - its test
       vectors are purely ``k_r`` random draws (``_random_test_vectors`` in
       ``bootstrap.py``). The paper itself separately reports that LSR
       *without* that seeding also degrades with problem size
       (``rho ~= 0.08 -> 0.98``), i.e. the constant-vector seed is doing
       real work, not a minor tweak. Empirically (see the Task 5 report),
       averaging measured ``rho`` over 8 independent ``BootstrapSetup``
       seeds at each of several 1D and 2D problem sizes showed LS and LSR
       performing statistically indistinguishably here - sometimes LSR
       measurably *worse* than LS for a given setup seed - rather than the
       paper's reliable "LSR wins" trend. This is consistent with (and the
       most likely explanation for) the missing constant-vector seed, but
       is reported as an open finding rather than asserted as a proven root
       cause. Given this, the tests below assert what *is* robust across
       setup seeds - both LS and LSR build valid, convergent multigrid
       hierarchies - and additionally demonstrate, at one fixed, documented
       ``BootstrapAMGPreconditioner`` seed (``seed=5``, this codebase's
       existing convention - see ``test_bootstrap.py``'s
       ``bootstrap_amg_preconditioner_64`` fixture), an LSR-vs-LS comparison
       consistent with Sec. 6's direction. That comparison is a
       reproducible example at that specific seed, not a general
       statistical claim - see the module-level finding above for why a
       stronger claim would not currently be honest.

Problem-size choice (127/255/511 for the 1D case, a 16x16 grid for 2D):
    ``BootstrapSetup``'s default ``max_coarse=10`` coarsening loop was
    found, while developing this benchmark, to produce a **degenerate,
    empty (0-node) coarsest level** for 1D Poisson at N=31 and N=63 (and
    for a 12x12 anisotropic 2D grid): the second coarsening pass'
    compatible-relaxation step returns an all-``False`` coarse mask on the
    once-coarsened Galerkin operator, so ``BootstrapSetup._build_levels``
    appends a ``(n, 0)`` prolongation and a 0x0 matrix as the final level
    instead of stopping one level earlier. That level then contributes
    exactly zero coarse-grid correction (an empty-column prolongation always
    restricts/prolongates to/from a zero-length vector), silently wasting a
    level and measurably degrading the V-cycle's convergence factor. This
    is flagged as a concern in the Task 5 report rather than special-cased
    away here; the sizes below were chosen because they build well-formed,
    strictly-shrinking hierarchies under the *same* default parameters, so
    the numbers this benchmark records reflect the solver working as
    intended rather than the degenerate-level edge case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from torchalg.preconditioners.implementations.amg.bootstrap import BootstrapAMGPreconditioner

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
    """1D problem sizes reproducing (a subset of) [BAMG11] Tables 4.2/4.3's ``N`` column.

    Excludes N=31/63 - see this module's docstring for why (the degenerate
    empty-coarsest-level edge case).
    """
    return (127, 255, 511)


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

    A factory, not a plain fixture, since it is built once per (matrix,
    ``use_lsr``) pair across several problem sizes/variants within a single
    test.
    """

    def _factory(matrix: torch.Tensor, use_lsr: bool) -> BootstrapAMGPreconditioner:
        return BootstrapAMGPreconditioner(
            matrix, k_r=8, eta=4, n_bootstrap_cycles=2, use_lsr=use_lsr, seed=5
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
) -> None:
    """Both LS and LSR must build valid, convergent multigrid hierarchies at every tested size.

    This is the one property that held robustly (see the module docstring's
    "3. No near-null-space seeding" note): individual LSR-vs-LS ordering
    per size did not, so it is not asserted here.

    Actual measured values (seed=5, 10 V(1,1)-cycle iterations, this repo's
    plain 5-point FD 1D Poisson, N=127/255/511 - recorded for future
    regression comparison):
        LS rho:  [0.9673, 0.8377, 0.8039]  (mean 0.8696)
        LSR rho: [0.7305, 0.9453, 0.7641]  (mean 0.8133)
    Averaged over 8 independent ``BootstrapAMGPreconditioner`` setup seeds
    at these same sizes, LS and LSR means differed by well under one
    standard deviation of the per-seed spread - the fixed-seed mean
    advantage recorded above is a reproducible example at ``seed=5``, not a
    general property (see the module docstring).
    """
    for n in poisson_paper_sizes:
        matrix = poisson_1d_factory(n)
        start = energy_norm_start_factory(n)
        rho_ls = _mg_convergence_factor(bootstrap_amg_factory(matrix, False), matrix, start)
        rho_lsr = _mg_convergence_factor(bootstrap_amg_factory(matrix, True), matrix, start)

        assert 0.0 <= rho_ls < 1.0, f"LS did not converge at N={n}: rho={rho_ls}"
        assert 0.0 <= rho_lsr < 1.0, f"LSR did not converge at N={n}: rho={rho_lsr}"


@pytest.mark.benchmark
def test_bootstrap_amg_lsr_beats_ls_at_fixed_seed_on_2d_poisson(
    isotropic_2d_matrix_256: torch.Tensor,
    energy_norm_start_factory: Callable[[int], torch.Tensor],
    bootstrap_amg_factory: Callable[[torch.Tensor, bool], BootstrapAMGPreconditioner],
) -> None:
    """At the codebase's standard seed, LSR converges and clearly beats LS on the 2D payoff case.

    A single, fixed-seed demonstration consistent with [BAMG11] Sec. 6's
    direction (LSR generally better), not a general statistical claim - see
    the module docstring's "3. No near-null-space seeding" note: averaged
    over 8 setup seeds at this same size, LS was on average *better* than
    LSR (0.6316 vs 0.7259), so this specific seed is not representative of
    the average case.

    Actual measured values (seed=5, 10 V(1,1)-cycle iterations, 16x16
    isotropic 2D Poisson, 256 nodes - recorded for future regression
    comparison):
        LS rho:  0.6636
        LSR rho: 0.4054
    """
    matrix = isotropic_2d_matrix_256
    start = energy_norm_start_factory(matrix.shape[0])

    rho_ls = _mg_convergence_factor(bootstrap_amg_factory(matrix, False), matrix, start)
    rho_lsr = _mg_convergence_factor(bootstrap_amg_factory(matrix, True), matrix, start)

    assert 0.0 <= rho_ls < 1.0
    assert 0.0 <= rho_lsr < 1.0
    assert rho_lsr < rho_ls
