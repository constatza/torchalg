"""Multigrid eigensolver (MGE) for Bootstrap AMG test-vector enrichment.

Pure functions with no coarsening-strategy state, following the module style
of ``_algebraic_distance.py``/``_least_squares.py``: given a (partial)
Bootstrap AMG hierarchy, compute ``k_e`` eigenvector approximations per
level by solving a small generalized eigenproblem on the coarsest grid and
propagating the result up through the hierarchy via interpolation and
relaxation.

``docs/bootstrap-amg.md`` Sec. 4.2 ("Algorithm 1") is the authoritative
spec, transcribed directly from [BAMG11]/[STATUS14]; every equation number
cited below refers to it.

Index-convention note: the transcribed algorithm iterates "for l = L, ...,
1" using the papers' own 1-indexed, finest-labeled-differently convention
(the doc's adjacent notation, e.g. ``P_l^{l+1}``, is consistent with a
scheme where the loop's lower bound is the finest level, not "level 1 out of
0..L"). Translated to this module's ``levels`` lists (0-indexed, finest
first, coarsest at ``len(levels) - 1``), the algorithm covers *every* level
from the coarsest down to and including the finest - this module's
``multigrid_eigensolver`` produces one entry per level in
``range(len(levels) - 1)`` (every level that has a coarser level to have
been seeded from), which is that same full range translated to 0-indexing.

Not implemented: [STATUS14] eq. 4.4's ``tau_lambda`` rebuild-or-skip test
(``tau_lambda(l, l-1) = |lambda^(l) - lambda^(l-1)| / |lambda^(l-1)|``,
directly confirmed - ``docs/bamg/status14_raw.md``). That measure belongs to
the W-cycle setup scheme (rebuild the hierarchy at every coarser grid
immediately, only when relaxing an eigenvector changed its Rayleigh quotient
by more than a tolerance) - ``bootstrap.py`` only implements V^mu-style
setup, which always rebuilds the whole hierarchy once per bootstrap cycle
regardless of how much any single eigenvector changed, so
``tau_lambda``-gated skipping has no consumer here.

``test_vector_weights`` (``_algebraic_distance.py``) computes ``omega_kappa``
using the ``T = I`` reduction only where that precondition ("on the finest
level, or before any MGE enrichment," ``docs/bootstrap-amg.md`` line 234)
actually holds: ``BAMGCoarsening`` tracks the composite prolongation
incrementally (``_current_T`` in ``bootstrap.py``) and passes the resulting
``T_l`` (this module's own ``T_l = P_l^H P_l`` and
composite-prolongation-chaining formula, both directly confirmed against
[STATUS14] Sec. 4, prose before eq. 4.2 - ``docs/bamg/status14_raw.md``)
into both ``test_vector_weights`` and ``algebraic_distance`` once ``k_e > 0``
makes it genuinely differ from ``I`` at the levels this module enriches.
**TODO(bamg-fidelity, needs-check), not yet resolved:** whether
``omega_kappa`` is actually meant to use this same ``T_l`` is unconfirmed.
[STATUS14] Sec. 3 (``docs/bamg/status14_raw.md`` line 593) states
``omega_kappa`` only via ``||v||_A^2 = <Av,v>``, with no ``T`` term and no
equation number of its own; nothing in [STATUS14] or [AD11], as directly
read from their raw text, links a ``T`` operator to these weights - not a
contradiction (STATUS14's own tables are MGE-free) but not confirmation
either. A prior read of this module attributed the ``T``-weighted
generalization to "[BAMG11] eq. 4.1" - **wrong on its face**: [STATUS14]'s
own eq. 4.1 is the unrelated homogeneous relaxation system ``A_l x_l = 0``,
not a weight formula, so that citation could not have been correct even
setting aside [BAMG11]'s unconfirmed numbering; the underlying uncertainty
about ``T``-weighted ``omega_kappa`` itself remains, just without a
fabricated equation number attached to it. This module's own use of ``T_l``
(the Rayleigh quotient in ``refine_eigenpair``/``coarsest_eigenpairs``) is
unaffected and independently confirmed; only the weight-formula application
is in doubt.

Measured effect (``k_r=8``, ``eta=4``, ``n_bootstrap_cycles=2``, GS smoother,
``seed=5``, this repo's 1D FD Poisson - the same harness as
``tests/benchmarks/preconditioners/test_bootstrap_amg.py``, which itself
targets [STATUS14] Table 2's MGE-*free* baseline (confirmed:
``docs/bamg/status14_raw.md`` - ``k_r=8``, ``eta=4``, varying ``h``) and so
does not exercise ``k_e``): ``k_e=8`` alongside the same ``k_r=8`` roughly
halves the measured convergence factor at the larger sizes - ``rho`` from
``0.51 -> 0.20`` at N=127, ``0.72 -> 0.39`` at N=255, ``0.59 -> 0.31`` at
N=511 - consistent with [STATUS14] Table 4's qualitative claim, confirmed
verbatim, that "using the MGE to enhance the test vectors consistently
improves the performance of the resulting solvers when compared to the
results reported in Table 2." At the smallest
sizes (N=31, N=63) with ``k_e`` comparable to ``k_r``, MGE's enriched test
vectors can instead push compatible-relaxation coarsening to a degenerate
single-level result for some setup seeds (measured: 2 of 8 seeds at N=31) -
see ``bootstrap.py``'s module docstring for why this is a pre-existing
CR-coarsening fragility at small problem sizes, not a defect in this module.

References:
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2015). Bootstrap
      algebraic multigrid: status report, open problems, and outlook. Numer.
      Math. Theor. Meth. Appl. 8(1). arXiv:1406.1819. Cited as [STATUS14]:
      eq. 4.2 (the coarse/fine Rayleigh-quotient transfer identity
      ``bootstrap-amg.md`` Sec. 4.2 follows). [STATUS14] describes the
      composite-interpolation/``T_l = P_l^H P_l`` identity only in prose,
      immediately before eq. 4.2 - no explicit equation number - and never
      labels this procedure "Algorithm 1"; it appears only in prose and in
      Figure 3's schematic.
    - Brandt, A., Brannick, J., Kahl, K., & Livshits, I. (2011). Bootstrap
      AMG. SIAM J. Sci. Comput. 33(2), 612-632. Cited as [BAMG11] where a
      prior read of this module attributed "Sec. 3.1, Algorithm 1, eq. 3.2,
      eq. 3.3-3.5" to the same material: **unconfirmed** - [BAMG11] could
      not be fetched this session (no arXiv listing, every mirror failed),
      so whether it actually uses that section/equation numbering, or a
      formal "Algorithm 1" label [STATUS14] does not use, is not verified
      against the primary source. See ``docs/bootstrap-amg.md`` Sec. 9.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    _Relaxation = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]

_NEAR_ZERO_NORM_TOL = 1e-14
"""Below this ``T``-norm, a relaxed eigenvector is treated as a degenerate
collapse to (near) zero rather than divided by near-zero - the package-wide
treat-as-degenerate convention, shared with ``_least_squares.py``'s
``_NEAR_ZERO_DIAGONAL_TOL``/``_algebraic_distance.py``'s
``_NEAR_ZERO_ENERGY_TOL``. Reachable when the shifted matrix happens to make
relaxation converge exactly (e.g. a diagonal shifted matrix, where
Gauss-Seidel solves the homogeneous system exactly in one sweep, to the
unique solution ``0``); not expected for the dense, non-diagonal Galerkin
operators this module actually runs on, but guarded regardless."""


def composite_transfer_metrics(
    prolongations: list[torch.Tensor] | tuple[torch.Tensor, ...],
    n0: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    """``T_l = P_l^H P_l`` for every level, ``P_l`` the composite interpolation ([STATUS14] Sec. 4, prose before eq. 4.2).

    ``P_l`` maps level ``l``'s space up to the finest level's space by
    chaining every prolongation from ``l`` to ``0``; ``P_0`` is the identity,
    so ``T_0 = I`` (``docs/bootstrap-amg.md`` line 234: "on the finest level
    ... T = I").

    Args:
        prolongations (list[torch.Tensor] | tuple[torch.Tensor, ...]):
            Prolongations, one per level except the coarsest;
            ``prolongations[i]`` maps level ``i + 1`` up to level ``i``.
        n0 (int): Finest-level dimension.
        dtype (torch.dtype): Dtype of the returned matrices.
        device (torch.device): Device of the returned matrices.

    Returns:
        list[torch.Tensor]: ``T_l`` for ``l = 0, ..., len(prolongations)``,
        one more entry than ``prolongations`` (finest through coarsest).
    """
    composite = torch.eye(n0, dtype=dtype, device=device)
    metrics = [composite.T @ composite]
    for prolongation in prolongations:
        composite = composite @ prolongation
        metrics.append(composite.T @ composite)
    return metrics


def coarsest_eigenpairs(
    matrix: torch.Tensor, T: torch.Tensor, k_e: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve ``A w = lambda T w`` directly for the ``k_e`` smallest eigenvalues ([STATUS14] Sec. 4's MGE procedure).

    Reduced to a standard symmetric eigenproblem by Cholesky-whitening ``T``:
    ``T = L L^H``, ``A' = L^{-1} A L^{-H}``, ``eigh(A') = (mu, y)``,
    ``w = L^{-H} y`` (``A'``'s eigenvalues equal the generalized problem's;
    ``w`` satisfies ``A w = mu T w`` by substitution). ``torch.linalg.eigh``
    returns eigenvalues ascending, matching "the smallest-eigenvalue ones"
    ([STATUS14]'s own selection rule for this step) with a plain slice.

    Args:
        matrix (torch.Tensor): Coarsest-level SPD matrix ``A_L``, shape
            ``(n, n)``.
        T (torch.Tensor): Coarsest-level composite-interpolation Gram
            operator ``T_L``, shape ``(n, n)``, SPD.
        k_e (int): Number of eigenpairs to keep; clamped to ``n`` if larger.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(eigenvalues, eigenvectors)``,
        shapes ``(k,)`` and ``(n, k)``, ``k = min(k_e, n)``, ascending.
    """
    cholesky_factor = torch.linalg.cholesky(T)
    identity = torch.eye(cholesky_factor.shape[0], dtype=T.dtype, device=T.device)
    inverse_factor = torch.linalg.solve_triangular(cholesky_factor, identity, upper=False)
    reduced = inverse_factor @ matrix @ inverse_factor.T
    reduced = (reduced + reduced.T) / 2  # guard against float asymmetry before eigh
    eigenvalues, eigenvectors = torch.linalg.eigh(reduced)
    k = min(k_e, eigenvalues.shape[0])
    return eigenvalues[:k], inverse_factor.T @ eigenvectors[:, :k]


def refine_eigenpair(
    matrix: torch.Tensor,
    T: torch.Tensor,
    eigenvalue: float,
    eigenvector: torch.Tensor,
    relaxation: _Relaxation,
    sweeps: int,
) -> tuple[float, torch.Tensor]:
    """Relax an interpolated eigenvector on the shifted homogeneous problem, refresh its Rayleigh quotient.

    ``relax on (A_l - lambda_l T_l) w_l = 0``, then ``lambda_l = <A_l w_l,
    w_l> / <T_l w_l, w_l>`` ([STATUS14] Sec. 4's per-level step). This
    "resembles inverse Rayleigh-quotient iteration with the inverse replaced
    by relaxation sweeps" (``docs/bootstrap-amg.md`` Sec. 4.2) - and, like
    ordinary Rayleigh-quotient iteration, renormalizes to unit ``T``-norm
    after *every single sweep* (not once after all ``sweeps``). This is not
    an extra step beyond what "resembles RQI" already implies:
    renormalization is intrinsic to RQI (without it, the iteration is not
    RQI, merely plain relaxation on a matrix that ``A - lambda T`` need not
    be positive definite for - any eigenvalue past the smallest few makes it
    indefinite, where unnormalized relaxation on a homogeneous system has no
    fixed point to converge to and diverges geometrically). Renormalizing
    only after a whole multi-sweep batch is not sufficient: growth can
    overflow to ``inf``/``nan`` *within* that batch before a single post-hoc
    renormalization ever runs - measured directly (eigenvector-column norms
    reaching ``1e99`` within a handful of sweeps with an unnormalized
    multi-sweep call; ``nan`` with ``sweeps=8`` in one batch even with
    post-hoc renormalization only at the end).

    Args:
        matrix (torch.Tensor): Level matrix ``A_l``, shape ``(n, n)``.
        T (torch.Tensor): Composite-interpolation Gram operator ``T_l``,
            shape ``(n, n)``.
        eigenvalue (float): Eigenvalue approximation ``lambda_l`` to shift by
            (inherited from the coarser level before relaxation).
        eigenvector (torch.Tensor): Interpolated eigenvector approximation
            ``w_l``, shape ``(n,)``.
        relaxation (_Relaxation): Relaxation callable, ``(A, rhs, x, steps)
            -> x``.
        sweeps (int): Number of relaxation sweeps.

    Returns:
        tuple[float, torch.Tensor]: Refreshed ``(lambda_l, w_l)``.
    """
    shifted = matrix - eigenvalue * T
    zero_rhs = torch.zeros_like(eigenvector)
    relaxed = eigenvector
    for _ in range(sweeps):
        # One sweep at a time, renormalizing after each: `shifted` need not
        # be positive definite (any eigenvalue past the smallest few makes
        # it indefinite), so unnormalized relaxation has no fixed point and
        # can overflow to inf/nan *within* a multi-sweep call before a
        # single post-hoc renormalization ever runs - measured directly.
        candidate = relaxation(shifted, zero_rhs, relaxed, 1)
        t_norm = torch.sqrt(candidate @ (T @ candidate))
        if t_norm.abs() <= _NEAR_ZERO_NORM_TOL:
            break  # degenerate collapse (see _NEAR_ZERO_NORM_TOL) - keep the last valid iterate
        relaxed = candidate / t_norm
    rayleigh = float((relaxed @ (matrix @ relaxed)) / (relaxed @ (T @ relaxed)))
    return rayleigh, relaxed


def multigrid_eigensolver(
    levels: list[torch.Tensor] | tuple[torch.Tensor, ...],
    prolongations: list[torch.Tensor] | tuple[torch.Tensor, ...],
    k_e: int,
    relaxation: _Relaxation,
    sweeps: int,
) -> dict[int, torch.Tensor]:
    """[STATUS14] Sec. 4's MGE procedure: ``k_e`` eigenvector approximations for every level but the coarsest.

    Solves the coarsest level's generalized eigenproblem directly
    (``coarsest_eigenpairs``), then interpolates and relaxes up through
    every finer level (``refine_eigenpair``), one level at a time - see the
    module docstring's index-convention note for why this covers every
    level down to and including the finest, matching the papers' own
    ``l = L, ..., 1`` range under their differing level-numbering.

    The coarsest level itself is never a *consumer* of enrichment (there is
    no coarser level below it to have seeded it from) - it is only
    Algorithm 1's starting point, so it has no entry in the returned dict,
    matching ``BootstrapSetup._improve_levels``'s own
    ``range(len(levels) - 1)`` shape for the same reason.

    Args:
        levels (list[torch.Tensor] | tuple[torch.Tensor, ...]): Level
            matrices, finest first.
        prolongations (list[torch.Tensor] | tuple[torch.Tensor, ...]):
            Prolongations, one per level except the coarsest.
        k_e (int): Number of eigenvector approximations to keep (clamped to
            each level's dimension where smaller).
        relaxation (_Relaxation): Relaxation callable used to refine
            interpolated eigenvectors, ``(A, rhs, x, steps) -> x``.
        sweeps (int): Relaxation sweeps per level (reuses the caller's
            ``eta``; the transcribed algorithm does not specify a separate
            MGE-only sweep count).

    Returns:
        dict[int, torch.Tensor]: ``k_e``-column eigenvector approximations
        keyed by matrix dimension, for every level except the coarsest.
    """
    metrics = composite_transfer_metrics(
        prolongations, n0=levels[0].shape[0], dtype=levels[0].dtype, device=levels[0].device
    )
    coarsest_index = len(levels) - 1
    current_values, current_vectors = coarsest_eigenpairs(
        levels[coarsest_index], metrics[coarsest_index], k_e
    )

    enriched: dict[int, torch.Tensor] = {}
    for index in range(coarsest_index - 1, -1, -1):
        interpolated = prolongations[index] @ current_vectors
        T = metrics[index]
        refined = [
            refine_eigenpair(
                levels[index], T, float(current_values[k]), interpolated[:, k], relaxation, sweeps
            )
            for k in range(interpolated.shape[1])
        ]
        current_values = torch.tensor(
            [value for value, _ in refined], dtype=levels[index].dtype, device=levels[index].device
        )
        current_vectors = torch.stack([vector for _, vector in refined], dim=1)
        enriched[levels[index].shape[0]] = current_vectors
    return enriched
