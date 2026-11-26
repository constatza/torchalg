"""Concrete AMG preconditioner presets.

Ported from ``neuralls.domain.solver.preconditioners.implementations.amg.variants``
with ``NDArray`` translated to ``torch.Tensor``. Each
factory function bundles a fixed multigrid cycle and smoothed-aggregation
coarsening with a weighted-Jacobi smoother by default, returning a plain
``AMGPreconditioner``. Use these parametrizable presets instead of wiring
``AMGPreconditioner`` manually when the standard SA-AMG configuration
suffices.

Note (DIP):
    These are preset/factory functions - they wire concrete domain objects
    and forward them to ``AMGPreconditioner``, the same pattern as
    ``JacobiPreconditioner``. For custom wiring (neural coarsening,
    polynomial smoothers, etc.) use ``AMGPreconditioner`` directly.

References:
    - Vanek, P., Mandel, J., & Brezina, M. (1996). Algebraic multigrid by
      smoothed aggregation for second and fourth order elliptic problems.
      Computing, 56(3), 179-196.
    - Briggs, W. L., Henson, V. E., & McCormick, S. F. (2000).
      A Multigrid Tutorial (2nd ed.). SIAM.
    - Stuben, K. (2001). A review of algebraic multigrid.
      J. Comput. Appl. Math., 128(1-2), 281-309.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .amg import AMGPreconditioner
from .coarsening import AggregationCoarsening
from .cycle import VCycle, WCycle
from .smoothers import resolve_jacobi_default

if TYPE_CHECKING:
    from .protocols import MultigridSmoother


def vcycle_amg(
    n_levels: int = 3,
    smoother_omega: float | None = None,
    prolongation_omega: float | None = None,
    n_pre: int = 2,
    n_post: int = 2,
    theta: float = 0.25,
    smoother: MultigridSmoother | None = None,
) -> AMGPreconditioner:
    """Build an AMG preconditioner with V-cycle and smoothed aggregation (SA-AMG).

    Standard choice for SPD problems arising from elliptic PDEs. One coarse-grid
    correction per level (gamma = 1). Optimal for isotropic problems; use
    ``wcycle_amg`` when the convergence factor is poor.

    Cost:
        Dense levels ``n_0 = n > n_1 > ... > n_{N-1} = c`` (``N = n_levels``), one
        right-hand side, ``nu = n_pre + n_post``, ``V = 8`` bytes per ``float64``,
        weighted Jacobi with the default spectral damping (an explicit ``smoother``
        replaces the sweep terms with its own cost).
        Setup, ``AMGPreconditioner.setup``: one ``AggregationCoarsening.build_transfer``
        per non-coarsest level (``W_agg,l = 4n_l^2 n_{l+1} + 2n_{l+1}^2 n_l + 5n_l^2 +
        W_rho,l``, see that method), one Jacobi ``prepare`` per non-coarsest level
        (``O(n_l)``: the ``rho`` estimate is memoized on the level matrix and shared
        with the prolongator smoothing when both dampings are spectral), then the LU
        factor of the coarsest matrix (``2c^3 / 3``).
        ``W = sum_l W_agg,l + 2c^3 / 3 + O(sum_l n_l)``,
        ``D = sum_l (3n_l + D_rho,l) + c`` (each level's host aggregation and Arnoldi
        chain, then the elimination), ``L = sum_l (30 + L_rho,l) + 1`` approximately
        and ``S = sum_l (3 + P_l) + 1`` (per level the aggregation reads and ``P_l``
        Hessenberg copies; the singularity read of the LU).
        Apply, one V-cycle: per non-coarsest level ``nu`` dense matvecs (``n_pre - 1``
        sweeps from zero, the residual, ``n_post`` sweeps) and one restriction and one
        prolongation with the dense ``n_l x n_{l+1}`` prolongator, then one LU solve:
        ``W = sum_l [2 nu n_l^2 + 4n_l n_{l+1} + 3 nu n_l] + 2c^2``,
        ``Q = V(sum_l [nu n_l^2 + 2n_l n_{l+1} + O(nu n_l)] + c^2)``,
        ``D = sum_l (nu + 3)(ceil(log2 n_l) + 4) + 2c`` approximately (the levels are
        visited one after the other, then the coarse substitutions),
        ``L = sum_l (4 nu + 2) + 1`` and ``S = 0``.
        Roofline: apply is memory-bound (``W/Q`` about 0.25 flop per byte), the finest
        level's matvecs dominate.
        Work-depth: every term but the ``2c``-step coarse solve is a parallel matvec or
        vector update; with ``c`` in the hundreds the coarse solve sets the depth.

    Args:
        n_levels (int): Number of hierarchy levels; must be at least 2.
            3 is an unsourced-but-reasonable default for medium-sized
            problems (n ~ 100-10 000); not derived from a specific paper -
            tune per problem, or use ``TargetDimensionCoarsening``
            (``coarsening.py``) to target a coarse dimension directly.
        smoother_omega (float | None): Weighted-Jacobi relaxation damping.
            ``None`` (default) is ``1 / rho(D^{-1}A)`` per level (PyAMG's
            rule); a float fixes it.
        prolongation_omega (float | None): Damping of the Jacobi step that
            smooths the tentative prolongator. ``None`` (default) is
            ``(4/3) / rho(D^{-1}A)`` (Vanek et al. 1996, Section 3); a float
            fixes it. ``rho`` is estimated once per level while the level is
            coarsened; the smoother's own estimate happens in ``setup()``.
        n_pre (int): Pre-smoothing steps (symmetric pre/post preserves SPD).
        n_post (int): Post-smoothing steps.
        theta (float): Strength-of-connection threshold theta in (0, 1).
            Default 0.25 is a literature/practice-standard default (e.g.
            hypre BoomerAMG's ``strong_threshold``, PyAMG's
            ``smoothed_aggregation_solver``), not from Stuben (2001) - see
            ``strength_of_connection``'s docstring in ``_aggregation.py``.
        smoother (MultigridSmoother | None): Explicit solve-time smoother.
            ``None`` selects weighted Jacobi. When supplied,
            ``smoother_omega`` must remain ``None``.

    Returns:
        AMGPreconditioner: Wired with smoothed-aggregation coarsening and a
        V-cycle; call ``.setup(A)`` with the system matrix to build it.

    References:
        - Vanek, Mandel & Brezina (1996), Sections 3-4 (SA-AMG, V-cycle
          convergence).
        - Briggs, Henson & McCormick (2000), Algorithm 3.7 (V-cycle).
    """
    return AMGPreconditioner(
        coarsening=AggregationCoarsening(theta=theta, omega=prolongation_omega),
        cycle=VCycle(
            resolve_jacobi_default(smoother, smoother_omega),
            n_pre=n_pre,
            n_post=n_post,
        ),
        n_levels=n_levels,
        linear=True,
    )


def wcycle_amg(
    n_levels: int = 3,
    smoother_omega: float | None = None,
    prolongation_omega: float | None = None,
    n_pre: int = 2,
    n_post: int = 2,
    theta: float = 0.25,
    smoother: MultigridSmoother | None = None,
) -> AMGPreconditioner:
    """Build an AMG preconditioner with W-cycle and smoothed aggregation (SA-AMG).

    More robust than ``vcycle_amg``: applies two coarse-grid corrections per
    level (gamma = 2), recomputing the fine-grid residual between them. Better
    convergence factors for anisotropic problems at roughly twice the coarse-grid
    cost. For isotropic SPD problems the iteration counts will be similar to or
    fewer than V-cycle.

    Cost:
        Notation and setup as in ``vcycle_amg`` (the coarsening and preparation are
        identical, only the cycle differs).
        Setup: ``W = sum_l W_agg,l + 2c^3 / 3 + O(sum_l n_l)``,
        ``D = sum_l (3n_l + D_rho,l) + c``, ``L = sum_l (30 + L_rho,l) + 1``
        approximately and ``S = sum_l (3 + P_l) + 1``.
        Apply, one W-cycle (``gamma = 2``): fine level ``l`` is visited ``2^l`` times
        and the coarsest ``2^{N-1}`` times; each visit pre-smooths, then twice forms the
        residual, restricts, recurses and prolongates, then post-smooths:
        ``W = sum_l 2^l [2(nu + 1) n_l^2 + 8n_l n_{l+1} + (3 nu + 2) n_l]
        + 2^{N-1} 2c^2``,
        ``Q = V(sum_l 2^l [(nu + 1) n_l^2 + 4n_l n_{l+1} + O(nu n_l)] + 2^{N-1} c^2)``,
        ``D = sum_l 2^l (nu + 4)(ceil(log2 n_l) + 4) + 2^{N-1} 2c`` approximately (every
        visit is sequential), ``L = sum_l 2^l (4 nu + 7) + 2^{N-1}`` and ``S = 0``.
        Against the V-cycle the finest level costs one extra matvec and two extra
        transfers per cycle; the coarse work doubles per level, so the coarsest solve's
        ``2^{N-1}`` repetitions dominate the depth for deep hierarchies.

    Args:
        n_levels (int): Number of hierarchy levels; must be at least 2.
        smoother_omega (float | None): Relaxation damping (same role as in
            ``vcycle_amg``).
        prolongation_omega (float | None): Prolongation-smoothing damping
            (same role as in ``vcycle_amg``).
        n_pre (int): Pre-smoothing steps.
        n_post (int): Post-smoothing steps.
        theta (float): Strength-of-connection threshold theta in (0, 1).
        smoother (MultigridSmoother | None): Explicit solve-time smoother.
            ``None`` selects weighted Jacobi. When supplied,
            ``smoother_omega`` must remain ``None``.

    Returns:
        AMGPreconditioner: Wired with smoothed-aggregation coarsening and a
        W-cycle.

    References:
        - Briggs, Henson & McCormick (2000), Section 3.3 (W-cycle /
          gamma-cycle).
        - Trottenberg, Oosterlee & Schuller (2001), Section 2.2.2, Algorithm
          2.2 (mu = 2).
        - Stuben (2001), "A review of algebraic multigrid", J. Comput. Appl.
          Math. 128(1-2), discusses AMG cycle complexity generally; unlike
          the strength-of-connection formula elsewhere in this codebase
          (which was verified against this same paper and found to be
          misattributed - see ``_aggregation.py``), the specific section
          number for this citation was not independently verified against
          the paper text (paywalled) - cited for the general topic only.
    """
    return AMGPreconditioner(
        coarsening=AggregationCoarsening(theta=theta, omega=prolongation_omega),
        cycle=WCycle(
            resolve_jacobi_default(smoother, smoother_omega),
            n_pre=n_pre,
            n_post=n_post,
        ),
        n_levels=n_levels,
        linear=True,
    )
