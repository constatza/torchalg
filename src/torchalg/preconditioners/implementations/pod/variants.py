"""POD-2G preconditioner preset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchalg.utils.pod_basis import compute_pod_basis

from ..amg.amg import AMGPreconditioner
from ..amg.cycle import VCycle
from ..amg.smoothers import resolve_jacobi_default
from .coarsening import PODCoarseningStrategy

if TYPE_CHECKING:
    import torch

    from ..amg.protocols import MultigridSmoother


def pod2g_preconditioner(
    snapshots: torch.Tensor,
    rank: float,
    n_levels: int = 2,
    smoother_omega: float | None = None,
    n_pre: int = 2,
    n_post: int = 2,
    smoother: MultigridSmoother | None = None,
) -> AMGPreconditioner:
    """Build a POD-2G preconditioner (Nikolopoulos et al. 2022, §3.3-3.4).

    Bundles a V-cycle, weighted-Jacobi relaxation, and POD coarsening into a
    single, parametrizable ``AMGPreconditioner`` - the AMG-inspired two-grid
    method where the prolongation/restriction operator is a POD basis fit to
    a snapshot ensemble, instead of algebraic aggregation. Use this instead
    of wiring ``AMGPreconditioner`` manually when the standard POD-2G
    configuration suffices; for custom coarsening/cycle combinations, use
    ``AMGPreconditioner`` directly with a ``PODCoarseningStrategy``.

    Two deliberate deviations from Nikolopoulos et al. (2022) worth
    disclosing explicitly, since the module-level citation otherwise implies
    a straight implementation of their §3.3-3.4 two-grid algorithm:

    - **Relaxation**: the paper's Algorithm 2 uses undamped Gauss-Seidel
      pre/post smoothing (``K = L + V``, ``u_{m+1} = L^{-1} r_m``), which
      has no damping parameter. This function uses weighted Jacobi instead
      (``smoother_omega`` below), reusing the same smoother as this codebase's
      algebraic ``vcycle_amg``/``wcycle_amg`` for a uniform, more easily
      parallelized ``SmootherBase`` across all AMG-family presets. The
      default damping ``1 / rho(D^-1 A)`` (PyAMG's relaxation rule, see
      ``JacobiSmoother``) is not derived from - or needed by - the
      Nikolopoulos et al. paper.
      Pass ``smoother=GaussSeidelSmoother()`` if matching the paper's exact
      relaxer matters for a given comparison.
    - **Snapshot source**: the paper collects snapshots as fully converged
      FEM solutions for sampled parameter values (§3.2). The ``snapshots``
      passed in here may instead be CG solve trajectories (intermediate
      iterates) - an intentional variant, not an error; see the caller that
      builds ``snapshots`` for details. Everything downstream of
      ``snapshots`` (the POD-via-SVD basis construction in
      ``compute_pod_basis``, the Galerkin coarse operator, the two-grid
      correction formula) is unchanged from the paper regardless of which
      snapshot source is used.

    Note (DIP):
        This is a preset/factory function - it wires concrete domain
        objects and forwards them to ``AMGPreconditioner``, the same
        pattern as ``vcycle_amg``/``wcycle_amg``
        (``preconditioners.implementations.amg.variants``). The system
        matrix is bound later by ``.setup(A)``; all persistent tensor state
        (the multigrid hierarchy, whose finest level carries the matrix) is
        owned by ``AMGPreconditioner``.

    Cost:
        Dense ``(n, n)`` matrix, ``r`` the basis rank, the default two levels and
        weighted Jacobi with spectral damping, one right-hand side,
        ``nu = n_pre + n_post``, ``V = 8`` bytes per ``float64``. The call itself fits
        the basis (``compute_pod_basis``, documented there) before any matrix exists.
        Setup, ``AMGPreconditioner.setup``: the Galerkin product of
        ``PODCoarseningStrategy.build_transfer`` (``2n^2 r + 2nr^2``), the Jacobi
        ``prepare`` of the fine level, whose ``rho(D^-1 A)`` estimate ``W_rho`` (``P``
        Arnoldi passes of ``2n^2 m + 4nm(m + 1)`` flops, ``m = min(n, 15)``) is not
        shared with the coarsening here, and the LU factor of the ``r x r`` coarse
        matrix: ``W = 2n^2 r + 2nr^2 + W_rho + 2r^3 / 3 + O(n)``,
        ``Q`` about ``V(n^2 (3 + Pm) + 4nr + 2r^2)``, ``D = O(P m log n) + r`` (the
        Arnoldi chain, then the elimination), ``L = O(P m) + 12`` on a GPU
        (``O(P m^2)`` on the CPU) and ``S = P + 1`` (the Hessenberg copies, the LU
        singularity read).
        Apply, one two-grid V-cycle: ``nu`` dense fine matvecs, the dense restriction
        and prolongation with the ``n x r`` basis and one ``r x r`` LU solve,
        ``W = 2 nu n^2 + 4nr + (3 nu - 1) n + 2r^2``,
        ``Q = V(nu n^2 + 2nr + O(nu n) + r^2)``,
        ``D = (nu + 3)(ceil(log2 n) + 4) + 2r`` approximately, ``L = 4 nu + 3`` and
        ``S = 0``.
        Roofline: apply is memory-bound on the ``nu`` matrix reads; the transfers add
        ``2nr`` flops each, a large share of the cycle at ``r`` in the tens (the
        basis is dense, unlike an aggregation prolongator).
        Work-depth: apply is parallel matvecs and products except the ``2r``-step coarse
        substitution.

    Args:
        snapshots (torch.Tensor): Solution snapshot ensemble, shape
            (n_samples, n_dofs).
        rank (int | float): Fixed mode count (int) or minimum cumulative
            captured energy (float in (0, 1]) - see ``compute_pod_basis``.
        n_levels (int): Total number of grid levels; must be at least 2.
            2 (the paper's POD-2G) is the default; higher values are accepted for a future
            hierarchical-POD coarsening strategy but are not meaningful with
            the single-basis ``PODCoarseningStrategy`` used here.
        smoother_omega (float | None): Weighted-Jacobi damping for pre/post
            relaxation; ``None`` is ``1 / rho(D^-1 A)``.
        n_pre (int): Pre-smoothing steps.
        n_post (int): Post-smoothing steps.
        smoother (MultigridSmoother | None): Explicit solve-time smoother.
            ``None`` selects weighted Jacobi. When supplied,
            ``smoother_omega`` must remain ``None``.

    Returns:
        AMGPreconditioner: Wired with POD coarsening and a V-cycle; call
        ``.setup(A)`` with the SPD system matrix to build it. Its
        ``__str__`` is ``AMGPreconditioner``'s generic
        ``"AMG(n_levels=..., <coarsening>)"`` format (e.g.
        ``"AMG(n_levels=2, POD-2G(rank=10))"``), not a POD-specific override.

    References:
        - Nikolopoulos, S., Kalogeris, I., Stavroulakis, G., & Papadopoulos,
          V. (2022). AI-enhanced iterative solvers for accelerating the
          solution of large-scale parametrized systems. arXiv:2207.02543.
    """
    return AMGPreconditioner(
        coarsening=PODCoarseningStrategy(compute_pod_basis(snapshots, rank)),
        cycle=VCycle(
            resolve_jacobi_default(smoother, smoother_omega),
            n_pre=n_pre,
            n_post=n_post,
        ),
        n_levels=n_levels,
        linear=True,
    )
