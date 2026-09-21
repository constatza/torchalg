"""POD-2G preconditioner preset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..amg import AMGPreconditioner, JacobiSmoother, VCycle
from .coarsening import PODCoarseningStrategy

if TYPE_CHECKING:
    import torch


class POD2GPreconditioner(AMGPreconditioner):
    """POD-2G preconditioner (Nikolopoulos et al. 2022, §3.3-3.4).

    Bundles a V-cycle, weighted-Jacobi relaxation, and POD coarsening into a
    single, parametrizable object - the AMG-inspired two-grid method where
    the prolongation/restriction operator is a POD basis fit to a snapshot
    ensemble, instead of algebraic aggregation. Use this instead of wiring
    ``AMGPreconditioner`` manually when the standard POD-2G configuration
    suffices; for custom coarsening/cycle combinations, use
    ``AMGPreconditioner`` directly with a ``PODCoarseningStrategy``.

    Two deliberate deviations from Nikolopoulos et al. (2022) worth
    disclosing explicitly, since the class-level citation otherwise implies
    a straight implementation of their §3.3-3.4 two-grid algorithm:

    - **Relaxation**: the paper's Algorithm 2 uses undamped Gauss-Seidel
      pre/post smoothing (``K = L + V``, ``u_{m+1} = L^{-1} r_m``), which
      has no damping parameter. This class uses weighted Jacobi instead
      (``smoother_omega`` below), reusing the same smoother as this codebase's
      algebraic ``VCycleAMG``/``WCycleAMG`` for a uniform, more easily
      parallelized ``SmootherBase`` across all AMG-family presets. The
      default damping ``1 / rho(D^-1 A)`` (PyAMG's relaxation rule, see
      ``JacobiSmoother``) is not derived from - or needed by - the
      Nikolopoulos et al. paper.
      Swap in a Gauss-Seidel smoother via ``AMGPreconditioner`` directly if
      matching the paper's exact relaxer matters for a given comparison.
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
        This is a preset/factory leaf class - it wires concrete domain
        objects in ``__init__``, the same pattern as ``VCycleAMG``/
        ``WCycleAMG`` (``preconditioners.implementations.amg.variants``).
        All persistent tensor state (the finest-level matrix, and - once
        built - the multigrid hierarchy) is owned and buffer-managed by the
        inherited ``AMGPreconditioner``/``nn.Module`` machinery; this class
        adds no state of its own.

    Args:
        matrix (torch.Tensor): System matrix A (n x n), SPD.
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

    References:
        - Nikolopoulos, S., Kalogeris, I., Stavroulakis, G., & Papadopoulos,
          V. (2022). AI-enhanced iterative solvers for accelerating the
          solution of large-scale parametrized systems. arXiv:2207.02543.
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        snapshots: torch.Tensor,
        rank: float,
        n_levels: int = 2,
        smoother_omega: float | None = None,
        n_pre: int = 2,
        n_post: int = 2,
    ) -> None:
        """Wire POD coarsening and a V-cycle into an AMGPreconditioner.

        Args:
            matrix (torch.Tensor): System matrix A (n x n), SPD.
            snapshots (torch.Tensor): Solution snapshot ensemble, shape
                (n_samples, n_dofs).
            rank (int | float): Fixed mode count (int) or minimum cumulative
                captured energy (float in (0, 1]).
            n_levels (int): Total number of grid levels; must be at least 2.
            smoother_omega (float | None): Weighted-Jacobi damping for pre/post
                relaxation; ``None`` is ``1 / rho(D^-1 A)``.
            n_pre (int): Pre-smoothing steps.
            n_post (int): Post-smoothing steps.
        """
        coarsening = PODCoarseningStrategy(rank=rank)
        coarsening.fit(snapshots)
        super().__init__(
            matrix=matrix,
            coarsening=coarsening,
            cycle=VCycle(JacobiSmoother(omega=smoother_omega), n_pre=n_pre, n_post=n_post),
            n_levels=n_levels,
            linear=True,
        )
