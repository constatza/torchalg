"""Concrete AMG preconditioner presets.

Ported from ``neuralls.domain.solver.preconditioners.implementations.amg.variants``
(see ``docs/plan.md``) with ``NDArray`` translated to ``torch.Tensor``. Each
class bundles a fixed multigrid cycle, smoothed-aggregation coarsening, and
weighted-Jacobi smoother into a single, parametrizable object. Use these
instead of wiring ``AMGPreconditioner`` manually when the standard SA-AMG
configuration suffices.

Note (DIP):
    These are preset/factory leaf classes - they wire concrete domain objects
    in their ``__init__``, the same pattern as ``JacobiPreconditioner``.
    For custom wiring (neural coarsening, polynomial smoothers, etc.) use
    ``AMGPreconditioner`` directly.

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
from .smoothers import JacobiSmoother

if TYPE_CHECKING:
    import torch


class VCycleAMG(AMGPreconditioner):
    """AMG preconditioner with V-cycle and smoothed aggregation (SA-AMG).

    Standard choice for SPD problems arising from elliptic PDEs. One coarse-grid
    correction per level (gamma = 1). Optimal for isotropic problems; use
    ``WCycleAMG`` when the convergence factor is poor.

    Args:
        matrix (torch.Tensor): System matrix A (n x n), SPD.
        n_levels (int): Number of hierarchy levels; must be at least 2.
            3 is an unsourced-but-reasonable default for medium-sized
            problems (n ~ 100-10 000); not derived from a specific paper -
            tune per problem, or use ``TargetDimensionCoarsening``
            (``coarsening.py``) to target a coarse dimension directly.
        omega (float): Jacobi damping factor omega used for both the
            smoother and the prolongation smoother. Default 0.67 ~=
            4/(3*rho(D^{-1}A)) for isotropic SPD with rho ~= 2 (Vanek et al.
            1996, Section 3).
        n_pre (int): Pre-smoothing steps (symmetric pre/post preserves SPD).
        n_post (int): Post-smoothing steps.
        theta (float): Strength-of-connection threshold theta in (0, 1).
            Default 0.25 is a literature/practice-standard default (e.g.
            hypre BoomerAMG's ``strong_threshold``, PyAMG's
            ``smoothed_aggregation_solver``), not from Stuben (2001) - see
            ``strength_of_connection``'s docstring in ``_aggregation.py``.

    References:
        - Vanek, Mandel & Brezina (1996), Sections 3-4 (SA-AMG, V-cycle
          convergence).
        - Briggs, Henson & McCormick (2000), Algorithm 3.7 (V-cycle).
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        n_levels: int = 3,
        omega: float = 0.67,
        n_pre: int = 2,
        n_post: int = 2,
        theta: float = 0.25,
    ) -> None:
        """Wire smoothed-aggregation coarsening and a V-cycle into an AMGPreconditioner.

        Args:
            matrix (torch.Tensor): System matrix A (n x n), SPD.
            n_levels (int): Number of hierarchy levels; must be at least 2.
            omega (float): Jacobi damping factor omega.
            n_pre (int): Pre-smoothing steps.
            n_post (int): Post-smoothing steps.
            theta (float): Strength-of-connection threshold theta in (0, 1).
        """
        # TODO: Split this into smoother_omega and prolongation_omega. They
        # control different Jacobi operations but currently share one preset
        # parameter for API compatibility with the reference implementation.
        super().__init__(
            matrix=matrix,
            coarsening=AggregationCoarsening(theta=theta, omega=omega),
            cycle=VCycle(JacobiSmoother(omega=omega), n_pre=n_pre, n_post=n_post),
            n_levels=n_levels,
            linear=True,
        )


class WCycleAMG(AMGPreconditioner):
    """AMG preconditioner with W-cycle and smoothed aggregation (SA-AMG).

    More robust than ``VCycleAMG``: applies two coarse-grid corrections per
    level (gamma = 2), recomputing the fine-grid residual between them. Better
    convergence factors for anisotropic problems at roughly twice the coarse-grid
    cost. For isotropic SPD problems the iteration counts will be similar to or
    fewer than V-cycle.

    Args:
        matrix (torch.Tensor): System matrix A (n x n), SPD.
        n_levels (int): Number of hierarchy levels; must be at least 2.
        omega (float): Jacobi damping factor omega (same role as in
            ``VCycleAMG``).
        n_pre (int): Pre-smoothing steps.
        n_post (int): Post-smoothing steps.
        theta (float): Strength-of-connection threshold theta in (0, 1).

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

    def __init__(
        self,
        matrix: torch.Tensor,
        n_levels: int = 3,
        omega: float = 0.67,
        n_pre: int = 2,
        n_post: int = 2,
        theta: float = 0.25,
    ) -> None:
        """Wire smoothed-aggregation coarsening and a W-cycle into an AMGPreconditioner.

        Args:
            matrix (torch.Tensor): System matrix A (n x n), SPD.
            n_levels (int): Number of hierarchy levels; must be at least 2.
            omega (float): Jacobi damping factor omega.
            n_pre (int): Pre-smoothing steps.
            n_post (int): Post-smoothing steps.
            theta (float): Strength-of-connection threshold theta in (0, 1).
        """
        # TODO: Split this into smoother_omega and prolongation_omega. They
        # control different Jacobi operations but currently share one preset
        # parameter for API compatibility with the reference implementation.
        super().__init__(
            matrix=matrix,
            coarsening=AggregationCoarsening(theta=theta, omega=omega),
            cycle=WCycle(JacobiSmoother(omega=omega), n_pre=n_pre, n_post=n_post),
            n_levels=n_levels,
            linear=True,
        )
