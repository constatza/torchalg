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
        omega (float): Weighted-Jacobi damping factor for pre/post
            relaxation.
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
        rank: int | float,
        n_levels: int = 2,
        omega: float = 0.67,
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
            omega (float): Weighted-Jacobi damping factor for pre/post
                relaxation.
            n_pre (int): Pre-smoothing steps.
            n_post (int): Post-smoothing steps.
        """
        coarsening = PODCoarseningStrategy(rank=rank)
        coarsening.fit(snapshots)
        super().__init__(
            matrix=matrix,
            coarsening=coarsening,
            cycle=VCycle(JacobiSmoother(omega=omega), n_pre=n_pre, n_post=n_post),
            n_levels=n_levels,
            linear=True,
        )
