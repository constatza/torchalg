"""Multigrid smoothers.

Ported from ``neuralls.domain.solver.preconditioners.implementations.amg.smoothers``
(see ``docs/plan.md``) with ``NDArray`` translated to ``torch.Tensor`` and
dense ``torch`` ops replacing numpy. ``JacobiSmoother`` holds no precomputed
tensor state - unlike ``JacobiPreconditioner`` (Stage 3), it is applied to a
*different* matrix ``A`` on every call (one per hierarchy level), so there is
nothing to cache in ``__init__``; only the scalar damping factor ``omega`` is
stored. No ``nn.Module`` inheritance is needed here for the same reason
``CoarseningStrategy``/``MultigridCycle`` implementations don't need it: no
tensor buffers to move under ``.to(device/dtype)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

_NEAR_ZERO_DIAGONAL_TOL = 1e-14
"""Diagonal entries with magnitude below this are excluded from the Jacobi
update (treated as a zero step for that component)."""


class SmootherBase(ABC):
    """Base class for fixed-step iterative error dampers used in multigrid cycles.

    Smoothers are stationary iterative methods applied for a fixed number of steps
    rather than to convergence. Their role is to damp high-frequency error components
    on a single grid level, complementing the coarse-grid correction. They satisfy
    the ``MultigridSmoother`` protocol structurally (parallel nominative/structural
    hierarchies; no import dependency between them).

    Note:
        This ABC and ``MultigridSmoother`` (Protocol) are intentionally separate:
        the Protocol provides structural typing for cycle extension points; this ABC
        provides the concrete implementation hierarchy for shared documentation and
        type-safe subclassing.
    """

    @abstractmethod
    def smooth(
        self,
        A: torch.Tensor,
        rhs: torch.Tensor,
        x: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """Apply ``steps`` smoothing iterations to the system Ax = rhs.

        Args:
            A (torch.Tensor): System matrix on this level (n x n).
            rhs (torch.Tensor): Right-hand side vector (n,).
            x (torch.Tensor): Current iterate (n,).
            steps (int): Number of sweeps (fixed, not convergence-driven).

        Returns:
            torch.Tensor: Updated iterate after ``steps`` sweeps, same shape
                as ``x``.
        """


class JacobiSmoother(SmootherBase):
    """Weighted Jacobi smoother: x <- x + omega * D^{-1} (rhs - Ax).

    Applies ``steps`` stationary Jacobi iterations with damping omega. For smoothed
    aggregation AMG, omega ~= 2/3 is optimal: it annihilates the high-frequency error
    components that the coarse-grid correction cannot reach.

    Args:
        omega (float): Damping factor omega in (0, 1). Default 0.67 ~= 2/3 is
            the standard choice for isotropic SPD problems where
            rho(D^{-1}A) ~= 2 (see References).

    References:
        - Young, D. M. (1954). Iterative methods for solving partial difference
          equations of elliptic type. Trans. Amer. Math. Soc., 76, 92-111.
        - Vanek, P., Mandel, J., & Brezina, M. (1996). Algebraic multigrid by
          smoothed aggregation for second and fourth order elliptic problems.
          Computing, 56(3), 179-196. Section 3: omega = 4 / (3 * rho(D^{-1}A)).
    """

    def __init__(self, omega: float = 0.67) -> None:
        """Store the Jacobi damping factor.

        Args:
            omega (float): Damping factor omega in (0, 1).
        """
        self._omega = omega

    def smooth(
        self,
        A: torch.Tensor,
        rhs: torch.Tensor,
        x: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """Apply ``steps`` weighted Jacobi iterations.

        Args:
            A (torch.Tensor): System matrix (n x n), dense.
            rhs (torch.Tensor): Right-hand side vector (n,).
            x (torch.Tensor): Current iterate (n,).
            steps (int): Number of sweeps.

        Returns:
            torch.Tensor: Updated iterate after ``steps`` sweeps.
        """
        diag = torch.diagonal(A)
        diag_inv = torch.where(
            diag.abs() > _NEAR_ZERO_DIAGONAL_TOL,
            self._omega / diag,
            torch.zeros_like(diag),
        )
        x = x.clone()
        for _ in range(steps):
            x = x + diag_inv * (rhs - A @ x)
        return x
