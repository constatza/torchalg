"""Multigrid smoothers.

Ported from ``neuralls.domain.solver.preconditioners.implementations.amg.smoothers``
with ``NDArray`` translated to ``torch.Tensor`` and
dense ``torch`` ops replacing numpy. A smoother object holds only configuration
(the Jacobi damping override); ``prepare(A)`` binds it to one level matrix and does
every matrix-dependent one-off computation there (diagonal inverse, spectral-radius
damping, triangular factors), returning a ``PreparedSmoother`` that only sweeps.
No ``nn.Module`` inheritance is needed for the same reason ``CoarseningStrategy``/
``MultigridCycle`` implementations don't need it: no tensor buffers to move under
``.to(device/dtype)``; the prepared state lives on the hierarchy level.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from torchalg.preconditioners.base import Placement

from ._jacobi_omega import RELAXATION_NOMINAL, jacobi_omega
from ._relaxation import PreparedSymmetricGaussSeidel, prepare_symmetric_gauss_seidel

if TYPE_CHECKING:
    from .protocols import MultigridSmoother, PreparedSmoother

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

    Attributes:
        placement (Placement): What bounds one sweep; each concrete smoother sets it.
    """

    placement: Placement

    @abstractmethod
    def prepare(self, A: torch.Tensor) -> PreparedSmoother:
        """Bind the smoother to ``A``, doing all matrix-dependent work once.

        Args:
            A (torch.Tensor): System matrix on this level (n x n), dense.

        Returns:
            PreparedSmoother: Smoother whose sweeps reuse the prepared state.
        """


@dataclass(frozen=True, slots=True)
class PreparedJacobi:
    """Weighted Jacobi bound to one matrix: ``x <- x + diag_inv * (rhs - A x)``.

    Cost:
        Setup: none of its own (``W = 0``, ``D = 0``, ``L = 0``, ``S = 0``); it is
        constructed with a ready ``diag_inv`` by ``JacobiSmoother.prepare``, which
        carries the work.
        Apply, one sweep of ``smooth``, dense ``(n, n)`` level matrix, ``k`` columns,
        ``V = 8`` bytes per ``float64``: ``W = 2n^2 k + 3nk`` (the dense matvec, then
        subtract, scale and add), ``Q = V(n^2 + 11nk)`` (the matrix read once, the
        ``n x k`` operands of four kernels), ``D = ceil(log2 n) + 4`` (the matvec's row
        reduction, then three dependent elementwise kernels), ``L = 4`` and ``S = 0``.
        ``smooth_from_zero`` with ``s`` sweeps skips the first matvec:
        ``W = nk + (s - 1)(2n^2 k + 3nk)``, ``L = 1 + 4(s - 1)``. ``residual`` is one
        matvec and a subtraction: ``W = 2n^2 k + nk``, ``L = 2``.
        Roofline: a sweep's ``W/Q`` is about 0.25 flop per byte for ``k = 1``
        (memory-bound on the matrix read).
        Work-depth: ``W/D = 2n^2 k / log2 n``, ample parallelism (``SPMV_BOUND``).

    Attributes:
        matrix (torch.Tensor): The level matrix ``A``, shape ``(n, n)``.
        diag_inv (torch.Tensor): ``omega / diag(A)``, zero where the diagonal is
            near zero, shape ``(n,)``. The damping is already folded in.
    """

    matrix: torch.Tensor
    diag_inv: torch.Tensor

    def residual(self, rhs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return ``rhs - A x`` with the bound level matrix.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            x (torch.Tensor): Current iterate, same shape as ``rhs``.

        Returns:
            torch.Tensor: Residual, same shape as ``rhs``.
        """
        return rhs - self.matrix @ x

    def _scale_for(self, rhs: torch.Tensor) -> torch.Tensor:
        return self.diag_inv.unsqueeze(-1) if rhs.ndim == 2 else self.diag_inv

    def smooth(self, rhs: torch.Tensor, x: torch.Tensor, steps: int) -> torch.Tensor:
        """Apply ``steps`` weighted Jacobi iterations from ``x``.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            x (torch.Tensor): Current iterate (not modified), same shape as ``rhs``.
            steps (int): Number of sweeps.

        Returns:
            torch.Tensor: Updated iterate after ``steps`` sweeps.
        """
        scale = self._scale_for(rhs)
        for _ in range(steps):
            x = x + scale * (rhs - self.matrix @ x)
        return x

    def smooth_from_zero(self, rhs: torch.Tensor, steps: int) -> torch.Tensor:
        """Apply ``steps`` weighted Jacobi iterations from the zero iterate.

        The first sweep is ``diag_inv * rhs`` (no ``A @ 0`` matvec).

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            steps (int): Number of sweeps.

        Returns:
            torch.Tensor: Iterate after smoothing, same shape as ``rhs``.
        """
        if steps == 0:
            return torch.zeros_like(rhs)
        scale = self._scale_for(rhs)
        x = scale * rhs
        for _ in range(steps - 1):
            x = x + scale * (rhs - self.matrix @ x)
        return x


class JacobiSmoother(SmootherBase):
    """Weighted Jacobi smoother: x <- x + omega * D^{-1} (rhs - Ax).

    Applies ``steps`` stationary Jacobi iterations with damping omega, which must
    stay below ``2 / rho(D^{-1}A)`` for the iteration to converge.

    Cost:
        Setup, ``prepare`` on a dense ``(n, n)`` matrix, ``V = 8`` bytes per
        ``float64``: with the default ``omega = None`` the damping is ``1 / rho``,
        estimated by ``jacobi_spectral_radius``: ``D^-1 A`` (``W = n^2 + n``,
        ``Q = 2Vn^2``), then ``P`` Arnoldi passes of ``m = min(n, 15)`` steps
        (``P`` usually 1 to 2, see ``utils.spectral.approximate_spectral_radius``), so
        ``W = n^2 + P(2n^2 m + 4nm(m + 1)) + O(n)``, ``Q`` about ``V n^2 (2 + Pm)``,
        ``D = O(P m log n)`` (the Krylov steps are a dependent chain),
        ``L = O(P m)`` on a GPU (block Gram-Schmidt) and ``O(P m^2)`` on the CPU
        (modified), and ``S = P`` (one Hessenberg copy per pass). The estimate is
        memoized per matrix object, so a level whose aggregation coarsening already
        smoothed its prolongator with the same estimate pays none of it here.
        An explicit ``omega`` leaves the damped diagonal inverse: ``W = 2n``
        (comparison, division), ``Q = 5Vn``, ``D = 3``, ``L = 5`` and ``S = 0``.
        Apply, one sweep: ``PreparedJacobi``'s, ``W = 2n^2 k + 3nk``,
        ``D = ceil(log2 n) + 4``, ``L = 4`` and ``S = 0``.

    Args:
        omega (float | torch.Tensor | None): Damping factor. ``None``
            (default) is the relaxation rule ``1 / rho(D^{-1}A)`` (PyAMG's),
            estimated once per matrix inside ``prepare`` (see ``_jacobi_omega``);
            a float or 0-d tensor fixes it (0.67 ~= 2/3 is ``4/(3 rho)`` for
            ``rho ~= 2``) - a tensor is accepted so callers holding an
            already-computed spectral-radius-derived value never have to
            round-trip it through Python before passing it back in.

    Attributes:
        placement (Placement): ``SPMV_BOUND``; a sweep is one matvec plus vector updates.

    References:
        - Young, D. M. (1954). Iterative methods for solving partial difference
          equations of elliptic type. Trans. Amer. Math. Soc., 76, 92-111.
        - Vanek, P., Mandel, J., & Brezina, M. (1996). Algebraic multigrid by
          smoothed aggregation for second and fourth order elliptic problems.
          Computing, 56(3), 179-196. Section 3: omega = 4 / (3 * rho(D^{-1}A)).
    """

    placement = Placement.SPMV_BOUND

    def __init__(self, omega: float | torch.Tensor | None = None) -> None:
        """Store the Jacobi damping override.

        Args:
            omega (float | torch.Tensor | None): Damping factor, or ``None`` for ``1 / rho``.
        """
        self._omega = omega

    def prepare(self, A: torch.Tensor) -> PreparedJacobi:
        """Resolve the damping and the damped diagonal inverse of ``A``, eagerly.

        Args:
            A (torch.Tensor): System matrix (n x n), dense.

        Returns:
            PreparedJacobi: Jacobi sweeps bound to ``A``.
        """
        diag = torch.diagonal(A)
        omega = jacobi_omega(A, RELAXATION_NOMINAL, self._omega)
        diag_inv = torch.where(
            diag.abs() > _NEAR_ZERO_DIAGONAL_TOL,
            omega / diag,
            torch.zeros_like(diag),
        )
        return PreparedJacobi(matrix=A, diag_inv=diag_inv)


class GaussSeidelSmoother(SmootherBase):
    """Symmetric Gauss-Seidel smoother: one forward then one backward sweep per step.

    The smoother PyAMG's adaptive SA uses by default (``('gauss_seidel',
    {'sweep': 'symmetric'})``); with equal pre- and post-smoothing counts the
    resulting cycle is symmetric, so plain PCG is valid. Rows with a zero
    diagonal are left unchanged.

    Cost:
        Setup, ``prepare`` on a dense ``(n, n)`` matrix: ``prepare_symmetric_gauss_seidel``,
        with no zero diagonal ``W = 0`` (two strict-triangle copies),
        ``Q = 4Vn^2``, ``D = 3``, ``L = 6`` and ``S = 2`` (one frozen-row test per
        direction); see that function for frozen rows.
        Apply, one symmetric iteration, ``k`` columns: ``PreparedSymmetricGaussSeidel``'s,
        ``W = 6n^2 k + 2nk``, ``D = 2(n + ceil(log2 n) + 2)``, ``L = 6`` and ``S = 0``.

    Attributes:
        placement (Placement): ``SEQUENTIAL_SWEEP``; each sweep is a chain of dependent
            triangular solves.
    """

    placement = Placement.SEQUENTIAL_SWEEP

    def prepare(self, A: torch.Tensor) -> PreparedSymmetricGaussSeidel:
        """Build the forward and backward triangular systems of ``A`` once.

        Args:
            A (torch.Tensor): System matrix (n x n), dense.

        Returns:
            PreparedSymmetricGaussSeidel: Gauss-Seidel sweeps bound to ``A``.
        """
        return prepare_symmetric_gauss_seidel(A)


def resolve_jacobi_default(
    smoother: MultigridSmoother | None,
    smoother_omega: float | None,
) -> MultigridSmoother:
    """Select an injected smoother or construct the preset's Jacobi default.

    ``smoother_omega`` belongs specifically to the default Jacobi strategy;
    accepting it alongside an injected smoother would silently ignore user
    configuration.

    Args:
        smoother (MultigridSmoother | None): Explicit smoother strategy.
        smoother_omega (float | None): Damping for the Jacobi default.

    Returns:
        MultigridSmoother: The explicit strategy or a weighted-Jacobi smoother.

    Raises:
        ValueError: If both an explicit smoother and Jacobi damping are supplied.
    """
    if smoother is not None and smoother_omega is not None:
        raise ValueError("smoother_omega only applies when smoother is not provided")
    return smoother if smoother is not None else JacobiSmoother(omega=smoother_omega)
