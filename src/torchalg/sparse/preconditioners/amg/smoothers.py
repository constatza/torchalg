"""Multigrid smoothers, sparse-CSR siblings.

Sparse-only counterparts of ``preconditioners.implementations.amg.smoothers``'s
``JacobiSmoother``/``GaussSeidelSmoother`` (dense; kept unmodified for
comparison - see ``docs/plan.md``'s "Correction: dense and sparse must be
separate implementations, not an internal branch"). Both classes here are
only ever called with sparse CSR input - no ``is_sparse_csr`` check inside
either ``smooth()`` body.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchalg.sparse.kernels.diagonal import sparse_diagonal

from ._gauss_seidel import sparse_symmetric_gauss_seidel
from ._jacobi_omega import RELAXATION_NOMINAL, jacobi_omega

if TYPE_CHECKING:
    from torchalg.multigrid.protocols import MultigridSmoother

_NEAR_ZERO_DIAGONAL_TOL = 1e-14
"""Diagonal entries with magnitude below this are excluded from the Jacobi
update (treated as a zero step for that component)."""


class JacobiSmoother:
    """Weighted Jacobi smoother for sparse CSR matrices: x <- x + omega * D^{-1} (rhs - Ax).

    Sparse-CSR sibling of the dense
    ``preconditioners.implementations.amg.smoothers.JacobiSmoother`` - same
    name, disambiguated by package (matches ``torch.mm`` vs.
    ``torch.sparse.mm``'s own convention), never imported by the dense
    class.

    Args:
        omega (float | torch.Tensor | None): Damping factor. ``None``
            (default) is the relaxation rule ``1 / rho(D^{-1}A)``, estimated
            once per matrix and cached; a float or 0-d tensor fixes it.
    """

    def __init__(self, omega: float | torch.Tensor | None = None) -> None:
        """Store the Jacobi damping factor.

        Args:
            omega (float | None): Damping factor, or ``None`` for ``1 / rho``.
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
            A (torch.Tensor): System matrix (n x n), sparse CSR.
            rhs (torch.Tensor): Right-hand side vector (n,).
            x (torch.Tensor): Current iterate (n,).
            steps (int): Number of sweeps.

        Returns:
            torch.Tensor: Updated iterate after ``steps`` sweeps.
        """
        diag = sparse_diagonal(A)
        omega = jacobi_omega(A, RELAXATION_NOMINAL, self._omega)
        diag_inv = torch.where(
            diag.abs() > _NEAR_ZERO_DIAGONAL_TOL,
            omega / diag,
            torch.zeros_like(diag),
        )
        x = x.clone()
        for _ in range(steps):
            x = x + diag_inv * (rhs - A @ x)
        return x


class GaussSeidelSmoother:
    """Symmetric Gauss-Seidel smoother for sparse CSR matrices.

    Sparse-CSR sibling of the dense
    ``preconditioners.implementations.amg.smoothers.GaussSeidelSmoother``.
    No factorization step: each sweep is a direct triangular solve against
    ``A``'s own entries, via ``sparse.preconditioners.amg._gauss_seidel
    .sparse_symmetric_gauss_seidel``.
    """

    def smooth(
        self,
        A: torch.Tensor,
        rhs: torch.Tensor,
        x: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """Apply ``steps`` symmetric Gauss-Seidel iterations.

        Args:
            A (torch.Tensor): System matrix (n x n), sparse CSR.
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            x (torch.Tensor): Current iterate, shape ``(n,)`` or ``(n, k)``.
            steps (int): Number of symmetric iterations.

        Returns:
            torch.Tensor: Updated iterate, same shape as ``x``.
        """
        return sparse_symmetric_gauss_seidel(A, x, rhs, steps)


def resolve_jacobi_default(
    smoother: MultigridSmoother | None,
    smoother_omega: float | None,
) -> MultigridSmoother:
    """Select an injected smoother or construct the preset's sparse Jacobi default.

    Sparse-CSR sibling of
    ``preconditioners.implementations.amg.smoothers.resolve_jacobi_default``
    - same name and logic, own copy, since the sparse tree never imports
    the dense tree (``docs/plan.md``).

    Args:
        smoother (MultigridSmoother | None): Explicit smoother strategy.
        smoother_omega (float | None): Damping for the sparse Jacobi default.

    Returns:
        MultigridSmoother: The explicit strategy or a sparse weighted-Jacobi
            smoother.

    Raises:
        ValueError: If both an explicit smoother and Jacobi damping are supplied.
    """
    if smoother is not None and smoother_omega is not None:
        raise ValueError("smoother_omega only applies when smoother is not provided")
    return smoother if smoother is not None else JacobiSmoother(omega=smoother_omega)
