"""Multigrid smoothers, sparse-CSR siblings.

Sparse-only counterparts of ``preconditioners.implementations.amg.smoothers``'s
``JacobiSmoother``/``GaussSeidelSmoother`` (dense; kept separate - see
``docs/plan.md``'s "Correction: dense and sparse must be separate
implementations, not an internal branch"). A smoother object holds only
configuration; ``prepare(A)`` does every matrix-dependent one-off computation
(diagonal inverse and spectral-radius damping for Jacobi, level schedules and
strict-triangle split for Gauss-Seidel) and returns the object that sweeps.
Only sparse CSR input is ever prepared - no ``is_sparse_csr`` branch anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from torchalg.sparse.kernels.diagonal import sparse_diagonal

from ._gauss_seidel_format import SymmetricGaussSeidel, prepare_gauss_seidel
from ._jacobi_omega import RELAXATION_NOMINAL, jacobi_omega

if TYPE_CHECKING:
    from torchalg.multigrid.protocols import MultigridSmoother

_NEAR_ZERO_DIAGONAL_TOL = 1e-14
"""Diagonal entries with magnitude below this are excluded from the Jacobi
update (treated as a zero step for that component)."""


@dataclass(frozen=True, slots=True)
class PreparedJacobi:
    """Weighted Jacobi bound to one sparse CSR matrix.

    Attributes:
        matrix (torch.Tensor): Sparse CSR matrix ``A``, shape ``(n, n)``.
        diag_inv (torch.Tensor): ``omega / diag(A)`` with ``0`` where the
            diagonal is (near) zero, shape ``(n,)``.
    """

    matrix: torch.Tensor
    diag_inv: torch.Tensor

    def _scale(self, residual: torch.Tensor) -> torch.Tensor:
        """``diag_inv * residual`` for a vector ``(n,)`` or a block ``(n, k)``."""
        if residual.ndim == 1:
            return self.diag_inv * residual
        return self.diag_inv.unsqueeze(-1) * residual

    def smooth(self, rhs: torch.Tensor, x: torch.Tensor, steps: int) -> torch.Tensor:
        """Apply ``steps`` sweeps ``x <- x + omega D^-1 (rhs - A x)`` from ``x``.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            x (torch.Tensor): Current iterate, same shape; not modified.
            steps (int): Number of sweeps.

        Returns:
            torch.Tensor: Updated iterate, same shape as ``x``.
        """
        if steps == 0:
            return x.clone()
        for _ in range(steps):
            x = x + self._scale(rhs - self.matrix @ x)
        return x

    def smooth_from_zero(self, rhs: torch.Tensor, steps: int) -> torch.Tensor:
        """Apply ``steps`` sweeps from the zero guess, skipping the first ``A @ 0``.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            steps (int): Number of sweeps.

        Returns:
            torch.Tensor: Iterate after ``steps`` sweeps, same shape as ``rhs``.
        """
        if steps == 0:
            return torch.zeros_like(rhs)
        x = self._scale(rhs)
        for _ in range(steps - 1):
            x = x + self._scale(rhs - self.matrix @ x)
        return x


class JacobiSmoother:
    """Weighted Jacobi smoother for sparse CSR matrices: x <- x + omega * D^{-1} (rhs - Ax).

    Sparse-CSR sibling of the dense
    ``preconditioners.implementations.amg.smoothers.JacobiSmoother``.

    Args:
        omega (float | torch.Tensor | None): Damping factor. ``None``
            (default) is the relaxation rule ``1 / rho(D^{-1}A)``, estimated
            in ``prepare`` (never during sweeps); a float or 0-d tensor fixes it.
    """

    def __init__(self, omega: float | torch.Tensor | None = None) -> None:
        """Store the Jacobi damping factor.

        Args:
            omega (float | None): Damping factor, or ``None`` for ``1 / rho``.
        """
        self._omega = omega

    def prepare(self, A: torch.Tensor) -> PreparedJacobi:
        """Extract the diagonal and resolve the damping, once, for ``A``.

        Args:
            A (torch.Tensor): System matrix (n x n), sparse CSR.

        Returns:
            PreparedJacobi: Jacobi sweeps bound to ``A``.
        """
        diag = sparse_diagonal(A)
        omega = jacobi_omega(A, RELAXATION_NOMINAL, self._omega)
        diag_inv = torch.where(
            diag.abs() > _NEAR_ZERO_DIAGONAL_TOL,
            omega / diag,
            torch.zeros_like(diag),
        )
        return PreparedJacobi(matrix=A, diag_inv=diag_inv)


class GaussSeidelSmoother:
    """Symmetric Gauss-Seidel smoother for sparse CSR matrices.

    Sparse-CSR sibling of the dense
    ``preconditioners.implementations.amg.smoothers.GaussSeidelSmoother``.
    No factorization step: each sweep is a direct triangular solve against
    ``A``'s own entries; ``prepare`` builds the level schedules once.
    """

    def prepare(self, A: torch.Tensor) -> SymmetricGaussSeidel:
        """Build both level schedules and the strict-triangle split of ``A``.

        Args:
            A (torch.Tensor): System matrix (n x n), sparse CSR.

        Returns:
            SymmetricGaussSeidel: Symmetric Gauss-Seidel bound to ``A``.
        """
        return prepare_gauss_seidel(A)


class GaussSeidelStage:
    """Gauss-Seidel that prepares each distinct matrix once, for one setup stage.

    Setup algorithms (compatible relaxation, adaptive and bootstrap stages)
    relax the same level matrices many times, through callables and cycles that
    only receive the matrix. A stage owns one of these for its duration: the
    first sweep on a matrix builds its ``SymmetricGaussSeidel`` and
    every later sweep on that same tensor object reuses it. The cache is keyed
    by tensor identity and holds the matrices alive, so it must not outlive the
    stage (and matrices must not be mutated in place within it); there is no
    process-global instance.

    It is a ``MultigridSmoother``: compatible relaxation, test-vector
    relaxation and the setup cycle all bind it to level matrices with ``prepare``.
    """

    def __init__(self) -> None:
        """Start with no prepared matrices."""
        self._prepared: dict[int, tuple[torch.Tensor, SymmetricGaussSeidel]] = {}

    def prepare(self, A: torch.Tensor) -> SymmetricGaussSeidel:
        """Return the prepared Gauss-Seidel of ``A``, building it on first use.

        Args:
            A (torch.Tensor): System matrix (n x n), sparse CSR.

        Returns:
            SymmetricGaussSeidel: Shared by every later call with this tensor.
        """
        key = id(A)
        entry = self._prepared.get(key)
        if entry is None:
            # The matrix is stored beside its preparation so its id cannot be reused
            # while the entry lives, whichever format the preparation chose.
            entry = (A, prepare_gauss_seidel(A))
            self._prepared[key] = entry
        return entry[1]


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
