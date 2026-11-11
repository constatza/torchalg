"""Incomplete LU (ILU(0)) preconditioner.

Dense masked port of the reference's `scipy.sparse.linalg.spilu`-backed
factorization: a natural-ordering, no-pivoting ILU(0) (Saad Algorithm 10.4)
computed on a dense ``torch.Tensor`` clone of ``A``, restricted to ``A``'s
original non-zero pattern - see ``_masked_factorization.py`` and
``docs/plan.md``'s dense-only directive.

Capability reduction from the reference, disclosed explicitly: the
reference's `scipy.sparse.linalg.spilu` accepted `drop_tol`/`fill_factor`,
allowing configurable fill-in beyond `A`'s original sparsity pattern (a
general incomplete LU, not strictly ILU(0)). This port is strict ILU(0)
only - no fill-in beyond the original pattern, and no `drop_tol`/
`fill_factor` parameters at all. `ILUPreconditioner.__init__` takes only
`matrix`.
"""

from __future__ import annotations

from typing import Self

import torch
from torch import nn

from ..base import LinearPreconditioner, PreconditionerContext
from ._masked_factorization import dense_ilu0


class ILUPreconditioner(LinearPreconditioner[torch.Tensor], nn.Module):
    """Incomplete LU preconditioner: ``z = (LU)^{-1}r``.

    Incomplete LU factorization preconditioner. More effective than Jacobi,
    but more expensive to apply.

    Follows the ``Preconditioner`` + ``nn.Module`` pattern established by
    ``JacobiPreconditioner``: ``nn.Module.__init__()`` is called explicitly
    and the combined ``L``/``U`` factor is stored via ``register_buffer``
    (never ``register_parameter`` - it is never learned or
    autograd-tracked) so it participates in ``.to(device=..., dtype=...)``
    propagation.

    Mathematical Properties:
        - ``M ≈ A`` via incomplete LU factorization (no fill-in beyond
          ``A``'s original non-zero pattern - natural ordering, no
          pivoting).
        - ``z = (LU)^{-1}r`` via forward/backward triangular solves.
        - Not guaranteed SPD even if ``A`` is SPD.

    Complexity (corrected from a prior ``O(nnz)`` claim, which described the
    sparse-textbook ILU(0) algorithm, not this dense implementation):
        - Storage: ``O(n^2)`` — the combined ``L``/``U`` factor is a dense
          ``(n, n)`` buffer, not a sparse structure sized by ``nnz(A)``.
        - Setup: ``O(sum_k nnz(L[:, k]) * nnz(U[k, :]))`` plus ``O(n^2)`` for
          the pattern scan — ``_masked_factorization.dense_ilu0`` is a
          right-looking elimination with one masked rank-1 update per pivot,
          over the stored rows and columns only, with no host
          synchronization per entry; a dense pattern reaches the classical
          dense-LU ``O(n^3)``.
        - Application: ``O(n^2)`` — ``apply()`` runs two dense
          ``torch.linalg.solve_triangular`` calls directly on the packed
          ``(n, n)`` factor (each reads only its own triangle), not a
          sparse-structured solve bounded by ``nnz``.

    Example:
        >>> import torch
        >>> matrix = torch.eye(3, dtype=torch.float64) * 2.0
        >>> precond = ILUPreconditioner()
        >>> precond.setup(matrix)
        >>> z = precond.apply(torch.ones(3, dtype=torch.float64))
    """

    def __init__(self) -> None:
        """Register the (not yet computed) combined LU-factor buffer."""
        nn.Module.__init__(self)
        self._operator: torch.Tensor
        self.register_buffer("_operator", None)

    def setup(
        self,
        matrix: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> Self:
        """Compute and register the dense ILU(0) factorization of ``matrix``.

        Args:
            matrix (torch.Tensor): System matrix ``A``, shape ``(n, n)``.
            context (PreconditionerContext | None): Ignored.
        """
        self._operator = self._compute_operator(matrix)
        self._mark_ready()
        return self

    def _compute_operator(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute the dense ILU(0) factorization of the system matrix.

        Args:
            matrix (torch.Tensor): Dense system matrix ``A``.

        Returns:
            torch.Tensor: Combined ``L``/``U`` factor tensor.
        """
        return dense_ilu0(matrix)

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Solve ``(LU) z = r`` via forward/backward triangular solves.

        Args:
            residual (torch.Tensor): Residual vector(s) ``r``, shape
                ``(n,)`` or ``(n, k)``.
            context (PreconditionerContext | None): Ignored (ILU doesn't
                need context).

        Returns:
            torch.Tensor: Preconditioned residual ``z = (LU)^{-1}r``.
        """
        # ``solve_triangular`` reads only the triangle it is told about, and
        # ``unitriangular=True`` ignores the diagonal, so the packed factor
        # serves as both ``L`` (unit diagonal) and ``U`` without building
        # ``tril``/``triu``/identity copies on every application.
        residual_matrix = residual if residual.ndim > 1 else residual.unsqueeze(-1)
        intermediate = torch.linalg.solve_triangular(
            self._operator, residual_matrix, upper=False, unitriangular=True
        )
        solution = torch.linalg.solve_triangular(self._operator, intermediate, upper=True)
        return solution if residual.ndim > 1 else solution.squeeze(-1)
