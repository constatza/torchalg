"""Incomplete LU (ILU(0)) preconditioner.

Dense masked port of the reference's `scipy.sparse.linalg.spilu`-backed
factorization: a natural-ordering, no-pivoting ILU(0) (Saad Algorithm 10.4)
computed on a dense ``torch.Tensor`` clone of ``A``, restricted to ``A``'s
original non-zero pattern - see ``_masked_factorization.py`` and
``docs/plan.md``'s dense-only directive.
"""

from __future__ import annotations

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
        - ``O(nnz)`` storage, ``O(nnz)`` application cost.
        - Not guaranteed SPD even if ``A`` is SPD.

    Example:
        >>> import torch
        >>> matrix = torch.eye(3, dtype=torch.float64) * 2.0
        >>> precond = ILUPreconditioner(matrix)
        >>> z = precond.apply(torch.ones(3, dtype=torch.float64))
    """

    def __init__(self, matrix: torch.Tensor) -> None:
        """Initialize from system matrix, registering the combined LU factor as a buffer.

        Args:
            matrix (torch.Tensor): System matrix ``A``, shape ``(n, n)``.
        """
        nn.Module.__init__(self)
        self._operator: torch.Tensor
        self.register_buffer("_operator", self._compute_operator(matrix))

    def _compute_operator(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute the dense ILU(0) factorization of the system matrix.

        Args:
            matrix (torch.Tensor): System matrix ``A``.

        Returns:
            torch.Tensor: Combined ``L``/``U`` factor tensor (strictly-lower
                part is ``L`` with an implicit unit diagonal; upper part
                including the diagonal is ``U``).
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
        n = self._operator.shape[0]
        unit_diagonal = torch.eye(n, dtype=self._operator.dtype, device=self._operator.device)
        lower = torch.tril(self._operator, diagonal=-1) + unit_diagonal
        upper = torch.triu(self._operator)

        residual_matrix = residual if residual.ndim > 1 else residual.unsqueeze(-1)
        intermediate = torch.linalg.solve_triangular(
            lower, residual_matrix, upper=False, unitriangular=True
        )
        solution = torch.linalg.solve_triangular(upper, intermediate, upper=True)
        return solution if residual.ndim > 1 else solution.squeeze(-1)
