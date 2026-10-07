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
        - Setup: at least ``O(n^2)``, up to ``O(n^3)`` for a dense pattern —
          ``_masked_factorization.dense_ilu0`` is a Python-level triple loop
          (``i in range(1, n)``, ``k in range(i)``, ``j in range(k+1, n)``)
          that always visits the full ``(i, k)`` grid to check the sparsity
          mask (an ``O(n^2)`` floor regardless of how few entries are
          actually nonzero), and does real elimination work in the inner
          ``j`` loop for every mask hit, reaching the classical dense-LU
          ``O(n^3)`` when the pattern is dense.
        - Application: ``O(n^2)`` — ``apply()`` runs two dense
          ``torch.linalg.solve_triangular`` calls against the full
          ``(n, n)`` ``L``/``U`` factors, not a sparse-structured solve
          bounded by ``nnz``.

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
