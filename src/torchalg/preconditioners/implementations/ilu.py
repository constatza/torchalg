"""Incomplete LU (ILU(0)) preconditioner.

Dense masked port of the reference's `scipy.sparse.linalg.spilu`-backed
factorization: a natural-ordering, no-pivoting ILU(0) (Saad Algorithm 10.4)
computed on a dense ``torch.Tensor`` clone of ``A``, restricted to ``A``'s
original non-zero pattern - see ``_masked_factorization.py``;
the dense tree stays dense-only; sparse CSR lives in the separate
``torchalg.sparse`` tree.

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

from ..base import LinearPreconditioner, Placement, PreconditionerContext
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

    Storage: ``O(n^2)`` — the combined ``L``/``U`` factor is a dense ``(n, n)``
    buffer, not a sparse structure sized by ``nnz(A)``.

    Cost:
        Setup, dense ``(n, n)`` matrix, ``V = 8`` bytes per ``float64``, ``r_k`` and
        ``c_k`` the stored entries of column ``k`` below and of row ``k`` right of the
        diagonal: ``_masked_factorization.dense_ilu0`` is a right-looking elimination
        with one masked rank-1 update of the ``r_k x c_k`` block per pivot.
        ``W = sum_k (r_k + 2 r_k c_k)`` (column scale, outer product, subtraction),
        reaching the dense LU's ``2n^3 / 3`` for a dense pattern;
        ``Q`` about ``8Vn^2 + 6V sum_k r_k c_k`` (the ``n^2`` clone and pattern scans,
        then the gathers, outer product, mask and scatter per pivot);
        ``D = 8n`` approximately (the ``n - 1`` pivots are a dependent chain of about
        eight kernels each); ``L = 12n`` approximately (one host-driven step per pivot);
        ``S = 4`` (the row counts and the ``nonzero`` of each of the two pattern
        splits; there is no pivot check). Setup is launch-bound.
        Apply, ``k`` columns: two dense ``solve_triangular`` calls on the packed
        factor (each reads only its own triangle, the first with a unit diagonal),
        independent of the pattern: ``W = 2n^2 k``, ``Q = V(n^2 + 4nk)``,
        ``D = 2n`` (forward then backward substitution), ``L = 2`` and ``S = 0``.
        Roofline: apply ``W/Q`` is about 0.25 flop per byte for ``k = 1``
        (memory-bound).
        Work-depth: apply ``W/D = nk``; the substitution depth keeps the solve on the
        CPU (``Placement.SEQUENTIAL_SWEEP``).

    Example:
        >>> import torch
        >>> matrix = torch.eye(3, dtype=torch.float64) * 2.0
        >>> precond = ILUPreconditioner()
        >>> precond.setup(matrix)
        >>> z = precond.apply(torch.ones(3, dtype=torch.float64))
    """

    placement = Placement.SEQUENTIAL_SWEEP

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
