"""Zero-level incomplete Cholesky (IC(0)) preconditioner.

Dense masked port of the reference's numba-JIT IC(0) kernel: computes
updates only on entries within the original sparsity pattern of the lower
triangle of ``A`` (the IC(0) property), using plain dense ``torch.Tensor``
operations instead of a sparse/JIT kernel - see ``_masked_factorization.py``;
the dense tree stays dense-only; sparse CSR lives in the separate
``torchalg.sparse`` tree.
"""

from __future__ import annotations

from typing import Self

import torch
from torch import nn

from ..base import LinearPreconditioner, Placement, PreconditionerContext
from ._masked_factorization import dense_ic0
from ._triangular import cholesky_factor_solve


class IC0Preconditioner(LinearPreconditioner[torch.Tensor], nn.Module):
    """Zero-level incomplete Cholesky preconditioner for SPD systems.

    Computes IC(0) factorization: ``L @ L.T ≈ A`` where ``L`` maintains the
    sparsity pattern of the lower triangle of ``A``: every non-zero stored
    value is pattern, there is no drop tolerance.

    Best suited for:
    - Sparse symmetric positive definite matrices.
    - Problems where structure preservation is important.
    - Systems where ILU is too expensive but Jacobi insufficient.

    Follows the ``Preconditioner`` + ``nn.Module`` pattern established by
    ``JacobiPreconditioner``: ``nn.Module.__init__()`` is called explicitly
    and the factor ``L`` is stored via ``register_buffer`` (never
    ``register_parameter`` - it is never learned or autograd-tracked) so it
    participates in ``.to(device=..., dtype=...)`` propagation.

    Mathematical background:
        Preconditioner ``M = L @ L.T``. Solve ``M z = r`` with two
        triangular solves: forward ``L y = r``, backward ``L.T z = y``.

    Complexity (derived from this class's actual dense implementation, not
    the sparse-textbook IC(0) algorithm):
        - Storage: ``O(n^2)`` — ``_masked_factorization.dense_ic0`` returns a
          full ``(n, n)`` buffer; entries outside the sparsity mask are
          zeroed, not omitted, so this is a dense masked port (see the
          module docstring), not a true sparse structure.
        - Setup: ``O(sum_k nnz(L[:, k])^2)`` plus ``O(n^2)`` for the
          pattern scan — ``dense_ic0`` is a right-looking Cholesky that, at
          step ``k``, updates only the block spanned by the stored rows of
          column ``k`` with one masked outer product; a dense pattern reaches
          the standard dense-Cholesky ``O(n^3)``. The positive-pivot check is
          a single host read after the loop.
        - Application: ``O(n^2)`` — two dense ``solve_triangular`` calls
          (triangular solves) against the full ``(n, n)`` factor, not a
          sparse-structured solve bounded by ``nnz(L)``.

    Breakdown:
        IC(0) is not guaranteed to exist for every SPD matrix (it can
        require a square root of a non-positive diagonal value).
        ``setup()`` raises ``ValueError`` naming the first non-positive
        pivot, instead of letting ``nan``/``inf`` propagate silently through
        ``apply()`` - a caller that wants to treat breakdown as a soft
        failure (e.g. mark one preconditioner in a sweep as broken rather
        than aborting) should catch ``ValueError`` around ``setup()``.

    Attributes:
        _operator (torch.Tensor): Lower triangular factor ``L``.

    Example:
        >>> import torch
        >>> matrix = torch.eye(3, dtype=torch.float64) * 2.0
        >>> precond = IC0Preconditioner()
        >>> precond.setup(matrix)
        >>> z = precond.apply(torch.ones(3, dtype=torch.float64))
    """

    placement = Placement.SEQUENTIAL_SWEEP

    def __init__(self) -> None:
        """Configure IC(0) preconditioner; the factorization is built by ``setup()``."""
        nn.Module.__init__(self)
        self._operator: torch.Tensor
        self.register_buffer("_operator", None)

    def setup(
        self,
        matrix: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> Self:
        """Compute and register the dense IC(0) factorization of ``matrix``.

        Args:
            matrix (torch.Tensor): Symmetric positive-definite system
                matrix ``A``, shape ``(n, n)``.
            context (PreconditionerContext | None): Ignored.
        """
        self._operator = self._compute_operator(matrix)
        return self

    def _compute_operator(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute the dense IC(0) factorization of the system matrix.

        Args:
            matrix (torch.Tensor): Symmetric positive-definite dense system
                matrix ``A``.

        Returns:
            torch.Tensor: Lower triangular incomplete Cholesky factor ``L``.
        """
        return dense_ic0(matrix)

    def __str__(self) -> str:
        """Human-readable structural summary.

        Returns:
            str: ``"IC0()"``.
        """
        return "IC0()"

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Solve ``(L @ L.T) z = r`` by two triangular solves.

        Args:
            residual (torch.Tensor): Residual vector(s) ``r``, shape
                ``(n,)`` or ``(n, k)``.
            context (PreconditionerContext | None): Ignored (IC(0) doesn't
                need context).

        Returns:
            torch.Tensor: Preconditioned residual ``z = M^{-1}r``.
        """
        return cholesky_factor_solve(self._operator, residual)
