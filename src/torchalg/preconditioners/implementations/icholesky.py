"""Incomplete Cholesky preconditioner using an externally supplied factor."""

from __future__ import annotations

from typing import Self

import torch
from torch import nn

from ..base import LinearPreconditioner, Placement, PreconditionerContext
from ._triangular import cholesky_factor_solve


class ICholeskyPreconditioner(LinearPreconditioner[torch.Tensor], nn.Module):
    """Incomplete Cholesky preconditioner using a provided factor ``L``.

    Uses an externally supplied lower triangular matrix ``L`` (e.g. an
    approximate Cholesky factor computed elsewhere) to precondition via
    ``M = L @ L.T``.

    Unlike other preconditioners, this class expects the *input* it's
    constructed with to already be the factor ``L``, not the system matrix
    ``A`` - no factorization happens here.

    Follows the ``Preconditioner`` + ``nn.Module`` pattern established by
    ``JacobiPreconditioner``: ``nn.Module.__init__()`` is called explicitly
    and ``L`` is stored via ``register_buffer`` (never ``register_parameter``
    - it is never learned or autograd-tracked) so it participates in
    ``.to(device=..., dtype=...)`` propagation.

    Mathematical Properties:
        - ``M = L @ L.T``.
        - ``z = M^{-1}r = (L @ L.T)^{-1}r``, solved by
          forward then backward ``torch.linalg.solve_triangular`` (replaces
          the reference's ``scipy.linalg.cho_solve`` - no scipy involved).

    Storage: ``O(n^2)`` — ``L`` is stored exactly as supplied, a dense ``(n, n)``
    buffer (no factorization, hence no sparsity to exploit even if the caller's
    ``L`` happens to be sparse in practice).

    Cost:
        Setup: ``_compute_operator`` returns the supplied ``L`` unchanged, so
        ``W = 0``, ``Q = 0``, ``D = 0``, ``L = 0`` and ``S = 0`` (the factor's own
        cost is the caller's).
        Apply, ``k`` columns, ``V = 8`` bytes per ``float64``: two dense
        ``solve_triangular`` calls against the full factor, shared with
        ``IC0Preconditioner`` through ``_triangular.cholesky_factor_solve``:
        ``W = 2n^2 k``, ``Q = V(n^2 + 4nk)``, ``D = 2n`` (forward then backward
        substitution), ``L = 2`` and ``S = 0``.
        Roofline: apply ``W/Q`` is about 0.25 flop per byte for ``k = 1``
        (memory-bound).
        Work-depth: apply ``W/D = nk``; the substitution depth keeps the solve on the
        CPU (``Placement.SEQUENTIAL_SWEEP``).

    Example:
        >>> import torch
        >>> L = torch.linalg.cholesky(torch.eye(3, dtype=torch.float64) * 2.0)
        >>> precond = ICholeskyPreconditioner()
        >>> precond.setup(L)
        >>> z = precond.apply(torch.ones(3, dtype=torch.float64))
    """

    placement = Placement.SEQUENTIAL_SWEEP

    def __init__(self) -> None:
        """Register the (not yet stored) lower-triangular-factor buffer."""
        nn.Module.__init__(self)
        self._operator: torch.Tensor
        self.register_buffer("_operator", None)

    def setup(
        self,
        matrix: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> Self:
        """Store the lower triangular factor ``L`` (``matrix``) as a buffer.

        Args:
            matrix (torch.Tensor): Lower triangular factor ``L``, shape
                ``(n, n)``.
            context (PreconditionerContext | None): Ignored.
        """
        self._operator = self._compute_operator(matrix)
        return self

    def _compute_operator(self, matrix: torch.Tensor) -> torch.Tensor:
        """Return the supplied factor ``L`` unchanged.

        Args:
            matrix (torch.Tensor): Lower triangular factor ``L``.

        Returns:
            torch.Tensor: ``matrix`` itself - no factorization to compute.
        """
        return matrix

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Solve ``(L @ L.T) z = r`` by two triangular solves.

        Args:
            residual (torch.Tensor): Residual vector(s) ``r``, shape
                ``(n,)`` or ``(n, k)``.
            context (PreconditionerContext | None): Ignored (ICholesky
                doesn't need context).

        Returns:
            torch.Tensor: Preconditioned residual ``z = (L @ L.T)^{-1}r``.
        """
        return cholesky_factor_solve(self._operator, residual)
