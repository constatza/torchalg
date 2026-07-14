"""Incomplete Cholesky preconditioner using an externally supplied factor."""

from __future__ import annotations

import torch
from torch import nn

from ..base import LinearPreconditioner, PreconditionerContext
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
        - ``z = M^{-1}r = (L @ L.T)^{-1}r``, solved via
          ``torch.cholesky_solve`` (replaces the reference's
          ``scipy.linalg.cho_solve`` outright - no scipy involved).

    Example:
        >>> import torch
        >>> L = torch.linalg.cholesky(torch.eye(3, dtype=torch.float64) * 2.0)
        >>> precond = ICholeskyPreconditioner(L)
        >>> z = precond.apply(torch.ones(3, dtype=torch.float64))
    """

    def __init__(self, matrix: torch.Tensor) -> None:
        """Store the lower triangular factor ``L`` as a buffer.

        Args:
            matrix (torch.Tensor): Lower triangular factor ``L``, shape
                ``(n, n)``.
        """
        nn.Module.__init__(self)
        self._operator: torch.Tensor
        self.register_buffer("_operator", self._compute_operator(matrix))

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
        """Solve ``(L @ L.T) z = r`` via ``torch.cholesky_solve``.

        Args:
            residual (torch.Tensor): Residual vector(s) ``r``, shape
                ``(n,)`` or ``(n, k)``.
            context (PreconditionerContext | None): Ignored (ICholesky
                doesn't need context).

        Returns:
            torch.Tensor: Preconditioned residual ``z = (L @ L.T)^{-1}r``.
        """
        return cholesky_factor_solve(self._operator, residual)
