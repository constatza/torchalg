"""Jacobi (diagonal scaling) preconditioner."""

from __future__ import annotations

import torch
from torch import nn

from ..base import LinearPreconditioner, PreconditionerContext

_NEAR_ZERO_DIAGONAL_TOL = 1e-14
"""Diagonal entries with magnitude below this are treated as 1.0 (protects
against division by near-zero)."""


class JacobiPreconditioner(LinearPreconditioner[torch.Tensor], nn.Module):
    """Jacobi (diagonal scaling) preconditioner: z = D^{-1}r.

    Diagonal preconditioning using the inverse of the matrix diagonal. Fast
    to apply, effective for diagonally dominant matrices.

    This is the first buffer-holding preconditioner in the port, and
    establishes the pattern every later stage's stateful preconditioner
    (ILU, IC0, ICholesky, AMG, POD) reuses: multiply-inherit
    ``Preconditioner`` (via ``LinearPreconditioner``) and ``nn.Module``,
    call ``nn.Module.__init__()`` explicitly, then store the precomputed
    tensor via ``register_buffer`` rather than a plain attribute. A plain
    attribute would not move under ``.to(device=...)``/``.to(dtype=...)`` -
    ``register_buffer`` is what gets that propagation for free from
    ``nn.Module``'s already-trusted implementation, with
    ``register_parameter`` deliberately avoided since ``inv_diag`` is never
    learned or autograd-tracked.

    Because ``__init__`` builds and registers ``inv_diag`` directly, it does
    not call ``LinearPreconditioner.__init__`` (which would instead store
    the result as a plain ``self._operator`` attribute, invisible to
    ``nn.Module``'s buffer machinery); ``_compute_operator`` is still
    implemented to satisfy ``LinearPreconditioner``'s interface contract.

    Mathematical Properties:
        - M = diag(A) (diagonal of system matrix).
        - z = D^{-1}r where D = diag(A).
        - O(n) storage, O(n) application cost.
        - SPD if A is SPD.

    Example:
        >>> import torch
        >>> matrix = torch.diag(torch.tensor([2.0, 4.0, 1.0]))
        >>> precond = JacobiPreconditioner(matrix)
        >>> z = precond.apply(torch.tensor([2.0, 4.0, 1.0]))  # z = D^{-1}r
    """

    def __init__(self, matrix: torch.Tensor) -> None:
        """Initialize from system matrix, registering the inverse diagonal as a buffer.

        Args:
            matrix (torch.Tensor): System matrix A, shape ``(n, n)``.
        """
        nn.Module.__init__(self)
        self.inv_diag: torch.Tensor
        self.register_buffer("inv_diag", self._compute_operator(matrix))

    def _compute_operator(self, matrix: torch.Tensor) -> torch.Tensor:
        """Extract and invert the diagonal from matrix.

        Args:
            matrix (torch.Tensor): System matrix A, shape ``(n, n)``.

        Returns:
            torch.Tensor: Diagonal inverse for fast elementwise
                multiplication, shape ``(n,)``.
        """
        diag = torch.diagonal(matrix)
        safe_diag = torch.where(
            diag.abs() < _NEAR_ZERO_DIAGONAL_TOL,
            torch.ones_like(diag),
            diag,
        )
        return 1.0 / safe_diag

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Apply diagonal scaling.

        Args:
            residual (torch.Tensor): Residual vector r.
            context (PreconditionerContext | None): Ignored (Jacobi doesn't
                need context).

        Returns:
            torch.Tensor: Preconditioned residual z = D^{-1}r.
        """
        return self.inv_diag * residual
