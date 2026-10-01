"""Jacobi (diagonal scaling) preconditioner, sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.jacobi.JacobiPreconditioner`` (dense; kept
unmodified for comparison - see ``docs/plan.md``'s "Correction: dense and
sparse must be separate implementations, not an internal branch"). Same name,
disambiguated by package (``torch.mm`` vs. ``torch.sparse.mm`` convention),
never imported by the dense class. The only behavioral difference from the
dense sibling is how the diagonal is extracted: ``kernels.diagonal.
sparse_diagonal`` instead of ``torch.diagonal``, which has no sparse-CSR
overload.
"""

from __future__ import annotations

import torch
from torch import nn

from torchalg.preconditioners.base import LinearPreconditioner, PreconditionerContext
from torchalg.sparse.kernels.diagonal import sparse_diagonal

_NEAR_ZERO_DIAGONAL_TOL = 1e-14
"""Matches the dense sibling's tolerance - see its docstring."""


class JacobiPreconditioner(LinearPreconditioner[torch.Tensor], nn.Module):
    """Jacobi (diagonal scaling) preconditioner for sparse CSR systems: z = D^{-1}r.

    Follows the ``Preconditioner`` + ``nn.Module`` pattern established by the
    dense sibling: ``nn.Module.__init__()`` is called explicitly and
    ``inv_diag`` is stored via ``register_buffer`` (never
    ``register_parameter`` - it is never learned or autograd-tracked) so it
    participates in ``.to(device=..., dtype=...)`` propagation.

    Mathematical Properties:
        - M = diag(A) (diagonal of system matrix).
        - z = D^{-1}r where D = diag(A).
        - O(n) storage, O(nnz) extraction cost, O(n) application cost.
        - SPD if A is SPD.

    Example:
        >>> import torch
        >>> matrix = torch.diag(torch.tensor([2.0, 4.0, 1.0])).to_sparse_csr()
        >>> precond = JacobiPreconditioner(matrix)
        >>> z = precond.apply(torch.tensor([2.0, 4.0, 1.0]))  # z = D^{-1}r
    """

    def __init__(self, matrix: torch.Tensor) -> None:
        """Initialize from a sparse CSR system matrix, registering the inverse diagonal as a buffer.

        Args:
            matrix (torch.Tensor): Sparse CSR system matrix ``A``, shape
                ``(n, n)``.
        """
        nn.Module.__init__(self)
        self.inv_diag: torch.Tensor
        self.register_buffer("inv_diag", self._compute_operator(matrix))

    def _compute_operator(self, matrix: torch.Tensor) -> torch.Tensor:
        """Extract and invert the diagonal of a sparse CSR matrix.

        Args:
            matrix (torch.Tensor): Sparse CSR system matrix ``A``, shape
                ``(n, n)``.

        Returns:
            torch.Tensor: Dense diagonal inverse for fast elementwise
                multiplication, shape ``(n,)``.

        Raises:
            ValueError: If ``matrix`` is not sparse CSR (raised by
                ``sparse_diagonal``).
        """
        diag = sparse_diagonal(matrix)
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
