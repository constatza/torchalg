"""Jacobi (diagonal scaling) preconditioner, sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.jacobi.JacobiPreconditioner`` (dense; kept
unmodified for comparison).
Same algorithm, separate implementation, tested against the dense result: the
sparse tree never imports the dense tree (enforced by ``tach.toml``). Same name,
disambiguated by package (``torch.mm`` vs. ``torch.sparse.mm`` convention),
never imported by the dense class. The only behavioral difference from the
dense sibling is how the diagonal is extracted: ``kernels.diagonal.
sparse_diagonal`` instead of ``torch.diagonal``, which has no sparse-CSR
overload.
"""

from __future__ import annotations

from typing import Self

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

    Cost:
        Setup, ``z = nnz``, ``m`` the largest row count: ``W = n + z`` (the
        reciprocal, plus the zeros the masked scatter adds), ``I = O(z)`` for
        the row-index expansion, ``Q`` a constant number of passes over the
        ``z``-sized index and value arrays (the algorithmic minimum is one
        pass), ``D = m + O(1)`` (the scatter's same-row additions are the only
        chain; every other kernel is a map), ``L = 15`` (row expansion, masked
        scatter, guard and reciprocal) and ``S = 0`` (``sparse_diagonal``
        reads no count back).
        Apply, ``k`` columns: ``W = nk``, ``Q = 3Vnk`` (read ``r``, write
        ``z``, plus the ``n``-vector ``inv_diag``), ``D = 1``, ``L = 1``,
        ``S = 0``, with ``V = 8`` bytes per ``float64``.
        Roofline: apply ``W/Q`` is about 0.04 flop per byte (memory-bound).
        Work-depth: ``W/D = nk``, fully parallel; the preconditioner adds no
        depth to the PCG iteration.

    Example:
        >>> import torch
        >>> matrix = torch.diag(torch.tensor([2.0, 4.0, 1.0])).to_sparse_csr()
        >>> precond = JacobiPreconditioner()
        >>> precond.setup(matrix)
        >>> z = precond.apply(torch.tensor([2.0, 4.0, 1.0]))  # z = D^{-1}r
    """

    def __init__(self) -> None:
        """Register the (not yet computed) inverse-diagonal buffer."""
        nn.Module.__init__(self)
        self.inv_diag: torch.Tensor
        self.register_buffer("inv_diag", None)

    def setup(
        self,
        matrix: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> Self:
        """Compute and register the inverse diagonal of a sparse CSR ``matrix``.

        Args:
            matrix (torch.Tensor): Sparse CSR system matrix ``A``, shape
                ``(n, n)``.
            context (PreconditionerContext | None): Ignored.
        """
        self.inv_diag = self._compute_operator(matrix)
        return self

    @property
    def is_ready(self) -> bool:
        """Whether the inverse diagonal has been computed.

        Returns:
            bool: ``True`` once ``setup()`` has stored ``inv_diag``.
        """
        return self.inv_diag is not None

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
            residual (torch.Tensor): Residual vector ``(n,)`` or block ``(n, k)``.
            context (PreconditionerContext | None): Ignored (Jacobi doesn't
                need context).

        Returns:
            torch.Tensor: Preconditioned residual z = D^{-1}r, same shape as
                ``residual``. A block is scaled row-wise (one column per
                right-hand side), never broadcast over its columns.
        """
        if residual.ndim == 1:
            return self.inv_diag * residual
        return self.inv_diag.unsqueeze(-1) * residual
