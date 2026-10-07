"""Incomplete Cholesky preconditioner using an externally supplied factor, sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.icholesky.ICholeskyPreconditioner`` (dense;
kept unmodified for comparison - see ``docs/plan.md``'s "Correction: dense
and sparse must be separate implementations, not an internal branch"). Same
name, disambiguated by package, never imported by the dense class. Like its
dense sibling, no factorization happens here - the caller supplies the
sparse CSR lower triangular factor ``L`` directly (e.g. from
``sparse_ic0``). Reuses ``_triangular.sparse_cholesky_factor_solve``, the
exact level-scheduled forward/backward solve ``IC0Preconditioner.apply()``
already uses - the plan's own framing for this class: "mostly reuse, not new
numerical code."
"""

from __future__ import annotations

from typing import Self

import torch
from torch import nn

from torchalg.preconditioners.base import LinearPreconditioner, PreconditionerContext
from torchalg.sparse.preconditioners._triangular import sparse_cholesky_factor_solve


class ICholeskyPreconditioner(LinearPreconditioner[torch.Tensor], nn.Module):
    """Incomplete Cholesky preconditioner for sparse CSR systems, using a provided factor ``L``.

    Uses an externally supplied sparse CSR lower triangular matrix ``L``
    (e.g. from ``sparse_ic0``) to precondition via ``M = L @ L.T``.

    Unlike other preconditioners, this class expects the *input* it's
    constructed with to already be the factor ``L``, not the system matrix
    ``A`` - no factorization happens here.

    Follows the ``Preconditioner`` + ``nn.Module`` pattern established by the
    dense sibling: ``nn.Module.__init__()`` is called explicitly and ``L``
    is stored via ``register_buffer`` (never ``register_parameter`` - it is
    never learned or autograd-tracked) so it participates in
    ``.to(device=..., dtype=...)`` propagation.

    Mathematical Properties:
        - ``M = L @ L.T``.
        - ``z = M^{-1}r = (L @ L.T)^{-1}r``, solved via two level-scheduled
          sparse triangular solves (forward against ``L``, backward against
          ``L.T``) - the dense sibling's single ``torch.cholesky_solve``
          call has no sparse-CSR overload, which is exactly the capability
          gap ``kernels.triangular`` exists to close.

    Complexity:
        - Storage: ``O(nnz(L))`` - ``L`` is stored exactly as supplied, a
          true sparse CSR structure (no factorization, hence no additional
          fill-in beyond whatever the caller's ``L`` already has).
        - Setup: none - ``_compute_operator`` returns the supplied ``L``
          unchanged.
        - Application: two ``O(nnz(L))`` level-scheduled triangular solves,
          same as ``IC0Preconditioner``'s application cost (both share
          ``_triangular.sparse_cholesky_factor_solve``).

    Example:
        >>> import torch
        >>> from torchalg.sparse.preconditioners.ic0 import sparse_ic0
        >>> matrix = (torch.eye(3, dtype=torch.float64) * 2.0).to_sparse_csr()
        >>> L = sparse_ic0(matrix)
        >>> precond = ICholeskyPreconditioner()
        >>> precond.setup(L)
        >>> z = precond.apply(torch.ones(3, dtype=torch.float64))
    """

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
        """Store the sparse CSR lower triangular factor ``L`` (``matrix``) as a buffer.

        Args:
            matrix (torch.Tensor): Sparse CSR lower triangular factor ``L``,
                shape ``(n, n)``.
            context (PreconditionerContext | None): Ignored.
        """
        self._operator = self._compute_operator(matrix)
        self._mark_ready()
        return self

    def _compute_operator(self, matrix: torch.Tensor) -> torch.Tensor:
        """Return the supplied factor ``L`` unchanged.

        Args:
            matrix (torch.Tensor): Sparse CSR lower triangular factor ``L``.

        Returns:
            torch.Tensor: ``matrix`` itself - no factorization to compute.
        """
        return matrix

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Solve ``(L @ L.T) z = r`` via forward/backward sparse triangular solves.

        Args:
            residual (torch.Tensor): Residual vector(s) ``r``, shape
                ``(n,)`` or ``(n, k)``.
            context (PreconditionerContext | None): Ignored (ICholesky
                doesn't need context).

        Returns:
            torch.Tensor: Preconditioned residual ``z = (L @ L.T)^{-1}r``.
        """
        return sparse_cholesky_factor_solve(self._operator, residual)
