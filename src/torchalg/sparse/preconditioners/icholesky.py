"""Incomplete Cholesky preconditioner using an externally supplied factor, sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.icholesky.ICholeskyPreconditioner`` (dense;
kept unmodified for comparison).
Same algorithm, separate implementation, tested against the dense result: the
sparse tree never imports the dense tree (enforced by ``tach.toml``). Same
name, disambiguated by package, never imported by the dense class. Like its
dense sibling, no factorization happens here - the caller supplies the
sparse CSR lower triangular factor ``L`` directly (e.g. from
``sparse_ic0``). Reuses ``_triangular.PreparedTriangularPair``, the exact
setup-prepared forward/backward solve state ``IC0Preconditioner`` already
uses - the plan's own framing for this class: "mostly reuse, not new numerical
code."
"""

from __future__ import annotations

from typing import Self

import torch
from torch import nn

from torchalg.preconditioners.base import LinearPreconditioner, Placement, PreconditionerContext
from torchalg.sparse.preconditioners._triangular import (
    PreparedTriangularPair,
    build_sparse_cholesky_solve_cache,
)


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
        - ``z = M^{-1}r = (L @ L.T)^{-1}r``, solved via two sparse triangular
          solves (forward against ``L``, backward against ``L.T``) prepared by
          ``kernels.triangular.prepare_triangular_solve`` (the vendor solve on
          CPU and CUDA) - the dense sibling's single ``torch.cholesky_solve``
          call has no sparse-CSR overload, which is exactly the capability
          gap ``kernels.triangular`` exists to close.

    Storage: ``O(nnz(L))`` - ``L`` is stored exactly as supplied, a true sparse
    CSR structure (no factorization, hence no fill-in beyond the caller's ``L``).

    Cost:
        Setup, ``z_L = nnz(L)``, ``w`` the index width, ``V = 8`` bytes per
        ``float64``: no factorization (``_compute_operator`` returns ``L``). The
        transpose is a COO round trip, a sort of the stored entries
        (``W = 0``, ``I = O(z_L log z_L)``, ``Q = O(z_L(V + w))``, ``D = O(log z_L)``),
        and each of the two ``prepare_triangular_solve`` calls extracts its triangle
        and tests its pivots (``Q = O(z_L(V + w))``, ``D = O(log z_L)``): about
        ``L = 60`` in all and ``S = 8`` (per direction one zero-pivot read and three
        count reads of the triangle extraction).
        Apply, ``k`` columns: two sparse triangular solves, ``W = 4 z_L k``,
        ``Q = 2 [z_L(V + w) + (n + 1)w + 2Vnk]``, ``D = lv_f + lv_b`` (each solve a
        recurrence over the dependency levels of ``L`` and ``L.T``), ``L = 2``
        through the vendor solve (CPU and CUDA; ``L = O(lv)`` per solve when
        level-scheduled) and ``S = 0`` - the same apply as ``IC0Preconditioner``,
        since both share ``_triangular.PreparedTriangularPair``.
        Roofline: apply ``W/Q`` is about 0.17 flop per byte (memory-bound).
        Work-depth: apply ``W/D = 2 z_L k / lv(L)``, low: depth limits parallel
        speed-up (``Placement.SEQUENTIAL_SWEEP``).

    Example:
        >>> import torch
        >>> from torchalg.sparse.preconditioners.ic0 import sparse_ic0
        >>> matrix = (torch.eye(3, dtype=torch.float64) * 2.0).to_sparse_csr()
        >>> L = sparse_ic0(matrix)
        >>> precond = ICholeskyPreconditioner()
        >>> precond.setup(L)
        >>> z = precond.apply(torch.ones(3, dtype=torch.float64))
    """

    placement = Placement.SEQUENTIAL_SWEEP

    def __init__(self) -> None:
        """Register the (not yet stored) lower-triangular-factor buffer."""
        nn.Module.__init__(self)
        self._operator: torch.Tensor
        self._solve_cache: PreparedTriangularPair
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
        operator = self._compute_operator(matrix)
        solve_cache = build_sparse_cholesky_solve_cache(operator)
        self._operator = operator
        self._solve_cache = solve_cache
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
        return self._solve_cache.solve(residual)
