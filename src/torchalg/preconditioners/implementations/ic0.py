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

    Storage: ``O(n^2)`` — ``_masked_factorization.dense_ic0`` returns a full
    ``(n, n)`` buffer; entries outside the sparsity mask are zeroed, not omitted,
    so this is a dense masked port (see the module docstring), not a true sparse
    structure.

    Cost:
        Setup, dense ``(n, n)`` matrix, ``V = 8`` bytes per ``float64``, ``d_k`` the
        number of pattern entries of column ``k`` below the diagonal:
        ``dense_ic0`` is a right-looking Cholesky that at pivot ``k`` scales the
        stored rows of column ``k`` and applies one masked outer product to the
        ``d_k x d_k`` block they span. ``W = sum_k (2 d_k^2 + d_k + 1)`` (outer
        product and subtraction over the whole block, of which the mask keeps the
        lower half; the column scale; the square root), reaching the dense
        Cholesky's ``2n^3 / 3`` for a dense pattern;
        ``Q`` about ``10Vn^2 + 6V sum_k d_k^2`` (the ``n^2`` pattern scan, then a
        gather, the outer product, the mask, the block gather and the block scatter
        per pivot); ``D = 8n`` approximately (the pivots are a dependent chain of
        about eight kernels each: square root, column gather, scale, scatter, outer
        product, mask, block subtract, block scatter); ``L = 14n`` approximately (one
        host-driven step per pivot, three for a pivot with no stored rows below);
        ``S = 3`` (the row counts and the ``nonzero`` of the pattern split, and the
        single positive-pivot read after the loop). Setup is launch-bound
        (``L`` and ``D`` grow with ``n``, not with the pattern).
        Apply, ``k`` columns: two dense ``solve_triangular`` calls against the full
        factor, independent of the pattern: ``W = 2n^2 k``, ``Q = V(n^2 + 4nk)``
        (each solve reads its triangle once), ``D = 2n`` (forward then backward
        substitution, each a recurrence), ``L = 2`` and ``S = 0``.
        Roofline: apply ``W/Q`` is about 0.25 flop per byte for ``k = 1``
        (memory-bound), growing with ``k``.
        Work-depth: apply ``W/D = nk``; the substitution depth keeps the solve on the
        CPU (``Placement.SEQUENTIAL_SWEEP``).

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
