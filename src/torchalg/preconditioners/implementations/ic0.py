"""Zero-level incomplete Cholesky (IC(0)) preconditioner.

Dense masked port of the reference's numba-JIT IC(0) kernel: computes
updates only on entries within the original sparsity pattern of the lower
triangle of ``A`` (the IC(0) property), using plain dense ``torch.Tensor``
operations instead of a sparse/JIT kernel - see ``_masked_factorization.py``
and ``docs/plan.md``'s dense-only directive.
"""

from __future__ import annotations

from typing import Self

import torch
from torch import nn

from ..base import LinearPreconditioner, PreconditionerContext
from ._masked_factorization import dense_ic0
from ._triangular import cholesky_factor_solve

_DEFAULT_THRESHOLD = 0.0
"""Default drop tolerance: entries with |value| <= threshold are treated as
zero and excluded from the sparsity pattern.

0.0 only drops exact zeros. A nonzero absolute constant is unsafe as a
library default: callers may factorize the same matrix at different scales
(e.g. before/after a data-dependent normalization), and a fixed absolute
threshold's effective strictness shifts with that scale - it can silently
drop entries that are structurally real at one scale but fall under the
threshold at another, corrupting the IC(0) sparsity pattern in a way that's
invisible until elimination breaks down many steps later. Callers that want
drop-tolerance behavior should pass an explicit threshold sized relative to
their own matrix's scale."""


class IC0Preconditioner(LinearPreconditioner[torch.Tensor], nn.Module):
    """Zero-level incomplete Cholesky preconditioner for SPD systems.

    Computes IC(0) factorization: ``L @ L.T ≈ A`` where ``L`` maintains the
    sparsity pattern of the lower triangle of ``A``.

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
        Preconditioner ``M = L @ L.T``. Solve ``M z = r`` via
        ``torch.cholesky_solve`` (equivalent to the two triangular solves:
        forward ``L y = r``, backward ``L.T z = y``).

    Complexity (derived from this class's actual dense implementation, not
    the sparse-textbook IC(0) algorithm):
        - Storage: ``O(n^2)`` — ``_masked_factorization.dense_ic0`` returns a
          full ``(n, n)`` buffer; entries outside the sparsity mask are
          zeroed, not omitted, so this is a dense masked port (see the
          module docstring), not a true sparse structure.
        - Setup: ``O(n^3)`` — ``dense_ic0`` is a vectorized right-looking
          Cholesky: at step ``k`` it updates the whole trailing
          ``(n-k, n-k)`` block with one masked outer product. Summed over
          ``k = 0..n-1`` this is the standard dense-Cholesky FLOP count,
          independent of how sparse the mask actually is (the mask decides
          *which* entries of the trailing block survive, not how much of the
          block gets touched).
        - Application: ``O(n^2)`` — ``torch.cholesky_solve`` runs two dense
          triangular solves against the full ``(n, n)`` factor, not a
          sparse-structured solve bounded by ``nnz(L)``.

    Breakdown:
        IC(0) is not guaranteed to exist for every SPD matrix (it can
        require a square root of a non-positive diagonal value).
        Construction raises ``ValueError`` as soon as elimination hits a
        non-positive pivot, instead of letting ``nan``/``inf`` propagate
        silently through ``apply()`` - a caller that wants to treat
        breakdown as a soft failure (e.g. mark one preconditioner in a
        sweep as broken rather than aborting) should catch ``ValueError``
        around construction.

    Attributes:
        _operator (torch.Tensor): Lower triangular factor ``L``.

    Example:
        >>> import torch
        >>> matrix = torch.eye(3, dtype=torch.float64) * 2.0
        >>> precond = IC0Preconditioner()
        >>> precond.setup(matrix)
        >>> z = precond.apply(torch.ones(3, dtype=torch.float64))
    """

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD) -> None:
        """Configure IC(0) preconditioner; the factorization is built by ``setup()``.

        Args:
            threshold (float): Drop tolerance - entries with ``|value| <=
                threshold`` are treated as zero. Default: ``0.0`` (only
                exact zeros are dropped; see ``_DEFAULT_THRESHOLD``).
        """
        nn.Module.__init__(self)
        self._threshold = threshold
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
        self._mark_ready()
        return self

    def _compute_operator(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute the dense IC(0) factorization of the system matrix.

        Args:
            matrix (torch.Tensor): Symmetric positive-definite dense system
                matrix ``A``.

        Returns:
            torch.Tensor: Lower triangular incomplete Cholesky factor ``L``.
        """
        return dense_ic0(matrix, self._threshold)

    @property
    def threshold(self) -> float:
        """Drop tolerance used to build the sparsity pattern.

        Returns:
            float: The value passed at construction.
        """
        return self._threshold

    def __str__(self) -> str:
        """Human-readable structural summary.

        Returns:
            str: e.g. ``"IC0(threshold=0e+00)"``.
        """
        return f"IC0(threshold={self._threshold:.0e})"

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Solve ``(L @ L.T) z = r`` via ``torch.cholesky_solve``.

        Args:
            residual (torch.Tensor): Residual vector(s) ``r``, shape
                ``(n,)`` or ``(n, k)``.
            context (PreconditionerContext | None): Ignored (IC(0) doesn't
                need context).

        Returns:
            torch.Tensor: Preconditioned residual ``z = M^{-1}r``.
        """
        return cholesky_factor_solve(self._operator, residual)
