"""Zero-level incomplete Cholesky (IC(0)) preconditioner.

Dense masked port of the reference's numba-JIT IC(0) kernel: computes
updates only on entries within the original sparsity pattern of the lower
triangle of ``A`` (the IC(0) property), using plain dense ``torch.Tensor``
operations instead of a sparse/JIT kernel - see ``_masked_factorization.py``
and ``docs/plan.md``'s dense-only directive.
"""

from __future__ import annotations

import torch
from torch import nn

from ..base import LinearPreconditioner, PreconditionerContext
from ._masked_factorization import dense_ic0
from ._triangular import cholesky_factor_solve

_DEFAULT_THRESHOLD = 1e-14
"""Default drop tolerance: entries with |value| <= threshold are treated as
zero and excluded from the sparsity pattern."""


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
        >>> precond = IC0Preconditioner(matrix)
        >>> z = precond.apply(torch.ones(3, dtype=torch.float64))
    """

    def __init__(self, matrix: torch.Tensor, threshold: float = _DEFAULT_THRESHOLD) -> None:
        """Initialize IC(0) preconditioner.

        Args:
            matrix (torch.Tensor): Symmetric positive-definite system
                matrix ``A``, shape ``(n, n)``.
            threshold (float): Drop tolerance - entries with ``|value| <=
                threshold`` are treated as zero. Improves numerical
                stability and maintains sparsity. Default: ``1e-14``.
        """
        nn.Module.__init__(self)
        self._threshold = threshold
        self._operator: torch.Tensor
        self.register_buffer("_operator", self._compute_operator(matrix))

    def _compute_operator(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute the IC(0) factorization of the system matrix.

        Args:
            matrix (torch.Tensor): Symmetric positive-definite system
                matrix ``A``.

        Returns:
            torch.Tensor: Lower triangular incomplete Cholesky factor ``L``.
        """
        return dense_ic0(matrix, self._threshold)

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
