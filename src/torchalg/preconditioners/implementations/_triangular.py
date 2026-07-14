"""Private helpers for dense Cholesky-style triangular solves.

Shared by ``icholesky.py`` and ``ic0.py``: both preconditioners hold a lower
triangular factor ``L`` and solve ``(L @ L.T) z = r`` for ``z`` — the only
difference between them is how ``L`` is obtained (externally supplied vs.
computed via ``IC(0)``). Isolated here per the project convention of keeping
the preconditioner classes themselves declarative and free of branching
logic.
"""

from __future__ import annotations

import torch


def cholesky_factor_solve(factor: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Solve ``(factor @ factor.T) z = residual`` via ``torch.cholesky_solve``.

    ``torch.cholesky_solve`` requires a right-hand side of shape ``(*, n,
    k)``; this promotes a 1D residual to a column matrix before the solve
    and squeezes the result back, so callers can pass either a plain vector
    or a batch of columns.

    Args:
        factor (torch.Tensor): Lower triangular factor ``L``, shape
            ``(n, n)``, such that the preconditioner matrix is
            ``L @ L.T``.
        residual (torch.Tensor): Residual vector(s) ``r``, shape ``(n,)`` or
            ``(n, k)``.

    Returns:
        torch.Tensor: Preconditioned residual ``z``, same shape as
            ``residual``.
    """
    residual_matrix = residual if residual.ndim > 1 else residual.unsqueeze(-1)
    solution = torch.cholesky_solve(residual_matrix, factor, upper=False)
    return solution if residual.ndim > 1 else solution.squeeze(-1)
