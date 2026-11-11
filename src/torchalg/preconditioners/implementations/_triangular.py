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
    """Solve ``(factor @ factor.T) z = residual`` by forward then backward substitution.

    Two ``torch.linalg.solve_triangular`` calls instead of ``torch.cholesky_solve``:
    both perform the same substitutions, but LAPACK ``potrs`` behind
    ``cholesky_solve`` measured 4-9x slower on CPU for one right-hand side
    (19 ms vs 4 ms at n = 2744). The backward solve reads ``factor.mT`` as an
    upper-triangular view, so no transpose is materialized. A 1D residual is
    promoted to a column for the solve and squeezed back.

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
    intermediate = torch.linalg.solve_triangular(factor, residual_matrix, upper=False)
    solution = torch.linalg.solve_triangular(factor.mT, intermediate, upper=True)
    return solution if residual.ndim > 1 else solution.squeeze(-1)
