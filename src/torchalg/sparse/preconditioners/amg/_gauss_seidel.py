"""Sparse-CSR symmetric Gauss-Seidel sweep, consumed by the sparse ``GaussSeidelSmoother``.

No factorization step: each sweep is a direct triangular solve against
``A``'s own entries, split by position relative to the diagonal - see
``docs/plan.md``'s "AMG/POD smoother constraint" section. Mirrors the dense
``_relaxation.py``'s D/L/U convention: forward sweep solves ``(D + L)
x_new = rhs - U @ x_old``, backward solves ``(D + U) x_new = rhs - L @
x_old``, one iteration = one forward sweep followed by one backward sweep.
"""

from __future__ import annotations

from typing import Literal

import torch

from torchalg.sparse.kernels.triangular import LevelSchedule, level_schedule, triangular_solve


def _strict_side_matvec(
    matrix: torch.Tensor, x: torch.Tensor, *, side: Literal["upper", "lower"]
) -> torch.Tensor:
    """Multiply ``x`` by the strict upper or lower part of a sparse CSR matrix.

    Uses the same crow-expansion + boolean mask + functional ``scatter_add``
    pattern as ``triangular.triangular_solve`` - no intermediate
    ``torch.sparse_csr_tensor`` is built.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix, shape ``(n, n)``.
        x (torch.Tensor): Vector to multiply, shape ``(n,)`` or ``(n, k)``.
        side (Literal["upper", "lower"]): Strict upper (``col > row``) or
            strict lower (``col < row``) part.

    Returns:
        torch.Tensor: Result of the restricted matvec, same shape as ``x``.
    """
    n = matrix.shape[0]
    crow = matrix.crow_indices()
    col = matrix.col_indices()
    values = matrix.values()
    row_nnz = crow[1:] - crow[:-1]
    row_index = torch.repeat_interleave(torch.arange(n, device=matrix.device), row_nnz)

    mask = col > row_index if side == "upper" else col < row_index
    selected_row = row_index[mask]
    selected_col = col[mask]
    selected_val = values[mask]

    if x.ndim == 1:
        return torch.zeros(n, dtype=x.dtype, device=x.device).scatter_add(
            0, selected_row, selected_val * x[selected_col]
        )
    k = x.shape[1]
    return torch.zeros(n, k, dtype=x.dtype, device=x.device).scatter_add(
        0, selected_row.unsqueeze(-1).expand(-1, k), selected_val.unsqueeze(-1) * x[selected_col]
    )


def sparse_symmetric_gauss_seidel(
    matrix: torch.Tensor,
    x: torch.Tensor,
    rhs: torch.Tensor,
    iterations: int,
    *,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply ``iterations`` symmetric Gauss-Seidel iterations on a sparse CSR matrix.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix ``A``, shape ``(n, n)``.
        x (torch.Tensor): Initial iterate, shape ``(n,)`` or ``(n, k)``.
        rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
        iterations (int): Number of symmetric iterations.
        rows (torch.Tensor | None): If given, only these rows (indices) are
            updated - sparse-CSR sibling of the dense
            ``_relaxation.symmetric_gauss_seidel``'s ``rows=`` parameter
            (PyAMG's ``gauss_seidel_indexed``). Rows outside ``rows`` stay
            frozen at their current value for the whole call.

    Returns:
        torch.Tensor: Updated iterate, same shape as ``x``.
    """
    forward_schedule: LevelSchedule = level_schedule(matrix, direction="forward")
    backward_schedule: LevelSchedule = level_schedule(matrix, direction="backward")

    active = None
    if rows is not None:
        active = torch.zeros(matrix.shape[0], dtype=torch.bool, device=matrix.device)
        active[rows] = True

    for _ in range(iterations):
        target = rhs - _strict_side_matvec(matrix, x, side="upper")
        x = triangular_solve(
            matrix, forward_schedule, target, direction="forward", active=active, current=x
        )
        target = rhs - _strict_side_matvec(matrix, x, side="lower")
        x = triangular_solve(
            matrix, backward_schedule, target, direction="backward", active=active, current=x
        )

    return x
