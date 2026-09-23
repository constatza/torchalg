"""Symmetric Gauss-Seidel relaxation, dense torch port of PyAMG's ``gauss_seidel``.

A forward sweep followed by a backward sweep, each equal to a single
triangular solve: ``(D + L) x_new = b - U x_old`` (forward) and
``(D + U) x_new = b - L x_old`` (backward). Rows with a zero diagonal - and,
for the indexed variant, rows outside the selected set - are left unchanged,
exactly as PyAMG's ``amg_core::gauss_seidel`` / ``gauss_seidel_indexed`` skip
them. PyAMG's block (BSR) Gauss-Seidel sweeps the dofs of each block
point-wise in the same direction, so it is identical to the scalar sweep on
the full matrix and needs no separate block implementation.

References:
    - PyAMG 5.3.0, ``pyamg/relaxation/relaxation.py`` and
      ``pyamg/amg_core/relaxation.h`` (``gauss_seidel``,
      ``bsr_gauss_seidel``, ``gauss_seidel_indexed``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _PreparedSweep:
    """A direction's fixed triangular system, reusable across every sweep of a call.

    ``frozen``/``triangle``/``other`` depend only on ``matrix`` and
    ``active`` - never on the iterate ``x`` or ``rhs`` - so building them
    once per direction and reusing across ``iterations`` sweeps (instead of
    reconstructing these dense ``(n, n)`` tensors on every single sweep, as
    a naive per-sweep implementation would) turns ``iterations`` redundant
    O(n^2) rebuilds into one.
    """

    frozen: torch.Tensor
    triangle: torch.Tensor
    other: torch.Tensor
    upper: bool


def _prepare_sweep(matrix: torch.Tensor, *, forward: bool, active: torch.Tensor) -> _PreparedSweep:
    """Build one direction's fixed triangular system, freezing inactive/zero-diagonal rows.

    Args:
        matrix (torch.Tensor): Dense matrix ``A``, shape ``(n, n)``.
        forward (bool): Ascending (``True``) or descending row order.
        active (torch.Tensor): Boolean mask of rows allowed to change.

    Returns:
        _PreparedSweep: Reusable triangular system for this direction.
    """
    frozen = (torch.diagonal(matrix) == 0) | ~active
    triangle, other = (
        (torch.tril(matrix), torch.triu(matrix, 1))
        if forward
        else (torch.triu(matrix), torch.tril(matrix, -1))
    )
    triangle = triangle * (~frozen).to(matrix.dtype).unsqueeze(1) + torch.diag(
        frozen.to(matrix.dtype)
    )
    return _PreparedSweep(frozen=frozen, triangle=triangle, other=other, upper=not forward)


def _apply_sweep(prepared: _PreparedSweep, x: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve one already-prepared triangular system for the current iterate.

    Args:
        prepared (_PreparedSweep): Direction's fixed triangular system.
        x (torch.Tensor): Current iterate, shape ``(n,)``.
        rhs (torch.Tensor): Right-hand side, shape ``(n,)``.

    Returns:
        torch.Tensor: Iterate after the sweep.
    """
    target = torch.where(prepared.frozen, x, rhs - prepared.other @ x)
    return torch.linalg.solve_triangular(
        prepared.triangle, target.unsqueeze(1), upper=prepared.upper
    ).squeeze(1)


def symmetric_gauss_seidel(
    matrix: torch.Tensor,
    x: torch.Tensor,
    rhs: torch.Tensor,
    iterations: int,
    *,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply ``iterations`` symmetric Gauss-Seidel iterations (forward then backward).

    Args:
        matrix (torch.Tensor): Dense matrix ``A``, shape ``(n, n)``.
        x (torch.Tensor): Initial iterate, shape ``(n,)``.
        rhs (torch.Tensor): Right-hand side, shape ``(n,)``.
        iterations (int): Number of symmetric iterations.
        rows (torch.Tensor | None): If given, only these rows (ascending
            indices) are updated - PyAMG's ``gauss_seidel_indexed``.

    Returns:
        torch.Tensor: Updated iterate, shape ``(n,)``.
    """
    active = torch.ones_like(x, dtype=torch.bool)
    if rows is not None:
        active = torch.zeros_like(x, dtype=torch.bool)
        active[rows] = True
    forward_sweep = _prepare_sweep(matrix, forward=True, active=active)
    backward_sweep = _prepare_sweep(matrix, forward=False, active=active)
    for _ in range(iterations):
        x = _apply_sweep(forward_sweep, x, rhs)
        x = _apply_sweep(backward_sweep, x, rhs)
    return x
