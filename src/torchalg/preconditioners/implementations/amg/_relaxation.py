"""Prepared symmetric Gauss-Seidel relaxation, dense torch port of PyAMG's ``gauss_seidel``.

A forward sweep followed by a backward sweep, each equal to a single
triangular solve: ``(D + L) x_new = b - U x_old`` (forward) and
``(D + U) x_new = b - L x_old`` (backward). Rows with a zero diagonal - and,
for the indexed variant, rows outside the selected set - are left unchanged,
exactly as PyAMG's ``amg_core::gauss_seidel`` / ``gauss_seidel_indexed`` skip
them. PyAMG's block (BSR) Gauss-Seidel sweeps the dofs of each block
point-wise in the same direction, so it is identical to the scalar sweep on
the full matrix and needs no separate block implementation.

``prepare_symmetric_gauss_seidel`` does the matrix-dependent work once (the forward and
backward triangular systems); the returned ``PreparedSymmetricGaussSeidel`` then only
sweeps, so setup stages that relax the same level matrix many times (compatible
relaxation, adaptive candidate relaxation) and the solve-time smoother share one
preparation. Triangular solves read the relevant triangle of ``A`` in place; a
triangle is copied only when rows are frozen (zero diagonal, or outside ``rows``),
and the freezing mask is skipped entirely when nothing is frozen.

References:
    - PyAMG 5.3.0, ``pyamg/relaxation/relaxation.py`` and
      ``pyamg/amg_core/relaxation.h`` (``gauss_seidel``,
      ``bsr_gauss_seidel``, ``gauss_seidel_indexed``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class _PreparedSweep:
    """A direction's fixed triangular system, reusable across every sweep.

    ``triangle``, ``other`` and ``frozen`` depend only on the matrix and the
    updatable row set, never on the iterate or the right-hand side, so they are
    built once per direction.

    Attributes:
        triangle (torch.Tensor): System matrix of the triangular solve. The matrix
            itself when no row is frozen (``solve_triangular`` reads only the
            ``upper`` triangle), otherwise a copy with frozen rows replaced by
            identity rows.
        other (torch.Tensor): Strict opposite triangle, applied to the old iterate.
        frozen (torch.Tensor | None): Boolean mask of rows kept at their old value,
            or ``None`` when every row is updated.
        upper (bool): Whether ``triangle`` is upper triangular (backward sweep).
    """

    triangle: torch.Tensor
    other: torch.Tensor
    frozen: torch.Tensor | None
    upper: bool


def _prepare_sweep(
    matrix: torch.Tensor, *, forward: bool, active: torch.Tensor | None
) -> _PreparedSweep:
    """Build one direction's fixed triangular system, freezing inactive/zero-diagonal rows.

    Args:
        matrix (torch.Tensor): Dense matrix ``A``, shape ``(n, n)``.
        forward (bool): Ascending (``True``) or descending row order.
        active (torch.Tensor | None): Boolean mask of rows allowed to change, or
            ``None`` for all rows.

    Returns:
        _PreparedSweep: Reusable triangular system for this direction.
    """
    frozen = torch.diagonal(matrix) == 0
    if active is not None:
        frozen = frozen | ~active
    other = torch.triu(matrix, 1) if forward else torch.tril(matrix, -1)
    if not bool(frozen.any()):
        return _PreparedSweep(triangle=matrix, other=other, frozen=None, upper=not forward)
    triangle = torch.tril(matrix) if forward else torch.triu(matrix)
    triangle = triangle * (~frozen).to(matrix.dtype).unsqueeze(1) + torch.diag(
        frozen.to(matrix.dtype)
    )
    return _PreparedSweep(triangle=triangle, other=other, frozen=frozen, upper=not forward)


def _solve_sweep(prepared: _PreparedSweep, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.solve_triangular(prepared.triangle, target, upper=prepared.upper)


def _apply_sweep(prepared: _PreparedSweep, x: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve one already-prepared triangular system for the current iterate.

    Args:
        prepared (_PreparedSweep): Direction's fixed triangular system.
        x (torch.Tensor): Current iterate, shape ``(n, k)``.
        rhs (torch.Tensor): Right-hand side, shape ``(n, k)``.

    Returns:
        torch.Tensor: Iterate after the sweep, shape ``(n, k)``.
    """
    target = rhs - prepared.other @ x
    if prepared.frozen is not None:
        target = torch.where(prepared.frozen.unsqueeze(-1), x, target)
    return _solve_sweep(prepared, target)


def _apply_first_sweep_from_zero(prepared: _PreparedSweep, rhs: torch.Tensor) -> torch.Tensor:
    """Solve one prepared triangular system starting from the zero iterate.

    The strict-triangle product vanishes for ``x = 0``, so the matmul is skipped.

    Args:
        prepared (_PreparedSweep): Direction's fixed triangular system.
        rhs (torch.Tensor): Right-hand side, shape ``(n, k)``.

    Returns:
        torch.Tensor: Iterate after the sweep, shape ``(n, k)``.
    """
    if prepared.frozen is None:
        return _solve_sweep(prepared, rhs)
    return _solve_sweep(prepared, rhs.masked_fill(prepared.frozen.unsqueeze(-1), 0.0))


def _as_columns(vector: torch.Tensor) -> torch.Tensor:
    return vector.unsqueeze(-1) if vector.ndim == 1 else vector


@dataclass(frozen=True, slots=True)
class PreparedSymmetricGaussSeidel:
    """Symmetric Gauss-Seidel bound to one matrix (``PreparedSmoother``).

    Attributes:
        matrix (torch.Tensor): The dense matrix ``A`` (for cycle residuals).
        forward (_PreparedSweep): Ascending sweep ``(D + L) x_new = b - U x_old``.
        backward (_PreparedSweep): Descending sweep ``(D + U) x_new = b - L x_old``.
    """

    matrix: torch.Tensor
    forward: _PreparedSweep
    backward: _PreparedSweep

    def residual(self, rhs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return ``rhs - A x`` with the bound level matrix.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            x (torch.Tensor): Current iterate, same shape as ``rhs``.

        Returns:
            torch.Tensor: Residual, same shape as ``rhs``.
        """
        return rhs - self.matrix @ x

    def smooth(self, rhs: torch.Tensor, x: torch.Tensor, steps: int) -> torch.Tensor:
        """Apply ``steps`` symmetric iterations (forward then backward) from ``x``.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            x (torch.Tensor): Current iterate (not modified), same shape as ``rhs``.
            steps (int): Number of symmetric iterations.

        Returns:
            torch.Tensor: Updated iterate, same shape as ``x``.
        """
        if steps == 0:
            return x
        current, target = _as_columns(x), _as_columns(rhs)
        for _ in range(steps):
            current = _apply_sweep(self.forward, current, target)
            current = _apply_sweep(self.backward, current, target)
        return current.reshape(x.shape)

    def smooth_from_zero(self, rhs: torch.Tensor, steps: int) -> torch.Tensor:
        """Apply ``steps`` symmetric iterations from the zero iterate.

        Equal to ``smooth(rhs, zeros_like(rhs), steps)`` without the first forward
        sweep's strict-triangle product.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            steps (int): Number of symmetric iterations.

        Returns:
            torch.Tensor: Iterate after smoothing, same shape as ``rhs``.
        """
        if steps == 0:
            return torch.zeros_like(rhs)
        target = _as_columns(rhs)
        current = _apply_sweep(
            self.backward, _apply_first_sweep_from_zero(self.forward, target), target
        )
        for _ in range(steps - 1):
            current = _apply_sweep(self.forward, current, target)
            current = _apply_sweep(self.backward, current, target)
        return current.reshape(rhs.shape)


def prepare_symmetric_gauss_seidel(
    matrix: torch.Tensor, *, rows: torch.Tensor | None = None
) -> PreparedSymmetricGaussSeidel:
    """Build the forward and backward triangular systems of ``matrix`` once.

    Args:
        matrix (torch.Tensor): Dense matrix ``A``, shape ``(n, n)``.
        rows (torch.Tensor | None): If given, only these rows (ascending
            indices) are updated - PyAMG's ``gauss_seidel_indexed``.

    Returns:
        PreparedSymmetricGaussSeidel: Smoother bound to ``matrix``.
    """
    active = None
    if rows is not None:
        active = torch.zeros(matrix.shape[0], dtype=torch.bool, device=matrix.device)
        active[rows] = True
    return PreparedSymmetricGaussSeidel(
        matrix=matrix,
        forward=_prepare_sweep(matrix, forward=True, active=active),
        backward=_prepare_sweep(matrix, forward=False, active=active),
    )
