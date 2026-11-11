"""Sparse-CSR symmetric Gauss-Seidel, prepared once per matrix.

No factorization step: each sweep is a direct triangular solve against
``A``'s own entries, split by position relative to the diagonal - see
``docs/plan.md``'s "AMG/POD smoother constraint" section. Mirrors the dense
``_relaxation.py``'s D/L/U convention: forward sweep solves ``(D + L)
x_new = rhs - U @ x_old``, backward solves ``(D + U) x_new = rhs - L @
x_old``, one iteration = one forward sweep followed by one backward sweep.

``PreparedSymmetricGaussSeidel.build`` does every matrix-only computation once
(both prepared triangular solves - level schedule, diagonal and level-sorted
entries each - and the strict-triangle entries the off-triangle matvecs read); ``smooth``/``smooth_from_zero`` then
only sweep. Callers that relax the same matrix many times (smoothers,
compatible relaxation, adaptive setup stages) hold one prepared object per
matrix instead of rebuilding schedules per call.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchalg.sparse.kernels.index_dtype import compact_csr_indices
from torchalg.sparse.kernels.triangular import (
    PreparedTriangularSolve,
    _expand_row_index,
    _require_csr,
)


@dataclass(frozen=True, slots=True)
class _StrictTriangle:
    """Stored entries of one strict triangle of a sparse CSR matrix.

    Attributes:
        row (torch.Tensor): Row of each selected entry.
        col (torch.Tensor): Column of each selected entry.
        values (torch.Tensor): Value of each selected entry.
    """

    row: torch.Tensor
    col: torch.Tensor
    values: torch.Tensor

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply ``x`` by this strict triangle with a functional ``scatter_add``.

        No intermediate ``torch.sparse_csr_tensor`` is built.

        Args:
            x (torch.Tensor): Vector ``(n,)`` or block ``(n, k)``.

        Returns:
            torch.Tensor: Product, same shape as ``x``.
        """
        gathered = x[self.col]
        if x.ndim == 1:
            return torch.zeros_like(x).scatter_add(0, self.row, self.values * gathered)
        return torch.zeros_like(x).scatter_add(
            0,
            self.row.unsqueeze(-1).expand(-1, x.shape[1]),
            self.values.unsqueeze(-1) * gathered,
        )


@dataclass(frozen=True, slots=True)
class PreparedSymmetricGaussSeidel:
    """Symmetric Gauss-Seidel bound to one sparse CSR matrix.

    Build with ``build``; every field is matrix-only structure computed once.

    Attributes:
        matrix (torch.Tensor): The sparse CSR matrix ``A``, shape ``(n, n)``, with
            int32 indices where it fits (used only for residuals).
        upper (_StrictTriangle): Strict upper entries (``col > row``).
        lower (_StrictTriangle): Strict lower entries (``col < row``).
        forward (PreparedTriangularSolve): ``(D + L)`` substitution of the forward sweep.
        backward (PreparedTriangularSolve): ``(D + U)`` substitution of the backward sweep.
    """

    matrix: torch.Tensor
    upper: _StrictTriangle
    lower: _StrictTriangle
    forward: PreparedTriangularSolve
    backward: PreparedTriangularSolve

    def residual(self, rhs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return ``rhs - A x`` with the bound level matrix.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            x (torch.Tensor): Current iterate, same shape as ``rhs``.

        Returns:
            torch.Tensor: Residual, same shape as ``rhs``.
        """
        return rhs - self.matrix @ x

    @classmethod
    def build(cls, matrix: torch.Tensor) -> PreparedSymmetricGaussSeidel:
        """Compute both level schedules and the strict-triangle split of ``matrix``.

        Args:
            matrix (torch.Tensor): Sparse CSR matrix ``A``, shape ``(n, n)``.

        Returns:
            PreparedSymmetricGaussSeidel: Reusable sweeps over ``matrix``.

        Raises:
            ValueError: If ``matrix`` is not sparse CSR.
        """
        _require_csr(matrix, "PreparedSymmetricGaussSeidel.build")
        row_index = _expand_row_index(matrix)
        col = matrix.col_indices()
        values = matrix.values()

        def triangle(selected: torch.Tensor) -> _StrictTriangle:
            return _StrictTriangle(row_index[selected], col[selected], values[selected])

        return cls(
            matrix=compact_csr_indices(matrix),
            upper=triangle(col > row_index),
            lower=triangle(col < row_index),
            forward=PreparedTriangularSolve(matrix, "forward", row_index=row_index),
            backward=PreparedTriangularSolve(matrix, "backward", row_index=row_index),
        )

    def smooth(
        self,
        rhs: torch.Tensor,
        x: torch.Tensor,
        steps: int,
        *,
        rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply ``steps`` symmetric Gauss-Seidel iterations from ``x``.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            x (torch.Tensor): Current iterate, same shape; not modified.
            steps (int): Number of symmetric iterations.
            rows (torch.Tensor | None): If given, only these rows (indices) are
                updated - sparse sibling of the dense indexed sweep (PyAMG's
                ``gauss_seidel_indexed``). Rows outside ``rows`` stay frozen
                at their current value for the whole call.

        Returns:
            torch.Tensor: Updated iterate, same shape as ``x``.
        """
        if steps == 0:
            return x.clone()
        active = None
        if rows is not None:
            active = torch.zeros(self.matrix.shape[0], dtype=torch.bool, device=self.matrix.device)
            active[rows] = True
        for _ in range(steps):
            x = self._sweep_pair(rhs, rhs - self.upper.matvec(x), x, active)
        return x

    def smooth_from_zero(self, rhs: torch.Tensor, steps: int) -> torch.Tensor:
        """Apply ``steps`` symmetric iterations from the zero guess.

        The first forward sweep's right-hand side is ``rhs - U @ 0 = rhs``, so
        that strict-triangle matvec is skipped.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            steps (int): Number of symmetric iterations.

        Returns:
            torch.Tensor: Iterate after ``steps`` iterations, same shape as ``rhs``.
        """
        if steps == 0:
            return torch.zeros_like(rhs)
        x = self._sweep_pair(rhs, rhs, None, None)
        return x if steps == 1 else self.smooth(rhs, x, steps - 1)

    def _sweep_pair(
        self,
        rhs: torch.Tensor,
        forward_target: torch.Tensor,
        current: torch.Tensor | None,
        active: torch.Tensor | None,
    ) -> torch.Tensor:
        """One forward sweep followed by one backward sweep.

        Args:
            rhs (torch.Tensor): Right-hand side of the system.
            forward_target (torch.Tensor): ``rhs - U @ x_old`` for the forward solve.
            current (torch.Tensor | None): ``x_old``; needed only to freeze inactive rows.
            active (torch.Tensor | None): Boolean mask of updatable rows, or ``None``.

        Returns:
            torch.Tensor: Iterate after the forward and backward sweeps.
        """
        x = self.forward.solve(forward_target, active=active, current=current)
        return self.backward.solve(rhs - self.lower.matvec(x), active=active, current=x)
