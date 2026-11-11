"""Dense-backed symmetric Gauss-Seidel for small and mid-size sparse levels.

Same smoother as ``_gauss_seidel.PreparedSymmetricGaussSeidel`` - same sweeps,
same frozen-pivot and indexed-row semantics, same ``smooth``/``smooth_from_zero``
interface - but the two triangular systems are densified once and solved with
``solve_triangular``. A level-scheduled CSR sweep costs one kernel launch per
dependency level, which dominates below a few thousand unknowns; a dense
triangular solve costs ``O(n^2)`` regardless of sparsity, so it wins there (the
crossover is chosen in ``_gauss_seidel_format``). The input is still a sparse CSR
matrix: the dense copy is private to this object, and nothing in the sparse tree
imports the dense tree.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchalg.sparse.kernels.index_dtype import compact_csr_indices

_PIVOT_TOL = 1e-14
"""Diagonal magnitude at or below this freezes the row, as in ``PreparedTriangularSolve``."""


def _as_columns(vector: torch.Tensor) -> torch.Tensor:
    return vector.unsqueeze(-1) if vector.ndim == 1 else vector


def _with_identity_rows(triangle: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``triangle`` with the rows flagged in ``rows`` replaced by identity rows."""
    keep = (~rows).to(triangle.dtype).unsqueeze(1)
    return triangle * keep + torch.diag(rows.to(triangle.dtype))


@dataclass(frozen=True, slots=True)
class DenseBackedSymmetricGaussSeidel:
    """Symmetric Gauss-Seidel bound to one matrix, solved against dense triangles.

    Attributes:
        matrix: The sparse CSR matrix ``A`` (int32 indices where it fits), used only
            for cycle residuals so they stay O(nnz).
        frozen: Rows whose pivot is (near) zero; they take the sweep's target value.
        upper: Strict upper triangle ``U`` (applied to the old iterate in the forward sweep).
        lower: Strict lower triangle ``L`` (applied in the backward sweep).
        forward: ``D + L`` with frozen rows replaced by identity rows.
        backward: ``D + U`` with frozen rows replaced by identity rows.
    """

    matrix: torch.Tensor
    frozen: torch.Tensor
    upper: torch.Tensor
    lower: torch.Tensor
    forward: torch.Tensor
    backward: torch.Tensor

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
    def build(cls, matrix: torch.Tensor) -> DenseBackedSymmetricGaussSeidel:
        """Densify ``matrix`` once and split it into the sweep triangles.

        Args:
            matrix (torch.Tensor): Sparse CSR matrix ``A``, shape ``(n, n)``.

        Returns:
            DenseBackedSymmetricGaussSeidel: Reusable sweeps over ``matrix``.
        """
        dense = matrix.to_dense()
        frozen = dense.diagonal().abs() <= _PIVOT_TOL
        return cls(
            matrix=compact_csr_indices(matrix),
            frozen=frozen,
            upper=torch.triu(dense, 1),
            lower=torch.tril(dense, -1),
            forward=_with_identity_rows(torch.tril(dense), frozen),
            backward=_with_identity_rows(torch.triu(dense), frozen),
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
            rows (torch.Tensor | None): If given, only these rows are updated
                (PyAMG's ``gauss_seidel_indexed``); the others keep their value
                from ``x`` for the whole call.

        Returns:
            torch.Tensor: Updated iterate, same shape as ``x``.
        """
        if steps == 0:
            return x.clone()
        target, current = _as_columns(rhs), _as_columns(x)
        forward, backward, inactive = self.forward, self.backward, None
        if rows is not None:
            inactive = torch.ones(self.frozen.shape, dtype=torch.bool, device=self.frozen.device)
            inactive[rows] = False
            forward = _with_identity_rows(forward, inactive)
            backward = _with_identity_rows(backward, inactive)
        for _ in range(steps):
            current = self._forward_sweep(target - self.upper @ current, current, forward, inactive)
            current = self._backward_sweep(target, current, backward, inactive)
        return current.reshape(x.shape)

    def smooth_from_zero(self, rhs: torch.Tensor, steps: int) -> torch.Tensor:
        """Apply ``steps`` symmetric iterations from the zero guess.

        The first forward sweep's right-hand side is ``rhs - U @ 0 = rhs``, so
        that product is skipped.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            steps (int): Number of symmetric iterations.

        Returns:
            torch.Tensor: Iterate after ``steps`` iterations, same shape as ``rhs``.
        """
        if steps == 0:
            return torch.zeros_like(rhs)
        target = _as_columns(rhs)
        first = torch.linalg.solve_triangular(self.forward, target, upper=False)
        current = self._backward_sweep(target, first, self.backward, None)
        out = current.reshape(rhs.shape)
        return out if steps == 1 else self.smooth(rhs, out, steps - 1)

    @staticmethod
    def _forward_sweep(
        target: torch.Tensor,
        current: torch.Tensor,
        forward: torch.Tensor,
        inactive: torch.Tensor | None,
    ) -> torch.Tensor:
        if inactive is not None:
            target = torch.where(inactive.unsqueeze(-1), current, target)
        return torch.linalg.solve_triangular(forward, target, upper=False)

    def _backward_sweep(
        self,
        rhs: torch.Tensor,
        current: torch.Tensor,
        backward: torch.Tensor,
        inactive: torch.Tensor | None,
    ) -> torch.Tensor:
        target = rhs - self.lower @ current
        if inactive is not None:
            target = torch.where(inactive.unsqueeze(-1), current, target)
        return torch.linalg.solve_triangular(backward, target, upper=True)
