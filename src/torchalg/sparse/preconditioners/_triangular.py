"""Private reusable state for sparse pairs of triangular solves.

Shared by ``icholesky.py`` and ``ic0.py``: both preconditioners hold a lower
triangular sparse CSR factor ``L`` and solve ``(L @ L.T) z = r`` for ``z`` -
the only difference between them is how ``L`` is obtained (externally
supplied vs. computed via ``sparse_ic0``). Mirrors the dense sibling's
``preconditioners.implementations._triangular.cholesky_factor_solve`` - same
shared-helper precedent. Factor transposition and schedule construction
happen during preconditioner setup; repeated application performs only the
two numerical triangular solves.
"""

from __future__ import annotations

import torch
from torch import nn

from torchalg.sparse.kernels.triangular import LevelSchedule, level_schedule, triangular_solve


class SparseTriangularSolveCache(nn.Module):
    """Setup-owned upper factor and schedules for a triangular solve pair.

    The lower factor remains owned by the parent preconditioner. All derived
    tensors here are registered as non-persistent buffers: they follow
    ``nn.Module.to()`` but are rebuilt from the canonical operator by
    ``setup()`` rather than serialized as a second source of truth.

    Args:
        upper: Explicit sparse CSR upper factor.
        forward_schedule: Schedule for the lower solve.
        backward_schedule: Schedule for the upper solve.
    """

    def __init__(
        self,
        upper: torch.Tensor,
        forward_schedule: LevelSchedule,
        backward_schedule: LevelSchedule,
    ) -> None:
        """Register a complete, internally consistent solve cache."""
        super().__init__()
        self.upper: torch.Tensor
        self.forward_row_order: torch.Tensor
        self.forward_level_sizes: torch.Tensor
        self.backward_row_order: torch.Tensor
        self.backward_level_sizes: torch.Tensor
        self.register_buffer("upper", upper, persistent=False)
        self.register_buffer("forward_row_order", forward_schedule.row_order, persistent=False)
        self.register_buffer("forward_level_sizes", forward_schedule.level_sizes, persistent=False)
        self.register_buffer("backward_row_order", backward_schedule.row_order, persistent=False)
        self.register_buffer(
            "backward_level_sizes", backward_schedule.level_sizes, persistent=False
        )

    @classmethod
    def build(cls, lower: torch.Tensor, upper: torch.Tensor) -> SparseTriangularSolveCache:
        """Construct schedules for a fixed lower/upper factor pair.

        Args:
            lower: Explicit sparse CSR lower factor.
            upper: Explicit sparse CSR upper factor.

        Returns:
            Complete reusable solve cache.
        """
        return cls(
            upper=upper,
            forward_schedule=level_schedule(lower, direction="forward"),
            backward_schedule=level_schedule(upper, direction="backward"),
        )

    def solve(self, lower: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Apply the cached lower/upper triangular solve pair.

        Args:
            lower: Explicit sparse CSR lower factor matching the cache.
            residual: Vector or matrix right-hand side.

        Returns:
            Preconditioned residual with the same shape as ``residual``.
        """
        forward_schedule = LevelSchedule(
            row_order=self.forward_row_order,
            level_sizes=self.forward_level_sizes,
        )
        backward_schedule = LevelSchedule(
            row_order=self.backward_row_order,
            level_sizes=self.backward_level_sizes,
        )

        y = triangular_solve(lower, forward_schedule, residual, direction="forward")
        return triangular_solve(self.upper, backward_schedule, y, direction="backward")


def build_sparse_cholesky_solve_cache(factor: torch.Tensor) -> SparseTriangularSolveCache:
    """Materialize ``factor.T`` and both schedules for repeated application.

    Args:
        factor: Sparse CSR lower triangular Cholesky-style factor.

    Returns:
        Reusable solve cache for ``factor @ factor.T``.
    """
    upper = factor.to_sparse_coo().t().to_sparse_csr()
    return SparseTriangularSolveCache.build(factor, upper)
