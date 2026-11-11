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

from collections.abc import Callable
from typing import Self

import torch
from torch import nn

from torchalg.sparse.kernels.triangular import TriangularSolver, prepare_triangular_solve


class SparseTriangularSolveCache(nn.Module):
    """Setup-owned prepared forward and backward solves for a triangular pair.

    Each half comes from ``prepare_triangular_solve``, which picks the implementation
    registered for the factors' device type (``TRIANGULAR_SOLVER_BY_DEVICE``). The
    factors are kept as the canonical state; when ``.to()`` moves the cache to another
    device type, both halves are re-resolved for the new device instead of carrying the
    old device's implementation along (the CPU library solve faults on ROCm). ``apply``
    never branches on the device.

    Args:
        lower: Explicit sparse CSR lower factor.
        upper: Explicit sparse CSR upper factor.
    """

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        """Prepare both substitutions for a fixed lower/upper factor pair."""
        super().__init__()
        self.lower: torch.Tensor
        self.upper: torch.Tensor
        self.register_buffer("lower", lower, persistent=False)
        self.register_buffer("upper", upper, persistent=False)
        self._resolve_solvers()

    @classmethod
    def build(cls, lower: torch.Tensor, upper: torch.Tensor) -> SparseTriangularSolveCache:
        """Prepare both substitutions for a fixed lower/upper factor pair.

        Args:
            lower: Explicit sparse CSR lower factor.
            upper: Explicit sparse CSR upper factor.

        Returns:
            Complete reusable solve cache.
        """
        return cls(lower, upper)

    def _resolve_solvers(self) -> None:
        self.forward_solve: TriangularSolver = prepare_triangular_solve(self.lower, "forward")
        self.backward_solve: TriangularSolver = prepare_triangular_solve(self.upper, "backward")

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> Self:
        device_type = self.lower.device.type
        super()._apply(fn, recurse)
        if self.lower.device.type != device_type:
            self._resolve_solvers()
        return self

    def solve(self, residual: torch.Tensor) -> torch.Tensor:
        """Apply the cached lower/upper triangular solve pair.

        Args:
            residual: Vector or matrix right-hand side.

        Returns:
            Preconditioned residual with the same shape as ``residual``.
        """
        return self.backward_solve.solve(self.forward_solve.solve(residual))


def build_sparse_cholesky_solve_cache(factor: torch.Tensor) -> SparseTriangularSolveCache:
    """Materialize ``factor.T`` and both schedules for repeated application.

    Args:
        factor: Sparse CSR lower triangular Cholesky-style factor.

    Returns:
        Reusable solve cache for ``factor @ factor.T``.
    """
    upper = factor.to_sparse_coo().t().to_sparse_csr()
    return SparseTriangularSolveCache.build(factor, upper)
