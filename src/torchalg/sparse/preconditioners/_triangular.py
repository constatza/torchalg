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

from torchalg.sparse.kernels.triangular import PreparedTriangularSolve


class SparseTriangularSolveCache(nn.Module):
    """Setup-owned prepared forward and backward solves for a triangular pair.

    Each half is a ``PreparedTriangularSolve`` (schedule, diagonal and level-sorted
    entries computed once); ``nn.Module.to()`` moves their non-persistent buffers, and
    ``setup()`` rebuilds them from the canonical operator rather than serializing a
    second source of truth.

    Args:
        forward: Prepared solve against the lower factor.
        backward: Prepared solve against the upper factor.
    """

    def __init__(self, forward: PreparedTriangularSolve, backward: PreparedTriangularSolve) -> None:
        """Register a complete, internally consistent solve cache."""
        super().__init__()
        self.forward_solve = forward
        self.backward_solve = backward

    @classmethod
    def build(cls, lower: torch.Tensor, upper: torch.Tensor) -> SparseTriangularSolveCache:
        """Prepare both substitutions for a fixed lower/upper factor pair.

        Args:
            lower: Explicit sparse CSR lower factor.
            upper: Explicit sparse CSR upper factor.

        Returns:
            Complete reusable solve cache.
        """
        return cls(
            PreparedTriangularSolve(lower, "forward"),
            PreparedTriangularSolve(upper, "backward"),
        )

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
