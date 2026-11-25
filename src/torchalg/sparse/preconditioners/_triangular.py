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
from torchalg.sparse.preconditioners._ic0_pattern import IC0Pattern
from torchalg.sparse.preconditioners._ilu0_pattern import ILU0Pattern


class PreparedTriangularPair(nn.Module):
    """Setup-owned prepared forward and backward solves for a triangular pair.

    Each half comes from ``prepare_triangular_solve``, which picks the implementation
    registered for the factors' device type (``TRIANGULAR_SOLVER_BY_DEVICE``). The
    factors are kept as the canonical state; when ``.to()`` moves the pair to another
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


def build_sparse_cholesky_solve_cache(
    factor: torch.Tensor, pattern: IC0Pattern | None = None
) -> PreparedTriangularPair:
    """Materialize ``factor.T`` and both schedules for repeated application.

    Args:
        factor: Sparse CSR lower triangular Cholesky-style factor.
        pattern: IC(0) plan whose lower structure ``factor`` has; its
            precomputed transpose permutation replaces the COO round trip.
            ``None`` transposes ``factor`` generically.

    Returns:
        Reusable solve pair for ``factor @ factor.T``.
    """
    if pattern is None:
        upper = factor.to_sparse_coo().t().to_sparse_csr()
    else:
        upper = torch.sparse_csr_tensor(
            pattern.upper_crow,
            pattern.upper_col,
            factor.values()[pattern.transpose_perm],
            size=factor.shape,
            check_invariants=False,
        )
    return PreparedTriangularPair(factor, upper)


def build_sparse_lu_solve_cache(
    factor: torch.Tensor, pattern: ILU0Pattern
) -> PreparedTriangularPair:
    """Split a combined ILU(0) factor into its two triangles for repeated application.

    ``L`` keeps its implicit unit diagonal out of the combined factor, but the
    triangular solve divides by whatever diagonal is stored, so the lower
    triangle gets an explicit ``1.0`` diagonal. ``U`` is the combined factor's
    own diagonal-and-upper part. Both are gathered by the pattern's
    precomputed positions.

    Args:
        factor: Combined ``L``/``U`` sparse CSR factor from ``sparse_ilu0``.
        pattern: ILU(0) plan whose stored structure ``factor`` has.

    Returns:
        Reusable solve pair for ``(L @ U)^{-1}``.
    """
    values = factor.values()
    with_unit = torch.cat([values, values.new_ones(1)])
    lower = torch.sparse_csr_tensor(
        pattern.lower_crow,
        pattern.lower_col,
        with_unit[pattern.lower_source],
        size=factor.shape,
        check_invariants=False,
    )
    upper = torch.sparse_csr_tensor(
        pattern.upper_crow,
        pattern.upper_col,
        values[pattern.upper_source],
        size=factor.shape,
        check_invariants=False,
    )
    return PreparedTriangularPair(lower, upper)
