"""Private helpers for sparse Cholesky-style triangular solves.

Shared by ``icholesky.py`` and ``ic0.py``: both preconditioners hold a lower
triangular sparse CSR factor ``L`` and solve ``(L @ L.T) z = r`` for ``z`` -
the only difference between them is how ``L`` is obtained (externally
supplied vs. computed via ``sparse_ic0``). Mirrors the dense sibling's
``preconditioners.implementations._triangular.cholesky_factor_solve`` - same
shared-helper precedent (keeping the preconditioner classes themselves
declarative, free of branching logic), now built on the level-scheduled
sparse solve instead of ``torch.cholesky_solve``.
"""

from __future__ import annotations

import torch

from torchalg.sparse.kernels.triangular import level_schedule, triangular_solve


def sparse_cholesky_factor_solve(factor: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Solve ``(factor @ factor.T) z = residual`` via two level-scheduled sparse triangular solves.

    ``factor`` is stored lower-only; ``factor.T`` has to be materialized as
    its own sparse CSR tensor (not just "the same values read upper") since
    ``triangular_solve``/``level_schedule`` operate on whatever pattern is
    literally stored in the tensor they're handed - a lower-only ``factor``
    has nothing stored above the diagonal for a ``direction="backward"``
    solve to find. ``.to_sparse_coo().t().to_sparse_csr()`` builds that
    transpose; recomputed per call rather than cached, matching
    ``IC0Preconditioner.apply()``'s existing tradeoff (correctness over
    amortization for this first pass).

    Args:
        factor (torch.Tensor): Sparse CSR lower triangular factor ``L``,
            such that the preconditioner matrix is ``L @ L.T``.
        residual (torch.Tensor): Residual vector(s) ``r``, shape ``(n,)`` or
            ``(n, k)``.

    Returns:
        torch.Tensor: Preconditioned residual ``z``, same shape as
            ``residual``.
    """
    if residual.ndim > 1:
        columns = [
            sparse_cholesky_factor_solve(factor, residual[:, k]) for k in range(residual.shape[1])
        ]
        return torch.stack(columns, dim=1)

    lower = factor
    upper = lower.to_sparse_coo().t().to_sparse_csr()

    forward_schedule = level_schedule(lower, direction="forward")
    backward_schedule = level_schedule(upper, direction="backward")

    y = triangular_solve(lower, forward_schedule, residual, direction="forward")
    return triangular_solve(upper, backward_schedule, y, direction="backward")
