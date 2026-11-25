"""Spectral damping rules for weighted Jacobi, sparse-CSR sibling.

Sparse-only counterpart of
``preconditioners.implementations.amg._jacobi_omega`` (dense; kept
unmodified for comparison - see ``docs/plan.md``'s "Correction: dense and
sparse must be separate implementations, not an internal branch"). Shares
``torchalg.utils.spectral.approximate_spectral_radius`` with the dense
sibling - the one piece of code genuinely common to both trees, since it
never itself branches on format (only ever calls ``matrix @ vector``, which
both dense and sparse CSR support via the same operator).

References:
    - PyAMG 5.3.0, ``relaxation/smoothing.py`` (``rho_D_inv_A``) and
      ``aggregation/smooth.py`` (``jacobi_prolongation_smoother``).
    - Vanek, P., Mandel, J., & Brezina, M. (1996). Computing, 56(3), 179-196.
"""

from __future__ import annotations

import torch

from torchalg.sparse.kernels.diagonal import sparse_diagonal
from torchalg.sparse.kernels.rowscale import sparse_row_scale
from torchalg.utils.spectral import approximate_spectral_radius
from torchalg.utils.weak_cache import tensor_memo

RELAXATION_NOMINAL = 1.0
"""Nominal Jacobi damping for relaxation: ``omega = 1 / rho``."""

PROLONGATION_NOMINAL = 4.0 / 3.0
"""Nominal Jacobi damping for prolongator smoothing: ``omega = (4/3) / rho``."""

_SEED = 0


def scaled_by_inverse_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """``D^-1 A`` with ``D^-1`` zero where the diagonal is zero, sparse CSR input.

    Args:
        matrix (torch.Tensor): Square sparse CSR matrix ``A``.

    Returns:
        torch.Tensor: Row-scaled sparse CSR matrix.
    """
    diagonal = sparse_diagonal(matrix)
    inverse = torch.where(diagonal != 0, 1.0 / diagonal, torch.zeros_like(diagonal))
    return sparse_row_scale(matrix, inverse)


def _seeded_draw(size: int) -> torch.Tensor:
    """Fixed-seed uniform ``[0, 1)`` start vector for the Arnoldi estimate.

    Args:
        size (int): Vector length.

    Returns:
        torch.Tensor: The same vector on every call.
    """
    generator = torch.Generator().manual_seed(_SEED)
    return torch.rand(size, generator=generator, dtype=torch.float64)


@tensor_memo
def jacobi_spectral_radius(matrix: torch.Tensor) -> torch.Tensor:
    """Deterministic Arnoldi estimate of ``rho(D^-1 A)``, sparse CSR input.

    Memoized per matrix object (see ``torchalg.utils.weak_cache.tensor_memo``): the
    start vector is fixed, so the estimate is a pure function of ``A``, and the
    prolongator smoothing (``AggregationCoarsening``) and the relaxation
    (``JacobiSmoother.prepare``) of one level both receive only that level's matrix,
    so the shared object identity is the only channel to share one estimate. Nothing
    may call it from ``apply``.

    Cost:
        Algorithmic: one Arnoldi run (see
        ``torchalg.utils.spectral.approximate_spectral_radius``). Paid once per matrix
        object instead of once per consumer, so a Jacobi-smoothed,
        aggregation-coarsened level pays one run, not two.

        Per memo miss, ``z = nnz``, ``P`` Arnoldi passes of ``m = 15`` steps:
        ``W = z + n`` for the diagonal scaling plus ``P (2zm + 4nm(m + 1))``
        for the estimate, ``D = P m (ceil(log2 n) + 1) * O(1)`` and
        ``S = P``; a memo hit costs nothing. ``P`` is data dependent
        (restarting stops once the Ritz pair is within tolerance), usually 1
        to 2. ``Q`` is the SpMV and basis traffic of those passes, with
        ``W/Q`` below 0.2 flop per byte (memory-bound).

    Args:
        matrix (torch.Tensor): Square sparse CSR matrix ``A``.

    Returns:
        torch.Tensor: Estimated spectral radius of ``D^-1 A``, 0-d.
    """
    return approximate_spectral_radius(scaled_by_inverse_diagonal(matrix), _seeded_draw)


def jacobi_omega(
    matrix: torch.Tensor, nominal: float, override: float | torch.Tensor | None
) -> torch.Tensor:
    """Resolve a Jacobi damping factor: the explicit value, else ``nominal / rho``.

    Cost:
        An explicit ``override`` costs one scalar conversion (``W = 0``, ``L = 1``,
        ``S = 0``). Otherwise the cost is that of ``jacobi_spectral_radius`` on
        a memo miss: ``P`` Arnoldi passes of ``m = 15`` steps,
        ``W = z + n + P (2zm + 4nm(m + 1))``, ``D = P m (ceil(log2 n) + 1) * O(1)``,
        ``S = P``, with ``z = nnz``; a memo hit (a second consumer of the same level
        matrix) costs one division. Memory-bound (``W/Q`` below 0.2 flop per byte).

    Args:
        matrix (torch.Tensor): Sparse CSR matrix the damping is applied to.
        nominal (float): ``RELAXATION_NOMINAL`` or ``PROLONGATION_NOMINAL``.
        override (float | torch.Tensor | None): Explicit omega; ``None``
            selects the spectral rule.

    Returns:
        torch.Tensor: Damping factor, 0-d, matching ``matrix``'s dtype/device.
    """
    if override is not None:
        return torch.as_tensor(override, dtype=matrix.dtype, device=matrix.device)
    return nominal / jacobi_spectral_radius(matrix)
