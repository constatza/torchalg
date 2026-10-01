"""Sparse-native strength-of-connection kernel for SA-AMG coarsening.

Companion to the dense
``preconditioners.implementations.amg._aggregation.strength_of_connection``
(kept unmodified for comparison - see ``docs/plan.md``'s "Correction:
strength-of-connection/aggregation must be sparse-native too"):
``strength_of_connection`` only ever touches ``A``'s own existing nonzero
entries (no multi-hop neighborhood, no fill-in), so a sparse version is a
pure vectorized gather/compare over CSR's flat arrays - O(nnz), no ``(n,
n)`` dense intermediate, no Python loop. A leaf primitive - no standalone
breakdown/convergence story of its own, consumed by the SA-AMG algorithm in
``torchalg.sparse.preconditioners.amg.aggregation``.

References:
    - Vanek, P., Mandel, J., & Brezina, M. (1996). Algebraic multigrid by
      smoothed aggregation for second and fourth order elliptic problems.
      Computing, 56(3), 179-196.
"""

from __future__ import annotations

import torch

from .diagonal import sparse_diagonal
from .row_max import sparse_row_max
from .triangular import _expand_row_index, _require_csr

_NEAR_ZERO_DIAGONAL_TOL = 1e-14
"""Diagonal entries with magnitude below this are treated as 1.0 when
normalizing strength-of-connection ratios (protects against division by
near-zero) - matches the dense ``_aggregation.py`` constant."""


def sparse_strength_of_connection(matrix: torch.Tensor, theta: float) -> torch.Tensor:
    """Build a sparse strength-of-connection graph from a sparse CSR matrix.

    Entry (i, j) is strong if ``|a_ij| >= theta * sqrt(|a_ii| * |a_jj|)`` -
    the same Cauchy-Schwarz measure as the dense
    ``_aggregation.strength_of_connection`` - restricted to ``matrix``'s
    already-stored nonzero entries (never a candidate pair outside that
    pattern) and diagonal entries excluded.

    Args:
        matrix (torch.Tensor): Sparse CSR fine-grid matrix, shape ``(n, n)``.
        theta (float): Strength-of-connection threshold theta in (0, 1).

    Returns:
        torch.Tensor: Sparse CSR boolean strength matrix, shape ``(n, n)``.
    """
    n = matrix.shape[0]
    crow = matrix.crow_indices()
    col = matrix.col_indices()
    values = matrix.values()

    diag = sparse_diagonal(matrix).abs()
    diag_safe = torch.where(diag > _NEAR_ZERO_DIAGONAL_TOL, diag, torch.ones_like(diag))

    row_nnz = crow[1:] - crow[:-1]
    row_index = torch.repeat_interleave(torch.arange(n, device=matrix.device), row_nnz)

    normalizer = torch.sqrt(diag_safe[row_index] * diag_safe[col])
    strengths = values.abs() / normalizer
    keep = (strengths >= theta) & (row_index != col) & (values != 0)

    kept_row = row_index[keep]
    kept_col = col[keep]
    new_crow = torch.zeros(n + 1, dtype=torch.long, device=matrix.device)
    new_crow[1:] = torch.cumsum(torch.bincount(kept_row, minlength=n), dim=0)

    return torch.sparse_csr_tensor(
        new_crow,
        kept_col,
        torch.ones_like(kept_col, dtype=torch.bool),
        size=(n, n),
        check_invariants=False,
    )


def sparse_strength_graph(
    distance: torch.Tensor, fine_mask: torch.Tensor, theta_ad: float = 0.5
) -> torch.Tensor:
    """Prune a sparse algebraic-distance matrix into the strength graph ``M_d`` ([AD11] eq. 4.4).

    Sparse-CSR sibling of
    ``preconditioners.implementations.amg._algebraic_distance.strength_graph``:
    entry ``(i, j)`` is kept iff both ``i`` and ``j`` are fine points
    (``fine_mask[i] == fine_mask[j] == True``) and ``distance``'s stored
    value there exceeds ``theta_ad`` times row ``i``'s strongest connection
    to any node - evaluated only at ``distance``'s already-stored pattern
    positions, via ``sparse_row_max`` for the per-row threshold instead of
    a dense ``distance.max(dim=1)``.

    Args:
        distance (torch.Tensor): Sparse CSR algebraic-distance matrix ``r``
            (as returned by ``kernels.algebraic_distance.
            sparse_algebraic_distance``), shape ``(n, n)``.
        fine_mask (torch.Tensor): Boolean mask, shape ``(n,)``, ``True`` at
            ``F``-points.
        theta_ad (float): Strength threshold theta_ad in (0, 1) (paper
            default ``0.5``).

    Returns:
        torch.Tensor: Sparse CSR boolean strength graph, shape ``(n, n)``.

    Raises:
        ValueError: If ``distance`` is not sparse CSR.
    """
    _require_csr(distance, "sparse_strength_graph")
    n = distance.shape[0]
    row = _expand_row_index(distance)
    col = distance.col_indices()
    values = distance.values()

    strongest = sparse_row_max(distance)
    keep = (values > theta_ad * strongest[row]) & fine_mask[row] & fine_mask[col]

    kept_row = row[keep]
    kept_col = col[keep]
    crow = torch.zeros(n + 1, dtype=torch.long, device=distance.device)
    crow[1:] = torch.cumsum(torch.bincount(kept_row, minlength=n), dim=0)

    return torch.sparse_csr_tensor(
        crow,
        kept_col,
        torch.ones_like(kept_col, dtype=torch.bool),
        size=(n, n),
        check_invariants=False,
    )
