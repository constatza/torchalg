"""Sparse-native strength-of-connection and aggregation for SA-AMG coarsening.

Companion to the dense
``preconditioners.implementations.amg._aggregation.strength_of_connection``/
``standard_aggregation`` (kept unmodified for comparison - see
``docs/plan.md``'s "Correction: strength-of-connection/aggregation must be
sparse-native too"): ``strength_of_connection`` only ever touches ``A``'s own
existing nonzero entries (no multi-hop neighborhood, no fill-in), so a sparse
version is a pure vectorized gather/compare over CSR's flat arrays - O(nnz),
no ``(n, n)`` dense intermediate, no Python loop. ``standard_aggregation``'s
three-pass greedy algorithm is inherently sequential regardless of storage
format (a genuinely new, independent transcription here, not a port, per the
same "new code, not shared code" precedent this plan uses for IC0 - the only
thing sparse storage changes is how the per-node neighbor lists are built:
directly from CSR's row-grouped ``col_indices()``, replacing the dense
version's O(n^2) ``torch.nonzero()`` scan.

References:
    - Vanek, P., Mandel, J., & Brezina, M. (1996). Algebraic multigrid by
      smoothed aggregation for second and fourth order elliptic problems.
      Computing, 56(3), 179-196.
"""

from __future__ import annotations

import torch

from .diagonal import sparse_diagonal
from .rowscale import sparse_row_scale

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
        new_crow, kept_col, torch.ones_like(kept_col, dtype=torch.bool), size=(n, n)
    )


def sparse_standard_aggregation(strength: torch.Tensor) -> torch.Tensor:
    """Assign each node to an aggregate via VMB96's three-pass algorithm, sparse input.

    Identical three-pass algorithm to the dense
    ``_aggregation.standard_aggregation`` (see that function's docstring for
    the full description of each pass); only the neighbor-list construction
    changes, reading CSR's already row-grouped ``col_indices()`` directly
    instead of scanning a dense boolean matrix.

    Args:
        strength (torch.Tensor): Sparse CSR boolean strength-of-connection
            matrix, shape ``(n, n)``. Must be symmetric (as produced by
            ``sparse_strength_of_connection`` on a symmetric matrix).

    Returns:
        torch.Tensor: Long tensor of length n where entry i is the
            aggregate index of node i, or ``-1`` if node i is isolated.
    """
    n = strength.shape[0]
    crow = strength.crow_indices().tolist()
    col = strength.col_indices().tolist()
    neighbor_lists: list[list[int]] = [col[crow[i] : crow[i + 1]] for i in range(n)]

    isolated_marker = -n
    state = [0] * n  # 0 = unmarked
    next_aggregate = 1

    # Pass 1.
    for i in range(n):
        if state[i]:
            continue
        neighbors = neighbor_lists[i]
        if not neighbors:
            state[i] = isolated_marker
            continue
        has_aggregated_neighbor = any(state[j] for j in neighbors)
        if not has_aggregated_neighbor:
            state[i] = next_aggregate
            for j in neighbors:
                state[j] = next_aggregate
            next_aggregate += 1

    # Pass 2.
    for i in range(n):
        if state[i]:
            continue
        for j in neighbor_lists[i]:
            if state[j] > 0:
                state[i] = -state[j]
                break

    next_aggregate -= 1

    # Pass 3.
    for i in range(n):
        current = state[i]
        if current != 0:
            if current > 0:
                state[i] = current - 1
            elif current == isolated_marker:
                state[i] = -1
            else:
                state[i] = -current - 1
            continue
        state[i] = next_aggregate
        for j in neighbor_lists[i]:
            if state[j] == 0:
                state[j] = next_aggregate
        next_aggregate += 1

    return torch.tensor(state, dtype=torch.long, device=strength.device)


def sparse_piecewise_constant_prolongation(
    aggregate: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Build the sparse piecewise-constant tentative prolongation P0.

    One nonzero (1.0) per assigned row, at the column of that row's
    aggregate - a scatter, cheap to construct sparse from the start rather
    than building dense and converting. An isolated node (aggregate index
    ``-1``, see ``sparse_standard_aggregation``) gets no stored entry in its
    row, matching the dense reference's all-zero row.

    Args:
        aggregate (torch.Tensor): Long tensor of length n mapping each fine
            node to its aggregate index, or ``-1`` if isolated.
        dtype (torch.dtype): Floating dtype for the returned matrix.

    Returns:
        torch.Tensor: Sparse CSR indicator matrix, shape ``(n, n_coarse)``.
    """
    n = aggregate.shape[0]
    n_coarse = int(aggregate.max().item()) + 1
    assigned = aggregate >= 0
    rows = torch.arange(n, device=aggregate.device)[assigned]
    cols = aggregate[assigned]
    values = torch.ones(rows.shape[0], dtype=dtype, device=aggregate.device)
    indices = torch.stack([rows, cols])
    return torch.sparse_coo_tensor(indices, values, size=(n, n_coarse)).to_sparse_csr()


def sparse_smoothed_prolongation(
    matrix: torch.Tensor, tentative: torch.Tensor, omega: float | torch.Tensor
) -> torch.Tensor:
    """Apply one Jacobi smoothing step to the tentative prolongation P0, sparse input.

    Computes ``P = (I - omega * D^{-1} A) P0`` (Vanek et al. 1996, Eq. 3.2) -
    identical formula to the dense ``_aggregation.smoothed_prolongation``,
    via sparse-sparse matmul (``A @ P0``) and per-row scaling
    (``sparse_row_scale``) instead of dense broadcasting.

    Args:
        matrix (torch.Tensor): Sparse CSR fine-grid matrix ``A``, shape
            ``(n, n)``.
        tentative (torch.Tensor): Sparse CSR piecewise-constant tentative
            prolongation P0, shape ``(n, n_coarse)``.
        omega (float | torch.Tensor): Jacobi damping factor omega.

    Returns:
        torch.Tensor: Sparse CSR smoothed prolongation matrix ``P``, shape
            ``(n, n_coarse)``.
    """
    diag = sparse_diagonal(matrix)
    diag_safe = torch.where(
        diag.abs() > _NEAR_ZERO_DIAGONAL_TOL,
        diag,
        torch.ones_like(diag),
    )
    update = matrix @ tentative
    scaled_update = sparse_row_scale(update, omega / diag_safe)
    # CSR - CSR has no kernel in this torch build ("unsupported tensor
    # layout: SparseCsr"); COO subtraction does, and round-trips losslessly.
    difference = tentative.to_sparse_coo() - scaled_update.to_sparse_coo()
    return difference.to_sparse_csr()
