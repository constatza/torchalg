"""Sparse-native SA-AMG aggregation algorithm: standard aggregation and prolongation smoothing.

Companion to the dense ``preconditioners.implementations.amg._aggregation``
module (kept unmodified for comparison - see ``docs/plan.md``'s "Correction:
strength-of-connection/aggregation must be sparse-native too"):
``standard_aggregation``'s three-pass greedy algorithm is inherently
sequential regardless of storage format (a genuinely new, independent
transcription here, not a port, per the same "new code, not shared code"
precedent this plan uses for IC0 - the only thing sparse storage changes is
how the per-node neighbor lists are built: directly from CSR's row-grouped
``col_indices()``, replacing the dense version's O(n^2) ``torch.nonzero()``
scan.

References:
    - Vanek, P., Mandel, J., & Brezina, M. (1996). Algebraic multigrid by
      smoothed aggregation for second and fourth order elliptic problems.
      Computing, 56(3), 179-196.
"""

from __future__ import annotations

import torch

from torchalg.sparse.kernels.diagonal import sparse_diagonal
from torchalg.sparse.kernels.rowscale import sparse_row_scale

_NEAR_ZERO_DIAGONAL_TOL = 1e-14
"""Diagonal entries with magnitude below this are treated as 1.0 when
normalizing (protects against division by near-zero) - matches the dense
``_aggregation.py`` constant."""

# TODO(parallel-algorithm): Add a distinct MIS-based aggregation coarsening
# strategy for tensor-parallel CPU/GPU setup. Do not vectorize or replace this
# VMB96 three-pass greedy algorithm: a parallel MIS aggregation has different
# selection semantics and must be exposed as a separate coarsening strategy,
# with convergence/operator-complexity validation rather than dense-output
# parity. Brannick et al., "Parallel Unsmoothed Aggregation Algebraic
# Multigrid Algorithms on GPUs," arXiv:1302.2547,
# https://arxiv.org/abs/1302.2547.


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
