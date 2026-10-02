"""Symmetric strength of connection on a (possibly block) node graph, sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.amg._node_strength.node_strength`` (dense;
kept unmodified for comparison). Reimplemented directly rather than
delegating to ``torchalg.sparse.kernels.strength.sparse_strength_of_connection``
even though the two are mathematically equivalent for ``dofs_per_node=1``
when every diagonal entry is nonzero (``|a_ij| >= theta*sqrt(|a_ii|*|a_jj|)``
squares to the same inequality as ``node_strength``'s ``a_ij^2 >=
theta^2*|a_ii|*|a_jj|``): the dense formula never divides, so a zero
diagonal entry leaves its row unconditionally "strong" wherever the
coupling itself is nonzero, whereas ``sparse_strength_of_connection``'s
normalized-ratio form clamps a zero diagonal to 1.0 before dividing - a
different result in that edge case. Reimplementing directly (no division
anywhere) reproduces ``node_strength``'s exact behavior, including that
edge case, rather than an approximation that only agrees when the matrix
is well-posed.

For ``dofs_per_node > 1``, the needed quantity per node-pair ``(I, J)`` is
the block Frobenius-norm-squared ``sum over (i in I, j in J) of a_ij^2`` -
computed with zero Python loop via sparse COO's own duplicate-index
summation (``torch.sparse_coo_tensor(...).coalesce()``), not a block-CSR
storage format (which ``torch.sparse`` has no native support for and which
this computation does not actually need).
"""

from __future__ import annotations

import torch

from torchalg.sparse.kernels.triangular import _expand_row_index, _require_csr


def sparse_node_strength(matrix: torch.Tensor, dofs_per_node: int, theta: float) -> torch.Tensor:
    """Boolean strong-connection graph between nodes (diagonal excluded), sparse CSR.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix, shape ``(n_dofs, n_dofs)``.
        dofs_per_node (int): Block size (1 for a scalar matrix).
        theta (float): Strength threshold, at least 0.

    Returns:
        torch.Tensor: Sparse CSR boolean ``(n_nodes, n_nodes)`` strength matrix.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR, or if ``theta`` is negative.
    """
    _require_csr(matrix, "sparse_node_strength")
    if theta < 0:
        raise ValueError("expected a positive theta")

    n_dofs = matrix.shape[0]
    n_nodes = n_dofs // dofs_per_node
    row = _expand_row_index(matrix)
    col = matrix.col_indices()
    values = matrix.values()

    node_row = row // dofs_per_node
    node_col = col // dofs_per_node

    # Block Frobenius-norm-squared per node-pair: sparse COO's own
    # coalesce() sums duplicate (node_row, node_col) entries, which is
    # exactly `sum over the block of a_ij^2` - node[I,J]**2 directly, no
    # sqrt needed except to recover the diagonal blocks' own `node[I,I]`
    # below.
    block_sq_sum = torch.sparse_coo_tensor(
        torch.stack([node_row, node_col]),
        values**2,
        size=(n_nodes, n_nodes),
        check_invariants=False,
    ).coalesce()
    block_row, block_col = block_sq_sum.indices()
    block_values = block_sq_sum.values()

    diag_sq = torch.zeros(n_nodes, dtype=values.dtype, device=values.device)
    on_diag = block_row == block_col
    diag_sq[block_row[on_diag]] = block_values[on_diag]
    diag = torch.sqrt(diag_sq)

    threshold = theta**2 * diag[block_row] * diag[block_col]
    strong = (block_values >= threshold) & (block_values != 0) & (block_row != block_col)

    kept_row = block_row[strong]
    kept_col = block_col[strong]
    # `coalesce()`'s indices are lexicographically sorted (row-major), and a
    # boolean-mask filter preserves relative order, so `kept_row`/
    # `kept_col` are already in valid per-row-ascending CSR order.
    crow = torch.zeros(n_nodes + 1, dtype=torch.long, device=matrix.device)
    crow[1:] = torch.cumsum(torch.bincount(kept_row, minlength=n_nodes), dim=0)

    return torch.sparse_csr_tensor(
        crow,
        kept_col,
        torch.ones_like(kept_col, dtype=torch.bool),
        size=(n_nodes, n_nodes),
        check_invariants=False,
    )
