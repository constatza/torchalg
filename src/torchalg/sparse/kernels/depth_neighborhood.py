"""Sparse boolean matrix power (depth-``d`` graph adjacency) for Bootstrap AMG.

Sparse-CSR sibling of
``preconditioners.implementations.amg._graph.depth_neighborhood`` (dense;
kept unmodified for comparison). The dense version computes the depth-``d``
off-diagonal adjacency pattern via repeated *dense* boolean matmul over an
``(n, n)`` tensor - exactly the ``O(n^2)`` cost this sparse sibling exists to
avoid. Depth 1 needs no matmul at all: it is directly ``matrix``'s own
off-diagonal nonzero pattern, read straight from ``col_indices()``/
``crow_indices()``. Depth ``> 1`` repeats a sparse-sparse boolean matmul
(COO, mirroring ``galerkin.form_sparse_sparse``'s documented CSR->COO
transpose/matmul workaround), collapsing each product's values back to a
0/1 pattern after every multiply so fill-in never accumulates numeric
magnitude, only pattern growth.

Consumed by the sparse Bootstrap-AMG algebraic-distance kernel
(``kernels.algebraic_distance``) for both the depth-``d`` neighborhood
``V_i`` ([AD11] eq. 4.2) and the LS-ring candidate neighborhood (depth
``d + 2``, [AD11] eq. 4.5).
"""

from __future__ import annotations

import torch

from .triangular import _expand_row_index, _require_csr


def sparse_depth_neighborhood(matrix: torch.Tensor, depth: int) -> torch.Tensor:
    """Sparse CSR boolean off-diagonal adjacency of ``matrix`` raised to ``depth``.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix ``A``, shape ``(n, n)``.
        depth (int): Search depth ``d``, ``>= 1``.

    Returns:
        torch.Tensor: Sparse CSR boolean neighborhood matrix, shape ``(n, n)``.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR.
    """
    _require_csr(matrix, "sparse_depth_neighborhood")
    n = matrix.shape[0]
    row = _expand_row_index(matrix)
    col = matrix.col_indices()
    off_diagonal = row != col

    pattern = torch.sparse_coo_tensor(
        torch.stack([row[off_diagonal], col[off_diagonal]]),
        torch.ones(int(off_diagonal.sum()), dtype=matrix.values().dtype, device=matrix.device),
        size=(n, n),
        check_invariants=False,
    ).coalesce()

    reachable = pattern
    for _ in range(depth - 1):
        product = torch.sparse.mm(reachable, pattern).coalesce()
        reachable = torch.sparse_coo_tensor(
            product.indices(),
            torch.ones_like(product.values()),
            size=(n, n),
            check_invariants=False,
        ).coalesce()

    indices = reachable.indices()
    off_diagonal_final = indices[0] != indices[1]
    final_indices = indices[:, off_diagonal_final]
    return (
        torch.sparse_coo_tensor(
            final_indices,
            torch.ones(final_indices.shape[1], dtype=torch.bool, device=matrix.device),
            size=(n, n),
            check_invariants=False,
        )
        .coalesce()
        .to_sparse_csr()
    )
