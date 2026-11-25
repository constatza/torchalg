"""Sparse boolean matrix power (depth-``d`` graph adjacency) for Bootstrap AMG.

Sparse-CSR sibling of
``preconditioners.implementations.amg._graph.depth_neighborhood`` (dense;
kept unmodified for comparison). The dense version computes the depth-``d``
off-diagonal adjacency pattern via repeated *dense* boolean matmul over an
``(n, n)`` tensor - exactly the ``O(n^2)`` cost this sparse sibling exists to
avoid. Depth 1 needs no matmul at all: it is directly ``matrix``'s own
off-diagonal nonzero pattern, read straight from ``col_indices()``/
``crow_indices()``. Depth ``> 1`` repeats a sparse-sparse CSR matmul, collapsing each product's
values back to 1 after every multiply so fill-in never accumulates numeric
magnitude, only pattern growth. The CSR product leaves each row's columns
unsorted and keeps the diagonal, so one final key sort and diagonal drop
restore the ascending-column, off-diagonal CSR the consumers rely on (row
lookups by ``searchsorted``).

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

    Cost:
        Per call, ``z = nnz(matrix)``, ``z_d`` the stored size of the depth-``d``
        reach. ``depth - 1`` CSR boolean products ``reach @ pattern`` plus one
        ``O(z_d log z_d)`` key sort. ``W = 0`` flops of use (the products run on
        ones), ``I`` is the SpGEMM index work, ``L = O(depth)`` launches. The former
        COO path coalesced (sorted) after every product and again at the end;
        this one sorts once.

        Per product ``k`` the multiply-add count is
        ``2 sum_{r_ij != 0} nnz(pattern_j:)`` over the current reach ``r``, so
        ``I = O(z_d log z_d)`` is dominated by the sort at depth 2 and above
        and by the products at larger ``depth``, and ``Q`` is the bytes of
        the reach, the pattern and the products (``w`` bytes per index; no
        values are needed beyond the ones the product reads). ``D = (depth - 1)
        (ceil(log2 (m_r m_p)) + 1) + O(log z_d)`` for the products and the sort,
        and ``S = 0``. The reach grows toward ``n^2`` with ``depth``
        (13% of ``n^2`` at depth 3 on a 3D elasticity matrix of ``n = 2946``),
        so ``Q`` and ``I`` grow with the fill, not with ``z``.
        Roofline: ``W/Q`` is below 1 flop per byte (memory-bound).
        Work-depth: ``W/D`` is large (rows of a product and the sort are
        parallel).

    Args:
        matrix (torch.Tensor): Sparse CSR matrix ``A``, shape ``(n, n)``.
        depth (int): Search depth ``d``, ``>= 1``.

    Returns:
        torch.Tensor: Sparse CSR boolean neighborhood matrix, shape ``(n, n)``,
        columns ascending within each row.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR.
    """
    _require_csr(matrix, "sparse_depth_neighborhood")
    n = matrix.shape[0]
    row = _expand_row_index(matrix)
    col = matrix.col_indices()
    off_diagonal = row != col

    pattern = _csr_of_ones(row[off_diagonal], col[off_diagonal], n, matrix.values().dtype)
    reachable = pattern
    for _ in range(depth - 1):
        product = reachable @ pattern
        reachable = torch.sparse_csr_tensor(
            product.crow_indices(),
            product.col_indices(),
            torch.ones_like(product.values()),
            size=(n, n),
            check_invariants=False,
        )

    reach_row = _expand_row_index(reachable)
    reach_col = reachable.col_indices()
    keep = reach_row != reach_col
    # int64 keys: the row * n product overflows the compact int32 index width
    keys = torch.sort(reach_row[keep].long() * n + reach_col[keep].long()).values
    return _csr_of_ones(keys // n, keys % n, n, torch.bool)


def _csr_of_ones(
    row: torch.Tensor,
    col: torch.Tensor,
    n: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """CSR ``(n, n)`` matrix of ones at the row-major ``(row, col)`` entries.

    Args:
        row (torch.Tensor): Row index per entry, non-decreasing.
        col (torch.Tensor): Column index per entry.
        n (int): Matrix dimension.
        dtype (torch.dtype): Value dtype.

    Returns:
        torch.Tensor: Sparse CSR tensor, one stored one per entry.
    """
    crow = torch.zeros(n + 1, dtype=row.dtype, device=row.device)
    crow[1:] = torch.cumsum(torch.bincount(row, minlength=n), 0)
    return torch.sparse_csr_tensor(
        crow,
        col,
        torch.ones(col.numel(), dtype=dtype, device=row.device),
        size=(n, n),
        check_invariants=False,
    )
