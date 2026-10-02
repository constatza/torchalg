"""Sparse-native tentative prolongation kernel for SA-AMG coarsening.

A leaf primitive - no standalone breakdown/convergence story of its own,
consumed by the SA-AMG algorithm in
``torchalg.sparse.preconditioners.amg.aggregation``. See ``docs/plan.md``'s
"Correction: strength-of-connection/aggregation must be sparse-native too".
"""

from __future__ import annotations

import torch


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
    return torch.sparse_coo_tensor(
        indices, values, size=(n, n_coarse), check_invariants=False
    ).to_sparse_csr()


def sparse_interpolation_prolongation(
    row_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    """Build a sparse CSR prolongation from ``(row, col, value)`` triples, multiple per row.

    General COO-accumulate-then-CSR-convert builder - the sparse-native
    sibling of Bootstrap AMG's dense ``prolongation[i,
    coarse_index_cpu[interp_set]] = row`` scatter assignment
    (``preconditioners.implementations.amg.bootstrap``'s
    ``BAMGCoarsening._prolongation``): each fine row contributes at most
    ``caliber`` LS-fitted nonzeros, accumulated across a CPU-resident
    per-row loop into flat ``(row, col, value)`` triples, then built once
    here rather than ever materializing a dense ``(n, n_coarse)``
    intermediate. Unlike ``sparse_piecewise_constant_prolongation`` above
    (exactly one nonzero per row, already in row-ascending order by
    construction), triples here may arrive in any order and span multiple
    columns per row, so ``.coalesce()`` is needed before the CSR conversion
    to get the row-major-sorted storage CSR requires.

    Args:
        row_indices (torch.Tensor): Long tensor, fine-row index per triple.
        col_indices (torch.Tensor): Long tensor, coarse-column index per
            triple, same shape as ``row_indices``.
        values (torch.Tensor): Values per triple, same shape as
            ``row_indices``.
        size (tuple[int, int]): ``(n, n_coarse)`` shape of the result.

    Returns:
        torch.Tensor: Sparse CSR matrix, shape ``size``.
    """
    indices = torch.stack([row_indices, col_indices])
    return (
        torch.sparse_coo_tensor(indices, values, size=size, check_invariants=False)
        .coalesce()
        .to_sparse_csr()
    )
