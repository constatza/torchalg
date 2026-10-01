"""Diagonal extraction for sparse CSR matrices.

``torch.diagonal`` has no sparse-CSR overload, so
``JacobiSmoother.smooth()`` (``preconditioners.implementations.amg.smoothers``)
needs this to dispatch on ``A.is_sparse_csr`` instead - see ``docs/plan.md``.
"""

from __future__ import annotations

import torch


def sparse_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """Extract the diagonal of a sparse CSR matrix without densifying it.

    Vectorized via a row-index expansion of ``crow_indices()`` (mirroring
    ``scripts/bench_sparse_vs_dense/operations.py``'s COO
    ``extract_inv_diag``): no Python loop, no ``(n, n)`` dense intermediate.
    A row with no stored diagonal entry (structural zero) reads back as
    ``0.0``, matching ``torch.diagonal``'s dense behavior.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix, shape ``(n, n)``.

    Returns:
        torch.Tensor: Dense ``(n,)`` diagonal.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR.
    """
    if matrix.layout != torch.sparse_csr:
        raise ValueError(
            f"sparse_diagonal requires a sparse CSR tensor, got layout {matrix.layout}"
        )

    n = matrix.shape[0]
    crow = matrix.crow_indices()
    col = matrix.col_indices()
    values = matrix.values()

    row_nnz = crow[1:] - crow[:-1]
    row_index = torch.repeat_interleave(torch.arange(n, device=matrix.device), row_nnz)
    on_diagonal = row_index == col

    diag = torch.zeros(n, dtype=matrix.dtype, device=matrix.device)
    diag[row_index[on_diagonal]] = values[on_diagonal]
    return diag
