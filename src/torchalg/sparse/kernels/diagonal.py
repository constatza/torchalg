"""Diagonal extraction for sparse CSR matrices.

``torch.diagonal`` has no sparse-CSR overload, so the sparse smoothers,
strength-of-connection and prolongation kernels read the diagonal through this
function.
"""

from __future__ import annotations

import torch


def sparse_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """Extract the diagonal of a sparse CSR matrix without densifying it.

    Vectorized via a row-index expansion of ``crow_indices()`` (mirroring
    ``benchmarks.sparse_dense.operations``' COO
    ``extract_inv_diag``): no Python loop, no ``(n, n)`` dense intermediate.
    A row with no stored diagonal entry (structural zero) reads back as
    ``0.0``, matching ``torch.diagonal``'s dense behavior. The scatter masks
    off-diagonal values to zero instead of boolean-indexing them out, so the
    call never synchronizes with the device (boolean indexing and
    ``repeat_interleave`` without ``output_size`` both read a count back).

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
    row_index = torch.repeat_interleave(
        torch.arange(n, device=matrix.device), row_nnz, output_size=col.numel()
    )
    diagonal_values = torch.where(row_index == col, values, torch.zeros_like(values))
    return torch.zeros(n, dtype=matrix.dtype, device=matrix.device).scatter_add(
        0, row_index, diagonal_values
    )
