"""Per-row scaling of a sparse CSR matrix by a dense vector.

Dense broadcasting (``scale.unsqueeze(1) * matrix``) has no sparse-CSR
equivalent (``sparse_mask()`` rejects the shape mismatch) - this multiplies
only the stored values, keeping the sparsity pattern fixed. Consumed by
``_jacobi_omega.scaled_by_inverse_diagonal`` (``D^-1 A``, shared by
``JacobiSmoother`` and prolongation smoothing) and by
``torchalg.sparse.aggregation``'s smoothed-prolongation step.
"""

from __future__ import annotations

import torch


def sparse_row_scale(matrix: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Scale each row ``i`` of a sparse CSR matrix by ``scale[i]``.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix, shape ``(n, m)``.
        scale (torch.Tensor): Dense per-row scale, shape ``(n,)``.

    Returns:
        torch.Tensor: Sparse CSR matrix with the same sparsity pattern as
            ``matrix``, values scaled per row.
    """
    crow = matrix.crow_indices()
    row_nnz = crow[1:] - crow[:-1]
    row_index = torch.repeat_interleave(
        torch.arange(matrix.shape[0], device=matrix.device), row_nnz
    )
    scaled_values = matrix.values() * scale[row_index]
    return torch.sparse_csr_tensor(
        crow,
        matrix.col_indices(),
        scaled_values,
        size=matrix.shape,
        check_invariants=False,
    )
