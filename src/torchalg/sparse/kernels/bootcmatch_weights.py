"""BootCMatch edge-weight kernel for graph-weighted-matching coarsening, sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.amg._bootcmatch_weights.bootcmatch_edge_weights``
(dense; kept unmodified for comparison - see that module's own docstring for
the formula derivation). Reformulates the same [BootCMatch18] eq. 10
computation as a gather over stored edges only (the same idiom as
``kernels.algebraic_distance.sparse_algebraic_distance``), then removes the
degenerate edges from the returned pattern via the same "filter then rebuild
``crow`` via ``cumsum(bincount)``" recipe already used by
``kernels.strength.sparse_strength_graph`` - no dense ``(n, n)`` intermediate
anywhere.

References:
    - D'Ambra, P., Filippone, S., & Vassilevski, P. S. (2018). BootCMatch: A
      Software Package for Bootstrap AMG Based on Graph Weighted Matching.
      ACM Transactions on Mathematical Software, 44(4). Cited as
      [BootCMatch18]: eq. 10 (edge weight), its stated degeneracy guards.
"""

from __future__ import annotations

import torch

from .diagonal import sparse_diagonal
from .triangular import _expand_row_index, _require_csr

__all__ = ["bootcmatch_edge_weights"]


def bootcmatch_edge_weights(
    w: torch.Tensor, matrix: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-edge BootCMatch weights ``a_hat_ij`` ([BootCMatch18] eq. 10), sparse CSR.

    Args:
        w (torch.Tensor): Smooth vector, shape ``(n,)``.
        matrix (torch.Tensor): Sparse CSR fine-grid matrix ``A``, shape
            ``(n, n)``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(weights, fine_only_mask)``.
        ``weights`` is sparse CSR, shape ``(n, n)``, with the same sparsity
        pattern as ``matrix`` minus its diagonal and minus every degenerate
        edge (``sqrt(w_i**2/a_ii + w_j**2/a_jj) < TOL``) - those positions
        are structurally absent, not merely zeroed. ``fine_only_mask`` has
        shape ``(n,)``, ``True`` where ``|w_i| < TOL``.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR.
    """
    _require_csr(matrix, "bootcmatch_edge_weights")
    n = matrix.shape[0]
    tol = torch.finfo(matrix.dtype).eps

    diag = sparse_diagonal(matrix)

    row = _expand_row_index(matrix)
    col = matrix.col_indices()
    values = matrix.values()

    is_edge = (row != col) & (values != 0)

    diag_i = diag[row]
    diag_j = diag[col]
    w_i = w[row]
    w_j = w[col]

    denominator = diag_i * w_i**2 + diag_j * w_j**2
    edge_weights = 1.0 - 2.0 * values * w_i * w_j / denominator

    degenerate = torch.sqrt(w_i**2 / diag_i + w_j**2 / diag_j) < tol

    keep = is_edge & ~degenerate
    kept_row = row[keep]
    kept_col = col[keep]
    kept_values = edge_weights[keep]

    crow = torch.zeros(n + 1, dtype=torch.long, device=matrix.device)
    crow[1:] = torch.cumsum(torch.bincount(kept_row, minlength=n), dim=0)

    weights = torch.sparse_csr_tensor(
        crow,
        kept_col,
        kept_values,
        size=(n, n),
        check_invariants=False,
    )

    fine_only_mask = w.abs() < tol

    return weights, fine_only_mask
