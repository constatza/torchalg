"""Residual-based "adaptive relaxation" test-vector correction, sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.amg._least_squares.lsr_correction`` (dense;
kept unmodified for comparison). ``matrix[target_rows] @ test_vectors`` -
sparse-CSR row-fancy-indexing, not reliably supported - is rewritten as
``(matrix @ test_vectors)[target_rows]``: selecting rows of a matrix
product commutes with selecting rows of the left operand, so this is exact,
not an approximation, and avoids sparse row-slicing entirely (``matrix @
test_vectors`` is a plain sparse-dense matmul, fully supported). This is
also already the pattern
``preconditioners.implementations.amg.bootstrap._fit_vectors`` uses at its
own call site today (``(A @ test_vectors)[fine_points]``), not a novel
rewrite.
"""

from __future__ import annotations

import torch

from .diagonal import sparse_diagonal

_NEAR_ZERO_DIAGONAL_TOL = 1e-14
"""Matches the dense sibling's tolerance - see its docstring."""


def sparse_lsr_correction(
    test_vectors: torch.Tensor,
    matrix: torch.Tensor,
    target_rows: torch.Tensor,
) -> torch.Tensor:
    """Residual-based "adaptive relaxation" correction ([STATUS14] eq. 3.2), sparse CSR.

    ``v_i^(kappa) <- v_i^(kappa) - (A v^(kappa))_i / a_ii``, applied only at
    ``target_rows``. Pure function: returns a new tensor, never mutates
    ``test_vectors``.

    Args:
        test_vectors (torch.Tensor): Test vectors ``V``, shape ``(n, k)``.
        matrix (torch.Tensor): Sparse CSR matrix ``A``, shape ``(n, n)``.
        target_rows (torch.Tensor): Long tensor of row indices to correct,
            shape ``(t,)``.

    Returns:
        torch.Tensor: New test vectors, shape ``(n, k)``, equal to
        ``test_vectors`` except at ``target_rows``.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR.
    """
    diagonal = sparse_diagonal(matrix)[target_rows]
    diagonal_safe = torch.where(
        diagonal.abs() > _NEAR_ZERO_DIAGONAL_TOL, diagonal, torch.ones_like(diagonal)
    )
    residual_at_targets = (matrix @ test_vectors)[target_rows]
    corrected = test_vectors.clone()
    corrected[target_rows] = test_vectors[
        target_rows
    ] - residual_at_targets / diagonal_safe.unsqueeze(1)
    return corrected
