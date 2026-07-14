"""Dense masked incomplete-factorization kernels (ILU(0), IC(0)).

Pure functions with no preconditioner-class state: given a dense system
matrix, each returns a dense factor (or combined ``L``/``U`` factor) tensor
that only fills in entries where ``A`` was already non-zero — the "0" in
ILU(0)/IC(0) — mirroring the reference's `scipy.sparse.linalg.spilu`-backed
ILU and numba-JIT IC(0) kernels without any sparse or JIT machinery.

Isolated here, separate from the ``Preconditioner`` classes that call them
(``ilu.py``, ``ic0.py``), per the project's helpers-isolated-in-their-own-file
convention: the classes stay declarative orchestration; the elimination
loops (inherently branchy - "only touch entries in the original sparsity
pattern") live here.

References:
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd ed.,
      Algorithm 10.4 (ILU(0), IKJ variant, no pivoting - natural ordering).
"""

from __future__ import annotations

import torch


def dense_ilu0(matrix: torch.Tensor) -> torch.Tensor:
    """Compute a dense ILU(0) factorization: combined ``L``/``U`` factors.

    Standard Doolittle-style Gaussian elimination without pivoting (natural
    ordering), restricted to ``matrix``'s original non-zero pattern: any
    entry that would fill in outside that pattern is simply never written,
    which is exactly the "incomplete" in ILU(0) (Saad Algorithm 10.4, IKJ
    variant).

    The result packs both factors into a single ``(n, n)`` tensor: the
    strictly-lower part is ``L`` (with an implicit unit diagonal), and the
    upper part (including the diagonal) is ``U``.

    Args:
        matrix (torch.Tensor): Dense system matrix ``A``, shape ``(n, n)``.

    Returns:
        torch.Tensor: Combined ``L``/``U`` factor tensor, shape ``(n, n)``.
    """
    n = matrix.shape[0]
    mask = matrix != 0
    factors = matrix.clone()
    for i in range(1, n):
        for k in range(i):
            if not mask[i, k]:
                continue
            factors[i, k] = factors[i, k] / factors[k, k]
            for j in range(k + 1, n):
                if mask[i, j]:
                    factors[i, j] = factors[i, j] - factors[i, k] * factors[k, j]
    return factors


def dense_ic0(matrix: torch.Tensor, threshold: float) -> torch.Tensor:
    """Compute a dense zero-fill incomplete Cholesky (IC(0)) factor ``L``.

    For a symmetric positive-definite ``matrix``, computes a lower
    triangular ``L`` such that ``L @ L.T`` approximates ``matrix``, where
    ``L`` keeps the same sparsity pattern as the lower triangle of
    ``matrix`` (entries with ``|value| <= threshold`` are treated as zero
    and never filled in).

    Direct torch port of the reference's numba-JIT IC(0) kernel: at each
    step ``k``, the diagonal is scaled by its square root, column ``k`` is
    scaled by the new diagonal, and the rank-1 update
    ``L[i, j] -= L[i, k] * L[j, k]`` is applied only where ``(i, j)``,
    ``(i, k)`` and ``(j, k)`` are all within the original sparsity pattern
    - no fill-in beyond it.

    Args:
        matrix (torch.Tensor): Symmetric positive-definite system matrix
            ``A``, shape ``(n, n)``.
        threshold (float): Drop tolerance - entries with ``|value| <=
            threshold`` are treated as zero and excluded from the sparsity
            pattern.

    Returns:
        torch.Tensor: Lower triangular incomplete Cholesky factor ``L``,
            shape ``(n, n)``.
    """
    n = matrix.shape[0]
    row_index = torch.arange(n, device=matrix.device).unsqueeze(1)
    col_index = torch.arange(n, device=matrix.device).unsqueeze(0)
    lower_triangle = row_index >= col_index
    sparsity_mask = lower_triangle & (matrix.abs() > threshold)

    factor = torch.where(sparsity_mask, torch.tril(matrix), torch.zeros_like(matrix))

    for k in range(n):
        factor[k, k] = torch.sqrt(factor[k, k])

        for i in range(k + 1, n):
            if sparsity_mask[i, k]:
                factor[i, k] = factor[i, k] / factor[k, k]

        for i in range(k + 1, n):
            if not sparsity_mask[i, k]:
                continue
            pivot_column_value = factor[i, k].clone()
            for j in range(k + 1, i + 1):
                if sparsity_mask[j, k] and sparsity_mask[i, j]:
                    factor[i, j] = factor[i, j] - pivot_column_value * factor[j, k]

    return factor
