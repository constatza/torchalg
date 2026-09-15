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

    Vectorized right-looking Cholesky-Crout: at each step ``k``, column
    ``k`` is scaled by its (square-rooted) pivot in one slice op, then the
    rank-1 update ``L[i, j] -= L[i, k] * L[j, k]`` is applied to the whole
    trailing block as a single masked outer product, rather than as scalar
    Python-level updates over ``i``/``j``. Entries outside the original
    sparsity pattern are exactly zero from the initial mask, and stay zero
    permanently (they are only ever combined with, or overwritten by, other
    exact zeros) - so ``factor[i, k]`` and ``factor[j, k]`` being outside the
    column-``k`` pattern falls out of the outer product for free, and the
    remaining ``(i, j)``-in-pattern condition is applied by masking the
    trailing block before subtracting. Same elimination order as a scalar
    IKJ loop, so results are bit-identical; only the inner two loops become
    one vectorized op each, per ``k``.

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
        pivot = factor[k, k]
        if pivot <= 0:
            raise ValueError(
                f"IC(0) breakdown at pivot {k}: diagonal value {pivot.item()} is "
                "non-positive, so no real factor L exists for this matrix "
                "(Saad, Iterative Methods for Sparse Linear Systems, 2nd ed., "
                "Sec. 10.3) - not every SPD matrix admits an IC(0) factorization."
            )
        factor[k, k] = torch.sqrt(pivot)
        if k + 1 >= n:
            break

        column = factor[k + 1 :, k] / factor[k, k]
        factor[k + 1 :, k] = column

        trailing_mask = sparsity_mask[k + 1 :, k + 1 :]
        update = torch.outer(column, column)
        factor[k + 1 :, k + 1 :] -= torch.where(trailing_mask, update, torch.zeros_like(update))

    return factor
