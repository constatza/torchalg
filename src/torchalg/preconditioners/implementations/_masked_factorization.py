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


def _indices_per_row(pattern: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Split a boolean ``(n, n)`` pattern into per-row column-index tensors.

    One ``nonzero`` and one host transfer of the ``n`` row counts for the whole
    pattern, so the elimination loops below index with ready-made views and
    never synchronize per pivot.
    """
    counts = pattern.sum(dim=1).tolist()
    columns = pattern.nonzero()[:, 1]
    return torch.split(columns, counts)


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

    Evaluated right-looking (KJI order, the form ``dense_ic0`` uses): at
    pivot ``k`` the stored entries of column ``k`` below the diagonal are
    scaled by the pivot, then one masked rank-1 update
    ``F[i, j] -= L[i, k] * U[k, j]`` is applied to the block spanned by those
    rows and by the stored entries of row ``k`` right of the diagonal. Every
    entry receives the same updates, in the same order and with the same
    operands, as the scalar IKJ triple loop, so results are bit-identical;
    only the structurally zero ``i, j`` pairs are never visited, which makes
    the cost ``O(sum_k nnz(L[:, k]) * nnz(U[k, :]))`` instead of
    ``O(n^3)`` Python-level scalar steps. (Entries outside the pattern are
    exactly zero, so skipping them only differs from the scalar loop on
    breakdown inputs where a zero pivot already produced ``inf``/``nan``.)

    Args:
        matrix (torch.Tensor): Dense system matrix ``A``, shape ``(n, n)``.

    Returns:
        torch.Tensor: Combined ``L``/``U`` factor tensor, shape ``(n, n)``.
    """
    n = matrix.shape[0]
    mask = matrix != 0
    factors = matrix.clone()
    rows_below = _indices_per_row(mask.tril(-1).T)
    columns_right = _indices_per_row(mask.triu(1))
    for k in range(n - 1):
        rows, columns = rows_below[k], columns_right[k]
        if rows.numel() == 0:
            continue
        lower_column = factors[rows, k] / factors[k, k]
        factors[rows, k] = lower_column
        if columns.numel() == 0:
            continue
        block = (rows.unsqueeze(1), columns.unsqueeze(0))
        update = torch.outer(lower_column, factors[k, columns])
        update.masked_fill_(~mask[block], 0)
        factors[block] -= update
    return factors


def ic0_sparsity_mask(matrix: torch.Tensor, threshold: float) -> torch.Tensor:
    """Compute IC(0)'s target sparsity pattern without factorizing.

    This is the structural half of ``dense_ic0``, split out because it
    answers "does IC(0) stay non-filling at this threshold?" on its own:
    the pattern is fixed by ``matrix`` and ``threshold`` alone (an ``&`` of
    two boolean tensors), before any elimination, square root, or pivot
    check runs. Checking it directly is ``O(n^2)`` with no loop and no
    breakdown risk, versus ``dense_ic0``'s ``O(n^3)`` elimination that can
    raise ``ValueError`` on an indefinite pivot - a question about pattern
    shouldn't have to survive numerics to get answered.

    Args:
        matrix (torch.Tensor): Symmetric positive-definite system matrix
            ``A``, shape ``(n, n)``.
        threshold (float): Drop tolerance - entries with ``|value| <=
            threshold`` are treated as zero and excluded from the sparsity
            pattern.

    Returns:
        torch.Tensor: Boolean mask, shape ``(n, n)``, true where the IC(0)
            factor is allowed to be non-zero (lower triangle, ``|A_ij| >
            threshold``).
    """
    n = matrix.shape[0]
    row_index = torch.arange(n, device=matrix.device).unsqueeze(1)
    col_index = torch.arange(n, device=matrix.device).unsqueeze(0)
    lower_triangle = row_index >= col_index
    return lower_triangle & (matrix.abs() > threshold)


def dense_ic0(matrix: torch.Tensor, threshold: float) -> torch.Tensor:
    """Compute a dense zero-fill incomplete Cholesky (IC(0)) factor ``L``.

    For a symmetric positive-definite ``matrix``, computes a lower
    triangular ``L`` such that ``L @ L.T`` approximates ``matrix``, where
    ``L`` keeps the same sparsity pattern as the lower triangle of
    ``matrix`` (entries with ``|value| <= threshold`` are treated as zero
    and never filled in).

    Right-looking Cholesky-Crout: at each step ``k``, the stored entries
    of column ``k`` below the diagonal are scaled by the (square-rooted)
    pivot, then the rank-1 update ``L[i, j] -= L[i, k] * L[j, k]`` is applied
    as one masked outer product on the block spanned by those rows. Entries
    outside the original sparsity pattern are exactly zero from the initial
    mask and are never visited, so the cost per pivot is
    ``O(nnz(L[:, k])^2)`` instead of the trailing block's ``O((n-k)^2)``;
    the per-row index lists come from a single ``nonzero`` up front. The
    remaining ``(i, j)``-in-pattern condition is applied by masking the
    block (in place with ``masked_fill_``, on the outer product's own
    buffer) before subtracting. Same elimination order and operands as a
    scalar IKJ loop, so results are bit-identical.

    The positive-pivot check runs once after the loop (see
    ``_raise_on_breakdown``) rather than synchronizing with the device at
    every pivot.

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
    sparsity_mask = ic0_sparsity_mask(matrix, threshold)
    rows_below = _indices_per_row(sparsity_mask.tril(-1).T)

    factor = torch.where(sparsity_mask, torch.tril(matrix), torch.zeros_like(matrix))
    pivots = torch.empty(n, dtype=matrix.dtype, device=matrix.device)

    for k in range(n):
        pivot = factor[k, k]
        pivots[k] = pivot
        diagonal = torch.sqrt(pivot)
        factor[k, k] = diagonal
        rows = rows_below[k]
        if rows.numel() == 0:
            continue

        column = factor[rows, k] / diagonal
        factor[rows, k] = column

        block = (rows.unsqueeze(1), rows.unsqueeze(0))
        update = torch.outer(column, column)
        update.masked_fill_(~sparsity_mask[block], 0)
        factor[block] -= update

    _raise_on_breakdown(pivots)
    return factor


def _raise_on_breakdown(pivots: torch.Tensor) -> None:
    """Raise ``ValueError`` at the first non-positive IC(0) pivot, in one host read.

    A non-positive pivot makes ``sqrt`` return ``nan`` (or ``0``) and the
    elimination simply carries on with garbage, so the check can wait for the
    end of the loop instead of synchronizing with the device at every pivot.
    The first offending index and its value are unaffected: every pivot before
    it was computed from valid data. A ``nan`` pivot (``nan`` input) is not
    flagged, matching the earlier per-pivot test ``pivot <= 0``.
    """
    non_positive = pivots <= 0
    if not bool(non_positive.any()):
        return
    k = int(non_positive.nonzero()[0])
    raise ValueError(
        f"IC(0) breakdown at pivot {k}: diagonal value {pivots[k].item()} is "
        "non-positive, so no real factor L exists for this matrix "
        "(Saad, Iterative Methods for Sparse Linear Systems, 2nd ed., "
        "Sec. 10.3) - not every SPD matrix admits an IC(0) factorization."
    )
