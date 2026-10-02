"""Incomplete LU (ILU(0)) preconditioner, sparse-CSR sibling.

Sparse-CSR counterpart of ``preconditioners.implementations.ilu
.ILUPreconditioner`` (dense; kept unmodified for comparison - see
``docs/plan.md``'s "Correction: dense and sparse must be separate
implementations, not an internal branch"). Row-oriented ("left-looking")
level-scheduled factorization - see ``sparse.preconditioners.ic0``'s module
docstring and ``docs/plan.md``'s "Correction: the row-oriented recurrence
sparse_ic0/sparse_ilu0 actually need" section for the shared design
rationale; ``_lookup``/``_build_lookup`` are reused from ``.ic0`` rather
than duplicated, since both factorizations need the identical global
sparse-element lookup primitive.

The Python loop is bounded by ``num_levels * max_row_degree_in_level``, not
``nnz``, same as ``.ic0``'s batching - see that module's docstring. ILU0's
own wrinkle: a row's L-side entries (``j < row``) need exactly ``d`` of the
row's own earlier columns at degree-position ``d`` (uniform across a
batch, like IC0), but its U-side entries (``j >= row``) each need the
row's *entire* L-side count (``idx_diag``), which differs row-to-row - so
U-side batches are padded to the batch's max L-count at that step and the
padding is masked out of the sum, rather than forcing one unified,
non-uniform-batch formula.
"""

from __future__ import annotations

import torch
from torch import nn

from torchalg.preconditioners.base import LinearPreconditioner, PreconditionerContext
from torchalg.sparse.kernels.triangular import _require_csr, level_schedule, triangular_solve

from .ic0 import LevelSchedule, _build_lookup, _lookup


def _expand_row_index(matrix: torch.Tensor) -> torch.Tensor:
    """Expand ``crow_indices()`` into a per-nonzero-entry row index.

    Same crow-expansion idiom as ``torchalg.sparse.kernels.triangular``'s
    private helper of the same name, duplicated here rather than imported
    across the kernels/preconditioners module boundary (matches the dense
    sibling's existing precedent for this exact idiom).

    Args:
        matrix (torch.Tensor): Sparse CSR matrix.

    Returns:
        torch.Tensor: int64, one row index per stored nonzero entry.
    """
    n = matrix.shape[0]
    crow = matrix.crow_indices()
    row_nnz = crow[1:] - crow[:-1]
    return torch.repeat_interleave(torch.arange(n, device=matrix.device), row_nnz)


def _csr_from_mask(matrix: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Rebuild a sparse CSR tensor from a boolean submask of ``matrix``'s own entries.

    Same crow-rebuild idiom as ``.ic0``'s ``_csr_from_boolean_submask``,
    duplicated here rather than imported, matching the dense sibling's
    precedent of not sharing this exact helper across its own ic0/ilu split.

    Args:
        matrix (torch.Tensor): Source sparse CSR matrix.
        mask (torch.Tensor): Boolean mask, same shape as ``matrix.values()``.

    Returns:
        torch.Tensor: Sparse CSR tensor containing only the masked-in entries.
    """
    n = matrix.shape[0]
    row_index = _expand_row_index(matrix)[mask]
    col = matrix.col_indices()[mask]
    values = matrix.values()[mask]

    counts = torch.bincount(row_index, minlength=n)
    crow = torch.zeros(n + 1, dtype=torch.int64, device=matrix.device)
    crow[1:] = torch.cumsum(counts, 0)
    return torch.sparse_csr_tensor(crow, col, values, size=matrix.shape, check_invariants=False)


def _materialize_unit_lower(combined: torch.Tensor) -> torch.Tensor:
    """Extract ``combined``'s strictly-lower part with an explicit unit diagonal.

    ``sparse_ilu0``'s combined factor packs ``L`` with an *implicit* unit
    diagonal - never actually stored as ``1.0``. ``triangular_solve`` has no
    unit-diagonal convention of its own; it divides by whatever diagonal is
    actually stored, so solving ``L``'s system against it requires
    materializing that unit diagonal as real stored entries first.

    Args:
        combined (torch.Tensor): Sparse CSR combined ``L``/``U`` factor.

    Returns:
        torch.Tensor: Sparse CSR lower-triangular matrix with ``combined``'s
            strictly-lower entries plus an explicit ``1.0`` diagonal.
    """
    n = combined.shape[0]
    row_index = _expand_row_index(combined)
    col = combined.col_indices()
    values = combined.values()

    strictly_lower = col < row_index
    lower_row = row_index[strictly_lower]
    lower_col = col[strictly_lower]
    lower_val = values[strictly_lower]

    diag_index = torch.arange(n, device=combined.device)
    diag_val = torch.ones(n, dtype=combined.dtype, device=combined.device)

    all_row = torch.cat([lower_row, diag_index])
    all_col = torch.cat([lower_col, diag_index])
    all_val = torch.cat([lower_val, diag_val])

    order = torch.argsort(all_row * n + all_col, stable=True)
    all_row, all_col, all_val = all_row[order], all_col[order], all_val[order]

    counts = torch.bincount(all_row, minlength=n)
    crow = torch.zeros(n + 1, dtype=torch.int64, device=combined.device)
    crow[1:] = torch.cumsum(counts, 0)
    return torch.sparse_csr_tensor(
        crow, all_col, all_val, size=combined.shape, check_invariants=False
    )


def sparse_ilu0(matrix: torch.Tensor) -> torch.Tensor:
    """Compute a sparse zero-fill incomplete LU (ILU(0)) combined factor.

    Row-oriented recurrence (see ``.ic0``'s module docstring), general (not
    necessarily symmetric):
        ``L[i,j] = (A[i,j] - sum_{k<j, k in pattern(i)} L[i,k]*U[k,j]) /
        U[j,j]`` for ``j < i``;
        ``U[i,j] = A[i,j] - sum_{k<i, k in pattern(i)} L[i,k]*U[k,j]`` for
        ``j >= i``.

    Packs both factors into one combined sparse CSR tensor, same convention
    as the dense sibling: the strictly-lower part is ``L`` (implicit unit
    diagonal, never stored); the diagonal-and-upper part is ``U``.

    Unlike ``sparse_ic0``, no breakdown check: the dense equivalence target
    divides unconditionally, so this does too.

    Args:
        matrix (torch.Tensor): Sparse CSR system matrix ``A``, shape
            ``(n, n)``. Uses ``A``'s own full pattern - no threshold, no
            lower-only filtering.

    Returns:
        torch.Tensor: Combined ``L``/``U`` sparse CSR factor tensor.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR.
    """
    _require_csr(matrix, "sparse_ilu0")

    n = matrix.shape[0]
    col = matrix.col_indices()
    crow = matrix.crow_indices()
    row_start = crow[:-1]
    row_end = crow[1:]
    degrees = row_end - row_start
    row_index_full = _expand_row_index(matrix)

    # Position of each row's own diagonal entry within that row (= count of
    # that row's strictly-lower stored entries, since CSR columns are
    # ascending and L-side entries therefore come first).
    is_lower = col < row_index_full
    idx_diag = torch.bincount(row_index_full[is_lower], minlength=n)
    diag_flat_pos = row_start + idx_diag

    keys = _build_lookup(row_index_full, col, n)

    # See ``.ic0.sparse_ic0``'s identical comment: ``values`` is rebuilt
    # out-of-place at every step (``torch.scatter``, never ``scatter_``/
    # indexed assignment), to avoid invalidating autograd's saved-tensor
    # versions across row-slice lookups.
    values = matrix.values()

    schedule: LevelSchedule = level_schedule(matrix, direction="forward")
    level_sizes = schedule.level_sizes.tolist()
    row_order = schedule.row_order

    offset = 0
    for level_size in level_sizes:
        level_rows = row_order[offset : offset + level_size]
        offset += level_size

        level_degrees = degrees[level_rows]
        max_degree = int(level_degrees.max()) if level_size > 0 else 0

        for d in range(max_degree):
            active_mask = level_degrees > d
            active_rows = level_rows[active_mask]
            if active_rows.numel() == 0:
                continue

            active_start = row_start[active_rows]
            flat_pos = active_start + d
            j_d = col[flat_pos]
            idx_diag_active = idx_diag[active_rows]
            is_l_side = d < idx_diag_active
            boundary = torch.where(is_l_side, torch.full_like(idx_diag_active, d), idx_diag_active)
            max_boundary = int(boundary.max())

            if max_boundary > 0:
                k_idx = torch.arange(max_boundary, device=values.device)
                valid = k_idx.unsqueeze(0) < boundary.unsqueeze(1)
                row_degree_active = degrees[active_rows]
                safe_k = torch.minimum(
                    k_idx.unsqueeze(0).expand(active_rows.numel(), -1),
                    (row_degree_active - 1).unsqueeze(1),
                )
                pos2d = active_start.unsqueeze(1) + safe_k
                k_cols = col[pos2d]
                l_row_k = values[pos2d]
                query_rows = k_cols.reshape(-1)
                query_cols = j_d.unsqueeze(1).expand(-1, max_boundary).reshape(-1)
                found, u_kj = _lookup(keys, values, query_rows, query_cols, n)
                found = found.reshape(-1, max_boundary) & valid
                u_kj = u_kj.reshape(-1, max_boundary)
                dot = (l_row_k * u_kj * found).sum(dim=1)
            else:
                dot = torch.zeros(active_rows.numel(), dtype=values.dtype, device=values.device)

            a_val = values[flat_pos]
            u_jj = values[diag_flat_pos[j_d]]
            new_val_l = (a_val - dot) / u_jj
            new_val_u = a_val - dot
            new_val = torch.where(is_l_side, new_val_l, new_val_u)
            values = values.scatter(0, flat_pos, new_val)

    return torch.sparse_csr_tensor(crow, col, values, size=matrix.shape, check_invariants=False)


class ILUPreconditioner(LinearPreconditioner[torch.Tensor], nn.Module):
    """Incomplete LU preconditioner for sparse CSR systems: ``z = (LU)^{-1}r``.

    Sparse-CSR sibling of the dense
    ``preconditioners.implementations.ilu.ILUPreconditioner`` - same name,
    disambiguated by package. More effective than Jacobi, but more
    expensive to apply.

    Follows the ``Preconditioner`` + ``nn.Module`` pattern established by
    the dense preconditioners: ``nn.Module.__init__()`` is called explicitly
    and the combined ``L``/``U`` factor is stored via ``register_buffer``
    (never ``register_parameter`` - it is never learned or
    autograd-tracked) so it participates in ``.to(device=..., dtype=...)``
    propagation.

    Mathematical Properties:
        - ``M ≈ A`` via incomplete LU factorization (no fill-in beyond
          ``A``'s original non-zero pattern - natural ordering, no
          pivoting).
        - ``z = (LU)^{-1}r`` via forward/backward sparse triangular solves.
        - Not guaranteed SPD even if ``A`` is SPD.

    Example:
        >>> import torch
        >>> matrix = (torch.eye(3, dtype=torch.float64) * 2.0).to_sparse_csr()
        >>> precond = ILUPreconditioner(matrix)
        >>> z = precond.apply(torch.ones(3, dtype=torch.float64))
    """

    def __init__(self, matrix: torch.Tensor) -> None:
        """Initialize from sparse CSR system matrix, registering the combined LU factor.

        Args:
            matrix (torch.Tensor): Sparse CSR system matrix ``A``, shape
                ``(n, n)``.
        """
        nn.Module.__init__(self)
        self._operator: torch.Tensor
        self.register_buffer("_operator", self._compute_operator(matrix))

    def _compute_operator(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute the sparse ILU(0) factorization of the system matrix.

        Args:
            matrix (torch.Tensor): Sparse CSR system matrix ``A``.

        Returns:
            torch.Tensor: Combined ``L``/``U`` sparse CSR factor tensor.
        """
        return sparse_ilu0(matrix)

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Solve ``(LU) z = r`` via sparse forward/backward triangular solves.

        ``L``'s implicit unit diagonal (never stored in the combined
        factor) is materialized explicitly via ``_materialize_unit_lower``
        before being handed to ``triangular_solve`` - which has no
        unit-diagonal convention of its own and would otherwise divide by
        a structurally-zero diagonal (frozen to ``target``, per
        ``triangular_solve``'s own zero-diagonal handling - silently wrong
        here, not just inefficient). ``U`` is the combined factor's own
        diagonal-and-upper part, used as-is.

        Args:
            residual (torch.Tensor): Residual vector(s) ``r``, shape
                ``(n,)`` or ``(n, k)``.
            context (PreconditionerContext | None): Ignored (ILU doesn't
                need context).

        Returns:
            torch.Tensor: Preconditioned residual ``z = (LU)^{-1}r``.
        """
        if residual.ndim > 1:
            columns = [self.apply(residual[:, k]) for k in range(residual.shape[1])]
            return torch.stack(columns, dim=1)

        combined = self._operator
        row_index = _expand_row_index(combined)
        col = combined.col_indices()
        upper_mask = col >= row_index

        lower = _materialize_unit_lower(combined)
        upper = _csr_from_mask(combined, upper_mask)

        forward_schedule = level_schedule(lower, direction="forward")
        backward_schedule = level_schedule(upper, direction="backward")

        y = triangular_solve(lower, forward_schedule, residual, direction="forward")
        return triangular_solve(upper, backward_schedule, y, direction="backward")
