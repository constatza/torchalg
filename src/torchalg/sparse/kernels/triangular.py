"""Level-scheduled sparse triangular systems over a CSR matrix's own entries.

Generic machinery, not IC0-specific: consumed both by ``GaussSeidelSmoother``'s
sparse path (triangular solve against ``A``'s own ``D``/``L``/``U`` split, no
factorization) and, in a later phase, by IC0/ILU's apply step against a
computed incomplete factor. See ``docs/plan.md``'s "Design" section and its
"Correction: a reversed forward schedule is not a valid backward schedule"
section - the latter is the authoritative spec this module implements:
forward and backward level schedules must each be computed independently,
never one derived from the other by reversal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class LevelSchedule:
    """A row partition into independence levels for triangular substitution.

    Attributes:
        row_order (torch.Tensor): int64 ``(n,)``, rows grouped by level,
            ascending.
        level_sizes (torch.Tensor): int64 ``(num_levels,)``, rows per level.
    """

    row_order: torch.Tensor
    level_sizes: torch.Tensor


def _require_csr(matrix: torch.Tensor, caller: str) -> None:
    """Raise ``ValueError`` unless ``matrix`` is sparse CSR.

    Args:
        matrix (torch.Tensor): Candidate matrix.
        caller (str): Name of the calling function, for the error message.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR.
    """
    if matrix.layout != torch.sparse_csr:
        raise ValueError(f"{caller} requires a sparse CSR tensor, got layout {matrix.layout}")


def _expand_row_index(matrix: torch.Tensor) -> torch.Tensor:
    """Expand ``crow_indices()`` into a per-nonzero-entry row index.

    Mirrors ``sparse_diagonal``'s vectorized expansion trick.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix.

    Returns:
        torch.Tensor: int64, one row index per stored nonzero entry.
    """
    n = matrix.shape[0]
    crow = matrix.crow_indices()
    row_nnz = crow[1:] - crow[:-1]
    return torch.repeat_interleave(torch.arange(n, device=matrix.device), row_nnz)


def level_schedule(
    matrix: torch.Tensor, *, direction: Literal["forward", "backward"] = "forward"
) -> LevelSchedule:
    """Build a direction-specific level schedule from ``matrix``'s sparsity pattern.

    Forward: ``level[i] = 0`` if row ``i`` has no strictly-lower (``col <
    i``) stored neighbor, else ``1 + max(level[j])`` over those neighbors.
    Backward: the mirror, over strictly-upper (``col > i``) neighbors.
    Computed independently per direction - never one derived from the other
    by reversal, see module docstring.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix, shape ``(n, n)``.
        direction (Literal["forward", "backward"]): Which triangular half's
            dependency graph to schedule.

    Returns:
        LevelSchedule: Rows partitioned into independence levels.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR.
    """
    _require_csr(matrix, "level_schedule")

    n = matrix.shape[0]
    if n == 0:
        return LevelSchedule(
            row_order=torch.empty(0, dtype=torch.int64, device=matrix.device),
            level_sizes=torch.empty(0, dtype=torch.int64, device=matrix.device),
        )

    row_index = _expand_row_index(matrix)
    col = matrix.col_indices()
    is_strict_dependency = col < row_index if direction == "forward" else col > row_index

    dep_row = row_index[is_strict_dependency]
    dep_col = col[is_strict_dependency]
    boundaries = torch.searchsorted(dep_row, torch.arange(n + 1, device=matrix.device))

    level = torch.zeros(n, dtype=torch.int64, device=matrix.device)
    row_range = range(n) if direction == "forward" else range(n - 1, -1, -1)
    boundaries_list = boundaries.tolist()
    dep_col_list = dep_col.tolist()
    for row in row_range:
        start, end = boundaries_list[row], boundaries_list[row + 1]
        if start == end:
            level[row] = 0
        else:
            level[row] = 1 + max(level[dep_col_list[k]] for k in range(start, end))

    num_levels = int(level.max()) + 1
    order = torch.argsort(level, stable=True)
    level_sizes = torch.bincount(level[order], minlength=num_levels)
    return LevelSchedule(row_order=order, level_sizes=level_sizes)


def triangular_solve(
    matrix: torch.Tensor,
    schedule: LevelSchedule,
    target: torch.Tensor,
    *,
    direction: Literal["forward", "backward"] = "forward",
    diagonal_tol: float = 1e-14,
    active: torch.Tensor | None = None,
    current: torch.Tensor | None = None,
) -> torch.Tensor:
    """Solve the triangular system selected by ``direction`` against ``matrix``.

    ``matrix`` is always the full sparse CSR matrix - never pre-masked -
    internally restricted here to the half ``direction`` selects: forward is
    ``col <= row`` (lower, incl. diagonal), backward is ``col >= row``
    (upper, incl. diagonal). Rows with a structurally-zero diagonal freeze to
    ``target``'s value rather than dividing.

    ``active``/``current`` add PyAMG's ``gauss_seidel_indexed`` masking:
    rows where ``active`` is ``False`` are excluded from this solve and
    instead frozen to their ``current`` value - distinct from the
    zero-diagonal freeze, which keeps using ``target`` (see
    ``_relaxation.py``'s dense ``_prepare_sweep``/``_apply_sweep`` for the
    semantics this mirrors: a frozen row's own equation becomes the
    identity, but it still couples into other rows' equations via its
    frozen value).

    Args:
        matrix (torch.Tensor): Sparse CSR matrix, shape ``(n, n)``.
        schedule (LevelSchedule): Matching ``direction``'s level schedule.
        target (torch.Tensor): Right-hand side, shape ``(n,)``.
        direction (Literal["forward", "backward"]): Which triangular half to solve.
        diagonal_tol (float): Diagonal magnitude below this is treated as zero.
        active (torch.Tensor | None): Boolean mask of rows allowed to
            change, shape ``(n,)``. ``None`` (default) updates every row.
        current (torch.Tensor | None): Current iterate, shape ``(n,)``,
            required whenever ``active`` excludes at least one row - its
            values are what those rows freeze to.

    Returns:
        torch.Tensor: Solution, shape ``(n,)``.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR, or if ``active``
            excludes a row but ``current`` is not given.
    """
    _require_csr(matrix, "triangular_solve")

    n = matrix.shape[0]
    row_index = _expand_row_index(matrix)
    col = matrix.col_indices()
    values = matrix.values()

    is_diag = col == row_index
    diag = torch.zeros(n, dtype=matrix.dtype, device=matrix.device)
    diag = diag.scatter_add(0, row_index[is_diag], values[is_diag])
    frozen_diagonal = diag.abs() <= diagonal_tol

    inactive = torch.zeros(n, dtype=torch.bool, device=matrix.device) if active is None else ~active
    if bool(inactive.any()) and current is None:
        raise ValueError("current is required when active excludes at least one row")
    freeze_value = target if current is None else torch.where(inactive, current, target)
    frozen = frozen_diagonal | inactive

    is_offdiag_in_triangle = col < row_index if direction == "forward" else col > row_index
    offdiag_row = row_index[is_offdiag_in_triangle]
    offdiag_col = col[is_offdiag_in_triangle]
    offdiag_val = values[is_offdiag_in_triangle]

    num_levels = len(schedule.level_sizes)
    row_to_level = torch.empty(n, dtype=torch.int64, device=matrix.device)
    row_to_level[schedule.row_order] = torch.repeat_interleave(
        torch.arange(num_levels, device=matrix.device), schedule.level_sizes
    )
    entry_level = row_to_level[offdiag_row]

    x = torch.zeros_like(target)
    level_start = 0
    for level_idx in range(num_levels):
        level_size = int(schedule.level_sizes[level_idx])
        rows = schedule.row_order[level_start : level_start + level_size]
        level_start += level_size

        mask = entry_level == level_idx
        contribution = torch.zeros(n, dtype=target.dtype, device=target.device).scatter_add(
            0, offdiag_row[mask], offdiag_val[mask] * x[offdiag_col[mask]]
        )

        solved = (target[rows] - contribution[rows]) / diag[rows]
        x[rows] = torch.where(frozen[rows], freeze_value[rows], solved)

    return x
