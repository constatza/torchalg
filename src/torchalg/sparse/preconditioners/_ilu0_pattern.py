"""Structure-only analysis for sparse ILU(0): everything that depends on the pattern, not the values.

ILU(0)'s pattern is the stored structure of the matrix itself (no threshold, no
value mask), so the elimination order, the storage positions of every
``L[i,k]``/``U[k,j]`` product it gathers and the split of the combined factor
into the two triangles to solve with follow from ``crow``/``col`` alone.
``analyze_ilu0`` computes them once into a frozen ``ILU0Pattern``;
``sparse_ilu0`` then only gathers by those positions, so a matrix whose values
change but whose structure does not (the usual re-setup) skips the searches,
the schedule construction, the sorts and every host synchronization.

The reuse key is ``crow``/``col`` (``ILU0Pattern.matches``): an explicitly
stored zero stays in the pattern, unlike IC(0)'s non-zero mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchalg.sparse.kernels.triangular import (
    _expand_row_index,
    _require_csr,
    level_schedule,
)
from torchalg.sparse.preconditioners._pattern_index import (
    build_lookup,
    lookup,
    row_pointer,
    structural_index,
)


@dataclass(frozen=True, slots=True)
class ILU0Gather:
    """Positions of the ``sum_k L[i,k] * U[k,j]`` products of one elimination step.

    The step's rows are padded to the widest row's product count; ``found``
    masks the padding and the ``(k, j)`` pairs outside the pattern.

    Attributes:
        pos_ik (torch.Tensor): int64 ``(m, b)``, storage position of ``L[i,k]``.
        pos_kj (torch.Tensor): int64 ``(m, b)``, position of ``U[k,j]``; ``0``
            where absent (a valid index, masked out by ``found``).
        found (torch.Tensor): bool ``(m, b)``, whether the product is a term of the sum.
    """

    pos_ik: torch.Tensor
    pos_kj: torch.Tensor
    found: torch.Tensor


@dataclass(frozen=True, slots=True)
class ILU0Step:
    """One degree position ``d`` of a level: the ``d``-th stored entry of every row that has one.

    Attributes:
        flat_pos (torch.Tensor): int64 ``(m,)``, storage position of ``A[i,j]``.
        diag_pos_of_j (torch.Tensor): int64 ``(m,)``, position of ``U[j,j]``.
        is_l_side (torch.Tensor): bool ``(m,)``, whether the entry is in ``L``
            (``j < i``, divided by ``U[j,j]``) or in ``U``.
        gather (ILU0Gather | None): The products to subtract; ``None`` when no
            row of the step has any.
    """

    flat_pos: torch.Tensor
    diag_pos_of_j: torch.Tensor
    is_l_side: torch.Tensor
    gather: ILU0Gather | None


@dataclass(frozen=True, slots=True)
class ILU0Pattern:
    """Reusable, values-free ILU(0) plan for every matrix with the same stored structure.

    Attributes:
        crow (torch.Tensor): Copy of the analyzed matrix's ``crow_indices`` (reuse
            key). A copy, not a reference, so an in-place edit of the matrix's own
            indices after analysis is seen by ``matches``.
        col (torch.Tensor): Copy of the analyzed matrix's ``col_indices`` (reuse key).
        steps (tuple[ILU0Step, ...]): Elimination steps in dependency order:
            levels of the forward schedule, then degree positions within a level.
        factor_crow (torch.Tensor): ``crow_indices`` of the combined factor.
        factor_col (torch.Tensor): ``col_indices`` of the combined factor.
        lower_crow (torch.Tensor): ``crow_indices`` of the unit-lower factor.
        lower_col (torch.Tensor): ``col_indices`` of the unit-lower factor.
        lower_source (torch.Tensor): Unit-lower values are ``cat([values, 1])[lower_source]``:
            the strictly-lower entries, then the explicit unit diagonal (position ``nnz``).
        upper_crow (torch.Tensor): ``crow_indices`` of the upper factor.
        upper_col (torch.Tensor): ``col_indices`` of the upper factor.
        upper_source (torch.Tensor): Upper values are ``values[upper_source]``.

    The ``factor_*``, ``lower_*`` and ``upper_*`` arrays only index storage and
    are held as int32 when ``max(n, nnz)`` fits (``structural_index``), else int64.
    """

    crow: torch.Tensor
    col: torch.Tensor
    steps: tuple[ILU0Step, ...]
    factor_crow: torch.Tensor
    factor_col: torch.Tensor
    lower_crow: torch.Tensor
    lower_col: torch.Tensor
    lower_source: torch.Tensor
    upper_crow: torch.Tensor
    upper_col: torch.Tensor
    upper_source: torch.Tensor

    def matches(self, matrix: torch.Tensor) -> bool:
        """Whether ``matrix`` has the analyzed stored structure.

        One O(nnz) comparison, cheap next to the factorization it lets the
        caller skip re-planning for.

        Args:
            matrix (torch.Tensor): Candidate matrix.

        Returns:
            bool: ``True`` only for a sparse CSR matrix on the same device, with
                the same index dtypes, ``crow`` and ``col``.
        """
        if matrix.layout != torch.sparse_csr:
            return False
        crow, col = matrix.crow_indices(), matrix.col_indices()
        if (
            crow.device != self.crow.device
            or crow.dtype != self.crow.dtype
            or col.dtype != self.col.dtype
        ):
            return False
        return torch.equal(crow, self.crow) and torch.equal(col, self.col)


def _analyze_step(
    d: int,
    level_degrees: torch.Tensor,
    rows: torch.Tensor,
    degrees: torch.Tensor,
    row_start: torch.Tensor,
    idx_diag: torch.Tensor,
    diag_flat_pos: torch.Tensor,
    col: torch.Tensor,
    keys: torch.Tensor,
    positions: torch.Tensor,
    n: int,
) -> ILU0Step:
    """Resolve the storage positions of degree position ``d`` of one level."""
    active_rows = rows[level_degrees > d]
    active_start = row_start[active_rows]
    flat_pos = active_start + d
    j_d = col[flat_pos]
    idx_diag_active = idx_diag[active_rows]
    is_l_side = d < idx_diag_active
    # L-side entries sum over the ``d`` earlier columns of their row; U-side
    # entries over all of the row's L-side columns, so the batch is padded to
    # the widest and the padding masked out.
    boundary = torch.where(is_l_side, torch.full_like(idx_diag_active, d), idx_diag_active)
    max_boundary = int(boundary.max())

    gather = None
    if max_boundary > 0:
        k_idx = torch.arange(max_boundary, device=col.device)
        valid = k_idx.unsqueeze(0) < boundary.unsqueeze(1)
        safe_k = torch.minimum(
            k_idx.unsqueeze(0).expand(active_rows.numel(), -1),
            (degrees[active_rows] - 1).unsqueeze(1),
        )
        pos_ik = active_start.unsqueeze(1) + safe_k
        query_cols = j_d.unsqueeze(1).expand(-1, max_boundary).reshape(-1)
        # Looking up each entry's own storage position (instead of a value)
        # makes the result a gather index that later factorizations reuse.
        found, pos_kj = lookup(keys, positions, col[pos_ik].reshape(-1), query_cols, n)
        gather = ILU0Gather(
            pos_ik, pos_kj.reshape(-1, max_boundary), found.reshape(-1, max_boundary) & valid
        )

    return ILU0Step(
        flat_pos=flat_pos,
        diag_pos_of_j=diag_flat_pos[j_d],
        is_l_side=is_l_side,
        gather=gather,
    )


def analyze_ilu0(matrix: torch.Tensor) -> ILU0Pattern:
    """Plan the ILU(0) factorization of every matrix sharing ``matrix``'s stored structure.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix, shape ``(n, n)``, with every
            diagonal entry stored. Only its structure is read, never the values.

    Returns:
        ILU0Pattern: Reusable plan for ``sparse_ilu0``.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR.
    """
    _require_csr(matrix, "analyze_ilu0")

    n = matrix.shape[0]
    crow = matrix.crow_indices()
    stored_col = matrix.col_indices()
    col = stored_col.to(torch.int64)  # int64: ``* n`` in the lookup keys must not overflow int32
    row = _expand_row_index(matrix)
    nnz = col.numel()
    extent = max(n, nnz)

    row_start = crow[:-1].to(torch.int64)
    degrees = crow[1:].to(torch.int64) - row_start
    # Columns ascend within a row, so a row's L-side entries come first and
    # their count is the position of its own diagonal entry.
    is_lower = col < row
    idx_diag = torch.bincount(row[is_lower], minlength=n)
    diag_flat_pos = row_start + idx_diag
    keys = build_lookup(row, col, n)
    positions = torch.arange(nnz, device=matrix.device)

    schedule = level_schedule(matrix, direction="forward")
    steps: list[ILU0Step] = []
    offset = 0
    for level_size in schedule.level_sizes.tolist():
        rows = schedule.row_order[offset : offset + level_size]
        offset += level_size
        level_degrees = degrees[rows]
        steps.extend(
            _analyze_step(
                d,
                level_degrees,
                rows,
                degrees,
                row_start,
                idx_diag,
                diag_flat_pos,
                col,
                keys,
                positions,
                n,
            )
            for d in range(int(level_degrees.max()))
        )

    # The unit-lower factor holds the strictly-lower entries plus an explicit
    # unit diagonal (the triangular solve divides by the stored diagonal).
    lower_entries = is_lower.nonzero().squeeze(1)
    diagonal = torch.arange(n, device=matrix.device)
    lower_row = torch.cat([row[lower_entries], diagonal])
    lower_col_unsorted = torch.cat([col[lower_entries], diagonal])
    order = torch.argsort(lower_row * n + lower_col_unsorted, stable=True)
    unit_source = torch.full((n,), nnz, dtype=torch.int64, device=matrix.device)

    upper_entries = (col >= row).nonzero().squeeze(1)

    return ILU0Pattern(
        crow=crow.clone(),
        col=stored_col.clone(),
        steps=tuple(steps),
        factor_crow=structural_index(crow.to(torch.int64), extent),
        factor_col=structural_index(col, extent),
        lower_crow=structural_index(row_pointer(lower_row, n), extent),
        lower_col=structural_index(lower_col_unsorted[order], extent),
        lower_source=structural_index(torch.cat([lower_entries, unit_source])[order], extent),
        upper_crow=structural_index(row_pointer(row[upper_entries], n), extent),
        upper_col=structural_index(col[upper_entries], extent),
        upper_source=structural_index(upper_entries, extent),
    )
