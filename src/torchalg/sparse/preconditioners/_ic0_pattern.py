"""Structure-only analysis for sparse IC(0): everything that depends on the pattern, not the values.

IC(0)'s elimination order, the storage positions of every ``L[i,k]``/``L[j,k]``
product it gathers, and the transpose permutation of ``L`` follow from the
sparsity pattern alone. ``analyze_ic0`` computes them once into a frozen
``IC0Pattern``; ``sparse_ic0`` then only gathers by those positions, so a
matrix whose values change but whose pattern does not (the usual re-setup)
skips the searches, schedule construction and host synchronizations.

The pattern is the stored entries of the lower triangle whose value is not
zero ("any non-zero is pattern"), the same mask the dense IC(0) uses with its
default zero drop tolerance. The mask is part of the reuse key
(``IC0Pattern.matches``): an entry turning zero or non-zero changes the
factorization's structure, so it must not silently reuse a stale pattern.
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
class IC0Gather:
    """Positions of the ``sum_k L[i,k] * L[j,k]`` products of one elimination step.

    Attributes:
        pos_ik (torch.Tensor): int64 ``(m, d)``, lower-storage position of
            ``L[i,k]`` for the ``d`` entries of row ``i`` left of column ``j``.
        pos_jk (torch.Tensor): int64 ``(m, d)``, position of ``L[j,k]``; ``0``
            where absent (a valid index, masked out by ``found``).
        found (torch.Tensor): bool ``(m, d)``, whether ``(j, k)`` is in the pattern.
    """

    pos_ik: torch.Tensor
    pos_jk: torch.Tensor
    found: torch.Tensor


@dataclass(frozen=True, slots=True)
class IC0Step:
    """One degree position ``d`` of a level: the ``d``-th off-diagonal of every row that has one.

    Attributes:
        flat_pos (torch.Tensor): int64 ``(m,)``, lower-storage position of ``L[i,j]``.
        diag_pos_of_j (torch.Tensor): int64 ``(m,)``, position of ``L[j,j]``.
        level_local_idx (torch.Tensor): int64 ``(m,)``, index of each active row
            inside its level.
        gather (IC0Gather | None): The ``k < j`` products; ``None`` for ``d == 0``,
            which has no earlier column.
    """

    flat_pos: torch.Tensor
    diag_pos_of_j: torch.Tensor
    level_local_idx: torch.Tensor
    gather: IC0Gather | None


@dataclass(frozen=True, slots=True)
class IC0Level:
    """Rows eliminated together because none depends on another in the level.

    Attributes:
        rows (torch.Tensor): int64 ``(level_size,)``, the rows of the level.
        diag_pos (torch.Tensor): int64 ``(level_size,)``, position of each row's diagonal.
        steps (tuple[IC0Step, ...]): One step per degree position, up to the
            level's maximum off-diagonal count (a host-known loop bound).
    """

    rows: torch.Tensor
    diag_pos: torch.Tensor
    steps: tuple[IC0Step, ...]


@dataclass(frozen=True, slots=True)
class IC0Pattern:
    """Reusable, values-free IC(0) plan for every matrix with the same pattern.

    Attributes:
        crow (torch.Tensor): Copy of the analyzed matrix's ``crow_indices`` (reuse
            key). A copy, not a reference, so an in-place edit of the matrix's own
            indices after analysis is seen by ``matches``.
        col (torch.Tensor): Copy of the analyzed matrix's ``col_indices`` (reuse key).
        row (torch.Tensor): Row of every stored entry, derived from ``crow``.
            Like ``lower_crow``, ``lower_col``, ``upper_crow``, ``upper_col`` and
            ``transpose_perm``, it only indexes storage and is held as int32 when
            ``max(n, nnz)`` fits (``structural_index``), else int64.
        lower_mask (torch.Tensor): bool, stored entries that are in the lower
            triangle and non-zero (reuse key; selects ``L``'s input values).
        lower_crow (torch.Tensor): ``crow_indices`` of the factor ``L``.
        lower_col (torch.Tensor): ``col_indices`` of the factor ``L``.
        levels (tuple[IC0Level, ...]): Forward level schedule with the gather
            positions of every step.
        upper_crow (torch.Tensor): ``crow_indices`` of ``L.T``.
        upper_col (torch.Tensor): ``col_indices`` of ``L.T``.
        transpose_perm (torch.Tensor): ``L.T.values() == L.values()[transpose_perm]``.
    """

    crow: torch.Tensor
    col: torch.Tensor
    row: torch.Tensor
    lower_mask: torch.Tensor
    lower_crow: torch.Tensor
    lower_col: torch.Tensor
    levels: tuple[IC0Level, ...]
    upper_crow: torch.Tensor
    upper_col: torch.Tensor
    transpose_perm: torch.Tensor

    def matches(self, matrix: torch.Tensor) -> bool:
        """Whether ``matrix`` has the analyzed stored structure and non-zero lower mask.

        One O(nnz) comparison, cheap next to the factorization it lets the
        caller skip re-planning for.

        Args:
            matrix (torch.Tensor): Candidate matrix.

        Returns:
            bool: ``True`` only for a sparse CSR matrix on the same device, with
                the same index dtypes, ``crow``, ``col`` and lower non-zero mask.
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
        return (
            torch.equal(crow, self.crow)
            and torch.equal(col, self.col)
            and torch.equal(_lower_nonzero_mask(self.row, col, matrix.values()), self.lower_mask)
        )


def _lower_nonzero_mask(row: torch.Tensor, col: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Stored lower-triangle entries that are pattern: ``|value| > 0``, so zero and NaN drop out."""
    return (col <= row) & (values.abs() > 0)


def _analyze_step(
    d: int,
    degrees: torch.Tensor,
    rows: torch.Tensor,
    row_start: torch.Tensor,
    diag_pos: torch.Tensor,
    lower_col: torch.Tensor,
    keys: torch.Tensor,
    positions: torch.Tensor,
    n: int,
) -> IC0Step:
    """Resolve the storage positions of degree position ``d`` of one level."""
    active_mask = degrees > d
    active_start = row_start[rows[active_mask]]
    flat_pos = active_start + d
    j = lower_col[flat_pos]

    gather = None
    if d > 0:
        pos_ik = active_start.unsqueeze(1) + torch.arange(d, device=rows.device).unsqueeze(0)
        query_rows = j.unsqueeze(1).expand(-1, d).reshape(-1)
        # Looking up each entry's own storage position (instead of a value)
        # makes the result a gather index that later factorizations reuse.
        found, pos_jk = lookup(keys, positions, query_rows, lower_col[pos_ik].reshape(-1), n)
        gather = IC0Gather(pos_ik, pos_jk.reshape(-1, d), found.reshape(-1, d))

    return IC0Step(
        flat_pos=flat_pos,
        diag_pos_of_j=diag_pos[j],
        level_local_idx=torch.arange(rows.numel(), device=rows.device)[active_mask],
        gather=gather,
    )


def analyze_ic0(matrix: torch.Tensor) -> IC0Pattern:
    """Plan the IC(0) factorization of every matrix sharing ``matrix``'s pattern.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix, shape ``(n, n)``. Only its
            stored structure and non-zero mask are read, never the magnitudes.

    Returns:
        IC0Pattern: Reusable plan for ``sparse_ic0``.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR.
    """
    _require_csr(matrix, "analyze_ic0")

    n = matrix.shape[0]
    crow = matrix.crow_indices()
    col = matrix.col_indices()
    row = _expand_row_index(matrix)
    lower_mask = _lower_nonzero_mask(row, col, matrix.values())
    lower_row = row[lower_mask]
    lower_col = col[lower_mask].to(torch.int64)  # int64: ``* n`` below must not overflow int32
    lower_crow = row_pointer(lower_row, n)

    lower = torch.sparse_csr_tensor(
        lower_crow,
        lower_col,
        torch.zeros(lower_col.numel(), dtype=matrix.dtype, device=matrix.device),
        size=matrix.shape,
        check_invariants=False,
    )
    schedule = level_schedule(lower, direction="forward")

    row_start = lower_crow[:-1]
    diag_pos = lower_crow[1:] - 1  # flat position of each row's own diagonal entry
    offdiag_count = diag_pos - row_start
    keys = build_lookup(lower_row, lower_col, n)
    positions = torch.arange(lower_col.numel(), device=matrix.device)

    levels: list[IC0Level] = []
    offset = 0
    for level_size in schedule.level_sizes.tolist():
        rows = schedule.row_order[offset : offset + level_size]
        offset += level_size
        degrees = offdiag_count[rows]
        steps = tuple(
            _analyze_step(d, degrees, rows, row_start, diag_pos, lower_col, keys, positions, n)
            for d in range(int(degrees.max()))
        )
        levels.append(IC0Level(rows=rows, diag_pos=diag_pos[rows], steps=steps))

    # (col, row) is unique per entry, so sorting by it lists L.T's storage order.
    transpose_perm = torch.argsort(lower_col * n + lower_row)

    extent = max(n, col.numel())
    return IC0Pattern(
        crow=crow.clone(),
        col=col.clone(),
        row=structural_index(row, extent),
        lower_mask=lower_mask,
        lower_crow=structural_index(lower_crow, extent),
        lower_col=structural_index(lower_col, extent),
        levels=tuple(levels),
        upper_crow=structural_index(row_pointer(lower_col, n), extent),
        upper_col=structural_index(lower_row[transpose_perm], extent),
        transpose_perm=structural_index(transpose_perm, extent),
    )
