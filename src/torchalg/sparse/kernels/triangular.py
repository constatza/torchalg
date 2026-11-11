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

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import torch
from torch import nn

from torchalg.sparse.kernels.diagonal import sparse_diagonal


@dataclass(frozen=True, slots=True)
class LevelSchedule:
    """A row partition into independence levels for triangular substitution.

    Attributes:
        row_order (torch.Tensor): int64 ``(n,)``, rows grouped by level,
            ascending.
        level_sizes (torch.Tensor): int64 ``(num_levels,)``, rows per level.
    """

    row_order: torch.Tensor
    level_sizes: torch.Tensor


def _cpu_dependency_levels(
    dep_row: torch.Tensor,
    dep_col: torch.Tensor,
    n: int,
    direction: Literal["forward", "backward"],
) -> torch.Tensor:
    """Compute triangular dependency levels with scalar-free tensor access.

    The recurrence is inherently ordered, so on CPU a linear Python pass
    over native integers avoids the overhead of one scalar tensor operation
    per dependency while retaining ``O(n + nnz)`` work.

    Args:
        dep_row: Row owning each strict triangular dependency.
        dep_col: Dependency column for each entry in ``dep_row``.
        n: Matrix dimension.
        direction: Triangular traversal direction.

    Returns:
        One int64 level per row on the CPU.
    """
    boundaries = torch.searchsorted(dep_row, torch.arange(n + 1)).tolist()
    dependencies = dep_col.tolist()
    levels = [0] * n
    row_range = range(n) if direction == "forward" else range(n - 1, -1, -1)

    for row in row_range:
        start, end = boundaries[row], boundaries[row + 1]
        if start != end:
            levels[row] = 1 + max(levels[dependencies[position]] for position in range(start, end))

    return torch.tensor(levels, dtype=torch.int64)


def _ragged_positions(starts: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Expand grouped ``start``/``length`` pairs into flat storage positions.

    Args:
        starts: Start position of each selected group.
        lengths: Number of positions in each selected group.

    Returns:
        Concatenated positions for all selected groups.
    """
    repeated_starts = torch.repeat_interleave(starts, lengths)
    repeated_offsets = torch.repeat_interleave(torch.cumsum(lengths, dim=0) - lengths, lengths)
    within_group = torch.arange(repeated_starts.numel(), device=starts.device) - repeated_offsets
    return repeated_starts + within_group


def _device_dependency_schedule(
    dep_row: torch.Tensor,
    dep_col: torch.Tensor,
    n: int,
) -> LevelSchedule:
    """Build a level schedule by processing each dependency edge once.

    Reverse adjacency groups dependent rows by the row they depend on. Each
    wavefront therefore touches only edges leaving that frontier instead of
    rescanning the full dependency set for every level.

    Args:
        dep_row: Row owning each strict triangular dependency.
        dep_col: Dependency column for each entry in ``dep_row``.
        n: Matrix dimension.

    Returns:
        Rows grouped into deterministic ascending wavefronts.

    Raises:
        RuntimeError: If the dependency graph cannot schedule every row.
    """
    device = dep_row.device
    unresolved = torch.bincount(dep_row, minlength=n)

    reverse_order = torch.argsort(dep_col, stable=True)
    reverse_dependency = dep_col[reverse_order]
    reverse_dependent = dep_row[reverse_order]
    reverse_counts = torch.bincount(reverse_dependency, minlength=n)
    reverse_crow = torch.zeros(n + 1, dtype=torch.int64, device=device)
    reverse_crow[1:] = torch.cumsum(reverse_counts, dim=0)

    frontier = torch.nonzero(unresolved == 0, as_tuple=True)[0]
    frontiers: list[torch.Tensor] = []
    level_sizes: list[int] = []
    scheduled = 0

    while scheduled < n:
        frontier_size = frontier.numel()
        if frontier_size == 0:
            raise RuntimeError("triangular dependency graph contains an unschedulable cycle")

        frontiers.append(frontier)
        level_sizes.append(frontier_size)
        scheduled += frontier_size

        starts = reverse_crow[frontier]
        lengths = reverse_counts[frontier]
        positions = _ragged_positions(starts, lengths)
        affected_rows = reverse_dependent[positions]
        decrements = torch.bincount(affected_rows, minlength=n)
        unresolved = unresolved - decrements

        candidates = torch.unique(affected_rows, sorted=True)
        frontier = candidates[unresolved[candidates] == 0]

    return LevelSchedule(
        row_order=torch.cat(frontiers),
        level_sizes=torch.tensor(level_sizes, dtype=torch.int64, device=device),
    )


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
    return torch.repeat_interleave(
        torch.arange(n, device=matrix.device), row_nnz, output_size=matrix.col_indices().numel()
    )


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
    if matrix.device.type != "cpu":
        return _device_dependency_schedule(dep_row, dep_col, n)

    level = _cpu_dependency_levels(dep_row, dep_col, n, direction)
    num_levels = int(level.max()) + 1
    order = torch.argsort(level, stable=True)
    level_sizes = torch.bincount(level[order], minlength=num_levels)
    return LevelSchedule(row_order=order, level_sizes=level_sizes)


class PreparedTriangularSolve(nn.Module):
    """Level-scheduled triangular substitution bound to one sparse CSR matrix.

    Everything that depends only on the matrix is computed once, at construction: the
    level schedule, the diagonal and its zero-pivot mask, and the strictly-off-diagonal
    entries of the selected triangle sorted by level with their within-level row
    position. ``solve`` then runs one gather, one segment-sum and one divide per level -
    O(nnz + n) work in total, no host synchronization, no mask over all stored entries per
    level (the previous per-call implementation was O(levels * nnz)).

    ``matrix`` is always the full CSR matrix, restricted here to the half ``direction``
    selects: forward is ``col <= row`` (lower, incl. diagonal), backward is ``col >= row``
    (upper, incl. diagonal). Rows with a structurally-zero diagonal freeze to ``target``.

    The derived tensors are non-persistent buffers, so ``.to()`` moves them with the module.
    The values are captured at construction: gradients flow through that construction to
    ``matrix.values()``, so for differentiable use build and solve inside one forward pass.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix, shape ``(n, n)``.
        direction (Literal["forward", "backward"]): Which triangular half to solve.
        diagonal_tol (float): Diagonal magnitude below this is treated as zero.
        schedule (LevelSchedule | None): Matching ``direction``'s schedule if the caller
            already built it; computed here when ``None``.
        row_index (torch.Tensor | None): Per-stored-entry row index of ``matrix`` if the
            caller already built it; computed here when ``None``.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR.
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        direction: Literal["forward", "backward"] = "forward",
        diagonal_tol: float = 1e-14,
        *,
        schedule: LevelSchedule | None = None,
        row_index: torch.Tensor | None = None,
    ) -> None:
        """Precompute the schedule, diagonal and level-sorted off-diagonal entries."""
        super().__init__()
        _require_csr(matrix, "PreparedTriangularSolve")
        n = matrix.shape[0]
        device = matrix.device
        if schedule is None:
            schedule = level_schedule(matrix, direction=direction)
        if row_index is None:
            row_index = _expand_row_index(matrix)
        col = matrix.col_indices()
        values = matrix.values()

        is_diag = col == row_index
        diag = torch.zeros(n, dtype=matrix.dtype, device=device)
        diag = diag.scatter_add(0, row_index[is_diag], values[is_diag])

        in_triangle = col < row_index if direction == "forward" else col > row_index
        entry_row = row_index[in_triangle]
        entry_col = col[in_triangle]
        entry_val = values[in_triangle]

        level_sizes = schedule.level_sizes
        num_levels = len(level_sizes)
        level_start = torch.cumsum(level_sizes, 0) - level_sizes
        row_to_level = torch.empty(n, dtype=torch.int64, device=device)
        row_to_level[schedule.row_order] = torch.repeat_interleave(
            torch.arange(num_levels, device=device), level_sizes
        )
        rank = torch.empty(n, dtype=torch.int64, device=device)
        rank[schedule.row_order] = torch.arange(n, device=device)
        entry_level = row_to_level[entry_row]
        by_level = torch.argsort(entry_level, stable=True)
        entry_counts = torch.bincount(entry_level, minlength=num_levels)

        self.row_order: torch.Tensor
        self.diag: torch.Tensor
        self.frozen_diagonal: torch.Tensor
        self.entry_local_row: torch.Tensor
        self.entry_col: torch.Tensor
        self.entry_val: torch.Tensor
        local_row = rank[entry_row] - level_start[entry_level]
        self.register_buffer("row_order", schedule.row_order, persistent=False)
        self.register_buffer("diag", diag, persistent=False)
        self.register_buffer("frozen_diagonal", diag.abs() <= diagonal_tol, persistent=False)
        self.register_buffer("entry_local_row", local_row[by_level], persistent=False)
        self.register_buffer("entry_col", entry_col[by_level], persistent=False)
        self.register_buffer("entry_val", entry_val[by_level], persistent=False)
        # One host transfer at build time buys sync-free slicing in every solve.
        self._level_sizes: tuple[int, ...] = tuple(level_sizes.tolist())
        counts = entry_counts.tolist()
        self._entry_bounds: tuple[int, ...] = (0, *itertools.accumulate(counts))

    def solve(
        self,
        target: torch.Tensor,
        *,
        active: torch.Tensor | None = None,
        current: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve the triangular system for ``target``.

        ``active``/``current`` add PyAMG's ``gauss_seidel_indexed`` masking: rows where
        ``active`` is ``False`` are excluded from this solve and instead frozen to their
        ``current`` value - distinct from the zero-diagonal freeze, which keeps using
        ``target``. A frozen row's own equation becomes the identity, but it still couples
        into other rows' equations through its frozen value.

        Args:
            target (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            active (torch.Tensor | None): Boolean mask of rows allowed to change, shape
                ``(n,)``. ``None`` (default) updates every row.
            current (torch.Tensor | None): Current iterate, shape matching ``target``,
                required whenever ``active`` excludes at least one row.

        Returns:
            torch.Tensor: Solution, same shape as ``target``.

        Raises:
            ValueError: If ``active`` excludes a row but ``current`` is not given.
        """
        batched = target.ndim == 2
        if active is None:
            frozen = self.frozen_diagonal
            freeze_value = target
        else:
            inactive = ~active
            if current is None:
                if bool(inactive.any()):
                    raise ValueError("current is required when active excludes at least one row")
                freeze_value = target
            else:
                freeze_value = torch.where(
                    inactive.unsqueeze(-1) if batched else inactive, current, target
                )
            frozen = self.frozen_diagonal | inactive

        x = torch.zeros_like(target)
        row_start = 0
        for level, size in enumerate(self._level_sizes):
            rows = self.row_order[row_start : row_start + size]
            row_start += size
            entries = slice(self._entry_bounds[level], self._entry_bounds[level + 1])
            rhs = target[rows]
            if entries.stop > entries.start:
                weights = self.entry_val[entries]
                gathered = x[self.entry_col[entries]]
                if batched:
                    weights = weights.unsqueeze(-1)
                contribution = torch.zeros_like(rhs).index_add_(
                    0, self.entry_local_row[entries], weights * gathered
                )
                rhs = rhs - contribution
            diag = self.diag[rows].unsqueeze(-1) if batched else self.diag[rows]
            is_frozen = frozen[rows].unsqueeze(-1) if batched else frozen[rows]
            x[rows] = torch.where(is_frozen, freeze_value[rows], rhs / diag)
        return x


def _triangle(matrix: torch.Tensor, direction: Literal["forward", "backward"]) -> torch.Tensor:
    """CSR copy of ``matrix`` keeping only its lower (forward) or upper (backward) half."""
    row = _expand_row_index(matrix)
    col = matrix.col_indices()
    keep = col <= row if direction == "forward" else col >= row
    counts = torch.zeros(matrix.shape[0] + 1, dtype=torch.int64, device=matrix.device)
    counts[1:] = torch.bincount(row[keep], minlength=matrix.shape[0])
    return torch.sparse_csr_tensor(
        torch.cumsum(counts, 0),
        col[keep],
        matrix.values()[keep],
        matrix.shape,
    )


class LibraryTriangularSolve(nn.Module):
    """Triangular substitution through ``torch.triangular_solve`` on a CSR triangle.

    Solves the same system as ``PreparedTriangularSolve`` (``direction`` selects the half
    of ``matrix`` incl. the diagonal) with one library call per solve instead of a
    level-scheduled loop. It has no zero-pivot freeze and no ``active`` masking, so build
    it through ``prepare_triangular_solve``, which only picks it when every pivot is
    nonzero. Gradients are not supported by the library call: a target that requires
    grad is solved by a lazily built ``PreparedTriangularSolve`` instead.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix, shape ``(n, n)``.
        direction (Literal["forward", "backward"]): Which triangular half to solve.
    """

    def __init__(
        self, matrix: torch.Tensor, direction: Literal["forward", "backward"] = "forward"
    ) -> None:
        """Extract and store the selected triangle."""
        super().__init__()
        _require_csr(matrix, "LibraryTriangularSolve")
        self.triangle: torch.Tensor
        self.register_buffer("triangle", _triangle(matrix, direction), persistent=False)
        self._direction: Literal["forward", "backward"] = direction

    def solve(self, target: torch.Tensor) -> torch.Tensor:
        """Solve the triangular system for ``target`` (``(n,)`` or ``(n, k)``).

        Args:
            target (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.

        Returns:
            torch.Tensor: Solution, same shape as ``target``.
        """
        if target.requires_grad or self.triangle.requires_grad:
            return PreparedTriangularSolve(self.triangle, self._direction).solve(target)
        column = target if target.ndim == 2 else target.unsqueeze(-1)
        solution = torch.triangular_solve(
            column, self.triangle, upper=self._direction == "backward"
        ).solution
        return solution if target.ndim == 2 else solution.squeeze(-1)


TriangularSolver = LibraryTriangularSolve | PreparedTriangularSolve
"""Any prepared substitution: both expose ``solve(target)`` with identical results."""

TriangularSolverFactory = Callable[
    [torch.Tensor, Literal["forward", "backward"], float], TriangularSolver
]
"""Builds a prepared substitution for ``(matrix, direction, diagonal_tol)``."""


def _level_scheduled(
    matrix: torch.Tensor, direction: Literal["forward", "backward"], diagonal_tol: float
) -> TriangularSolver:
    return PreparedTriangularSolve(matrix, direction, diagonal_tol)


def _library_unless_zero_pivot(
    matrix: torch.Tensor, direction: Literal["forward", "backward"], diagonal_tol: float
) -> TriangularSolver:
    # One host read at setup; the library call has no zero-pivot freeze.
    if matrix.shape[0] == 0 or bool((sparse_diagonal(matrix).abs() <= diagonal_tol).any()):
        return _level_scheduled(matrix, direction, diagonal_tol)
    return LibraryTriangularSolve(matrix, direction)


TRIANGULAR_SOLVER_BY_DEVICE: Mapping[str, TriangularSolverFactory] = MappingProxyType(
    {"cpu": _library_unless_zero_pivot}
)
"""The one place that maps a device type to its triangular-solve implementation.

Unlisted device types get the level-scheduled ``PreparedTriangularSolve``. CPU uses
``torch.triangular_solve`` on CSR (MKL): 0.28 ms against 14.7 ms per IC(0) apply at
n = 10000, equal to rounding. GPUs are not listed: on gfx1030 with the rocm7.2 wheels the
library solve faults on its second call, and a CUDA device must be measured with
``benchmarks/tmp/compare_sparse_backends.py`` before it is added here. Callers never
branch on the device; they call ``prepare_triangular_solve``."""


def prepare_triangular_solve(
    matrix: torch.Tensor,
    direction: Literal["forward", "backward"] = "forward",
    diagonal_tol: float = 1e-14,
) -> TriangularSolver:
    """Prepare the substitution registered for ``matrix``'s device type.

    Resolved once, at setup (or when a holder moves to another device); ``solve`` itself
    never consults the device. See ``TRIANGULAR_SOLVER_BY_DEVICE``.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix, shape ``(n, n)``.
        direction (Literal["forward", "backward"]): Which triangular half to solve.
        diagonal_tol (float): Diagonal magnitude below this counts as a zero pivot.

    Returns:
        TriangularSolver: An object with ``solve(target)``.
    """
    _require_csr(matrix, "prepare_triangular_solve")
    factory = TRIANGULAR_SOLVER_BY_DEVICE.get(matrix.device.type, _level_scheduled)
    return factory(matrix, direction, diagonal_tol)
