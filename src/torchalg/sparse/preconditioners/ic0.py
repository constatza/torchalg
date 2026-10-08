"""Zero-level incomplete Cholesky (IC(0)) preconditioner, sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.ic0.IC0Preconditioner`` (dense; kept
unmodified for comparison - see ``docs/plan.md``'s "Correction: dense and
sparse must be separate implementations, not an internal branch"). Row-
oriented ("left-looking") level-scheduled factorization - not a sparse
transliteration of the dense column-oriented ("right-looking") Crout loop,
see ``docs/plan.md``'s "Correction: the row-oriented recurrence
sparse_ic0/sparse_ilu0 actually need" section, the authoritative spec this
module implements. Rows are processed in
``kernels.triangular.level_schedule``'s row order - not strict ``0..n-1``
order - since the recurrence's result is independent of visitation order as
long as every row's dependencies are visited first (same property
``triangular_solve`` already relies on).

Both the factorization and ``apply()`` are differentiable by construction:
every value computed into the output buffer is a composition of
autograd-tracked tensor ops; ``.item()``/``.tolist()`` are used only for
Python-level control flow (loop bounds, breakdown checks), never to compute
a value that ends up in the returned factor.

The Python loop driving the factorization is bounded by ``num_levels *
max_row_degree_in_level`` (outer loop over levels, inner loop over
degree-position ``d``), not by ``nnz`` - every row in a level that has a
``d``-th off-diagonal entry is updated in one vectorized batch per ``d``,
via the global ``_lookup``/``_build_lookup`` sparse-element lookup below
(replaces the former one-row-at-a-time ``_row_value_at``). See
``docs/plan.md``'s "Known, separate, not-yet-fixed defect" note for the
performance rationale this batching addresses.
"""

from __future__ import annotations

from typing import Self

import torch
from torch import nn

from torchalg.preconditioners.base import LinearPreconditioner, PreconditionerContext
from torchalg.sparse.kernels.triangular import (
    LevelSchedule,
    _expand_row_index,
    _require_csr,
    level_schedule,
)
from torchalg.sparse.preconditioners._triangular import (
    SparseTriangularSolveCache,
    build_sparse_cholesky_solve_cache,
)

_DEFAULT_THRESHOLD = 0.0
"""Default drop tolerance: entries with |value| <= threshold are treated as
zero and excluded from the sparsity pattern. See the dense sibling's
docstring for the full rationale (0.0 only drops exact zeros; a nonzero
absolute default would be unsafe across differently-scaled matrices)."""


def _build_lookup(row_index: torch.Tensor, col_index: torch.Tensor, n: int) -> torch.Tensor:
    """Global sorted key ``row * n + col`` for every stored pattern entry.

    CSR storage is already row-major with ascending columns per row, so
    ``row * n + col`` is already strictly increasing in storage order - no
    sort needed, just the key computation. Since the factorization's
    sparsity pattern is fixed throughout (zero fill-in), this is computed
    once upfront and reused by every ``_lookup`` call.

    Args:
        row_index (torch.Tensor): int64, one row index per stored entry
            (storage order).
        col_index (torch.Tensor): int64, one column index per stored entry
            (storage order), same shape as ``row_index``.
        n (int): Matrix dimension.

    Returns:
        torch.Tensor: int64, sorted ascending, one key per stored entry.
    """
    return row_index * n + col_index


def _lookup(
    keys: torch.Tensor,
    values: torch.Tensor,
    query_rows: torch.Tensor,
    query_cols: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve a batch of ``(row, col)`` queries against the global sorted keys.

    Replaces the former one-row-at-a-time ``_row_value_at``: a single
    ``torch.searchsorted`` call resolves an entire batch of queries at
    once, regardless of how many distinct rows they touch - exactly what
    per-level, per-degree-position batching needs (a batch's queries
    routinely span every row active in a level).

    Args:
        keys (torch.Tensor): int64, sorted ascending, from ``_build_lookup``.
        values (torch.Tensor): Values aligned with ``keys`` (same storage
            order as the pattern ``keys`` was built from).
        query_rows (torch.Tensor): int64, rows to look up.
        query_cols (torch.Tensor): int64, columns to look up, same shape as
            ``query_rows``.
        n (int): Matrix dimension (must match the ``n`` used to build
            ``keys``).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(found_mask, gathered_values)``,
            both shaped like ``query_rows``; ``gathered_values`` is 0
            wherever ``found_mask`` is ``False``.
    """
    query_keys = query_rows * n + query_cols
    positions = torch.searchsorted(keys, query_keys)
    clamped = positions.clamp(max=keys.numel() - 1)
    found = (positions < keys.numel()) & (keys[clamped] == query_keys)
    gathered = torch.where(found, values[clamped], torch.zeros_like(values[clamped]))
    return found, gathered


def _csr_from_boolean_submask(matrix: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Rebuild a sparse CSR tensor from a boolean submask of ``matrix``'s own entries.

    ``crow_indices`` is recomputed via ``torch.bincount``+``torch.cumsum`` of
    the per-row kept-entry counts; ``col_indices``/``values`` are a plain
    boolean-mask selection, which preserves per-row ascending column order
    (a subsequence of an already-sorted sequence stays sorted).

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


def sparse_ic0(matrix: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Compute a sparse zero-fill incomplete Cholesky (IC(0)) factor ``L``.

    Row-oriented recurrence (see module docstring):
        ``L[i,j] = (A[i,j] - sum_{k<j, k in pattern(i) ∩ pattern(j)}
        L[i,k]*L[j,k]) / L[j,j]`` for ``j < i``, and
        ``L[i,i] = sqrt(A[i,i] - sum_{k<i, k in pattern(i)} L[i,k]^2)``.

    Args:
        matrix (torch.Tensor): Symmetric positive-definite sparse CSR system
            matrix ``A``, shape ``(n, n)``.
        threshold (float): Drop tolerance - entries with ``|value| <=
            threshold`` are treated as zero and excluded from the sparsity
            pattern (matches the dense sibling's ``ic0_sparsity_mask``).

    Returns:
        torch.Tensor: Sparse CSR lower-triangular incomplete Cholesky
            factor ``L``.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR, or if elimination hits
            a non-positive pivot (IC(0) breakdown).
    """
    _require_csr(matrix, "sparse_ic0")

    n = matrix.shape[0]
    row_index = _expand_row_index(matrix)
    col = matrix.col_indices()
    values = matrix.values()
    keep = (col <= row_index) & (values.abs() > threshold)
    pattern = _csr_from_boolean_submask(matrix, keep)

    pat_row = _expand_row_index(pattern)
    pat_col = pattern.col_indices()
    pat_crow = pattern.crow_indices()
    row_start = pat_crow[:-1]
    row_end = pat_crow[1:]
    diag_pos = row_end - 1  # flat position of each row's own diagonal entry
    row_offdiag_count = row_end - row_start - 1  # off-diagonal entries per row

    keys = _build_lookup(pat_row, pat_col, n)

    # ``values`` is rebuilt out-of-place at every step (``torch.scatter``,
    # never ``scatter_``/indexed assignment): slicing the same backing
    # storage for a lookup and then writing back into it in-place would
    # bump that storage's autograd version between the forward read and
    # the later backward pass, breaking gradient tracking. A fresh tensor
    # per step has no such aliasing hazard.
    values = pattern.values()

    schedule: LevelSchedule = level_schedule(pattern, direction="forward")
    level_sizes = schedule.level_sizes.tolist()
    row_order = schedule.row_order

    offset = 0
    for level_size in level_sizes:
        level_rows = row_order[offset : offset + level_size]
        offset += level_size

        degrees = row_offdiag_count[level_rows]
        max_degree = int(degrees.max()) if level_size > 0 else 0
        dot_diag = torch.zeros(level_size, dtype=values.dtype, device=values.device)

        for d in range(max_degree):
            active_mask = degrees > d
            active_rows = level_rows[active_mask]
            active_start = row_start[active_rows]
            flat_pos_d = active_start + d
            j_d = pat_col[flat_pos_d]

            if d > 0:
                k_idx = torch.arange(d, device=values.device)
                pos2d = active_start.unsqueeze(1) + k_idx.unsqueeze(0)
                k_cols = pat_col[pos2d]
                l_row_k = values[pos2d]
                query_rows = j_d.unsqueeze(1).expand(-1, d).reshape(-1)
                query_cols = k_cols.reshape(-1)
                found, l_j_k = _lookup(keys, values, query_rows, query_cols, n)
                found = found.reshape(-1, d)
                l_j_k = l_j_k.reshape(-1, d)
                dot = (l_row_k * l_j_k * found).sum(dim=1)
            else:
                dot = torch.zeros(active_rows.numel(), dtype=values.dtype, device=values.device)

            l_jj = values[diag_pos[j_d]]
            a_val = values[flat_pos_d]
            new_val = (a_val - dot) / l_jj
            values = values.scatter(0, flat_pos_d, new_val)

            level_local_idx = torch.arange(level_size, device=values.device)[active_mask]
            dot_diag = dot_diag.scatter_add(0, level_local_idx, new_val**2)

        level_diag_pos = diag_pos[level_rows]
        pivot = values[level_diag_pos] - dot_diag
        if bool((pivot <= 0).any()):
            bad = int(level_rows[(pivot <= 0).nonzero(as_tuple=True)[0][0]])
            bad_value = float(pivot[(pivot <= 0).nonzero(as_tuple=True)[0][0]])
            raise ValueError(
                f"IC(0) breakdown at pivot {bad}: diagonal value {bad_value} is "
                "non-positive, so no real factor L exists for this matrix "
                "(Saad, Iterative Methods for Sparse Linear Systems, 2nd ed., "
                "Sec. 10.3) - not every SPD matrix admits an IC(0) factorization."
            )
        values = values.scatter(0, level_diag_pos, torch.sqrt(pivot))

    return torch.sparse_csr_tensor(
        pattern.crow_indices(),
        pattern.col_indices(),
        values,
        size=matrix.shape,
        check_invariants=False,
    )


class IC0Preconditioner(LinearPreconditioner[torch.Tensor], nn.Module):
    """Zero-level incomplete Cholesky preconditioner for sparse CSR SPD systems.

    Sparse-CSR sibling of the dense
    ``preconditioners.implementations.ic0.IC0Preconditioner`` - same name,
    disambiguated by package (matches ``torch.mm`` vs. ``torch.sparse.mm``'s
    own convention), never imported by the dense class. Computes IC(0)
    factorization: ``L @ L.T ≈ A`` where ``L`` maintains the sparsity
    pattern of the lower triangle of ``A``.

    Follows the ``Preconditioner`` + ``nn.Module`` pattern established by
    the dense preconditioners: ``nn.Module.__init__()`` is called explicitly
    and the factor ``L`` is stored via ``register_buffer`` (never
    ``register_parameter`` - it is never learned or autograd-tracked) so it
    participates in ``.to(device=..., dtype=...)`` propagation.

    Mathematical background:
        Preconditioner ``M = L @ L.T``. Solve ``M z = r`` via forward
        ``L y = r`` then backward ``L.T z = y``, both level-scheduled sparse
        triangular solves.

    Complexity:
        - Storage: ``O(nnz(L))`` - a true sparse CSR structure, not a dense
          ``(n, n)`` buffer.
        - Setup: ``O(sum of row_degree^2)`` arithmetic work - ``O(nnz)`` for
          bounded-degree matrices (FEM-like stencils); the driving Python
          loop itself is bounded by ``num_levels * max_row_degree_in_level``,
          not ``nnz`` or row count - see the module docstring's per-level,
          per-degree-position batching description.
        - Application: two ``O(nnz(L))`` level-scheduled triangular solves;
          the transpose and both schedules are prepared once during setup.

    Breakdown:
        IC(0) is not guaranteed to exist for every SPD matrix (it can
        require a square root of a non-positive diagonal value).
        Construction raises ``ValueError`` as soon as elimination hits a
        non-positive pivot, instead of letting ``nan``/``inf`` propagate
        silently through ``apply()``.

    Attributes:
        _operator (torch.Tensor): Sparse CSR lower triangular factor ``L``.

    Example:
        >>> import torch
        >>> matrix = (torch.eye(3, dtype=torch.float64) * 2.0).to_sparse_csr()
        >>> precond = IC0Preconditioner()
        >>> precond.setup(matrix)
        >>> z = precond.apply(torch.ones(3, dtype=torch.float64))
    """

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD) -> None:
        """Configure IC(0) preconditioner; the factorization is built by ``setup()``.

        Args:
            threshold (float): Drop tolerance - entries with ``|value| <=
                threshold`` are treated as zero. Default: ``0.0`` (only
                exact zeros are dropped; see ``_DEFAULT_THRESHOLD``).
        """
        nn.Module.__init__(self)
        self._threshold = threshold
        self._operator: torch.Tensor
        self._solve_cache: SparseTriangularSolveCache
        self.register_buffer("_operator", None)

    def setup(
        self,
        matrix: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> Self:
        """Compute and register the sparse IC(0) factorization of ``matrix``.

        Args:
            matrix (torch.Tensor): Symmetric positive-definite sparse CSR
                system matrix ``A``, shape ``(n, n)``.
            context (PreconditionerContext | None): Ignored.
        """
        operator = self._compute_operator(matrix)
        solve_cache = build_sparse_cholesky_solve_cache(operator)
        self._operator = operator
        self._solve_cache = solve_cache
        self._mark_ready()
        return self

    def _compute_operator(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute the sparse IC(0) factorization of the system matrix.

        Args:
            matrix (torch.Tensor): Symmetric positive-definite sparse CSR
                system matrix ``A``.

        Returns:
            torch.Tensor: Sparse CSR lower triangular incomplete Cholesky
                factor ``L``.
        """
        return sparse_ic0(matrix, self._threshold)

    @property
    def threshold(self) -> float:
        """Drop tolerance used to build the sparsity pattern.

        Returns:
            float: The value passed at construction.
        """
        return self._threshold

    def __str__(self) -> str:
        """Human-readable structural summary.

        Returns:
            str: e.g. ``"IC0(threshold=0e+00)"``.
        """
        return f"IC0(threshold={self._threshold:.0e})"

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Solve ``(L @ L.T) z = r`` via forward/backward sparse triangular solves.

        Delegates to the setup-owned ``_triangular.SparseTriangularSolveCache``
        shared with ``ICholeskyPreconditioner``, which solves the identical
        ``(L @ L.T) z = r`` system for an externally supplied ``L`` rather
        than one computed by ``sparse_ic0``.

        Args:
            residual (torch.Tensor): Residual vector(s) ``r``, shape
                ``(n,)`` or ``(n, k)``.
            context (PreconditionerContext | None): Ignored (IC(0) doesn't
                need context).

        Returns:
            torch.Tensor: Preconditioned residual ``z = M^{-1}r``.
        """
        return self._solve_cache.solve(self._operator, residual)
