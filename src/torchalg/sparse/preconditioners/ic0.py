"""Zero-level incomplete Cholesky (IC(0)) preconditioner, sparse-CSR sibling.

Sparse-CSR counterpart of
``preconditioners.implementations.ic0.IC0Preconditioner`` (dense; kept
unmodified for comparison).
Same algorithm, separate implementation, tested against the dense result: the
sparse tree never imports the dense tree (enforced by ``tach.toml``). Row-
oriented ("left-looking") level-scheduled factorization - not a sparse
transliteration of the dense column-oriented ("right-looking") Crout loop:
each row depends only on already-finished rows, so the rows of one dependency
level factor in one batch. Rows are processed in
``kernels.triangular.level_schedule``'s row order - not strict ``0..n-1``
order - since the recurrence's result is independent of visitation order as
long as every row's dependencies are visited first (same property
``triangular_solve`` already relies on).

Both the factorization and ``apply()`` are differentiable by construction:
every value computed into the output buffer is a composition of
autograd-tracked tensor ops; ``.item()``/``.tolist()`` are used only for
Python-level control flow (loop bounds, breakdown checks), never to compute
a value that ends up in the returned factor.

Everything that depends only on the sparsity pattern (schedule, gather
positions, transpose permutation) is planned once by
``_ic0_pattern.analyze_ic0``; the numeric loop below consumes that
``IC0Pattern`` and is bounded by ``num_levels * max_row_degree_in_level``
(outer loop over levels, inner loop over degree-position ``d``), not by
``nnz`` - every row in a level that has a ``d``-th off-diagonal entry is
updated in one vectorized batch per ``d``.
"""

from __future__ import annotations

from typing import Self

import torch
from torch import nn

from torchalg.preconditioners.base import (
    MatrixSetupPreconditioner,
    Placement,
    PreconditionerContext,
)
from torchalg.sparse.kernels.triangular import _require_csr
from torchalg.sparse.preconditioners._ic0_pattern import IC0Pattern, analyze_ic0
from torchalg.sparse.preconditioners._triangular import (
    PreparedTriangularPair,
    build_sparse_cholesky_solve_cache,
    tracks_autograd,
    write_positions,
)


def _factorize(matrix: torch.Tensor, pattern: IC0Pattern) -> torch.Tensor:
    """Run the IC(0) recurrence over ``pattern``'s precomputed gather positions.

    Cost:
        W: ``O(sum of row_degree^2)`` arithmetic, as the algorithmic minimum.
        Executed: each elimination step writes only its own positions (no
        whole-array copy) unless autograd tracks the matrix values.
        S: one host sync per call (the breakdown test after the last level),
        not one per level.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix whose structure ``pattern`` matches.
        pattern (IC0Pattern): Plan from ``analyze_ic0``.

    Returns:
        torch.Tensor: Sparse CSR lower-triangular factor ``L``.

    Raises:
        ValueError: If elimination hits a non-positive pivot (IC(0) breakdown).
    """
    # The out-of-place path (a fresh tensor per step) exists only for autograd:
    # slicing the same backing storage for a gather and then writing back into
    # it in place would bump that storage's version between the forward read
    # and the later backward pass. Without tracking there is nothing to save,
    # and ``values`` is already a private copy (an integer-index gather).
    in_place = not tracks_autograd(matrix)
    values = matrix.values()[pattern.lower_positions]
    level_pivots: list[torch.Tensor] = []

    for level in pattern.levels:
        dot_diag = torch.zeros(level.rows.numel(), dtype=values.dtype, device=values.device)

        for step in level.steps:
            numerator = values[step.flat_pos]
            if step.gather is not None:
                gather = step.gather
                # Unfound ``(j, k)`` positions point at storage index 0; masking the
                # gathered value (not multiplying the product by ``found``) keeps an
                # ``inf``/``nan`` stored there from leaking in as ``inf * 0 = nan``.
                l_jk = torch.where(gather.found, values[gather.pos_jk], 0.0)
                dot = (values[gather.pos_ik] * l_jk).sum(dim=1)
                numerator = numerator - dot

            new_val = numerator / values[step.diag_pos_of_j]
            values = write_positions(values, step.flat_pos, new_val, in_place=in_place)
            dot_diag = dot_diag.scatter_add(0, step.level_local_idx, new_val**2)

        pivot = values[level.diag_pos] - dot_diag
        level_pivots.append(pivot)
        values = write_positions(values, level.diag_pos, torch.sqrt(pivot), in_place=in_place)

    _raise_on_breakdown(pattern, level_pivots)
    return torch.sparse_csr_tensor(
        pattern.lower_crow,
        pattern.lower_col,
        values,
        size=matrix.shape,
        check_invariants=False,
    )


def _raise_on_breakdown(pattern: IC0Pattern, level_pivots: list[torch.Tensor]) -> None:
    """Raise for the first non-positive pivot in level order, with one host sync.

    Elimination does not stop at a breakdown. After a zero pivot, later pivots
    that depend on it become ``-inf`` (the division by ``sqrt(0)`` feeds
    ``inf**2`` into their dot products) and so also compare non-positive;
    after a negative one they become ``nan``. Either way ``torch.cat`` keeps
    level order and a pivot depends only on earlier levels, so the first
    non-positive pivot found here is the first that occurred.

    Args:
        pattern (IC0Pattern): Plan whose levels produced ``level_pivots``.
        level_pivots (list[torch.Tensor]): One pivot vector per level, level order.

    Raises:
        ValueError: If any pivot is non-positive.
    """
    if not level_pivots:
        return
    pivots = torch.cat(level_pivots)
    non_positive = pivots <= 0
    if not bool(non_positive.any()):
        return
    first = int(non_positive.nonzero(as_tuple=True)[0][0])
    bad = int(torch.cat([level.rows for level in pattern.levels])[first])
    raise ValueError(
        f"IC(0) breakdown at pivot {bad}: diagonal value {float(pivots[first])} is "
        "non-positive, so no real factor L exists for this matrix "
        "(Saad, Iterative Methods for Sparse Linear Systems, 2nd ed., "
        "Sec. 10.3) - not every SPD matrix admits an IC(0) factorization."
    )


def sparse_ic0(matrix: torch.Tensor, pattern: IC0Pattern | None = None) -> torch.Tensor:
    """Compute a sparse zero-fill incomplete Cholesky (IC(0)) factor ``L``.

    Row-oriented recurrence (see module docstring):
        ``L[i,j] = (A[i,j] - sum_{k<j, k in pattern(i) ∩ pattern(j)}
        L[i,k]*L[j,k]) / L[j,j]`` for ``j < i``, and
        ``L[i,i] = sqrt(A[i,i] - sum_{k<i, k in pattern(i)} L[i,k]^2)``.

    The sparsity pattern is the stored lower-triangle entries with a non-zero value.

    Args:
        matrix (torch.Tensor): Symmetric positive-definite sparse CSR system
            matrix ``A``, shape ``(n, n)``.
        pattern (IC0Pattern | None): Plan from ``analyze_ic0`` for a matrix
            with the same structure; ``None`` analyzes ``matrix`` here, which
            suits one-off use.

    Returns:
        torch.Tensor: Sparse CSR lower-triangular incomplete Cholesky
            factor ``L``.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR, if ``pattern`` does not
            match ``matrix``, or if elimination hits a non-positive pivot
            (IC(0) breakdown).
    """
    _require_csr(matrix, "sparse_ic0")
    if pattern is None:
        pattern = analyze_ic0(matrix)
    elif not pattern.matches(matrix):
        raise ValueError(
            "sparse_ic0: pattern does not match the matrix's structure or non-zero mask"
        )
    return _factorize(matrix, pattern)


class IC0Preconditioner(MatrixSetupPreconditioner, nn.Module):
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
          per-degree-position batching description. The first ``setup`` on a
          pattern also pays ``analyze_ic0`` (the same order of work); later
          ones reuse it.
        - Application: two ``O(nnz(L))`` level-scheduled triangular solves;
          the transpose and both schedules are prepared once during setup.

    Cost:
        Setup, once per matrix values (the pattern analysis, ``O(n + nnz)`` host index
        work, is reused for later matrices of the same structure), ``V = 8`` bytes per
        ``float64``: algorithmic ``W = sum_{(i,j)} (2 c_ij + 1) + sum_i (2 d_i + 2)``
        over the lower pattern, with ``c_ij`` the number of common earlier columns of
        rows ``i`` and ``j`` and ``d_i`` the row degree, and ``Q = 2V nnz(L)``. The
        executed loop is one batch per (level, degree position): depth
        ``D = sum over levels of the maximum row degree``, ``L`` proportional to that
        step count (a constant dozen kernels per step), and ``S = 1`` (the breakdown
        test after the last level). The padded gathers and the found-mask make the
        executed ``W`` a small multiple of the algorithmic count. Setup is
        latency-bound (``L`` and ``D`` dominate), not flop-bound.
        Apply, ``k`` right-hand-side columns, ``w`` the index width: two sparse
        triangular solves, ``W = 4 nnz(L) k``,
        ``Q = 2 [nnz(L)(V + w) + (n + 1)w + 2Vnk]``, ``D = lv_f + lv_b``
        (about ``2 lv(L)``, each solve a recurrence over its dependency levels),
        ``L = 2`` through the vendor solve (CPU and CUDA; the level-scheduled solve
        has ``L = O(lv)`` per solve) and ``S = 0``.
        Roofline: apply ``W/Q`` is about 0.17 flop per byte (memory-bound).
        Work-depth: apply ``W/D = 2 nnz(L) k / lv(L)``, the lowest of the methods
        here: depth, not flops, limits parallel speed-up, so the solve stays on the
        CPU or a library kernel (``Placement.SEQUENTIAL_SWEEP``).

    Breakdown:
        IC(0) is not guaranteed to exist for every SPD matrix (it can
        require a square root of a non-positive diagonal value).
        Construction raises ``ValueError`` as soon as elimination hits a
        non-positive pivot, instead of letting ``nan``/``inf`` propagate
        silently through ``apply()``.

    Pattern reuse:
        The instance keeps the ``IC0Pattern`` of its last ``setup`` and reuses it
        for the next matrix whose stored structure and non-zero mask are
        unchanged (``IC0Pattern.matches``); otherwise it analyzes anew. Re-running
        ``setup`` after the values change therefore skips all structural work.

    Attributes:
        _operator (torch.Tensor): Sparse CSR lower triangular factor ``L``.
        _pattern (IC0Pattern | None): Pattern of the last successful ``setup``.

    Example:
        >>> import torch
        >>> matrix = (torch.eye(3, dtype=torch.float64) * 2.0).to_sparse_csr()
        >>> precond = IC0Preconditioner()
        >>> precond.setup(matrix)
        >>> z = precond.apply(torch.ones(3, dtype=torch.float64))
    """

    placement = Placement.SEQUENTIAL_SWEEP

    def __init__(self) -> None:
        """Configure IC(0) preconditioner; the factorization is built by ``setup()``."""
        nn.Module.__init__(self)
        self._pattern: IC0Pattern | None = None
        self._operator: torch.Tensor
        self._solve_cache: PreparedTriangularPair
        self.register_buffer("_operator", None)

    def setup(
        self,
        matrix: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> Self:
        """Compute and register the sparse IC(0) factorization of ``matrix``.

        State is assigned only after every step succeeded, so a failed setup
        keeps the previous complete factor, solve pair and pattern.

        Args:
            matrix (torch.Tensor): Symmetric positive-definite sparse CSR
                system matrix ``A``, shape ``(n, n)``.
            context (PreconditionerContext | None): Ignored.
        """
        pattern = self._pattern_for(matrix)
        operator = _factorize(matrix, pattern)
        solve_cache = build_sparse_cholesky_solve_cache(operator, pattern)
        self._pattern = pattern
        self._operator = operator
        self._solve_cache = solve_cache
        return self

    def _pattern_for(self, matrix: torch.Tensor) -> IC0Pattern:
        """Return the held pattern when it matches ``matrix``, else a fresh analysis."""
        if self._pattern is not None and self._pattern.matches(matrix):
            return self._pattern
        return analyze_ic0(matrix)

    @property
    def is_ready(self) -> bool:
        """Whether ``setup`` has stored a factor.

        Returns:
            bool: ``True`` once ``setup()`` has replaced the ``None`` placeholder.
        """
        return self._operator is not None

    def __str__(self) -> str:
        """Human-readable structural summary.

        Returns:
            str: ``"IC0()"``.
        """
        return "IC0()"

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Solve ``(L @ L.T) z = r`` via forward/backward sparse triangular solves.

        Delegates to the setup-owned ``_triangular.PreparedTriangularPair``
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
        return self._solve_cache.solve(residual)
