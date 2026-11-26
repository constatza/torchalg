"""Incomplete LU (ILU(0)) preconditioner, sparse-CSR sibling.

Sparse-CSR counterpart of ``preconditioners.implementations.ilu
.ILUPreconditioner`` (dense; kept unmodified for comparison).
Same algorithm, separate implementation, tested against the dense result: the
sparse tree never imports the dense tree (enforced by ``tach.toml``). Row-oriented ("left-looking")
level-scheduled factorization - see ``sparse.preconditioners.ic0``'s module
docstring for the shared design rationale.

Everything that depends only on the stored structure (schedule, gather
positions, the split into the two solve triangles) is planned once by
``_ilu0_pattern.analyze_ilu0``; the numeric loop below consumes that
``ILU0Pattern`` and is bounded by ``num_levels * max_row_degree_in_level``,
not ``nnz``, same as ``.ic0``'s batching - see that module's docstring. ILU0's
own wrinkle: a row's L-side entries (``j < row``) need exactly ``d`` of the
row's own earlier columns at degree-position ``d`` (uniform across a
batch, like IC0), but its U-side entries (``j >= row``) each need the
row's *entire* L-side count (``idx_diag``), which differs row-to-row - so
U-side batches are padded to the batch's max L-count at that step and the
padding is masked out of the sum, rather than forcing one unified,
non-uniform-batch formula.
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
from torchalg.sparse.preconditioners._ilu0_pattern import ILU0Pattern, analyze_ilu0
from torchalg.sparse.preconditioners._triangular import (
    PreparedTriangularPair,
    build_sparse_lu_solve_cache,
    tracks_autograd,
    write_positions,
)


def _factorize(matrix: torch.Tensor, pattern: ILU0Pattern) -> torch.Tensor:
    """Run the ILU(0) recurrence over ``pattern``'s precomputed gather positions.

    Cost:
        W: algorithmic minimum. Executed: each step writes only its own
        positions (no whole-array copy) unless autograd tracks the matrix values.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix whose structure ``pattern`` matches.
        pattern (ILU0Pattern): Plan from ``analyze_ilu0``.

    Returns:
        torch.Tensor: Combined ``L``/``U`` sparse CSR factor.
    """
    # See ``.ic0._factorize``'s comment on why autograd forces the out-of-place
    # path. In place, ``values`` must be a private copy: ``matrix.values()``
    # aliases the caller's storage.
    in_place = not tracks_autograd(matrix)
    values = matrix.values().clone() if in_place else matrix.values()

    for step in pattern.steps:
        numerator = values[step.flat_pos]
        if step.gather is not None:
            gather = step.gather
            # Masking the gathered value (not the product) keeps an ``inf``/``nan``
            # at the unfound positions' index 0 from leaking in as ``inf * 0 = nan``.
            u_kj = torch.where(gather.found, values[gather.pos_kj], 0.0)
            dot = (values[gather.pos_ik] * u_kj).sum(dim=1)
            numerator = numerator - dot

        u_jj = values[step.diag_pos_of_j]
        new_val = torch.where(step.is_l_side, numerator / u_jj, numerator)
        values = write_positions(values, step.flat_pos, new_val, in_place=in_place)

    return torch.sparse_csr_tensor(
        pattern.factor_crow,
        pattern.factor_col,
        values,
        size=matrix.shape,
        check_invariants=False,
    )


def sparse_ilu0(matrix: torch.Tensor, pattern: ILU0Pattern | None = None) -> torch.Tensor:
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
        pattern (ILU0Pattern | None): Plan from ``analyze_ilu0`` for a matrix
            with the same stored structure; ``None`` analyzes ``matrix`` here,
            which suits one-off use.

    Returns:
        torch.Tensor: Combined ``L``/``U`` sparse CSR factor tensor.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR or ``pattern`` does not
            match ``matrix``.
    """
    _require_csr(matrix, "sparse_ilu0")
    if pattern is None:
        pattern = analyze_ilu0(matrix)
    elif not pattern.matches(matrix):
        raise ValueError("sparse_ilu0: pattern does not match the matrix's stored structure")
    return _factorize(matrix, pattern)


class ILUPreconditioner(MatrixSetupPreconditioner, nn.Module):
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

    Pattern reuse:
        The instance keeps the ``ILU0Pattern`` of its last ``setup`` and reuses it
        for the next matrix with the same stored structure
        (``ILU0Pattern.matches``); otherwise it analyzes anew. Re-running
        ``setup`` after the values change therefore skips all structural work.

    Cost:
        Setup, once per matrix values (the pattern analysis, ``O(n + nnz)`` host index
        work, is reused for later matrices of the same structure), ``z = nnz``,
        ``V = 8`` bytes per ``float64``: algorithmic
        ``W = 2 sum_{(i,j)} |{k < min(i, j) : (i,k), (k,j) in pattern}| + nnz(L)``
        (the multiply-adds of the row recurrence plus the divisions), ``Q`` a small
        multiple of ``V z``. The executed loop is one batch per (level, degree
        position), the upper side padded to the batch's widest lower count: depth
        ``D = sum over levels of the maximum row degree``, ``L = 11 D`` (eleven
        kernels per step) plus about 50 for the two triangle gathers and the two
        prepared solves, and ``S = 8``: per solve direction one zero-pivot read and
        three count reads of the triangle extraction in ``prepare_triangular_solve``;
        the recurrence itself reads nothing back. Setup is latency-bound, not
        flop-bound.
        Apply, ``k`` columns, ``w`` the index width: two sparse triangular solves
        against ``L`` (unit diagonal stored explicitly) and ``U``,
        ``W = 2 (z + n) k``, ``Q = (z + n)(V + w) + 2(n + 1)w + 4Vnk``,
        ``D = lv_L + lv_U`` (each solve a recurrence over its dependency levels),
        ``L = 2`` through the vendor solve (CPU and CUDA; ``O(lv)`` per solve when
        level-scheduled) and ``S = 0``.
        Roofline: apply ``W/Q`` is about 0.17 flop per byte (memory-bound).
        Work-depth: apply ``W/D = (z + n) k / (lv_L + lv_U)``, low: depth, not
        flops, limits parallel speed-up (``Placement.SEQUENTIAL_SWEEP``).

    Mathematical Properties:
        - ``M ≈ A`` via incomplete LU factorization (no fill-in beyond
          ``A``'s original non-zero pattern - natural ordering, no
          pivoting).
        - ``z = (LU)^{-1}r`` via forward/backward sparse triangular solves.
        - Not guaranteed SPD even if ``A`` is SPD.

    Example:
        >>> import torch
        >>> matrix = (torch.eye(3, dtype=torch.float64) * 2.0).to_sparse_csr()
        >>> precond = ILUPreconditioner()
        >>> precond.setup(matrix)
        >>> z = precond.apply(torch.ones(3, dtype=torch.float64))
    """

    placement = Placement.SEQUENTIAL_SWEEP

    def __init__(self) -> None:
        """Register the (not yet computed) combined LU-factor buffer."""
        nn.Module.__init__(self)
        self._pattern: ILU0Pattern | None = None
        self._operator: torch.Tensor
        self._solve_cache: PreparedTriangularPair
        self.register_buffer("_operator", None)

    def setup(
        self,
        matrix: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> Self:
        """Compute and register the sparse ILU(0) factorization of ``matrix``.

        State is assigned only after every step succeeded, so a failed setup
        keeps the previous complete factor, solve pair and pattern.

        Args:
            matrix (torch.Tensor): Sparse CSR system matrix ``A``, shape
                ``(n, n)``.
            context (PreconditionerContext | None): Ignored.
        """
        pattern = self._pattern_for(matrix)
        operator = _factorize(matrix, pattern)
        solve_cache = build_sparse_lu_solve_cache(operator, pattern)

        self._pattern = pattern
        self._operator = operator
        self._solve_cache = solve_cache
        return self

    def _pattern_for(self, matrix: torch.Tensor) -> ILU0Pattern:
        """Return the held pattern when it matches ``matrix``, else a fresh analysis."""
        if self._pattern is not None and self._pattern.matches(matrix):
            return self._pattern
        return analyze_ilu0(matrix)

    @property
    def is_ready(self) -> bool:
        """Whether ``setup`` has stored a factor.

        Returns:
            bool: ``True`` once ``setup()`` has replaced the ``None`` placeholder.
        """
        return self._operator is not None

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Solve ``(LU) z = r`` via sparse forward/backward triangular solves.

        ``L``'s implicit unit diagonal (never stored in the combined
        factor) is materialized explicitly by ``build_sparse_lu_solve_cache``
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
        return self._solve_cache.solve(residual)
