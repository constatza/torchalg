"""Sparse-CSR symmetric Gauss-Seidel, prepared once per matrix.

No factorization step: each sweep is a direct triangular solve against ``A``'s own
entries, split by position relative to the diagonal. Mirrors the dense
``_relaxation.py``'s D/L/U convention: forward sweep solves ``(D + L) x_new = rhs - U @
x_old``, backward solves ``(D + U) x_new = rhs - L @ x_old``, one iteration = one forward
sweep followed by one backward sweep.

``prepare_csr_gauss_seidel`` does every computation on the matrix once and returns the
frozen ``PreparedSymmetricGaussSeidel``; ``smooth``/``smooth_from_zero`` then only sweep.
Every sweep is a vendor triangular solve (``kernels.triangular.prepare_triangular_solve``,
resolved per device through ``TRIANGULAR_SOLVER_BY_DEVICE``) on int32-indexed triangles.

The caller picks how a sweep gets its strict-triangle products (``SweepProducts``).
With ``FROM_SOLVES`` and every pivot nonzero, each solve satisfies its triangular system,
so the products come from the solves themselves: ``(D + L) x_f = rhs - U x_old`` gives
``rhs - L x_f = U x_old + D x_f``, and ``(D + U) x_b = t_b`` gives ``U x_b = t_b - D x_b``.
A call then applies ``U`` once (to the incoming iterate) and no strict product per step;
the results agree with the explicit products to rounding (about ``1e-16``). ``EXPLICIT``
applies ``L`` and ``U`` every sweep. Setup stages (compatible relaxation, adaptive and
bootstrap test-vector relaxation, their setup cycles) use ``EXPLICIT``: their coarsening
and fitting amplify that rounding to ``1e-10`` in the built hierarchy, so explicit
products keep setups bit-identical to the textbook sweep. The solve-time smoother uses
``FROM_SOLVES``, where the rounding stays at ``1e-16``. A zero pivot freezes its row to
the target, which breaks both identities, so such a matrix always applies the products
explicitly.

A masked sweep (``rows=``, the adaptive smoothed-aggregation general stage) runs on
triangles whose frozen rows are identity rows, solved for the target ``current`` on those
rows - the same construction as the dense ``_relaxation._prepare_sweep``. Its mask is
drawn from the iterate (the nonzero rows of a prolongated candidate), so it differs on
every call and a per-mask cache would never hit. Each call instead derives both
identity-row solves from the prepared ones (``with_identity_rows``): the vendor triangles
keep their pattern and only their values are masked, a level-scheduled solve only widens
its freeze mask, so no triangle is sorted or rebuilt. Callers that relax the same matrix
many times (smoothers, compatible relaxation, adaptive setup stages) hold one prepared
object per matrix instead of rebuilding per call.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import torch

from torchalg.sparse.kernels.diagonal import sparse_diagonal
from torchalg.sparse.kernels.triangular import (
    ZERO_PIVOT_TOL,
    TriangularSolver,
    _expand_row_index,
    _require_csr,
    prepare_triangular_solve,
)
from torchalg.utils.csr_index import compact_csr_indices


def _strict_triangle(
    row_index: torch.Tensor, matrix: torch.Tensor, keep: torch.Tensor
) -> torch.Tensor:
    """CSR matrix of the stored entries of ``matrix`` selected by ``keep``.

    Args:
        row_index (torch.Tensor): Row of each stored entry of ``matrix``.
        matrix (torch.Tensor): Sparse CSR matrix ``A``, shape ``(n, n)``.
        keep (torch.Tensor): Boolean mask over the stored entries.

    Returns:
        torch.Tensor: Row-wise filter of ``matrix`` (int32 indices where they fit).
    """
    n = matrix.shape[0]
    crow = torch.zeros(n + 1, dtype=row_index.dtype, device=matrix.device)
    crow[1:] = torch.cumsum(torch.bincount(row_index[keep], minlength=n), 0)
    # A row-wise filter of a valid CSR is valid; checking it would add O(nnz) to the setup.
    triangle = torch.sparse_csr_tensor(
        crow,
        matrix.col_indices()[keep],
        matrix.values()[keep],
        matrix.shape,
        check_invariants=False,
    )
    return compact_csr_indices(triangle)


class SweepProducts(Enum):
    """How a sweep gets the strict-triangle products ``L x`` and ``U x``.

    Attributes:
        FROM_SOLVES: Read them off the triangular solves (solve-time smoothing).
        EXPLICIT: Apply ``L`` and ``U`` every sweep (setup stages, bit-identical setups).
    """

    FROM_SOLVES = auto()
    EXPLICIT = auto()


@dataclass(frozen=True, slots=True)
class _ProductsFromSolves:
    """Every pivot is nonzero and ``FROM_SOLVES`` was chosen: sweeps skip strict products.

    Attributes:
        diagonal (torch.Tensor): ``D``, the diagonal of ``A``, shape ``(n,)``.
    """

    diagonal: torch.Tensor


@dataclass(frozen=True, slots=True)
class _ExplicitProducts:
    """Sweeps apply ``L`` and ``U``: ``EXPLICIT`` was chosen or some pivot is zero.

    Attributes:
        frozen (torch.Tensor): Boolean mask of the zero-pivot rows, which freeze to the
            target, shape ``(n,)``; all ``False`` when every pivot is nonzero.
    """

    frozen: torch.Tensor


def _as_column(diagonal: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return diagonal.unsqueeze(-1) if like.ndim == 2 else diagonal


@dataclass(frozen=True, slots=True)
class PreparedSymmetricGaussSeidel:
    """Symmetric Gauss-Seidel bound to one sparse CSR matrix.

    Create with ``prepare_csr_gauss_seidel``; every field is computed from the
    matrix once and captures its values.

    Attributes:
        matrix (torch.Tensor): The sparse CSR matrix ``A``, shape ``(n, n)``, with
            int32 indices where it fits.
        upper (torch.Tensor): Strict upper triangle ``U`` as sparse CSR.
        lower (torch.Tensor): Strict lower triangle ``L`` as sparse CSR (explicit and
            masked sweeps only).
        forward (TriangularSolver): ``(D + L)`` substitution of the forward sweep.
        backward (TriangularSolver): ``(D + U)`` substitution of the backward sweep.
        products (_ProductsFromSolves | _ExplicitProducts): How a sweep gets its
            strict-triangle products.

    Cost:
        Apply, one symmetric iteration (forward plus backward sweep), ``k`` columns,
        ``z = nnz``, ``V = 8`` bytes per ``float64``, ``w`` the index width,
        ``lv_f`` and ``lv_b`` the dependency levels of ``D + L`` and ``D + U``.
        Products from the solves: ``W = (2(z + n) + 5n)k`` per iteration (two
        triangular solves plus three ``O(n)`` vector kernels: one subtraction and
        two ``addcmul``), plus ``(z - n)k`` once per ``smooth`` call for ``U x``;
        ``smooth_from_zero`` applies no strict product. ``Q = (z + n)(V + w)`` per
        iteration plus ``O(Vnk)`` vector passes, ``D = lv_f + lv_b + 3`` (each solve
        is a recurrence over its levels), ``L = 5`` per iteration plus one per
        ``smooth`` call, and ``S = 0``.
        Explicit products (setup stages, zero pivots): two strict-triangle products
        per iteration, ``W = (4z + 2n)k``, ``Q = 2z(V + w)`` per iteration,
        ``D = lv_f + lv_b + 2 ceil(log2 m) + O(1)`` (``m`` the largest row count),
        ``L = 6`` per iteration, one product fewer for ``smooth_from_zero``.
        A masked call (``rows=``) masks the values of its two triangles once
        (``W = 0``, ``Q = O(z(V + w))``, ``L = 18``, no sort) and then sweeps at
        ``L = 8`` per iteration.
        Roofline: ``W/Q`` is about 0.16 flop per byte (memory-bound).
        Work-depth: ``W/D = (2z + 7n)k / (lv_f + lv_b)``, about ``10^3`` at
        ``nnz/n`` near 37, far below SpMV and Jacobi: the depth, not the flops,
        limits parallel speed-up. The library solves keep ``L`` independent of the
        depth; on a device without a registered library solve each solve is
        level-scheduled (``PreparedTriangularSolve``, ``L = O(lv)``).
    """

    matrix: torch.Tensor
    upper: torch.Tensor
    lower: torch.Tensor
    forward: TriangularSolver
    backward: TriangularSolver
    products: _ProductsFromSolves | _ExplicitProducts

    def residual(self, rhs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return ``rhs - A x`` with the bound level matrix.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            x (torch.Tensor): Current iterate, same shape as ``rhs``.

        Returns:
            torch.Tensor: Residual, same shape as ``rhs``.
        """
        return rhs - self.matrix @ x

    def smooth(
        self,
        rhs: torch.Tensor,
        x: torch.Tensor,
        steps: int,
        *,
        rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply ``steps`` symmetric Gauss-Seidel iterations from ``x``.

        Cost: per iteration two vendor triangular solves; ``U x`` once per call with products
        from the solves, two strict-triangle products per iteration otherwise (class
        ``Cost:``).

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            x (torch.Tensor): Current iterate, same shape; not modified.
            steps (int): Number of symmetric iterations.
            rows (torch.Tensor | None): If given, only these rows (indices) are
                updated - sparse sibling of the dense indexed sweep (PyAMG's
                ``gauss_seidel_indexed``). Rows outside ``rows`` stay frozen
                at their current value for the whole call.

        Returns:
            torch.Tensor: Updated iterate, same shape as ``x``.
        """
        if steps == 0:
            return x.clone()
        if rows is not None:
            return self._smooth_masked(rhs, x, steps, rows)
        match self.products:
            case _ProductsFromSolves(diagonal=diagonal):
                return self._exact_sweeps(rhs, self.upper @ x, steps, diagonal)
            case _ExplicitProducts():
                return self._explicit_sweeps(rhs, x, steps)

    def smooth_from_zero(self, rhs: torch.Tensor, steps: int) -> torch.Tensor:
        """Apply ``steps`` symmetric iterations from the zero guess.

        The first forward sweep's right-hand side is ``rhs - U @ 0 = rhs``, so
        that strict-triangle matvec is skipped.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            steps (int): Number of symmetric iterations.

        Returns:
            torch.Tensor: Iterate after ``steps`` iterations, same shape as ``rhs``.
        """
        if steps == 0:
            return torch.zeros_like(rhs)
        match self.products:
            case _ProductsFromSolves(diagonal=diagonal):
                return self._exact_sweeps(rhs, None, steps, diagonal)
            case _ExplicitProducts():
                x = self.backward.solve(rhs - self.lower @ self.forward.solve(rhs))
                return self._explicit_sweeps(rhs, x, steps - 1)

    def _exact_sweeps(
        self,
        rhs: torch.Tensor,
        upper_x: torch.Tensor | None,
        steps: int,
        diagonal: torch.Tensor,
    ) -> torch.Tensor:
        """Sweep pairs whose strict-triangle products come from the solves.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            upper_x (torch.Tensor | None): ``U x_old``, or ``None`` for the zero iterate.
            steps (int): Number of symmetric iterations, at least one.
            diagonal (torch.Tensor): ``D``, shape ``(n,)``.

        Returns:
            torch.Tensor: Iterate after ``steps`` iterations.
        """
        d = _as_column(diagonal, rhs)
        x = rhs
        for _ in range(steps):
            if upper_x is None:
                # ``rhs - L x_f = D x_f`` when ``x_old = 0``.
                x_forward = self.forward.solve(rhs)
                backward_target = d * x_forward
            else:
                # ``rhs - L x_f = U x_old + D x_f``.
                x_forward = self.forward.solve(rhs - upper_x)
                backward_target = torch.addcmul(upper_x, d, x_forward)
            x = self.backward.solve(backward_target)
            # ``U x_b = t_b - D x_b``: the next forward sweep's strict product.
            upper_x = torch.addcmul(backward_target, d, x, value=-1)
        return x

    def _explicit_sweeps(self, rhs: torch.Tensor, x: torch.Tensor, steps: int) -> torch.Tensor:
        """Sweep pairs with explicit strict-triangle products.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            x (torch.Tensor): Current iterate, same shape.
            steps (int): Number of symmetric iterations.

        Returns:
            torch.Tensor: Iterate after ``steps`` iterations.
        """
        for _ in range(steps):
            x_forward = self.forward.solve(rhs - self.upper @ x)
            x = self.backward.solve(rhs - self.lower @ x_forward)
        return x

    def _smooth_masked(
        self, rhs: torch.Tensor, x: torch.Tensor, steps: int, rows: torch.Tensor
    ) -> torch.Tensor:
        """Sweep only ``rows``; every other row keeps its value from ``x``.

        Inactive rows and zero-pivot rows become identity rows of both triangles; the
        target on an inactive row is its current value, on a zero-pivot row the sweep's own
        target (the unmasked freeze).

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            x (torch.Tensor): Current iterate, same shape.
            steps (int): Number of symmetric iterations, at least one.
            rows (torch.Tensor): Indices of the updatable rows.

        Returns:
            torch.Tensor: Iterate after ``steps`` iterations.
        """
        inactive = torch.ones(self.matrix.shape[0], dtype=torch.bool, device=x.device)
        inactive[rows] = False
        frozen = inactive
        if isinstance(self.products, _ExplicitProducts):
            frozen = frozen | self.products.frozen
        forward = self.forward.with_identity_rows(frozen)
        backward = self.backward.with_identity_rows(frozen)
        keep = _as_column(inactive, x)
        for _ in range(steps):
            x_forward = forward.solve(torch.where(keep, x, rhs - self.upper @ x))
            x = backward.solve(torch.where(keep, x_forward, rhs - self.lower @ x_forward))
        return x


def prepare_csr_gauss_seidel(
    matrix: torch.Tensor, products: SweepProducts
) -> PreparedSymmetricGaussSeidel:
    """Split ``matrix`` into its strict triangles and prepare both vendor substitutions.

    Cost:
        Once per matrix, ``z = nnz``, ``w`` the index width, ``V = 8`` bytes per
        ``float64``: four triangle extractions (strict ``L`` and ``U`` here, ``D + L``
        and ``D + U`` in ``prepare_triangular_solve``), each a comparison, a bincount, a
        cumulative sum and three masked selects over the stored entries, plus three
        masked diagonal scatters. ``W = 3z + O(n)`` (the split itself only moves
        entries; the library solve has no separate analysis to pay here),
        ``Q = O(z(V + w))``, ``D = O(log z)`` (a constant chain of maps, scans and
        compactions), about ``L = 100`` and ``S = 15`` with ``FROM_SOLVES``: three count
        reads per masked triangle extraction, one zero-pivot read per
        ``prepare_triangular_solve`` and one here (``S = 14`` for ``EXPLICIT``, which
        skips that read). A device without a registered library solve builds a level
        schedule per direction instead (``level_schedule``'s Cost, an ``O(n + nnz)``
        host recurrence).

    Args:
        matrix (torch.Tensor): Sparse CSR matrix ``A``, shape ``(n, n)``.
        products (SweepProducts): How a sweep gets its strict-triangle products; a zero
            pivot forces ``EXPLICIT``.

    Returns:
        PreparedSymmetricGaussSeidel: Reusable sweeps over ``matrix``.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR.
    """
    _require_csr(matrix, "prepare_csr_gauss_seidel")
    # int32 triangles: the CPU library solve otherwise narrows int64 indices on every call.
    compact = compact_csr_indices(matrix)
    row_index = _expand_row_index(matrix)
    col = matrix.col_indices()
    diagonal = sparse_diagonal(matrix)
    zero_pivot = diagonal.abs() <= ZERO_PIVOT_TOL
    sweep_products: _ProductsFromSolves | _ExplicitProducts = _ExplicitProducts(zero_pivot)
    if products is SweepProducts.FROM_SOLVES and not bool(zero_pivot.any()):
        sweep_products = _ProductsFromSolves(diagonal)
    return PreparedSymmetricGaussSeidel(
        matrix=compact,
        upper=_strict_triangle(row_index, matrix, col > row_index),
        lower=_strict_triangle(row_index, matrix, col < row_index),
        forward=prepare_triangular_solve(compact, "forward"),
        backward=prepare_triangular_solve(compact, "backward"),
        products=sweep_products,
    )
