"""Sparse-CSR symmetric Gauss-Seidel, prepared once per matrix.

No factorization step: each sweep is a direct triangular solve against
``A``'s own entries, split by position relative to the diagonal - see
``docs/plan.md``'s "AMG/POD smoother constraint" section. Mirrors the dense
``_relaxation.py``'s D/L/U convention: forward sweep solves ``(D + L)
x_new = rhs - U @ x_old``, backward solves ``(D + U) x_new = rhs - L @
x_old``, one iteration = one forward sweep followed by one backward sweep.

``prepare_csr_gauss_seidel`` does every computation on the matrix once and returns
the frozen ``PreparedSymmetricGaussSeidel``; ``smooth``/``smooth_from_zero`` then
only sweep. An unmasked sweep is two vendor triangular solves
(``kernels.triangular.prepare_triangular_solve``, resolved per device through
``TRIANGULAR_SOLVER_BY_DEVICE``) plus two strict-triangle SpMVs. A masked sweep
(``rows=``, compatible relaxation) has no vendor form, so it runs on the level-scheduled
``PreparedTriangularSolve``, built on the first masked sweep of a matrix. Callers that
relax the same matrix many times (smoothers, compatible relaxation, adaptive setup
stages) hold one prepared object per matrix instead of rebuilding per call.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchalg.sparse.kernels.triangular import (
    PreparedTriangularSolve,
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


@dataclass(slots=True)
class _MaskedSolves:
    """Level-scheduled sweeps of one matrix, built when its first masked sweep needs them.

    Mutable on purpose: the schedules cost a host pass over the matrix, which the many
    matrices that are only ever swept unmasked must not pay.

    Attributes:
        matrix (torch.Tensor): The sparse CSR matrix ``A``.
        solves (tuple[PreparedTriangularSolve, PreparedTriangularSolve] | None): The
            ``(D + L)`` and ``(D + U)`` substitutions, ``None`` until first use.
    """

    matrix: torch.Tensor
    solves: tuple[PreparedTriangularSolve, PreparedTriangularSolve] | None = None

    def get(self) -> tuple[PreparedTriangularSolve, PreparedTriangularSolve]:
        """Return the forward and backward substitutions, building them on first call."""
        if self.solves is None:
            self.solves = (
                PreparedTriangularSolve(self.matrix, "forward"),
                PreparedTriangularSolve(self.matrix, "backward"),
            )
        return self.solves


@dataclass(frozen=True, slots=True)
class PreparedSymmetricGaussSeidel:
    """Symmetric Gauss-Seidel bound to one sparse CSR matrix.

    Create with ``prepare_csr_gauss_seidel``; every field is computed from the
    matrix once and captures its values.

    Attributes:
        matrix (torch.Tensor): The sparse CSR matrix ``A``, shape ``(n, n)``, with
            int32 indices where it fits (used only for residuals).
        upper (torch.Tensor): Strict upper triangle ``U`` as sparse CSR.
        lower (torch.Tensor): Strict lower triangle ``L`` as sparse CSR.
        forward (TriangularSolver): ``(D + L)`` substitution of the forward sweep.
        backward (TriangularSolver): ``(D + U)`` substitution of the backward sweep.
        masked (_MaskedSolves): Level-scheduled substitutions for masked sweeps.

    Cost:
        One symmetric iteration (forward plus backward sweep), ``k`` columns,
        ``z = nnz``, ``V = 8`` bytes per ``float64``, ``w`` the index width,
        ``lv_f`` and ``lv_b`` the dependency levels of ``D + L`` and ``D + U``.
        Unmasked, vendor substitutions: ``W = (4z + 2n)k`` (two triangular
        solves ``2(z + n)``, two strict-triangle SpMVs ``2(z - n)``, two
        right-hand-side updates ``2n``), ``Q = 2z(V + w)`` plus ``O(Vnk)``
        vector passes, ``D = lv_f + lv_b`` (each solve is a recurrence over
        its levels), ``L = 4`` and ``S = 0``. This is the algorithmic minimum.
        Masked (``rows=``), level-scheduled: the same ``W`` up to the frozen
        rows, but ``D = lv_f + lv_b`` with one small gather, segment sum and
        divide per level, so ``L = O(lv_f + lv_b)`` and ``Q`` is the
        touched-row bytes plus those per-kernel temporaries. Launches, not
        flops, dominate it, and it is the slower path.
        Roofline: ``W/Q`` is about 0.16 flop per byte unmasked (memory-bound)
        and an order of magnitude lower masked.
        Work-depth: ``W/D = (4z + 2n)k / (lv_f + lv_b)``, about ``10^3`` at
        ``nnz/n`` near 37, far below SpMV and Jacobi: the depth, not the
        flops, limits parallel speed-up. The cost is independent of ``n^2``
        (unlike the dense-backed sweep).
    """

    matrix: torch.Tensor
    upper: torch.Tensor
    lower: torch.Tensor
    forward: TriangularSolver
    backward: TriangularSolver
    masked: _MaskedSolves

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

        Cost: per iteration two vendor triangular solves and two strict-triangle
        SpMVs, ``W = (4 nnz + 2n)k`` at the algorithmic minimum, with launches independent
        of the dependency depth. A masked call executes the level-scheduled substitution
        instead, one gather, segment sum and divide per level (class ``Cost:`` has the
        depth and launch terms).

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
        active = None
        if rows is not None:
            active = torch.zeros(self.matrix.shape[0], dtype=torch.bool, device=self.matrix.device)
            active[rows] = True
        for _ in range(steps):
            x = self._sweep_pair(rhs, rhs - self.upper @ x, x, active)
        return x

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
        x = self._sweep_pair(rhs, rhs, None, None)
        return x if steps == 1 else self.smooth(rhs, x, steps - 1)

    def _sweep_pair(
        self,
        rhs: torch.Tensor,
        forward_target: torch.Tensor,
        current: torch.Tensor | None,
        active: torch.Tensor | None,
    ) -> torch.Tensor:
        """One forward sweep followed by one backward sweep.

        Args:
            rhs (torch.Tensor): Right-hand side of the system.
            forward_target (torch.Tensor): ``rhs - U @ x_old`` for the forward solve.
            current (torch.Tensor | None): ``x_old``; needed only to freeze inactive rows.
            active (torch.Tensor | None): Boolean mask of updatable rows, or ``None``.

        Returns:
            torch.Tensor: Iterate after the forward and backward sweeps.
        """
        if active is None:
            x = self.forward.solve(forward_target)
            return self.backward.solve(rhs - self.lower @ x)
        forward, backward = self.masked.get()
        x = forward.solve(forward_target, active=active, current=current)
        return backward.solve(rhs - self.lower @ x, active=active, current=x)


# Rounding differs from the level-scheduled substitution it replaced (adaptive SA applications
# differ by up to 2e-10 on lap2d n = 10 000, the size at which the old tree differs from itself
# between thread counts); hierarchy sizes and PCG iterations are identical.
def prepare_csr_gauss_seidel(matrix: torch.Tensor) -> PreparedSymmetricGaussSeidel:
    """Split ``matrix`` into its strict triangles and prepare both vendor substitutions.

    Cost: ``W = O(nnz)`` for the split and the library's triangular analysis, ``Q`` one
    pass over the stored entries, no level schedule (``L = 1``).

    Args:
        matrix (torch.Tensor): Sparse CSR matrix ``A``, shape ``(n, n)``.

    Returns:
        PreparedSymmetricGaussSeidel: Reusable sweeps over ``matrix``.

    Raises:
        ValueError: If ``matrix`` is not sparse CSR.
    """
    _require_csr(matrix, "prepare_csr_gauss_seidel")
    row_index = _expand_row_index(matrix)
    col = matrix.col_indices()
    return PreparedSymmetricGaussSeidel(
        matrix=compact_csr_indices(matrix),
        upper=_strict_triangle(row_index, matrix, col > row_index),
        lower=_strict_triangle(row_index, matrix, col < row_index),
        forward=prepare_triangular_solve(matrix, "forward"),
        backward=prepare_triangular_solve(matrix, "backward"),
        masked=_MaskedSolves(matrix),
    )
