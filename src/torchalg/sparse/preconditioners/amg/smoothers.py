"""Multigrid smoothers, sparse-CSR siblings.

Sparse-only counterparts of ``preconditioners.implementations.amg.smoothers``'s
``JacobiSmoother``/``GaussSeidelSmoother`` (dense; kept separate).
Same algorithm, separate implementation, tested against the dense result: the
sparse tree never imports the dense tree (enforced by ``tach.toml``). A smoother object holds only
configuration; ``prepare(A)`` does every matrix-dependent one-off computation
(diagonal inverse and spectral-radius damping for Jacobi, level schedules and
strict-triangle split for Gauss-Seidel) and returns the object that sweeps.
Only sparse CSR input is ever prepared - no ``is_sparse_csr`` branch anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from torchalg.preconditioners.base import Placement
from torchalg.sparse.kernels.diagonal import sparse_diagonal
from torchalg.utils.csr_index import compact_csr_indices

from ._gauss_seidel import PreparedSymmetricGaussSeidel, SweepProducts, prepare_csr_gauss_seidel
from ._jacobi_omega import RELAXATION_NOMINAL, jacobi_omega

if TYPE_CHECKING:
    from torchalg.multigrid.protocols import MultigridSmoother

_NEAR_ZERO_DIAGONAL_TOL = 1e-14
"""Diagonal entries with magnitude below this are excluded from the Jacobi
update (treated as a zero step for that component)."""


@dataclass(frozen=True, slots=True)
class PreparedJacobi:
    """Weighted Jacobi bound to one sparse CSR matrix.

    Cost:
        Setup: construction only binds two tensors, ``W = 0``, ``D = 0``, ``L = 0``,
        ``S = 0``; the matrix-dependent work is ``JacobiSmoother.prepare``'s.
        Apply, one sweep ``x + D^-1 omega (rhs - A x)``, ``k`` columns, ``z = nnz``,
        ``m`` the largest row count, ``w`` the index width, ``V = 8`` bytes per
        ``float64``: ``W = (2z + 3n)k`` (SpMV, subtract, scale, add),
        ``Q = z(V + w) + (n + 1)w + 11Vnk`` (the SpMV reads ``A`` and ``x`` and writes
        ``A x``; the three vector kernels read two operands and write one),
        ``D = ceil(log2 m) + 4``, ``L = 4`` and ``S = 0``. ``smooth_from_zero``'s first
        sweep is one scaling (``W = nk``, ``L = 1``); ``residual`` is the SpMV and the
        subtraction (``W = (2z + n)k``, ``L = 2``).
        Roofline: ``W/Q`` is about 0.15 flop per byte with 4-byte indices
        (memory-bound).
        Work-depth: ``W/D`` is about ``2zk / log2 m``, fully parallel across rows,
        which is why Jacobi-smoothed cycles are ``SPMV_BOUND`` and may run on a GPU.

    Attributes:
        matrix (torch.Tensor): Sparse CSR matrix ``A``, shape ``(n, n)``, with int32
            indices where it fits (``utils.csr_index``), since every sweep and
            residual multiplies by it.
        diag_inv (torch.Tensor): ``omega / diag(A)`` with ``0`` where the
            diagonal is (near) zero, shape ``(n,)``.
    """

    matrix: torch.Tensor
    diag_inv: torch.Tensor

    def residual(self, rhs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return ``rhs - A x`` with the bound level matrix.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            x (torch.Tensor): Current iterate, same shape as ``rhs``.

        Returns:
            torch.Tensor: Residual, same shape as ``rhs``.
        """
        return rhs - self.matrix @ x

    def _scale(self, residual: torch.Tensor) -> torch.Tensor:
        """``diag_inv * residual`` for a vector ``(n,)`` or a block ``(n, k)``."""
        if residual.ndim == 1:
            return self.diag_inv * residual
        return self.diag_inv.unsqueeze(-1) * residual

    def smooth(self, rhs: torch.Tensor, x: torch.Tensor, steps: int) -> torch.Tensor:
        """Apply ``steps`` sweeps ``x <- x + omega D^-1 (rhs - A x)`` from ``x``.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            x (torch.Tensor): Current iterate, same shape; not modified.
            steps (int): Number of sweeps.

        Returns:
            torch.Tensor: Updated iterate, same shape as ``x``.
        """
        if steps == 0:
            return x.clone()
        for _ in range(steps):
            x = x + self._scale(rhs - self.matrix @ x)
        return x

    def smooth_from_zero(self, rhs: torch.Tensor, steps: int) -> torch.Tensor:
        """Apply ``steps`` sweeps from the zero guess, skipping the first ``A @ 0``.

        Args:
            rhs (torch.Tensor): Right-hand side, shape ``(n,)`` or ``(n, k)``.
            steps (int): Number of sweeps.

        Returns:
            torch.Tensor: Iterate after ``steps`` sweeps, same shape as ``rhs``.
        """
        if steps == 0:
            return torch.zeros_like(rhs)
        x = self._scale(rhs)
        for _ in range(steps - 1):
            x = x + self._scale(rhs - self.matrix @ x)
        return x


class JacobiSmoother:
    """Weighted Jacobi smoother for sparse CSR matrices: x <- x + omega * D^{-1} (rhs - Ax).

    Sparse-CSR sibling of the dense
    ``preconditioners.implementations.amg.smoothers.JacobiSmoother``.

    Args:
        omega (float | torch.Tensor | None): Damping factor. ``None``
            (default) is the relaxation rule ``1 / rho(D^{-1}A)``, estimated
            in ``prepare`` (never during sweeps); a float or 0-d tensor fixes it.

    Cost:
        Setup (``prepare``, once per matrix), ``z = nnz``, ``m`` the largest row
        count: the diagonal, ``omega`` and ``omega / diag`` cost ``W = z + n``,
        ``Q = O(z(V + w))``, ``D = m + O(1)``, ``L = 19`` and ``S = 0`` for a fixed
        ``omega``. The default ``omega = 1 / rho(D^-1 A)`` adds one Arnoldi estimate
        on a memo miss (``jacobi_spectral_radius``'s Cost: ``P`` passes of
        ``m_A = 15`` steps, ``W = z + n + P(2z m_A + 4n m_A(m_A + 1))``,
        ``D = P m_A (ceil(log2 n) + 1) * O(1)``, ``L = O(P m_A)`` on a GPU and
        ``O(P m_A^2)`` on the CPU, ``S = P``); an aggregation-coarsened level has
        already paid it for the same matrix, so its smoother adds nothing.
        Apply, one sweep of the returned ``PreparedJacobi``, ``k`` columns:
        ``W = (2z + 3n)k``, ``D = ceil(log2 m) + 4``, ``L = 4``, ``S = 0``.

    Attributes:
        placement (Placement): ``SPMV_BOUND``; a sweep is one SpMV plus vector updates.
    """

    placement = Placement.SPMV_BOUND

    def __init__(self, omega: float | torch.Tensor | None = None) -> None:
        """Store the Jacobi damping factor.

        Args:
            omega (float | None): Damping factor, or ``None`` for ``1 / rho``.
        """
        self._omega = omega

    def prepare(self, A: torch.Tensor) -> PreparedJacobi:
        """Extract the diagonal and resolve the damping, once, for ``A``.

        Args:
            A (torch.Tensor): System matrix (n x n), sparse CSR.

        Returns:
            PreparedJacobi: Jacobi sweeps bound to ``A``.
        """
        diag = sparse_diagonal(A)
        omega = jacobi_omega(A, RELAXATION_NOMINAL, self._omega)
        diag_inv = torch.where(
            diag.abs() > _NEAR_ZERO_DIAGONAL_TOL,
            omega / diag,
            torch.zeros_like(diag),
        )
        return PreparedJacobi(matrix=compact_csr_indices(A), diag_inv=diag_inv)


class GaussSeidelSmoother:
    """Symmetric Gauss-Seidel smoother for sparse CSR matrices.

    Sparse-CSR sibling of the dense
    ``preconditioners.implementations.amg.smoothers.GaussSeidelSmoother``.
    No factorization step: each sweep is a direct triangular solve against
    ``A``'s own entries; ``prepare`` splits ``A`` and prepares the vendor
    triangular solves once. Its sweeps take their strict-triangle products from
    the solves (``SweepProducts.FROM_SOLVES``), within rounding of the explicit
    products.

    Cost:
        Setup (``prepare``, once per matrix) is ``prepare_csr_gauss_seidel`` with
        ``FROM_SOLVES``, ``z = nnz``: ``W = 3z + O(n)`` (three diagonal scatters; the
        split only moves entries), ``Q = O(z(V + w))``, ``D = O(log z)``, about
        ``L = 100`` and ``S = 15`` (the count reads of the masked triangle extraction and
        the pivot tests).
        Apply, one symmetric iteration (forward plus backward sweep), ``k`` columns,
        ``lv_f`` and ``lv_b`` the dependency levels of ``D + L`` and ``D + U``:
        ``W = (2z + 7n)k`` (two vendor triangular solves and three vector kernels),
        plus ``(z - n)k`` once per ``smooth`` call for ``U x``, ``D = lv_f + lv_b + 3``,
        ``L = 5`` (plus one per call for ``U x``) and ``S = 0``; see
        ``PreparedSymmetricGaussSeidel`` for the bytes and the explicit-product sweep.

    Attributes:
        placement (Placement): ``SEQUENTIAL_SWEEP``; each sweep is a chain of dependent
            triangular solves.
    """

    placement = Placement.SEQUENTIAL_SWEEP

    def prepare(self, A: torch.Tensor) -> PreparedSymmetricGaussSeidel:
        """Prepare symmetric Gauss-Seidel on ``A``.

        Args:
            A (torch.Tensor): System matrix (n x n), sparse CSR.

        Returns:
            PreparedSymmetricGaussSeidel: Symmetric Gauss-Seidel bound to ``A``.
        """
        return prepare_csr_gauss_seidel(A, SweepProducts.FROM_SOLVES)


class GaussSeidelMemo:
    """Identity memo of the prepared Gauss-Seidel of each distinct matrix, for one setup stage.

    Setup algorithms (compatible relaxation, adaptive and bootstrap stages)
    relax the same level matrices many times, through callables and cycles that
    only receive the matrix. A stage owns one of these for its duration: the
    first sweep on a matrix prepares its ``PreparedSymmetricGaussSeidel`` and
    every later sweep on that same tensor object reuses it. The memo is keyed
    by tensor identity and holds the matrices alive, so it must not outlive the
    stage (and matrices must not be mutated in place within it); there is no
    process-global instance.

    It is a ``MultigridSmoother``: compatible relaxation, test-vector
    relaxation and the setup cycle all bind it to level matrices with ``prepare``,
    so that protocol verb is memoized rather than renamed; the result is the
    same kind of frozen object a fresh ``GaussSeidelSmoother().prepare(A)`` returns,
    except that its sweeps apply the strict-triangle products explicitly
    (``SweepProducts.EXPLICIT``): setup coarsening and fitting amplify the
    ``1e-16`` rounding of products read off the solves to ``1e-10`` in the
    hierarchy, so explicit products keep every setup bit-identical.
    """

    placement = Placement.SEQUENTIAL_SWEEP

    def __init__(self) -> None:
        """Start with no prepared matrices."""
        self._prepared: dict[int, tuple[torch.Tensor, PreparedSymmetricGaussSeidel]] = {}

    def prepare(self, A: torch.Tensor) -> PreparedSymmetricGaussSeidel:
        """Return the prepared Gauss-Seidel of ``A``, preparing it on first use.

        Args:
            A (torch.Tensor): System matrix (n x n), sparse CSR.

        Returns:
            PreparedSymmetricGaussSeidel: Shared by every later call with this tensor.
        """
        key = id(A)
        entry = self._prepared.get(key)
        if entry is None:
            # The matrix is stored beside its preparation so its id cannot be reused
            # while the entry lives.
            entry = (A, prepare_csr_gauss_seidel(A, SweepProducts.EXPLICIT))
            self._prepared[key] = entry
        return entry[1]


def resolve_jacobi_default(
    smoother: MultigridSmoother | None,
    smoother_omega: float | None,
) -> MultigridSmoother:
    """Select an injected smoother or construct the preset's sparse Jacobi default.

    Sparse-CSR sibling of
    ``preconditioners.implementations.amg.smoothers.resolve_jacobi_default``
    - same name and logic, own copy, since the sparse tree never imports
    the dense tree (enforced by ``tach.toml``).

    Args:
        smoother (MultigridSmoother | None): Explicit smoother strategy.
        smoother_omega (float | None): Damping for the sparse Jacobi default.

    Returns:
        MultigridSmoother: The explicit strategy or a sparse weighted-Jacobi
            smoother.

    Raises:
        ValueError: If both an explicit smoother and Jacobi damping are supplied.
    """
    if smoother is not None and smoother_omega is not None:
        raise ValueError("smoother_omega only applies when smoother is not provided")
    return smoother if smoother is not None else JacobiSmoother(omega=smoother_omega)
