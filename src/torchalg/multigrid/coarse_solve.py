"""Prepared coarsest-level solvers: factorize once per hierarchy, apply per cycle.

A multigrid cycle solves the coarsest level on every application. Re-factorizing the
(constant) coarse matrix there costs O(n_c^3) per application (an SVD for ``pinv``, an LU
for ``solve``) and, on GPU, a host synchronization. A ``CoarseSolverFactory`` instead
turns the coarse matrix into a ``CoarseSolve`` once, at hierarchy preparation, and the
cycle only applies it.

The prepared solve is stored on the ``PreparedCoarsestLevel`` (never on a cycle object:
the setup cycles are shared module-level instances used for many matrices).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from torchalg.preconditioners.base import PreconditionerNotReadyError


@dataclass(frozen=True, slots=True)
class LUCoarseSolve:
    """Pivoted LU factors of the coarsest matrix; solves by triangular substitution.

    A frozen dataclass (not a closure) so ``utils.device.moved_to`` can move the factors
    with the rest of a prepared hierarchy.

    Cost:
        One call, ``c`` the coarsest dimension, ``k`` columns, ``V = 8`` bytes per
        ``float64``: ``W = 2c^2 k`` (two triangular substitutions), ``Q = V(c^2 + 2ck)``,
        ``D = 2c`` (forward then backward substitution, each a recurrence), ``L = 1`` and
        ``S = 0``.
        Roofline: ``W/Q`` is about 0.25 flop per byte for ``k = 1`` (memory-bound) and
        grows with ``k``.
        Work-depth: ``W/D = ck`` is low, so the dense coarse solve, not the sparse levels,
        limits the depth of a cycle when ``c`` is a few hundred. The factorization is
        ``lu_coarse_solver``'s cost.
    """

    lu: torch.Tensor
    pivots: torch.Tensor

    def __call__(self, rhs: torch.Tensor) -> torch.Tensor:
        """Solve for a vector or a matrix right-hand side."""
        columns = rhs.unsqueeze(-1) if rhs.ndim == 1 else rhs
        solution = torch.linalg.lu_solve(self.lu, self.pivots, columns)
        return solution.squeeze(-1) if rhs.ndim == 1 else solution


@dataclass(frozen=True, slots=True)
class PseudoInverseCoarseSolve:
    """Cached inverse (pseudo-inverse when singular) of the coarsest matrix; solves by one product.

    Cost:
        One call, ``c`` the coarsest dimension, ``k`` columns, ``V = 8`` bytes per
        ``float64``: ``W = 2c^2 k`` (one GEMV or GEMM), ``Q = V(c^2 + 2ck)``,
        ``D = ceil(log2 c) + 1`` (a row reduction; no recurrence, unlike the LU solve),
        ``L = 1`` and ``S = 0``.
        Roofline: ``W/Q`` is about 0.25 flop per byte for ``k = 1`` (memory-bound).
        Work-depth: ``W/D`` is large, so this apply parallelizes well and is the
        shallower of the two coarse solves. The factorization is
        ``pseudo_inverse_coarse_solver``'s cost.
    """

    inverse: torch.Tensor

    def __call__(self, rhs: torch.Tensor) -> torch.Tensor:
        """Apply the cached inverse, the minimum-norm least-squares one when singular."""
        return self.inverse @ rhs


type CoarseSolve = Callable[[torch.Tensor], torch.Tensor]
"""Maps a coarsest-level right-hand side to the solution. The built-in factories return
frozen dataclasses (``LUCoarseSolve``, ``PseudoInverseCoarseSolve``), which move with the
hierarchy; a custom solve must be a dataclass of tensors or expose ``moved_to``."""

type CoarseSolverFactory = Callable[[torch.Tensor], CoarseSolve]
"""Builds a ``CoarseSolve`` from a coarsest-level matrix (the one-off expensive step)."""


class HierarchyNotPreparedError(PreconditionerNotReadyError):
    """Raised when a cycle is applied to a hierarchy that ``prepare`` has not produced.

    A ``PreconditionerNotReadyError``, so one ``except`` clause covers "used before it
    was prepared or set up" for preconditioners and bare cycles alike.

    Call ``cycle.prepare(hierarchy)`` once after building a hierarchy; this error makes a
    forgotten preparation fail fast instead of silently re-factorizing on every cycle.
    """


def lu_coarse_solver(matrix: torch.Tensor) -> CoarseSolve:
    """Factor ``matrix`` with a pivoted LU once; solve by triangular substitution.

    Equivalent to ``torch.linalg.solve(matrix, rhs)`` (including failing on a singular
    matrix, now at preparation time instead of on the first cycle).

    Cost:
        Once per hierarchy, ``c`` the coarsest dimension, ``V = 8`` bytes per ``float64``:
        ``W = 2c^3 / 3``, ``Q = 2Vc^2`` (the matrix is read and the factors written once),
        ``D = c`` (the elimination steps are a dependent chain), ``L = 1`` and ``S = 1``
        (the singularity read).
        Roofline: ``W/Q = c / 24`` flops per byte, so the factorization is compute-bound
        from ``c`` of a few hundred, the one kernel in a setup that is.
        Work-depth: ``W/D = 2c^2 / 3``, ample parallelism inside each elimination step.
        Each later solve is ``LUCoarseSolve``.

    Args:
        matrix (torch.Tensor): Dense nonsingular coarsest-level matrix.

    Returns:
        CoarseSolve: Solves ``matrix @ x = rhs`` for a vector or a matrix right-hand side.

    Raises:
        torch.linalg.LinAlgError: If ``matrix`` is singular.
    """
    lu, pivots, info = torch.linalg.lu_factor_ex(matrix)
    if bool(info.item()):
        raise torch.linalg.LinAlgError(
            "Coarsest-level matrix is singular; use a pinv coarse solver."
        )
    return LUCoarseSolve(lu=lu, pivots=pivots)


# Below this width the SVD costs under 2 ms, and the Cholesky inverse shifted a one-application
# result by 1.9e-11 on a 17 x 17 block (condition number about 860), so small blocks stay on pinv.
CHOLESKY_MIN_DIMENSION = 128
"""Smallest coarsest dimension that takes the Cholesky route.

Below it the SVD costs at most a couple of milliseconds once per hierarchy, so ``pinv`` is
kept and the prepared solve stays bit-identical to the historical ``pinv``; at and above it
the SVD is ``>= 5x`` the Cholesky and grows to ``26x`` at ``c = 1024``."""

_SYMMETRY_TOLERANCE_EPS = 16.0
"""Largest relative asymmetry, in units of machine epsilon, still treated as symmetric.

A Galerkin product ``P.T @ A @ P`` is symmetric only up to summation-order rounding
(below one epsilon in practice), so the bound separates that from a genuinely nonsymmetric
coarse operator, whose Cholesky factor (built from one triangle) would solve a different
system."""

_MIN_PIVOT_RATIO_EXPONENT = 0.5
"""Cholesky is trusted when ``(min pivot / max pivot)^2 > eps ** exponent``.

A positive semidefinite singular matrix finishes its Cholesky with a rounding-level pivot
(ratio ``~eps``) instead of failing, and the plain inverse would then return garbage where
``pinv`` returns the minimum-norm solution. Pivot ratios bound the condition number from
below, so a tiny ratio proves near-singularity; matrices below the bound keep ``pinv``."""


def _spd_inverse(matrix: torch.Tensor) -> torch.Tensor | None:
    """Inverse of a large, symmetric, well-conditioned positive definite ``matrix``, else ``None``.

    Cost:
        Cholesky ``c^3 / 3`` plus the triangular inverse and product ``2 c^3 / 3``, so
        ``W ~ c^3`` against ``~ 24 c^3`` for the SVD in ``pinv``. One host synchronization
        decides the branch (``S = 1``). A rejected matrix pays the Cholesky on top of the
        ``pinv``. ``Q = O(Vc^2)`` (a constant number of passes over the ``c x c`` matrix),
        ``D = O(c)`` (the factorization is a recurrence over the columns).

    Args:
        matrix (torch.Tensor): Dense square coarsest-level matrix.

    Returns:
        torch.Tensor | None: ``matrix^-1`` when it is symmetric, positive definite and not
        near-singular and at least ``CHOLESKY_MIN_DIMENSION`` wide; ``None`` when ``pinv``
        must decide.
    """
    if matrix.shape[0] < CHOLESKY_MIN_DIMENSION:
        return None
    eps = torch.finfo(matrix.dtype).eps
    asymmetry = (matrix - matrix.mT).abs().amax()
    symmetric = asymmetry <= _SYMMETRY_TOLERANCE_EPS * eps * matrix.abs().amax()
    factor, info = torch.linalg.cholesky_ex(matrix)
    pivots = factor.diagonal()
    well_conditioned = (pivots.amin() / pivots.amax()) ** 2 > eps**_MIN_PIVOT_RATIO_EXPONENT
    if not bool((info == 0) & symmetric & well_conditioned):
        return None
    return torch.cholesky_inverse(factor)


def pseudo_inverse_coarse_solver(matrix: torch.Tensor) -> CoarseSolve:
    """Invert ``matrix`` once (Cholesky if SPD, else ``pinv``); solve by a matrix-vector product.

    PyAMG's default coarse solver (``'pinv'``). Valid for singular coarse matrices, which
    arise when dropped candidates leave zero rows/columns, and returns the minimum-norm
    least-squares solution there. A symmetric, well-conditioned positive definite matrix
    of dimension ``>= CHOLESKY_MIN_DIMENSION``, the usual Galerkin coarse operator, is inverted through its Cholesky factor instead,
    which agrees with ``pinv`` to rounding at a fraction of the SVD cost; every other
    matrix (small, nonsymmetric, singular, indefinite, near-singular) keeps the plain (not
    Hermitian) ``pinv``, bit-identical to the previous per-call ``pinv(matrix) @ rhs``.

    Cost:
        Setup, once per hierarchy, ``c`` the coarsest dimension, ``V = 8`` bytes per
        ``float64``: ``W ~ c^3`` on the SPD path (Cholesky and its inverse), ``W ~ 24 c^3``
        through ``pinv`` (a standard SVD coefficient, not a counted value), ``Q = O(Vc^2)``
        (about ``2Vc^2`` for the SVD), ``D = O(c)``, ``L`` a small constant and ``S = 1``
        (the SPD test; none below ``CHOLESKY_MIN_DIMENSION``). Apply is one ``c x c``
        product on both (``PseudoInverseCoarseSolve``).
        Roofline: ``W/Q`` is about ``1.5c`` flops per byte for the SVD, so setup is
        compute-bound beyond tiny ``c``; the SVD is the largest single flop term of
        an adaptive smoothed-aggregation setup.
        Work-depth: ``W/D`` is large inside the dense kernels, but the factorization
        recurrence bounds the depth at ``O(c)``.

    Args:
        matrix (torch.Tensor): Dense coarsest-level matrix, possibly singular.

    Returns:
        CoarseSolve: Applies the cached inverse.
    """
    inverse = _spd_inverse(matrix)
    return PseudoInverseCoarseSolve(
        inverse=inverse if inverse is not None else torch.linalg.pinv(matrix)
    )
