"""Prepared coarsest-level solvers: factorize once per hierarchy, apply per cycle.

A multigrid cycle solves the coarsest level on every application. Re-factorizing the
(constant) coarse matrix there costs O(n_c^3) per application (an SVD for ``pinv``, an LU
for ``solve``) and, on GPU, a host synchronization. A ``CoarseSolverFactory`` instead
turns the coarse matrix into a ``CoarseSolve`` once, at hierarchy preparation, and the
cycle only applies it.

The prepared solve is stored on the coarsest ``MultigridLevel`` (never on a cycle object:
the setup cycles are shared module-level instances used for many matrices).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

type CoarseSolve = Callable[[torch.Tensor], torch.Tensor]
"""Maps a coarsest-level right-hand side to the solution; closes over the factorization."""

type CoarseSolverFactory = Callable[[torch.Tensor], CoarseSolve]
"""Builds a ``CoarseSolve`` from a coarsest-level matrix (the one-off expensive step)."""


class HierarchyNotPreparedError(RuntimeError):
    """Raised when a cycle is applied to a hierarchy whose coarsest level has no prepared solve.

    Call ``cycle.prepare(hierarchy)`` once after building a hierarchy; this error makes a
    forgotten preparation fail fast instead of silently re-factorizing on every cycle.
    """


def lu_coarse_solver(matrix: torch.Tensor) -> CoarseSolve:
    """Factor ``matrix`` with a pivoted LU once; solve by triangular substitution.

    Equivalent to ``torch.linalg.solve(matrix, rhs)`` (including failing on a singular
    matrix, now at preparation time instead of on the first cycle).

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

    def solve(rhs: torch.Tensor) -> torch.Tensor:
        columns = rhs.unsqueeze(-1) if rhs.ndim == 1 else rhs
        solution = torch.linalg.lu_solve(lu, pivots, columns)
        return solution.squeeze(-1) if rhs.ndim == 1 else solution

    return solve


def pseudo_inverse_coarse_solver(matrix: torch.Tensor) -> CoarseSolve:
    """Compute ``pinv(matrix)`` once; solve by a matrix-vector product.

    PyAMG's default coarse solver (``'pinv'``). Valid for singular coarse matrices, which
    arise when dropped candidates leave zero rows/columns, and returns the minimum-norm
    least-squares solution there. The plain (not Hermitian) ``pinv`` is kept so the result
    is bit-identical to the previous per-call ``pinv(matrix) @ rhs`` and stays correct for
    non-symmetric coarse operators.

    Args:
        matrix (torch.Tensor): Dense coarsest-level matrix, possibly singular.

    Returns:
        CoarseSolve: Applies the cached pseudo-inverse.
    """
    inverse = torch.linalg.pinv(matrix)

    def solve(rhs: torch.Tensor) -> torch.Tensor:
        return inverse @ rhs

    return solve
