"""Coarsest-level direct solve factories for sparse AMG hierarchies.

``torchalg.multigrid``'s cycles (``VCycle``/``WCycle``) are format-agnostic
everywhere except the coarsest-level solve, which they obtain from an injectable
``CoarseSolverFactory`` (see ``torchalg.multigrid.coarse_solve``). The factories here are
the one deliberate, intentional sparse-aware branch in this whole sparse AMG tree: the
coarsest level is small by construction (bounded coarsening ratio over ``n_levels - 1``
steps), so densifying just that one matrix - once, when the hierarchy is prepared, not on
every cycle - for the dense factorization is the correct, narrow adapter, not a violation
of the sparse/dense separation enforced elsewhere (``docs/plan.md``). PyTorch has no
sparse-CSR direct-solve path.
"""

from __future__ import annotations

import torch

from torchalg.multigrid.coarse_solve import (
    CoarseSolve,
    lu_coarse_solver,
    pseudo_inverse_coarse_solver,
)


def _densified(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.to_dense() if matrix.is_sparse_csr else matrix


def dense_lu_coarse_solver(matrix: torch.Tensor) -> CoarseSolve:
    """Densify ``matrix`` if sparse, then factor it with a pivoted LU once.

    Args:
        matrix (torch.Tensor): Coarsest-level matrix, sparse CSR or dense.

    Returns:
        CoarseSolve: Solves ``matrix @ x = rhs``.
    """
    return lu_coarse_solver(_densified(matrix))


def dense_pseudo_inverse_coarse_solver(matrix: torch.Tensor) -> CoarseSolve:
    """Densify ``matrix`` if sparse, then compute its pseudo-inverse once.

    Used by setup-time stages (adaptive-SA candidate construction, Bootstrap AMG test-vector
    improvement) and the prebuilt presets whose coarse matrix can become singular when
    dropped candidates leave a zero row/column - an LU solve would raise there, which is
    exactly why PyAMG's own default coarse solver is ``'pinv'``.

    Args:
        matrix (torch.Tensor): Coarsest-level matrix, sparse CSR or dense.

    Returns:
        CoarseSolve: Applies the cached minimum-norm least-squares inverse.
    """
    return pseudo_inverse_coarse_solver(_densified(matrix))
