"""Coarsest-level direct solve for sparse AMG hierarchies.

``torchalg.multigrid``'s cycles (``VCycle``/``WCycle``) are format-agnostic
everywhere except the coarsest-level solve, which they delegate to an
injectable ``coarse_solver`` callable (see ``torchalg.multigrid.cycle``'s
module docstring). ``dense_coarse_solve`` is the one deliberate,
intentional sparse-aware branch in this whole sparse AMG tree: the coarsest
level is small by construction (bounded coarsening ratio over
``n_levels - 1`` steps), so densifying just that one matrix for
``torch.linalg.solve`` - which has no sparse-CSR direct-solve path in
PyTorch - is the correct, narrow adapter, not a violation of the
sparse/dense separation enforced elsewhere (``docs/plan.md``).
"""

from __future__ import annotations

import torch


def dense_coarse_solve(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve ``matrix @ x = rhs`` at the coarsest level, densifying first if sparse.

    Args:
        matrix (torch.Tensor): Coarsest-level matrix, sparse CSR or dense.
        rhs (torch.Tensor): Right-hand side.

    Returns:
        torch.Tensor: Solution ``x``.
    """
    if matrix.is_sparse_csr:
        matrix = matrix.to_dense()
    return torch.linalg.solve(matrix, rhs)


def dense_pseudo_inverse_solve(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Coarsest-level solve ``pinv(A) @ rhs``, densifying first if sparse.

    Sparse sibling of ``torchalg.multigrid.cycle.pseudo_inverse_solve`` -
    same one deliberate, intentional sparse-aware densification as
    ``dense_coarse_solve`` above, needed because ``torch.linalg.pinv`` has
    no sparse-CSR path either. Used by setup-time stages (adaptive-SA
    candidate construction, Bootstrap AMG test-vector improvement) whose
    coarse matrix can become singular when dropped candidates leave a zero
    row/column - ``torch.linalg.solve`` would raise there, which is exactly
    why PyAMG's own default coarse solver is ``'pinv'``, not a direct solve.

    Args:
        matrix (torch.Tensor): Coarsest-level matrix, sparse CSR or dense.
        rhs (torch.Tensor): Right-hand side.

    Returns:
        torch.Tensor: Minimum-norm least-squares solution.
    """
    if matrix.is_sparse_csr:
        matrix = matrix.to_dense()
    return torch.linalg.pinv(matrix) @ rhs
