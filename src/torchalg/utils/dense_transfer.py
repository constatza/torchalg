"""Dense transfer operator for multigrid-style prolongation and restriction.

Promoted out of ``preconditioners.implementations.amg.transfer`` (see
``docs/plan.md``): this class only ever requires its ``P`` argument to be a
dense ``torch.Tensor`` and does plain ``P @ x`` / ``P.T @ x`` - nothing about
it assumes anything about any other tensor in the system (not the fine-grid
matrix ``A``, not any result format). That makes it format-agnostic in
exactly the same sense as ``torchalg.utils.spectral``'s Arnoldi routine, so
it lives here as a dependency-free leaf consumed by both the dense AMG/POD
trees and (via ``torchalg.sparse.preconditioners.pod.coarsening``, whose POD
basis ``Phi_r`` is always dense even when the fine-grid matrix is sparse CSR)
the sparse tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class DenseTransferOperator:
    """Transfer operator backed by a dense prolongation matrix P.

    Prolongation: ``P @ coarse`` (inject coarse correction to fine grid).
    Restriction: ``P.T @ fine`` (Galerkin restriction, R = P^T).

    Args:
        P (torch.Tensor): Prolongation matrix (n_fine x n_coarse), dense.
    """

    def __init__(self, P: torch.Tensor) -> None:
        """Store the prolongation matrix and its transpose (Galerkin restriction).

        Args:
            P (torch.Tensor): Prolongation matrix (n_fine x n_coarse), dense.
        """
        self._P = P
        self._R = P.T

    def prolongate(self, coarse: torch.Tensor) -> torch.Tensor:
        """Interpolate coarse-grid vector to fine grid.

        Args:
            coarse (torch.Tensor): Coarse-grid vector of length n_coarse.

        Returns:
            torch.Tensor: Fine-grid vector of length n_fine.
        """
        return self._P @ coarse

    def restrict(self, fine: torch.Tensor) -> torch.Tensor:
        """Restrict fine-grid vector to coarse grid (R = P^T).

        Args:
            fine (torch.Tensor): Fine-grid vector of length n_fine.

        Returns:
            torch.Tensor: Coarse-grid vector of length n_coarse.
        """
        return self._R @ fine
