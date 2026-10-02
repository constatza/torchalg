"""Sparse-backed transfer operator for multigrid prolongation and restriction.

Second implementation of ``preconditioners.implementations.amg.protocols
.TransferOperator`` (dense ``DenseTransferOperator`` is the first) - see
``docs/plan.md``: uses ``torch.sparse.mm`` instead of a dense ``@`` so the
fine-grid-sized prolongation matrix ``P`` never needs a dense ``(n,
n_coarse)`` materialization.
"""

from __future__ import annotations

import torch


class SparseTransferOperator:
    """Transfer operator backed by a sparse CSR prolongation matrix P.

    Prolongation: ``P @ coarse`` (inject coarse correction to fine grid).
    Restriction: ``P.T @ fine`` (Galerkin restriction, R = P^T).

    Args:
        P (torch.Tensor): Sparse CSR prolongation matrix (n_fine x
            n_coarse).
    """

    def __init__(self, P: torch.Tensor) -> None:
        """Store the sparse prolongation matrix and its transpose.

        Args:
            P (torch.Tensor): Sparse CSR prolongation matrix (n_fine x
                n_coarse).
        """
        # Regression: lazy Bootstrap/adaptive sparse hierarchies built during
        # PCG's default inference-mode execution failed at ``P.t()`` with
        # "Cannot set version_counter for inference tensor". Cache ordinary
        # tensors for the persistent transfer operator state.
        with torch.inference_mode(False):
            self._P = P.clone()
            self._R = self._P.t().to_sparse_csr()

    def prolongate(self, coarse: torch.Tensor) -> torch.Tensor:
        """Interpolate coarse-grid vector to fine grid.

        Args:
            coarse (torch.Tensor): Dense coarse-grid vector of length
                n_coarse.

        Returns:
            torch.Tensor: Dense fine-grid vector of length n_fine.
        """
        return _sparse_matvec(self._P, coarse)

    def restrict(self, fine: torch.Tensor) -> torch.Tensor:
        """Restrict fine-grid vector to coarse grid (R = P^T).

        Args:
            fine (torch.Tensor): Dense fine-grid vector of length n_fine.

        Returns:
            torch.Tensor: Dense coarse-grid vector of length n_coarse.
        """
        return _sparse_matvec(self._R, fine)


def _sparse_matvec(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Sparse-CSR-matrix @ dense-vector product, dense output.

    Args:
        matrix (torch.Tensor): Sparse CSR matrix, shape ``(m, n)``.
        vector (torch.Tensor): Dense vector, shape ``(n,)`` or ``(n, k)``.

    Returns:
        torch.Tensor: Dense result, same ``ndim`` as ``vector``.
    """
    if vector.ndim == 1:
        return (matrix @ vector.unsqueeze(-1)).squeeze(-1)
    return matrix @ vector
