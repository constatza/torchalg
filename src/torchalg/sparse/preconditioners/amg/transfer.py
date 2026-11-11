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
        """Store the sparse prolongation matrix and build its transpose once.

        ``P`` is kept by reference (an operator never mutates it, and a
        hierarchy never mutates a built ``P`` in place); only an inference
        tensor is copied, because ``P.t()`` fails on one with "Cannot set
        version_counter for inference tensor" and the operator must persist
        past the inference-mode block that built it.

        Args:
            P (torch.Tensor): Sparse CSR prolongation matrix (n_fine x
                n_coarse).
        """
        with torch.inference_mode(False):
            self._P = P.clone() if P.is_inference() else P
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


class TransferOperatorCache:
    """Builds one ``SparseTransferOperator`` per prolongation tensor, for one setup stage.

    Setup algorithms rebuild sub-hierarchies over the same prolongations many
    times (adaptive ``_solve`` per candidate, bootstrap improvement passes);
    wrapping each ``P`` through this cache transposes it once per stage
    instead of once per sub-hierarchy. Keyed by tensor identity and holding
    the prolongations alive, so it must not outlive the stage and a cached
    ``P`` must not be mutated in place. Callable as the
    ``prolongation -> TransferOperator`` factory ``BootstrapSetup`` expects.
    """

    def __init__(self) -> None:
        """Start with no operators."""
        self._operators: dict[int, tuple[torch.Tensor, SparseTransferOperator]] = {}

    def __call__(self, P: torch.Tensor) -> SparseTransferOperator:
        """Return the operator of ``P``, building it on first use.

        Args:
            P (torch.Tensor): Sparse CSR prolongation matrix.

        Returns:
            SparseTransferOperator: Shared by every later call with this tensor.
        """
        key = id(P)
        entry = self._operators.get(key)
        if entry is None:
            # ``P`` is stored beside its operator so its id cannot be recycled.
            entry = (P, SparseTransferOperator(P))
            self._operators[key] = entry
        return entry[1]


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
