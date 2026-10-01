"""Sparse Galerkin coarse-operator formation for AMG/POD coarsening.

Promotes ``scripts/bench_sparse_vs_dense/operations.py``'s already-validated
``form_galerkin_spgemm``/``form_galerkin_spmm`` into production (see
``docs/plan.md``): both forms compute ``A_coarse = P.T @ A @ P`` without ever
materializing the fine-grid ``(n, n)`` matrix densely.

``form_sparse_sparse`` (AMG shape, both operands sparse) returns a sparse
CSR coarse operator, matching every other kernel in this package - it must
never densify, since a multi-level hierarchy (``n_levels > 2``) feeds its
own output back in as the next level's fine-grid matrix, and ``AMGPreconditioner``
itself is the one place that eventually densifies, at the coarsest level
only (via ``torchalg.sparse.preconditioners.amg.coarse_solve
.dense_coarse_solve``). ``form_sparse_dense`` (POD shape, dense basis) still
returns dense, since a POD coarse operator is small and dense by
construction (SVD right-singular vectors, no structural zeros).
"""

from __future__ import annotations

import torch


def form_sparse_sparse(P: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Form ``A_coarse = P.T @ A @ P`` with both ``P`` and ``A`` sparse CSR (SpGEMM).

    AMG shape: real AMG prolongation operators are themselves sparse (each
    fine node interpolates from a handful of coarse neighbors). Uses COO for
    the transpose step, since ``sparse_csr.t()`` yields CSC and mixed
    CSC/CSR sparse-sparse matmul support isn't something to assume.

    Args:
        P (torch.Tensor): Sparse CSR prolongation operator, shape
            ``(n, n_coarse)``.
        A (torch.Tensor): Sparse CSR fine-grid matrix, shape ``(n, n)``.

    Returns:
        torch.Tensor: Sparse CSR coarse operator, shape
            ``(n_coarse, n_coarse)``.
    """
    p_coo = P.to_sparse_coo()
    a_coo = A.to_sparse_coo()
    ap = torch.sparse.mm(a_coo, p_coo)
    return torch.sparse.mm(p_coo.t(), ap).to_sparse_csr()


def form_sparse_dense(A: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
    """Form ``A_coarse = Phi.T @ A @ Phi`` with sparse ``A``, dense ``Phi`` (SpMM).

    POD shape: the POD basis ``Phi`` is fundamentally dense (SVD
    right-singular vectors, no structural zeros), while the fine-grid
    operator ``A`` is a real sparse FEM matrix.

    Args:
        A (torch.Tensor): Sparse CSR fine-grid matrix, shape ``(n, n)``.
        Phi (torch.Tensor): Dense basis/prolongation operator, shape
            ``(n, n_coarse)``.

    Returns:
        torch.Tensor: Dense coarse operator, shape ``(n_coarse, n_coarse)``.
    """
    return Phi.T @ (A @ Phi)
