"""Sparse Galerkin coarse-operator formation for AMG coarsening.

Promotes ``benchmarks.sparse_dense.operations``' already-validated
``form_galerkin_spgemm`` into production (see
``docs/plan.md``): it computes ``A_coarse = P.T @ A @ P`` without ever
materializing the fine-grid ``(n, n)`` matrix densely.

``form_sparse_sparse`` (AMG shape, both operands sparse) returns a sparse
CSR coarse operator, matching every other kernel in this package - it must
never densify, since a multi-level hierarchy (``n_levels > 2``) feeds its
own output back in as the next level's fine-grid matrix, and ``AMGPreconditioner``
itself is the one place that eventually densifies, at the coarsest level
only (via ``torchalg.sparse.preconditioners.amg.coarse_solve
.dense_coarse_solve``). POD needs no kernel here: its basis is dense, so
``PODCoarseningStrategy`` forms ``Phi.T @ (A @ Phi)`` with a plain SpMM.
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
