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

    Cost:
        Two SpGEMMs, ``AP`` then ``P^T (AP)``, with ``V = 8`` bytes per
        ``float64``: ``W = 2 sum_{a_ik != 0} nnz(P_k:) + 2 sum_{p_ji != 0}
        nnz((AP)_i:)`` (the exact multiply-add count of each product; for a
        tentative ``P`` with one entry per row the first term is ``2 nnz(A)``).
        ``Q`` is the bytes of the operands and results, about twice the
        one-pass minimum here because the COO path converts from CSR and
        coalesces (sorts) each product. ``D = 2 (ceil(log2 (m_A m_P)) + 1)``
        with ``m_A``, ``m_P`` the largest row counts, ``L`` a constant (two
        products plus the format conversions) and ``S = 0``.
        Roofline: ``W/Q`` is about 0.5 flop per byte (memory-bound).
        Work-depth: ``W/D`` is large (rows of a product are independent), so
        it parallelizes well; a CSR-to-CSR product would halve ``Q``.

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
