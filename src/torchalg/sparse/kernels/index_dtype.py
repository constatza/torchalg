"""CSR index-width compaction for operators applied many times.

PyTorch builds CSR tensors with int64 ``crow_indices``/``col_indices``, but the CPU
sparse BLAS behind ``csr @ x`` (MKL) works on 32-bit indices, so every product with
an int64 matrix first copies and narrows both index arrays. Narrowing once at setup
removes that copy: SpMV measured 1.5-2.5x faster at n = 1.7k-90k and an adaptive SA
V-cycle 1.45-1.57x faster, with bit-identical results (the same kernel runs on the
same values). Only apply-time operators are narrowed; setup products (SpGEMM)
promote back to int64 on their own.
"""

from __future__ import annotations

import torch

INT32_INDEX_LIMIT = 2**31 - 1
"""Largest dimension or stored-entry count an int32 CSR index can address."""


def compact_csr_indices(matrix: torch.Tensor) -> torch.Tensor:
    """Return ``matrix`` with int32 CSR indices, sharing its values tensor.

    A non-CSR tensor, an already-int32 CSR tensor, or one too large for 32-bit
    indices is returned unchanged.

    Args:
        matrix (torch.Tensor): Any tensor; only sparse CSR is converted.

    Returns:
        torch.Tensor: The same operator, with int32 indices where possible.
    """
    if matrix.layout != torch.sparse_csr or matrix.crow_indices().dtype == torch.int32:
        return matrix
    if max(*matrix.shape, matrix.values().numel()) > INT32_INDEX_LIMIT:
        return matrix
    return torch.sparse_csr_tensor(
        matrix.crow_indices().to(torch.int32),
        matrix.col_indices().to(torch.int32),
        matrix.values(),
        matrix.shape,
    )
