"""Structural checks for ``benchmarks.sparse_dense.matrices`` helpers."""

from __future__ import annotations

import torch
from benchmarks.sparse_dense import matrices as bench_matrices


def test_sparse_prolongation_has_exact_nnz_per_row(
    test_seed: int, sparse_prolongation_shape: tuple[int, int, int]
) -> None:
    torch.manual_seed(test_seed)
    n, rank, nnz_per_row = sparse_prolongation_shape

    p = bench_matrices.sparse_prolongation(n, rank, nnz_per_row=nnz_per_row, dtype=torch.float64)

    assert p.shape == (n, rank)
    assert p.dtype == torch.float64
    row_nnz = p.crow_indices().diff()
    assert torch.all(row_nnz == nnz_per_row)


def test_sparse_prolongation_clamps_nnz_per_row_to_rank(
    test_seed: int, rank_clamped_prolongation_shape: tuple[int, int, int]
) -> None:
    torch.manual_seed(test_seed)
    n, rank, nnz_per_row = rank_clamped_prolongation_shape

    p = bench_matrices.sparse_prolongation(n, rank, nnz_per_row=nnz_per_row, dtype=torch.float64)

    row_nnz = p.crow_indices().diff()
    assert torch.all(row_nnz == rank)
