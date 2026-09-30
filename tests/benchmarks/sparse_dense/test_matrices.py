"""Structural checks for ``scripts/bench_sparse_vs_dense/matrices.py`` helpers."""

from __future__ import annotations

import pytest
import torch
from bench_sparse_vs_dense import matrices as bench_matrices

pytestmark = pytest.mark.benchmark


def test_sparse_prolongation_has_exact_nnz_per_row(test_seed: int) -> None:
    torch.manual_seed(test_seed)
    n, rank, nnz_per_row = 50, 12, 4

    p = bench_matrices.sparse_prolongation(n, rank, nnz_per_row=nnz_per_row, dtype=torch.float64)

    assert p.shape == (n, rank)
    assert p.dtype == torch.float64
    row_nnz = p.crow_indices().diff()
    assert torch.all(row_nnz == nnz_per_row)


def test_sparse_prolongation_clamps_nnz_per_row_to_rank(test_seed: int) -> None:
    torch.manual_seed(test_seed)
    n, rank = 10, 3

    p = bench_matrices.sparse_prolongation(n, rank, nnz_per_row=100, dtype=torch.float64)

    row_nnz = p.crow_indices().diff()
    assert torch.all(row_nnz == rank)
