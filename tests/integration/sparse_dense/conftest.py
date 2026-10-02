"""Fixtures for the dense-vs-sparse operation equivalence checks.

Small, fixed-size fixtures only - this suite verifies each
``benchmarks.sparse_dense.operations`` function's sparse/native/scipy leg
against its dense reference. It is correctness coverage, not timing coverage.
"""

from __future__ import annotations

import pytest
import torch
from benchmarks.sparse_dense import matrices as bench_matrices

_SMALL_N = 200
_RANK = 20


@pytest.fixture
def sparse_prolongation_shape() -> tuple[int, int, int]:
    """Small ``(n, rank, nnz_per_row)`` case for structural prolongation tests."""
    return 50, 12, 4


@pytest.fixture
def rank_clamped_prolongation_shape() -> tuple[int, int, int]:
    """Case whose requested row sparsity exceeds its coarse rank."""
    return 10, 3, 100


@pytest.fixture
def small_dense_matrix(torch_dtype: torch.dtype) -> torch.Tensor:
    """Small SPD lattice-Laplacian matrix, dense."""
    scipy_matrix, _ = bench_matrices.synthetic_matrix(_SMALL_N, dims=2)
    return bench_matrices.to_torch_dense(scipy_matrix, dtype=torch_dtype)


@pytest.fixture
def small_sparse_matrix(torch_dtype: torch.dtype) -> torch.Tensor:
    """The same small SPD lattice-Laplacian matrix, sparse CSR."""
    scipy_matrix, _ = bench_matrices.synthetic_matrix(_SMALL_N, dims=2)
    return bench_matrices.to_torch_sparse_csr(scipy_matrix, dtype=torch_dtype)


@pytest.fixture
def matrix_size(small_dense_matrix: torch.Tensor) -> int:
    """Actual matrix size (the lattice generator only approximates a target N)."""
    return small_dense_matrix.shape[0]


@pytest.fixture
def small_vector(matrix_size: int, torch_dtype: torch.dtype, test_seed: int) -> torch.Tensor:
    """Fixed-seed random vector matching ``small_dense_matrix``'s size."""
    generator = torch.Generator().manual_seed(test_seed)
    return torch.randn(matrix_size, dtype=torch_dtype, generator=generator)


@pytest.fixture
def prolongation_dense(matrix_size: int, torch_dtype: torch.dtype, test_seed: int) -> torch.Tensor:
    """Dense random ``(n, r)`` transfer operator ``P`` for the Galerkin-product checks."""
    generator = torch.Generator().manual_seed(test_seed)
    return torch.randn(matrix_size, _RANK, dtype=torch_dtype, generator=generator)


@pytest.fixture
def prolongation_sparse(prolongation_dense: torch.Tensor) -> torch.Tensor:
    """The same transfer operator, sparse CSR (dense-random values, sparse *format* only)."""
    return prolongation_dense.to_sparse_csr()


@pytest.fixture
def coarse_vector(torch_dtype: torch.dtype, test_seed: int) -> torch.Tensor:
    """Fixed-seed random ``(r,)`` vector for coarse-space operations."""
    generator = torch.Generator().manual_seed(test_seed)
    return torch.randn(_RANK, dtype=torch_dtype, generator=generator)


@pytest.fixture
def snapshot_matrix(matrix_size: int, torch_dtype: torch.dtype, test_seed: int) -> torch.Tensor:
    """Thin ``(n_samples, n_dofs)`` snapshot matrix, matching POD's actual SVD call shape."""
    generator = torch.Generator().manual_seed(test_seed)
    return torch.randn(30, matrix_size, dtype=torch_dtype, generator=generator)
