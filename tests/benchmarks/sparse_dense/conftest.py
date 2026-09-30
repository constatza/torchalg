"""Fixtures for the dense-vs-sparse operation equivalence checks.

Small, fixed-size fixtures only - this suite exists to verify each
``scripts/bench_sparse_vs_dense/operations.py`` function's sparse/native/
scipy leg agrees numerically with its dense reference, not to time
anything (that's ``scripts/bench_sparse_vs_dense/run_benchmark.py``, opt-in
and not part of the test suite at all).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# scripts/ isn't an installed package - add it to sys.path the same way the
# scripts themselves do, rather than duplicating matrices.py/operations.py
# under tests/.
_SCRIPTS_DIR = Path(__file__).parents[3] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from bench_sparse_vs_dense import matrices as bench_matrices

_SMALL_N = 200
_RANK = 20


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
