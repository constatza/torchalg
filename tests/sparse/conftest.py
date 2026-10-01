"""Shared fixtures for ``torchalg.sparse`` tests.

All test data is created via fixtures - never inline in test functions -
per project convention. Reuses ``tests/conftest.py``'s ``poisson_1d_factory``
as the canonical sparse-capable SPD matrix source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def poisson_1d_dense(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    """Dense 12x12 1D Poisson matrix (tridiagonal [-1, 2, -1]).

    Args:
        poisson_1d_factory: Size-parametrized Poisson matrix factory
            (``tests/conftest.py``).

    Returns:
        torch.Tensor: Dense 12x12 SPD tridiagonal matrix.
    """
    return poisson_1d_factory(12)


@pytest.fixture
def poisson_1d_csr(poisson_1d_dense: torch.Tensor) -> torch.Tensor:
    """``poisson_1d_dense`` converted to sparse CSR.

    Args:
        poisson_1d_dense: Dense 12x12 Poisson matrix fixture.

    Returns:
        torch.Tensor: Sparse CSR twin of ``poisson_1d_dense``.
    """
    return poisson_1d_dense.to_sparse_csr()


@pytest.fixture
def poisson_1d_large_dense(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    """Dense 64x64 1D Poisson matrix (tridiagonal [-1, 2, -1]).

    Large enough to coarsen meaningfully across more than two AMG levels
    (``poisson_1d_dense``'s 12 dofs collapse too fast for an ``n_levels=3+``
    hierarchy to stay interesting).

    Args:
        poisson_1d_factory: Size-parametrized Poisson matrix factory
            (``tests/conftest.py``).

    Returns:
        torch.Tensor: Dense 64x64 SPD tridiagonal matrix.
    """
    return poisson_1d_factory(64)


@pytest.fixture
def poisson_1d_large_csr(poisson_1d_large_dense: torch.Tensor) -> torch.Tensor:
    """``poisson_1d_large_dense`` converted to sparse CSR.

    Args:
        poisson_1d_large_dense: Dense 64x64 Poisson matrix fixture.

    Returns:
        torch.Tensor: Sparse CSR twin of ``poisson_1d_large_dense``.
    """
    return poisson_1d_large_dense.to_sparse_csr()


@pytest.fixture
def poisson_1d_xlarge_dense(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    """Dense 300x300 1D Poisson matrix (tridiagonal [-1, 2, -1]).

    Larger than ``poisson_1d_large_dense`` specifically to exercise
    ``BAMGCoarsening._prolongation``'s vectorized candidate-gathering at a
    scale where fine rows plausibly land with several distinct
    coarse-neighbor candidate-set sizes (exercising the padded/masked
    layout's ragged-to-padded packing, not just the uniform case a small
    fixture could pass by accident).

    Args:
        poisson_1d_factory: Size-parametrized Poisson matrix factory
            (``tests/conftest.py``).

    Returns:
        torch.Tensor: Dense 300x300 SPD tridiagonal matrix.
    """
    return poisson_1d_factory(300)


@pytest.fixture
def bamg_default_test_vectors(
    poisson_1d_large_dense: torch.Tensor, torch_dtype: torch.dtype, test_seed: int
) -> torch.Tensor:
    """Seeded default-width BAMG vectors for dense/CSR value-parity checks."""
    generator = torch.Generator().manual_seed(test_seed)
    return torch.randn((poisson_1d_large_dense.shape[0], 8), dtype=torch_dtype, generator=generator)


@pytest.fixture
def poisson_1d_large_rhs(poisson_1d_large_dense: torch.Tensor) -> torch.Tensor:
    """Deterministic right-hand side for large sparse/dense solver parity tests."""
    return torch.ones(poisson_1d_large_dense.shape[0], dtype=poisson_1d_large_dense.dtype)


@pytest.fixture
def poisson_2d_dense(anisotropic_2d_factory: Callable[[int, float], torch.Tensor]) -> torch.Tensor:
    """Dense 9x9 2D isotropic Poisson matrix (5-point stencil on a 3x3 grid).

    Unlike ``poisson_1d_dense`` (tridiagonal, exactly one row per dependency
    level), this gives a genuine multi-row-per-level schedule, exercising the
    vectorized multi-row path in ``triangular.level_schedule``/
    ``triangular_solve``.

    Args:
        anisotropic_2d_factory: 2D anisotropic Poisson factory
            (``tests/conftest.py``).

    Returns:
        torch.Tensor: Dense 9x9 SPD 5-point-stencil matrix.
    """
    return anisotropic_2d_factory(3, 1.0)


@pytest.fixture
def poisson_2d_csr(poisson_2d_dense: torch.Tensor) -> torch.Tensor:
    """``poisson_2d_dense`` converted to sparse CSR.

    Args:
        poisson_2d_dense: Dense 9x9 Poisson matrix fixture.

    Returns:
        torch.Tensor: Sparse CSR twin of ``poisson_2d_dense``.
    """
    return poisson_2d_dense.to_sparse_csr()
