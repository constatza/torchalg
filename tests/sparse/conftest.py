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
