"""Fixtures for ``torchalg.sparse.preconditioners.pod`` tests.

All test data is created via fixtures - never inline in test functions -
per project convention.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def pod_snapshots(poisson_1d_dense: torch.Tensor, test_seed: int) -> torch.Tensor:
    """Snapshot ensemble spanning ``poisson_1d_dense``'s solution space.

    Solves ``poisson_1d_dense @ x = b_i`` for several random RHS vectors, so
    a full-rank POD basis recovers the solution space exactly and a
    truncated basis approximates it.

    Args:
        poisson_1d_dense: The dense 12x12 Poisson matrix fixture
            (``tests/sparse/conftest.py``).
        test_seed: Fixed random seed for reproducibility.

    Returns:
        torch.Tensor: Snapshot ensemble, shape (8, 12) - one solution
            vector per row.
    """
    generator = torch.Generator().manual_seed(test_seed)
    n = poisson_1d_dense.shape[0]
    rhs_batch = torch.randn(8, n, dtype=poisson_1d_dense.dtype, generator=generator)
    return torch.linalg.solve(poisson_1d_dense, rhs_batch.T).T
