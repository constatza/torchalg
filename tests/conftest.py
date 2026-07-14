"""Session-wide fixtures shared across the torchalg test suite.

Solver-specific fixtures (SPD matrices, RHS vectors, tolerance tiers,
preconditioner factories, ...) live in ``tests/solver/conftest.py``.
Benchmark/exactness fixtures and helpers live in
``tests/benchmarks/exactness/conftest.py``. This module mostly holds
fixtures that are generic across the whole suite: the fixed random seed and
the numpy -> torch adapter layer that later-stage torch tests will use to
consume the same numpy data the scipy oracle sees. It also hosts
``poisson_1d_factory`` (Stage 5, AMG): not itself suite-generic, but placed
here anyway because it is the only conftest that is an ancestor of both its
consumers (``tests/solver/preconditioners/implementations/test_amg.py`` and
``tests/benchmarks/preconditioners/test_amg_variants.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

TEST_SEED = 42
"""Fixed random seed for reproducible fixture data (reference convention)."""


@pytest.fixture
def test_seed() -> int:
    """Fixed random seed for reproducible fixture data.

    Returns:
        The seed value, matching the reference test suite's convention of
        seeding ``np.random.default_rng`` with 42.
    """
    return TEST_SEED


@pytest.fixture
def torch_dtype() -> torch.dtype:
    """Default dtype for torch-side fixture data.

    Returns:
        ``torch.float64``, matching the reference's float64-first numerics,
        which is required for correctness parity against the scipy oracle.
    """
    return torch.float64


@pytest.fixture
def to_torch(torch_dtype: torch.dtype) -> Callable[[NDArray], torch.Tensor]:
    """Adapter converting a numpy fixture array into an independent torch tensor.

    The base SPD matrix / RHS / system fixtures (``tests/solver/conftest.py``)
    stay plain numpy so they are bit-identical inputs to both the scipy
    oracle and the torch code under test; this fixture is the single,
    explicit bridge from that shared numpy data into a torch tensor.

    Args:
        torch_dtype: Default dtype to convert into.

    Returns:
        Callable ``(array, dtype=None) -> torch.Tensor`` that returns a
        cloned tensor (never a view into the numpy array), so mutating the
        tensor in a torch-side test can never corrupt the shared numpy
        fixture data.
    """

    def _to_torch(array: NDArray, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Convert a numpy array to a cloned torch tensor.

        Args:
            array: Source numpy array.
            dtype: Target torch dtype; defaults to ``torch_dtype``.

        Returns:
            Cloned torch tensor with no aliasing to ``array``.
        """
        return torch.from_numpy(np.asarray(array)).to(dtype=dtype or torch_dtype).clone()

    return _to_torch


@pytest.fixture
def poisson_1d_factory(torch_dtype: torch.dtype) -> Callable[[int], torch.Tensor]:
    """Factory building a 1D Poisson matrix (tridiagonal [-1, 2, -1]) of size n.

    Shared between ``tests/solver/preconditioners/implementations/test_amg.py``
    (fixed size, AMG unit tests) and
    ``tests/benchmarks/preconditioners/test_amg_variants.py`` (varying sizes,
    AMG benchmark tests) - consolidated here rather than duplicated per-file
    (the reference duplicates a ``_poisson_1d`` helper across both test
    files; this repo's DRY convention keeps a single source of truth
    instead). Lives in the root conftest, not ``tests/solver/conftest.py``,
    because it is the only conftest that is an ancestor of both consumer
    directories.

    Args:
        torch_dtype: Default dtype for the returned matrix.

    Returns:
        Callable ``(n) -> torch.Tensor`` building a dense ``(n, n)`` SPD
        tridiagonal matrix with diagonal 2 and off-diagonals -1.
    """

    def _factory(n: int) -> torch.Tensor:
        """Build the ``(n, n)`` 1D Poisson matrix.

        Args:
            n: System size.

        Returns:
            Dense SPD tridiagonal matrix, diag=2, off-diag=-1.
        """
        return (
            2 * torch.eye(n, dtype=torch_dtype)
            - torch.diag(torch.ones(n - 1, dtype=torch_dtype), 1)
            - torch.diag(torch.ones(n - 1, dtype=torch_dtype), -1)
        )

    return _factory
