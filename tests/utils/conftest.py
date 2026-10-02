"""Shared fixtures for ``tests/utils/test_ls_interpolation.py``.

Kept minimal and local to this directory since the batched LS-interpolation
tests are the only consumers so far; promote to the root conftest if a
second consumer appears (single source of truth, no premature sharing).
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def ls_test_vectors(test_seed: int) -> torch.Tensor:
    """Small random dense test-vector matrix ``V``, shape ``(12, 7)``.

    ``k=7`` is deliberately larger than the test suite's ``caliber=4``, so
    every trial interpolatory set stays strictly rank-deficient relative to
    ``k`` and its LS residual stays safely away from the ``~0`` floating-
    point noise floor a square/overdetermined (``|C_i| >= k``) exact fit
    would hit - at ``|C_i| == k`` the normal-equations residual for *every*
    remaining candidate collapses to numerical noise, making "the minimizing
    candidate" an unstable, batched-vs-looped-computation-order-dependent
    tie rather than a real comparison, which previously made this fixture
    (at ``k=3``) produce spurious batched/single-row mismatches unrelated to
    the functions under test.

    Args:
        test_seed: Fixed seed for reproducibility.

    Returns:
        Dense float64 tensor, shape ``(n=12, k=7)``.
    """
    generator = torch.Generator().manual_seed(test_seed)
    return torch.randn(12, 7, generator=generator, dtype=torch.float64)


@pytest.fixture
def ls_weights() -> torch.Tensor:
    """Per-test-vector weights ``omega``, shape ``(7,)``.

    Returns:
        Positive float64 weights.
    """
    return torch.tensor([1.0, 2.0, 0.5, 1.5, 0.8, 1.2, 0.6], dtype=torch.float64)
