"""Fixtures for ``torchalg.strategies`` tests.

Modular, composable fixtures per project convention - no inline test data.
Builds on the session-wide ``torch_dtype`` fixture from ``tests/conftest.py``.
"""

from __future__ import annotations

import pytest
import torch


# =============================================================================
# norms.py fixtures
# =============================================================================


@pytest.fixture
def three_four_vector(torch_dtype: torch.dtype) -> torch.Tensor:
    """A 2-vector with a hand-computable Euclidean norm of 5.0.

    Returns:
        torch.Tensor: ``[3.0, 4.0]``.
    """
    return torch.tensor([3.0, 4.0], dtype=torch_dtype)


@pytest.fixture
def energy_norm_diag_entries(torch_dtype: torch.dtype) -> torch.Tensor:
    """Diagonal SPD entries ``[2.0, 8.0]`` for ``energy_norm`` tests.

    Returns:
        torch.Tensor: 1-D tensor of diagonal entries.
    """
    return torch.tensor([2.0, 8.0], dtype=torch_dtype)


@pytest.fixture
def energy_norm_full_matrix(
    energy_norm_diag_entries: torch.Tensor,
) -> torch.Tensor:
    """Full 2x2 SPD matrix with the same spectrum as ``energy_norm_diag_entries``.

    Returns:
        torch.Tensor: ``diag([2.0, 8.0])`` as a dense 2x2 matrix, so the
            diagonal and full-matrix branches of ``energy_norm`` can be
            checked against each other.
    """
    return torch.diag(energy_norm_diag_entries)


@pytest.fixture
def ones_vector_2(torch_dtype: torch.dtype) -> torch.Tensor:
    """A 2-vector of ones, the probe vector for the energy-norm fixtures.

    Returns:
        torch.Tensor: ``[1.0, 1.0]``.
    """
    return torch.tensor([1.0, 1.0], dtype=torch_dtype)


# =============================================================================
# convergence.py fixtures
# =============================================================================


@pytest.fixture
def converged_residual(torch_dtype: torch.dtype) -> torch.Tensor:
    """A residual vector well within a ``rtol=1e-6`` tolerance for ``rhs_norm=1.0``.

    Returns:
        torch.Tensor: A vector with Euclidean norm ``1e-8``.
    """
    return torch.tensor([1e-8, 0.0], dtype=torch_dtype)


@pytest.fixture
def unconverged_residual(torch_dtype: torch.dtype) -> torch.Tensor:
    """A residual vector well outside a ``rtol=1e-6`` tolerance for ``rhs_norm=1.0``.

    Returns:
        torch.Tensor: A vector with Euclidean norm ``1.0``.
    """
    return torch.tensor([1.0, 0.0], dtype=torch_dtype)


@pytest.fixture
def nan_residual(torch_dtype: torch.dtype) -> torch.Tensor:
    """A residual vector whose norm is non-finite (contains ``nan``).

    Returns:
        torch.Tensor: A vector with a ``nan`` entry.
    """
    return torch.tensor([float("nan"), 0.0], dtype=torch_dtype)


# =============================================================================
# direction.py / orthogonalization.py fixtures
# =============================================================================


@pytest.fixture
def orthogonalization_probe(torch_dtype: torch.dtype) -> torch.Tensor:
    """Vector to orthogonalize against a one-vector A-conjugacy history."""
    return torch.tensor([2.0, 1.0], dtype=torch_dtype)


@pytest.fixture
def previous_direction(torch_dtype: torch.dtype) -> torch.Tensor:
    """Previous search direction for hand-computable orthogonalization tests."""
    return torch.tensor([1.0, 0.0], dtype=torch_dtype)


@pytest.fixture
def previous_matrix_product(torch_dtype: torch.dtype) -> torch.Tensor:
    """Matrix product A @ previous_direction for A=diag([2, 3])."""
    return torch.tensor([2.0, 0.0], dtype=torch_dtype)


@pytest.fixture
def two_term_current_residual(torch_dtype: torch.dtype) -> torch.Tensor:
    """Current residual for Fletcher-Reeves direction tests."""
    return torch.tensor([1.0, 1.0], dtype=torch_dtype)


@pytest.fixture
def two_term_preconditioned_residual(torch_dtype: torch.dtype) -> torch.Tensor:
    """Current preconditioned residual for Fletcher-Reeves direction tests."""
    return torch.tensor([0.5, 0.5], dtype=torch_dtype)


@pytest.fixture
def two_term_previous_direction(torch_dtype: torch.dtype) -> torch.Tensor:
    """Previous direction for Fletcher-Reeves direction tests."""
    return torch.tensor([2.0, 0.0], dtype=torch_dtype)
