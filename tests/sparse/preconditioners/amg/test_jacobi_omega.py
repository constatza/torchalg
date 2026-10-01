"""Tests for ``torchalg.sparse.preconditioners.amg._jacobi_omega``.

The 2 sparse test cases from
``tests/solver/preconditioners/implementations/test_jacobi_omega.py``,
moved here per ``docs/plan.md``'s migration notes (the pre-existing dense
cases stay at that path).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torchalg.preconditioners.implementations.amg import _jacobi_omega as dense_jacobi_omega_module
from torchalg.preconditioners.implementations.amg._jacobi_omega import (
    jacobi_spectral_radius as dense_jacobi_spectral_radius,
)
from torchalg.sparse.preconditioners.amg import _jacobi_omega as jacobi_omega_module
from torchalg.sparse.preconditioners.amg._jacobi_omega import jacobi_spectral_radius


@pytest.fixture
def poisson_matrix(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    """64-node 1D Poisson matrix."""
    return poisson_1d_factory(64)


class TestSparseJacobiSpectralRadius:
    """``jacobi_spectral_radius``/``scaled_by_inverse_diagonal`` accept sparse CSR input."""

    def test_matches_dense_reference(self, poisson_matrix: torch.Tensor) -> None:
        """The Arnoldi estimate is identical whether ``A`` is dense or sparse CSR."""
        dense_rho = dense_jacobi_spectral_radius(poisson_matrix)
        sparse_rho = jacobi_spectral_radius(poisson_matrix.to_sparse_csr())
        torch.testing.assert_close(sparse_rho, dense_rho)

    def test_scaled_by_inverse_diagonal_matches_dense_reference(
        self, poisson_matrix: torch.Tensor
    ) -> None:
        """``D^-1 A`` densified from the sparse path equals the dense path exactly."""
        dense = dense_jacobi_omega_module.scaled_by_inverse_diagonal(poisson_matrix)
        sparse = jacobi_omega_module.scaled_by_inverse_diagonal(poisson_matrix.to_sparse_csr())
        torch.testing.assert_close(sparse.to_dense(), dense)
