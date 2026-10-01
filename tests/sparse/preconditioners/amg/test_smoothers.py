"""Sparse-CSR tests for ``torchalg.sparse.preconditioners.amg.smoothers``.

Cross-checks the standalone sparse ``JacobiSmoother``/``GaussSeidelSmoother``
classes against their dense siblings
(``torchalg.preconditioners.implementations.amg.smoothers``) on the same
matrix - same algorithm, two classes now instead of one class's two
branches (``docs/plan.md``'s "Correction: dense and sparse must be separate
implementations, not an internal branch").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from torchalg.preconditioners.implementations.amg.smoothers import (
    GaussSeidelSmoother as DenseGaussSeidelSmoother,
)
from torchalg.preconditioners.implementations.amg.smoothers import (
    JacobiSmoother as DenseJacobiSmoother,
)
from torchalg.sparse.preconditioners.amg.smoothers import GaussSeidelSmoother, JacobiSmoother

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def poisson_2d(anisotropic_2d_factory: Callable[[int, float], torch.Tensor]) -> torch.Tensor:
    """Dense 9x9 2D isotropic Poisson matrix (5-point stencil on a 3x3 grid).

    Gives genuine multi-row dependency levels, unlike the tridiagonal 1D
    fixture (one row per level).
    """
    return anisotropic_2d_factory(3, 1.0)


@pytest.fixture
def nonzero_x(poisson_1d_dense: torch.Tensor) -> torch.Tensor:
    """Nonzero initial iterate, matching ``poisson_1d_dense``'s size."""
    return torch.arange(1, poisson_1d_dense.shape[0] + 1, dtype=poisson_1d_dense.dtype) * 0.1


@pytest.fixture
def nonzero_x_2d(poisson_2d: torch.Tensor) -> torch.Tensor:
    """Nonzero initial iterate, matching ``poisson_2d``'s size."""
    return torch.arange(1, poisson_2d.shape[0] + 1, dtype=poisson_2d.dtype) * 0.1


@pytest.fixture
def nonzero_rhs(poisson_1d_dense: torch.Tensor) -> torch.Tensor:
    """Nonzero right-hand side, matching ``poisson_1d_dense``'s size."""
    return torch.ones(poisson_1d_dense.shape[0], dtype=poisson_1d_dense.dtype) * 2.0


@pytest.fixture
def nonzero_rhs_2d(poisson_2d: torch.Tensor) -> torch.Tensor:
    """Nonzero right-hand side, matching ``poisson_2d``'s size."""
    return torch.ones(poisson_2d.shape[0], dtype=poisson_2d.dtype) * 2.0


class TestJacobiSmootherSparseMatchesDense:
    """The sparse ``JacobiSmoother`` matches the dense sibling exactly."""

    @pytest.mark.parametrize("steps", [1, 3])
    def test_matches_dense_poisson_1d(
        self,
        poisson_1d_dense: torch.Tensor,
        nonzero_x: torch.Tensor,
        nonzero_rhs: torch.Tensor,
        steps: int,
    ) -> None:
        dense_result = DenseJacobiSmoother().smooth(poisson_1d_dense, nonzero_rhs, nonzero_x, steps)
        sparse_result = JacobiSmoother().smooth(
            poisson_1d_dense.to_sparse_csr(), nonzero_rhs, nonzero_x, steps
        )
        assert not torch.allclose(dense_result, torch.zeros_like(dense_result))
        torch.testing.assert_close(sparse_result, dense_result)

    @pytest.mark.parametrize("steps", [1, 3])
    def test_matches_dense_poisson_2d(
        self,
        poisson_2d: torch.Tensor,
        nonzero_x_2d: torch.Tensor,
        nonzero_rhs_2d: torch.Tensor,
        steps: int,
    ) -> None:
        dense_result = DenseJacobiSmoother().smooth(poisson_2d, nonzero_rhs_2d, nonzero_x_2d, steps)
        sparse_result = JacobiSmoother().smooth(
            poisson_2d.to_sparse_csr(), nonzero_rhs_2d, nonzero_x_2d, steps
        )
        assert not torch.allclose(dense_result, torch.zeros_like(dense_result))
        torch.testing.assert_close(sparse_result, dense_result)


class TestGaussSeidelSmootherSparseMatchesDense:
    """The sparse ``GaussSeidelSmoother`` matches the dense sibling exactly."""

    @pytest.mark.parametrize("steps", [1, 3])
    def test_matches_dense_poisson_1d(
        self,
        poisson_1d_dense: torch.Tensor,
        nonzero_x: torch.Tensor,
        nonzero_rhs: torch.Tensor,
        steps: int,
    ) -> None:
        dense_result = DenseGaussSeidelSmoother().smooth(
            poisson_1d_dense, nonzero_rhs, nonzero_x, steps
        )
        sparse_result = GaussSeidelSmoother().smooth(
            poisson_1d_dense.to_sparse_csr(), nonzero_rhs, nonzero_x, steps
        )
        assert not torch.allclose(dense_result, torch.zeros_like(dense_result))
        torch.testing.assert_close(sparse_result, dense_result)

    @pytest.mark.parametrize("steps", [1, 3])
    def test_matches_dense_poisson_2d(
        self,
        poisson_2d: torch.Tensor,
        nonzero_x_2d: torch.Tensor,
        nonzero_rhs_2d: torch.Tensor,
        steps: int,
    ) -> None:
        dense_result = DenseGaussSeidelSmoother().smooth(
            poisson_2d, nonzero_rhs_2d, nonzero_x_2d, steps
        )
        sparse_result = GaussSeidelSmoother().smooth(
            poisson_2d.to_sparse_csr(), nonzero_rhs_2d, nonzero_x_2d, steps
        )
        assert not torch.allclose(dense_result, torch.zeros_like(dense_result))
        torch.testing.assert_close(sparse_result, dense_result)
