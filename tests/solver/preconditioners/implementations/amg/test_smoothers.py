"""Dense-path tests for ``JacobiSmoother``/``GaussSeidelSmoother``.

See ``tests/sparse/preconditioners/amg/test_smoothers.py`` for the sparse
CSR siblings' tests - rewritten as "two classes, same result" rather than
mechanically moved, since the old shape here ("one class, assert both
branches agree") no longer applies once the sparse dispatch branch was
removed from these dense classes (``docs/plan.md``'s "Correction: dense and
sparse must be separate implementations, not an internal branch").
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations.amg.smoothers import (
    GaussSeidelSmoother,
    JacobiSmoother,
)


class TestJacobiSmootherDense:
    """Weighted Jacobi smoothing reduces the residual on a dense SPD system."""

    @pytest.mark.parametrize("steps", [1, 3])
    def test_reduces_residual_norm(self, poisson_1d: torch.Tensor, steps: int) -> None:
        rhs = torch.ones(poisson_1d.shape[0], dtype=poisson_1d.dtype)
        x0 = torch.zeros_like(rhs)
        smoother = JacobiSmoother()

        x = smoother.smooth(poisson_1d, rhs, x0, steps)

        residual_before = torch.linalg.norm(rhs - poisson_1d @ x0)
        residual_after = torch.linalg.norm(rhs - poisson_1d @ x)
        assert residual_after < residual_before


class TestGaussSeidelSmootherDense:
    """Symmetric Gauss-Seidel smoothing reduces the residual on a dense SPD system."""

    @pytest.mark.parametrize("steps", [1, 3])
    def test_reduces_residual_norm(self, poisson_1d: torch.Tensor, steps: int) -> None:
        rhs = torch.ones(poisson_1d.shape[0], dtype=poisson_1d.dtype)
        x0 = torch.zeros_like(rhs)
        smoother = GaussSeidelSmoother()

        x = smoother.smooth(poisson_1d, rhs, x0, steps)

        residual_before = torch.linalg.norm(rhs - poisson_1d @ x0)
        residual_after = torch.linalg.norm(rhs - poisson_1d @ x)
        assert residual_after < residual_before
