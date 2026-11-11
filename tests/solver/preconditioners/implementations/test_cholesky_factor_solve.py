"""``cholesky_factor_solve`` against ``torch.cholesky_solve`` as the oracle.

The helper solves ``(L L^T) z = r`` with two ``solve_triangular`` calls because
LAPACK ``potrs`` (behind ``cholesky_solve``) runs several times slower on CPU for
one right-hand side; the two are the same substitution, so results agree to
rounding.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations._triangular import cholesky_factor_solve

SIZE = 64
SEED = 7


@pytest.fixture
def lower_factor(torch_dtype: torch.dtype) -> torch.Tensor:
    """Lower Cholesky factor of a seeded, well-conditioned SPD matrix."""
    generator = torch.Generator().manual_seed(SEED)
    base = torch.randn(SIZE, SIZE, dtype=torch_dtype, generator=generator)
    spd = base @ base.T + SIZE * torch.eye(SIZE, dtype=torch_dtype)
    return torch.linalg.cholesky(spd)


@pytest.fixture(params=[(SIZE,), (SIZE, 3)], ids=["vector", "block"])
def right_hand_side(request: pytest.FixtureRequest, torch_dtype: torch.dtype) -> torch.Tensor:
    """Seeded residual vector or residual block."""
    generator = torch.Generator().manual_seed(SEED + 1)
    return torch.randn(request.param, dtype=torch_dtype, generator=generator)


def test_matches_cholesky_solve(lower_factor: torch.Tensor, right_hand_side: torch.Tensor) -> None:
    rhs_matrix = right_hand_side if right_hand_side.ndim > 1 else right_hand_side.unsqueeze(-1)
    expected = torch.cholesky_solve(rhs_matrix, lower_factor, upper=False)
    if right_hand_side.ndim == 1:
        expected = expected.squeeze(-1)

    result = cholesky_factor_solve(lower_factor, right_hand_side)

    assert result.shape == right_hand_side.shape
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=0.0)
