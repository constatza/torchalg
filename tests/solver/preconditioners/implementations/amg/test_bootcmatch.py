"""End-to-end PCG convergence test for ``BootCMatchPreconditioner`` (dense).

Correctness/convergence only, per this project's acceptance criteria for
BootCMatch: explicitly not required to match ``BootstrapAMGPreconditioner``'s
iteration count or coarse grids (a different published algorithm).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torchalg import pcg
from torchalg.preconditioners.implementations.amg.bootcmatch import BootCMatchPreconditioner


@pytest.fixture
def poisson_64(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    return poisson_1d_factory(64)


@pytest.fixture
def poisson_64_rhs(poisson_64: torch.Tensor) -> torch.Tensor:
    return torch.ones(poisson_64.shape[0], dtype=poisson_64.dtype)


def test_pcg_converges_with_bootcmatch_preconditioner(
    poisson_64: torch.Tensor, poisson_64_rhs: torch.Tensor
) -> None:
    preconditioner = BootCMatchPreconditioner(
        poisson_64,
        seed=3,
        max_levels=4,
        max_coarse=4,
        k_max=4,
        rho_desired=0.8,
        max_hierarchies=3,
    )

    assert not preconditioner.requires_flexible_cg
    assert len(preconditioner.hierarchies) >= 1

    solution, info = pcg(
        poisson_64,
        poisson_64_rhs,
        preconditioner=preconditioner,
        rtol=1e-8,
        maxiter=200,
    )

    assert info.converged
    residual_norm = torch.linalg.norm(poisson_64_rhs - poisson_64 @ solution)
    assert residual_norm < 1e-6
