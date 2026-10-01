"""Tests for ``torchalg.multigrid.bootcmatch_setup.BootCMatchSetup``.

Exercises the outer bootstrap loop's own bookkeeping (hierarchy count,
monitored convergence rate, per-level coarsening sanity) - not per-level
correctness, which is already covered by
``tests/preconditioners/implementations/amg/test_bootcmatch_coarsening.py``/
``tests/sparse/preconditioners/amg/test_bootcmatch_coarsening.py``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import pytest
import torch

from torchalg.multigrid import VCycle
from torchalg.multigrid.bootcmatch_setup import BootCMatchSetup
from torchalg.preconditioners.implementations.amg.bootcmatch_coarsening import (
    BootCMatchCoarsening,
)
from torchalg.preconditioners.implementations.amg.smoothers import JacobiSmoother


@pytest.fixture
def seeded_draw() -> Callable[[int], torch.Tensor]:
    generator = torch.Generator().manual_seed(0)

    def _draw(n: int) -> torch.Tensor:
        return torch.rand(n, generator=generator, dtype=torch.float64)

    return _draw


@pytest.fixture
def poisson_matrix(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    return poisson_1d_factory(32)


class TestBootCMatchSetupRun:
    def test_returns_at_least_one_valid_hierarchy(
        self, poisson_matrix: torch.Tensor, seeded_draw: Callable[[int], torch.Tensor]
    ) -> None:
        setup = BootCMatchSetup(
            coarsening_factory=BootCMatchCoarsening,
            cycle=VCycle(JacobiSmoother(omega=0.5)),
            draw=seeded_draw,
            max_levels=4,
            max_coarse=4,
            k_max=3,
            rho_desired=0.99,
            max_hierarchies=1,
        )

        hierarchies = setup.run(poisson_matrix)

        assert len(hierarchies) >= 1
        for hierarchy in hierarchies:
            sizes = [level.matrix.shape[0] for level in hierarchy.levels]
            assert sizes == sorted(sizes, reverse=True)
            assert all(a > b for a, b in itertools.pairwise(sizes))
            assert hierarchy.levels[-1].transfer is None
            assert all(level.transfer is not None for level in hierarchy.levels[:-1])

    def test_low_rho_desired_grows_composite_and_computes_finite_rate(
        self, poisson_matrix: torch.Tensor, seeded_draw: Callable[[int], torch.Tensor]
    ) -> None:
        setup = BootCMatchSetup(
            coarsening_factory=BootCMatchCoarsening,
            cycle=VCycle(JacobiSmoother(omega=0.5)),
            draw=seeded_draw,
            max_levels=4,
            max_coarse=4,
            k_max=2,
            rho_desired=0.0,
            max_hierarchies=3,
        )

        hierarchies = setup.run(poisson_matrix)

        assert len(hierarchies) > 1
        assert len(hierarchies) <= 3

    def test_seed_vector_is_used_for_the_first_hierarchy(
        self, poisson_matrix: torch.Tensor, seeded_draw: Callable[[int], torch.Tensor]
    ) -> None:
        n = poisson_matrix.shape[0]
        seed_vector = torch.ones(n, dtype=poisson_matrix.dtype)
        setup = BootCMatchSetup(
            coarsening_factory=BootCMatchCoarsening,
            cycle=VCycle(JacobiSmoother(omega=0.5)),
            draw=seeded_draw,
            max_levels=3,
            max_coarse=4,
            k_max=2,
            rho_desired=0.99,
            max_hierarchies=1,
        )

        coarsening = BootCMatchCoarsening(seed_vector.clone())
        expected_coarse, _ = coarsening.build_transfer(poisson_matrix)

        hierarchies = setup.run(poisson_matrix, seed_vector=seed_vector.clone())

        torch.testing.assert_close(hierarchies[0].levels[1].matrix, expected_coarse)
