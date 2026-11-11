"""Cost contract of the adaptive setup's trial cycles (``adaptive._solve``)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from tests.support.matvec_counter import MatvecCounter
from torchalg.preconditioners.implementations.amg import adaptive
from torchalg.preconditioners.implementations.amg._presets import seeded_draw
from torchalg.preconditioners.implementations.amg.adaptive import (
    _Level,
    _Params,
    _smoothed_aggregation_levels,
    _solve,
)

_CYCLES = 3


@pytest.fixture
def levels(poisson_1d_factory: Callable[[int], torch.Tensor]) -> list[_Level]:
    matrix = poisson_1d_factory(64)
    params = _Params(0.0, 4.0 / 3.0, 2, 3, 4, seeded_draw(0))
    candidates = torch.ones(matrix.shape[0], 1, dtype=matrix.dtype)
    return _smoothed_aggregation_levels(matrix, candidates, params, None)


@pytest.fixture
def start(levels: list[_Level]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(2)
    return torch.rand(levels[0].A.shape[0], generator=generator, dtype=levels[0].A.dtype)


def _count_fine_matvecs(levels: list[_Level], run: Callable[[], object]) -> int:
    with MatvecCounter(levels[0].A) as counter:
        run()
    return counter.count


class TestSolveMatvecs:
    def test_one_fine_matvec_per_cycle_beyond_the_cycle_itself(
        self, levels: list[_Level], start: torch.Tensor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The tolerance never triggers, so every cycle runs.
        monkeypatch.setattr(adaptive, "_SOLVE_TOLERANCE", -1.0)
        hierarchy = adaptive.GS_SETUP_CYCLE.prepare(adaptive._hierarchy_of(levels))
        inside_cycles = _count_fine_matvecs(
            levels,
            lambda: [adaptive.GS_SETUP_CYCLE.apply(hierarchy, start) for _ in range(_CYCLES)],
        )

        total = _count_fine_matvecs(levels, lambda: _solve(levels, start, _CYCLES))

        # One residual for the first cycle, then one new residual per cycle (reused as the
        # next cycle's input and as the convergence test).
        assert total - inside_cycles == _CYCLES + 1
