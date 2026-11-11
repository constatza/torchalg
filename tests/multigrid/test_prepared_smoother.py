"""Per-level prepared smoothers: bound to the level matrix once, used by every cycle."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torchalg.multigrid.coarse_solve import HierarchyNotPreparedError
from torchalg.multigrid.cycle import VCycle, WCycle
from torchalg.multigrid.hierarchy import MultigridHierarchy, build_hierarchy
from torchalg.multigrid.protocols import PreparedSmoother
from torchalg.preconditioners.implementations.amg import AggregationCoarsening

N_LEVELS = 3
POISSON_SIZE = 40
N_PRE = 2
N_POST = 3


class RecordingPrepared:
    """Damped-Jacobi sweeps that log which entry point the cycle used."""

    def __init__(self, matrix: torch.Tensor, log: list[str]) -> None:
        self._matrix = matrix
        self._log = log

    def smooth(self, rhs: torch.Tensor, x: torch.Tensor, steps: int) -> torch.Tensor:
        self._log.append(f"smooth:{steps}")
        diag = torch.diagonal(self._matrix)
        for _ in range(steps):
            x = x + (rhs - self._matrix @ x) / diag * 0.6
        return x

    def smooth_from_zero(self, rhs: torch.Tensor, steps: int) -> torch.Tensor:
        self._log.append(f"zero:{steps}")
        return self.smooth(rhs, torch.zeros_like(rhs), steps)

    def residual(self, rhs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._log.append("residual")
        return rhs - self._matrix @ x


class RecordingSmoother:
    """Smoother factory that counts ``prepare`` calls."""

    def __init__(self) -> None:
        self.prepared_for: list[int] = []
        self.sweep_log: list[str] = []

    def prepare(self, A: torch.Tensor) -> PreparedSmoother:
        self.prepared_for.append(A.shape[0])
        return RecordingPrepared(A, self.sweep_log)


@pytest.fixture
def hierarchy(poisson_1d_factory: Callable[[int], torch.Tensor]) -> MultigridHierarchy:
    return build_hierarchy(poisson_1d_factory(POISSON_SIZE), AggregationCoarsening(), N_LEVELS)


@pytest.fixture
def rhs() -> torch.Tensor:
    return torch.linspace(1.0, 2.0, POISSON_SIZE, dtype=torch.float64)


@pytest.mark.parametrize("cycle_class", [VCycle, WCycle])
class TestPreparedSmoothers:
    def test_every_non_coarsest_level_is_prepared_once(
        self, cycle_class: type[VCycle | WCycle], hierarchy: MultigridHierarchy, rhs: torch.Tensor
    ) -> None:
        smoother = RecordingSmoother()
        cycle = cycle_class(smoother, n_pre=N_PRE, n_post=N_POST)

        prepared = cycle.prepare(hierarchy)
        for _ in range(3):
            cycle.apply(prepared, rhs)

        expected = [level.matrix.shape[0] for level in hierarchy.levels[:-1]]
        assert smoother.prepared_for == expected
        assert prepared.levels[-1].smoother is None
        assert all(level.smoother is not None for level in prepared.levels[:-1])

    def test_prepare_is_idempotent(
        self, cycle_class: type[VCycle | WCycle], hierarchy: MultigridHierarchy
    ) -> None:
        smoother = RecordingSmoother()
        cycle = cycle_class(smoother)
        prepared = cycle.prepare(hierarchy)

        again = cycle.prepare(prepared)

        assert smoother.prepared_for == [level.matrix.shape[0] for level in hierarchy.levels[:-1]]
        assert all(a is b for a, b in zip(again.levels, prepared.levels, strict=True))

    def test_pre_smoothing_starts_from_zero_and_post_smoothing_does_not(
        self, cycle_class: type[VCycle | WCycle], hierarchy: MultigridHierarchy, rhs: torch.Tensor
    ) -> None:
        smoother = RecordingSmoother()
        cycle = cycle_class(smoother, n_pre=N_PRE, n_post=N_POST)

        cycle.apply(cycle.prepare(hierarchy), rhs)

        assert smoother.sweep_log[0] == f"zero:{N_PRE}"
        assert f"smooth:{N_POST}" in smoother.sweep_log

    def test_unprepared_smoother_level_is_rejected(
        self, cycle_class: type[VCycle | WCycle], hierarchy: MultigridHierarchy, rhs: torch.Tensor
    ) -> None:
        cycle = cycle_class(RecordingSmoother())

        with pytest.raises(HierarchyNotPreparedError):
            cycle.apply(hierarchy, rhs)

    def test_residual_comes_from_the_prepared_smoother(
        self, cycle_class: type[VCycle | WCycle], hierarchy: MultigridHierarchy, rhs: torch.Tensor
    ) -> None:
        """The cycle forms each level's residual through the smoother's bound operator."""
        smoother = RecordingSmoother()
        cycle = cycle_class(smoother, n_pre=N_PRE, n_post=N_POST)

        cycle.apply(cycle.prepare(hierarchy), rhs)

        assert "residual" in smoother.sweep_log
