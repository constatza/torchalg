"""Tests for ``VCycle``/``WCycle``, relocated alongside the promotion of
``cycle.py`` from ``torchalg.preconditioners.implementations.amg`` into
``torchalg.multigrid`` (see ``docs/plan.md``).

New here: ``TestWCycleCoarseSolver`` - regression coverage for the
``WCycle``/``VCycle`` coarse-solver parity gap found during the promotion
(``WCycle`` hardcoded ``torch.linalg.solve`` at the coarsest level instead
of accepting an injectable ``coarse_solver``, unlike ``VCycle``).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torchalg.multigrid.cycle import WCycle, pseudo_inverse_solve
from torchalg.multigrid.hierarchy import build_hierarchy
from torchalg.preconditioners.implementations.amg import AggregationCoarsening, JacobiSmoother


@pytest.fixture
def poisson_matrix(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    """20-node 1D Poisson matrix."""
    return poisson_1d_factory(20)


class TestWCycleCoarseSolver:
    def test_default_coarse_solver_is_torch_linalg_solve(self) -> None:
        """``WCycle``'s default must match ``VCycle``'s (``torch.linalg.solve``)."""
        cycle = WCycle(smoother=JacobiSmoother())
        assert cycle._coarse_solver is torch.linalg.solve

    def test_injected_coarse_solver_is_used(
        self, poisson_matrix: torch.Tensor, poisson_1d_factory: Callable[[int], torch.Tensor]
    ) -> None:
        """A custom ``coarse_solver`` must actually be called at the coarsest level."""
        calls: list[tuple[torch.Tensor, torch.Tensor]] = []

        def recording_solver(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            calls.append((matrix, rhs))
            return torch.linalg.solve(matrix, rhs)

        hierarchy = build_hierarchy(poisson_matrix, AggregationCoarsening(), n_levels=2)
        cycle = WCycle(smoother=JacobiSmoother(), coarse_solver=recording_solver)
        rhs = torch.ones(poisson_matrix.shape[0], dtype=poisson_matrix.dtype)

        cycle.apply(hierarchy, rhs)

        # gamma=2: the coarsest-level solve runs twice per W-cycle.
        assert len(calls) == 2

    def test_pseudo_inverse_solve_works_as_coarse_solver(
        self, poisson_matrix: torch.Tensor
    ) -> None:
        """``pseudo_inverse_solve`` (singular-safe) must be usable as ``WCycle``'s coarse solver.

        This is impossible before the fix: the coarsest-level solve was
        hardcoded to ``torch.linalg.solve``, which raises on a singular
        coarse matrix.
        """
        hierarchy = build_hierarchy(poisson_matrix, AggregationCoarsening(), n_levels=2)
        cycle = WCycle(smoother=JacobiSmoother(), coarse_solver=pseudo_inverse_solve)
        rhs = torch.ones(poisson_matrix.shape[0], dtype=poisson_matrix.dtype)

        result = cycle.apply(hierarchy, rhs)

        assert result.shape == rhs.shape
