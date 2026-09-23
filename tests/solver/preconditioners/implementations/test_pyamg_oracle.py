"""Precision tests: torchalg's AMG pieces against PyAMG on identical operators.

Exact comparisons only (same ``A_l``/``P_l`` fed to both implementations), so
any disagreement is a bug in the cycle, the smoother, or the tentative
prolongator - not algorithmic slack. The adaptive-SA kernels and the complete
setup are checked in ``test_adaptive_sa_port.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pyamg
import pytest
import torch
from pyamg.relaxation.smoothing import change_smoothers

from tests.support.pyamg_reference import (
    as_torch_preconditioner,
    multilevel_from_hierarchy,
    replay_draw,
    to_csr,
)
from torchalg import pcg
from torchalg.preconditioners.implementations.amg import (
    AggregationCoarsening,
    JacobiSmoother,
    MultigridSmoother,
    VCycle,
    WCycle,
)
from torchalg.preconditioners.implementations.amg.adaptive import (
    AdaptiveSAPreconditioner,
)
from torchalg.preconditioners.implementations.amg.hierarchy import build_hierarchy

OMEGA = 0.67
N_PRE = 2
N_POST = 2


@pytest.fixture
def aniso_matrix(anisotropic_2d_factory: Callable[[int, float], torch.Tensor]) -> torch.Tensor:
    """12x12 anisotropic (eps=0.05) 2D Poisson matrix, 144 unknowns."""
    return anisotropic_2d_factory(12, 0.05)


@pytest.fixture
def rhs(aniso_matrix: torch.Tensor, torch_dtype: torch.dtype) -> torch.Tensor:
    """Seeded random right-hand side."""
    generator = torch.Generator().manual_seed(12)
    return torch.randn(aniso_matrix.shape[0], generator=generator, dtype=torch_dtype)


@pytest.fixture(params=[("V", VCycle), ("W", WCycle)], ids=["V-cycle", "W-cycle"])
def cycle_pair(request: pytest.FixtureRequest) -> tuple[str, type]:
    """PyAMG cycle name paired with the matching torchalg cycle class."""
    return request.param


class TestCycleAgainstPyAMG:
    def test_single_cycle_matches_on_sa_hierarchy(
        self,
        aniso_matrix: torch.Tensor,
        rhs: torch.Tensor,
        cycle_pair: tuple[str, type],
        torch_dtype: torch.dtype,
    ) -> None:
        name, cycle_class = cycle_pair
        hierarchy = build_hierarchy(aniso_matrix, AggregationCoarsening(), 3)
        ours = cycle_class(JacobiSmoother(OMEGA), n_pre=N_PRE, n_post=N_POST).apply(hierarchy, rhs)
        oracle = as_torch_preconditioner(
            multilevel_from_hierarchy(hierarchy, OMEGA, N_PRE, N_POST), name, torch_dtype
        )
        torch.testing.assert_close(ours, oracle(rhs), rtol=1e-10, atol=1e-10)


class TestResidualHistoryAgainstPyAMG:
    def test_adaptive_sa_pcg_residual_history_matches_pyamg(
        self, aniso_matrix: torch.Tensor, rhs: torch.Tensor, torch_dtype: torch.dtype
    ) -> None:
        options: dict[str, Any] = {"num_candidates": 3, "max_levels": 3, "max_coarse": 5}
        np.random.seed(41)
        reference, _ = pyamg.aggregation.adaptive_sa_solver(
            to_csr(aniso_matrix), strength=("symmetric", {"theta": 0.25}), **options
        )
        smoother = ("jacobi", {"omega": OMEGA, "iterations": 1, "withrho": False})
        change_smoothers(reference, smoother, smoother)
        ours = AdaptiveSAPreconditioner(
            aniso_matrix,
            theta=0.25,
            draw=replay_draw(41),
            smoother_omega=OMEGA,
            **options,
        )
        oracle = as_torch_preconditioner(reference, "V", torch_dtype)
        _, ours_info = pcg(aniso_matrix, rhs, preconditioner=ours, rtol=1e-10, trace_mode="minimal")
        _, oracle_info = pcg(
            aniso_matrix, rhs, preconditioner=oracle, rtol=1e-10, trace_mode="minimal"
        )
        assert ours_info.converged and oracle_info.converged
        assert ours_info.residual_history_abs is not None
        assert oracle_info.residual_history_abs is not None
        assert len(ours_info.residual_history_abs) == len(oracle_info.residual_history_abs)
        np.testing.assert_allclose(
            ours_info.residual_history_abs, oracle_info.residual_history_abs, rtol=1e-6
        )

    def test_adaptive_sa_uses_configured_solve_smoother(
        self,
        aniso_matrix: torch.Tensor,
        rhs: torch.Tensor,
        raising_smoother: MultigridSmoother,
    ) -> None:
        """Adaptive setup remains GS-faithful while its solve-time smoother is injectable."""
        preconditioner = AdaptiveSAPreconditioner(
            aniso_matrix,
            num_candidates=1,
            max_levels=2,
            max_coarse=5,
            smoother=raising_smoother,
        )
        with pytest.raises(RuntimeError, match="configured smoother used"):
            preconditioner.apply(rhs)

    def test_adaptive_sa_defaults_to_jacobi_at_solve_time(
        self,
        aniso_matrix: torch.Tensor,
    ) -> None:
        """The quick solve path defaults to Jacobi without changing GS-based setup."""
        preconditioner = AdaptiveSAPreconditioner(
            aniso_matrix,
            num_candidates=1,
            max_levels=2,
            max_coarse=5,
        )
        assert isinstance(
            preconditioner._cycle._smoother,  # ty: ignore[unresolved-attribute]
            JacobiSmoother,
        )
