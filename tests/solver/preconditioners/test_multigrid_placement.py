"""Placement of multigrid preconditioners follows their solve-time smoother (dense tree)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from torchalg.multigrid import VCycle, WCycle
from torchalg.preconditioners.base import Placement
from torchalg.preconditioners.implementations.amg import (
    AdaptiveSAPreconditioner,
    BootstrapAMGPreconditioner,
    GaussSeidelSmoother,
    JacobiSmoother,
    vcycle_amg,
    wcycle_amg,
)
from torchalg.preconditioners.implementations.amg.bootcmatch import BootCMatchPreconditioner
from torchalg.preconditioners.implementations.pod.variants import pod2g_preconditioner

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchalg.preconditioners.base import Preconditioner

POD_RANK = 2


@pytest.fixture
def snapshots() -> torch.Tensor:
    """Tiny snapshot ensemble for the POD preset."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(4, 8, generator=generator, dtype=torch.float64)


@pytest.mark.parametrize(
    ("smoother", "expected"),
    [
        (JacobiSmoother(), Placement.SPMV_BOUND),
        (GaussSeidelSmoother(), Placement.SEQUENTIAL_SWEEP),
    ],
    ids=["jacobi", "gauss_seidel"],
)
class TestSmootherPlacement:
    def test_smoother_declares_its_kind(
        self, smoother: JacobiSmoother | GaussSeidelSmoother, expected: Placement
    ) -> None:
        assert smoother.placement is expected

    @pytest.mark.parametrize("cycle_class", [VCycle, WCycle])
    def test_cycle_exposes_the_smoother_kind(
        self,
        cycle_class: type[VCycle | WCycle],
        smoother: JacobiSmoother | GaussSeidelSmoother,
        expected: Placement,
    ) -> None:
        assert cycle_class(smoother).placement is expected


# Each entry builds a preset around ``smoother``; ``None`` leaves the preset's default.
_PRESET_FACTORIES: dict[str, Callable[..., Preconditioner]] = {
    "vcycle_amg": lambda smoother, snapshots: vcycle_amg(smoother=smoother),
    "wcycle_amg": lambda smoother, snapshots: wcycle_amg(smoother=smoother),
    "pod2g": lambda smoother, snapshots: pod2g_preconditioner(
        snapshots, POD_RANK, smoother=smoother
    ),
    "adaptive_sa": lambda smoother, snapshots: AdaptiveSAPreconditioner(smoother=smoother),
    "bootstrap": lambda smoother, snapshots: BootstrapAMGPreconditioner(smoother=smoother),
    "bootcmatch": lambda smoother, snapshots: BootCMatchPreconditioner(smoother=smoother),
}


@pytest.mark.parametrize("factory", _PRESET_FACTORIES.values(), ids=list(_PRESET_FACTORIES))
class TestPresetPlacement:
    def test_default_jacobi_is_spmv_bound(
        self, factory: Callable[..., Preconditioner], snapshots: torch.Tensor
    ) -> None:
        assert factory(None, snapshots).placement is Placement.SPMV_BOUND

    def test_gauss_seidel_is_sequential(
        self, factory: Callable[..., Preconditioner], snapshots: torch.Tensor
    ) -> None:
        precond = factory(GaussSeidelSmoother(), snapshots)
        assert precond.placement is Placement.SEQUENTIAL_SWEEP

    def test_instance_placement_does_not_leak_to_the_class(
        self, factory: Callable[..., Preconditioner], snapshots: torch.Tensor
    ) -> None:
        sequential = factory(GaussSeidelSmoother(), snapshots)
        jacobi = factory(None, snapshots)
        assert sequential.placement is Placement.SEQUENTIAL_SWEEP
        assert jacobi.placement is Placement.SPMV_BOUND
