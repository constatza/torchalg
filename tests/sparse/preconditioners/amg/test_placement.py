"""Placement of sparse multigrid preconditioners follows their solve-time smoother."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from torchalg.multigrid import VCycle, WCycle
from torchalg.preconditioners.base import Placement
from torchalg.sparse.preconditioners.amg.adaptive import AdaptiveSAPreconditioner
from torchalg.sparse.preconditioners.amg.bootcmatch import BootCMatchPreconditioner
from torchalg.sparse.preconditioners.amg.bootstrap import BootstrapAMGPreconditioner
from torchalg.sparse.preconditioners.amg.smoothers import (
    GaussSeidelMemo,
    GaussSeidelSmoother,
    JacobiSmoother,
)
from torchalg.sparse.preconditioners.amg.variants import vcycle_amg, wcycle_amg

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchalg.multigrid.protocols import MultigridSmoother
    from torchalg.preconditioners.base import Preconditioner


@pytest.mark.parametrize(
    ("smoother", "expected"),
    [
        (JacobiSmoother(), Placement.SPMV_BOUND),
        (GaussSeidelSmoother(), Placement.SEQUENTIAL_SWEEP),
        (GaussSeidelMemo(), Placement.SEQUENTIAL_SWEEP),
    ],
    ids=["jacobi", "gauss_seidel", "gauss_seidel_memo"],
)
class TestSmootherPlacement:
    def test_smoother_declares_its_kind(
        self, smoother: MultigridSmoother, expected: Placement
    ) -> None:
        assert smoother.placement is expected

    @pytest.mark.parametrize("cycle_class", [VCycle, WCycle])
    def test_cycle_exposes_the_smoother_kind(
        self, cycle_class: type[VCycle | WCycle], smoother: MultigridSmoother, expected: Placement
    ) -> None:
        assert cycle_class(smoother).placement is expected


_PRESET_FACTORIES: dict[str, Callable[[MultigridSmoother | None], Preconditioner]] = {
    "vcycle_amg": lambda smoother: vcycle_amg(smoother=smoother),
    "wcycle_amg": lambda smoother: wcycle_amg(smoother=smoother),
    "adaptive_sa": lambda smoother: AdaptiveSAPreconditioner(smoother=smoother),
    "bootstrap": lambda smoother: BootstrapAMGPreconditioner(smoother=smoother),
    "bootcmatch": lambda smoother: BootCMatchPreconditioner(smoother=smoother),
}


@pytest.mark.parametrize("factory", _PRESET_FACTORIES.values(), ids=list(_PRESET_FACTORIES))
class TestPresetPlacement:
    def test_default_jacobi_is_spmv_bound(
        self, factory: Callable[[MultigridSmoother | None], Preconditioner]
    ) -> None:
        assert factory(None).placement is Placement.SPMV_BOUND

    def test_gauss_seidel_is_sequential(
        self, factory: Callable[[MultigridSmoother | None], Preconditioner]
    ) -> None:
        assert factory(GaussSeidelSmoother()).placement is Placement.SEQUENTIAL_SWEEP

    def test_instance_placement_does_not_leak_between_instances(
        self, factory: Callable[[MultigridSmoother | None], Preconditioner]
    ) -> None:
        sequential = factory(GaussSeidelSmoother())
        jacobi = factory(None)
        assert sequential.placement is Placement.SEQUENTIAL_SWEEP
        assert jacobi.placement is Placement.SPMV_BOUND
