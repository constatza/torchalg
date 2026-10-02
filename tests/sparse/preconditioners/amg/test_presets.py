"""Tests for ``torchalg.sparse.preconditioners.amg._presets``.

Sparse-CSR sibling of
``preconditioners.implementations.amg._presets`` - same shared wiring
(setup cycle, solve-time cycle factory, seeded draw, placeholder
coarsening), built from this package's own sparse-native
``GaussSeidelSmoother``/``dense_pseudo_inverse_solve`` instead of the dense
versions.
"""

from __future__ import annotations

import torch

from torchalg.multigrid import VCycle
from torchalg.sparse.preconditioners.amg._presets import (
    GS_SETUP_CYCLE,
    PrebuiltCoarsening,
    prebuilt_cycle,
    seeded_draw,
)
from torchalg.sparse.preconditioners.amg.coarse_solve import dense_pseudo_inverse_solve
from torchalg.sparse.preconditioners.amg.smoothers import GaussSeidelSmoother


class TestSeededDraw:
    def test_same_seed_is_deterministic(self) -> None:
        draw_a = seeded_draw(0)
        draw_b = seeded_draw(0)
        torch.testing.assert_close(draw_a(5), draw_b(5))

    def test_different_seeds_differ(self) -> None:
        draw_a = seeded_draw(0)
        draw_b = seeded_draw(1)
        assert not torch.allclose(draw_a(5), draw_b(5))


class TestGSSetupCycle:
    def test_is_a_vcycle_with_sparse_gauss_seidel(self) -> None:
        assert isinstance(GS_SETUP_CYCLE, VCycle)
        assert isinstance(GS_SETUP_CYCLE._smoother, GaussSeidelSmoother)
        assert GS_SETUP_CYCLE._coarse_solver is dense_pseudo_inverse_solve


class TestPrebuiltCycle:
    def test_wraps_smoother_in_vcycle(self) -> None:
        smoother = GaussSeidelSmoother()
        cycle = prebuilt_cycle(smoother, n_pre=2, n_post=3)
        assert isinstance(cycle, VCycle)
        assert cycle._smoother is smoother
        assert cycle._n_pre == 2
        assert cycle._n_post == 3
        assert cycle._coarse_solver is dense_pseudo_inverse_solve


class TestPrebuiltCoarsening:
    def test_build_transfer_always_raises(self) -> None:
        coarsening = PrebuiltCoarsening("test preset")
        matrix = torch.eye(3).to_sparse_csr()
        try:
            coarsening.build_transfer(matrix)
        except RuntimeError as error:
            assert "test preset" in str(error)
        else:
            raise AssertionError("expected RuntimeError")
