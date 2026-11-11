"""Tests for ``torchalg.sparse.preconditioners.amg._presets``.

Sparse-CSR sibling of
``preconditioners.implementations.amg._presets`` - same shared wiring
(setup cycle, solve-time cycle factory, seeded draw, placeholder
coarsening), built from this package's own sparse-native
``GaussSeidelStage``/``dense_pseudo_inverse_coarse_solver`` instead of the dense
versions.
"""

from __future__ import annotations

import torch

from torchalg.multigrid import VCycle
from torchalg.sparse.preconditioners.amg._presets import (
    PrebuiltCoarsening,
    gs_setup_cycle,
    prebuilt_cycle,
    seeded_draw,
)
from torchalg.sparse.preconditioners.amg.coarse_solve import dense_pseudo_inverse_coarse_solver
from torchalg.sparse.preconditioners.amg.smoothers import GaussSeidelSmoother, GaussSeidelStage


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
    def test_is_a_vcycle_smoothing_through_the_given_stage(self) -> None:
        stage = GaussSeidelStage()
        cycle = gs_setup_cycle(stage)
        assert isinstance(cycle, VCycle)
        assert cycle._smoother is stage
        assert cycle._n_pre == 1
        assert cycle._n_post == 1
        assert cycle._coarse_solver_factory is dense_pseudo_inverse_coarse_solver

    def test_cycles_of_distinct_stages_do_not_share_state(self) -> None:
        assert (
            gs_setup_cycle(GaussSeidelStage())._smoother
            is not gs_setup_cycle(GaussSeidelStage())._smoother
        )


class TestPrebuiltCycle:
    def test_wraps_smoother_in_vcycle(self) -> None:
        smoother = GaussSeidelSmoother()
        cycle = prebuilt_cycle(smoother, n_pre=2, n_post=3)
        assert isinstance(cycle, VCycle)
        assert cycle._smoother is smoother
        assert cycle._n_pre == 2
        assert cycle._n_post == 3
        assert cycle._coarse_solver_factory is dense_pseudo_inverse_coarse_solver


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
