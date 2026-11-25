"""Dense-backed symmetric Gauss-Seidel equals the level-scheduled CSR sweep, and the format policy."""

from __future__ import annotations

import gc
import weakref
from typing import Literal

import pytest
import torch

from torchalg.sparse.kernels import triangular as triangular_module
from torchalg.sparse.kernels.triangular import LevelSchedule
from torchalg.sparse.preconditioners.amg import _gauss_seidel_format as policy
from torchalg.sparse.preconditioners.amg._dense_gauss_seidel import (
    PreparedDenseGaussSeidel,
    prepare_dense_gauss_seidel,
)
from torchalg.sparse.preconditioners.amg._gauss_seidel import (
    PreparedSymmetricGaussSeidel,
    prepare_csr_gauss_seidel,
)
from torchalg.sparse.preconditioners.amg.smoothers import GaussSeidelMemo, GaussSeidelSmoother

STEPS = (0, 1, 3)


@pytest.fixture(params=["poisson_2d_csr", "zero_diagonal_csr"])
def matrix(request: pytest.FixtureRequest) -> torch.Tensor:
    return request.getfixturevalue(request.param)


class TestPreparedDenseEqualsCsr:
    """Same smoothing result as the CSR path for every entry point."""

    @pytest.mark.parametrize("steps", STEPS)
    def test_smooth(
        self, matrix: torch.Tensor, gauss_seidel_block: tuple[torch.Tensor, ...], steps: int
    ) -> None:
        rhs, x, _ = gauss_seidel_block
        expected = prepare_csr_gauss_seidel(matrix).smooth(rhs, x, steps)
        actual = prepare_dense_gauss_seidel(matrix).smooth(rhs, x, steps)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("steps", STEPS)
    def test_smooth_from_zero(
        self, matrix: torch.Tensor, gauss_seidel_block: tuple[torch.Tensor, ...], steps: int
    ) -> None:
        rhs, _, _ = gauss_seidel_block
        expected = prepare_csr_gauss_seidel(matrix).smooth_from_zero(rhs, steps)
        actual = prepare_dense_gauss_seidel(matrix).smooth_from_zero(rhs, steps)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_indexed_rows(
        self, matrix: torch.Tensor, gauss_seidel_block: tuple[torch.Tensor, ...]
    ) -> None:
        rhs, x, rows = gauss_seidel_block
        expected = prepare_csr_gauss_seidel(matrix).smooth(rhs, x, 2, rows=rows)
        actual = prepare_dense_gauss_seidel(matrix).smooth(rhs, x, 2, rows=rows)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_vector_input_keeps_its_shape(
        self, matrix: torch.Tensor, gauss_seidel_block: tuple[torch.Tensor, ...]
    ) -> None:
        rhs, x, _ = gauss_seidel_block
        out = prepare_dense_gauss_seidel(matrix).smooth(rhs[:, 0], x[:, 0], 1)
        assert out.shape == x[:, 0].shape

    def test_input_iterate_is_not_modified(
        self, matrix: torch.Tensor, gauss_seidel_block: tuple[torch.Tensor, ...]
    ) -> None:
        rhs, x, _ = gauss_seidel_block
        before = x.clone()
        prepare_dense_gauss_seidel(matrix).smooth(rhs, x, 2)
        torch.testing.assert_close(x, before, rtol=0.0, atol=0.0)


class TestFormatPolicy:
    """Dense below the band, CSR above it, a cost comparison inside it."""

    def test_small_matrix_is_dense_backed(self, poisson_2d_csr: torch.Tensor) -> None:
        assert isinstance(policy.prepare_gauss_seidel(poisson_2d_csr), PreparedDenseGaussSeidel)

    def test_large_matrix_is_csr(
        self, poisson_2d_csr: torch.Tensor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(policy, "DENSE_ALWAYS_MAX_N", 2)
        monkeypatch.setattr(policy, "CSR_ALWAYS_MIN_N", 3)
        assert isinstance(policy.prepare_gauss_seidel(poisson_2d_csr), PreparedSymmetricGaussSeidel)

    @pytest.mark.parametrize(("ms_per_level", "dense"), [(0.06, False), (1e4, True)])
    def test_band_compares_dense_cost_with_level_launch_cost(
        self,
        poisson_2d_csr: torch.Tensor,
        monkeypatch: pytest.MonkeyPatch,
        ms_per_level: float,
        dense: bool,
    ) -> None:
        monkeypatch.setattr(policy, "DENSE_ALWAYS_MAX_N", 2)
        monkeypatch.setattr(policy, "CSR_ALWAYS_MIN_N", 10**6)
        monkeypatch.setattr(policy, "DENSE_MS_PER_N_SQUARED", 1.0)  # dense cost 81 ms at n = 9
        monkeypatch.setattr(policy, "CSR_MS_PER_LEVEL", ms_per_level)
        expected = PreparedDenseGaussSeidel if dense else PreparedSymmetricGaussSeidel
        assert isinstance(policy.prepare_gauss_seidel(poisson_2d_csr), expected)

    def test_band_computes_each_level_schedule_once(
        self, poisson_2d_csr: torch.Tensor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The forward schedule that sizes the cost estimate is the one the CSR sweep uses."""
        monkeypatch.setattr(policy, "DENSE_ALWAYS_MAX_N", 2)
        monkeypatch.setattr(policy, "CSR_ALWAYS_MIN_N", 10**6)
        monkeypatch.setattr(policy, "DENSE_MS_PER_N_SQUARED", 1.0)  # dense loses: CSR is built
        directions: list[str] = []
        real_schedule = triangular_module.level_schedule

        def spy(
            matrix: torch.Tensor, *, direction: Literal["forward", "backward"] = "forward"
        ) -> LevelSchedule:
            directions.append(direction)
            return real_schedule(matrix, direction=direction)

        monkeypatch.setattr(policy, "level_schedule", spy)
        monkeypatch.setattr(triangular_module, "level_schedule", spy)

        prepared = policy.prepare_gauss_seidel(poisson_2d_csr)

        assert isinstance(prepared, PreparedSymmetricGaussSeidel)
        assert sorted(directions) == ["backward", "forward"]

    def test_schedule_passed_through_gives_the_same_sweeps(
        self, poisson_2d_csr: torch.Tensor, gauss_seidel_block: tuple[torch.Tensor, ...]
    ) -> None:
        rhs, x, _ = gauss_seidel_block
        schedule = triangular_module.level_schedule(poisson_2d_csr, direction="forward")
        reused = prepare_csr_gauss_seidel(poisson_2d_csr, forward_schedule=schedule)
        fresh = prepare_csr_gauss_seidel(poisson_2d_csr)
        assert torch.equal(reused.smooth(rhs, x, 2), fresh.smooth(rhs, x, 2))

    def test_smoother_and_stage_use_the_policy(self, poisson_2d_csr: torch.Tensor) -> None:
        assert isinstance(GaussSeidelSmoother().prepare(poisson_2d_csr), PreparedDenseGaussSeidel)
        stage = GaussSeidelMemo()
        first = stage.prepare(poisson_2d_csr)
        assert isinstance(first, PreparedDenseGaussSeidel)
        assert stage.prepare(poisson_2d_csr) is first


class TestStageOwnsItsMatrices:
    """The id-keyed stage cache must keep each matrix alive, whichever format it chose."""

    def test_stage_keeps_the_matrix_alive(self, poisson_2d_dense: torch.Tensor) -> None:
        matrix = poisson_2d_dense.to_sparse_csr()
        reference = weakref.ref(matrix)
        stage = GaussSeidelMemo()
        stage.prepare(matrix)
        del matrix
        gc.collect()
        assert reference() is not None
        assert stage  # the stage, not the caller, now owns the matrix
