"""Unmasked Gauss-Seidel sweeps run on the registered vendor triangular solve."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest
import torch

from torchalg.sparse.kernels import triangular as triangular_module
from torchalg.sparse.kernels.triangular import (
    LevelSchedule,
    LibraryTriangularSolve,
    PreparedTriangularSolve,
)
from torchalg.sparse.preconditioners.amg._gauss_seidel import prepare_csr_gauss_seidel
from torchalg.sparse.preconditioners.amg.smoothers import GaussSeidelMemo, GaussSeidelSmoother

if TYPE_CHECKING:
    from tests.sparse.conftest import DenseSymmetricGaussSeidel

STEPS = (0, 1, 3)


@pytest.fixture
def scheduled_directions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Directions of every level schedule built while the test runs."""
    directions: list[str] = []
    real_schedule = triangular_module.level_schedule

    def spy(
        matrix: torch.Tensor, *, direction: Literal["forward", "backward"] = "forward"
    ) -> LevelSchedule:
        directions.append(direction)
        return real_schedule(matrix, direction=direction)

    monkeypatch.setattr(triangular_module, "level_schedule", spy)
    return directions


class TestVendorSweeps:
    def test_sweeps_use_the_registered_library_solve(self, poisson_2d_csr: torch.Tensor) -> None:
        prepared = prepare_csr_gauss_seidel(poisson_2d_csr)
        assert isinstance(prepared.forward, LibraryTriangularSolve)
        assert isinstance(prepared.backward, LibraryTriangularSolve)

    def test_zero_pivot_matrix_keeps_the_freezing_level_scheduled_solve(
        self, zero_diagonal_csr: torch.Tensor
    ) -> None:
        prepared = prepare_csr_gauss_seidel(zero_diagonal_csr)
        assert isinstance(prepared.forward, PreparedTriangularSolve)
        assert isinstance(prepared.backward, PreparedTriangularSolve)

    def test_unmasked_use_builds_no_level_schedule(
        self,
        poisson_2d_csr: torch.Tensor,
        gauss_seidel_block: tuple[torch.Tensor, ...],
        scheduled_directions: list[str],
    ) -> None:
        rhs, x, _ = gauss_seidel_block
        prepared = prepare_csr_gauss_seidel(poisson_2d_csr)
        prepared.smooth(rhs, x, 2)
        prepared.smooth_from_zero(rhs, 2)
        assert scheduled_directions == []

    def test_masked_sweeps_build_their_schedules_once_per_matrix(
        self,
        poisson_2d_csr: torch.Tensor,
        gauss_seidel_block: tuple[torch.Tensor, ...],
        scheduled_directions: list[str],
    ) -> None:
        rhs, x, rows = gauss_seidel_block
        prepared = prepare_csr_gauss_seidel(poisson_2d_csr)
        prepared.smooth(rhs, x, 1, rows=rows)
        prepared.smooth(rhs, x, 2, rows=rows)
        assert sorted(scheduled_directions) == ["backward", "forward"]


class TestMatchesDenseOracle:
    """Both the vendor and the frozen-pivot level-scheduled sweeps equal the dense algorithm."""

    @pytest.fixture(params=["poisson_2d_csr", "zero_diagonal_csr"])
    def matrix(self, request: pytest.FixtureRequest) -> torch.Tensor:
        return request.getfixturevalue(request.param)

    @pytest.mark.parametrize("steps", STEPS)
    def test_smooth(
        self,
        matrix: torch.Tensor,
        gauss_seidel_block: tuple[torch.Tensor, ...],
        dense_symmetric_gauss_seidel: DenseSymmetricGaussSeidel,
        steps: int,
    ) -> None:
        rhs, x, _ = gauss_seidel_block
        expected = dense_symmetric_gauss_seidel(matrix.to_dense(), x, rhs, steps)
        actual = prepare_csr_gauss_seidel(matrix).smooth(rhs, x, steps)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("steps", STEPS)
    def test_smooth_from_zero(
        self,
        matrix: torch.Tensor,
        gauss_seidel_block: tuple[torch.Tensor, ...],
        dense_symmetric_gauss_seidel: DenseSymmetricGaussSeidel,
        steps: int,
    ) -> None:
        rhs, _, _ = gauss_seidel_block
        expected = dense_symmetric_gauss_seidel(
            matrix.to_dense(), torch.zeros_like(rhs), rhs, steps
        )
        actual = prepare_csr_gauss_seidel(matrix).smooth_from_zero(rhs, steps)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_indexed_rows(
        self,
        matrix: torch.Tensor,
        gauss_seidel_block: tuple[torch.Tensor, ...],
        dense_symmetric_gauss_seidel: DenseSymmetricGaussSeidel,
    ) -> None:
        rhs, x, rows = gauss_seidel_block
        expected = dense_symmetric_gauss_seidel(matrix.to_dense(), x, rhs, 2, rows=rows)
        actual = prepare_csr_gauss_seidel(matrix).smooth(rhs, x, 2, rows=rows)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_vector_input_keeps_its_shape_and_iterate_is_not_modified(
        self, matrix: torch.Tensor, gauss_seidel_block: tuple[torch.Tensor, ...]
    ) -> None:
        rhs, x, _ = gauss_seidel_block
        before = x[:, 0].clone()
        out = prepare_csr_gauss_seidel(matrix).smooth(rhs[:, 0], x[:, 0], 1)
        assert out.shape == before.shape
        torch.testing.assert_close(x[:, 0], before, rtol=0.0, atol=0.0)


class TestSmootherPreparation:
    def test_smoother_and_stage_prepare_the_vendor_sweeps(
        self, poisson_2d_csr: torch.Tensor
    ) -> None:
        assert isinstance(
            GaussSeidelSmoother().prepare(poisson_2d_csr).forward, LibraryTriangularSolve
        )
        stage = GaussSeidelMemo()
        first = stage.prepare(poisson_2d_csr)
        assert isinstance(first.forward, LibraryTriangularSolve)
        assert stage.prepare(poisson_2d_csr) is first
