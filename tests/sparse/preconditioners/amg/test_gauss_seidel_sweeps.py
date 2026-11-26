"""Gauss-Seidel sweeps run on the registered vendor triangular solve.

Both ``SweepProducts`` kinds are covered: the solve-time smoother takes its strict-triangle
products from the solves, setup stages apply them explicitly (bit-identical setups).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest
import torch

from tests.support.dispatch_counter import DispatchCounter
from torchalg.sparse.kernels import triangular as triangular_module
from torchalg.sparse.kernels.triangular import (
    LevelSchedule,
    LibraryTriangularSolve,
    PreparedTriangularSolve,
    TriangularSolver,
)
from torchalg.sparse.preconditioners.amg import _gauss_seidel as gauss_seidel_module
from torchalg.sparse.preconditioners.amg._gauss_seidel import (
    SweepProducts,
    prepare_csr_gauss_seidel,
)
from torchalg.sparse.preconditioners.amg.smoothers import GaussSeidelMemo, GaussSeidelSmoother

if TYPE_CHECKING:
    from tests.sparse.conftest import DenseSymmetricGaussSeidel

STEPS = (0, 1, 3)
PRODUCT_STEPS = 3
FROM_SOLVES = SweepProducts.FROM_SOLVES
EXPLICIT = SweepProducts.EXPLICIT


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
        prepared = prepare_csr_gauss_seidel(poisson_2d_csr, FROM_SOLVES)
        assert isinstance(prepared.forward, LibraryTriangularSolve)
        assert isinstance(prepared.backward, LibraryTriangularSolve)

    def test_zero_pivot_matrix_keeps_the_freezing_level_scheduled_solve(
        self, zero_diagonal_csr: torch.Tensor
    ) -> None:
        prepared = prepare_csr_gauss_seidel(zero_diagonal_csr, FROM_SOLVES)
        assert isinstance(prepared.forward, PreparedTriangularSolve)
        assert isinstance(prepared.backward, PreparedTriangularSolve)

    @pytest.mark.parametrize("products", list(SweepProducts))
    def test_unmasked_use_builds_no_level_schedule(
        self,
        poisson_2d_csr: torch.Tensor,
        gauss_seidel_block: tuple[torch.Tensor, ...],
        scheduled_directions: list[str],
        products: SweepProducts,
    ) -> None:
        rhs, x, _ = gauss_seidel_block
        prepared = prepare_csr_gauss_seidel(poisson_2d_csr, products)
        prepared.smooth(rhs, x, 2)
        prepared.smooth_from_zero(rhs, 2)
        assert scheduled_directions == []

    @pytest.mark.parametrize("matrix_name", ["poisson_2d_csr", "zero_diagonal_csr"])
    def test_masked_sweeps_build_no_level_schedule(
        self,
        matrix_name: str,
        request: pytest.FixtureRequest,
        gauss_seidel_block: tuple[torch.Tensor, ...],
        scheduled_directions: list[str],
    ) -> None:
        rhs, x, rows = gauss_seidel_block
        prepared = prepare_csr_gauss_seidel(request.getfixturevalue(matrix_name), EXPLICIT)
        built_by_prepare = len(scheduled_directions)
        prepared.smooth(rhs, x, 2, rows=rows)
        assert len(scheduled_directions) == built_by_prepare

    @pytest.mark.parametrize("matrix_name", ["poisson_2d_csr", "zero_diagonal_csr"])
    def test_masked_sweeps_prepare_no_triangular_solve(
        self,
        matrix_name: str,
        request: pytest.FixtureRequest,
        gauss_seidel_block: tuple[torch.Tensor, ...],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A masked call derives its identity-row solves from the prepared ones."""
        rhs, x, rows = gauss_seidel_block
        prepared = prepare_csr_gauss_seidel(request.getfixturevalue(matrix_name), EXPLICIT)
        builds: list[str] = []
        real_prepare = gauss_seidel_module.prepare_triangular_solve

        def spy(
            matrix: torch.Tensor, direction: Literal["forward", "backward"] = "forward"
        ) -> TriangularSolver:
            builds.append(direction)
            return real_prepare(matrix, direction)

        monkeypatch.setattr(gauss_seidel_module, "prepare_triangular_solve", spy)
        prepared.smooth(rhs, x, 2, rows=rows)
        assert builds == []

    def test_triangles_carry_int32_indices(self, poisson_2d_csr: torch.Tensor) -> None:
        assert poisson_2d_csr.crow_indices().dtype == torch.int64
        prepared = prepare_csr_gauss_seidel(poisson_2d_csr, FROM_SOLVES)
        assert isinstance(prepared.forward, LibraryTriangularSolve)
        assert isinstance(prepared.backward, LibraryTriangularSolve)
        for triangle in (prepared.forward.triangle, prepared.backward.triangle):
            assert triangle.crow_indices().dtype == torch.int32
            assert triangle.col_indices().dtype == torch.int32


class TestStrictTriangleProducts:
    """``FROM_SOLVES`` reads the strict products off the solves; ``EXPLICIT`` applies them."""

    @pytest.mark.parametrize(
        ("products", "expected"), [(FROM_SOLVES, 1), (EXPLICIT, 2 * PRODUCT_STEPS)]
    )
    def test_smooth_strict_products(
        self,
        poisson_2d_csr: torch.Tensor,
        gauss_seidel_block: tuple[torch.Tensor, ...],
        products: SweepProducts,
        expected: int,
    ) -> None:
        rhs, x, _ = gauss_seidel_block
        prepared = prepare_csr_gauss_seidel(poisson_2d_csr, products)
        with DispatchCounter() as counter:
            prepared.smooth(rhs[:, 0], x[:, 0], PRODUCT_STEPS)
        assert counter.ops["aten.mv.default"] == expected

    @pytest.mark.parametrize(
        ("products", "expected"), [(FROM_SOLVES, 0), (EXPLICIT, 2 * PRODUCT_STEPS - 1)]
    )
    def test_smooth_from_zero_strict_products(
        self,
        poisson_2d_csr: torch.Tensor,
        gauss_seidel_block: tuple[torch.Tensor, ...],
        products: SweepProducts,
        expected: int,
    ) -> None:
        rhs, _, _ = gauss_seidel_block
        prepared = prepare_csr_gauss_seidel(poisson_2d_csr, products)
        with DispatchCounter() as counter:
            prepared.smooth_from_zero(rhs[:, 0], PRODUCT_STEPS)
        assert counter.ops["aten.mv.default"] == expected

    def test_zero_pivots_force_explicit_products(
        self, zero_diagonal_csr: torch.Tensor, gauss_seidel_block: tuple[torch.Tensor, ...]
    ) -> None:
        rhs, x, _ = gauss_seidel_block
        prepared = prepare_csr_gauss_seidel(zero_diagonal_csr, FROM_SOLVES)
        with DispatchCounter() as counter:
            prepared.smooth(rhs[:, 0], x[:, 0], PRODUCT_STEPS)
        assert counter.ops["aten.mv.default"] == 2 * PRODUCT_STEPS


class TestMatchesDenseOracle:
    """Both sweep kinds, vendor and frozen-pivot level-scheduled, equal the dense algorithm."""

    @pytest.fixture(params=["poisson_2d_csr", "zero_diagonal_csr"])
    def matrix(self, request: pytest.FixtureRequest) -> torch.Tensor:
        return request.getfixturevalue(request.param)

    @pytest.fixture(params=list(SweepProducts))
    def products(self, request: pytest.FixtureRequest) -> SweepProducts:
        return request.param

    @pytest.mark.parametrize("steps", STEPS)
    def test_smooth(
        self,
        matrix: torch.Tensor,
        products: SweepProducts,
        gauss_seidel_block: tuple[torch.Tensor, ...],
        dense_symmetric_gauss_seidel: DenseSymmetricGaussSeidel,
        steps: int,
    ) -> None:
        rhs, x, _ = gauss_seidel_block
        expected = dense_symmetric_gauss_seidel(matrix.to_dense(), x, rhs, steps)
        actual = prepare_csr_gauss_seidel(matrix, products).smooth(rhs, x, steps)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("steps", STEPS)
    def test_smooth_from_zero(
        self,
        matrix: torch.Tensor,
        products: SweepProducts,
        gauss_seidel_block: tuple[torch.Tensor, ...],
        dense_symmetric_gauss_seidel: DenseSymmetricGaussSeidel,
        steps: int,
    ) -> None:
        rhs, _, _ = gauss_seidel_block
        expected = dense_symmetric_gauss_seidel(
            matrix.to_dense(), torch.zeros_like(rhs), rhs, steps
        )
        actual = prepare_csr_gauss_seidel(matrix, products).smooth_from_zero(rhs, steps)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_indexed_rows(
        self,
        matrix: torch.Tensor,
        products: SweepProducts,
        gauss_seidel_block: tuple[torch.Tensor, ...],
        dense_symmetric_gauss_seidel: DenseSymmetricGaussSeidel,
    ) -> None:
        rhs, x, rows = gauss_seidel_block
        expected = dense_symmetric_gauss_seidel(matrix.to_dense(), x, rhs, 2, rows=rows)
        actual = prepare_csr_gauss_seidel(matrix, products).smooth(rhs, x, 2, rows=rows)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_indexed_rows_leave_frozen_rows_bit_equal(
        self,
        matrix: torch.Tensor,
        products: SweepProducts,
        gauss_seidel_block: tuple[torch.Tensor, ...],
    ) -> None:
        rhs, x, rows = gauss_seidel_block
        frozen = torch.ones(matrix.shape[0], dtype=torch.bool)
        frozen[rows] = False
        actual = prepare_csr_gauss_seidel(matrix, products).smooth(rhs, x, 2, rows=rows)
        torch.testing.assert_close(actual[frozen], x[frozen], rtol=0.0, atol=0.0)

    def test_vector_input_keeps_its_shape_and_iterate_is_not_modified(
        self,
        matrix: torch.Tensor,
        products: SweepProducts,
        gauss_seidel_block: tuple[torch.Tensor, ...],
    ) -> None:
        rhs, x, _ = gauss_seidel_block
        before = x[:, 0].clone()
        out = prepare_csr_gauss_seidel(matrix, products).smooth(rhs[:, 0], x[:, 0], 1)
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

    @pytest.mark.parametrize(
        ("make", "expected"),
        [(GaussSeidelSmoother, 1), (GaussSeidelMemo, 2 * PRODUCT_STEPS)],
        ids=["solve_time_from_solves", "setup_stage_explicit"],
    )
    def test_strict_products_by_preparer(
        self,
        poisson_2d_csr: torch.Tensor,
        gauss_seidel_block: tuple[torch.Tensor, ...],
        make: type[GaussSeidelSmoother | GaussSeidelMemo],
        expected: int,
    ) -> None:
        rhs, x, _ = gauss_seidel_block
        prepared = make().prepare(poisson_2d_csr)
        with DispatchCounter() as counter:
            prepared.smooth(rhs[:, 0], x[:, 0], PRODUCT_STEPS)
        assert counter.ops["aten.mv.default"] == expected
