"""Sparse-CSR tests for ``torchalg.sparse.preconditioners.amg.smoothers``.

Each sparse smoother is prepared once per matrix (``prepare``) and then swept
(``smooth``/``smooth_from_zero``). Results are checked against dense
first-principles oracles (``tests/sparse/conftest.py``), not against the
dense smoother classes, so the oracle stays independent of that API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from torchalg.sparse.kernels import triangular as triangular_kernel
from torchalg.sparse.preconditioners.amg import smoothers
from torchalg.sparse.preconditioners.amg._jacobi_omega import (
    RELAXATION_NOMINAL,
    jacobi_omega,
)
from torchalg.sparse.preconditioners.amg.smoothers import (
    GaussSeidelSmoother,
    JacobiSmoother,
    resolve_jacobi_default,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.sparse.conftest import DenseJacobiSweeps, DenseSymmetricGaussSeidel

_NUM_COLUMNS = 3
_FIXED_OMEGA = 0.6


@pytest.fixture
def poisson_2d(anisotropic_2d_factory: Callable[[int, float], torch.Tensor]) -> torch.Tensor:
    """Dense 9x9 2D isotropic Poisson matrix (genuine multi-row dependency levels)."""
    return anisotropic_2d_factory(3, 1.0)


@pytest.fixture(params=["poisson_1d_dense", "poisson_2d"])
def dense_matrix(request: pytest.FixtureRequest) -> torch.Tensor:
    """Each dense test matrix: tridiagonal (one row per level) and 2D (wide levels)."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def csr_matrix(dense_matrix: torch.Tensor) -> torch.Tensor:
    """Sparse CSR twin of ``dense_matrix``."""
    return dense_matrix.to_sparse_csr()


@pytest.fixture
def x0(dense_matrix: torch.Tensor) -> torch.Tensor:
    """Nonzero initial iterate."""
    n = dense_matrix.shape[0]
    return torch.arange(1, n + 1, dtype=dense_matrix.dtype) * 0.1


@pytest.fixture
def rhs(dense_matrix: torch.Tensor) -> torch.Tensor:
    """Non-constant right-hand side."""
    n = dense_matrix.shape[0]
    return torch.cos(torch.arange(n, dtype=dense_matrix.dtype)) + 2.0


@pytest.fixture
def x0_block(dense_matrix: torch.Tensor, test_seed: int) -> torch.Tensor:
    """Block of ``_NUM_COLUMNS`` initial iterates."""
    generator = torch.Generator().manual_seed(test_seed)
    return torch.randn(
        dense_matrix.shape[0], _NUM_COLUMNS, dtype=dense_matrix.dtype, generator=generator
    )


@pytest.fixture
def rhs_block(dense_matrix: torch.Tensor, test_seed: int) -> torch.Tensor:
    """Block of ``_NUM_COLUMNS`` right-hand sides."""
    generator = torch.Generator().manual_seed(test_seed + 1)
    return torch.randn(
        dense_matrix.shape[0], _NUM_COLUMNS, dtype=dense_matrix.dtype, generator=generator
    )


@pytest.fixture
def count_level_schedules(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counter list; ``len`` is the number of ``level_schedule`` calls made by Gauss-Seidel."""
    calls: list[int] = []
    original = triangular_kernel.level_schedule

    def counting(*args: Any, **kwargs: Any):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(triangular_kernel, "level_schedule", counting)
    return calls


class TestPreparedJacobi:
    """``JacobiSmoother.prepare`` returns a Jacobi bound to one matrix."""

    @pytest.mark.parametrize("steps", [1, 3])
    def test_matches_dense_oracle(
        self,
        dense_matrix: torch.Tensor,
        csr_matrix: torch.Tensor,
        x0: torch.Tensor,
        rhs: torch.Tensor,
        dense_jacobi_sweeps: DenseJacobiSweeps,
        steps: int,
    ) -> None:
        expected = dense_jacobi_sweeps(dense_matrix, rhs, x0, steps)
        actual = JacobiSmoother().prepare(csr_matrix).smooth(rhs, x0, steps)
        torch.testing.assert_close(actual, expected)

    def test_explicit_omega_matches_dense_oracle(
        self,
        dense_matrix: torch.Tensor,
        csr_matrix: torch.Tensor,
        x0: torch.Tensor,
        rhs: torch.Tensor,
        dense_jacobi_sweeps: DenseJacobiSweeps,
    ) -> None:
        expected = dense_jacobi_sweeps(dense_matrix, rhs, x0, 2, omega=_FIXED_OMEGA)
        actual = JacobiSmoother(omega=_FIXED_OMEGA).prepare(csr_matrix).smooth(rhs, x0, 2)
        torch.testing.assert_close(actual, expected)

    def test_omega_is_resolved_in_prepare(
        self, csr_matrix: torch.Tensor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spectral estimation happens in ``prepare``; sweeps never estimate again."""
        prepared = JacobiSmoother().prepare(csr_matrix)

        def forbidden(*_: object, **__: object) -> None:
            raise AssertionError("spectral radius estimated after prepare")

        monkeypatch.setattr(
            "torchalg.sparse.preconditioners.amg._jacobi_omega.approximate_spectral_radius",
            forbidden,
        )
        rhs_ = torch.ones(csr_matrix.shape[0], dtype=csr_matrix.dtype)
        prepared.smooth(rhs_, torch.zeros_like(rhs_), 2)
        prepared.smooth_from_zero(rhs_, 2)

    def test_diag_inv_matches_resolved_omega(self, csr_matrix: torch.Tensor) -> None:
        prepared = JacobiSmoother().prepare(csr_matrix)
        omega = jacobi_omega(csr_matrix, RELAXATION_NOMINAL, None)
        expected = omega / torch.diagonal(csr_matrix.to_dense())
        torch.testing.assert_close(prepared.diag_inv, expected)

    def test_diagonal_is_extracted_once(
        self,
        csr_matrix: torch.Tensor,
        rhs: torch.Tensor,
        x0: torch.Tensor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[int] = []
        original = smoothers.sparse_diagonal

        def counting(matrix: torch.Tensor) -> torch.Tensor:
            calls.append(1)
            return original(matrix)

        monkeypatch.setattr(smoothers, "sparse_diagonal", counting)
        prepared = JacobiSmoother(omega=_FIXED_OMEGA).prepare(csr_matrix)
        prepared.smooth(rhs, x0, 5)
        prepared.smooth_from_zero(rhs, 5)
        assert len(calls) == 1

    @pytest.mark.parametrize("steps", [1, 4])
    def test_smooth_from_zero_equals_smooth_of_zeros(
        self, csr_matrix: torch.Tensor, rhs: torch.Tensor, steps: int
    ) -> None:
        prepared = JacobiSmoother().prepare(csr_matrix)
        torch.testing.assert_close(
            prepared.smooth_from_zero(rhs, steps),
            prepared.smooth(rhs, torch.zeros_like(rhs), steps),
            rtol=1e-13,
            atol=1e-13,
        )

    def test_zero_steps_from_zero_is_zero(
        self, csr_matrix: torch.Tensor, rhs: torch.Tensor
    ) -> None:
        prepared = JacobiSmoother().prepare(csr_matrix)
        torch.testing.assert_close(prepared.smooth_from_zero(rhs, 0), torch.zeros_like(rhs))

    def test_block_equals_per_column(
        self, csr_matrix: torch.Tensor, rhs_block: torch.Tensor, x0_block: torch.Tensor
    ) -> None:
        prepared = JacobiSmoother().prepare(csr_matrix)
        block = prepared.smooth(rhs_block, x0_block, 3)
        from_zero = prepared.smooth_from_zero(rhs_block, 3)
        for column in range(_NUM_COLUMNS):
            torch.testing.assert_close(
                block[:, column], prepared.smooth(rhs_block[:, column], x0_block[:, column], 3)
            )
            torch.testing.assert_close(
                from_zero[:, column], prepared.smooth_from_zero(rhs_block[:, column], 3)
            )

    def test_input_iterate_is_not_modified(
        self, csr_matrix: torch.Tensor, rhs: torch.Tensor, x0: torch.Tensor
    ) -> None:
        before = x0.clone()
        JacobiSmoother().prepare(csr_matrix).smooth(rhs, x0, 2)
        torch.testing.assert_close(x0, before, rtol=0, atol=0)

    def test_near_zero_diagonal_row_takes_no_step(self, poisson_1d_dense: torch.Tensor) -> None:
        matrix = poisson_1d_dense.clone()
        matrix[0, 0] = 0.0
        prepared = JacobiSmoother(omega=_FIXED_OMEGA).prepare(matrix.to_sparse_csr())
        assert prepared.diag_inv[0] == 0.0


class TestPreparedGaussSeidel:
    """``GaussSeidelSmoother.prepare`` returns a symmetric GS bound to one matrix."""

    @pytest.mark.parametrize("steps", [1, 3])
    def test_matches_dense_oracle(
        self,
        dense_matrix: torch.Tensor,
        csr_matrix: torch.Tensor,
        x0: torch.Tensor,
        rhs: torch.Tensor,
        dense_symmetric_gauss_seidel: DenseSymmetricGaussSeidel,
        steps: int,
    ) -> None:
        expected = dense_symmetric_gauss_seidel(dense_matrix, x0, rhs, steps)
        actual = GaussSeidelSmoother().prepare(csr_matrix).smooth(rhs, x0, steps)
        torch.testing.assert_close(actual, expected)

    def test_block_matches_dense_oracle(
        self,
        dense_matrix: torch.Tensor,
        csr_matrix: torch.Tensor,
        x0_block: torch.Tensor,
        rhs_block: torch.Tensor,
        dense_symmetric_gauss_seidel: DenseSymmetricGaussSeidel,
    ) -> None:
        expected = dense_symmetric_gauss_seidel(dense_matrix, x0_block, rhs_block, 2)
        actual = GaussSeidelSmoother().prepare(csr_matrix).smooth(rhs_block, x0_block, 2)
        torch.testing.assert_close(actual, expected)

    def test_block_equals_per_column(
        self, csr_matrix: torch.Tensor, rhs_block: torch.Tensor, x0_block: torch.Tensor
    ) -> None:
        prepared = GaussSeidelSmoother().prepare(csr_matrix)
        block = prepared.smooth(rhs_block, x0_block, 2)
        from_zero = prepared.smooth_from_zero(rhs_block, 2)
        for column in range(_NUM_COLUMNS):
            torch.testing.assert_close(
                block[:, column], prepared.smooth(rhs_block[:, column], x0_block[:, column], 2)
            )
            torch.testing.assert_close(
                from_zero[:, column], prepared.smooth_from_zero(rhs_block[:, column], 2)
            )

    @pytest.mark.parametrize("steps", [1, 3])
    def test_smooth_from_zero_equals_smooth_of_zeros(
        self, csr_matrix: torch.Tensor, rhs: torch.Tensor, steps: int
    ) -> None:
        prepared = GaussSeidelSmoother().prepare(csr_matrix)
        torch.testing.assert_close(
            prepared.smooth_from_zero(rhs, steps),
            prepared.smooth(rhs, torch.zeros_like(rhs), steps),
            rtol=1e-13,
            atol=1e-13,
        )

    def test_zero_steps_from_zero_is_zero(
        self, csr_matrix: torch.Tensor, rhs: torch.Tensor
    ) -> None:
        prepared = GaussSeidelSmoother().prepare(csr_matrix)
        torch.testing.assert_close(prepared.smooth_from_zero(rhs, 0), torch.zeros_like(rhs))

    def test_indexed_rows_match_dense_oracle(
        self,
        dense_matrix: torch.Tensor,
        csr_matrix: torch.Tensor,
        x0: torch.Tensor,
        rhs: torch.Tensor,
        dense_symmetric_gauss_seidel: DenseSymmetricGaussSeidel,
    ) -> None:
        rows = torch.nonzero(torch.arange(dense_matrix.shape[0]) % 3 != 0).flatten()
        expected = dense_symmetric_gauss_seidel(dense_matrix, x0, rhs, 2, rows=rows)
        actual = GaussSeidelSmoother().prepare(csr_matrix).smooth(rhs, x0, 2, rows=rows)
        torch.testing.assert_close(actual, expected)

    def test_input_iterate_is_not_modified(
        self, csr_matrix: torch.Tensor, rhs: torch.Tensor, x0: torch.Tensor
    ) -> None:
        before = x0.clone()
        GaussSeidelSmoother().prepare(csr_matrix).smooth(rhs, x0, 2)
        torch.testing.assert_close(x0, before, rtol=0, atol=0)

    def test_schedules_are_built_once_across_many_sweeps(
        self,
        csr_matrix: torch.Tensor,
        rhs: torch.Tensor,
        x0: torch.Tensor,
        count_level_schedules: list[int],
    ) -> None:
        prepared = GaussSeidelSmoother().prepare(csr_matrix)
        built_by_prepare = len(count_level_schedules)
        assert built_by_prepare == 2  # forward and backward, independently
        x = x0
        for _ in range(4):
            x = prepared.smooth(rhs, x, 3)
        prepared.smooth_from_zero(rhs, 3)
        prepared.smooth(rhs, x0, 2, rows=torch.arange(0, csr_matrix.shape[0], 2))
        assert len(count_level_schedules) == built_by_prepare


class TestGaussSeidelStage:
    """Stage-scoped cache: each distinct matrix is prepared once per stage."""

    def test_repeated_calls_on_one_matrix_build_schedules_once(
        self,
        csr_matrix: torch.Tensor,
        rhs: torch.Tensor,
        x0: torch.Tensor,
        count_level_schedules: list[int],
    ) -> None:
        stage = smoothers.GaussSeidelStage()
        for _ in range(5):
            stage.prepare(csr_matrix).smooth(rhs, x0, 1)
        assert len(count_level_schedules) == 2

    def test_distinct_matrices_are_prepared_separately(
        self,
        csr_matrix: torch.Tensor,
        rhs: torch.Tensor,
        x0: torch.Tensor,
        count_level_schedules: list[int],
    ) -> None:
        stage = smoothers.GaussSeidelStage()
        other = csr_matrix.clone()
        stage.prepare(csr_matrix).smooth(rhs, x0, 1)
        stage.prepare(other).smooth(rhs, x0, 1)
        assert len(count_level_schedules) == 4

    def test_matches_prepared_smoother(
        self, csr_matrix: torch.Tensor, rhs: torch.Tensor, x0: torch.Tensor
    ) -> None:
        stage = smoothers.GaussSeidelStage()
        torch.testing.assert_close(
            stage.prepare(csr_matrix).smooth(rhs, x0, 2),
            GaussSeidelSmoother().prepare(csr_matrix).smooth(rhs, x0, 2),
        )

    def test_stages_do_not_share_state(
        self,
        csr_matrix: torch.Tensor,
        rhs: torch.Tensor,
        x0: torch.Tensor,
        count_level_schedules: list[int],
    ) -> None:
        smoothers.GaussSeidelStage().prepare(csr_matrix)
        smoothers.GaussSeidelStage().prepare(csr_matrix)
        assert len(count_level_schedules) == 4


class TestResolveJacobiDefault:
    def test_returns_injected_smoother(self) -> None:
        smoother = GaussSeidelSmoother()
        assert resolve_jacobi_default(smoother, None) is smoother

    def test_defaults_to_jacobi_with_omega(self) -> None:
        assert isinstance(resolve_jacobi_default(None, _FIXED_OMEGA), JacobiSmoother)

    def test_rejects_smoother_with_omega(self) -> None:
        with pytest.raises(ValueError, match="smoother_omega"):
            resolve_jacobi_default(GaussSeidelSmoother(), _FIXED_OMEGA)
