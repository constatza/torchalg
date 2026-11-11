"""Dense prepared smoothers: ``prepare(A)`` does the matrix work once, sweeps only sweep.

See ``tests/sparse/preconditioners/amg/test_smoothers.py`` for the sparse CSR siblings.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torchalg.preconditioners.implementations.amg import _jacobi_omega, _relaxation
from torchalg.preconditioners.implementations.amg._jacobi_omega import (
    RELAXATION_NOMINAL,
    jacobi_spectral_radius,
)
from torchalg.preconditioners.implementations.amg._relaxation import (
    PreparedSymmetricGaussSeidel,
    prepare_symmetric_gauss_seidel,
)
from torchalg.preconditioners.implementations.amg.smoothers import (
    GaussSeidelSmoother,
    JacobiSmoother,
    SmootherBase,
)

SIZE = 24
BLOCK_COLUMNS = 3
EXPLICIT_OMEGA = 0.6
DEAD_ROW = 5
FROZEN_ROWS = (2, 3, 7, 11, 12)
STEP_COUNTS = [0, 1, 3]
TOLERANCE = 1e-12


@pytest.fixture
def poisson(poisson_1d_factory: Callable[[int], torch.Tensor]) -> torch.Tensor:
    return poisson_1d_factory(SIZE)


@pytest.fixture
def matrix_with_dead_row(poisson: torch.Tensor) -> torch.Tensor:
    """Poisson matrix whose ``DEAD_ROW`` has a zero diagonal (row and column keep couplings)."""
    dead = poisson.clone()
    dead[DEAD_ROW, DEAD_ROW] = 0.0
    return dead


@pytest.fixture
def rhs_vector(torch_dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(SIZE, generator=torch.Generator().manual_seed(1), dtype=torch_dtype)


@pytest.fixture
def x_vector(torch_dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(SIZE, generator=torch.Generator().manual_seed(2), dtype=torch_dtype)


@pytest.fixture
def rhs_block(torch_dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(
        SIZE, BLOCK_COLUMNS, generator=torch.Generator().manual_seed(3), dtype=torch_dtype
    )


@pytest.fixture
def x_block(torch_dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(
        SIZE, BLOCK_COLUMNS, generator=torch.Generator().manual_seed(4), dtype=torch_dtype
    )


def reference_jacobi(
    matrix: torch.Tensor, rhs: torch.Tensor, x: torch.Tensor, omega: float, steps: int
) -> torch.Tensor:
    """Textbook weighted Jacobi, zero step on zero-diagonal rows."""
    diagonal = torch.diagonal(matrix)
    scale = torch.where(diagonal.abs() > 1e-14, omega / diagonal, torch.zeros_like(diagonal))
    for _ in range(steps):
        x = x + scale.reshape(-1, *[1] * (x.ndim - 1)) * (rhs - matrix @ x)
    return x


def reference_symmetric_gauss_seidel(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    x: torch.Tensor,
    steps: int,
    rows: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Row-by-row forward then backward Gauss-Seidel; skips zero-diagonal and non-listed rows."""
    x = x.clone()
    updated = set(range(matrix.shape[0])) if rows is None else set(rows)
    n = matrix.shape[0]
    for _ in range(steps):
        for order in (range(n), range(n - 1, -1, -1)):
            for i in order:
                if i not in updated or matrix[i, i] == 0:
                    continue
                residual = rhs[i] - matrix[i] @ x + matrix[i, i] * x[i]
                x[i] = residual / matrix[i, i]
    return x


class TestPreparedJacobi:
    @pytest.mark.parametrize("steps", STEP_COUNTS)
    @pytest.mark.parametrize(
        "omega", [EXPLICIT_OMEGA, torch.tensor(EXPLICIT_OMEGA, dtype=torch.float64)]
    )
    def test_explicit_omega_equals_the_textbook_formula(
        self,
        poisson: torch.Tensor,
        rhs_vector: torch.Tensor,
        x_vector: torch.Tensor,
        omega: float | torch.Tensor,
        steps: int,
    ) -> None:
        prepared = JacobiSmoother(omega=omega).prepare(poisson)

        torch.testing.assert_close(
            prepared.smooth(rhs_vector, x_vector, steps),
            reference_jacobi(poisson, rhs_vector, x_vector, EXPLICIT_OMEGA, steps),
            rtol=TOLERANCE,
            atol=TOLERANCE,
        )

    def test_default_omega_is_the_relaxation_rule_resolved_in_prepare(
        self, poisson: torch.Tensor, rhs_vector: torch.Tensor, x_vector: torch.Tensor
    ) -> None:
        omega = RELAXATION_NOMINAL / jacobi_spectral_radius(poisson)

        prepared = JacobiSmoother().prepare(poisson)

        torch.testing.assert_close(
            prepared.smooth(rhs_vector, x_vector, 3),
            reference_jacobi(poisson, rhs_vector, x_vector, float(omega), 3),
            rtol=TOLERANCE,
            atol=TOLERANCE,
        )

    def test_prepare_estimates_the_spectral_radius_once_and_sweeps_never(
        self,
        poisson: torch.Tensor,
        rhs_vector: torch.Tensor,
        x_vector: torch.Tensor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[int] = []
        original = _jacobi_omega.approximate_spectral_radius

        def counting(matrix: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            calls.append(matrix.shape[0])
            return original(matrix, *args, **kwargs)  # ty: ignore[invalid-argument-type]

        monkeypatch.setattr(_jacobi_omega, "approximate_spectral_radius", counting)

        prepared = JacobiSmoother().prepare(poisson)
        assert calls == [SIZE]
        prepared.smooth(rhs_vector, x_vector, 2)
        prepared.smooth_from_zero(rhs_vector, 2)

        assert calls == [SIZE]

    def test_explicit_omega_never_estimates(
        self, poisson: torch.Tensor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden(*args: object, **kwargs: object) -> torch.Tensor:
            raise AssertionError("spectral radius estimated despite an explicit omega")

        monkeypatch.setattr(_jacobi_omega, "approximate_spectral_radius", forbidden)

        JacobiSmoother(omega=EXPLICIT_OMEGA).prepare(poisson)

    def test_zero_diagonal_component_is_left_unchanged(
        self, matrix_with_dead_row: torch.Tensor, rhs_vector: torch.Tensor, x_vector: torch.Tensor
    ) -> None:
        prepared = JacobiSmoother(omega=EXPLICIT_OMEGA).prepare(matrix_with_dead_row)

        smoothed = prepared.smooth(rhs_vector, x_vector, 2)

        assert smoothed[DEAD_ROW] == x_vector[DEAD_ROW]

    @pytest.mark.parametrize("steps", STEP_COUNTS)
    def test_smooth_from_zero_equals_smooth_from_a_zero_iterate(
        self, poisson: torch.Tensor, rhs_vector: torch.Tensor, steps: int
    ) -> None:
        prepared = JacobiSmoother(omega=EXPLICIT_OMEGA).prepare(poisson)

        assert torch.equal(
            prepared.smooth_from_zero(rhs_vector, steps),
            prepared.smooth(rhs_vector, torch.zeros_like(rhs_vector), steps),
        )

    @pytest.mark.parametrize("steps", STEP_COUNTS)
    def test_block_equals_column_by_column(
        self,
        poisson: torch.Tensor,
        rhs_block: torch.Tensor,
        x_block: torch.Tensor,
        steps: int,
    ) -> None:
        prepared = JacobiSmoother(omega=EXPLICIT_OMEGA).prepare(poisson)

        block = prepared.smooth(rhs_block, x_block, steps)
        from_zero = prepared.smooth_from_zero(rhs_block, steps)

        for column in range(BLOCK_COLUMNS):
            torch.testing.assert_close(
                block[:, column],
                prepared.smooth(rhs_block[:, column], x_block[:, column], steps),
                rtol=TOLERANCE,
                atol=TOLERANCE,
            )
            torch.testing.assert_close(
                from_zero[:, column],
                prepared.smooth_from_zero(rhs_block[:, column], steps),
                rtol=TOLERANCE,
                atol=TOLERANCE,
            )

    def test_input_iterate_is_not_modified(
        self, poisson: torch.Tensor, rhs_vector: torch.Tensor, x_vector: torch.Tensor
    ) -> None:
        before = x_vector.clone()

        JacobiSmoother(omega=EXPLICIT_OMEGA).prepare(poisson).smooth(rhs_vector, x_vector, 3)

        assert torch.equal(x_vector, before)

    def test_is_a_smoother_base(self) -> None:
        assert isinstance(JacobiSmoother(), SmootherBase)


class TestPreparedGaussSeidel:
    @pytest.mark.parametrize("steps", STEP_COUNTS)
    def test_equals_the_row_by_row_reference(
        self, poisson: torch.Tensor, rhs_vector: torch.Tensor, x_vector: torch.Tensor, steps: int
    ) -> None:
        prepared = GaussSeidelSmoother().prepare(poisson)

        torch.testing.assert_close(
            prepared.smooth(rhs_vector, x_vector, steps),
            reference_symmetric_gauss_seidel(poisson, rhs_vector, x_vector, steps),
            rtol=TOLERANCE,
            atol=TOLERANCE,
        )

    def test_zero_diagonal_rows_are_left_unchanged(
        self, matrix_with_dead_row: torch.Tensor, rhs_vector: torch.Tensor, x_vector: torch.Tensor
    ) -> None:
        prepared = GaussSeidelSmoother().prepare(matrix_with_dead_row)

        smoothed = prepared.smooth(rhs_vector, x_vector, 2)

        assert smoothed[DEAD_ROW] == x_vector[DEAD_ROW]
        torch.testing.assert_close(
            smoothed,
            reference_symmetric_gauss_seidel(matrix_with_dead_row, rhs_vector, x_vector, 2),
            rtol=TOLERANCE,
            atol=TOLERANCE,
        )

    def test_row_subset_updates_only_those_rows(
        self, poisson: torch.Tensor, rhs_vector: torch.Tensor, x_vector: torch.Tensor
    ) -> None:
        rows = torch.tensor(FROZEN_ROWS)
        prepared = prepare_symmetric_gauss_seidel(poisson, rows=rows)

        smoothed = prepared.smooth(rhs_vector, x_vector, 2)

        torch.testing.assert_close(
            smoothed,
            reference_symmetric_gauss_seidel(poisson, rhs_vector, x_vector, 2, rows=FROZEN_ROWS),
            rtol=TOLERANCE,
            atol=TOLERANCE,
        )
        untouched = [i for i in range(SIZE) if i not in FROZEN_ROWS]
        assert torch.equal(smoothed[untouched], x_vector[untouched])

    @pytest.mark.parametrize("steps", STEP_COUNTS)
    def test_smooth_from_zero_equals_smooth_from_a_zero_iterate(
        self, poisson: torch.Tensor, rhs_vector: torch.Tensor, steps: int
    ) -> None:
        prepared = GaussSeidelSmoother().prepare(poisson)

        assert torch.equal(
            prepared.smooth_from_zero(rhs_vector, steps),
            prepared.smooth(rhs_vector, torch.zeros_like(rhs_vector), steps),
        )

    def test_smooth_from_zero_respects_frozen_rows(
        self, matrix_with_dead_row: torch.Tensor, rhs_vector: torch.Tensor
    ) -> None:
        prepared = prepare_symmetric_gauss_seidel(
            matrix_with_dead_row, rows=torch.tensor(FROZEN_ROWS + (DEAD_ROW,))
        )

        assert torch.equal(
            prepared.smooth_from_zero(rhs_vector, 2),
            prepared.smooth(rhs_vector, torch.zeros_like(rhs_vector), 2),
        )

    @pytest.mark.parametrize("steps", STEP_COUNTS)
    def test_block_equals_column_by_column(
        self, poisson: torch.Tensor, rhs_block: torch.Tensor, x_block: torch.Tensor, steps: int
    ) -> None:
        prepared = GaussSeidelSmoother().prepare(poisson)

        block = prepared.smooth(rhs_block, x_block, steps)
        from_zero = prepared.smooth_from_zero(rhs_block, steps)

        for column in range(BLOCK_COLUMNS):
            torch.testing.assert_close(
                block[:, column],
                prepared.smooth(rhs_block[:, column], x_block[:, column], steps),
                rtol=TOLERANCE,
                atol=TOLERANCE,
            )
            torch.testing.assert_close(
                from_zero[:, column],
                prepared.smooth_from_zero(rhs_block[:, column], steps),
                rtol=TOLERANCE,
                atol=TOLERANCE,
            )

    def test_factors_are_built_once_across_many_sweeps_and_calls(
        self,
        poisson: torch.Tensor,
        rhs_vector: torch.Tensor,
        x_vector: torch.Tensor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        directions: list[bool] = []
        original = _relaxation._prepare_sweep

        def counting(matrix: torch.Tensor, *, forward: bool, **kwargs: object) -> object:
            directions.append(forward)
            return original(matrix, forward=forward, **kwargs)  # ty: ignore[invalid-argument-type]

        monkeypatch.setattr(_relaxation, "_prepare_sweep", counting)

        prepared = GaussSeidelSmoother().prepare(poisson)
        for _ in range(5):
            x = prepared.smooth(rhs_vector, x_vector, 4)
            prepared.smooth_from_zero(rhs_vector, 4)

        assert sorted(directions) == [False, True]
        assert x.shape == x_vector.shape

    def test_nothing_frozen_skips_the_mask(self, poisson: torch.Tensor) -> None:
        prepared = prepare_symmetric_gauss_seidel(poisson)

        assert isinstance(prepared, PreparedSymmetricGaussSeidel)
        assert prepared.forward.frozen is None
        assert prepared.backward.frozen is None

    def test_a_zero_diagonal_row_is_frozen(self, matrix_with_dead_row: torch.Tensor) -> None:
        prepared = prepare_symmetric_gauss_seidel(matrix_with_dead_row)

        assert prepared.forward.frozen is not None
        assert prepared.forward.frozen.nonzero().flatten().tolist() == [DEAD_ROW]

    def test_input_iterate_is_not_modified(
        self, poisson: torch.Tensor, rhs_vector: torch.Tensor, x_vector: torch.Tensor
    ) -> None:
        before = x_vector.clone()

        GaussSeidelSmoother().prepare(poisson).smooth(rhs_vector, x_vector, 3)

        assert torch.equal(x_vector, before)

    def test_reduces_the_residual(
        self, poisson: torch.Tensor, rhs_vector: torch.Tensor, x_vector: torch.Tensor
    ) -> None:
        smoothed = GaussSeidelSmoother().prepare(poisson).smooth(rhs_vector, x_vector, 3)

        assert torch.linalg.norm(rhs_vector - poisson @ smoothed) < torch.linalg.norm(
            rhs_vector - poisson @ x_vector
        )

    def test_is_a_smoother_base(self) -> None:
        assert isinstance(GaussSeidelSmoother(), SmootherBase)
