"""Tests for ``torchalg.sparse.triangular``."""

from __future__ import annotations

from typing import Literal

import pytest
import torch

from torchalg.sparse.kernels.triangular import (
    PreparedTriangularSolve,
    _device_dependency_schedule,
    level_schedule,
)


@pytest.fixture
def diagonal_csr(torch_dtype: torch.dtype) -> torch.Tensor:
    """Return a diagonal CSR matrix whose rows are all independent."""
    return torch.eye(5, dtype=torch_dtype).to_sparse_csr()


@pytest.fixture
def empty_csr(torch_dtype: torch.dtype) -> torch.Tensor:
    """Return the empty square CSR matrix."""
    return torch.empty((0, 0), dtype=torch_dtype).to_sparse_csr()


class TestLevelScheduleAndTriangularSolve:
    """Forward/backward level-scheduled solves match a direct dense solve."""

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_matches_dense_triangular_solve_poisson_1d(
        self,
        poisson_1d_dense: torch.Tensor,
        poisson_1d_csr: torch.Tensor,
        direction: Literal["forward", "backward"],
    ) -> None:
        """Matches ``torch.linalg.solve_triangular`` on the 1D Poisson fixture."""
        torch.manual_seed(42)
        target = torch.randn(poisson_1d_dense.shape[0], dtype=poisson_1d_dense.dtype)

        actual = PreparedTriangularSolve(poisson_1d_csr, direction).solve(target)

        triangle = (
            torch.tril(poisson_1d_dense) if direction == "forward" else torch.triu(poisson_1d_dense)
        )
        expected = torch.linalg.solve_triangular(
            triangle, target.unsqueeze(1), upper=(direction == "backward")
        ).squeeze(1)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_matches_dense_triangular_solve_poisson_2d(
        self,
        poisson_2d_dense: torch.Tensor,
        poisson_2d_csr: torch.Tensor,
        direction: Literal["forward", "backward"],
    ) -> None:
        """Matches dense triangular solve on the 2D (multi-row-per-level) fixture."""
        torch.manual_seed(42)
        target = torch.randn(poisson_2d_dense.shape[0], dtype=poisson_2d_dense.dtype)

        actual = PreparedTriangularSolve(poisson_2d_csr, direction).solve(target)

        triangle = (
            torch.tril(poisson_2d_dense) if direction == "forward" else torch.triu(poisson_2d_dense)
        )
        expected = torch.linalg.solve_triangular(
            triangle, target.unsqueeze(1), upper=(direction == "backward")
        ).squeeze(1)
        torch.testing.assert_close(actual, expected)

    def test_rejects_non_csr_layout(self, poisson_1d_dense: torch.Tensor) -> None:
        """Dense (strided) input is rejected by both functions, not silently misread."""
        with pytest.raises(ValueError, match="sparse CSR"):
            level_schedule(poisson_1d_dense)
        with pytest.raises(ValueError, match="sparse CSR"):
            PreparedTriangularSolve(poisson_1d_dense, "forward")

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_diagonal_matrix_has_one_level(
        self,
        diagonal_csr: torch.Tensor,
        direction: Literal["forward", "backward"],
    ) -> None:
        """All rows of a diagonal matrix belong to the same level."""
        schedule = level_schedule(diagonal_csr, direction=direction)

        torch.testing.assert_close(schedule.row_order, torch.arange(5))
        torch.testing.assert_close(schedule.level_sizes, torch.tensor([5]))

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_empty_matrix_has_empty_schedule(
        self,
        empty_csr: torch.Tensor,
        direction: Literal["forward", "backward"],
    ) -> None:
        """The empty matrix has neither scheduled rows nor levels."""
        schedule = level_schedule(empty_csr, direction=direction)

        assert schedule.row_order.numel() == 0
        assert schedule.level_sizes.numel() == 0

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_every_dependency_precedes_its_row(
        self,
        poisson_2d_csr: torch.Tensor,
        direction: Literal["forward", "backward"],
    ) -> None:
        """Every triangular dependency is assigned to an earlier level."""
        schedule = level_schedule(poisson_2d_csr, direction=direction)
        levels = torch.empty(poisson_2d_csr.shape[0], dtype=torch.int64)
        levels[schedule.row_order] = torch.repeat_interleave(
            torch.arange(len(schedule.level_sizes)), schedule.level_sizes
        )

        rows = torch.repeat_interleave(
            torch.arange(poisson_2d_csr.shape[0]),
            poisson_2d_csr.crow_indices()[1:] - poisson_2d_csr.crow_indices()[:-1],
        )
        cols = poisson_2d_csr.col_indices()
        dependency = cols < rows if direction == "forward" else cols > rows

        assert bool((levels[cols[dependency]] < levels[rows[dependency]]).all())

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_wavefront_schedule_matches_cpu_recurrence(
        self,
        poisson_2d_csr: torch.Tensor,
        direction: Literal["forward", "backward"],
    ) -> None:
        """The accelerator wavefront algorithm preserves exact schedule semantics."""
        rows = torch.repeat_interleave(
            torch.arange(poisson_2d_csr.shape[0]),
            poisson_2d_csr.crow_indices()[1:] - poisson_2d_csr.crow_indices()[:-1],
        )
        cols = poisson_2d_csr.col_indices()
        dependency = cols < rows if direction == "forward" else cols > rows

        expected = level_schedule(poisson_2d_csr, direction=direction)
        actual = _device_dependency_schedule(
            rows[dependency], cols[dependency], poisson_2d_csr.shape[0]
        )

        torch.testing.assert_close(actual.row_order, expected.row_order)
        torch.testing.assert_close(actual.level_sizes, expected.level_sizes)

    def test_zero_diagonal_row_freezes_to_target(self, torch_dtype: torch.dtype) -> None:
        """A structurally-zero diagonal entry freezes that row to ``target`` instead of dividing."""
        dense = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 1.0, 3.0]], dtype=torch_dtype)
        matrix = dense.to_sparse_csr()
        target = torch.tensor([5.0, 1.0, 1.0], dtype=torch_dtype)

        actual = PreparedTriangularSolve(matrix, "forward").solve(target)

        assert actual[0] == target[0]

    def test_backward_schedule_is_not_reversed_forward_schedule(
        self, torch_dtype: torch.dtype
    ) -> None:
        """Pinned regression for the counterexample in ``docs/plan.md``.

        Symmetric edges (0,1), (1,3), (2,3) on a 4-node DAG: the forward
        schedule is ``[0, 1, 0, 2]``; the "reversed forward" shortcut would
        guess backward levels ``max(forward) - forward = [2, 1, 2, 0]``, but
        the true, independently-computed backward schedule is ``[2, 1, 1,
        0]`` (node 2's only path to a sink is the short one through node 3).
        """
        n = 4
        dense = torch.zeros((n, n), dtype=torch_dtype)
        dense[0, 0] = dense[1, 1] = dense[2, 2] = dense[3, 3] = 1.0
        dense[1, 0] = dense[0, 1] = 1.0
        dense[3, 1] = dense[1, 3] = 1.0
        dense[3, 2] = dense[2, 3] = 1.0
        matrix = dense.to_sparse_csr()

        forward_schedule = level_schedule(matrix, direction="forward")
        backward_schedule = level_schedule(matrix, direction="backward")

        forward_level = torch.empty(n, dtype=torch.int64)
        forward_level[forward_schedule.row_order] = torch.repeat_interleave(
            torch.arange(len(forward_schedule.level_sizes)), forward_schedule.level_sizes
        )
        backward_level = torch.empty(n, dtype=torch.int64)
        backward_level[backward_schedule.row_order] = torch.repeat_interleave(
            torch.arange(len(backward_schedule.level_sizes)), backward_schedule.level_sizes
        )

        reversed_forward_guess = int(forward_level.max()) - forward_level
        assert not torch.equal(backward_level, reversed_forward_guess)
        torch.testing.assert_close(backward_level, torch.tensor([2, 1, 1, 0]))

        torch.manual_seed(42)
        target = torch.randn(n, dtype=torch_dtype)
        actual = PreparedTriangularSolve(matrix, "backward").solve(target)
        expected = torch.linalg.solve_triangular(
            torch.triu(dense), target.unsqueeze(1), upper=True
        ).squeeze(1)
        torch.testing.assert_close(actual, expected)

    def test_differentiable_through_values_and_target(self, torch_dtype: torch.dtype) -> None:
        """Gradients flow from the solve output back to CSR ``values`` and to ``target``."""
        dense = torch.tensor([[2.0, 0.0], [1.0, 3.0]], dtype=torch_dtype)
        matrix = dense.to_sparse_csr()
        values = matrix.values().clone().requires_grad_(True)
        matrix = torch.sparse_csr_tensor(
            matrix.crow_indices(),
            matrix.col_indices(),
            values,
            size=matrix.shape,
            check_invariants=False,
        )
        target = torch.tensor([4.0, 5.0], dtype=torch_dtype, requires_grad=True)

        result = PreparedTriangularSolve(matrix, "forward").solve(target)
        result.sum().backward()

        assert values.grad is not None
        assert target.grad is not None


class TestTriangularSolveBatchedColumns:
    """``triangular_solve`` on ``(n, k)`` targets matches the looped ``(n,)`` calls."""

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_matches_looped_unbatched_columns(
        self,
        poisson_1d_csr: torch.Tensor,
        direction: Literal["forward", "backward"],
    ) -> None:
        """Each column of a batched solve matches an independent unbatched solve."""
        torch.manual_seed(42)
        n = poisson_1d_csr.shape[0]
        k = 3
        target = torch.randn(n, k, dtype=poisson_1d_csr.dtype)

        batched = PreparedTriangularSolve(poisson_1d_csr, direction).solve(target)

        for col in range(k):
            expected = PreparedTriangularSolve(poisson_1d_csr, direction).solve(target[:, col])
            torch.testing.assert_close(batched[:, col], expected)


class TestPreparedTriangularSolveReuse:
    """All matrix-only structure is built at construction, never per solve."""

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_solve_does_not_rebuild_structure(
        self,
        poisson_2d_csr: torch.Tensor,
        direction: Literal["forward", "backward"],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import torchalg.sparse.kernels.triangular as kernel

        prepared = PreparedTriangularSolve(poisson_2d_csr, direction)
        target = torch.ones(poisson_2d_csr.shape[0], dtype=poisson_2d_csr.dtype)
        first = prepared.solve(target)

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("structure must be built at construction")

        monkeypatch.setattr(kernel, "level_schedule", forbidden)
        monkeypatch.setattr(kernel, "_expand_row_index", forbidden)

        torch.testing.assert_close(prepared.solve(target), first)

    @pytest.mark.parametrize("direction", ["forward", "backward"])
    def test_precomputed_schedule_and_row_index_are_accepted(
        self,
        poisson_2d_csr: torch.Tensor,
        direction: Literal["forward", "backward"],
    ) -> None:
        from torchalg.sparse.kernels.triangular import _expand_row_index

        target = torch.linspace(1.0, 2.0, poisson_2d_csr.shape[0], dtype=poisson_2d_csr.dtype)
        reference = PreparedTriangularSolve(poisson_2d_csr, direction).solve(target)
        shared = PreparedTriangularSolve(
            poisson_2d_csr,
            direction,
            schedule=level_schedule(poisson_2d_csr, direction=direction),
            row_index=_expand_row_index(poisson_2d_csr),
        )

        torch.testing.assert_close(shared.solve(target), reference)

    def test_inactive_rows_freeze_to_current_value(self, poisson_1d_csr: torch.Tensor) -> None:
        n = poisson_1d_csr.shape[0]
        prepared = PreparedTriangularSolve(poisson_1d_csr, "forward")
        target = torch.ones(n, dtype=poisson_1d_csr.dtype)
        current = torch.full((n,), 7.0, dtype=poisson_1d_csr.dtype)
        active = torch.arange(n) % 2 == 0

        result = prepared.solve(target, active=active, current=current)

        assert torch.equal(result[~active], current[~active])

    def test_inactive_rows_without_current_are_rejected(self, poisson_1d_csr: torch.Tensor) -> None:
        n = poisson_1d_csr.shape[0]
        prepared = PreparedTriangularSolve(poisson_1d_csr, "forward")

        with pytest.raises(ValueError, match="current is required"):
            prepared.solve(
                torch.ones(n, dtype=poisson_1d_csr.dtype), active=torch.zeros(n, dtype=torch.bool)
            )
