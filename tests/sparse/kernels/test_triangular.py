"""Tests for ``torchalg.sparse.triangular``."""

from __future__ import annotations

from typing import Literal

import pytest
import torch

from torchalg.sparse.kernels.triangular import level_schedule, triangular_solve


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

        schedule = level_schedule(poisson_1d_csr, direction=direction)
        actual = triangular_solve(poisson_1d_csr, schedule, target, direction=direction)

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

        schedule = level_schedule(poisson_2d_csr, direction=direction)
        actual = triangular_solve(poisson_2d_csr, schedule, target, direction=direction)

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
        schedule = level_schedule(poisson_1d_dense.to_sparse_csr())
        with pytest.raises(ValueError, match="sparse CSR"):
            triangular_solve(poisson_1d_dense, schedule, torch.zeros(poisson_1d_dense.shape[0]))

    def test_zero_diagonal_row_freezes_to_target(self, torch_dtype: torch.dtype) -> None:
        """A structurally-zero diagonal entry freezes that row to ``target`` instead of dividing."""
        dense = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 1.0, 3.0]], dtype=torch_dtype)
        matrix = dense.to_sparse_csr()
        target = torch.tensor([5.0, 1.0, 1.0], dtype=torch_dtype)

        schedule = level_schedule(matrix, direction="forward")
        actual = triangular_solve(matrix, schedule, target, direction="forward")

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
        actual = triangular_solve(matrix, backward_schedule, target, direction="backward")
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

        schedule = level_schedule(matrix, direction="forward")
        result = triangular_solve(matrix, schedule, target, direction="forward")
        result.sum().backward()

        assert values.grad is not None
        assert target.grad is not None
