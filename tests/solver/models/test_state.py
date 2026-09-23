"""Tests for ``torchalg.models.state``."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import pytest
import torch

from torchalg.models.history import DirectionHistory, ResidualHistory
from torchalg.models.state import CGState, KrylovState, SolverState


class TestSolverState:
    """Tests for the base ``SolverState``."""

    def test_valid_state_constructs(self, solver_state: SolverState) -> None:
        """A well-formed state constructs without raising."""
        assert solver_state.iteration == 0
        assert solver_state.converged is False

    def test_negative_iteration_raises(self) -> None:
        """A negative iteration count violates the state invariant."""
        with pytest.raises(ValueError, match="iteration must be >= 0"):
            SolverState(
                iteration=-1,
                converged=False,
                breakdown=False,
                divergence=False,
                residual_norm=1.0,
                rhs_norm=1.0,
            )

    def test_negative_residual_norm_raises(self) -> None:
        """A negative residual norm violates the state invariant."""
        with pytest.raises(ValueError, match="residual_norm must be >= 0"):
            SolverState(
                iteration=0,
                converged=False,
                breakdown=False,
                divergence=False,
                residual_norm=-1.0,
                rhs_norm=1.0,
            )

    def test_negative_rhs_norm_raises(self) -> None:
        """A negative RHS norm violates the state invariant."""
        with pytest.raises(ValueError, match="rhs_norm must be >= 0"):
            SolverState(
                iteration=0,
                converged=False,
                breakdown=False,
                divergence=False,
                residual_norm=1.0,
                rhs_norm=-1.0,
            )

    def test_state_is_frozen(self, solver_state: SolverState) -> None:
        """Mutating a field after construction raises."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            solver_state.iteration = 1  # ty: ignore[invalid-assignment]


class TestKrylovState:
    """Tests for ``KrylovState``."""

    def test_vectors_are_preserved_without_swapping(
        self,
        krylov_state: KrylovState,
        krylov_vectors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Each field holds the tensor it was constructed with, not a swapped one."""
        u, r, w, d, q = krylov_vectors
        assert torch.equal(krylov_state.u, u)
        assert torch.equal(krylov_state.r, r)
        assert torch.equal(krylov_state.w, w)
        assert torch.equal(krylov_state.d, d)
        assert torch.equal(krylov_state.q, q)

    def test_inherits_solver_state_fields(self, krylov_state: KrylovState) -> None:
        """``KrylovState`` is a ``SolverState`` with the base fields intact."""
        assert isinstance(krylov_state, SolverState)
        assert krylov_state.iteration == 0
        assert krylov_state.residual_norm == 1.0

    def test_requires_keyword_arguments(
        self,
        krylov_vectors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """``kw_only=True`` rejects positional construction, preventing field mix-ups.

        The deliberately-wrong call is statically invalid too (``ty`` flags
        the same arity/keyword mismatch this test asserts a runtime
        ``TypeError`` for), hence the blanket ignore below.
        """
        u, r, w, d, q = krylov_vectors
        with pytest.raises(TypeError):
            KrylovState(0, False, False, False, 1.0, 1.0, u, r, w, d, q)  # ty: ignore[missing-argument, too-many-positional-arguments]


class TestCGState:
    """Tests for ``CGState`` and its ``create_initial`` factory."""

    def test_create_initial_sets_iteration_zero(self, cg_state: CGState) -> None:
        """A freshly created state starts at iteration 0, unconverged."""
        assert cg_state.iteration == 0
        assert cg_state.converged is False
        assert cg_state.breakdown is False
        assert cg_state.divergence is False

    def test_create_initial_seeds_residual_history_with_one_entry(
        self,
        cg_state: CGState,
    ) -> None:
        """``residual_history`` starts with exactly the initial residual norm."""
        assert len(cg_state.residual_history) == 1
        assert cg_state.residual_history.norms_abs[0] == cg_state.residual_norm
        assert cg_state.residual_history.norms_rel[0] == pytest.approx(
            cg_state.residual_norm / cg_state.rhs_norm
        )

    def test_create_initial_direction_history_is_empty(self, cg_state: CGState) -> None:
        """``direction_history`` starts empty, with the requested window size."""
        assert len(cg_state.direction_history) == 0
        assert cg_state.direction_history.max_size == 3

    def test_create_initial_has_no_previous_iterate(self, cg_state: CGState) -> None:
        """Two-term-recurrence lookback fields are unset at iteration 0."""
        assert cg_state.w_prev is None
        assert cg_state.r_prev is None
        assert cg_state.rw_prev == 0.0

    def test_create_initial_zero_rhs_norm_uses_absolute_norm(
        self,
        krylov_vectors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """A zero RHS norm falls back to the absolute residual norm (no division by zero)."""
        u, r, w, d, q = krylov_vectors
        state = CGState.create_initial(u=u, r=r, w=w, d=d, q=q, residual_norm=2.0, rhs_norm=0.0)
        assert state.residual_history.norms_rel[0] == 2.0

    def test_is_krylov_state_and_solver_state(self, cg_state: CGState) -> None:
        """``CGState`` extends the full ``SolverState`` -> ``KrylovState`` chain."""
        assert isinstance(cg_state, KrylovState)
        assert isinstance(cg_state, SolverState)

    def test_requires_keyword_arguments(
        self,
        krylov_vectors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """``kw_only=True`` is preserved down the inheritance chain.

        See ``KrylovState``'s analogous test for why the ``ty: ignore`` below
        is expected: the deliberately-wrong call is statically invalid too.
        """
        u, r, w, d, q = krylov_vectors
        with pytest.raises(TypeError):
            CGState(  # ty: ignore[missing-argument]
                0,
                False,
                False,
                False,
                1.0,
                1.0,
                u,  # ty: ignore[too-many-positional-arguments]
                r,
                w,
                d,
                q,
                DirectionHistory.empty(),
                ResidualHistory.empty(),
                None,
                None,
                0.0,
            )

    def test_state_is_frozen(self, cg_state: CGState) -> None:
        """Mutating a field after construction raises."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            cg_state.rw_prev = 1.0  # ty: ignore[invalid-assignment]

    def test_tensor_residual_norm_construction_no_sync(self) -> None:
        """Constructing SolverState with tensor residual_norm doesn't force bool conversion."""
        bool_call_count = 0
        original_bool = torch.Tensor.__bool__

        def counting_bool(self: Any) -> bool:
            nonlocal bool_call_count
            bool_call_count += 1
            return original_bool(self)

        torch.Tensor.__bool__ = cast(Any, counting_bool)
        try:
            state = SolverState(
                iteration=0,
                converged=False,
                breakdown=False,
                divergence=False,
                residual_norm=torch.tensor(1.0),
                rhs_norm=1.0,
            )
            assert bool_call_count == 0, f"Expected 0 bool() calls, got {bool_call_count}"
            assert isinstance(state.residual_norm, torch.Tensor)
        finally:
            torch.Tensor.__bool__ = original_bool

    def test_tensor_rhs_norm_construction_no_sync(self) -> None:
        """Constructing SolverState with tensor rhs_norm doesn't force bool conversion."""
        bool_call_count = 0
        original_bool = torch.Tensor.__bool__

        def counting_bool(self: Any) -> bool:
            nonlocal bool_call_count
            bool_call_count += 1
            return original_bool(self)

        torch.Tensor.__bool__ = cast(Any, counting_bool)
        try:
            state = SolverState(
                iteration=0,
                converged=False,
                breakdown=False,
                divergence=False,
                residual_norm=1.0,
                rhs_norm=torch.tensor(1.0),
            )
            assert bool_call_count == 0, f"Expected 0 bool() calls, got {bool_call_count}"
            assert isinstance(state.rhs_norm, torch.Tensor)
        finally:
            torch.Tensor.__bool__ = original_bool
