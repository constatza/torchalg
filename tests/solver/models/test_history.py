"""Tests for ``torchalg.models.history``."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from torchalg.models.history import DirectionHistory, ResidualHistory


class TestDirectionHistory:
    """Tests for ``DirectionHistory``."""

    def test_empty_has_no_vectors(self, empty_direction_history: DirectionHistory) -> None:
        """A freshly created empty history has no vectors and zero updates."""
        assert empty_direction_history.d_vectors == ()
        assert empty_direction_history.q_vectors == ()
        assert len(empty_direction_history) == 0
        assert empty_direction_history.total_updates == 0

    def test_add_appends_vectors(
        self,
        empty_direction_history: DirectionHistory,
        direction_vector_sequence: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """A single ``add`` grows the window by one and records the vector."""
        d0, q0 = direction_vector_sequence[0]
        history = empty_direction_history.add(d0, q0)
        assert len(history) == 1
        assert torch.equal(history.d_vectors[0], d0)
        assert torch.equal(history.q_vectors[0], q0)
        assert history.total_updates == 1

    def test_add_truncates_to_max_size(
        self,
        empty_direction_history: DirectionHistory,
        direction_vector_sequence: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """A window of size 2 keeps only the most recent 2 vectors after 3 adds."""
        history = empty_direction_history
        for d, q in direction_vector_sequence:
            history = history.add(d, q)

        assert len(history) == empty_direction_history.max_size == 2
        # The oldest pair (index 0) was dropped; the last two survive in order.
        assert torch.equal(history.d_vectors[0], direction_vector_sequence[1][0])
        assert torch.equal(history.d_vectors[1], direction_vector_sequence[2][0])

    def test_total_updates_is_monotonic_past_truncation(
        self,
        empty_direction_history: DirectionHistory,
        direction_vector_sequence: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """``total_updates`` keeps counting even once the window truncates."""
        history = empty_direction_history
        for d, q in direction_vector_sequence:
            history = history.add(d, q)
        assert history.total_updates == len(direction_vector_sequence) == 3

    def test_add_clones_input_tensors(
        self,
        empty_direction_history: DirectionHistory,
        direction_vector_sequence: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Stored vectors never alias the caller's tensor: later mutation is invisible."""
        d0, q0 = direction_vector_sequence[0]
        d0 = d0.clone()
        history = empty_direction_history.add(d0, q0)

        d0.fill_(999.0)

        assert not torch.equal(history.d_vectors[0], d0)

    def test_add_returns_new_instance(
        self,
        empty_direction_history: DirectionHistory,
        direction_vector_sequence: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """``add`` is a pure functional update: the original instance is untouched."""
        d0, q0 = direction_vector_sequence[0]
        updated = empty_direction_history.add(d0, q0)
        assert updated is not empty_direction_history
        assert len(empty_direction_history) == 0
        assert len(updated) == 1

    def test_history_is_frozen(self, empty_direction_history: DirectionHistory) -> None:
        """Mutating a field after construction raises."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            empty_direction_history.max_size = 5  # ty: ignore[invalid-assignment]


class TestResidualHistory:
    """Tests for ``ResidualHistory``."""

    def test_empty_has_no_norms(self, empty_residual_history: ResidualHistory) -> None:
        """A freshly created empty history has no recorded norms."""
        assert empty_residual_history.norms_abs == ()
        assert empty_residual_history.norms_rel == ()
        assert len(empty_residual_history) == 0

    def test_add_appends_norms_in_order(self, empty_residual_history: ResidualHistory) -> None:
        """Successive ``add`` calls append in call order."""
        history = empty_residual_history.add(norm_abs=1.0, norm_rel=0.1)
        history = history.add(norm_abs=0.5, norm_rel=0.05)
        assert history.norms_abs == (1.0, 0.5)
        assert history.norms_rel == (0.1, 0.05)
        assert len(history) == 2

    def test_add_coerces_to_float(self, empty_residual_history: ResidualHistory) -> None:
        """Non-float numeric input (e.g. an int) is coerced to ``float``."""
        history = empty_residual_history.add(norm_abs=1, norm_rel=0)
        assert history.norms_abs == (1.0,)
        assert isinstance(history.norms_abs[0], float)

    def test_add_returns_new_instance(self, empty_residual_history: ResidualHistory) -> None:
        """``add`` is a pure functional update: the original instance is untouched."""
        updated = empty_residual_history.add(norm_abs=1.0, norm_rel=1.0)
        assert updated is not empty_residual_history
        assert len(empty_residual_history) == 0

    def test_history_is_frozen(self, empty_residual_history: ResidualHistory) -> None:
        """Mutating a field after construction raises."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            empty_residual_history.norms_abs = (1.0,)  # ty: ignore[invalid-assignment]
