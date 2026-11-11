"""Tests for ``torchalg.models.history``."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from torchalg.models.history import DirectionHistory, ResidualHistory


def _append_all(
    history: DirectionHistory, pairs: list[tuple[torch.Tensor, torch.Tensor]]
) -> DirectionHistory:
    """Append every ``(d, q)`` pair with its dot-product curvature."""
    for d, q in pairs:
        history.append(d, q, torch.dot(d, q))
    return history


class TestDirectionHistory:
    """Tests for ``DirectionHistory`` and its ``DirectionWindow`` views."""

    def test_empty_has_no_vectors(
        self, empty_direction_history: DirectionHistory, direction_vector_sequence: list
    ) -> None:
        """A freshly created empty history has an empty window and zero updates."""
        like = direction_vector_sequence[0][0]
        window = empty_direction_history.window(like=like)
        assert len(window) == 0
        assert window.d.shape == (0, like.shape[0])
        assert window.d.dtype == like.dtype
        assert len(empty_direction_history) == 0
        assert empty_direction_history.total_updates == 0

    def test_append_records_pair_and_curvature(
        self,
        empty_direction_history: DirectionHistory,
        direction_vector_sequence: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """A single ``append`` grows the window by one and stores the pair and curvature."""
        d0, q0 = direction_vector_sequence[0]
        empty_direction_history.append(d0, q0, torch.dot(d0, q0))
        window = empty_direction_history.window(like=d0)
        assert len(empty_direction_history) == len(window) == 1
        assert torch.equal(window.d[0], d0)
        assert torch.equal(window.q[0], q0)
        assert torch.equal(window.curvature[0], torch.dot(d0, q0))
        assert empty_direction_history.total_updates == 1

    def test_window_truncates_to_max_size_keeping_order(
        self,
        empty_direction_history: DirectionHistory,
        direction_vector_sequence: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """A window of size 2 keeps only the most recent 2 pairs after 3 appends."""
        history = _append_all(empty_direction_history, direction_vector_sequence)
        window = history.window(like=direction_vector_sequence[0][0])

        assert len(history) == history.max_size == 2
        assert torch.equal(window.d[0], direction_vector_sequence[1][0])
        assert torch.equal(window.d[1], direction_vector_sequence[2][0])
        assert torch.equal(window.q[1], direction_vector_sequence[2][1])

    def test_total_updates_is_monotonic_past_truncation(
        self,
        empty_direction_history: DirectionHistory,
        direction_vector_sequence: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """``total_updates`` keeps counting even once the window truncates."""
        history = _append_all(empty_direction_history, direction_vector_sequence)
        assert history.total_updates == len(direction_vector_sequence) == 3

    @pytest.mark.parametrize("max_size", [1, 2, 3, 7, 20])
    def test_window_matches_list_slice_over_many_appends(
        self, max_size: int, torch_dtype: torch.dtype
    ) -> None:
        """Growth and compaction keep exactly the last ``max_size`` pairs, in order."""
        generator = torch.Generator().manual_seed(max_size)
        pairs = [
            (
                torch.randn(5, generator=generator, dtype=torch_dtype),
                torch.randn(5, generator=generator, dtype=torch_dtype),
            )
            for _ in range(4 * max_size + 3)
        ]
        history = DirectionHistory.empty(max_size=max_size)
        for count, (d, q) in enumerate(pairs, start=1):
            history.append(d, q, torch.dot(d, q))
            expected = pairs[max(count - max_size, 0) : count]
            window = history.window(like=d)
            assert len(window) == len(expected)
            assert torch.equal(window.d, torch.stack([p[0] for p in expected]))
            assert torch.equal(window.q, torch.stack([p[1] for p in expected]))

    def test_append_copies_input_tensors(
        self,
        empty_direction_history: DirectionHistory,
        direction_vector_sequence: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Stored vectors never alias the caller's tensor: later mutation is invisible."""
        d0, q0 = direction_vector_sequence[0]
        d0 = d0.clone()
        empty_direction_history.append(d0, q0, torch.dot(d0, q0))

        d0.fill_(999.0)

        window = empty_direction_history.window(like=d0)
        assert not torch.equal(window.d[0], d0)

    def test_append_rejects_zero_sized_window(
        self, direction_vector_sequence: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """A zero-size history exists (PCG keeps none) but cannot store pairs."""
        d, q = direction_vector_sequence[0]
        with pytest.raises(ValueError, match="max_size"):
            DirectionHistory.empty(max_size=0).append(d, q, torch.dot(d, q))

    def test_tail_returns_newest_pairs(
        self,
        direction_vector_sequence: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """``DirectionWindow.tail`` keeps the newest rows of every field."""
        history = _append_all(DirectionHistory.empty(max_size=5), direction_vector_sequence)
        window = history.window(like=direction_vector_sequence[0][0])
        tail = window.tail(2)
        assert len(tail) == 2
        assert torch.equal(tail.d, window.d[1:])
        assert torch.equal(tail.curvature, window.curvature[1:])
        assert len(window.tail(10)) == 3


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
