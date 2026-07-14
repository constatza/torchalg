"""Tests for ``torchalg.monitoring.storage``."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from torchalg.monitoring.storage import ScalarHistory, VectorHistory


class TestScalarHistory:
    """Tests for scalar iteration-history storage."""

    def test_empty_has_no_values(self) -> None:
        """A fresh scalar history has no values."""
        history = ScalarHistory.empty()
        assert history.values == ()
        assert len(history) == 0

    def test_add_and_prepend_return_new_instances(self) -> None:
        """Functional updates preserve the old instance and order values correctly."""
        history = ScalarHistory.empty()
        appended = history.add(2)
        prepended = appended.prepend(1)

        assert history.to_list() == []
        assert appended.to_list() == [2.0]
        assert prepended.to_list() == [1.0, 2.0]
        assert appended is not history
        assert prepended is not appended

    def test_getitem_returns_float(self) -> None:
        """Indexing exposes the stored scalar value."""
        history = ScalarHistory(values=(1.0, 2.0))
        assert history[0] == 1.0
        assert history[-1] == 2.0

    def test_history_is_frozen(self) -> None:
        """Mutating a field after construction raises."""
        history = ScalarHistory.empty()
        with pytest.raises(dataclasses.FrozenInstanceError):
            history.values = (1.0,)  # ty: ignore[invalid-assignment]


class TestVectorHistory:
    """Tests for torch tensor iteration-history storage."""

    def test_empty_to_tensor_is_empty_float64(self) -> None:
        """A fresh vector history stacks to an empty float64 tensor."""
        history = VectorHistory.empty()
        stacked = history.to_tensor()
        assert stacked.shape == (0,)
        assert stacked.dtype == torch.float64

    def test_add_and_prepend_stack_vectors(self, residual_vector: torch.Tensor) -> None:
        """Functional updates preserve vector order and stack into rows."""
        history = VectorHistory.empty()
        appended = history.add(residual_vector)
        prepended = appended.prepend(torch.zeros_like(residual_vector))

        expected = torch.stack((torch.zeros_like(residual_vector), residual_vector))
        assert torch.equal(prepended.to_tensor(), expected)
        assert len(history) == 0
        assert len(appended) == 1
        assert len(prepended) == 2

    def test_add_clones_input_tensor(self, residual_vector: torch.Tensor) -> None:
        """Stored vectors do not alias tensors later mutated by the caller."""
        vector = residual_vector.clone()
        history = VectorHistory.empty().add(vector)

        vector.fill_(999.0)

        assert not torch.equal(history[0], vector)

    def test_getitem_returns_clone(self, residual_vector: torch.Tensor) -> None:
        """Indexing returns a clone so callers cannot mutate stored history."""
        history = VectorHistory.empty().add(residual_vector)
        vector = history[0]

        vector.fill_(999.0)

        assert not torch.equal(history[0], vector)

    def test_history_is_frozen(self) -> None:
        """Mutating a field after construction raises."""
        history = VectorHistory.empty()
        with pytest.raises(dataclasses.FrozenInstanceError):
            history.vectors = (torch.ones(1),)  # ty: ignore[invalid-assignment]
