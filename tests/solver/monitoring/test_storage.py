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

    def test_add_and_prepend_store_tensors_internally(self) -> None:
        """Internally, values are stored as 0-d tensors, not Python floats."""
        history = ScalarHistory.empty()
        appended = history.add(2.0)
        prepended = appended.prepend(1.0)

        # Stored values are tensors, not floats
        assert isinstance(prepended.values[0], torch.Tensor)
        assert isinstance(prepended.values[1], torch.Tensor)
        assert prepended.values[0].ndim == 0  # 0-d tensor
        assert prepended.values[1].ndim == 0
        assert prepended.values[0].item() == 1.0
        assert prepended.values[1].item() == 2.0

    def test_getitem_returns_tensor(self) -> None:
        """Indexing exposes the stored scalar as a 0-d tensor."""
        v0 = torch.tensor(1.0)
        v1 = torch.tensor(2.0)
        history = ScalarHistory(values=(v0, v1))
        assert isinstance(history[0], torch.Tensor)
        assert isinstance(history[-1], torch.Tensor)
        assert history[0].item() == 1.0
        assert history[-1].item() == 2.0

    def test_history_is_frozen(self) -> None:
        """Mutating a field after construction raises."""
        history = ScalarHistory.empty()
        with pytest.raises(dataclasses.FrozenInstanceError):
            history.values = (torch.tensor(1.0),)  # ty: ignore[invalid-assignment]

    def test_empty_to_list_returns_empty_list(self) -> None:
        """to_list() on an empty history returns an empty list."""
        history = ScalarHistory.empty()
        assert history.to_list() == []

    def test_add_with_tensor_input(self) -> None:
        """add() accepts both float and torch.Tensor inputs."""
        history = ScalarHistory.empty()
        history = history.add(1.0)
        history = history.add(torch.tensor(2.0))
        history = history.add(3.0)

        assert history.to_list() == [1.0, 2.0, 3.0]

    def test_prepend_with_tensor_input(self) -> None:
        """prepend() accepts both float and torch.Tensor inputs."""
        history = ScalarHistory.empty()
        history = history.add(torch.tensor(3.0))
        history = history.prepend(2.0)
        history = history.prepend(torch.tensor(1.0))

        assert history.to_list() == [1.0, 2.0, 3.0]


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

    def test_add_moves_to_cpu(self, residual_vector: torch.Tensor) -> None:
        """Stored vectors are always CPU-resident, regardless of solve device."""
        history = VectorHistory.empty().add(residual_vector)
        assert history[0].device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
    def test_add_moves_cuda_tensor_to_cpu(self) -> None:
        """A CUDA-resident vector is copied to host memory immediately, not left on device."""
        vector = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device="cuda")
        history = VectorHistory.empty().add(vector)
        assert history[0].device.type == "cpu"

    def test_history_is_frozen(self) -> None:
        """Mutating a field after construction raises."""
        history = VectorHistory.empty()
        with pytest.raises(dataclasses.FrozenInstanceError):
            history.vectors = (torch.ones(1),)  # ty: ignore[invalid-assignment]
