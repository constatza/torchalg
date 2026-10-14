"""Test extra named inputs support in NeuralPreconditioner."""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations.neural import NeuralPreconditioner
from torchalg.preconditioners.ports import ExtraInputPredictorPort


class _RecordingPredictor(ExtraInputPredictorPort):
    """Mock predictor that records calls for testing."""

    def __init__(self, required_inputs: tuple[str, ...] = ()) -> None:
        """Initialize with empty call record."""
        self._required_inputs = required_inputs
        self.residuals: list[torch.Tensor] = []
        self.extras: list[dict[str, torch.Tensor]] = []

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Return predictor-declared required extra inputs.

        Returns:
            Required input names.
        """
        return self._required_inputs

    def apply(self, residual: torch.Tensor, **extra_inputs: torch.Tensor) -> torch.Tensor:
        """Record call details and return a clone of the residual.

        Args:
            residual: Input residual vector.
            **extra_inputs: Named extra tensors to record.

        Returns:
            Clone of the residual.
        """
        self.residuals.append(residual.clone())
        self.extras.append(dict(extra_inputs))
        return residual.clone()

    def cleanup(self) -> None:
        """No-op cleanup."""


@pytest.fixture
def predictor() -> _RecordingPredictor:
    """Fixture: recording predictor the preconditioner borrows."""
    return _RecordingPredictor()


@pytest.fixture
def neural_with_matrix(predictor: _RecordingPredictor) -> NeuralPreconditioner:
    """Fixture: NeuralPreconditioner expecting a matrix extra input."""
    return NeuralPreconditioner(predictor, extra_input_names=("matrix",))


@pytest.fixture
def small_matrix() -> torch.Tensor:
    """Fixture: small identity-like matrix."""
    return torch.eye(3, dtype=torch.float64) * 2.0


def test_extra_input_names_stored(neural_with_matrix: NeuralPreconditioner) -> None:
    """Test that extra_input_names property returns declared inputs."""
    assert neural_with_matrix.extra_input_names == ("matrix",)


def test_bind_inputs_filters_to_declared_names(
    predictor: _RecordingPredictor,
    neural_with_matrix: NeuralPreconditioner,
    small_matrix: torch.Tensor,
) -> None:
    """Test that bind_inputs only keeps keys declared in extra_input_names."""
    neural_with_matrix.bind_inputs(matrix=small_matrix, ignored_key=torch.ones(3))
    neural_with_matrix.apply(torch.ones(3))
    extra = predictor.extras[-1]
    assert "matrix" in extra
    assert "ignored_key" not in extra


def test_apply_without_bind_sends_no_extras(
    predictor: _RecordingPredictor, residual: torch.Tensor
) -> None:
    """Test that apply sends empty extras if bind_inputs was not called."""
    precond = NeuralPreconditioner(predictor, extra_input_names=("matrix",))
    precond.apply(residual)
    assert predictor.extras[-1] == {}


def test_apply_after_bind_forwards_extra(
    predictor: _RecordingPredictor,
    neural_with_matrix: NeuralPreconditioner,
    residual: torch.Tensor,
    small_matrix: torch.Tensor,
) -> None:
    """Test that apply forwards bound extra inputs to the predictor."""
    neural_with_matrix.bind_inputs(matrix=small_matrix)
    neural_with_matrix.apply(residual)
    extra = predictor.extras[-1]
    assert torch.equal(extra["matrix"], small_matrix)


def test_no_extra_input_names_by_default(predictor: _RecordingPredictor) -> None:
    """Test that NeuralPreconditioner has empty extra_input_names by default."""
    assert NeuralPreconditioner(predictor).extra_input_names == ()


def test_extra_input_names_fall_back_to_predictor_required_inputs() -> None:
    """NeuralPreconditioner exposes predictor-declared extras without redeclaration."""
    precond = NeuralPreconditioner(_RecordingPredictor(required_inputs=("matrix",)))

    assert precond.extra_input_names == ("matrix",)


def test_bind_inputs_keeps_predictor_required_inputs(
    residual: torch.Tensor,
    small_matrix: torch.Tensor,
) -> None:
    """Binding honors predictor-declared extras when constructor names are omitted."""
    predictor = _RecordingPredictor(required_inputs=("matrix",))
    precond = NeuralPreconditioner(predictor)

    precond.bind_inputs(matrix=small_matrix, ignored_key=torch.ones(3))
    precond.apply(residual)

    assert predictor.extras[-1] == {"matrix": small_matrix}
