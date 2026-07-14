"""Test extra named inputs support in NeuralPreconditioner."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torchalg.preconditioners.implementations.neural import NeuralPreconditioner
from torchalg.preconditioners.ports import ExtraInputPredictorPort, PredictorAdapter


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


class _MockAdapter(PredictorAdapter):
    """Mock adapter that provides a recording predictor."""

    def __init__(self, required_inputs: tuple[str, ...] = ()) -> None:
        """Initialize with a fresh recording predictor."""
        self.predictor = _RecordingPredictor(required_inputs)

    def create_predictor(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> ExtraInputPredictorPort:
        """Return the shared recording predictor instance.

        Args:
            checkpoint_path: Ignored in mock.
            config_path: Ignored in mock.
            data_config_path: Ignored in mock.

        Returns:
            The shared _RecordingPredictor instance.
        """
        return self.predictor


@pytest.fixture
def adapter() -> _MockAdapter:
    """Fixture: mock adapter with recording predictor."""
    return _MockAdapter()


@pytest.fixture
def dummy_checkpoint(tmp_path: Path) -> Path:
    """Fixture: dummy checkpoint file path."""
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.touch()
    return checkpoint


@pytest.fixture
def neural_with_matrix(adapter: _MockAdapter, dummy_checkpoint: Path) -> NeuralPreconditioner:
    """Fixture: NeuralPreconditioner expecting a matrix extra input."""
    return NeuralPreconditioner(
        checkpoint_path=dummy_checkpoint,
        adapter=adapter,
        extra_input_names=("matrix",),
    )


@pytest.fixture
def small_matrix() -> torch.Tensor:
    """Fixture: small identity-like matrix."""
    return torch.eye(3, dtype=torch.float64) * 2.0


def test_extra_input_names_stored(neural_with_matrix: NeuralPreconditioner) -> None:
    """Test that extra_input_names property returns declared inputs."""
    assert neural_with_matrix.extra_input_names == ("matrix",)


def test_bind_inputs_filters_to_declared_names(
    adapter: _MockAdapter, neural_with_matrix: NeuralPreconditioner, small_matrix: torch.Tensor
) -> None:
    """Test that bind_inputs only keeps keys declared in extra_input_names."""
    neural_with_matrix.bind_inputs(matrix=small_matrix, ignored_key=torch.ones(3))
    neural_with_matrix.apply(torch.ones(3))
    extra = adapter.predictor.extras[-1]
    assert "matrix" in extra
    assert "ignored_key" not in extra


def test_apply_without_bind_sends_no_extras(
    adapter: _MockAdapter, dummy_checkpoint: Path, residual: torch.Tensor
) -> None:
    """Test that apply sends empty extras if bind_inputs was not called."""
    precond = NeuralPreconditioner(
        checkpoint_path=dummy_checkpoint,
        adapter=adapter,
        extra_input_names=("matrix",),
    )
    precond.apply(residual)
    assert adapter.predictor.extras[-1] == {}


def test_apply_after_bind_forwards_extra(
    adapter: _MockAdapter,
    neural_with_matrix: NeuralPreconditioner,
    residual: torch.Tensor,
    small_matrix: torch.Tensor,
) -> None:
    """Test that apply forwards bound extra inputs to the predictor."""
    neural_with_matrix.bind_inputs(matrix=small_matrix)
    neural_with_matrix.apply(residual)
    extra = adapter.predictor.extras[-1]
    assert torch.equal(extra["matrix"], small_matrix)


def test_no_extra_input_names_by_default(adapter: _MockAdapter, dummy_checkpoint: Path) -> None:
    """Test that NeuralPreconditioner has empty extra_input_names by default."""
    precond = NeuralPreconditioner(checkpoint_path=dummy_checkpoint, adapter=adapter)
    assert precond.extra_input_names == ()


def test_extra_input_names_fall_back_to_predictor_required_inputs(dummy_checkpoint: Path) -> None:
    """NeuralPreconditioner exposes predictor-declared extras without redeclaration."""
    adapter = _MockAdapter(required_inputs=("matrix",))
    precond = NeuralPreconditioner(checkpoint_path=dummy_checkpoint, adapter=adapter)

    assert precond.extra_input_names == ("matrix",)


def test_bind_inputs_keeps_predictor_required_inputs(
    dummy_checkpoint: Path,
    residual: torch.Tensor,
    small_matrix: torch.Tensor,
) -> None:
    """Binding honors predictor-declared extras when constructor names are omitted."""
    adapter = _MockAdapter(required_inputs=("matrix",))
    precond = NeuralPreconditioner(checkpoint_path=dummy_checkpoint, adapter=adapter)

    precond.bind_inputs(matrix=small_matrix, ignored_key=torch.ones(3))
    precond.apply(residual)

    assert adapter.predictor.extras[-1] == {"matrix": small_matrix}
