"""Unit tests for predictor adapters and ports compliance.

Tests the adapter contract, resource management, and error boundaries.
Focuses on our own abstractions (``PredictorAdapter``/``ExtraInputPredictorPort``)
- does not test any specific model-loading framework's internals.

Ported from the reference (``dl-experiments``'s
``tests/solver/preconditioners/test_adapters.py``), with one omission: the
reference's ``test_adapter_can_be_injected_into_factory`` exercises the
TOML-config-driven composition factory (``NeuralPreconditionerConfig``,
``create_preconditioner``), which ``docs/plan.md``'s scope trim explicitly
excludes from this migration - that factory has no equivalent in
``torchalg``. DIP-style "adapter injected via constructor" coverage is kept
via ``NeuralPreconditioner`` itself in ``test_neural.py``.

Follows project principles:
    - Use fixtures for all test data.
    - Use ``tmp_path`` for temporary files (never ``tempfile``).
    - Type hints throughout.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torchalg.preconditioners.ports import ExtraInputPredictorPort, PredictorAdapter

# ==============================================================================
# Mock Predictor and Adapter
# ==============================================================================


class MockPredictor(ExtraInputPredictorPort):
    """Mock predictor for testing port contract."""

    def __init__(self) -> None:
        """Initialize mock predictor."""
        self.cleaned_up = False
        self.apply_count = 0

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Return empty tuple - this mock needs no extra inputs.

        Returns:
            Empty tuple of required input names.
        """
        return ()

    def apply(self, residual: torch.Tensor, **extra_inputs: torch.Tensor) -> torch.Tensor:
        """Mock apply that scales residual by 0.5.

        Args:
            residual: Input residual vector.
            **extra_inputs: Ignored extra inputs.

        Returns:
            Residual scaled by 0.5.
        """
        self.apply_count += 1
        return residual * 0.5

    def cleanup(self) -> None:
        """Mark as cleaned up."""
        self.cleaned_up = True

    def __enter__(self) -> MockPredictor:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager with cleanup."""
        self.cleanup()


class MockAdapter(PredictorAdapter):
    """Mock adapter for testing adapter contract."""

    def __init__(self, predictor: MockPredictor | None = None) -> None:
        """Initialize with optional predictor."""
        self.predictor = predictor or MockPredictor()

    def create_predictor(
        self,
        checkpoint_path: Path,
        config_path: Path | None = None,
        data_config_path: Path | None = None,
    ) -> ExtraInputPredictorPort:
        """Create predictor, validating checkpoint exists.

        Args:
            checkpoint_path: Path to checkpoint file (must exist).
            config_path: Unused.
            data_config_path: Unused.

        Returns:
            The shared MockPredictor instance.

        Raises:
            FileNotFoundError: If checkpoint does not exist.
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return self.predictor


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mock_checkpoint(tmp_path: Path) -> Path:
    """Create mock checkpoint file.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to mock checkpoint file
    """
    checkpoint = tmp_path / "test_checkpoint.ckpt"
    checkpoint.write_text("mock checkpoint data")
    return checkpoint


@pytest.fixture
def mock_predictor() -> MockPredictor:
    """Create mock predictor instance.

    Returns:
        Fresh MockPredictor instance
    """
    return MockPredictor()


@pytest.fixture
def mock_adapter(mock_predictor: MockPredictor) -> MockAdapter:
    """Create mock adapter with predictor.

    Args:
        mock_predictor: MockPredictor fixture

    Returns:
        MockAdapter instance
    """
    return MockAdapter(predictor=mock_predictor)


# ==============================================================================
# Port Contract Tests
# ==============================================================================


def test_predictor_context_manager_calls_cleanup(mock_predictor: MockPredictor) -> None:
    """Test that context manager protocol calls cleanup on exit."""
    assert not mock_predictor.cleaned_up

    with mock_predictor:
        assert not mock_predictor.cleaned_up  # Not cleaned yet

    # Should be cleaned after exit
    assert mock_predictor.cleaned_up


def test_predictor_apply_returns_tensor_with_correct_shape(
    mock_predictor: MockPredictor, residual_vector: torch.Tensor
) -> None:
    """Test that apply returns a tensor with correct shape and dtype."""
    result = mock_predictor.apply(residual_vector)

    assert isinstance(result, torch.Tensor)
    assert result.shape == residual_vector.shape
    assert result.dtype == residual_vector.dtype

    expected = residual_vector * 0.5
    torch.testing.assert_close(result, expected)


def test_predictor_supports_multiple_apply_calls(
    mock_predictor: MockPredictor, residual_vector: torch.Tensor
) -> None:
    """Test that predictor can be used multiple times before cleanup."""
    result1 = mock_predictor.apply(residual_vector)
    result2 = mock_predictor.apply(residual_vector * 2)
    result3 = mock_predictor.apply(residual_vector * 3)

    assert mock_predictor.apply_count == 3

    torch.testing.assert_close(result1, residual_vector * 0.5)
    torch.testing.assert_close(result2, residual_vector * 2 * 0.5)
    torch.testing.assert_close(result3, residual_vector * 3 * 0.5)


# ==============================================================================
# Adapter Integration Tests
# ==============================================================================


def test_adapter_validates_checkpoint_exists(
    mock_adapter: MockAdapter, mock_checkpoint: Path, residual_vector: torch.Tensor
) -> None:
    """Test that adapter validates checkpoint file exists."""
    predictor = mock_adapter.create_predictor(mock_checkpoint)
    result = predictor.apply(residual_vector)

    torch.testing.assert_close(result, residual_vector * 0.5)


def test_adapter_handles_missing_checkpoint_file(
    mock_adapter: MockAdapter,
    tmp_path: Path,
) -> None:
    """Test that adapter raises FileNotFoundError for missing checkpoint."""
    nonexistent_path = tmp_path / "missing" / "checkpoint.ckpt"

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        mock_adapter.create_predictor(nonexistent_path)
