"""Direct unit tests for the neural preconditioner."""

from __future__ import annotations

import gc

import pytest
import torch

from torchalg.preconditioners.implementations.neural import NeuralPreconditioner
from torchalg.preconditioners.ports import ExtraInputPredictorPort


class DummyPredictor(ExtraInputPredictorPort):
    """Minimal predictor stub for direct neural preconditioner tests."""

    def __init__(self) -> None:
        """Initialize with cleanup flag."""
        self.cleaned_up = False

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """Return empty tuple - this stub needs no extra inputs.

        Returns:
            Empty tuple of required input names.
        """
        return ()

    def apply(self, residual: torch.Tensor, **extra_inputs: torch.Tensor) -> torch.Tensor:
        """Scale residual by 0.5 and ignore extra inputs.

        Args:
            residual: Input residual vector.
            **extra_inputs: Ignored extra inputs.

        Returns:
            Residual scaled by 0.5.
        """
        return residual * 0.5

    def cleanup(self) -> None:
        """Mark as cleaned up."""
        self.cleaned_up = True


@pytest.fixture
def predictor() -> DummyPredictor:
    """A predictor the caller owns."""
    return DummyPredictor()


def test_neural_preconditioner_applies_predictor_output(predictor: DummyPredictor) -> None:
    """NeuralPreconditioner delegates `apply` to the predictor it was given."""
    preconditioner = NeuralPreconditioner(predictor)

    residual = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64)
    result = preconditioner.apply(residual)

    torch.testing.assert_close(result, residual * 0.5)


def test_neural_preconditioner_is_ready_after_construction(predictor: DummyPredictor) -> None:
    """The predictor is injected, so there is nothing left for `setup()` to load."""
    preconditioner = NeuralPreconditioner(predictor)
    residual = torch.ones(3, dtype=torch.float64)

    before = preconditioner.apply(residual)
    after = preconditioner.setup().apply(residual)

    torch.testing.assert_close(before, after)


def test_neural_preconditioner_requires_flexible_cg(predictor: DummyPredictor) -> None:
    """NeuralPreconditioner is non-linear, so it requires Flexible CG."""
    assert NeuralPreconditioner(predictor).requires_flexible_cg is True


def test_neural_preconditioner_never_releases_the_predictor(predictor: DummyPredictor) -> None:
    """The caller owns the predictor: dropping a preconditioner must not unload it."""
    preconditioner = NeuralPreconditioner(predictor)

    del preconditioner
    gc.collect()

    assert predictor.cleaned_up is False


def test_one_predictor_can_back_several_preconditioners(predictor: DummyPredictor) -> None:
    """Sharing is safe because no preconditioner releases the predictor on its own."""
    first = NeuralPreconditioner(predictor)
    second = NeuralPreconditioner(predictor)
    residual = torch.ones(2, dtype=torch.float64)

    del first
    gc.collect()

    torch.testing.assert_close(second.apply(residual), residual * 0.5)
