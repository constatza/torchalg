"""Tests for PredictorPort and ExtraInputPredictorPort contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchalg.preconditioners.ports import ExtraInputPredictorPort, PredictorPort

if TYPE_CHECKING:
    from tests.solver.preconditioners.conftest import _CapturingPredictor


def test_base_port_apply_no_extras(
    minimal_predictor: PredictorPort, residual: torch.Tensor
) -> None:
    """Base PredictorPort.apply works with residual only."""
    result = minimal_predictor.apply(residual)
    assert torch.equal(result, residual)


def test_base_port_is_context_manager(
    minimal_predictor: PredictorPort, residual: torch.Tensor
) -> None:
    """Base PredictorPort supports context manager protocol."""
    with minimal_predictor as pred:
        result = pred.apply(residual)
    assert torch.equal(result, residual)


def test_apply_extra_inputs_forwarded(
    capturing_predictor: _CapturingPredictor, residual: torch.Tensor
) -> None:
    """ExtraInputPredictorPort.apply forwards named extra tensors."""
    extra = torch.eye(4, dtype=residual.dtype)
    capturing_predictor.apply(residual, matrix=extra)
    assert "matrix" in capturing_predictor.last_extra
    assert torch.equal(capturing_predictor.last_extra["matrix"], extra)


def test_apply_no_extra_inputs_still_works(
    capturing_predictor: _CapturingPredictor, residual: torch.Tensor
) -> None:
    """ExtraInputPredictorPort.apply works with no extra inputs."""
    result = capturing_predictor.apply(residual)
    assert torch.equal(result, residual)
    assert capturing_predictor.last_extra == {}


def test_extra_input_port_is_subtype_of_base_port(
    capturing_predictor: _CapturingPredictor,
) -> None:
    """ExtraInputPredictorPort is a proper subtype of PredictorPort."""
    assert isinstance(capturing_predictor, PredictorPort)
    assert isinstance(capturing_predictor, ExtraInputPredictorPort)
