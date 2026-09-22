"""Tests for ``torchalg.monitoring.iteration_history``."""

from __future__ import annotations

import math

import torch

from torchalg.monitoring import IterationHistory, TraceMode


def test_minimal_mode_logs_residual_norms_only(
    minimal_trace_mode: TraceMode | str,
    residual_vector: torch.Tensor,
    solution_vector: torch.Tensor,
    direction_vector: torch.Tensor,
) -> None:
    """MINIMAL mode keeps scalar norms and ignores vectors."""
    history = IterationHistory(mode=minimal_trace_mode)

    history.log_iteration(
        residual_norm=2.0,
        residual=residual_vector,
        solution=solution_vector,
        direction=direction_vector,
    )
    history.log_iteration(residual_norm=1.0)

    assert history.mode == TraceMode.MINIMAL
    assert history.residual_norms.to_list() == [2.0, 1.0]
    assert history.residuals is None
    assert history.solutions is None
    assert history.directions is None
    assert history.iteration_count() == 2


def test_full_mode_logs_vectors(
    residual_vector: torch.Tensor,
    solution_vector: torch.Tensor,
    direction_vector: torch.Tensor,
) -> None:
    """FULL mode records scalar residual norms and supplied vectors."""
    history = IterationHistory(mode=TraceMode.FULL)

    history.log_iteration(
        residual_norm=torch.linalg.norm(residual_vector).item(),
        residual=residual_vector,
        solution=solution_vector,
        direction=direction_vector,
    )

    assert history.residuals is not None
    assert history.solutions is not None
    assert history.directions is not None
    assert torch.equal(history.residuals.to_tensor(), residual_vector.unsqueeze(0))
    assert torch.equal(history.solutions.to_tensor(), solution_vector.unsqueeze(0))
    assert torch.equal(history.directions.to_tensor(), direction_vector.unsqueeze(0))


def test_full_mode_skips_missing_vectors(residual_vector: torch.Tensor) -> None:
    """FULL mode permits logging only the vectors available for an iteration."""
    history = IterationHistory(mode=TraceMode.FULL)

    history.log_iteration(residual_norm=1.0, residual=residual_vector)

    assert history.residuals is not None
    assert history.solutions is not None
    assert history.directions is not None
    assert len(history.residuals) == 1
    assert len(history.solutions) == 0
    assert len(history.directions) == 0


def test_x_exact_populates_error_norms_in_minimal_mode(
    minimal_trace_mode: TraceMode | str,
    residual_vector: torch.Tensor,
    solution_vector: torch.Tensor,
    x_exact_vector: torch.Tensor,
) -> None:
    """Supplying ``x_exact`` records the exact A-norm error, even under MINIMAL."""
    history = IterationHistory(mode=minimal_trace_mode, x_exact=x_exact_vector)

    history.log_iteration(
        residual_norm=1.0,
        residual=residual_vector,
        solution=solution_vector,
    )

    expected = math.sqrt(
        max(float(torch.dot(x_exact_vector - solution_vector, residual_vector)), 0.0)
    )
    assert history.error_norms.to_list() == [expected]
    assert history.residuals is None
    assert history.solutions is None


def test_without_x_exact_error_norms_stays_empty(
    minimal_trace_mode: TraceMode | str,
    residual_vector: torch.Tensor,
    solution_vector: torch.Tensor,
) -> None:
    """No ``x_exact`` means ``error_norms`` never gets populated."""
    history = IterationHistory(mode=minimal_trace_mode)

    history.log_iteration(residual_norm=1.0, residual=residual_vector, solution=solution_vector)

    assert history.error_norms.to_list() == []


def test_energy_decrement_recorded_regardless_of_mode(minimal_trace_mode: TraceMode | str) -> None:
    """``energy_decrement`` is recorded even under MINIMAL - no vectors involved."""
    history = IterationHistory(mode=minimal_trace_mode)

    history.log_iteration(residual_norm=1.0, energy_decrement=0.25)
    history.log_iteration(residual_norm=0.5, energy_decrement=0.1)

    assert history.energy_decrements.to_list() == [0.25, 0.1]


def test_disabled_mode_does_not_log() -> None:
    """DISABLED mode is a no-op container for callers that still hold a history."""
    history = IterationHistory(mode=TraceMode.DISABLED)

    history.log_iteration(residual_norm=1.0)

    assert history.residual_norms.to_list() == []
    assert history.residuals is None
    assert history.solutions is None
    assert history.directions is None
    assert history.iteration_count() == 0
