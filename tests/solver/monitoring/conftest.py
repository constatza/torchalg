"""Fixtures for solver monitoring tests."""

from __future__ import annotations

import pytest
import torch

from torchalg.monitoring.trace_mode import TraceMode


@pytest.fixture
def residual_vector() -> torch.Tensor:
    """Fixture: representative residual vector."""
    return torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)


@pytest.fixture
def solution_vector() -> torch.Tensor:
    """Fixture: representative solution vector."""
    return torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)


@pytest.fixture
def direction_vector() -> torch.Tensor:
    """Fixture: representative search direction vector."""
    return torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)


@pytest.fixture
def x_exact_vector() -> torch.Tensor:
    """Fixture: known exact solution paired with ``solution_vector``/``residual_vector``."""
    return torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)


@pytest.fixture(params=[TraceMode.MINIMAL, "minimal"])
def minimal_trace_mode(request: pytest.FixtureRequest) -> TraceMode | str:
    """Fixture: minimal trace mode as enum and string input."""
    return request.param
