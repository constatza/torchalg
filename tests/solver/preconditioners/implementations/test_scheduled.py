"""Tests for scheduled preconditioner functionality.

This module tests the ``ScheduledPreconditioner`` class, which switches
between a primary and fallback preconditioner based on ``start_iter``/
``limit_iters``.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.base import PreconditionerContext
from torchalg.preconditioners.implementations import (
    Identity,
    JacobiPreconditioner,
    ScheduledPreconditioner,
)


@pytest.fixture
def diagonal_matrix(torch_dtype: torch.dtype) -> torch.Tensor:
    """Provide a diagonal matrix for Jacobi schedule tests.

    Returns:
        torch.Tensor: ``diag([2.0, 2.0])``.
    """
    return torch.diag(torch.tensor([2.0, 2.0], dtype=torch_dtype))


@pytest.fixture
def residual_vector(torch_dtype: torch.dtype) -> torch.Tensor:
    """Provide a residual vector for schedule tests.

    Returns:
        torch.Tensor: ``[2.0, 4.0]``.
    """
    return torch.tensor([2.0, 4.0], dtype=torch_dtype)


def test_scheduled_preconditioner_limit_iters(
    diagonal_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
) -> None:
    """Verify ScheduledPreconditioner switches at limit_iters."""
    primary = JacobiPreconditioner(diagonal_matrix)
    fallback = Identity()
    scheduled = ScheduledPreconditioner(primary, fallback, limit_iters=5)

    # Before limit: use primary (Jacobi: z = r / diag = [2/2, 4/2] = [1, 2]).
    ctx = PreconditionerContext(iteration=3, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(residual_vector, ctx)
    torch.testing.assert_close(z, torch.tensor([1.0, 2.0], dtype=residual_vector.dtype))

    # At limit: switch to fallback (Identity: z = r = [2, 4]).
    ctx = PreconditionerContext(iteration=5, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(residual_vector, ctx)
    torch.testing.assert_close(z, residual_vector)

    # After limit: still fallback.
    ctx = PreconditionerContext(iteration=10, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(residual_vector, ctx)
    torch.testing.assert_close(z, residual_vector)


def test_scheduled_preconditioner_default_fallback(
    diagonal_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
) -> None:
    """Verify ScheduledPreconditioner uses Identity as default fallback."""
    primary = JacobiPreconditioner(diagonal_matrix)
    scheduled = ScheduledPreconditioner(primary, fallback=None, limit_iters=2)

    # After limit: should use Identity (default fallback).
    ctx = PreconditionerContext(iteration=5, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(residual_vector, ctx)
    torch.testing.assert_close(z, residual_vector)


def test_scheduled_preconditioner_no_limit(
    diagonal_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
) -> None:
    """Verify ScheduledPreconditioner with no limit always uses primary."""
    primary = JacobiPreconditioner(diagonal_matrix)
    fallback = Identity()
    scheduled = ScheduledPreconditioner(primary, fallback, limit_iters=None)

    expected = torch.tensor([1.0, 2.0], dtype=residual_vector.dtype)
    for i in [0, 5, 10, 100]:
        ctx = PreconditionerContext(iteration=i, residual_norm=1.0, rhs_norm=1.0)
        z = scheduled.apply(residual_vector, ctx)
        torch.testing.assert_close(z, expected)


def test_scheduled_preconditioner_requires_context(
    diagonal_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
) -> None:
    """Verify ScheduledPreconditioner raises error when context is None."""
    primary = JacobiPreconditioner(diagonal_matrix)
    scheduled = ScheduledPreconditioner(primary, limit_iters=5)

    with pytest.raises(ValueError, match="requires context"):
        scheduled.apply(residual_vector, context=None)


def test_preconditioner_context_immutable() -> None:
    """Verify PreconditionerContext is immutable (frozen dataclass)."""
    ctx = PreconditionerContext(iteration=5, residual_norm=1.0, rhs_norm=2.0)

    with pytest.raises(AttributeError):
        ctx.iteration = 10  # ty: ignore[invalid-assignment]

    with pytest.raises(AttributeError):
        ctx.residual_norm = 5.0  # ty: ignore[invalid-assignment]


def test_scheduled_preconditioner_iteration_zero(
    diagonal_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
) -> None:
    """Test that iteration=0 uses primary preconditioner when limit_iters > 0."""
    primary = JacobiPreconditioner(diagonal_matrix)
    fallback = Identity()
    scheduled = ScheduledPreconditioner(primary, fallback, limit_iters=5)

    ctx = PreconditionerContext(iteration=0, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(residual_vector, ctx)
    torch.testing.assert_close(z, torch.tensor([1.0, 2.0], dtype=residual_vector.dtype))


def test_scheduled_preconditioner_uses_fallback_before_start_iter(
    diagonal_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
) -> None:
    """Verify delayed schedules use fallback before start_iter."""
    primary = JacobiPreconditioner(diagonal_matrix)
    fallback = Identity()
    scheduled = ScheduledPreconditioner(primary, fallback, limit_iters=None, start_iter=3)

    ctx = PreconditionerContext(iteration=2, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(residual_vector, ctx)

    torch.testing.assert_close(z, residual_vector)


def test_scheduled_preconditioner_starts_primary_at_start_iter(
    diagonal_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
) -> None:
    """Verify delayed schedules activate primary at start_iter."""
    primary = JacobiPreconditioner(diagonal_matrix)
    fallback = Identity()
    scheduled = ScheduledPreconditioner(primary, fallback, limit_iters=None, start_iter=3)

    ctx = PreconditionerContext(iteration=3, residual_norm=1.0, rhs_norm=1.0)
    z = scheduled.apply(residual_vector, ctx)

    torch.testing.assert_close(z, torch.tensor([1.0, 2.0], dtype=residual_vector.dtype))


def test_scheduled_preconditioner_delayed_limit_switches_back_to_fallback(
    diagonal_matrix: torch.Tensor,
    residual_vector: torch.Tensor,
) -> None:
    """Verify limit_iters is measured from start_iter."""
    primary = JacobiPreconditioner(diagonal_matrix)
    fallback = Identity()
    scheduled = ScheduledPreconditioner(primary, fallback, limit_iters=2, start_iter=3)

    active_ctx = PreconditionerContext(iteration=4, residual_norm=1.0, rhs_norm=1.0)
    inactive_ctx = PreconditionerContext(iteration=5, residual_norm=1.0, rhs_norm=1.0)

    torch.testing.assert_close(
        scheduled.apply(residual_vector, active_ctx),
        torch.tensor([1.0, 2.0], dtype=residual_vector.dtype),
    )
    torch.testing.assert_close(scheduled.apply(residual_vector, inactive_ctx), residual_vector)


def test_scheduled_preconditioner_rejects_negative_start_iter() -> None:
    """Verify delayed schedules reject negative activation iterations."""
    with pytest.raises(ValueError, match="start_iter"):
        ScheduledPreconditioner(Identity(), start_iter=-1)


def test_scheduled_preconditioner_rejects_negative_limit_iters() -> None:
    """Verify bounded schedules reject negative active windows."""
    with pytest.raises(ValueError, match="limit_iters"):
        ScheduledPreconditioner(Identity(), limit_iters=-1)


def test_primary_property_matches_constructor_arg(diagonal_matrix: torch.Tensor) -> None:
    """`primary` property returns exactly the preconditioner passed to `__init__`."""
    primary = JacobiPreconditioner(diagonal_matrix)
    scheduled = ScheduledPreconditioner(primary=primary, limit_iters=10, start_iter=0)
    assert scheduled.primary is primary
