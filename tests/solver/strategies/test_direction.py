"""Tests for ``torchalg.strategies.direction``."""

from __future__ import annotations

from dataclasses import replace

import torch

from torchalg.models.history import DirectionHistory, ResidualHistory
from torchalg.models.state import CGState
from torchalg.strategies.direction import (
    OrthogonalizationDirectionStrategy,
    TwoTermRecurrenceStrategy,
)
from torchalg.strategies.orthogonalization import TruncatedGramSchmidt


def test_two_term_recurrence_first_iteration_returns_preconditioned_residual(
    two_term_preconditioned_residual: torch.Tensor,
    two_term_previous_direction: torch.Tensor,
) -> None:
    """The first PCG direction is the preconditioned residual."""
    state = CGState.create_initial(
        u=torch.zeros_like(two_term_preconditioned_residual),
        r=two_term_preconditioned_residual,
        w=two_term_preconditioned_residual,
        d=two_term_previous_direction,
        q=two_term_previous_direction,
        residual_norm=float(torch.linalg.norm(two_term_preconditioned_residual)),
        rhs_norm=1.0,
    )

    direction, ortho_breakdown = TwoTermRecurrenceStrategy().compute_direction(
        two_term_preconditioned_residual,
        state,
    )

    assert torch.equal(direction, two_term_preconditioned_residual)
    assert direction is not two_term_preconditioned_residual
    assert ortho_breakdown is False


def test_two_term_recurrence_uses_fletcher_reeves_beta(
    two_term_current_residual: torch.Tensor,
    two_term_preconditioned_residual: torch.Tensor,
    two_term_previous_direction: torch.Tensor,
) -> None:
    """Subsequent PCG directions use beta=(r_k,w_k)/(r_{k-1},w_{k-1})."""
    base_state = CGState.create_initial(
        u=torch.zeros_like(two_term_current_residual),
        r=two_term_current_residual,
        w=two_term_preconditioned_residual,
        d=two_term_previous_direction,
        q=two_term_previous_direction,
        residual_norm=float(torch.linalg.norm(two_term_current_residual)),
        rhs_norm=1.0,
    )
    state = replace(base_state, iteration=1, rw_prev=2.0)

    direction, ortho_breakdown = TwoTermRecurrenceStrategy().compute_direction(
        two_term_preconditioned_residual,
        state,
    )

    expected = two_term_preconditioned_residual + 0.5 * two_term_previous_direction
    assert torch.allclose(direction, expected)
    assert ortho_breakdown is False


def test_orthogonalization_direction_strategy_uses_direction_history(
    orthogonalization_probe: torch.Tensor,
    previous_direction: torch.Tensor,
    previous_matrix_product: torch.Tensor,
) -> None:
    """FCG direction strategy delegates to configured A-conjugacy orthogonalization."""
    history = DirectionHistory.empty(max_size=2).add(
        previous_direction,
        previous_matrix_product,
    )
    state = CGState(
        iteration=1,
        converged=False,
        breakdown=False,
        divergence=False,
        residual_norm=float(torch.linalg.norm(orthogonalization_probe)),
        rhs_norm=1.0,
        u=torch.zeros_like(orthogonalization_probe),
        r=orthogonalization_probe,
        w=orthogonalization_probe,
        d=previous_direction,
        q=previous_matrix_product,
        direction_history=history,
        residual_history=ResidualHistory.empty(),
        w_prev=None,
        r_prev=None,
        rw_prev=0.0,
    )

    direction, ortho_breakdown = OrthogonalizationDirectionStrategy(
        TruncatedGramSchmidt(window_size=2),
    ).compute_direction(orthogonalization_probe, state)

    assert torch.allclose(direction, torch.tensor([0.0, 1.0], dtype=direction.dtype))
    assert ortho_breakdown is False
