"""Direction computation strategies for CG-family solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from torchalg.models.state import CGState
from torchalg.strategies.orthogonalization import OrthogonalizationStrategy
from torchalg.utils.numerics import stable_dot_product


class DirectionStrategy(ABC):
    """Strategy for computing a search direction from a preconditioned residual."""

    @property
    @abstractmethod
    def required_history_size(self) -> int | None:
        """Return the direction-history window this strategy requires."""

    @abstractmethod
    def compute_direction(self, w: torch.Tensor, state: CGState) -> torch.Tensor:
        """Compute the search direction for the current state."""


class TwoTermRecurrenceStrategy(DirectionStrategy):
    """Standard PCG two-term recurrence."""

    def __init__(self, beta_formula: str = "fletcher_reeves") -> None:
        """Initialize with the beta formula name."""
        if beta_formula != "fletcher_reeves":
            raise ValueError(f"Unsupported beta formula: {beta_formula}")
        self.beta_formula = beta_formula

    @property
    def required_history_size(self) -> int:
        """Two-term recurrence requires no direction history."""
        return 0

    def compute_direction(self, w: torch.Tensor, state: CGState) -> torch.Tensor:
        """Compute ``d_k = w_k + beta_k d_{k-1}``."""
        if state.iteration == 0:
            return w.clone()

        rw_curr = stable_dot_product(state.r, w)
        beta = rw_curr / state.rw_prev
        return w + beta * state.d


class OrthogonalizationDirectionStrategy(DirectionStrategy):
    """FCG direction strategy using explicit A-conjugacy orthogonalization."""

    def __init__(self, orthogonalization: OrthogonalizationStrategy) -> None:
        """Initialize with an orthogonalization strategy."""
        self.orthogonalization = orthogonalization

    @property
    def required_history_size(self) -> int | None:
        """Return the wrapped orthogonalization window size."""
        return self.orthogonalization.window_size

    def compute_direction(self, w: torch.Tensor, state: CGState) -> torch.Tensor:
        """Orthogonalize ``w`` against stored directions/products."""
        direction, _ = self.orthogonalization.orthogonalize(
            vector=w,
            d_vectors=state.direction_history.d_vectors,
            q_vectors=state.direction_history.q_vectors,
        )
        return direction


class CompositeDirectionStrategy(DirectionStrategy):
    """Base direction strategy with optional post reorthogonalization."""

    def __init__(
        self,
        base_strategy: DirectionStrategy,
        reorthog_strategy: OrthogonalizationStrategy | None = None,
    ) -> None:
        """Initialize composite direction computation."""
        self.base_strategy = base_strategy
        self.reorthog_strategy = reorthog_strategy

    @property
    def required_history_size(self) -> int | None:
        """Return reorthogonalization history size, or base requirement."""
        if self.reorthog_strategy is None:
            return self.base_strategy.required_history_size
        return self.reorthog_strategy.window_size

    def compute_direction(self, w: torch.Tensor, state: CGState) -> torch.Tensor:
        """Compute base direction and optionally reorthogonalize it."""
        direction = self.base_strategy.compute_direction(w, state)
        if self.reorthog_strategy is None:
            return direction
        reorthogonalized, _ = self.reorthog_strategy.orthogonalize(
            vector=direction,
            d_vectors=state.direction_history.d_vectors,
            q_vectors=state.direction_history.q_vectors,
        )
        return reorthogonalized
