"""Direction computation strategies for CG-family solvers.

References:
    - Hestenes, M.R. & Stiefel, E. (1952). Methods of Conjugate Gradients
      for Solving Linear Systems. J. Res. Natl. Bur. Stand. 49(6), 409-436.
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd ed.
      SIAM. Ch. 9 (Algorithm 9.1, preconditioned CG two-term recurrence).
    - Notay, Y. (2000). Flexible Conjugate Gradients. SIAM J. Sci. Comput.
      22(4), 1444-1460.
"""

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
    def compute_direction(self, w: torch.Tensor, state: CGState) -> tuple[torch.Tensor, bool]:
        """Compute the search direction for the current state.

        Returns:
            tuple[torch.Tensor, bool]: ``(direction, ortho_breakdown)`` -
                ``ortho_breakdown`` is ``True`` iff this step's
                orthogonalization (if any) reported a near-zero/degenerate
                result; strategies with no orthogonalization always return
                ``False``.
        """


class TwoTermRecurrenceStrategy(DirectionStrategy):
    """Standard PCG two-term recurrence.

    ``beta_k = (r_k, w_k) / (r_{k-1}, w_{k-1})``,
    ``d_k = w_k + beta_k d_{k-1})`` - Saad (2003) Algorithm 9.1, the
    preconditioned generalization of Hestenes & Stiefel (1952)'s
    Fletcher-Reeves-form recurrence (reduces to plain Fletcher-Reeves when
    ``M = I``). Requires no direction history and is exact only for a
    fixed SPD preconditioner ``M``; see ``OrthogonalizationDirectionStrategy``
    for the Flexible CG alternative.
    """

    def __init__(self, beta_formula: str = "fletcher_reeves") -> None:
        """Initialize with the beta formula name."""
        if beta_formula != "fletcher_reeves":
            raise ValueError(f"Unsupported beta formula: {beta_formula}")
        self.beta_formula = beta_formula

    @property
    def required_history_size(self) -> int:
        """Two-term recurrence requires no direction history."""
        return 0

    def compute_direction(self, w: torch.Tensor, state: CGState) -> tuple[torch.Tensor, bool]:
        """Compute ``d_k = w_k + beta_k d_{k-1}``. Never orthogonalizes."""
        if state.iteration == 0:
            return w.clone(), False

        rw_curr = stable_dot_product(state.r, w)
        beta = rw_curr / state.rw_prev
        return w + beta * state.d, False


class OrthogonalizationDirectionStrategy(DirectionStrategy):
    """FCG direction strategy using explicit A-conjugacy orthogonalization.

    Notay (2000) FCG(m): rather than a two-term recurrence, ``d_k`` is the
    preconditioned residual ``w_k`` after explicit Gram-Schmidt
    A-conjugation against up to ``m`` previously stored ``(d_j, q_j)``
    pairs (delegated to the injected ``OrthogonalizationStrategy`` -
    ``PeriodicRestartOrthogonalization`` implements Notay's own periodic
    window; ``TruncatedGramSchmidt``/``ModifiedGramSchmidt`` are simpler
    fixed/full-history variants). This is the general, multi-term FCG(m)
    from the paper - not the cheaper two-term Polak-Ribiere-type shortcut
    Notay derives for the ``m=1`` case, which is not implemented here.
    """

    def __init__(self, orthogonalization: OrthogonalizationStrategy) -> None:
        """Initialize with an orthogonalization strategy."""
        self.orthogonalization = orthogonalization

    @property
    def required_history_size(self) -> int | None:
        """Return the wrapped orthogonalization window size."""
        return self.orthogonalization.window_size

    def compute_direction(self, w: torch.Tensor, state: CGState) -> tuple[torch.Tensor, bool]:
        """Orthogonalize ``w`` against stored directions/products."""
        direction, report = self.orthogonalization.orthogonalize(
            vector=w,
            d_vectors=state.direction_history.d_vectors,
            q_vectors=state.direction_history.q_vectors,
        )
        return direction, report.breakdown


class CompositeDirectionStrategy(DirectionStrategy):
    """Base direction strategy with optional post reorthogonalization.

    Wraps ``base_strategy`` (typically ``TwoTermRecurrenceStrategy``) and,
    when ``reorthog_strategy`` is given, A-conjugates its output against
    stored history afterward - Notay (2000)'s periodic reorthogonalization
    layered on top of PCG, to counter loss of A-conjugacy from accumulated
    rounding error over long solves without switching to full FCG.
    """

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

    def compute_direction(self, w: torch.Tensor, state: CGState) -> tuple[torch.Tensor, bool]:
        """Compute base direction and optionally reorthogonalize it."""
        direction, base_breakdown = self.base_strategy.compute_direction(w, state)
        if self.reorthog_strategy is None:
            return direction, base_breakdown
        reorthogonalized, report = self.reorthog_strategy.orthogonalize(
            vector=direction,
            d_vectors=state.direction_history.d_vectors,
            q_vectors=state.direction_history.q_vectors,
        )
        return reorthogonalized, base_breakdown or report.breakdown
