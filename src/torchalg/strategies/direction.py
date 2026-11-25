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
from dataclasses import replace

import torch

from torchalg.models.state import CGState
from torchalg.strategies.orthogonalization import (
    OrthogonalizationReport,
    OrthogonalizationStrategy,
)


class DirectionStrategy(ABC):
    """Strategy for computing a search direction from a preconditioned residual."""

    @property
    @abstractmethod
    def required_history_size(self) -> int | None:
        """Return the direction-history window this strategy requires."""

    @abstractmethod
    def compute_direction(
        self,
        w: torch.Tensor,
        rw: torch.Tensor,
        state: CGState,
        *,
        overflow_safe_retry: bool = False,
    ) -> tuple[torch.Tensor, OrthogonalizationReport | None]:
        """Compute the search direction for the current state.

        Args:
            w: Preconditioned residual ``w_k = M^{-1} r_k``.
            rw: ``(r_k, w_k)``, already computed by the solver step.
            state: Current solver state. At ``state.iteration == 0`` the
                direction must be a copy of ``w`` (there is nothing to
                conjugate against); the solver reuses the precomputed
                ``state.q = A w`` for that first direction.
            overflow_safe_retry: Repeat the previous call, whose report had
                ``projections_finite`` false, with overflow-safe projections
                and without advancing a stateful orthogonalization (see
                ``OrthogonalizationStrategy.orthogonalize``).

        Returns:
            tuple[torch.Tensor, OrthogonalizationReport | None]:
                ``(direction, report)`` - the orthogonalization report, whose
                ``breakdown`` is ``True`` iff this step reported a
                near-zero/degenerate result and whose ``projections_finite``
                is ``False`` iff the direction must be recomputed with
                ``overflow_safe_retry``; or ``None`` for strategies that
                perform no orthogonalization (nothing to report, and no
                per-iteration flag allocation).
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

    def compute_direction(
        self,
        w: torch.Tensor,
        rw: torch.Tensor,
        state: CGState,
        *,
        overflow_safe_retry: bool = False,
    ) -> tuple[torch.Tensor, None]:
        """Compute ``d_k = w_k + beta_k d_{k-1}``. Never orthogonalizes.

        Takes no projection, so ``overflow_safe_retry`` changes nothing.

        Cost:
            One fused ``addcmul`` launch with no ``beta * d`` temporary, instead
            of a multiply and an add: ``2n`` flops and ``3Vn`` bytes (``V = 8``
            per ``float64``) against ``5Vn`` for the two-kernel form, one
            launch fewer per iteration, with the same ``w + beta d`` out of
            place so autograd sees no in-place write. ``addcmul`` may contract
            the multiply-add, so the result agrees with the unfused form to
            rounding. The first iteration is one copy of ``w``.
        """
        if state.iteration == 0:
            return w.clone(), None

        beta = rw / state.rw_prev
        return torch.addcmul(w, beta, state.d), None


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

    def compute_direction(
        self,
        w: torch.Tensor,
        rw: torch.Tensor,
        state: CGState,
        *,
        overflow_safe_retry: bool = False,
    ) -> tuple[torch.Tensor, OrthogonalizationReport]:
        """Orthogonalize ``w`` against stored directions/products."""
        return self.orthogonalization.orthogonalize(
            vector=w,
            window=state.direction_history.window(like=w),
            overflow_safe_retry=overflow_safe_retry,
        )


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

    def compute_direction(
        self,
        w: torch.Tensor,
        rw: torch.Tensor,
        state: CGState,
        *,
        overflow_safe_retry: bool = False,
    ) -> tuple[torch.Tensor, OrthogonalizationReport | None]:
        """Compute base direction and optionally reorthogonalize it.

        A retry is passed to both stages; the combined report breaks down
        if either stage did and has finite projections only if both have.
        """
        direction, base_report = self.base_strategy.compute_direction(
            w, rw, state, overflow_safe_retry=overflow_safe_retry
        )
        if self.reorthog_strategy is None:
            return direction, base_report
        reorthogonalized, report = self.reorthog_strategy.orthogonalize(
            vector=direction,
            window=state.direction_history.window(like=direction),
            overflow_safe_retry=overflow_safe_retry,
        )
        if base_report is None:
            return reorthogonalized, report
        return reorthogonalized, replace(
            report,
            breakdown=base_report.breakdown | report.breakdown,
            projections_finite=base_report.projections_finite & report.projections_finite,
        )
