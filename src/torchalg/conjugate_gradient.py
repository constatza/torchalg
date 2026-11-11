"""Conjugate Gradient solver family.

``ConjugateGradientSolver`` is a single template-method CG loop
(``IterativeSolverBase``, see ``base.py``) parameterized by a
``DirectionStrategy``: ``PCGSolver`` plugs in the standard two-term
recurrence, ``FCGSolver`` plugs in explicit A-conjugacy orthogonalization.
Both share the same per-iteration update (``_iterate_step``) -
``w = M^{-1}r``, ``d = strategy(w)``, ``q = A d``,
``alpha = (r, w) / (d, q)``, then the standard CG state update - only the
direction computation differs. See ``factories.py``'s ``pcg()``/
``flexible_cg()`` for the public entry points and full algorithm writeups.

References:
    - Hestenes, M.R. & Stiefel, E. (1952). Methods of Conjugate Gradients
      for Solving Linear Systems. J. Res. Natl. Bur. Stand. 49(6), 409-436.
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd ed.
      SIAM. Ch. 6, Ch. 9.
    - Notay, Y. (2000). Flexible Conjugate Gradients. SIAM J. Sci. Comput.
      22(4), 1444-1460.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Self

import torch

from torchalg.base import DEFAULT_ATOL, DEFAULT_RTOL, IterativeSolverBase
from torchalg.models.diagnostics import TerminationDiagnostics
from torchalg.models.history import DirectionHistory
from torchalg.models.result import SolverResult
from torchalg.models.state import CGState, SolverState
from torchalg.monitoring import IterationHistory, TraceMode
from torchalg.preconditioners.base import Preconditioner, PreconditionerContext
from torchalg.strategies.convergence import (
    CombinedToleranceCriterion,
    IConvergenceCriterion,
)
from torchalg.strategies.direction import (
    CompositeDirectionStrategy,
    DirectionStrategy,
    OrthogonalizationDirectionStrategy,
    TwoTermRecurrenceStrategy,
)
from torchalg.strategies.orthogonalization import OrthogonalizationStrategy
from torchalg.utils.numerics import stable_dot_product
from torchalg.utils.validation import check_solution_validity, describe_breakdown_reason


def _record_first_breakdown(
    recorded: torch.Tensor | None,
    flag: torch.Tensor | None,
    iteration: int,
) -> torch.Tensor | None:
    """Fold this iteration's breakdown ``flag`` into the running first-breakdown record.

    Pure device-side ``where`` so the solve loop never synchronizes on it and
    never accumulates a per-iteration list. ``recorded`` holds ``0`` until a
    flag fires, then stays at that first 1-based iteration.
    """
    if flag is None:
        return recorded
    if recorded is None:
        recorded = torch.zeros((), dtype=torch.int64, device=flag.device)
    return torch.where(flag & (recorded == 0), iteration, recorded)


def _resolve_first_breakdown(recorded: torch.Tensor | None) -> int | None:
    """Read the first-breakdown record once, at the end of the solve."""
    if recorded is None:
        return None
    iteration = int(recorded)
    return iteration if iteration > 0 else None


class _IdentityPreconditioner(Preconditioner):
    """Internal identity preconditioner used when solvers are constructed directly."""

    def __init__(self) -> None:
        """Mark ready immediately; identity has no matrix-dependent state."""
        self._mark_ready()

    def setup(
        self,
        matrix: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> Self:
        """No-op: identity has no matrix-dependent state to build."""
        self._mark_ready()
        return self

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Return a clone of ``residual``."""
        return residual.clone()


class ConjugateGradientSolver(IterativeSolverBase[CGState]):
    """Common CG iteration engine parameterized by a direction strategy."""

    def __init__(
        self,
        direction_strategy: DirectionStrategy,
        preconditioner: Preconditioner | None = None,
        convergence_criterion: IConvergenceCriterion | None = None,
        iteration_history: IterationHistory | None = None,
        trace_mode: TraceMode = TraceMode.MINIMAL,
    ) -> None:
        """Initialize CG solver."""
        self.direction_strategy = direction_strategy
        self.preconditioner = preconditioner or _IdentityPreconditioner()
        self.convergence_criterion = convergence_criterion or CombinedToleranceCriterion(
            rtol=DEFAULT_RTOL,
            atol=DEFAULT_ATOL,
        )
        super().__init__(iteration_history=iteration_history, trace_mode=trace_mode)

    def _initialize_state(
        self,
        linear_op: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        x0: torch.Tensor | None,
        maxiter: int | None = None,
    ) -> CGState:
        """Create iteration-zero CG state.

        Holds the first preconditioned residual ``w_0 = M^{-1} r_0``, the first
        direction ``d_0 = w_0`` and ``q_0 = A d_0``; the first iteration
        consumes them instead of recomputing, so a ``k``-iteration solve costs
        ``k`` preconditioner applies and ``k`` matvecs (``k + 1`` with a
        nonzero ``x0``, whose residual needs ``A x0``).
        """
        if x0 is None:
            # b - A @ 0 == b exactly, so the zero-guess matvec is skipped.
            u = torch.zeros_like(b)
            r = b.clone()
        else:
            u = x0.clone()
            r = b - linear_op(u)
        residual_norm = torch.linalg.norm(r)
        rhs_norm = self.convergence_criterion.norm(b)
        initial_state = SolverState(
            iteration=0,
            converged=False,
            breakdown=False,
            divergence=False,
            residual_norm=residual_norm,
            rhs_norm=rhs_norm,
        )
        w = self._apply_preconditioner(self.preconditioner, r, initial_state)
        d = w.clone()
        q = linear_op(d)
        max_history = self._resolve_history_size(maxiter=maxiter, dimension=b.numel())
        return CGState.create_initial(
            u=u,
            r=r,
            w=w,
            d=d,
            q=q,
            residual_norm=residual_norm,
            rhs_norm=rhs_norm,
            max_history=max_history,
        )

    def _iterate_step(
        self,
        linear_op: Callable[[torch.Tensor], torch.Tensor],
        state: CGState,
    ) -> CGState:
        """Execute one unified CG iteration.

        The first iteration (``state.iteration == 0``) reuses the
        preconditioned residual and ``q = A d`` that ``_initialize_state``
        already computed from the same ``r``; the direction strategy is still
        invoked so its per-iteration bookkeeping (e.g. the periodic-restart
        window counter) advances identically, and by contract returns a copy
        of ``w`` at iteration 0.
        """
        first_iteration = state.iteration == 0
        w = (
            state.w
            if first_iteration
            else self._apply_preconditioner(self.preconditioner, state.r, state)
        )
        rw_curr = stable_dot_product(state.r, w)
        d, ortho_breakdown = self.direction_strategy.compute_direction(w, rw_curr, state)
        q = state.q if first_iteration else linear_op(d)
        curvature = stable_dot_product(d, q)
        alpha = rw_curr / curvature
        energy_decrement = alpha * rw_curr
        u_new = state.u + alpha * d
        r_new = state.r - alpha * q
        residual_norm_new = torch.linalg.norm(r_new)
        direction_history = self._updated_direction_history(state, d, q, curvature)
        new_iteration = state.iteration + 1

        return replace(
            state,
            iteration=new_iteration,
            residual_norm=residual_norm_new,
            u=u_new,
            r=r_new,
            w=w,
            d=d,
            q=q,
            direction_history=direction_history,
            rw_prev=rw_curr,
            ortho_breakdown_at=_record_first_breakdown(
                state.ortho_breakdown_at, ortho_breakdown, new_iteration
            ),
            energy_decrement=energy_decrement,
        )

    def _check_stopping(
        self,
        state: CGState,
        rtol: float,
        atol: float,
        maxiter: int,
    ) -> bool:
        """Stop on convergence, breakdown, divergence, or iteration limit."""
        if state.converged or state.breakdown or state.divergence:
            return True
        if state.iteration >= maxiter:
            return True
        return self._check_convergence_with_criterion(state, self.convergence_criterion)

    def _build_result(
        self,
        state: CGState,
        rtol: float,
        atol: float,
        maxiter: int | None = None,
    ) -> SolverResult:
        """Build immutable solver result from final state and telemetry."""
        solution_valid = check_solution_validity(state.u)
        converged = solution_valid and self._check_convergence_with_criterion(
            state,
            self.convergence_criterion,
        )
        # Regression: ``float(state.rhs_norm)`` emitted PyTorch's
        # gradient-tracked-scalar warning for ``differentiable=True`` solves.
        # SolverResult/history values are non-differentiable telemetry, so
        # detach them without altering the returned solution's graph.
        rhs_norm_for_history = (
            float(state.rhs_norm.detach())
            if isinstance(state.rhs_norm, torch.Tensor)
            else state.rhs_norm
        )
        (
            residual_abs_hist,
            residual_rel_hist,
            residual_vectors,
            solution_vectors,
            direction_vectors,
            error_history_a_norm,
            energy_decrements,
        ) = self._extract_histories_from_iteration_history(rhs_norm_for_history)

        ortho_breakdown_at = _resolve_first_breakdown(state.ortho_breakdown_at)

        breakdown = state.breakdown or not solution_valid
        diagnostics = self._diagnostics(state, converged, breakdown, ortho_breakdown_at)
        stopping_criterion = self._stopping_criterion(state, converged, breakdown, maxiter)

        # Convert tensor-or-float norms to float for SolverResult
        residual_norm_float = (
            float(state.residual_norm.detach())
            if isinstance(state.residual_norm, torch.Tensor)
            else state.residual_norm
        )
        rhs_norm_float = (
            float(state.rhs_norm.detach())
            if isinstance(state.rhs_norm, torch.Tensor)
            else state.rhs_norm
        )

        return SolverResult(
            converged=converged,
            iterations=state.iteration,
            residual=self._compute_relative_residual(residual_norm_float, rhs_norm_float),
            residual_abs=residual_norm_float,
            rhs_norm=rhs_norm_float,
            breakdown=breakdown,
            info=self._info_code(converged, breakdown, state, maxiter),
            residual_history_rel=residual_rel_hist,
            residual_history_abs=residual_abs_hist,
            tol=rtol,
            atol=atol,
            maxiter=maxiter,
            diagnostics=diagnostics,
            stopping_criterion=stopping_criterion,
            residual_vectors=residual_vectors,
            solution_vectors=solution_vectors,
            direction_vectors=direction_vectors,
            error_history_a_norm=error_history_a_norm,
            energy_decrements=energy_decrements,
        )

    def _resolve_history_size(self, *, maxiter: int | None, dimension: int) -> int:
        """Resolve direction-history capacity requested by the strategy."""
        required = self.direction_strategy.required_history_size
        if required is None:
            return maxiter if maxiter is not None else dimension
        return required

    def _updated_direction_history(
        self,
        state: CGState,
        direction: torch.Tensor,
        matrix_product: torch.Tensor,
        curvature: torch.Tensor,
    ) -> DirectionHistory:
        """Append ``(d, q, (d, q))`` unless the strategy needs no direction history."""
        required = self.direction_strategy.required_history_size
        if required is not None and required <= 0:
            return state.direction_history
        state.direction_history.append(direction, matrix_product, curvature)
        return state.direction_history

    def _compute_relative_residual[T: (float, torch.Tensor)](
        self, abs_residual: T, rhs_norm: float
    ) -> T:
        """Compute relative residual with zero-RHS fallback.

        Generic over the type of abs_residual (float or 0-d tensor).
        Preserves the input type in the output.

        Args:
            abs_residual: The absolute residual (float or 0-d tensor).
            rhs_norm: The RHS norm (float).

        Returns:
            The relative residual, same type as abs_residual.
        """
        return abs_residual / rhs_norm if rhs_norm > 1e-14 else abs_residual  # type: ignore[return-value]

    def _diagnostics(
        self,
        state: CGState,
        converged: bool,
        breakdown: bool,
        ortho_breakdown_at: int | None,
    ) -> TerminationDiagnostics:
        """Build one-shot termination diagnostics."""
        reason = None
        if breakdown:
            reason = describe_breakdown_reason(state.u) or "curvature_breakdown"
        return TerminationDiagnostics(
            converged_at=state.iteration if converged else None,
            breakdown_at=state.iteration if breakdown else None,
            breakdown_reason=reason,
            ortho_breakdown_at=ortho_breakdown_at,
        )

    def _stopping_criterion(
        self,
        state: CGState,
        converged: bool,
        breakdown: bool,
        maxiter: int | None,
    ) -> str:
        """Classify the stopping condition."""
        if converged:
            return "converged"
        if breakdown:
            return "breakdown"
        if maxiter is not None and state.iteration >= maxiter:
            return "max_iter"
        return "stopped"

    def _info_code(
        self,
        converged: bool,
        breakdown: bool,
        state: CGState,
        maxiter: int | None,
    ) -> int:
        """Return SciPy-style status code."""
        if converged:
            return 0
        if breakdown:
            return -1
        if maxiter is not None and state.iteration >= maxiter:
            return maxiter
        return 1


class PCGSolver(ConjugateGradientSolver):
    """Preconditioned Conjugate Gradient with optional reorthogonalization.

    Standard preconditioned Hestenes & Stiefel (1952)/Saad (2003, Algorithm
    9.1) two-term recurrence: ``beta_k = (r_k, w_k) / (r_{k-1}, w_{k-1})``,
    ``d_k = w_k + beta_k d_{k-1}``. Valid for a single, fixed SPD
    preconditioner ``M`` (see ``factories.pcg()`` for the full writeup and
    parameter docs). ``reorthogonalization`` optionally layers Notay
    (2000)'s periodic-restart reorthogonalization on top, to counter loss of
    A-conjugacy from accumulated rounding error.
    """

    def __init__(
        self,
        preconditioner: Preconditioner | None = None,
        convergence_criterion: IConvergenceCriterion | None = None,
        reorthogonalization: OrthogonalizationStrategy | None = None,
        beta_formula: str = "fletcher_reeves",
        iteration_history: IterationHistory | None = None,
        trace_mode: TraceMode = TraceMode.MINIMAL,
    ) -> None:
        """Initialize PCG solver."""
        self.reorthogonalization = reorthogonalization
        self.beta_formula = beta_formula
        base_strategy = TwoTermRecurrenceStrategy(beta_formula=beta_formula)
        if reorthogonalization is None:
            direction_strategy = base_strategy
        else:
            direction_strategy = CompositeDirectionStrategy(base_strategy, reorthogonalization)
        super().__init__(
            direction_strategy=direction_strategy,
            preconditioner=preconditioner,
            convergence_criterion=convergence_criterion,
            iteration_history=iteration_history,
            trace_mode=trace_mode,
        )


class FCGSolver(ConjugateGradientSolver):
    """Flexible Conjugate Gradient with explicit orthogonalization.

    Notay (2000) FCG: each new search direction is built by explicitly
    A-conjugating the preconditioned residual against stored ``(d_j, q_j)``
    pairs (``OrthogonalizationDirectionStrategy`` / ``orthogonalization``
    strategy), rather than relying on PCG's short two-term recurrence -
    necessary whenever the preconditioner varies between iterations or is
    not exactly SPD, since the two-term recurrence's optimality proof
    assumes a fixed SPD ``M``. See ``factories.flexible_cg()`` for the full
    writeup and parameter docs.
    """

    def __init__(
        self,
        orthogonalization: OrthogonalizationStrategy,
        preconditioner: Preconditioner | None = None,
        convergence_criterion: IConvergenceCriterion | None = None,
        iteration_history: IterationHistory | None = None,
        trace_mode: TraceMode = TraceMode.MINIMAL,
    ) -> None:
        """Initialize FCG solver."""
        self.orthogonalization = orthogonalization
        super().__init__(
            direction_strategy=OrthogonalizationDirectionStrategy(orthogonalization),
            preconditioner=preconditioner,
            convergence_criterion=convergence_criterion,
            iteration_history=iteration_history,
            trace_mode=trace_mode,
        )
