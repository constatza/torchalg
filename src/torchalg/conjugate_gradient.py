"""Conjugate Gradient solver family."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import torch

from torchalg.base import DEFAULT_ATOL, DEFAULT_RTOL, IterativeSolverBase
from torchalg.models.diagnostics import TerminationDiagnostics
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


class _IdentityPreconditioner(Preconditioner):
    """Internal identity preconditioner used when solvers are constructed directly."""

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
        """Create iteration-zero CG state."""
        u = torch.zeros_like(b) if x0 is None else x0.clone()
        r = b - linear_op(u)
        residual_norm = float(torch.linalg.norm(r))
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
        breakdown_tol: float | None = None,
    ) -> CGState:
        """Execute one unified CG iteration."""
        w = self._apply_preconditioner(self.preconditioner, state.r, state)
        rw_curr = stable_dot_product(state.r, w)
        d = self.direction_strategy.compute_direction(w, state)
        q = linear_op(d)
        curvature = stable_dot_product(d, q)
        alpha = rw_curr / curvature
        u_new = state.u + alpha * d
        r_new = state.r - alpha * q
        residual_norm_new = float(torch.linalg.norm(r_new))
        residual_history = state.residual_history.add(
            norm_abs=residual_norm_new,
            norm_rel=self._compute_relative_residual(residual_norm_new, state.rhs_norm),
        )
        direction_history = self._updated_direction_history(state, d, q)

        return replace(
            state,
            iteration=state.iteration + 1,
            residual_norm=residual_norm_new,
            u=u_new,
            r=r_new,
            w=w,
            d=d,
            q=q,
            direction_history=direction_history,
            residual_history=residual_history,
            w_prev=w.clone(),
            r_prev=state.r.clone(),
            rw_prev=rw_curr,
        )

    def _check_stopping(
        self,
        state: CGState,
        rtol: float,
        atol: float,
        maxiter: int,
        breakdown_tol: float | None = None,
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
        breakdown_tol: float | None = None,
        maxiter: int | None = None,
    ) -> SolverResult:
        """Build immutable solver result from final state and telemetry."""
        solution_valid = check_solution_validity(state.u)
        converged = solution_valid and self._check_convergence_with_criterion(
            state,
            self.convergence_criterion,
        )
        (
            residual_abs_hist,
            residual_rel_hist,
            residual_vectors,
            solution_vectors,
            direction_vectors,
        ) = self._extract_histories_from_iteration_history(state.rhs_norm)
        if residual_abs_hist is None and self.iteration_history is not None:
            residual_abs_hist = tuple(state.residual_history.norms_abs)
            residual_rel_hist = tuple(state.residual_history.norms_rel)

        breakdown = state.breakdown or not solution_valid
        diagnostics = self._diagnostics(state, converged, breakdown)
        stopping_criterion = self._stopping_criterion(state, converged, breakdown, maxiter)

        return SolverResult(
            converged=converged,
            iterations=state.iteration,
            residual=self._compute_relative_residual(state.residual_norm, state.rhs_norm),
            residual_abs=state.residual_norm,
            rhs_norm=state.rhs_norm,
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
    ):
        """Return direction history updated only when the strategy needs it."""
        required = self.direction_strategy.required_history_size
        if required is not None and required <= 0:
            return state.direction_history
        return state.direction_history.add(direction, matrix_product)

    def _compute_relative_residual(self, abs_residual: float, rhs_norm: float) -> float:
        """Compute relative residual with zero-RHS fallback."""
        return abs_residual / rhs_norm if rhs_norm > 1e-14 else abs_residual

    def _diagnostics(
        self,
        state: CGState,
        converged: bool,
        breakdown: bool,
    ) -> TerminationDiagnostics:
        """Build one-shot termination diagnostics."""
        reason = None
        if breakdown:
            reason = describe_breakdown_reason(state.u) or "curvature_breakdown"
        return TerminationDiagnostics(
            converged_at=state.iteration if converged else None,
            breakdown_at=state.iteration if breakdown else None,
            breakdown_reason=reason,
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
    """Preconditioned Conjugate Gradient with optional reorthogonalization."""

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
    """Flexible Conjugate Gradient with explicit orthogonalization."""

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
