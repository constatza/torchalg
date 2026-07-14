"""Template-method base for torch-native iterative solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast

import torch

from torchalg.models.protocols import HasVectors
from torchalg.models.result import SolverResult
from torchalg.models.state import SolverState
from torchalg.monitoring import IterationHistory, TraceMode
from torchalg.preconditioners.base import Preconditioner, PreconditionerContext
from torchalg.strategies.convergence import IConvergenceCriterion
from torchalg.utils.validation import validate_matrix, validate_rhs_vector

DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-14
DEFAULT_BREAKDOWN_TOL = 1e-14

type LinearSystemOperator = Callable[[torch.Tensor], torch.Tensor] | torch.Tensor


class IterativeSolverBase[S: SolverState](ABC):
    """Abstract base class for iterative solvers using template method."""

    def __init__(
        self,
        *,
        iteration_history: IterationHistory | None = None,
        trace_mode: TraceMode = TraceMode.MINIMAL,
    ) -> None:
        """Initialize common solver monitoring configuration."""
        self.iteration_history = iteration_history
        self.trace_mode = trace_mode

    def solve(
        self,
        A: LinearSystemOperator,
        b: torch.Tensor,
        x0: torch.Tensor | None = None,
        *,
        rtol: float | None = None,
        atol: float | None = None,
        maxiter: int | None = None,
        breakdown_tol: float | None = None,
    ) -> tuple[torch.Tensor, SolverResult]:
        """Solve ``A x = b`` using the concrete iterative method."""
        self._validate_system(A, b, x0)

        rtol_eff = DEFAULT_RTOL if rtol is None else rtol
        atol_eff = DEFAULT_ATOL if atol is None else atol
        breakdown_tol_eff = DEFAULT_BREAKDOWN_TOL if breakdown_tol is None else breakdown_tol
        maxiter_eff = 10 * b.numel() if maxiter is None else maxiter
        linear_op = self._prepare_operator(A)

        state = self._initialize_state(linear_op, b, x0, maxiter=maxiter_eff)
        self._log_state(state)

        while not self._check_stopping(
            state,
            rtol_eff,
            atol_eff,
            maxiter_eff,
            breakdown_tol=breakdown_tol_eff,
        ):
            state = self._iterate_step(linear_op, state, breakdown_tol=breakdown_tol_eff)
            self._log_state(state)

        if not isinstance(state, HasVectors):
            raise RuntimeError("Final solution not found in state")

        return state.u, self._build_result(
            state,
            rtol_eff,
            atol_eff,
            breakdown_tol=breakdown_tol_eff,
            maxiter=maxiter_eff,
        )

    def check_convergence(
        self,
        residual: torch.Tensor,
        rhs_norm: float,
        criterion: IConvergenceCriterion,
    ) -> bool:
        """Delegate convergence checking to the injected criterion."""
        return criterion.has_converged(residual, rhs_norm)

    def _check_convergence_with_criterion(
        self,
        state: S,
        criterion: IConvergenceCriterion,
    ) -> bool:
        """Check convergence for states that expose residual vectors."""
        if isinstance(state, HasVectors):
            return self.check_convergence(state.r, state.rhs_norm, criterion)
        return False

    def _validate_system(
        self,
        A: LinearSystemOperator,
        b: torch.Tensor,
        x0: torch.Tensor | None,
    ) -> None:
        """Validate system dimensions and finiteness."""
        if b.ndim != 1:
            raise ValueError(f"b must be 1D vector, got shape {tuple(b.shape)}")
        validate_rhs_vector(b)

        if isinstance(A, torch.Tensor):
            validate_matrix(A)
            if A.shape[0] != b.shape[0]:
                raise ValueError(f"A shape {tuple(A.shape)} incompatible with b {tuple(b.shape)}")

        if x0 is not None:
            validate_rhs_vector(x0)
            if x0.shape != b.shape:
                raise ValueError(f"x0 shape {tuple(x0.shape)} != b shape {tuple(b.shape)}")

    def _prepare_operator(self, A: LinearSystemOperator) -> Callable[[torch.Tensor], torch.Tensor]:
        """Wrap dense tensors and callables behind a matvec callable."""
        if callable(A):
            return cast(Callable[[torch.Tensor], torch.Tensor], A)

        matrix = A

        def _matvec(vector: torch.Tensor) -> torch.Tensor:
            return matrix @ vector

        return _matvec

    @abstractmethod
    def _initialize_state(
        self,
        linear_op: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        x0: torch.Tensor | None,
        maxiter: int | None = None,
    ) -> S:
        """Initialize solver state before the iteration loop."""

    @abstractmethod
    def _iterate_step(
        self,
        linear_op: Callable[[torch.Tensor], torch.Tensor],
        state: S,
        breakdown_tol: float | None = None,
    ) -> S:
        """Execute one solver iteration."""

    @abstractmethod
    def _check_stopping(
        self,
        state: S,
        rtol: float,
        atol: float,
        maxiter: int,
        breakdown_tol: float | None = None,
    ) -> bool:
        """Return whether the solver should stop."""

    @abstractmethod
    def _build_result(
        self,
        state: S,
        rtol: float,
        atol: float,
        breakdown_tol: float | None = None,
        maxiter: int | None = None,
    ) -> SolverResult:
        """Build the final solver result."""

    def _log_state(self, state: S) -> None:
        """Log continuous iteration telemetry if enabled."""
        if self.iteration_history is None:
            return

        residual = None
        solution = None
        direction = None
        if self.trace_mode == TraceMode.FULL and isinstance(state, HasVectors):
            residual = state.r
            solution = state.u
            direction = state.d

        self.iteration_history.log_iteration(
            residual_norm=state.residual_norm,
            residual=residual,
            solution=solution,
            direction=direction,
        )

    def _apply_preconditioner(
        self,
        preconditioner: Preconditioner,
        residual: torch.Tensor,
        state: SolverState,
    ) -> torch.Tensor:
        """Apply preconditioner with a uniform context object."""
        context = PreconditionerContext(
            iteration=state.iteration,
            residual_norm=state.residual_norm,
            rhs_norm=state.rhs_norm,
        )
        result = preconditioner.apply(residual, context)
        if result.shape != residual.shape:
            raise ValueError(
                f"Preconditioner output shape {tuple(result.shape)} != input "
                f"{tuple(residual.shape)}"
            )
        return result.to(dtype=residual.dtype, device=residual.device)

    def _extract_histories_from_iteration_history(
        self,
        rhs_norm: float,
    ) -> tuple[
        tuple[float, ...] | None,
        tuple[float, ...] | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Extract result histories from the configured iteration history."""
        if self.iteration_history is None:
            return None, None, None, None, None

        residual_history_abs = tuple(self.iteration_history.residual_norms.to_list())
        if not residual_history_abs:
            return None, None, None, None, None

        if rhs_norm > 0:
            residual_history_rel = tuple(value / rhs_norm for value in residual_history_abs)
        else:
            residual_history_rel = residual_history_abs

        residual_vectors = None
        if self.iteration_history.residuals is not None:
            residual_vectors = self.iteration_history.residuals.to_tensor()

        solution_vectors = None
        if self.iteration_history.solutions is not None:
            solution_vectors = self.iteration_history.solutions.to_tensor()

        direction_vectors = None
        if self.iteration_history.directions is not None:
            direction_vectors = self.iteration_history.directions.to_tensor()

        return (
            residual_history_abs,
            residual_history_rel,
            residual_vectors,
            solution_vectors,
            direction_vectors,
        )
