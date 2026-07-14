"""SciPy CG solver with callback-based monitoring.

This module provides SciPyCGSolver, which wraps scipy.sparse.linalg.cg and uses
SciPyCallbackAdapter to capture iteration data via callbacks.

Design Principles:
    - Wrapper Pattern: Delegates to scipy.sparse.linalg.cg
    - Callback Integration: Uses SciPyCallbackAdapter for monitoring
    - Single Responsibility: Only wraps scipy CG, no custom algorithm
    - Composition: Uses ResidualHistoryTracker and EventLog via dependency injection

Use Cases:
    - Comparison workflows requiring scipy CG baseline
    - Validation against scipy reference implementation
    - When scipy's CG implementation is preferred over custom

References:
    - scipy.sparse.linalg.cg documentation
    - Hestenes & Stiefel (1952). Methods of Conjugate Gradients.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from collections.abc import Callable

import numpy as np
from scipy.sparse import spmatrix
from scipy.linalg import norm
from scipy.sparse.linalg import LinearOperator, aslinearoperator, cg

from .models.result import SolverResult
from .monitoring.callbacks import InitialStateComputer, SciPyCallbackAdapter
from .monitoring.residual_history_tracker import ResidualHistoryTracker
from .monitoring.trace_mode import TraceMode
from .monitoring.event_log import EventLog
from .monitoring.iteration_history import IterationHistory
from .preconditioners import Identity, Preconditioner

if TYPE_CHECKING:
    from numpy.typing import NDArray

type LinearSystemOperator = NDArray | spmatrix | LinearOperator | Callable[[NDArray], NDArray]


class SciPyCGSolver:
    """Wrapper for scipy.sparse.linalg.cg with callback monitoring.

    This solver delegates to scipy's CG implementation while capturing
    iteration data via callbacks. It provides the same interface as
    custom solvers but uses scipy under the hood.

    Attributes:
        preconditioner: Preconditioner strategy (M^{-1} operator).
        iteration_history: Optional IterationHistory for continuous monitoring.
        event_log: Optional EventLog for discrete event logging.

    Example:
        >>> from neuralls.domain.solver.preconditioners import Identity
        >>> from neuralls.domain.solver.monitoring.event_log import EventLog
        >>> from neuralls.domain.solver.monitoring.iteration_history import IterationHistory
        >>>
        >>> precond = Identity()
        >>> event_log = EventLog()
        >>> iteration_history = IterationHistory()
        >>> solver = SciPyCGSolver(
        ...     preconditioner=precond, event_log=event_log, iteration_history=iteration_history
        ... )
        >>>
        >>> x, result = solver.solve(A, b, rtol=1e-6)
        >>> print(f"Converged: {result.converged}, Iterations: {result.iterations}")
    """

    def __init__(
        self,
        preconditioner: Preconditioner | None = None,
        iteration_history: IterationHistory | None = None,
        event_log: EventLog | None = None,
    ) -> None:
        """Initialize scipy CG solver.

        Args:
            preconditioner: Preconditioner M (default: Identity).
            iteration_history: Optional IterationHistory for continuous monitoring.
            event_log: Optional EventLog for discrete event logging.

        Example:
            >>> solver = SciPyCGSolver()  # Identity preconditioner, no logging
            >>> solver = SciPyCGSolver(iteration_history=IterationHistory())  # With logging
        """
        self.preconditioner = preconditioner or Identity()
        self.iteration_history = iteration_history
        self.event_log = event_log

    def solve(
        self,
        A: LinearSystemOperator,
        b: NDArray,
        x0: NDArray | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-14,
        maxiter: int | None = None,
        trace_mode: TraceMode | str = TraceMode.MINIMAL,
        **kwargs,
    ) -> tuple[NDArray, SolverResult]:
        """Solve Ax = b using scipy.sparse.linalg.cg with callback monitoring.

        Args:
            A: System matrix or linear operator (n × n, SPD).
            b: Right-hand side vector (n,).
            x0: Initial guess (n,). If None, uses zero vector.
            rtol: Relative tolerance for convergence.
            atol: Absolute tolerance for convergence.
            maxiter: Maximum number of iterations (default: 10 * N).
            trace_mode: Logging granularity (MINIMAL or FULL).
            **kwargs: Additional parameters (event_log override).

        Returns:
            Tuple (x, result):
                - x: Solution vector (n,).
                - result: SolverResult with convergence info and diagnostics.

        Example:
            >>> x, result = solver.solve(A, b, rtol=1e-6, maxiter=100)
            >>> print(f"Residual: {result.residual}, Iterations: {result.iterations}")
        """
        # Convert trace_mode string to enum
        if isinstance(trace_mode, str):
            trace_mode = TraceMode(trace_mode)

        b_arr = np.asarray(b, dtype=np.float64)
        if callable(A) and not isinstance(A, np.ndarray):
            n = b_arr.shape[0]
            # ty's scipy stub only models LinearOperator.__init__, not the
            # matvec/matmat dispatch its __new__ performs at runtime (scipy's
            # own documented public API) — not a defect in this ported logic.
            A_op = LinearOperator(
                shape=(n, n),
                matvec=cast(Callable[[np.ndarray], np.ndarray], A),  # ty: ignore[unknown-argument]
                dtype=np.float64,
            )
        else:
            A_op = cast(Any, aslinearoperator)(A)

        # Prepare preconditioner as LinearOperator
        M_op = self._prepare_preconditioner(A_op.shape)

        # Get event log from kwargs or instance
        event_log = kwargs.get("event_log", self.event_log)

        # Create history tracker
        tracker = ResidualHistoryTracker()
        if trace_mode == TraceMode.FULL:
            tracker.enable_solution_tracking()

        # Compute initial residual and log iteration 0
        x_init, r_init, r0_norm = InitialStateComputer.compute_initial_residual(A_op, b_arr, x0)

        # Compute RHS norm for relative residuals
        rhs_norm = float(norm(b_arr))

        # Log initial state (iteration 0)
        r0_norm_rel = r0_norm / rhs_norm if rhs_norm > 0 else r0_norm
        tracker.record_residual(norm_abs=r0_norm, norm_rel=r0_norm_rel)

        if self.iteration_history is not None:
            self.iteration_history.log_iteration(
                residual_norm=r0_norm,
                residual=r_init,
                solution=x_init if trace_mode == TraceMode.FULL else None,
            )

        if trace_mode == TraceMode.FULL:
            tracker.record_solution(x_init)

        # Create callback adapter
        callback = SciPyCallbackAdapter(
            A=A_op,
            b=b_arr,
            callback_type="x",  # Capture solution vectors
            tracker=tracker,
            iteration_history=self.iteration_history,
        )

        # Solve using scipy CG
        # scipy convergence: ||r|| <= max(atol, rtol * ||b||)
        # which matches our convention exactly
        x_sol, info_code = cg(
            A=A_op,
            b=b_arr,
            x0=x_init,
            rtol=rtol,  # Relative tolerance
            atol=atol,  # Absolute tolerance
            maxiter=maxiter,
            M=M_op,
            callback=callback,
        )

        # Build result
        result = self._build_result(
            x_sol=x_sol,
            info_code=info_code,
            tracker=tracker,
            event_log=event_log,
            rtol=rtol,
            atol=atol,
            rhs_norm=rhs_norm,
        )

        return x_sol, result

    def _prepare_preconditioner(self, shape: tuple[int, int]) -> LinearOperator | None:
        """Convert preconditioner to scipy LinearOperator.

        Args:
            shape: Matrix shape (n, n).

        Returns:
            LinearOperator for preconditioner, or None for identity.

        Theory:
            scipy.sparse.linalg.cg expects preconditioner as LinearOperator
            with matvec(v) -> M^{-1} v operation.
        """
        if self.preconditioner is None:
            return None

        # Test if preconditioner is effectively identity via duck typing
        # instead of isinstance check (polymorphism instead of type checking)
        test_vec = np.ones(shape[0])
        result = self.preconditioner.apply(test_vec)

        if np.allclose(result, test_vec):
            return None  # scipy handles identity natively

        # Wrap in LinearOperator
        def matvec(v: NDArray) -> NDArray:
            return self.preconditioner.apply(v)

        # See the ty: ignore note in solve() above — same scipy stub gap.
        return LinearOperator(
            shape=shape,
            matvec=matvec,  # ty: ignore[unknown-argument]
            dtype=np.float64,
        )

    def _build_result(
        self,
        x_sol: NDArray,
        info_code: int,
        tracker: ResidualHistoryTracker,
        event_log: EventLog | None,
        rtol: float,
        atol: float,
        rhs_norm: float,
    ) -> SolverResult:
        """Build SolverResult from scipy output and tracked data.

        Args:
            x_sol: Final solution vector from scipy.
            info_code: scipy CG info code (0=success, >0=max iter, <0=error).
            tracker: ResidualHistoryTracker with recorded residuals.
            event_log: Optional EventLog with logged events.
            rtol: Relative tolerance used.
            atol: Absolute tolerance used.
            rhs_norm: Norm of RHS vector ||b||.

        Returns:
            SolverResult with all diagnostic information.

        Theory:
            scipy CG info codes:
            - 0: Convergence achieved
            - >0: Max iterations reached (not converged)
            - <0: Illegal input or breakdown
        """
        # Extract residual histories from tracker
        residual_history_abs = tracker.residual_norms_abs
        residual_history_rel = tracker.residual_norms_rel

        # Extract vectors if logged (FULL mode)
        residual_vectors = None
        solution_vectors = None
        if self.iteration_history is not None and self.iteration_history.residuals is not None:
            residual_vectors = self.iteration_history.residuals.to_array()
        if tracker.solutions is not None:
            solution_vectors = np.array(tracker.solutions)

        # Check solution validity and determine convergence
        from .utils.validation import check_solution_validity, record_breakdown_event

        # Determine effective iterations by checking history against tolerance
        # SciPy might take an extra step due to floating point differences or check timing.
        # We align with our strict criterion: first k where r_k converges.

        # Default to full history count
        raw_iterations = max(0, len(residual_history_abs) - 1)
        iterations = raw_iterations

        # Find first convergence point
        convergence_threshold = max(rtol * rhs_norm, atol)

        # Check if we converged earlier (excluding r0 if we took steps)
        # We start checking from index 0. If r0 converged, iterations=0.
        converged_idx = -1
        for i, norm_val in enumerate(residual_history_abs):
            if norm_val <= convergence_threshold:
                converged_idx = i
                break

        # If we found a convergence point
        if converged_idx != -1:
            iterations = converged_idx
            # Override info code to indicate success
            info_code = 0

            # Truncate histories to match the "virtual" stop point
            # Keep up to index `iterations` (so length is iterations + 1)
            end_idx = iterations + 1
            residual_history_abs = residual_history_abs[:end_idx]
            residual_history_rel = residual_history_rel[:end_idx]
            if solution_vectors is not None:
                solution_vectors = solution_vectors[:end_idx]
            if residual_vectors is not None:
                residual_vectors = residual_vectors[:end_idx]

        if not check_solution_validity(x_sol):
            record_breakdown_event(event_log, x_sol, iterations)
            converged = False
        else:
            converged = info_code == 0

        # Get final residuals (NaN if no data - clearly indicates failure, not convergence)
        final_residual_abs = residual_history_abs[-1] if residual_history_abs else float("nan")
        final_residual_rel = residual_history_rel[-1] if residual_history_rel else float("nan")

        return SolverResult(
            converged=converged,
            iterations=iterations,
            residual=final_residual_rel,
            residual_abs=final_residual_abs,
            rhs_norm=rhs_norm,
            breakdown=(info_code < 0),
            residual_history_rel=residual_history_rel,
            residual_history_abs=residual_history_abs,
            tol=rtol,
            atol=atol,
            event_log=event_log,
            iteration_history=self.iteration_history,
            residual_vectors=residual_vectors,
            solution_vectors=solution_vectors,
            info=info_code,
        )
