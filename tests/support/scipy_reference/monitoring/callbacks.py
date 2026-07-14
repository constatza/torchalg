"""SciPy callback adaptation for scipy.sparse.linalg.cg integration.

This module provides SciPyCallbackAdapter as a **necessary compatibility layer**
for SciPyCGSolver, which wraps scipy.sparse.linalg.cg.

Why this adapter exists:
    - scipy.sparse.linalg.cg uses a callback-based monitoring interface
    - We cannot use IterationHistory directly (scipy controls the iteration loop)
    - The adapter bridges scipy's callback protocol to our monitoring system

Components:
- SciPyCallbackAdapter: Bridges scipy.sparse.linalg.cg callbacks to ResidualHistoryTracker + IterationHistory
- InitialStateComputer: Computes initial state (x_0, r_0) for solvers

Usage by solver type:
    - SciPyCGSolver: Uses SciPyCallbackAdapter (required for scipy integration)
    - Native solvers (PCG, FCG): Use IterationHistory + EventLog directly (full iteration control)

Design:
    - SciPyCallbackAdapter: ONLY adapts callbacks (no history logic)
    - InitialStateComputer: ONLY computes initial state (pure function)
    - Composition: Adapter uses ResidualHistoryTracker and IterationHistory via dependency injection

Theory:
    Separating callback adaptation from history tracking enables:
    - Testing callback protocol compliance independently
    - Using history tracking without callbacks
    - Replacing callback mechanism without affecting history
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.sparse.linalg import LinearOperator

# Note: Using np.linalg.norm instead of scipy.linalg.norm to avoid
# NaN/Inf checks (scipy version raises ValueError on non-finite values)

if TYPE_CHECKING:
    from .iteration_history import IterationHistory
    from .residual_history_tracker import ResidualHistoryTracker


class SciPyCallbackAdapter:
    """Adapter for scipy.sparse.linalg.cg callback interface.

    This is a **necessary compatibility layer** for SciPyCGSolver, which wraps
    scipy.sparse.linalg.cg. Since scipy uses callbacks for monitoring, we cannot
    use IterationHistory directly and need this adapter.

    This class implements the SciPy CG callback protocol while delegating
    history tracking to ResidualHistoryTracker and event logging to IterationHistory.

    Usage by solver type:
        - SciPyCGSolver: Creates SciPyCallbackAdapter (required for scipy callbacks)
        - Native solvers (PCG, FCG): Use IterationHistory + EventLog directly (full iteration control)

    Responsibilities:
        - Implement __call__ for SciPy callback protocol
        - Convert callback arguments (xk or pr_norm) to residual norms
        - Delegate to ResidualHistoryTracker and IterationHistory (composition, not mixing)

    Attributes:
        A: System matrix (LinearOperator) for computing residuals.
        b: Right-hand side vector.
        callback_type: Type of callback ('x' for solution, 'pr_norm' for residual norm).
        tracker: ResidualHistoryTracker for recording history (optional).
        iteration_history: IterationHistory for continuous monitoring (optional).
        rhs_norm: Cached ||b||_2 for computing relative residuals.

    See Also:
        IterationHistory: Continuous monitoring for native solvers
        ResidualHistoryTracker: History tracking for scipy callbacks

    Example:
        >>> # Used by SciPyCGSolver
        >>> from neuralls.domain.solver.monitoring.residual_history_tracker import (
        ...     ResidualHistoryTracker,
        ... )
        >>> tracker = ResidualHistoryTracker()
        >>> adapter = SciPyCallbackAdapter(A, b, callback_type="x", tracker=tracker)
        >>> # Pass to scipy.sparse.linalg.cg
        >>> x, info = scipy.sparse.linalg.cg(A, b, callback=adapter)
        >>> tracker.residual_norms_abs
        [...]
    """

    def __init__(
        self,
        A: LinearOperator,
        b: np.ndarray,
        *,
        callback_type: Literal["x", "pr_norm"] = "x",
        tracker: ResidualHistoryTracker | None = None,
        iteration_history: IterationHistory | None = None,
    ) -> None:
        """Initialize SciPy callback adapter.

        Args:
            A: System matrix (LinearOperator).
            b: Right-hand side vector.
            callback_type: Type of callback:
                - 'x': Callback receives solution vector xk
                - 'pr_norm': Callback receives residual norm directly
            tracker: Optional ResidualHistoryTracker for recording history.
            iteration_history: Optional IterationHistory for continuous monitoring.

        Example:
            >>> adapter = SciPyCallbackAdapter(A, b, callback_type="x")
            >>> # Use with scipy.sparse.linalg.cg
        """
        self.A = A
        self.b = b
        self.callback_type = callback_type
        self.tracker = tracker
        self.iteration_history = iteration_history
        self.rhs_norm = float(np.linalg.norm(b))

    def __call__(self, xk_or_norm: np.ndarray | float) -> None:
        """SciPy callback protocol implementation.

        Called by SciPy during each iteration with either the solution vector
        or the residual norm (depending on callback_type).

        Args:
            xk_or_norm: Solution vector (if callback_type='x') or
                        residual norm (if callback_type='pr_norm').

        Example:
            >>> # Called automatically by SciPy
            >>> adapter(xk)  # Solution vector
        """
        residual_abs: float
        residual: np.ndarray | None = None
        xk: np.ndarray | None = None

        if self.callback_type == "pr_norm":
            # Callback gives residual norm directly
            residual_abs = float(xk_or_norm)
            residual = None
        else:
            # Callback gives solution vector, compute residual
            xk = np.asarray(xk_or_norm, dtype=np.float64, copy=False)
            residual = self.b - self.A @ xk
            residual_abs = float(np.linalg.norm(residual))

            # Record solution if tracking enabled
            if self.tracker and self.tracker.solutions is not None:
                self.tracker.record_solution(xk)

        # Compute relative residual
        residual_rel = residual_abs / self.rhs_norm if self.rhs_norm > 0 else residual_abs

        # Record residual norms if tracking enabled
        if self.tracker:
            self.tracker.record_residual(residual_abs, residual_rel)

        # Log to IterationHistory if enabled
        if self.iteration_history is not None:
            self.iteration_history.log_iteration(
                residual_norm=residual_abs,
                residual=residual,
                solution=xk if self.callback_type == "x" else None,
            )


class InitialStateComputer:
    """Computes initial state for iterative solvers (Single Responsibility).

    This class has a single responsibility: computing the initial solution
    and residual vectors from the linear system and optional initial guess.

    All methods are static (pure functions) for simplicity and testability.

    Theory:
        The initial state is:
        - r_0 = b             when x_0 = 0 (None)
        - r_0 = b - A @ x_0   when x_0 is provided

    Example:
        >>> x_init, r_init, r_norm = InitialStateComputer.compute_initial_residual(A, b, x0=None)
        >>> np.allclose(r_init, b)
        True
    """

    @staticmethod
    def compute_initial_residual(
        A: LinearOperator,
        b: np.ndarray,
        x0: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute initial solution, residual, and residual norm.

        Args:
            A: System matrix (LinearOperator).
            b: Right-hand side vector.
            x0: Optional initial guess. If None, uses zero vector.

        Returns:
            Tuple (x_initial, r_initial, r_norm_initial):
            - x_initial: Initial solution (x_0 or zeros)
            - r_initial: Initial residual (b - A @ x_0)
            - r_norm_initial: Initial residual norm ||r_0||_2

        Example:
            >>> # Zero initial guess
            >>> x, r, r_norm = InitialStateComputer.compute_initial_residual(A, b, x0=None)
            >>> np.allclose(x, 0)
            True
            >>> np.allclose(r, b)
            True

            >>> # Non-zero initial guess
            >>> x0 = np.random.rand(n)
            >>> x, r, r_norm = InitialStateComputer.compute_initial_residual(A, b, x0)
            >>> np.allclose(x, x0)
            True
            >>> np.allclose(r, b - A @ x0)
            True
        """
        if x0 is None:
            x_initial = np.zeros_like(b)
            r_initial = np.asarray(b, copy=True)
        else:
            x_initial = np.asarray(x0, dtype=np.float64, copy=True)
            r_initial = b - A @ x_initial

        r_norm = float(np.linalg.norm(r_initial))
        return x_initial, r_initial, r_norm
