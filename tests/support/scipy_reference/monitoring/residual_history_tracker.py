"""Residual history tracking for scipy.sparse.linalg.cg callback interface.

This module provides ResidualHistoryTracker as a **necessary compatibility layer** for
integrating with scipy.sparse.linalg.cg's callback protocol.

Why ResidualHistoryTracker exists:
    - scipy.sparse.linalg.cg controls the iteration loop (we can't use IterationHistory directly)
    - scipy callbacks receive limited info (xk or pr_norm only)
    - We need an adapter to bridge scipy's callback to our monitoring system

Usage by solver type:
    - SciPyCGSolver: Uses ResidualHistoryTracker (required for scipy callback interface)
    - Native solvers (PCG, FCG): Use IterationHistory + EventLog directly (full control over iteration)

Both systems are valid and necessary:
    - ResidualHistoryTracker: For scipy integration (callback-based)
    - IterationHistory + EventLog: For native solvers (direct logging)
    - SciPyCallbackAdapter: Composes both for unified monitoring

Design:
    - ResidualHistoryTracker: Simple mutable history accumulation
    - Single Responsibility: ONLY tracks history (no callback logic)
    - Composition: Used by SciPyCallbackAdapter, not standalone

Theory:
    Separating history tracking from callback adaptation enables:
    - Testing each responsibility independently
    - Using history tracking without SciPy callbacks
    - Clear dependency graph
"""

from __future__ import annotations

import numpy as np


class ResidualHistoryTracker:
    """Residual history tracker for scipy.sparse.linalg.cg callback interface.

    This is a **necessary compatibility layer** for SciPyCGSolver, which wraps
    scipy.sparse.linalg.cg. Since scipy controls the iteration loop, we cannot
    use IterationHistory directly and need this adapter-friendly tracker.

    Usage by solver type:
        - SciPyCGSolver: Uses ResidualHistoryTracker (required for scipy callbacks)
        - Native solvers (PCG, FCG): Use IterationHistory + EventLog (full iteration control)

    This class has a single responsibility: accumulating residual norms
    and optionally solution vectors as the solver iterates.

    Attributes:
        residual_norms_abs: Absolute residual norms ||r_k||_2.
        residual_norms_rel: Relative residual norms ||r_k||_2 / ||b||_2.
        solutions: Optional list of solution vectors (heavy, use sparingly).

    See Also:
        IterationHistory: Continuous monitoring for native solvers
        EventLog: Discrete event logging for native solvers
        SciPyCallbackAdapter: Composes ResidualHistoryTracker with IterationHistory

    Example:
        >>> # Used by SciPyCGSolver via SciPyCallbackAdapter
        >>> tracker = ResidualHistoryTracker()
        >>> tracker.record_residual(norm_abs=1.0, norm_rel=0.1)
        >>> tracker.record_residual(norm_abs=0.5, norm_rel=0.05)
        >>> len(tracker.residual_norms_abs)
        2
        >>> tracker.residual_norms_abs[-1]
        0.5
    """

    def __init__(self) -> None:
        """Initialize empty history tracker.

        History tracking is always enabled. Solution tracking is disabled
        by default and must be enabled via enable_solution_tracking().
        """
        self.residual_norms_abs: list[float] = []
        self.residual_norms_rel: list[float] = []
        self.solutions: list[np.ndarray] | None = None

    def record_residual(self, norm_abs: float, norm_rel: float) -> None:
        """Record residual norms.

        Args:
            norm_abs: Absolute residual norm ||r_k||_2.
            norm_rel: Relative residual norm ||r_k||_2 / ||b||_2.

        Example:
            >>> tracker = ResidualHistoryTracker()
            >>> tracker.record_residual(1.0, 0.1)
            >>> tracker.residual_norms_abs
            [1.0]
        """
        self.residual_norms_abs.append(float(norm_abs))
        self.residual_norms_rel.append(float(norm_rel))

    def record_solution(self, x: np.ndarray) -> None:
        """Record solution vector (if solution tracking enabled).

        Args:
            x: Solution vector to record.

        Raises:
            RuntimeError: If solution tracking not enabled.

        Example:
            >>> tracker = ResidualHistoryTracker()
            >>> tracker.enable_solution_tracking()
            >>> tracker.record_solution(np.zeros(10))
            >>> len(tracker.solutions)
            1
        """
        if self.solutions is None:
            raise RuntimeError(
                "Solution tracking not enabled. Call enable_solution_tracking() first."
            )
        self.solutions.append(np.asarray(x, copy=True))

    def enable_solution_tracking(self) -> None:
        """Enable solution tracking.

        This allocates the solutions list. Call before recording any solutions.

        Example:
            >>> tracker = ResidualHistoryTracker()
            >>> tracker.enable_solution_tracking()
            >>> tracker.solutions
            []
        """
        if self.solutions is None:
            self.solutions = []

    def __len__(self) -> int:
        """Return number of residual norms recorded.

        Returns:
            Length of residual_norms_abs (same as residual_norms_rel).

        Example:
            >>> tracker = ResidualHistoryTracker()
            >>> tracker.record_residual(1.0, 0.1)
            >>> len(tracker)
            1
        """
        return len(self.residual_norms_abs)
