"""Continuous monitoring of solver iterations.

This module implements the Metrics/Telemetry pattern for tracking continuous
solver diagnostics at each iteration.

Design:
    - Direct attribute access (not enum indexing): tracker.residual_norms
    - Immutable history objects (ScalarHistory, VectorHistory)
    - Functional updates (reassign attributes with new history)
    - TraceMode controls which vectors are collected
    - DirectionHistory lives in solver state, NOT here
    - No orthogonalization diagnostics (orthogonalization is internal)

Key Principle:
    This tracks CONTINUOUS data (collected every iteration), not discrete events.
    For discrete events (breakdown at iteration 42), use EventLog instead.

Follows Telemetry/Metrics pattern:
    - Like Prometheus, StatsD, OpenTelemetry
    - Direct attribute access for different metric types
    - Clear separation between continuous monitoring and discrete events
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .storage import ScalarHistory, VectorHistory
from .trace_mode import TraceMode

if TYPE_CHECKING:
    from numpy.typing import NDArray


class IterationHistory:
    """Tracks continuous solver diagnostics at each iteration.

    Follows Telemetry/Metrics pattern with direct attribute access.

    Design:
        - Immutable history objects (ScalarHistory, VectorHistory)
        - Functional updates (reassign attributes with new history)
        - DirectionHistory lives in solver state, NOT here
        - No orthogonalization diagnostics (orthogonalization is internal)

    Attributes:
        mode: Tracing mode (MINIMAL or FULL).
        residual_norms: Scalar history of ||r_k|| at each iteration (always tracked).
        residuals: Vector history of r_k vectors (FULL mode only).
        solutions: Vector history of x_k vectors (FULL mode only).
        directions: Vector history of p_k vectors (FULL mode only, for debugging).

    Note:
        DirectionHistory (d_k, q_k window for orthogonalization) lives in CGState,
        NOT in monitoring. This directions history is for debugging/visualization only.

    Example:
        >>> history = IterationHistory(mode=TraceMode.MINIMAL)
        >>> history.log_iteration(residual_norm=1.0)
        >>> history.log_iteration(residual_norm=0.5)
        >>> history.residual_norms.to_list()
        [1.0, 0.5]
        >>> history.iteration_count()
        2
    """

    def __init__(self, mode: TraceMode = TraceMode.MINIMAL) -> None:
        """Initialize iteration history.

        Args:
            mode: Tracing mode controlling which data is collected.
                MINIMAL: Only residual_norms (scalars).
                FULL: Residual_norms + residuals/solutions/directions (vectors).

        Example:
            >>> history = IterationHistory(mode=TraceMode.MINIMAL)
            >>> history.residuals is None
            True
            >>> history_full = IterationHistory(mode=TraceMode.FULL)
            >>> history_full.residuals is not None
            True
        """
        self.mode = mode

        # Always tracked (MINIMAL and FULL modes)
        self.residual_norms = ScalarHistory.empty()

        # Only tracked in FULL mode (for debugging/visualization)
        self.residuals: VectorHistory | None = (
            VectorHistory.empty() if mode == TraceMode.FULL else None
        )
        self.solutions: VectorHistory | None = (
            VectorHistory.empty() if mode == TraceMode.FULL else None
        )
        self.directions: VectorHistory | None = (
            VectorHistory.empty() if mode == TraceMode.FULL else None
        )

    def log_iteration(
        self,
        residual_norm: float,
        residual: NDArray | None = None,
        solution: NDArray | None = None,
        direction: NDArray | None = None,
    ) -> None:
        """Log data for one iteration (mutates by reassigning attributes).

        Uses functional updates to maintain immutability of history objects
        while allowing mutation of the IterationHistory container.

        Args:
            residual_norm: ||r_k|| at this iteration (always logged).
            residual: r_k vector (logged only in FULL mode).
            solution: x_k vector (logged only in FULL mode).
            direction: p_k vector (logged only in FULL mode, for debugging).

        Note:
            This direction is for visualization/debugging, NOT orthogonalization.
            DirectionHistory (d_k, q_k window) lives in CGState.

        Example:
            >>> import numpy as np
            >>> history = IterationHistory(mode=TraceMode.FULL)
            >>> r = np.array([1.0, 2.0])
            >>> x = np.array([0.1, 0.2])
            >>> p = np.array([0.5, 0.6])
            >>> history.log_iteration(residual_norm=2.236, residual=r, solution=x, direction=p)
            >>> history.residual_norms[0]
            2.236
            >>> len(history.residuals)
            1
        """
        # Always log residual norm
        self.residual_norms = self.residual_norms.add(residual_norm)

        # Log vectors only in FULL mode
        if self.mode == TraceMode.FULL:
            if residual is not None and self.residuals is not None:
                self.residuals = self.residuals.add(residual)
            if solution is not None and self.solutions is not None:
                self.solutions = self.solutions.add(solution)
            if direction is not None and self.directions is not None:
                self.directions = self.directions.add(direction)

    def iteration_count(self) -> int:
        """Number of iterations recorded.

        Returns:
            Number of iterations logged (length of residual_norms).

        Example:
            >>> history = IterationHistory()
            >>> history.log_iteration(residual_norm=1.0)
            >>> history.log_iteration(residual_norm=0.5)
            >>> history.iteration_count()
            2
        """
        return len(self.residual_norms)
