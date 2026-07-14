"""Continuous monitoring of solver iterations."""

from __future__ import annotations

import torch

from torchalg.monitoring.storage import ScalarHistory, VectorHistory
from torchalg.monitoring.trace_mode import TraceMode, coerce_trace_mode


class IterationHistory:
    """Track continuous solver diagnostics at each iteration.

    ``IterationHistory`` is intentionally a mutable container: solver loops
    append telemetry by replacing immutable ``ScalarHistory``/
    ``VectorHistory`` instances with updated copies. This keeps the stored
    values immutable while avoiding noisy return-value plumbing in the solver
    orchestration layer.
    """

    def __init__(self, mode: TraceMode | str = TraceMode.MINIMAL) -> None:
        """Initialize iteration history.

        Args:
            mode: Trace mode controlling which data is collected.
        """
        self.mode = coerce_trace_mode(mode)
        self.residual_norms = ScalarHistory.empty()
        self.residuals: VectorHistory | None = self._vector_history_for_mode()
        self.solutions: VectorHistory | None = self._vector_history_for_mode()
        self.directions: VectorHistory | None = self._vector_history_for_mode()

    def _vector_history_for_mode(self) -> VectorHistory | None:
        """Return vector storage only for full tracing."""
        if self.mode == TraceMode.FULL:
            return VectorHistory.empty()
        return None

    def log_iteration(
        self,
        residual_norm: float,
        residual: torch.Tensor | None = None,
        solution: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
    ) -> None:
        """Log data for one iteration.

        Args:
            residual_norm: Absolute residual norm for this iteration.
            residual: Residual vector, recorded only in ``FULL`` mode.
            solution: Solution vector, recorded only in ``FULL`` mode.
            direction: Search direction vector, recorded only in ``FULL`` mode.
        """
        if self.mode == TraceMode.DISABLED:
            return

        self.residual_norms = self.residual_norms.add(residual_norm)

        if self.mode != TraceMode.FULL:
            return

        self._log_full_vectors(residual=residual, solution=solution, direction=direction)

    def _log_full_vectors(
        self,
        *,
        residual: torch.Tensor | None,
        solution: torch.Tensor | None,
        direction: torch.Tensor | None,
    ) -> None:
        """Append supplied vectors to their histories in ``FULL`` mode."""
        if residual is not None and self.residuals is not None:
            self.residuals = self.residuals.add(residual)
        if solution is not None and self.solutions is not None:
            self.solutions = self.solutions.add(solution)
        if direction is not None and self.directions is not None:
            self.directions = self.directions.add(direction)

    def iteration_count(self) -> int:
        """Return the number of residual norms recorded."""
        return len(self.residual_norms)
