"""Continuous monitoring of solver iterations."""

from __future__ import annotations

import math

import torch

from torchalg.monitoring.storage import ScalarHistory, VectorHistory
from torchalg.monitoring.trace_mode import TraceMode, coerce_trace_mode
from torchalg.utils.numerics import stable_dot_product


def _a_norm_error(
    *, x_exact: torch.Tensor, residual: torch.Tensor, solution: torch.Tensor
) -> float:
    """Compute ``||u_k - x_exact||_A`` via ``(x_exact - u_k)-r_k`` (no matvec).

    Exact for SPD ``A``: since ``A@x_exact = b`` and ``r_k = b - A@u_k``,
    ``A@(u_k - x_exact) = -r_k``, so
    ``||u_k - x_exact||_A^2 = (u_k - x_exact)-A@(u_k - x_exact)
    = (x_exact - u_k)-r_k``.

    Args:
        x_exact: Known exact solution.
        residual: Current residual ``r_k = b - A@u_k``.
        solution: Current iterate ``u_k``.

    Returns:
        float: ``||u_k - x_exact||_A``, clamped to ``0`` before the square
            root to absorb floating-point noise.
    """
    quadratic_form = stable_dot_product(x_exact - solution, residual)
    return math.sqrt(max(quadratic_form, 0.0))


class IterationHistory:
    """Track continuous solver diagnostics at each iteration.

    ``IterationHistory`` is intentionally a mutable container: solver loops
    append telemetry by replacing immutable ``ScalarHistory``/
    ``VectorHistory`` instances with updated copies. This keeps the stored
    values immutable while avoiding noisy return-value plumbing in the solver
    orchestration layer.
    """

    def __init__(
        self,
        mode: TraceMode | str = TraceMode.MINIMAL,
        x_exact: torch.Tensor | None = None,
    ) -> None:
        """Initialize iteration history.

        Args:
            mode: Trace mode controlling which vector data is collected.
            x_exact: Known exact solution, if available (e.g. a synthetic
                benchmark system). When given, ``error_norms`` is populated
                every iteration regardless of ``mode``, via the identity
                ``||u_k - x_exact||_A^2 = (x_exact - u_k)-r_k`` (exact for
                SPD ``A`` since ``A@x_exact = b``) — a single dot product of
                vectors already resident, no extra matvec, no stored
                vectors.
        """
        self.mode = coerce_trace_mode(mode)
        self.x_exact = x_exact
        self.residual_norms = ScalarHistory.empty()
        self.error_norms = ScalarHistory.empty()
        self.energy_decrements = ScalarHistory.empty()
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
        residual_norm: torch.Tensor | float,
        residual: torch.Tensor | None = None,
        solution: torch.Tensor | None = None,
        direction: torch.Tensor | None = None,
        energy_decrement: torch.Tensor | float | None = None,
    ) -> None:
        """Log data for one iteration.

        Args:
            residual_norm: Absolute residual norm for this iteration. May be
                a 0-d tensor; ``ScalarHistory.add()`` casts to float itself,
                so callers should not pre-convert (that would be a second,
                redundant device sync for the same value).
            residual: Residual vector. Stored in ``FULL`` mode; also used
                (without being stored) to compute ``error_norms`` when
                ``x_exact`` is set.
            solution: Solution vector. Stored in ``FULL`` mode; also used
                (without being stored) to compute ``error_norms`` when
                ``x_exact`` is set.
            direction: Search direction vector, recorded only in ``FULL`` mode.
            energy_decrement: This iteration's ``alpha_k * rho_k`` (the exact
                CG decrement of ``||e_k||_A^2``), recorded regardless of
                ``mode`` — see ``torchalg.monitoring.analysis.golub_meurant_error_bound``.
        """
        if self.mode == TraceMode.DISABLED:
            return

        self.residual_norms = self.residual_norms.add(residual_norm)

        if energy_decrement is not None:
            self.energy_decrements = self.energy_decrements.add(energy_decrement)

        if self.x_exact is not None and residual is not None and solution is not None:
            error_norm = _a_norm_error(x_exact=self.x_exact, residual=residual, solution=solution)
            self.error_norms = self.error_norms.add(error_norm)

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
