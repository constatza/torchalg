"""Continuous monitoring of solver iterations."""

from __future__ import annotations

import torch

from torchalg.monitoring.storage import ScalarHistory, VectorHistory, as_scalar_value, to_cpu_copy
from torchalg.monitoring.trace_mode import TraceMode, coerce_trace_mode
from torchalg.utils.numerics import stable_dot_product


def _a_norm_error(
    *, x_exact: torch.Tensor, residual: torch.Tensor, solution: torch.Tensor
) -> torch.Tensor:
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
        torch.Tensor: 0-d tensor ``||u_k - x_exact||_A``, clamped to ``0``
            before the square root to absorb floating-point noise.
    """
    quadratic_form = stable_dot_product(x_exact - solution, residual)
    return torch.sqrt(torch.clamp_min(quadratic_form, 0.0))


class IterationHistory:
    """Track continuous solver diagnostics at each iteration.

    ``IterationHistory`` is intentionally a mutable container: solver loops
    append telemetry to private lists (amortized ``O(1)`` per iteration; the
    previous rebuild of an immutable tuple per iteration made a ``k``-iteration
    solve ``O(k^2)``). The ``residual_norms``/``error_norms``/
    ``energy_decrements`` and ``residuals``/``solutions``/``directions``
    properties materialize immutable ``ScalarHistory``/``VectorHistory``
    snapshots on access, so a snapshot never changes when more iterations are
    logged afterwards.
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
        self._residual_norms: list[torch.Tensor] = []
        self._error_norms: list[torch.Tensor] = []
        self._energy_decrements: list[torch.Tensor] = []
        full_trace = self.mode == TraceMode.FULL
        self._residuals: list[torch.Tensor] | None = [] if full_trace else None
        self._solutions: list[torch.Tensor] | None = [] if full_trace else None
        self._directions: list[torch.Tensor] | None = [] if full_trace else None

    @property
    def residual_norms(self) -> ScalarHistory:
        """Snapshot of the absolute residual norms logged so far."""
        return ScalarHistory(values=tuple(self._residual_norms))

    @property
    def error_norms(self) -> ScalarHistory:
        """Snapshot of the A-norm errors logged so far (empty without ``x_exact``)."""
        return ScalarHistory(values=tuple(self._error_norms))

    @property
    def energy_decrements(self) -> ScalarHistory:
        """Snapshot of the per-iteration energy decrements logged so far."""
        return ScalarHistory(values=tuple(self._energy_decrements))

    @property
    def residuals(self) -> VectorHistory | None:
        """Snapshot of the residual vectors (``FULL`` mode only, else ``None``)."""
        return None if self._residuals is None else VectorHistory(vectors=tuple(self._residuals))

    @property
    def solutions(self) -> VectorHistory | None:
        """Snapshot of the solution vectors (``FULL`` mode only, else ``None``)."""
        return None if self._solutions is None else VectorHistory(vectors=tuple(self._solutions))

    @property
    def directions(self) -> VectorHistory | None:
        """Snapshot of the search directions (``FULL`` mode only, else ``None``)."""
        return None if self._directions is None else VectorHistory(vectors=tuple(self._directions))

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

        self._residual_norms.append(as_scalar_value(residual_norm))

        if energy_decrement is not None:
            self._energy_decrements.append(as_scalar_value(energy_decrement))

        if self.x_exact is not None and residual is not None and solution is not None:
            error_norm = _a_norm_error(x_exact=self.x_exact, residual=residual, solution=solution)
            self._error_norms.append(as_scalar_value(error_norm))

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
        if residual is not None and self._residuals is not None:
            self._residuals.append(to_cpu_copy(residual))
        if solution is not None and self._solutions is not None:
            self._solutions.append(to_cpu_copy(solution))
        if direction is not None and self._directions is not None:
            self._directions.append(to_cpu_copy(direction))

    def iteration_count(self) -> int:
        """Return the number of residual norms recorded."""
        return len(self._residual_norms)
