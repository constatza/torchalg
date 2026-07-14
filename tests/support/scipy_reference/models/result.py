"""Result dataclasses for solver execution.

This module defines result containers returned by solvers after execution.
Includes diagnostic information, convergence status, and optional traces.

Design:
    - SolverResult: General solver result with SciPy-aligned metadata

Theory:
    Results contain both numerical outcomes (solution, residual) and diagnostic
    information (iterations, convergence status, breakdown flags). This separation
    enables post-analysis and debugging.

Note:
    This is a narrowed snapshot of the reference's ``models/result.py``: only
    ``SolverResult`` is kept (unchanged). The reference's other five
    dataclasses in this file (``CGComparisonResult``, ``IterationContext``,
    ``PlotPaths``, ``RankedRecommendation``, ``ComparisonRecommendations``,
    ``ComparisonResult``) belong to the ``comparison.py``/``utils/export``
    benchmark-runner subsystem, which this Stage-0 CG oracle does not use —
    and ``ComparisonResult`` depends on ``config.ComparisonGeneral``, which
    itself depends on the external ``neuralls.shared.types`` package this
    snapshot must not depend on. See ``tests.support.scipy_reference``'s
    package docstring for the full rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..monitoring.event_log import EventLog
    from ..monitoring.iteration_history import IterationHistory


@dataclass
class SolverResult:
    """Execution result for iterative solvers with diagnostic metadata.

    This dataclass contains all information about a solver run:
    - Convergence status and final residual
    - Iteration counts and history
    - Diagnostic flags (breakdown, divergence)
    - Optional traces (residual vectors, solution vectors)

    Attributes:
        converged: Whether convergence criterion satisfied.
        iterations: Number of iterations executed.
        residual: Relative residual norm ||r||/||b|| at termination.
        residual_abs: Absolute residual norm ||r|| at termination.
        rhs_norm: Right-hand side norm ||b||.
        breakdown: Whether numerical breakdown occurred.
        info: Status code (0=success, >0=warning, <0=error).
        residual_history_rel: Relative residual norms across iterations.
        residual_history_abs: Absolute residual norms across iterations.
        tol: Tolerance used (for diagnostics).
        atol: Absolute tolerance used.
        maxiter: Maximum iterations allowed.
        event_log: Optional EventLog with discrete solver events.
        iteration_history: Optional IterationHistory with continuous iteration diagnostics.
        stopping_criterion: Description of why solver stopped.
        residual_vectors: Optional full residual vectors (shape: iterations x n).
        solution_vectors: Optional full solution vectors (shape: iterations x n).

    Example:
        >>> result = SolverResult(
        ...     converged=True,
        ...     iterations=10,
        ...     residual=1e-8,
        ...     residual_abs=1e-8,
        ...     rhs_norm=1.0,
        ...     breakdown=False,
        ... )
        >>> result.converged
        True
    """

    converged: bool
    """Whether convergence criterion satisfied: ||r|| <= max(rtol*||b||, atol)."""

    iterations: int
    """Number of iterations executed."""

    residual: float
    """Relative residual norm ||r||/||b|| at termination."""

    residual_abs: float
    """Absolute residual norm ||r|| at termination."""

    rhs_norm: float
    """Right-hand side norm ||b||."""

    breakdown: bool
    """Whether numerical breakdown occurred (NaN/Inf). Terminal condition."""

    info: int = 0
    """Status code: 0=success, >0=warning (max iter), <0=error (breakdown)."""

    residual_history_rel: list[float] | None = None
    """Relative residual norms ||r_k||/||b|| across iterations."""

    residual_history_abs: list[float] | None = None
    """Absolute residual norms ||r_k|| across iterations."""

    tol: float | None = None
    """Relative tolerance used (for diagnostics)."""

    atol: float | None = None
    """Absolute tolerance used (for diagnostics)."""

    maxiter: int | None = None
    """Maximum iterations allowed."""

    event_log: EventLog | None = None
    """Optional EventLog with discrete solver events."""

    iteration_history: IterationHistory | None = None
    """Optional IterationHistory with continuous iteration diagnostics."""

    stopping_criterion: str | None = None
    """Description of why solver stopped (e.g., 'converged', 'max_iter', 'breakdown')."""

    residual_vectors: np.ndarray | None = None
    """Optional full residual vectors (shape: iterations x n). Heavy, use sparingly."""

    solution_vectors: np.ndarray | None = None
    """Optional full solution vectors (shape: iterations x n). Heavy, use sparingly."""
