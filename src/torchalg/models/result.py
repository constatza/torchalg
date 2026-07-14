"""Result dataclasses for solver execution.

Ported from ``neuralls.domain.solver.models.result`` (see ``docs/plan.md``)
with ``NDArray`` fields translated to ``torch.Tensor``, list fields
translated to immutable tuples, and the whole dataclass made ``frozen=True,
slots=True`` (the reference's ``SolverResult`` was plain, mutable
``@dataclass``) per this repo's immutability convention (see
``docs/plan.md``'s "nn.Module usage" architecture decision: state/result
records are pure immutable data, replaced via ``dataclasses.replace``,
never mutated in place).

Two deliberate deviations from a straight port, both per ``docs/plan.md``:

- ``event_log: EventLog | None`` is replaced by
  ``diagnostics: TerminationDiagnostics`` (see
  ``torchalg.models.diagnostics`` for the full rationale).
- ``iteration_history: IterationHistory | None`` is dropped from this
  module for now: ``IterationHistory`` lives in ``torchalg.monitoring``
  (ported in Stage 8), but ``torchalg.models`` must not depend on
  ``torchalg.monitoring`` (see the dependency DAG in ``docs/plan.md``).
  Wiring it back onto ``SolverResult`` happens when Stage 9 lands, from a
  module that is allowed to see both.

This module defines result containers returned by solvers after execution.
It includes diagnostic information, convergence status, and optional
traces.

Design:
    - ``SolverResult``: General solver result with SciPy-aligned metadata.
    - ``IterationContext``: Context passed to flexible preconditioners.

    The reference's comparison-workflow-specific result types
    (``CGComparisonResult``, ``PlotPaths``, ``RankedRecommendation``,
    ``ComparisonRecommendations``, ``ComparisonResult``) are not ported here;
    they belong to ``comparison.py``, which is Stage 10 (lowest priority) per
    ``docs/plan.md``'s scope trim.

Theory:
    Results contain both numerical outcomes (solution, residual) and
    diagnostic information (iterations, convergence status, breakdown
    flags). This separation enables post-analysis and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .diagnostics import TerminationDiagnostics


@dataclass(frozen=True, slots=True)
class SolverResult:
    """Execution result for iterative solvers with diagnostic metadata.

    This dataclass contains all information about a solver run:
    - Convergence status and final residual.
    - Iteration counts and history.
    - Diagnostic flags (breakdown, divergence).
    - Optional traces (residual vectors, solution vectors).

    Attributes:
        converged (bool): Whether convergence criterion satisfied.
        iterations (int): Number of iterations executed.
        residual (float): Relative residual norm ``||r||/||b||`` at
            termination.
        residual_abs (float): Absolute residual norm ``||r||`` at
            termination.
        rhs_norm (float): Right-hand side norm ``||b||``.
        breakdown (bool): Whether numerical breakdown occurred.
        info (int): Status code (0=success, >0=warning, <0=error).
        residual_history_rel (tuple[float, ...] | None): Relative residual
            norms across iterations.
        residual_history_abs (tuple[float, ...] | None): Absolute residual
            norms across iterations.
        tol (float | None): Relative tolerance used (for diagnostics).
        atol (float | None): Absolute tolerance used.
        maxiter (int | None): Maximum iterations allowed.
        diagnostics (TerminationDiagnostics): One-shot termination facts
            (convergence/breakdown iteration and reason); see
            ``torchalg.models.diagnostics``.
        stopping_criterion (str | None): Description of why solver stopped.
        residual_vectors (torch.Tensor | None): Optional full residual
            vectors (shape: iterations x n).
        solution_vectors (torch.Tensor | None): Optional full solution
            vectors (shape: iterations x n).

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

    residual_history_rel: tuple[float, ...] | None = None
    """Relative residual norms ||r_k||/||b|| across iterations."""

    residual_history_abs: tuple[float, ...] | None = None
    """Absolute residual norms ||r_k|| across iterations."""

    tol: float | None = None
    """Relative tolerance used (for diagnostics)."""

    atol: float | None = None
    """Absolute tolerance used (for diagnostics)."""

    maxiter: int | None = None
    """Maximum iterations allowed."""

    diagnostics: TerminationDiagnostics = field(default_factory=TerminationDiagnostics)
    """One-shot termination facts (see torchalg.models.diagnostics)."""

    stopping_criterion: str | None = None
    """Description of why solver stopped (e.g., 'converged', 'max_iter', 'breakdown')."""

    residual_vectors: torch.Tensor | None = None
    """Optional full residual vectors (shape: iterations x n). Heavy, use sparingly."""

    solution_vectors: torch.Tensor | None = None
    """Optional full solution vectors (shape: iterations x n). Heavy, use sparingly."""


@dataclass(frozen=True, slots=True)
class IterationContext:
    """Context passed to flexible preconditioners and step helpers.

    Provides iteration-specific information to preconditioners and helpers
    that need access to current solver state beyond just the residual.

    Attributes:
        iteration (int): Current iteration number (0-indexed).
        residual (torch.Tensor): Current residual vector r_k.
        solution (torch.Tensor): Current solution vector x_k.
        matrix (torch.Tensor): System matrix A (optional, may be expensive
            to store).
        rhs (torch.Tensor): Right-hand side vector b.

    Example:
        >>> import torch
        >>> ctx = IterationContext(
        ...     iteration=5,
        ...     residual=torch.ones(10),
        ...     solution=torch.zeros(10),
        ...     matrix=torch.eye(10),
        ...     rhs=torch.ones(10),
        ... )
        >>> ctx.iteration
        5

    Usage:
        Context-aware preconditioners can use iteration info for adaptive
        behavior::

            def adaptive_precond(r: torch.Tensor, ctx: IterationContext) -> torch.Tensor:
                if ctx.iteration < 10:
                    # Use simple preconditioner early.
                    return r / torch.diagonal(ctx.matrix)
                # Use expensive preconditioner later.
                return neural_network(r, ctx.solution)
    """

    iteration: int
    """Current iteration number (0-indexed)."""

    residual: torch.Tensor
    """Current residual vector r_k = b - A*x_k."""

    solution: torch.Tensor
    """Current solution vector x_k."""

    matrix: torch.Tensor
    """System matrix A (optional, may be expensive to store)."""

    rhs: torch.Tensor
    """Right-hand side vector b."""
