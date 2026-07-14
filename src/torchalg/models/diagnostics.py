"""Termination diagnostics for iterative solvers.

New in ``torchalg`` - not a port. Replaces the reference's
``EventLog``/``SolverEvent``/``EventType`` machinery
(``neuralls.domain.solver.monitoring.events``/``event_log``) with a single
frozen dataclass. See ``docs/plan.md``'s "Architectural corrections" section
for the full rationale; summarized here:

The reference frames ``EventLog`` as "Event Sourcing" - a replayable log
used to reconstruct state. But nothing in the solver ever replays or
queries ``EventLog`` as a stream: it only ever records three discrete,
one-shot facts and reads them back once, at the end of a solve. That is not
a logging subsystem, it is diagnostic data - a handful of
``int | None`` / ``str | None`` fields set once at termination. Replacing
the log with a plain frozen dataclass removes an entire module pair
(``events.py``, ``event_log.py``) and its query API for no loss of
information.

``IterationHistory`` (the genuinely continuous, per-iteration scalar/vector
trace - residual norms etc.) is unaffected by this change and is kept as-is
in ``torchalg.monitoring`` (Stage 8); it was never the overkill part.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TerminationDiagnostics:
    """One-shot diagnostic facts recorded at solver termination.

    Three discrete questions, answered once when the solve loop ends: did
    we converge (and when), did we break down (and when/why), and did FCG's
    truncated orthogonalization break down (and when). All fields default
    to ``None``, so ``TerminationDiagnostics()`` is the "nothing happened
    yet" state - the same role the reference's ``event_log=None`` played on
    ``SolverResult``, without needing an ``Optional`` wrapper around the
    whole record.

    Attributes:
        converged_at (int | None): Iteration index at which the convergence
            criterion was satisfied, or ``None`` if the solve never
            converged.
        breakdown_at (int | None): Iteration index at which numerical
            breakdown (NaN/Inf) was first detected, or ``None`` if no
            breakdown occurred.
        breakdown_reason (str | None): Human-readable classification of the
            breakdown (see
            ``torchalg.utils.validation.describe_breakdown_reason``,
            e.g. ``"nan_in_solution, inf_in_solution"``), or ``None`` if no
            breakdown occurred.
        ortho_breakdown_at (int | None): Iteration index at which FCG's
            truncated orthogonalization detected a breakdown (near-zero
            curvature against a stored direction), or ``None`` if it never
            occurred.

    Example:
        >>> diagnostics = TerminationDiagnostics()
        >>> diagnostics.converged_at is None
        True
        >>> diagnostics = TerminationDiagnostics(converged_at=12)
        >>> diagnostics.converged_at
        12
    """

    converged_at: int | None = None
    """Iteration index of convergence, or None if never converged."""

    breakdown_at: int | None = None
    """Iteration index of numerical breakdown, or None if none occurred."""

    breakdown_reason: str | None = None
    """Classification of the breakdown (e.g. "nan_in_solution"), or None."""

    ortho_breakdown_at: int | None = None
    """Iteration index of orthogonalization breakdown, or None."""
