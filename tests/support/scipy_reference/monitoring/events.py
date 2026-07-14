"""Type-safe event types for discrete solver events.

This module defines EventType enum for discrete events that occur during solver execution.

Key Principle:
    EventType represents DISCRETE EVENTS (occurrences), not continuous measurements.
    - Discrete: "Breakdown occurred at iteration 42"
    - NOT continuous: "Residual norm at every iteration" (use IterationHistory instead)

Design Changes:
    - Reduced from 17 event types to 3 discrete event types
    - Removed continuous measurements (RESIDUAL_NORM, ITERATION, etc.)
    - Removed vectors (RESIDUAL, SOLUTION, DIRECTION - use IterationHistory)
    - Removed is_scalar/is_vector properties (no longer needed)
    - Events record WHEN something happened, not continuous data

Pattern:
    Follows Event Sourcing pattern - events are immutable records of occurrences.

Usage:
    >>> from neuralls.domain.solver.monitoring.events import EventType
    >>> event_log.record(EventType.CONVERGED, iteration=42)
    >>> event_log.record(EventType.BREAKDOWN, iteration=10, reason="curvature")
    >>> event_log.record(EventType.ORTHO_BREAKDOWN, iteration=25)
"""

from __future__ import annotations

from enum import StrEnum


class EventType(StrEnum):
    """Type-safe discrete event types for solver execution.

    These represent OCCURRENCES during solving, not continuous measurements.

    Events:
        BREAKDOWN: Numerical breakdown detected (NaN/Inf/negative curvature).
        CONVERGED: Convergence criterion satisfied.
        ORTHO_BREAKDOWN: Orthogonalization breakdown detected.

    Removed (now in IterationHistory):
        - ITERATION: Just use list index
        - RESIDUAL_NORM: Continuous history, not event
        - RESIDUAL/SOLUTION/DIRECTION: Continuous histories, not events
        - BETA_COEFFICIENT, STEP_LENGTH, CURVATURE: Not needed
        - DIVERGENCE, REORTHOGONALIZED: Not currently used

    Note:
        CONVERGED is now stored as a discrete event with iteration number
        (e.g., converged_at = 42), not as a boolean flag every iteration
        (e.g., [0, 0, 0, 1, 1, 1, ...]).

    Example:
        >>> event = EventType.CONVERGED
        >>> event.value
        'converged'
    """

    BREAKDOWN = "breakdown"
    """Numerical breakdown detected (NaN/Inf, negative curvature, etc.)."""

    CONVERGED = "converged"
    """Convergence criterion satisfied (||r_k|| <= tolerance)."""

    ORTHO_BREAKDOWN = "ortho_breakdown"
    """Orthogonalization breakdown detected (FCG/PCG reorthogonalization)."""
