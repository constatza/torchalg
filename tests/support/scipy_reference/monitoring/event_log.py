"""Event log for discrete solver events.

This module implements the Event Sourcing pattern for tracking discrete
events that occur during solver execution.

Design:
    - Immutable event records (SolverEvent dataclass)
    - Events have type, iteration, and metadata
    - Query methods for finding events
    - Events are occurrences, not continuous data

Key Principle:
    This tracks DISCRETE events (breakdown at iteration 42, converged at iteration 100),
    not continuous data. For continuous monitoring (residual norms every iteration),
    use IterationHistory instead.

Follows Event Sourcing pattern:
    - Immutable event records
    - Append-only log
    - Query by event type
    - Events capture what happened and when
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .events import EventType


@dataclass(frozen=True)
class SolverEvent:
    """Immutable record of a discrete solver event.

    Events represent occurrences during solver execution, not continuous measurements.

    Attributes:
        type: Type of event (BREAKDOWN, CONVERGED, ORTHO_BREAKDOWN).
        iteration: Iteration number when event occurred.
        metadata: Additional event-specific data.

    Example:
        >>> from neuralls.domain.solver.monitoring.events import EventType
        >>> event = SolverEvent(
        ...     type=EventType.CONVERGED, iteration=42, metadata={"final_residual": 1e-8}
        ... )
        >>> event.iteration
        42
        >>> event.metadata["final_residual"]
        1e-08
    """

    type: EventType
    """Type of event that occurred."""

    iteration: int
    """Iteration number when event occurred."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional event-specific data."""


class EventLog:
    """Log of discrete events during solve.

    Follows Event Sourcing pattern - immutable event records in append-only log.

    Events are occurrences (breakdown at iteration 42), not continuous data
    (residual norms every iteration). For continuous monitoring, use IterationHistory.

    Attributes:
        _events: Internal list of recorded events.

    Example:
        >>> from neuralls.domain.solver.monitoring.events import EventType
        >>> log = EventLog()
        >>> log.record(EventType.CONVERGED, iteration=42, final_residual=1e-8)
        >>> log.converged_at()
        42
        >>> event = log.find_first(EventType.CONVERGED)
        >>> event.metadata["final_residual"]
        1e-08
    """

    def __init__(self) -> None:
        """Initialize empty event log.

        Example:
            >>> log = EventLog()
            >>> len(log.all_events())
            0
        """
        self._events: list[SolverEvent] = []

    def record(self, event_type: EventType, iteration: int, **metadata: Any) -> None:
        """Record a discrete event.

        Args:
            event_type: Type of event (BREAKDOWN, CONVERGED, ORTHO_BREAKDOWN).
            iteration: Iteration number when event occurred.
            **metadata: Additional event-specific data.

        Example:
            >>> from neuralls.domain.solver.monitoring.events import EventType
            >>> log = EventLog()
            >>> log.record(EventType.BREAKDOWN, iteration=10, reason="curvature_negative")
            >>> log.breakdown_at()
            10
        """
        event = SolverEvent(type=event_type, iteration=iteration, metadata=metadata)
        self._events.append(event)

    def all_events(self) -> list[SolverEvent]:
        """Get all recorded events.

        Returns:
            List of all events (copy for safety).

        Example:
            >>> from neuralls.domain.solver.monitoring.events import EventType
            >>> log = EventLog()
            >>> log.record(EventType.CONVERGED, iteration=5)
            >>> log.record(EventType.BREAKDOWN, iteration=10)
            >>> len(log.all_events())
            2
        """
        return list(self._events)

    def find_first(self, event_type: EventType) -> SolverEvent | None:
        """Find first occurrence of event type.

        Args:
            event_type: Type of event to find.

        Returns:
            First event of given type, or None if not found.

        Example:
            >>> from neuralls.domain.solver.monitoring.events import EventType
            >>> log = EventLog()
            >>> log.record(EventType.CONVERGED, iteration=42)
            >>> event = log.find_first(EventType.CONVERGED)
            >>> event.iteration
            42
            >>> log.find_first(EventType.BREAKDOWN) is None
            True
        """
        for event in self._events:
            if event.type == event_type:
                return event
        return None

    def converged_at(self) -> int | None:
        """Iteration where convergence occurred.

        Returns:
            Iteration number where CONVERGED event occurred, or None.

        Example:
            >>> from neuralls.domain.solver.monitoring.events import EventType
            >>> log = EventLog()
            >>> log.record(EventType.CONVERGED, iteration=100)
            >>> log.converged_at()
            100
        """
        from .events import EventType

        event = self.find_first(EventType.CONVERGED)
        return event.iteration if event else None

    def breakdown_at(self) -> int | None:
        """Iteration where breakdown occurred.

        Returns:
            Iteration number where BREAKDOWN event occurred, or None.

        Example:
            >>> from neuralls.domain.solver.monitoring.events import EventType
            >>> log = EventLog()
            >>> log.record(EventType.BREAKDOWN, iteration=50, reason="curvature")
            >>> log.breakdown_at()
            50
        """
        from .events import EventType

        event = self.find_first(EventType.BREAKDOWN)
        return event.iteration if event else None

    def ortho_breakdown_at(self) -> int | None:
        """Iteration where orthogonalization breakdown occurred.

        Returns:
            Iteration number where ORTHO_BREAKDOWN event occurred, or None.

        Example:
            >>> from neuralls.domain.solver.monitoring.events import EventType
            >>> log = EventLog()
            >>> log.record(EventType.ORTHO_BREAKDOWN, iteration=25)
            >>> log.ortho_breakdown_at()
            25
        """
        from .events import EventType

        event = self.find_first(EventType.ORTHO_BREAKDOWN)
        return event.iteration if event else None
