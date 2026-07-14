"""Monitoring and callback components for solver execution.

This package provides components for tracking solver execution:
- iteration_history: Continuous monitoring (IterationHistory)
- event_log: Discrete event logging (EventLog, SolverEvent)
- events: EventType enum for discrete event types
- trace_mode: TraceMode enum for controlling logging granularity
- storage: Immutable history classes (ScalarHistory, VectorHistory)
- residual_history_tracker: Scipy-specific residual tracking (ResidualHistoryTracker)
- callbacks: SciPy callback adaptation (SciPyCallbackAdapter, InitialStateComputer)

Design Principles:
    - Separation of Concerns: Continuous monitoring vs discrete events
    - Direct Access: iteration_history.residual_norms (not enum indexing)
    - Industry Patterns: Telemetry/Metrics + Event Sourcing
    - Single Responsibility: Each class has one clear purpose
    - Type Safety: EventType enum for discrete events only
    - Composition: Components compose via dependency injection
    - Optional: All monitoring is optional (zero overhead when unused)
"""

from .callbacks import InitialStateComputer, SciPyCallbackAdapter
from .event_log import EventLog, SolverEvent
from .events import EventType
from .iteration_history import IterationHistory
from .residual_history_tracker import ResidualHistoryTracker
from .storage import ScalarHistory, VectorHistory
from .trace_mode import TraceMode

__all__ = [
    # Core monitoring classes
    "IterationHistory",
    "EventLog",
    "SolverEvent",
    "EventType",
    "TraceMode",
    # Storage classes
    "ScalarHistory",
    "VectorHistory",
    # Scipy integration
    "ResidualHistoryTracker",
    "SciPyCallbackAdapter",
    "InitialStateComputer",
]
