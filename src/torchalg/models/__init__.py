"""Typed, immutable data models for iterative solvers.

This package contains all state, history, result, and configuration
dataclasses, plus the structural Protocols used to type-check against them:

- ``state``: Solver state hierarchy (``SolverState``, ``KrylovState``,
  ``CGState``).
- ``history``: Direction and residual history tracking.
- ``result``: Solver execution results (``SolverResult``,
  ``IterationContext``).
- ``config``: Solver configuration (``SolverConfig``, ``SolverParams``).
- ``diagnostics``: One-shot termination facts (``TerminationDiagnostics``),
  replacing the reference's ``EventLog`` (see ``docs/plan.md``).
- ``protocols``: Structural typing contracts (``HasVectors``,
  ``HasDirectionHistory``, ``SolverProtocol``) - imported directly from
  ``torchalg.models.protocols``, not re-exported here (matches the
  reference's own convention).

Design Principles:
    - Immutable (``frozen=True``) for thread safety.
    - Fully typed (no dynamic attributes).
    - Single Responsibility (each class has one purpose).
"""

from .config import SolverConfig, SolverParams
from .diagnostics import TerminationDiagnostics
from .history import DirectionHistory, ResidualHistory
from .result import IterationContext, SolverResult
from .state import CGState, KrylovState, SolverState

__all__ = [
    # State hierarchy
    "SolverState",
    "KrylovState",
    "CGState",
    # History
    "DirectionHistory",
    "ResidualHistory",
    # Results
    "SolverResult",
    "IterationContext",
    # Diagnostics
    "TerminationDiagnostics",
    # Config
    "SolverConfig",
    "SolverParams",
]
