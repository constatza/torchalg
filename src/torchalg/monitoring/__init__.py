"""Continuous monitoring utilities for iterative solvers.

This package tracks per-iteration telemetry for solver loops:

- ``TraceMode``: tracing granularity control.
- ``ScalarHistory`` and ``VectorHistory``: immutable storage for scalar and
  tensor-valued iteration data.
- ``IterationHistory``: mutable telemetry container that applies trace-mode
  policy while retaining immutable history values.
- ``energy_norm_history``/``golub_meurant_error_bound`` (``analysis.py``):
  explicit, opt-in post-hoc analysis of a completed solve's recorded
  history - never run automatically inside ``solve()``.

Discrete termination facts live in ``torchalg.models.diagnostics``;
the reference implementation's event-log subsystem is intentionally not
ported (see ``docs/plan.md``).
"""

from .analysis import energy_norm_history, golub_meurant_error_bound
from .iteration_history import IterationHistory
from .storage import ScalarHistory, VectorHistory
from .trace_mode import TraceMode, coerce_trace_mode

__all__ = [
    "IterationHistory",
    "ScalarHistory",
    "TraceMode",
    "VectorHistory",
    "coerce_trace_mode",
    "energy_norm_history",
    "golub_meurant_error_bound",
]
