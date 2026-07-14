"""Tracing granularity modes for solver iteration telemetry."""

from __future__ import annotations

from enum import StrEnum


class TraceMode(StrEnum):
    """Tracing granularity levels for solver monitoring."""

    DISABLED = "disabled"
    """Collect no iteration telemetry."""

    MINIMAL = "minimal"
    """Collect scalar residual norms only."""

    FULL = "full"
    """Collect scalar residual norms and per-iteration vectors."""


def coerce_trace_mode(value: TraceMode | str) -> TraceMode:
    """Convert string trace modes to ``TraceMode`` values."""
    if isinstance(value, TraceMode):
        return value
    return TraceMode(value)
