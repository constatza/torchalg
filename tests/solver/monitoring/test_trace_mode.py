"""Tests for ``torchalg.monitoring.trace_mode``."""

from __future__ import annotations

import pytest

from torchalg.monitoring.trace_mode import TraceMode, coerce_trace_mode


def test_trace_mode_values_match_public_strings() -> None:
    """Trace mode enum values are stable public configuration strings."""
    assert TraceMode.DISABLED == "disabled"
    assert TraceMode.MINIMAL == "minimal"
    assert TraceMode.FULL == "full"


def test_coerce_trace_mode_accepts_enum() -> None:
    """Coercion passes enum values through unchanged."""
    assert coerce_trace_mode(TraceMode.MINIMAL) is TraceMode.MINIMAL


def test_coerce_trace_mode_accepts_string() -> None:
    """Coercion converts valid strings to enum values."""
    assert coerce_trace_mode("full") is TraceMode.FULL


def test_coerce_trace_mode_rejects_unknown_string() -> None:
    """Unknown strings fail with the enum's standard ``ValueError``."""
    with pytest.raises(ValueError, match="unknown"):
        coerce_trace_mode("unknown")
