"""Tracing granularity modes for solver event logging.

This module defines TraceMode enum to control the level of detail in event logging.
Different modes trade off memory usage vs diagnostic information.

Design Principles:
    - Single Responsibility: Only defines tracing modes
    - Clear Documentation: Memory usage clearly documented
    - Performance Conscious: Default mode minimizes overhead

Usage:
    >>> from neuralls.domain.solver.monitoring.trace_mode import TraceMode
    >>> solver = FlexibleCGSolver(trace_mode=TraceMode.MINIMAL)
    >>> solver = FlexibleCGSolver(trace_mode=TraceMode.FULL)  # For debugging
"""

from __future__ import annotations

from enum import StrEnum


class TraceMode(StrEnum):
    """Tracing granularity levels for solver event logging.

    Controls what data is logged during solver iterations. Higher levels
    provide more diagnostic information at the cost of memory usage.

    Modes:

    DISABLED:
        No logging - event_logger is None
        Memory: 0 bytes
        Use Case: Production runs where diagnostics not needed

    MINIMAL (DEFAULT):
        Log scalar events only (iteration, residual_norm, convergence flags)
        Memory: ~8 bytes per scalar × num_scalars × iterations
        Scalars: ITERATION, RESIDUAL_NORM, CONVERGED, BREAKDOWN, DIVERGENCE
        Example: 100 iterations × 5 scalars = ~4 KB
        Use Case: Monitoring convergence, lightweight diagnostics

    FULL:
        Log scalars + vector events (residual, solution, direction vectors)
        Memory: ~8 bytes per element × problem_size × num_vectors × iterations
        Vectors: RESIDUAL, SOLUTION, DIRECTION, PRECOND_RESIDUAL, MATRIX_PRODUCT
        Example: n=10,000, 100 iterations, 3 vectors = ~24 MB
        Use Case: Deep debugging, algorithm analysis, research

    Memory Usage Examples:

    Small problem (n=100):
        MINIMAL: 100 iters × 5 scalars × 8 bytes = ~4 KB
        FULL: 4 KB + 100 iters × 100 × 3 vectors × 8 bytes = ~244 KB

    Medium problem (n=10,000):
        MINIMAL: 100 iters × 5 scalars × 8 bytes = ~4 KB
        FULL: 4 KB + 100 iters × 10,000 × 3 vectors × 8 bytes = ~24 MB

    Large problem (n=1,000,000):
        MINIMAL: 100 iters × 5 scalars × 8 bytes = ~4 KB
        FULL: 4 KB + 100 iters × 1,000,000 × 3 vectors × 8 bytes = ~2.4 GB (!)

    Performance Impact:
        DISABLED: Zero overhead
        MINIMAL: Negligible (< 1% overhead)
        FULL: Significant for large problems (memory + copy time)

    Recommendation:
        - Use MINIMAL by default (tests pass, low overhead)
        - Use FULL only for debugging specific convergence issues
        - Use DISABLED for production if event_log not needed

    Theory:
        Krylov methods maintain O(n) vectors at each iteration. Logging all
        vectors creates O(kn) memory usage where k = iteration count. For large
        problems, this can exceed available RAM. Scalar-only logging maintains
        O(k) memory, typically negligible even for thousands of iterations.
    """

    DISABLED = "disabled"
    """No event logging. event_logger set to None. Zero memory/performance overhead."""

    MINIMAL = "minimal"
    """Log scalar events only. Low memory (~KB). Default mode."""

    FULL = "full"
    """Log scalars + vectors. High memory (~MB-GB). Debugging/research mode."""


def coerce_trace_mode(value: TraceMode | str) -> TraceMode:
    """Convert string to TraceMode; pass TraceMode through unchanged.

    Args:
        value: A TraceMode enum or string representation

    Returns:
        TraceMode enum value

    Examples:
        >>> coerce_trace_mode("minimal")
        <TraceMode.MINIMAL: 'minimal'>
        >>> coerce_trace_mode(TraceMode.MINIMAL)
        <TraceMode.MINIMAL: 'minimal'>
    """
    return TraceMode(value) if isinstance(value, str) else value
