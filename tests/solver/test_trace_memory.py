"""CUDA-only regression test: ``FULL`` trace mode must not scale device memory with iterations.

See ``docs/bug-full-trace-history-exhausts-gpu-memory.md``: the original bug
grew GPU memory by ``O(3 * iterations * n)`` because iterate vectors were
cloned onto the solve device every step and never offloaded. The fix moves
each vector to host memory immediately (``monitoring/storage.py::VectorHistory.add``),
so device memory should only ever grow by ``O(n)`` regardless of
``maxiter``.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.factories import pcg
from torchalg.monitoring import TraceMode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")

_N = 2000
_MAXITER = 200
_FLOAT64_ITEMSIZE = 8


def _peak_bytes_for_trace_mode(trace_mode: TraceMode) -> int:
    """Run a CUDA solve and return peak allocated device bytes."""
    device = torch.device("cuda")
    diagonal = torch.linspace(1.0, 2.0, _N, dtype=torch.float64, device=device)
    matrix = torch.diag(diagonal)
    rhs = torch.ones(_N, dtype=torch.float64, device=device)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    pcg(matrix, rhs, trace_mode=trace_mode, maxiter=_MAXITER)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device)


def test_full_trace_memory_does_not_scale_with_iteration_count() -> None:
    """FULL trace's extra peak memory is O(n), not O(iterations * n)."""
    baseline = _peak_bytes_for_trace_mode(TraceMode.DISABLED)
    full = _peak_bytes_for_trace_mode(TraceMode.FULL)
    extra = full - baseline

    # Fixed behavior: at most a handful of live vectors' worth of overhead.
    one_vector = _N * _FLOAT64_ITEMSIZE
    assert extra < 20 * one_vector

    # The pre-fix bug would have added roughly 3 * maxiter vectors' worth
    # (residual + solution + direction, cloned every iteration).
    unfixed_growth = 3 * _MAXITER * one_vector
    assert extra < unfixed_growth / 2
