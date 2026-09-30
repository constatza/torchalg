"""Memory-footprint measurement, kept separate from timing.

Runtime alone hides the actual reason large FEM matrices force a sparse
representation: a dense float64 matrix at N=50,000 is ~20GB, which is a
memory problem, not a speed one. This module reports memory as its own
metric per tensor/cell rather than folding it into prose justification for
the size caps in ``run_benchmark.py``.

Three tiers, all stdlib/torch-native (no new dependency):

1. :func:`storage_bytes` - exact, deterministic size of a tensor's own
   backing storage, computed from its properties. This is the primary,
   noise-free number and the one that actually answers "how much RAM does
   holding A in this format cost."
2. :func:`gpu_peak_memory` - a context manager wrapping
   ``torch.cuda.max_memory_allocated()``, torch's own precise allocator
   statistic. Only meaningful on CUDA.
3. :func:`cpu_peak_rss_delta` - a context manager wrapping
   ``resource.getrusage().ru_maxrss``, a best-effort, explicitly
   approximate secondary signal (process-wide, includes Python/allocator
   overhead, and - because ``ru_maxrss`` is a running high-water mark that
   never resets - only detects a *new* peak reached during the block, not
   memory reused below a previous cell's peak).
"""

from __future__ import annotations

import resource
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch


def storage_bytes(tensor: torch.Tensor) -> int:
    """Exact backing-storage size of a tensor, dense or sparse, in bytes.

    Args:
        tensor (torch.Tensor): Dense (strided) or sparse (COO/CSR/CSC)
            tensor.

    Returns:
        int: Total bytes across all buffers backing this tensor (values
            plus, for sparse layouts, the index buffers).
    """
    if tensor.layout == torch.strided:
        return tensor.numel() * tensor.element_size()
    if tensor.layout == torch.sparse_coo:
        indices = tensor._indices()
        values = tensor._values()
        return indices.numel() * indices.element_size() + values.numel() * values.element_size()
    if tensor.layout in (torch.sparse_csr, torch.sparse_csc):
        crow_or_ccol = (
            tensor.crow_indices() if tensor.layout == torch.sparse_csr else tensor.ccol_indices()
        )
        col_or_row = (
            tensor.col_indices() if tensor.layout == torch.sparse_csr else tensor.row_indices()
        )
        values = tensor.values()
        return (
            crow_or_ccol.numel() * crow_or_ccol.element_size()
            + col_or_row.numel() * col_or_row.element_size()
            + values.numel() * values.element_size()
        )
    raise ValueError(f"Unsupported layout for storage_bytes: {tensor.layout}")


@dataclass(frozen=True)
class MemoryReading:
    """Result of a memory-measurement context manager.

    Attributes:
        peak_bytes: Peak measured during the block, or ``None`` if the
            measurement wasn't applicable (e.g. GPU stats requested with no
            CUDA device).
        approximate: True if this reading is a best-effort proxy
            (:func:`cpu_peak_rss_delta`) rather than an exact figure.
    """

    peak_bytes: int | None
    approximate: bool


@contextmanager
def gpu_peak_memory(device: torch.device) -> Iterator[MemoryReading]:
    """Measure peak CUDA allocator memory used inside the block.

    Resets torch's per-device peak-memory stat on entry and reads it back
    on exit, so the reading reflects only this block's allocations, not
    anything accumulated earlier in the process.

    Args:
        device (torch.device): CUDA device to measure. If not a CUDA
            device, no measurement is taken (``peak_bytes`` stays ``None``)
            since the underlying stat only exists for CUDA.

    Yields:
        MemoryReading: Mutated in place; ``peak_bytes`` is only valid after
            the ``with`` block exits.
    """
    reading = MemoryReading(peak_bytes=None, approximate=False)
    if device.type != "cuda":
        yield reading
        return
    torch.cuda.reset_peak_memory_stats(device)
    try:
        yield reading
    finally:
        object.__setattr__(reading, "peak_bytes", torch.cuda.max_memory_allocated(device))


@contextmanager
def cpu_peak_rss_delta() -> Iterator[MemoryReading]:
    """Best-effort peak resident-set-size delta during the block.

    ``ru_maxrss`` is a process-wide, monotonically non-decreasing
    high-water mark (kilobytes on Linux) - it cannot be reset mid-process,
    so this only detects a *new* peak reached during the block. A block
    that reuses memory already below a previous high-water mark reports a
    delta of 0 even though it genuinely allocated something - a known
    limitation of using the OS-level stat instead of a per-process
    isolated profiler, acceptable here since :func:`storage_bytes` is the
    primary, exact signal and this is only a secondary cross-check.

    Yields:
        MemoryReading: Mutated in place; ``peak_bytes`` (bytes, converted
            from the kilobytes ``ru_maxrss`` reports on Linux) and
            ``approximate=True`` are only valid after the ``with`` block
            exits.
    """
    reading = MemoryReading(peak_bytes=None, approximate=True)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        yield reading
    finally:
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        object.__setattr__(reading, "peak_bytes", max(0, after - before) * 1024)
