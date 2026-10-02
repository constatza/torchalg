"""Advisory CPU/CUDA device recommendation for ``torchalg.sparse`` operations.

See ``docs/plan.md``'s "The generic CPU-vs-CUDA rule" and "At-scale
evidence" sections, re-scoped by "Device placement: the algorithm decides,
not a blanket ``.to(device)``" - this is a library-internal helper now, not
only a caller-facing advisory one: a sparse orchestrator (e.g. an AMG
V-cycle) may consult it per pipeline stage (parallel-kernel vs. sequential-
graph-construction vs. already-dense-by-design, per that section), never to
silently migrate a tensor's device behind the caller's back. Nothing in this
module moves a tensor or checks CUDA availability - it only ever returns a
``torch.device`` recommendation; the caller (human or orchestrator) decides
whether and when to act on it.

There is no universal N threshold across operation shapes. The constants
below summarize one benchmark environment and one synthetic matrix family;
they are internal defaults, not a public performance guarantee. Re-measure
with ``python -m benchmarks.sparse_dense.report`` when hardware, PyTorch, or
the sparsity pattern changes materially.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

OperationCategory = Literal["parallel_kernel", "elementwise", "iterative"]

PARALLEL_KERNEL_CUDA_CROSSOVER_N = 2_000
"""Single parallel kernel ops (``mv``, ``formed_spgemm``, ``formed_spmm``,
``matrix_free_sparse``): CPU wins below N~1,000-2,000, CUDA wins above by
10-30x at N=15876 (measured ``/data/shared/mgroup-clusters/results.csv``)."""

ELEMENTWISE_CUDA_CROSSOVER_N = math.inf
"""Tiny elementwise ops (``jacobi_apply``-style diagonal scaling): no CUDA
crossover observed in the tested range - CPU is still ~1.6x faster than
CUDA at N=32,041, the largest size measured."""

ITERATIVE_CUDA_CROSSOVER_N = 15_000
"""Iterative, many-small-sequential-kernel-launch ops (``lobpcg``): CPU wins
below N~15,000, CUDA wins above but only by ~1.7x - host/device sync per
iteration dominates over FLOPs, so the crossover arrives late and the CUDA
win stays modest."""

_CROSSOVER_N: dict[OperationCategory, float] = {
    "parallel_kernel": PARALLEL_KERNEL_CUDA_CROSSOVER_N,
    "elementwise": ELEMENTWISE_CUDA_CROSSOVER_N,
    "iterative": ITERATIVE_CUDA_CROSSOVER_N,
}


def recommend_device(operation_category: OperationCategory, n: int) -> torch.device:
    """Recommend a device for an operation of this shape and problem size.

    Args:
        operation_category (OperationCategory): ``"parallel_kernel"``
            (matvec/GEMM-like, single-shot), ``"elementwise"`` (diagonal
            scaling), or ``"iterative"`` (many small sequential kernel
            launches, e.g. ``lobpcg``).
        n (int): Problem size (matrix dimension).

    Returns:
        torch.device: ``torch.device("cuda")`` if ``n`` is at or above the
            measured crossover for ``operation_category``, else
            ``torch.device("cpu")``.
    """
    return torch.device("cuda") if n >= _CROSSOVER_N[operation_category] else torch.device("cpu")
