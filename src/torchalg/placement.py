"""Cost-based CPU/GPU placement of a solve.

``solve()`` used to move every operand to CUDA whenever a GPU was visible. The cost audit
shows that this is wrong for most small and medium problems: a GPU kernel pays a launch and
synchronization latency of about 0.1 to 0.2 ms whatever its size, a sparse CPU kernel at
``n`` below 10^5 costs microseconds, and a level-scheduled triangular solve issues about 13
kernels per dependency level, which measured 30 to 165 times slower than the CPU.

The decision therefore depends on what bounds the work (:class:`Placement`) and on its size,
and is made in one place, :func:`choose_device`, from a :class:`PlacementProblem`. An
explicit ``device=`` argument of ``solve()`` bypasses it, and ``solve()`` consults it only
for a CPU-resident system matrix: one already on an accelerator stays there, since moving it
to the CPU would copy it back on every solve.

The two thresholds are provisional: they come from an AMD gfx1030 measurement (the only GPU
measured so far) and must be re-measured on CUDA, see ``plans/cost-audit-fixes.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchalg.preconditioners.base import Placement

# PROVISIONAL: both thresholds come from an AMD gfx1030 (ROCm) measurement, not CUDA; re-measure
# on CUDA (benchmarks/throughput/device_crossover.py) before trusting them.
SPMV_GPU_MIN_NNZ = 500_000
"""Stored entries from which SpMV-bound iterations beat the CPU (a Jacobi-PCG iteration won at
``n = 90 000``, 450 000 entries, by 1.1 times and at 250 000 by 2.1 times)."""

DENSE_GPU_MIN_N = 2_000
"""Rows from which dense products beat the CPU (GEMV won 22 times at ``n = 2 744``)."""

_CPU = torch.device("cpu")
_CUDA = torch.device("cuda")


@dataclass(slots=True, frozen=True)
class PlacementProblem:
    """The size and the bound of one solve, as the placement policy sees it.

    Attributes:
        n: Number of unknowns.
        nnz: Stored entries of the system matrix (``n * n`` for a dense one).
        kind: What bounds one preconditioned iteration.
    """

    n: int
    nnz: int
    kind: Placement


def choose_device(problem: PlacementProblem) -> torch.device:
    """Pick CUDA or CPU for ``problem``; CPU whenever no GPU is visible.

    Args:
        problem (PlacementProblem): Size and bound of the solve.

    Returns:
        torch.device: ``cuda`` only where the GPU is expected to beat the CPU.
    """
    if not torch.cuda.is_available():
        return _CPU
    match problem.kind:
        case Placement.SEQUENTIAL_SWEEP:
            return _CPU
        case Placement.SPMV_BOUND:
            return _CUDA if problem.nnz >= SPMV_GPU_MIN_NNZ else _CPU
        case Placement.DENSE_BOUND:
            return _CUDA if problem.n >= DENSE_GPU_MIN_N else _CPU


def placement_problem(matrix: torch.Tensor, preconditioner: object) -> PlacementProblem:
    """Describe the solve of ``matrix`` with ``preconditioner`` for :func:`choose_device`.

    A preconditioner with sequential sweeps decides the kind alone. Otherwise the matrix
    format does: a strided matrix makes every product dense, a sparse one keeps it
    SpMV-bound. Anything that is not a torchalg preconditioner counts as SpMV-bound.

    Args:
        matrix (torch.Tensor): System matrix.
        preconditioner (object): The solver's preconditioner, if any.

    Returns:
        PlacementProblem: Size and bound of the solve.
    """
    n = matrix.shape[0]
    dense = matrix.layout == torch.strided
    nnz = n * n if dense else matrix._nnz()
    declared = getattr(preconditioner, "placement", Placement.SPMV_BOUND)
    if declared is Placement.SPMV_BOUND and dense:
        return PlacementProblem(n=n, nnz=nnz, kind=Placement.DENSE_BOUND)
    return PlacementProblem(n=n, nnz=nnz, kind=declared)
