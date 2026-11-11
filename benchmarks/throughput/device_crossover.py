"""CPU against GPU per operation, as a function of n, using the best working path on each.

Run: ``uv run python -m benchmarks.throughput.device_crossover`` on a machine with a GPU.
The CPU side uses the paths torchalg picks there (int32 CSR SpMV, MKL triangular solve);
the GPU side uses COO SpMV and the level-scheduled triangular solve, because on gfx1030
with the rocm7.2 wheels CSR SpMV and the CSR library triangular solve fault (see
``docs/implementations/device-and-gpu.md``). Times are medians in milliseconds, with the
GPU synchronized around every call.

Ops:
- ``spmv``: one sparse matrix-vector product.
- ``pcg-iter``: one Jacobi-PCG iteration's arithmetic (SpMV, two dots, three axpys, one
  diagonal scaling, one norm), without host syncs.
- ``ic0-apply``: IC(0) forward plus backward triangular solve.
- ``dense-gemv``: dense ``A @ x`` with the densified matrix (n <= DENSE_MAX_N).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from benchmarks.cases import MatrixCase, box3d_case, lap2d_case
from benchmarks.harness import time_callable
from torchalg.sparse.kernels.index_dtype import compact_csr_indices
from torchalg.sparse.kernels.triangular import PreparedTriangularSolve
from torchalg.sparse.preconditioners.ic0 import sparse_ic0

CASES = (
    box3d_case(2744),
    lap2d_case(10000),
    box3d_case(27000),
    lap2d_case(90000),
    lap2d_case(250000),
)
DENSE_MAX_N = 12000
REPEATS = 20
CPU = torch.device("cpu")
GPU = torch.device("cuda")


def _time(fn: Callable[[], object], device: torch.device) -> float:
    return time_callable(fn, device=device, repeats=REPEATS)[0] * 1e3


def _pcg_iteration(
    matvec: Callable[[torch.Tensor], torch.Tensor], inv_diag: torch.Tensor, n: int
) -> Callable[[], torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    r, d, u = (
        torch.randn(n, dtype=torch.float64, generator=generator).to(inv_diag.device)
        for _ in range(3)
    )

    def step() -> torch.Tensor:
        w = inv_diag * r
        q = matvec(d)
        alpha = torch.dot(r, w) / torch.dot(d, q)
        u_new = u + alpha * d
        r_new = r - alpha * q
        return torch.linalg.norm(r_new) + u_new[0]

    return step


def _report(case: MatrixCase, op: str, cpu_ms: float, gpu_ms: float) -> None:
    winner = "GPU" if gpu_ms < cpu_ms else "CPU"
    ratio = max(cpu_ms, gpu_ms) / min(cpu_ms, gpu_ms)
    print(
        f"{case.id:15} {op:11} cpu {cpu_ms:9.3f} ms  gpu {gpu_ms:9.3f} ms  {winner} {ratio:5.1f}x"
    )


def _run_case(case: MatrixCase, generator: torch.Generator) -> None:
    csr = compact_csr_indices(case.csr(torch.float64, CPU))
    coo_gpu = case.csr(torch.float64, CPU).to_sparse_coo().coalesce().to(GPU)
    x_cpu = torch.randn(case.n, dtype=torch.float64, generator=generator)
    x_gpu = x_cpu.to(GPU)
    _report(case, "spmv", _time(lambda: csr @ x_cpu, CPU), _time(lambda: coo_gpu @ x_gpu, GPU))

    diag = torch.ones(case.n, dtype=torch.float64)
    _report(
        case,
        "pcg-iter",
        _time(_pcg_iteration(lambda v: csr @ v, diag, case.n), CPU),
        _time(_pcg_iteration(lambda v: coo_gpu @ v, diag.to(GPU), case.n), GPU),
    )

    lower = sparse_ic0(case.csr(torch.float64, CPU))
    upper = lower.to_sparse_coo().t().to_sparse_csr()
    forward = PreparedTriangularSolve(lower, "forward").to(GPU)
    backward = PreparedTriangularSolve(upper, "backward").to(GPU)
    column = x_cpu.unsqueeze(-1)
    _report(
        case,
        "ic0-apply",
        _time(
            lambda: (
                torch.triangular_solve(
                    torch.triangular_solve(column, lower, upper=False).solution,
                    upper,
                    upper=True,
                ).solution
            ),
            CPU,
        ),
        _time(lambda: backward.solve(forward.solve(x_gpu)), GPU),
    )

    if case.n <= DENSE_MAX_N:
        dense_cpu = case.dense(torch.float64, CPU)
        dense_gpu = dense_cpu.to(GPU)
        _report(
            case,
            "dense-gemv",
            _time(lambda: dense_cpu @ x_cpu, CPU),
            _time(lambda: dense_gpu @ x_gpu, GPU),
        )


def main() -> None:
    generator = torch.Generator().manual_seed(0)
    for case in CASES:
        _run_case(case, generator)


if __name__ == "__main__":
    main()
