"""Sparse-kernel backend probes on one device.

Run: ``uv run python -m benchmarks.throughput.sparse_backends [--device cuda] [--rows ...]``
(ROCm devices report as ``cuda``). Every row times a baseline and a candidate on identical
inputs and prints their max relative difference, so a fast but wrong candidate shows.

Rows (``--rows`` selects; the CSR SpMV rows abort the process on gfx1030 with the rocm7.2
wheels, a rocSPARSE memory fault, so skip them there):

- ``trsv``: level-scheduled IC(0) apply (``PreparedTriangularSolve`` pair) against
  ``torch.triangular_solve`` on the CSR factors, k = 1 and 8 right-hand sides.
- ``spmv-idx``: CSR ``A @ x`` with int64 against int32 indices.
- ``spmm``: CSR ``A @ X`` (k = 8) against a loop of 8 SpMVs.
- ``spmv-coo``: CSR ``A @ x`` against COO ``A @ x``.
- ``spmv-gather``: COO ``A @ x`` against a gather + ``index_add_`` product.
- ``dense-ic0``: ``torch.cholesky_solve`` against two ``solve_triangular`` calls.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from benchmarks.cases import MatrixCase, box3d_case, lap2d_case
from benchmarks.harness import time_callable
from torchalg.sparse.kernels.index_dtype import compact_csr_indices
from torchalg.sparse.kernels.triangular import PreparedTriangularSolve
from torchalg.sparse.preconditioners.ic0 import sparse_ic0

CASES = (box3d_case(2744), lap2d_case(10000), box3d_case(27000), lap2d_case(90000))
ALL_ROWS = ("trsv", "spmv-idx", "spmm", "spmv-coo", "spmv-gather", "dense-ic0")
DENSE_IC0_MAX_N = 6000
BLOCK_COLUMNS = (1, 8)
REPEATS = 20


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def _row(
    name: str,
    case: MatrixCase,
    base: Callable[[], torch.Tensor],
    cand: Callable[[], torch.Tensor],
    device: torch.device,
    note: str = "",
) -> None:
    try:
        reference = base()
        candidate = cand()
    except (RuntimeError, NotImplementedError) as error:
        print(f"{name:11} {case.id:14} unsupported: {str(error).splitlines()[0][:70]}")
        return
    t_base = time_callable(base, device=device, repeats=REPEATS)[0] * 1e3
    t_cand = time_callable(cand, device=device, repeats=REPEATS)[0] * 1e3
    print(
        f"{name:11} {case.id:14} {note:5} base {t_base:9.3f} ms  cand {t_cand:9.3f} ms  "
        f"speed-up {t_base / t_cand:7.1f}x  rel {_rel(candidate, reference):.1e}"
    )


def _gather_matvec(coo: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    rows, cols = coo.indices()
    values = coo.values()
    n = coo.shape[0]
    return lambda x: torch.zeros(n, dtype=x.dtype, device=x.device).index_add_(
        0, rows, values * x[cols]
    )


def _run_case(
    case: MatrixCase, device: torch.device, rows: set[str], generator: torch.Generator
) -> None:
    cpu_matrix = case.csr(torch.float64, torch.device("cpu"))
    x = torch.randn(case.n, dtype=torch.float64, generator=generator).to(device)
    if "trsv" in rows:
        lower = sparse_ic0(cpu_matrix)
        upper = lower.to_sparse_coo().t().to_sparse_csr()
        forward = PreparedTriangularSolve(lower, "forward").to(device)
        backward = PreparedTriangularSolve(upper, "backward").to(device)
        lower_d, upper_d = lower.to(device), upper.to(device)
        for k in BLOCK_COLUMNS:
            rhs = torch.randn(case.n, k, dtype=torch.float64, generator=generator).to(device)
            _row(
                "trsv",
                case,
                lambda r=rhs: backward.solve(forward.solve(r)),
                lambda r=rhs: (
                    torch.triangular_solve(
                        torch.triangular_solve(r, lower_d, upper=False).solution,
                        upper_d,
                        upper=True,
                    ).solution
                ),
                device,
                f"k={k}",
            )
    matrix = cpu_matrix.to(device)
    matrix32 = compact_csr_indices(matrix)
    coo = cpu_matrix.to_sparse_coo().coalesce().to(device)
    if "spmv-idx" in rows:
        _row("spmv-idx", case, lambda: matrix @ x, lambda: matrix32 @ x, device)
    if "spmm" in rows:
        block = torch.randn(case.n, 8, dtype=torch.float64, generator=generator).to(device)
        _row(
            "spmm",
            case,
            lambda: matrix32 @ block,
            lambda: torch.stack([matrix32 @ block[:, j] for j in range(8)], dim=1),
            device,
            "k=8",
        )
    if "spmv-coo" in rows:
        _row("spmv-coo", case, lambda: matrix32 @ x, lambda: coo @ x, device)
    if "spmv-gather" in rows:
        gather = _gather_matvec(coo)
        _row("spmv-gather", case, lambda: coo @ x, lambda: gather(x), device)
    if "dense-ic0" in rows and case.n <= DENSE_IC0_MAX_N:
        dense_factor = torch.linalg.cholesky(cpu_matrix.to_dense()).to(device)
        column = x.unsqueeze(-1)
        _row(
            "dense-ic0",
            case,
            lambda: torch.cholesky_solve(column, dense_factor),
            lambda: torch.linalg.solve_triangular(
                dense_factor.mT,
                torch.linalg.solve_triangular(dense_factor, column, upper=False),
                upper=True,
            ),
            device,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rows", nargs="*", default=list(ALL_ROWS), choices=ALL_ROWS)
    args = parser.parse_args()
    device = torch.device(args.device)
    rows = set(args.rows)
    generator = torch.Generator().manual_seed(0)
    for case in CASES:
        _run_case(case, device, rows, generator)


if __name__ == "__main__":
    main()
