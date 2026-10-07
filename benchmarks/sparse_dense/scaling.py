"""Benchmark dense-vs-CSR scaling through N=7,921, plus CSR kernels to 50k.

This is the focused companion to :mod:`run`: that runner compares a
wide operation registry and can reach very large sparse systems; this runner
answers the narrower question of how dense and CSR paths scale in time and
memory on systems that can still be represented densely.  It records:

* elementary resident operations used by PCG and AMG: matvec, Jacobi setup
  and apply, symmetric Gauss-Seidel, and a fixed-rank Galerkin product; and
* end-to-end work: preconditioner construction plus PCG for no
  preconditioner, Jacobi, SA-AMG, and BootCMatch.

Every shared-size cell uses the same SPD lattice-Laplacian system in dense or
CSR form. CSR cells continue through N=49,729 without a dense counterpart.
CUDA cells are emitted only when CUDA is available. They are full-pipeline
cells: A, b, setup, and PCG all reside on CUDA; no transfer is folded into
the timed operation. Every cell is warmed before memory and steady-state
timing; its warmup wall time is recorded separately. Cells whose algorithm
is dominated by host-side Python control flow (the BootCMatch kernels and
the ``pcg_bootcmatch`` end-to-end path) are warmed and timed uncompiled,
since ``torch.compile`` cannot usefully trace them; all other cells are
compiled and warmed before timing.

BootCMatch (compatible-weighted-matching Bootstrap AMG) replaces the
CR-based BAMG this benchmark used to measure: CR-based BAMG's coarsening is
now documented as dense-only in this codebase (its coarsening is
inherently sequential - see ``bootstrap.py``'s module docstring in both
trees), so it is no longer a meaningful sparse-scaling comparison point.
BootCMatch is measured on both formats here since it has a real dense and
sparse sibling, unlike the CR-based preconditioner this replaced.
"""

from __future__ import annotations

import argparse
import csv
import signal
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Self

import torch
from scipy.sparse import csr_matrix
from torch.utils.benchmark import Timer

from torchalg import pcg
from torchalg.preconditioners.implementations import JacobiPreconditioner
from torchalg.preconditioners.implementations.amg import vcycle_amg
from torchalg.preconditioners.implementations.amg.bootcmatch import (
    BootCMatchPreconditioner,
)
from torchalg.preconditioners.implementations.amg.smoothers import (
    GaussSeidelSmoother,
)
from torchalg.sparse.kernels.bootcmatch_matching import bootcmatch_parallel_matching
from torchalg.sparse.kernels.bootcmatch_weights import bootcmatch_edge_weights
from torchalg.sparse.preconditioners.amg.bootcmatch import (
    BootCMatchCoarsening,
)
from torchalg.sparse.preconditioners.amg.bootcmatch import (
    BootCMatchPreconditioner as SparseBootCMatchPreconditioner,
)
from torchalg.sparse.preconditioners.amg.smoothers import (
    GaussSeidelSmoother as SparseGaussSeidelSmoother,
)
from torchalg.sparse.preconditioners.amg.variants import vcycle_amg as sparse_vcycle_amg
from torchalg.sparse.preconditioners.jacobi import (
    JacobiPreconditioner as SparseJacobiPreconditioner,
)

from . import matrices, memory
from . import operations as ops

MAX_SYSTEM_SIZE = 8_000
"""Hard limit for this comparison: dense legs must remain practical."""

MAX_SPARSE_SYSTEM_SIZE = 50_000
"""Hard limit for the sparse-only extension of every benchmark suite."""

DEFAULT_SIZES = [484, 1_024, 2_025, 3_969, 7_921]
"""Exact 2-D grid sizes, all at or below :data:`MAX_SYSTEM_SIZE`."""

SPARSE_ONLY_SIZES = [15_876, 31_684, 49_729]
"""Exact 2-D CSR-only sizes above the dense-safe comparison cap."""

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_MEMORY_GB = 4.0
DEFAULT_ARTIFACTS_DIR = Path("artifacts/benchmarks/sparse_dense")

ELEMENTARY_ALGORITHMS = (
    "matvec",
    "jacobi_setup",
    "jacobi_apply",
    "symmetric_gauss_seidel",
    "galerkin_rank64",
    "bootcmatch_edge_weights",
    "bootcmatch_matching",
    "bootcmatch_coarsen_level",
)
"""Representative PCG/AMG kernels in the focused elementary sweep."""

GALERKIN_RANK = 64
"""Fixed coarse width: represents AMG formation without cubic dense growth."""

END_TO_END_ALGORITHMS = ("pcg", "pcg_jacobi", "pcg_sa_amg", "pcg_bootcmatch")
"""Complete solver/preconditioner paths measured in the focused sweep."""

_HOST_CONTROL_FLOW_ALGORITHMS = frozenset(
    {
        "bootcmatch_matching",
        "bootcmatch_coarsen_level",
        "pcg_bootcmatch",
    }
)
"""Algorithms dominated by host-side Python control flow (greedy loops,
``.tolist()``/``.item()`` calls) that ``torch.compile`` cannot usefully
trace; compiling them only churns through Dynamo's recompile limit and
falls back to eager, so they are run uncompiled instead."""


class _Timeout:
    """Unix wall-clock guard for one benchmark cell."""

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._supported = hasattr(signal, "SIGALRM")

    def __enter__(self) -> Self:
        if self._supported:
            signal.signal(signal.SIGALRM, self._raise)
            signal.setitimer(signal.ITIMER_REAL, self._seconds)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._supported:
            signal.setitimer(signal.ITIMER_REAL, 0)

    @staticmethod
    def _raise(signum: int, frame: object) -> None:
        raise TimeoutError("cell exceeded --timeout-s")


@dataclass(frozen=True)
class Cell:
    """A resident dense/CSR benchmark operation."""

    suite: str
    algorithm: str
    format_label: str
    device: torch.device
    n: int
    build: Callable[[], tuple[Callable[[], object], torch.Tensor, tuple[torch.Tensor, ...]]]


@dataclass(frozen=True)
class Result:
    """A completed or skipped scaling cell, ready for CSV output."""

    suite: str
    algorithm: str
    format_label: str
    device_label: str
    n: int
    compile_warmup_time_s: float | None
    median_time_s: float | None
    matrix_storage_bytes: int | None
    working_set_storage_bytes: int | None
    gpu_peak_bytes: int | None
    cpu_rss_delta_bytes: int | None
    skip_reason: str | None


def devices() -> tuple[torch.device, ...]:
    """Return CPU and, only when present, a real CUDA device."""
    return (
        (torch.device("cpu"), torch.device("cuda"))
        if torch.cuda.is_available()
        else (torch.device("cpu"),)
    )


def _matrix_builder(
    matrix: csr_matrix, fmt: str, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Materialize the requested resident dense or CSR system matrix."""
    if fmt == "dense":
        return matrices.to_torch_dense(matrix, dtype).to(device)
    return matrices.to_torch_sparse_csr(matrix, dtype).to(device)


def _bootcmatch_smooth_vector(n: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Build a deterministic smooth vector for BootCMatch setup kernels."""
    generator = torch.Generator(device=device).manual_seed(0)
    return torch.rand(n, generator=generator, dtype=dtype, device=device)


def _elementary_cells(
    matrix: csr_matrix,
    n: int,
    dtype: torch.dtype,
    device: torch.device,
    formats: tuple[str, ...] = ("dense", "sparse"),
) -> Iterator[Cell]:
    """Yield representative resident PCG/AMG kernel cells for one system/device."""
    for fmt in formats:

        def build_matvec(
            fmt: str = fmt,
        ) -> tuple[Callable[[], object], torch.Tensor, tuple[torch.Tensor, ...]]:
            operand = _matrix_builder(matrix, fmt, dtype, device)
            vector = torch.ones(n, dtype=dtype, device=device)
            return lambda: ops.mv(operand, vector), operand, (operand, vector)

        yield Cell("elementary", "matvec", fmt, device, n, build_matvec)

        def build_jacobi_setup(
            fmt: str = fmt,
        ) -> tuple[Callable[[], object], torch.Tensor, tuple[torch.Tensor, ...]]:
            operand = _matrix_builder(matrix, fmt, dtype, device)
            return lambda: ops.extract_inv_diag(operand), operand, (operand,)

        yield Cell("elementary", "jacobi_setup", fmt, device, n, build_jacobi_setup)

        def build_jacobi_apply(
            fmt: str = fmt,
        ) -> tuple[Callable[[], object], torch.Tensor, tuple[torch.Tensor, ...]]:
            operand = _matrix_builder(matrix, fmt, dtype, device)
            inverse_diagonal = ops.extract_inv_diag(operand)
            residual = torch.ones(n, dtype=dtype, device=device)
            return (
                lambda: ops.elementwise_scale(inverse_diagonal, residual),
                operand,
                (operand, inverse_diagonal, residual),
            )

        yield Cell("elementary", "jacobi_apply", fmt, device, n, build_jacobi_apply)

        def build_symmetric_gauss_seidel(
            fmt: str = fmt,
        ) -> tuple[Callable[[], object], torch.Tensor, tuple[torch.Tensor, ...]]:
            operand = _matrix_builder(matrix, fmt, dtype, device)
            rhs = torch.ones(n, dtype=dtype, device=device)
            initial = torch.zeros(n, dtype=dtype, device=device)
            smoother = GaussSeidelSmoother() if fmt == "dense" else SparseGaussSeidelSmoother()
            return (
                lambda: smoother.smooth(operand, rhs, initial, steps=1),
                operand,
                (operand, rhs, initial),
            )

        yield Cell(
            "elementary",
            "symmetric_gauss_seidel",
            fmt,
            device,
            n,
            build_symmetric_gauss_seidel,
        )

        def build_galerkin(
            fmt: str = fmt,
        ) -> tuple[Callable[[], object], torch.Tensor, tuple[torch.Tensor, ...]]:
            operand = _matrix_builder(matrix, fmt, dtype, device)
            rank = min(GALERKIN_RANK, n)
            prolongation = torch.ones((n, rank), dtype=dtype, device=device) / rank
            if fmt == "dense":
                run = lambda: ops.form_galerkin_dense(prolongation, operand)
            else:
                run = lambda: ops.form_galerkin_spmm(prolongation, operand)
            return run, operand, (operand, prolongation)

        yield Cell("elementary", "galerkin_rank64", fmt, device, n, build_galerkin)

        if fmt != "sparse":
            continue

        def build_bootcmatch_edge_weights(
            fmt: str = fmt,
        ) -> tuple[Callable[[], object], torch.Tensor, tuple[torch.Tensor, ...]]:
            operand = _matrix_builder(matrix, fmt, dtype, device)
            w = _bootcmatch_smooth_vector(n, dtype, device)
            return (
                lambda: bootcmatch_edge_weights(w, operand),
                operand,
                (operand, w),
            )

        yield Cell(
            "elementary", "bootcmatch_edge_weights", fmt, device, n, build_bootcmatch_edge_weights
        )

        def build_bootcmatch_matching(
            fmt: str = fmt,
        ) -> tuple[Callable[[], object], torch.Tensor, tuple[torch.Tensor, ...]]:
            operand = _matrix_builder(matrix, fmt, dtype, device)
            w = _bootcmatch_smooth_vector(n, dtype, device)
            weights, fine_only_mask = bootcmatch_edge_weights(w, operand)
            return (
                lambda: bootcmatch_parallel_matching(weights, fine_only_mask),
                operand,
                (operand, weights, fine_only_mask),
            )

        yield Cell("elementary", "bootcmatch_matching", fmt, device, n, build_bootcmatch_matching)

        def build_bootcmatch_coarsen_level(
            fmt: str = fmt,
        ) -> tuple[Callable[[], object], torch.Tensor, tuple[torch.Tensor, ...]]:
            operand = _matrix_builder(matrix, fmt, dtype, device)
            w = _bootcmatch_smooth_vector(n, dtype, device)
            return (
                lambda: BootCMatchCoarsening(w).build_transfer(operand),
                operand,
                (operand, w),
            )

        yield Cell(
            "elementary", "bootcmatch_coarsen_level", fmt, device, n, build_bootcmatch_coarsen_level
        )


def _pcg_cells(
    matrix: csr_matrix,
    n: int,
    dtype: torch.dtype,
    device: torch.device,
    formats: tuple[str, ...] = ("dense", "sparse"),
) -> Iterator[Cell]:
    """Yield setup-plus-solve PCG cells for equivalent dense and CSR systems."""
    for fmt in formats:
        for algorithm in END_TO_END_ALGORITHMS:

            def build(
                fmt: str = fmt,
                algorithm: str = algorithm,
            ) -> tuple[Callable[[], object], torch.Tensor, tuple[torch.Tensor, ...]]:
                operand = _matrix_builder(matrix, fmt, dtype, device)
                rhs = torch.ones(n, dtype=dtype, device=device)

                def run() -> object:
                    if algorithm == "pcg":
                        preconditioner = None
                    elif algorithm == "pcg_jacobi":
                        # setup() is intentionally inside the timed path for every
                        # algorithm here, now that torchalg's Preconditioner is an
                        # explicit two-phase (setup, apply) contract - this keeps
                        # every branch measuring the same "construct + setup +
                        # solve" cost a real caller pays, Jacobi's trivial setup
                        # included.
                        preconditioner = (
                            JacobiPreconditioner().setup(operand)
                            if fmt == "dense"
                            else SparseJacobiPreconditioner().setup(operand)
                        )
                    elif algorithm == "pcg_sa_amg":
                        preconditioner = (
                            vcycle_amg(operand, n_levels=3).setup(operand)
                            if fmt == "dense"
                            else sparse_vcycle_amg(operand, n_levels=3).setup(operand)
                        )
                    else:
                        # BootCMatch setup is intentionally inside the timed
                        # path: unlike Jacobi, its bootstrap hierarchy is the
                        # cost a caller pays before the first PCG iteration.
                        # max_levels=5 caps the benchmark's own hierarchy
                        # depth (the library default stays 10) - keeps this
                        # sweep's larger sparse-only sizes from building more
                        # levels than a solve actually benefits from.
                        preconditioner = (
                            BootCMatchPreconditioner(seed=0, max_levels=5).setup(operand)
                            if fmt == "dense"
                            else SparseBootCMatchPreconditioner(seed=0, max_levels=5).setup(operand)
                        )
                    return pcg(
                        operand,
                        rhs,
                        preconditioner=preconditioner,
                        rtol=1e-6,
                        maxiter=min(500, n),
                        device=device,
                    )

                return run, operand, (operand, rhs)

            yield Cell("end_to_end", algorithm, fmt, device, n, build)


def build_grid(
    sizes: list[int], sparse_only_sizes: list[int], dtype: torch.dtype
) -> Iterator[Cell]:
    """Build shared dense/CSR cells plus sparse-only extension cells."""
    for n in sizes:
        if n > MAX_SYSTEM_SIZE:
            raise ValueError(f"N={n} exceeds this benchmark's cap of {MAX_SYSTEM_SIZE}")
        matrix, actual_n = matrices.synthetic_matrix(n, dims=2)
        if actual_n > MAX_SYSTEM_SIZE:
            raise ValueError(f"generated N={actual_n} exceeds cap of {MAX_SYSTEM_SIZE}")
        for device in devices():
            yield from _elementary_cells(matrix, actual_n, dtype, device)
            yield from _pcg_cells(matrix, actual_n, dtype, device)
    for n in sparse_only_sizes:
        if n > MAX_SPARSE_SYSTEM_SIZE:
            raise ValueError(
                f"sparse-only N={n} exceeds this benchmark's cap of {MAX_SPARSE_SYSTEM_SIZE}"
            )
        matrix, actual_n = matrices.synthetic_matrix(n, dims=2)
        if actual_n > MAX_SPARSE_SYSTEM_SIZE:
            raise ValueError(
                f"generated sparse-only N={actual_n} exceeds cap of {MAX_SPARSE_SYSTEM_SIZE}"
            )
        for device in devices():
            yield from _elementary_cells(matrix, actual_n, dtype, device, formats=("sparse",))
            yield from _pcg_cells(matrix, actual_n, dtype, device, formats=("sparse",))


def run_cell(cell: Cell, timeout_s: float) -> Result:
    """Compile then measure one cell's storage, execution peak, and steady state.

    A bad algorithm configuration is a benchmark bug, not an unavailable
    configuration: ``ValueError`` and unexpected ``RuntimeError`` propagate
    rather than being disguised as a skipped row.  Only a PyTorch operation
    that explicitly has no backend implementation is recorded as unavailable.

    ``torch.compile`` is lazy: wrapping a callable does not do the expensive
    trace/code-generation work. ``compile_warmup_time_s`` therefore measures
    the wrapper creation *and* its first invocation. The following memory
    and ``Timer`` calls use that already-warmed compiled callable.
    """
    try:
        with _Timeout(timeout_s):
            run, operand, working_set = cell.build()
            matrix_storage = memory.storage_bytes(operand)
            working_set_storage = sum(memory.storage_bytes(tensor) for tensor in working_set)
            compile_started = perf_counter()
            compiled_run = (
                run if cell.algorithm in _HOST_CONTROL_FLOW_ALGORITHMS else torch.compile(run)
            )
            compiled_run()
            if cell.device.type == "cuda":
                torch.cuda.synchronize(cell.device)
            compile_warmup_time = perf_counter() - compile_started
            gpu_reading = memory.MemoryReading(peak_bytes=None, approximate=False)
            cpu_reading = memory.MemoryReading(peak_bytes=None, approximate=True)
            if cell.device.type == "cuda":
                with memory.gpu_peak_memory(cell.device) as gpu_reading:
                    compiled_run()
            else:
                with memory.cpu_peak_rss_delta() as cpu_reading:
                    compiled_run()
            median_time = (
                Timer(stmt="compiled_run()", globals={"compiled_run": compiled_run})
                .blocked_autorange()
                .median
            )
    except TimeoutError:
        return Result(
            cell.suite,
            cell.algorithm,
            cell.format_label,
            cell.device.type,
            cell.n,
            None,
            None,
            None,
            None,
            None,
            None,
            "timeout",
        )
    except RuntimeError as error:
        # PyTorch Inductor can catch our SIGALRM while compiling and rethrow
        # it as ``InductorError``. Preserve the harness outcome rather than
        # mislabeling a real timeout as a compiler/backend failure.
        if "TimeoutError: cell exceeded --timeout-s" not in str(error):
            raise
        return Result(
            cell.suite,
            cell.algorithm,
            cell.format_label,
            cell.device.type,
            cell.n,
            None,
            None,
            None,
            None,
            None,
            None,
            "timeout",
        )
    except NotImplementedError as error:
        return Result(
            cell.suite,
            cell.algorithm,
            cell.format_label,
            cell.device.type,
            cell.n,
            None,
            None,
            None,
            None,
            None,
            None,
            f"backend unavailable: {error}"[:200],
        )
    return Result(
        cell.suite,
        cell.algorithm,
        cell.format_label,
        cell.device.type,
        cell.n,
        compile_warmup_time,
        median_time,
        matrix_storage,
        working_set_storage,
        gpu_reading.peak_bytes,
        cpu_reading.peak_bytes,
        None,
    )


def write_results(results: list[Result], output: Path) -> None:
    """Write a stable CSV shared with :mod:`scaling_report`."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "suite",
                "algorithm",
                "format",
                "device",
                "n",
                "compile_warmup_time_s",
                "median_time_s",
                "matrix_storage_bytes",
                "working_set_storage_bytes",
                "gpu_peak_bytes",
                "cpu_rss_delta_bytes",
                "skip_reason",
            ]
        )
        writer.writerows(
            (
                result.suite,
                result.algorithm,
                result.format_label,
                result.device_label,
                result.n,
                result.compile_warmup_time_s,
                result.median_time_s,
                result.matrix_storage_bytes,
                result.working_set_storage_bytes,
                result.gpu_peak_bytes,
                result.cpu_rss_delta_bytes,
                result.skip_reason,
            )
            for result in results
        )


def main() -> None:
    """Run the capped scaling grid and write its CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--sparse-only-sizes", type=int, nargs="+", default=SPARSE_ONLY_SIZES)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS_DIR / "scaling.csv")
    args = parser.parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    started = perf_counter()
    results: list[Result] = []
    for cell in build_grid(args.sizes, args.sparse_only_sizes, dtype):
        result = run_cell(cell, args.timeout_s)
        results.append(result)
        status = result.skip_reason or f"{result.median_time_s:.3e}s"
        print(
            f"{result.suite:<12} {result.algorithm:<14} {result.format_label:<7} "
            f"{result.device_label:<4} n={result.n:<5} {status}"
        )
    write_results(results, args.output)
    print(f"\n{len(results)} cells in {perf_counter() - started:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
