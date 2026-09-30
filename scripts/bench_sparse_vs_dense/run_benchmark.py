#!/usr/bin/env python3
"""CLI: build the (operation x format x device x N) grid and time/measure every surviving cell.

Cartesian-product reductions applied before any cell runs (see
``scripts/README.md`` for the full rationale):

1. GPU legs only run when ``torch.cuda.is_available()`` - otherwise skipped
   for lack of hardware, not eliminated by theory.
2. Dense legs of the three families that are provably O(N^3) at full size
   (Galerkin product, factorization, full eigh) are capped at
   ``--dense-cubic-max-n`` and never attempted above it - a known
   ~20GB/hours-long blowup at N=50k, not something worth discovering via a
   live timeout. ``ic0`` gets a much lower, non-CLI-configurable cap of its
   own (``DENSE_CUBIC_MAX_N_OVERRIDES``) - unlike the other two capped ops,
   it's a Python-level loop rather than one BLAS/LAPACK call, and its
   measured wall time grows far worse than N^3 in practice.
3. A sparse/native leg that isn't available on this build/device (e.g.
   ``torch.sparse.spsolve`` on CPU, or on CUDA without a cuDSS-enabled
   PyTorch build - see ``operations.py`` for both) is caught and logged as
   a skip, not a crash.

Every surviving cell is timed with ``torch.utils.benchmark.Timer``
(handles warm-up, repetition statistics, and CUDA synchronization
correctly - no hand-rolled timing loop) and measured for memory via
``memory.py``, then written to one CSV row. Transfer/conversion costs
(``h2d``/``d2h``/``to_sparse``/``to_dense``) are timed as their own rows,
independent of any compute cell, per the same reasoning.
"""

from __future__ import annotations

import argparse
import csv
import signal
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Self

import torch
from scipy.sparse import csr_matrix
from scipy.sparse import tril as scipy_tril
from torch.utils.benchmark import Timer

sys.path.insert(0, str(Path(__file__).parents[1]))

from bench_sparse_vs_dense import matrices, memory
from bench_sparse_vs_dense import operations as ops

DEFAULT_SIZES = [500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 50_000]
DEFAULT_K_VALUES = [1, 5, 20, 100]
DEFAULT_DENSE_CUBIC_MAX_N = 8_000
DEFAULT_MAX_MEMORY_GB = 8.0
DEFAULT_TIMEOUT_S = 60.0
RESULTS_DIR = Path(__file__).with_name("results")

# Specific ops provably O(N^3) at full matrix size - these are the only
# ones a static size cap applies to (see module docstring point 2). Keyed
# by op, not family: a family can mix a capped dense-formation leg
# ("formed_dense") with an uncapped one that happens to also use dense
# operands ("matrix_free_dense" is O(N^2), not O(N^3) - not capped;
# "matrix_free_sparse" needs no dense A at all). Every other op runs the
# full sweep.
DENSE_CUBIC_OPS = {"formed_dense", "ic0", "eigvalsh"}

# Per-op override of --dense-cubic-max-n, for ops whose measured cost grows
# far worse than the other DENSE_CUBIC_OPS members at the same N.
# "formed_dense"/"eigvalsh" are single BLAS/LAPACK calls, so the general
# cap (tuned for an N^3 blowup, see module docstring point 2) fits them.
# "ic0" (``dense_ic0``) is a Python-level ``for k in range(n)`` loop over
# per-step tensor ops, not one fused kernel - measured wall time at
# float64 (0.19s @ n=1024, 3.7s @ n=2048, 18.5s @ n=3000) grows far faster
# than N^3 in practice, so it needs its own, much lower cap rather than
# sharing the 8000 default meant for a single BLAS/LAPACK call.
DENSE_CUBIC_MAX_N_OVERRIDES = {"ic0": 2_000}

# Rough multiplier on a matrix's own storage size for the peak working
# memory a family needs (factorization/eigh/svd allocate several
# same-sized intermediates; mv/elementwise need none beyond the operands
# themselves) - used only for the pre-flight memory estimate, not for
# anything timed.
WORKING_MEMORY_MULTIPLIER = {
    "mv": 1,
    "elementwise": 1,
    "galerkin_form": 2,
    "triangular_apply": 1,
    "factorize": 3,
    "eigh": 4,
    "svd": 4,
}


class _Timeout:
    """Unix ``SIGALRM``-based wall-clock timeout for a single cell.

    stdlib only, no subprocess-per-cell overhead. Not available on
    non-Unix platforms - degrades to "no timeout" there rather than
    failing, since this is a backstop guard (see module docstring point 2
    for why static caps, not this, are the primary defense).
    """

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


@dataclass
class Cell:
    """One (operation, format, device, N[, K]) grid cell.

    Attributes:
        family: Operation family name (matches keys used for static caps).
        op: Specific operation label (e.g. ``"mv"``, ``"lobpcg"``).
        format_label: ``"dense"``, ``"sparse"``, or ``"scipy"``.
        device_label: ``"cpu"`` or ``"cuda"``.
        n: Matrix size for this cell.
        build: Zero-arg factory returning a zero-arg callable to time, plus
            the primary operand tensor (for storage-size measurement).
        k: Reuse count, only meaningful for ``family == "galerkin_form"``
            (formed-vs-matrix-free comparison); ``None`` otherwise.
    """

    family: str
    op: str
    format_label: str
    device_label: str
    n: int
    build: Callable[[], tuple[Callable[[], object], torch.Tensor]]
    k: int | None = None


@dataclass
class CellResult:
    """One CSV row: a cell's outcome, timed or skipped."""

    family: str
    op: str
    format_label: str
    device_label: str
    n: int
    k: int | None
    median_time_s: float | None
    storage_bytes: int | None
    gpu_peak_bytes: int | None
    cpu_rss_delta_bytes: int | None
    skip_reason: str | None


def devices() -> list[torch.device]:
    """CPU always; CUDA only if actually available on this machine."""
    result = [torch.device("cpu")]
    if torch.cuda.is_available():
        result.append(torch.device("cuda"))
    return result


def estimate_memory_bytes(n: int, family: str, dtype: torch.dtype) -> int:
    """Pre-flight estimate of a dense cell's peak working memory, for the memory guard.

    Args:
        n: Matrix size.
        family: Operation family (looked up in ``WORKING_MEMORY_MULTIPLIER``).
        dtype: Element dtype.

    Returns:
        int: Estimated bytes.
    """
    multiplier = WORKING_MEMORY_MULTIPLIER.get(family, 1)
    return n * n * torch.tensor([], dtype=dtype).element_size() * multiplier


def run_cell(cell: Cell, *, timeout_s: float, device_for_memory: torch.device) -> CellResult:
    """Time and memory-measure one cell, catching unsupported-API and timeout failures.

    Args:
        cell: Grid cell to run.
        timeout_s: Per-cell wall-clock timeout.
        device_for_memory: Device to measure GPU peak memory on (only used
            when ``cell.device_label == "cuda"``).

    Returns:
        CellResult: Timed result, or a result with ``skip_reason`` set.
    """
    try:
        with _Timeout(timeout_s):
            fn, operand = cell.build()
            storage = memory.storage_bytes(operand)

            gpu_reading = memory.MemoryReading(peak_bytes=None, approximate=False)
            cpu_reading = memory.MemoryReading(peak_bytes=None, approximate=True)
            if cell.device_label == "cuda":
                with memory.gpu_peak_memory(device_for_memory) as gpu_reading:
                    fn()
            else:
                with memory.cpu_peak_rss_delta() as cpu_reading:
                    fn()

            median_time = Timer(stmt="fn()", globals={"fn": fn}).blocked_autorange().median
    except TimeoutError:
        return CellResult(
            cell.family,
            cell.op,
            cell.format_label,
            cell.device_label,
            cell.n,
            cell.k,
            None,
            None,
            None,
            None,
            "timeout",
        )
    except (NotImplementedError, RuntimeError, ValueError) as error:
        return CellResult(
            cell.family,
            cell.op,
            cell.format_label,
            cell.device_label,
            cell.n,
            cell.k,
            None,
            None,
            None,
            None,
            f"unsupported: {error}"[:200],
        )

    return CellResult(
        cell.family,
        cell.op,
        cell.format_label,
        cell.device_label,
        cell.n,
        cell.k,
        median_time,
        storage,
        gpu_reading.peak_bytes,
        cpu_reading.peak_bytes,
        None,
    )


def _cells_for_n_and_device(
    scipy_matrix: csr_matrix,
    n: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
    k_values: list[int],
) -> Iterator[Cell]:
    """Every cell for one (N, device) combination.

    Factored out of ``build_grid`` specifically so ``scipy_matrix``/``n``/
    ``rank``/``device`` are real parameters of a single call, not variables
    of an enclosing loop - every nested ``def build_*():`` below closes
    over them safely as a result (closing over a loop variable directly
    would silently capture whatever value the loop last left it at, if this
    generator were ever consumed by collecting all cells into a list before
    running them instead of one at a time).

    Args:
        scipy_matrix: This N's synthetic lattice-Laplacian matrix.
        n: Actual matrix size.
        rank: Coarsening rank for the Galerkin-product cells.
        device: Target device for this batch of cells.
        dtype: Element dtype for all cells.
        k_values: Reuse counts for the family-3 formed-vs-matrix-free split.

    Yields:
        Cell: One cell per (family, format, N[, K]) combination on this device.
    """
    device_label = device.type

    # --- Family 1: mv ---
    for fmt, matrix_builder in (
        ("dense", lambda: matrices.to_torch_dense(scipy_matrix, dtype).to(device)),
        ("sparse", lambda: matrices.to_torch_sparse_csr(scipy_matrix, dtype).to(device)),
    ):

        def build(matrix_builder=matrix_builder):
            matrix = matrix_builder()
            vector = torch.randn(matrix.shape[0], dtype=dtype, device=matrix.device)
            return (lambda: ops.mv(matrix, vector)), matrix

        yield Cell("mv", "mv", fmt, device_label, n, build)

    # --- Family 2: elementwise (Jacobi apply) ---
    for fmt, matrix_builder in (
        ("dense", lambda: matrices.to_torch_dense(scipy_matrix, dtype).to(device)),
        ("sparse", lambda: matrices.to_torch_sparse_csr(scipy_matrix, dtype).to(device)),
    ):

        def build(matrix_builder=matrix_builder):
            matrix = matrix_builder()
            inv_diag = ops.extract_inv_diag(matrix)
            residual = torch.randn(matrix.shape[0], dtype=dtype, device=matrix.device)
            return (lambda: ops.elementwise_scale(inv_diag, residual)), matrix

        yield Cell("elementwise", "jacobi_apply", fmt, device_label, n, build)

    # --- Family 3: Galerkin form + apply, per K (formed) / matrix-free ---
    # "formed_dense"/"matrix_free_dense" both need the full dense A
    # materialized as setup, so both are subject to the same N cap as any
    # other dense O(N^3)-adjacent leg (the cap check in `main` keys off
    # `cell.op`, not the whole family, precisely so a family can mix
    # capped and uncapped ops). "matrix_free_sparse" uses the sparse A
    # instead - no dense materialization, no cap - this is the leg that
    # actually demonstrates matrix-free's large-N advantage, since
    # "matrix_free_dense" would otherwise need to build a 20GB dense A at
    # N=50k just to set up the comparison, defeating its own point.
    for k in k_values:

        def build_formed_dense(k=k):
            dense = matrices.to_torch_dense(scipy_matrix, dtype).to(device)
            p = torch.randn(n, rank, dtype=dtype, device=device)
            vectors = torch.randn(rank, k, dtype=dtype, device=device)

            def run():
                coarse = ops.form_galerkin_dense(p, dense)
                for i in range(k):
                    ops.apply_formed(coarse, vectors[:, i])

            return run, dense

        yield Cell(
            "galerkin_form", "formed_dense", "dense", device_label, n, build_formed_dense, k=k
        )

        def build_matrix_free_dense(k=k):
            dense = matrices.to_torch_dense(scipy_matrix, dtype).to(device)
            p = torch.randn(n, rank, dtype=dtype, device=device)
            vectors = torch.randn(rank, k, dtype=dtype, device=device)

            def run():
                for i in range(k):
                    ops.apply_matrix_free(p, dense, vectors[:, i])

            return run, dense

        yield Cell(
            "galerkin_form",
            "matrix_free_dense",
            "dense",
            device_label,
            n,
            build_matrix_free_dense,
            k=k,
        )

        def build_matrix_free_sparse(k=k):
            sparse = matrices.to_torch_sparse_csr(scipy_matrix, dtype).to(device)
            p = torch.randn(n, rank, dtype=dtype, device=device)
            vectors = torch.randn(rank, k, dtype=dtype, device=device)

            def run():
                for i in range(k):
                    ops.apply_matrix_free(p, sparse, vectors[:, i])

            return run, sparse

        yield Cell(
            "galerkin_form",
            "matrix_free_sparse",
            "sparse",
            device_label,
            n,
            build_matrix_free_sparse,
            k=k,
        )

    def build_formed_spmm():
        sparse = matrices.to_torch_sparse_csr(scipy_matrix, dtype).to(device)
        p = torch.randn(n, rank, dtype=dtype, device=device)
        return (lambda: ops.form_galerkin_spmm(p, sparse)), sparse

    yield Cell("galerkin_form", "formed_spmm", "sparse", device_label, n, build_formed_spmm)

    def build_formed_spgemm():
        sparse = matrices.to_torch_sparse_csr(scipy_matrix, dtype).to(device)
        p_sparse = matrices.sparse_prolongation(n, rank, dtype=dtype).to(device)
        return (lambda: ops.form_galerkin_spgemm(p_sparse, sparse)), sparse

    yield Cell("galerkin_form", "formed_spgemm", "sparse", device_label, n, build_formed_spgemm)

    # --- Family 4: triangular apply (factor already computed) ---
    # The matrix's own lower triangle stands in for "an already-computed
    # Cholesky-style factor": same shape/bandwidth, and a positive
    # diagonal (true for this SPD lattice Laplacian), so it's a valid
    # operand for timing the O(N^2) apply - computing a *real* Cholesky
    # factor here would cost O(N^3) just to set up a benchmark meant to
    # run the full 50k sweep uncapped, which defeats the point of family 4
    # being apply-only.
    def build_triangular_dense():
        dense = matrices.to_torch_dense(scipy_matrix, dtype).to(device)
        factor = torch.tril(dense)
        residual = torch.randn(n, dtype=dtype, device=device)
        return (lambda: ops.triangular_apply_dense(factor, residual)), factor

    yield Cell(
        "triangular_apply", "cholesky_solve", "dense", device_label, n, build_triangular_dense
    )

    def build_triangular_sparse():
        factor_sparse = matrices.to_torch_sparse_csr(scipy_tril(scipy_matrix).tocsr(), dtype).to(
            device
        )
        residual = torch.randn(n, dtype=dtype, device=device)
        return (lambda: ops.triangular_apply_sparse(factor_sparse, residual)), factor_sparse

    yield Cell("triangular_apply", "spsolve", "sparse", device_label, n, build_triangular_sparse)

    # --- Family 5: factorization (dense capped by op, scipy uncapped) ---
    def build_ic0():
        dense = matrices.to_torch_dense(scipy_matrix, dtype).to(device)
        return (lambda: ops.factorize_ic0(dense)), dense

    yield Cell("factorize", "ic0", "dense", device_label, n, build_ic0)

    def build_spilu():
        sparse = matrices.to_torch_sparse_csr(scipy_matrix, dtype).to(device)
        csc = ops.to_scipy_csc(sparse)
        return (lambda: ops.factorize_spilu(csc)), sparse

    yield Cell("factorize", "spilu", "scipy", "cpu", n, build_spilu)

    # --- Family 6: eigendecomposition (dense capped by op, lobpcg/scipy uncapped) ---
    def build_eigh_dense():
        dense = matrices.to_torch_dense(scipy_matrix, dtype).to(device)
        return (lambda: ops.eigh_dense(dense)), dense

    yield Cell("eigh", "eigvalsh", "dense", device_label, n, build_eigh_dense)

    def build_lobpcg():
        sparse = matrices.to_torch_sparse_csr(scipy_matrix, dtype).to(device)
        return (lambda: ops.eigh_lobpcg(sparse, k=5)), sparse

    yield Cell("eigh", "lobpcg", "sparse", device_label, n, build_lobpcg)

    def build_eigh_scipy():
        sparse = matrices.to_torch_sparse_csr(scipy_matrix, dtype).to(device)
        csc = ops.to_scipy_csc(sparse)
        return (lambda: ops.eigh_scipy(csc.tocsr(), k=5)), sparse

    yield Cell("eigh", "eigsh", "scipy", "cpu", n, build_eigh_scipy)

    # --- Family 7: SVD (thin snapshot matrix, uncapped - near-linear in N) ---
    n_snapshots = 50

    def build_svd_dense():
        snapshots = torch.randn(n_snapshots, n, dtype=dtype, device=device)
        return (lambda: ops.svd_dense(snapshots)), snapshots

    yield Cell("svd", "svd", "dense", device_label, n, build_svd_dense)

    def build_svd_lowrank():
        snapshots = torch.randn(n_snapshots, n, dtype=dtype, device=device)
        return (lambda: ops.svd_lowrank_native(snapshots, rank=10)), snapshots

    yield Cell("svd", "svd_lowrank", "dense", device_label, n, build_svd_lowrank)

    def build_svd_scipy():
        import scipy.sparse as scipy_sparse_ns

        snapshots = scipy_sparse_ns.random(n_snapshots, n, density=1.0, format="csr")
        return (lambda: ops.svd_scipy(snapshots, rank=10)), torch.empty(0)

    yield Cell("svd", "svds", "scipy", "cpu", n, build_svd_scipy)

    # --- Standalone transfer/conversion costs ---
    def build_h2d():
        dense = matrices.to_torch_dense(scipy_matrix, dtype)
        return (lambda: ops.h2d(dense, device)), dense

    yield Cell("transfer", "h2d", "dense", device_label, n, build_h2d)

    def build_to_sparse():
        dense = matrices.to_torch_dense(scipy_matrix, dtype).to(device)
        return (lambda: ops.to_sparse(dense)), dense

    yield Cell("convert", "to_sparse", "dense", device_label, n, build_to_sparse)

    def build_to_dense():
        sparse = matrices.to_torch_sparse_csr(scipy_matrix, dtype).to(device)
        return (lambda: ops.to_dense(sparse)), sparse

    yield Cell("convert", "to_dense", "sparse", device_label, n, build_to_dense)


def build_grid(
    sizes: list[int], k_values: list[int], dims: int, dtype: torch.dtype
) -> Iterator[Cell]:
    """Construct every grid cell by sweeping N and device over :func:`_cells_for_n_and_device`.

    The static O(N^3) dense-leg cap is applied by the caller (``main`` - see
    module docstring point 2), not here.

    Args:
        sizes: Target matrix sizes to sweep.
        k_values: Reuse counts for the family-3 formed-vs-matrix-free split.
        dims: Grid dimensionality for the synthetic lattice Laplacian.
        dtype: Element dtype for all cells.

    Yields:
        Cell: One cell per (family, format, device, N[, K]) combination.
    """
    for target_n in sizes:
        scipy_matrix, n = matrices.synthetic_matrix(target_n, dims=dims)
        rank = max(2, n // 8)  # a representative coarsening ratio, not tuned per-N
        for device in devices():
            yield from _cells_for_n_and_device(scipy_matrix, n, rank, device, dtype, k_values)


def write_results(results: list[CellResult], output_path: Path) -> None:
    """Write every cell result (timed or skipped) to a CSV.

    Args:
        results: All cell results, in the order they were produced.
        output_path: Destination CSV path (parent directories created).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "family",
                "op",
                "format",
                "device",
                "n",
                "k",
                "median_time_s",
                "storage_bytes",
                "gpu_peak_bytes",
                "cpu_rss_delta_bytes",
                "skip_reason",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.family,
                    r.op,
                    r.format_label,
                    r.device_label,
                    r.n,
                    r.k,
                    r.median_time_s,
                    r.storage_bytes,
                    r.gpu_peak_bytes,
                    r.cpu_rss_delta_bytes,
                    r.skip_reason,
                ]
            )


def main() -> None:
    """Parse CLI args, run the full grid, and write results to CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--dims", type=int, default=2, choices=(2, 3))
    parser.add_argument("--dense-cubic-max-n", type=int, default=DEFAULT_DENSE_CUBIC_MAX_N)
    parser.add_argument("--max-memory-gb", type=float, default=DEFAULT_MAX_MEMORY_GB)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "results.csv")
    args = parser.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    max_memory_bytes = args.max_memory_gb * (1024**3)

    results: list[CellResult] = []
    started = perf_counter()
    for cell in build_grid(args.sizes, args.k_values, args.dims, dtype):
        cell_max_n = DENSE_CUBIC_MAX_N_OVERRIDES.get(cell.op, args.dense_cubic_max_n)
        if cell.op in DENSE_CUBIC_OPS and cell.n > cell_max_n:
            results.append(
                CellResult(
                    cell.family,
                    cell.op,
                    cell.format_label,
                    cell.device_label,
                    cell.n,
                    cell.k,
                    None,
                    None,
                    None,
                    None,
                    "known-cubic-blowup",
                )
            )
            continue
        if estimate_memory_bytes(cell.n, cell.family, dtype) > max_memory_bytes:
            results.append(
                CellResult(
                    cell.family,
                    cell.op,
                    cell.format_label,
                    cell.device_label,
                    cell.n,
                    cell.k,
                    None,
                    None,
                    None,
                    None,
                    "memory",
                )
            )
            continue

        device_for_memory = (
            torch.device("cuda") if cell.device_label == "cuda" else torch.device("cpu")
        )
        results.append(
            run_cell(cell, timeout_s=args.timeout_s, device_for_memory=device_for_memory)
        )
        print(
            f"{results[-1].family:<16} {results[-1].op:<16} {results[-1].format_label:<8} "
            f"{results[-1].device_label:<5} n={results[-1].n:<7} "
            f"{'skip: ' + results[-1].skip_reason if results[-1].skip_reason else f'{results[-1].median_time_s:.3e}s'}"
        )

    write_results(results, args.output)
    elapsed = perf_counter() - started
    print(f"\n{len(results)} cells in {elapsed:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
