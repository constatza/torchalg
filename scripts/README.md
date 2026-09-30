# scripts/

Standalone utilities that sit outside the `torchalg` package proper -
one-off visualizations, benchmarking harnesses. Each script is run
directly with `uv run python scripts/<script>.py`, not imported by
`src/torchalg`.

## `visualize_ic0_sparsity_dual_precision.py`

Plots IC(0)'s sparsity filter (dense, no elimination) on the
`rectangular-high-condition` real template matrix at float32/float64,
saving figures to `~/Pictures/`.

## `bench_sparse_vs_dense/`

Benchmarks `torch.sparse` (CSR/COO) against torchalg's current all-dense
tensors, across CPU/GPU and problem size, for the tensor-op families every
algorithm in the package actually performs. Exists to answer: which
operations should become sparse, should they run on CPU or GPU, and does
torchalg need two code paths (small-N vs large-N) for any of them.

### Layout

- `matrices.py` - real FEM template loader (`/data/shared/mgroup-clusters/`,
  used as size-840/3015 anchor points, not part of the size sweep) plus a
  synthetic structured lattice-Laplacian generator (`scipy.sparse.kronsum`
  of a 1D stencil across the grid's axes) giving exact, cheap control over
  N at any size up to 50k+ while staying SPD and banded like a real FEM
  stiffness matrix.
- `memory.py` - memory-footprint measurement, kept separate from timing:
  exact tensor storage size (dense or sparse), `torch.cuda.max_memory_allocated`
  for GPU cells, and a best-effort `resource.getrusage` RSS delta for CPU
  cells. A dense float64 matrix at N=50,000 is ~20GB - this is often the
  actual reason a large matrix forces a sparse representation, not runtime.
- `operations.py` - one representative, resident-only (operands already
  placed - compute cost only) function per tensor-op family, reusing
  torchalg's own kernels (`dense_ic0`, `dense_ilu0`,
  `cholesky_factor_solve`) where they already exist rather than
  reimplementing them. Also the standalone `h2d`/`d2h`/`to_sparse`/`to_dense`
  transfer/conversion helpers, timed independently of any compute cell
  since transfer/conversion cost is a function of size/format alone.
- `run_benchmark.py` - CLI: builds the (operation x format x device x N[, K])
  grid, drops cells eliminated by a static cap or a runtime guard (see
  below), times every surviving cell with `torch.utils.benchmark.Timer`,
  measures its memory footprint, writes one CSV row per cell.
- `report.py` - CLI: reads the CSV, prints a decision table (fastest leg at
  the smallest/largest measured N, and any crossover point), and saves a
  log-log time-vs-N plot per family plus a time-vs-K chart for the
  Galerkin formed-vs-matrix-free comparison.

### Operation families

| # | Family | Representative | Algorithms it stands in for | Sparse/native alternative |
|---|--------|-----------------|------------------------------|--------------|
| 1 | matvec | `A @ v` | CG/PCG/FCG inner loop | sparse CSR `mv` |
| 2 | elementwise scale | `D^{-1} r` | Jacobi apply | sparse-derived diagonal, same elementwise op |
| 3 | Galerkin triple product | `P.T @ A @ P` | AMG/POD coarsening | sparse-`A`@dense-`P` (SpMM), sparse-`A`@sparse-`P` (SpGEMM), and a matrix-free `P.T @ (A @ (P @ v))` alternative that never forms the product at all |
| 4 | triangular apply | `cholesky_factor_solve` | IC0/ILU/ICholesky apply | `torch.sparse.spsolve` (CUDA-only as of this torch build - raises `NotImplementedError` on CPU, logged as a skip) |
| 5 | factorization | `dense_ic0`/`dense_ilu0` | IC0/ILU setup | `scipy.sparse.linalg.spilu` (incomplete LU, CPU-only - no native `torch.sparse` factorization exists) |
| 6 | eigendecomposition | `torch.linalg.eigvalsh` | AMG's MGE, condition-number diagnostics | `torch.lobpcg` (native, sparse-capable, **CPU+GPU**) |
| 7 | SVD | `torch.linalg.svd` (thin snapshot matrix) | POD basis | `torch.svd_lowrank` (native randomized low-rank SVD, **CPU+GPU**) |

Families 6 and 7 have real native-torch sparse-capable alternatives that
also run on GPU - not just a CPU-only scipy fallback (scipy's
`eigsh`/`svds` are kept as a secondary reference point for both). Family 5
is the one genuine gap: torch has no native sparse Cholesky/LU.

### Cartesian-product reductions

Every dropped cell is still a row in the CSV (`skip_reason` set), never
silently absent:

1. **GPU columns** only run when `torch.cuda.is_available()` - a hardware
   limitation of the machine running the benchmark, not a theoretical
   elimination. The GPU code path is written and unmodified for when it
   runs on CUDA hardware.
2. **Static per-op N cap** (`--dense-cubic-max-n`, default 8,000): the
   `formed_dense` (Galerkin), `ic0` (factorization), and `eigvalsh` (full
   eigendecomposition) ops are O(N^3) at full matrix size - a dense
   float64 matrix at N=50,000 is already ~20GB and a cubic op there is
   plausibly hours on CPU. This is known ahead of time from complexity
   theory, so these three ops are never attempted above the cap rather
   than discovered via a live timeout. Every other op - including
   `matrix_free_dense`/`matrix_free_sparse`, which are O(N^2)/O(N) despite
   also touching dense operands - runs the full sweep.
3. **Runtime guards** (`--max-memory-gb`, `--timeout-s`) are a backstop for
   the cells that *are* attempted, not the primary defense against the
   known cubic blowups above.
4. A sparse/native leg unavailable on this build/device (`spsolve` on CPU)
   is caught and logged as `unsupported: ...`, not treated as a crash.

### Running it

```sh
# Correctness check first - confirms every sparse/native/scipy leg agrees
# numerically with its dense reference before trusting any timing.
uv run pytest -m benchmark tests/benchmarks/sparse_dense/

# Full sweep (CPU-only on a machine with no CUDA device; the GPU legs are
# skipped automatically, not run at reduced fidelity).
uv run python scripts/bench_sparse_vs_dense/run_benchmark.py

# Decision table + plots under scripts/bench_sparse_vs_dense/results/plots/
uv run python scripts/bench_sparse_vs_dense/report.py
```

Both CLIs take `--sizes`/`--k-values`/`--dense-cubic-max-n`/
`--max-memory-gb`/`--timeout-s`/`--dtype`/`--output` - see `--help`. A
smaller `--sizes`/`--dense-cubic-max-n` is worth using for a quick local
run; the defaults sweep to N=50,000.
