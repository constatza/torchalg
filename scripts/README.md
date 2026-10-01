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
| 4 | triangular apply | `cholesky_factor_solve` | IC0/ILU/ICholesky apply | none measurable - see below |
| 5 | factorization | `dense_ic0`/`dense_ilu0` | IC0/ILU setup | `scipy.sparse.linalg.spilu` (incomplete LU, CPU-only - no native `torch.sparse` factorization exists) |
| 6 | eigendecomposition | `torch.linalg.eigvalsh` | AMG's MGE, condition-number diagnostics | `torch.lobpcg` (native, sparse-capable, **CPU+GPU**) |
| 7 | SVD | `torch.linalg.svd` (thin snapshot matrix) | POD basis | `torch.svd_lowrank` (native randomized low-rank SVD, **CPU+GPU**) |

Families 6 and 7 have real native-torch sparse-capable alternatives that
also run on GPU - not just a CPU-only scipy fallback (scipy's
`eigsh`/`svds` are kept as a secondary reference point for both). Family 5
is the one genuine gap: torch has no native sparse Cholesky/LU.

**Legs that no longer appear in the grid at all** (removed outright, not
capped-and-skipped - see "Cartesian-product reductions" point 2 below):
dense `formed_dense` (family 3, O(N^3) Galerkin formation), dense `ic0`
(family 5, a Python-level loop whose measured cost grows far worse than
N^3), dense `eigvalsh` (family 6, O(N^3) full eigendecomposition), and
native sparse `spsolve` (family 4 - unsupported on CPU, and on CUDA without
a cuDSS-enabled build, at every N; `torchalg.sparse.kernels.triangular`'s
own level-scheduled solve is the real replacement this capability gap
motivated). Each removal's established finding (the blowup point, or the
capability gap) is already recorded in `docs/plan.md`'s "At-scale evidence"
and in historical `results.csv` runs - a future sweep gains nothing by
re-confirming it, and family 4 in particular has no sparse/native leg left
to compare against the dense one (hence "none measurable" above).

### Cartesian-product reductions

Every dropped cell is still a row in the CSV (`skip_reason` set), never
silently absent:

1. **GPU columns** only run when `torch.cuda.is_available()` - a hardware
   limitation of the machine running the benchmark, not a theoretical
   elimination. The GPU code path is written and unmodified for when it
   runs on CUDA hardware.
2. **Known-bad legs are removed from the grid, not capped.** The four legs
   listed above (dense `formed_dense`/`ic0`/`eigvalsh`, sparse `spsolve`)
   have an already-established, previously-measured outcome (an O(N^3)+
   blowup, or a complete capability gap at every N) - keeping them in the
   sweep would only re-derive an already-known conclusion (or, for
   `spsolve`, produce nothing but an `unsupported: ...` skip row at every
   single cell). `operations.py`'s underlying functions
   (`form_galerkin_dense`/`factorize_ic0`/`eigh_dense`/
   `triangular_apply_sparse`) are untouched and still correctness-checked
   against their sparse/scipy counterparts by
   `tests/benchmarks/sparse_dense/test_operations_equivalence.py` - only
   the wall-clock sweep stopped exercising them.
3. **Runtime guards** (`--max-memory-gb`, `--timeout-s`) are a backstop for
   the cells that *are* attempted (e.g. `matrix_free_dense` at very large
   N), not the primary defense against point 2's known-bad legs - those are
   gone from the grid before either guard would ever see them.
4. A sparse/native leg that turns out unavailable for some other,
   less-settled reason is still caught and logged as `unsupported: ...`,
   not treated as a crash - point 2 only removes legs whose
   unsupportedness/blowup is already fully established.

### `--compile`

Passing `--compile` adds a second, `torch.compile`-wrapped sibling cell for
every cell in a parallel-kernel-shaped family (`mv`, `galerkin_form`,
`triangular_apply` - see `run_benchmark.py`'s `COMPILABLE_FAMILIES`),
recorded as a `mode` column in the output CSV (`"eager"` vs. `"compiled"`).
Per `docs/plan.md`'s "Next steps" (re-benchmark under `torch.compile()`):
scoped to these families specifically because compilation can shift the
calculus for parallel-kernel ops (kernel fusion changes per-call overhead),
while the sequential-graph-construction-style ops this script measures
elsewhere would gain nothing - their control flow isn't the kind
`torch.compile` traces through. Off by default; omitting the flag reproduces
the exact pre-`--compile` grid and CSV shape (plus the now-always-present
`mode` column, constant `"eager"`).

### Running it

```sh
# Correctness check first - confirms every sparse/native/scipy leg agrees
# numerically with its dense reference before trusting any timing.
uv run pytest -m benchmark tests/benchmarks/sparse_dense/

# Full sweep (CPU-only on a machine with no CUDA device; the GPU legs are
# skipped automatically, not run at reduced fidelity). Add --compile to
# also measure the torch.compile-wrapped legs (see above).
uv run python scripts/bench_sparse_vs_dense/run_benchmark.py

# Decision table + plots under scripts/bench_sparse_vs_dense/results/plots/
uv run python scripts/bench_sparse_vs_dense/report.py
```

`run_benchmark.py` takes `--sizes`/`--k-values`/`--dims`/`--max-memory-gb`/
`--timeout-s`/`--dtype`/`--output`/`--compile` - see `--help`. A smaller
`--sizes` is worth using for a quick local run; the defaults sweep to
N=50,000. `report.py` just reads the resulting CSV and writes plots - it
takes only `--input`/`--plots-dir`.
