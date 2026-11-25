# Sparse implementation

`torchalg.sparse` contains the CSR-specific implementation tree. Dense
`torchalg` modules must not import from it; callers select a dense or sparse
algorithm based on the matrix representation they already own.

- `kernels/` provides reusable sparse primitives such as diagonal extraction,
  Galerkin formation, graph operations, and triangular solves.
- `preconditioners/` contains complete sparse algorithms, including AMG,
  IC(0), ICholesky, ILU, and Jacobi implementations. POD has no sparse
  sibling: `torchalg.preconditioners.implementations.pod.PODCoarseningStrategy`
  takes a CSR matrix directly, since its basis and coarse matrix are dense.
  Every sparse AMG preset (`vcycle_amg`, `wcycle_amg`, `AdaptiveSAPreconditioner`,
  `BootstrapAMGPreconditioner`, `BootCMatchPreconditioner`) is constructed from
  configuration alone and bound to a CSR matrix by `.setup(A)`.
- Device-specific implementations are chosen by per-operation registries keyed
  by device type (`kernels.triangular.TRIANGULAR_SOLVER_BY_DEVICE`,
  `LEVEL_SCHEDULER_BY_DEVICE`), never by device checks at call sites.

Re-run the opt-in experiments in `benchmarks/sparse_dense/` when a device,
PyTorch version, or matrix family changes materially.

## Triangular solve preparation

`kernels.triangular.prepare_triangular_solve(matrix, direction)` returns the exact
substitution registered for the matrix's device type in
`TRIANGULAR_SOLVER_BY_DEVICE`, the single place that maps devices to
implementations; callers never branch on the device, and `solve` never consults
it. Both implementations expose `solve(target)` with results equal to rounding:

- `LibraryTriangularSolve`: one `torch.triangular_solve` call on the CSR triangle
  (MKL on CPU, cuSPARSE `SpSV` on CUDA). Registered for `"cpu"` and `"cuda"`,
  used when no pivot is within `diagonal_tol` of zero (checked once at setup). At n = 10000 (5-point Laplacian, 199 dependency
  levels) an IC(0) apply takes 0.28 ms against 14.7 ms level-scheduled. A target
  that requires grad falls back to the level-scheduled solve, since the library
  call has no autograd. The triangle's `crow_indices` take the operand's
  `col_indices` dtype, so an int32 operand stays int32 end to end (MKL then does not
  narrow int64 indices on every call).
- `PreparedTriangularSolve`: level-scheduled substitution, the default for every
  unregistered device type. It handles zero pivots (row frozen to the target)
  and `active` masks. It is launch-bound (about 13 eager ops per dependency
  level). ROCm reports as `"cuda"` and is unsupported: its library solve faults
  on the second call (gfx1030, rocm7.2).

`PreparedTriangularPair` keeps the factor pair as its canonical state and
re-resolves both halves through the registry when `.to()` moves it to another
device type, so a preconditioner set up on the CPU and moved to a GPU uses the
GPU's implementation.

Level scheduling is structural preprocessing, dispatched the same way through
`LEVEL_SCHEDULER_BY_DEVICE`: CPU uses a linear native-integer recurrence; every
other device the default wavefront traversal with reverse adjacency, so each
dependency edge is processed once without per-row device scalar reads.

Sparse IC(0), ICholesky, and ILU preconditioners materialize their lower/upper
factors and prepare both substitutions during `setup()`. Their repeated
`apply()` path performs only the two numerical triangular solves and accepts
vector or matrix right-hand sides directly. The cached transpose or split
factors add `O(nnz)` storage in exchange for removing structural work from
every solver iteration.

### Pattern analysis and reuse (IC(0), ILU(0))

Everything IC(0) derives from the sparsity pattern alone - the forward level schedule,
the storage position of every `L[i,k]`/`L[j,k]` product, the diagonal positions and the
transpose permutation of `L` - is computed once by `analyze_ic0(A)` into a frozen
`IC0Pattern` (`preconditioners/_ic0_pattern.py`). `sparse_ic0(A, pattern)` then only
gathers by those positions, so its numeric loop makes no `searchsorted` and no host
read except the per-level breakdown check; `pattern=None` analyzes internally for
one-off use. The pattern is the stored lower-triangle entries whose value is not zero
(`|value| > 0`, so zero and NaN drop out; any other value is pattern and there is no drop
tolerance), and `pattern.matches(A)` compares
`crow`, `col` and that non-zero mask in one O(nnz) pass, so an entry turning zero or
non-zero is never silently reused. The numeric loop masks the gathered `L[j,k]`/`U[k,j]`
with `torch.where(found, values[pos], 0)` for positions outside the pattern (they index
storage 0), so an `inf`/`nan` stored there never leaks in as `inf * 0`.

The structural index arrays that only locate storage (`row`, `lower_crow`, `lower_col`,
`upper_crow`, `upper_col`, `transpose_perm`) are held as int32 when `max(n, nnz)` fits
(`INT32_INDEX_LIMIT`), int64 otherwise, halving them: 1.32 -> 0.69 MB (lap2d n = 10 000),
5.30 -> 2.75 MB (n = 40 000), 1.42 -> 0.74 MB (box3d n = 2 744), against 0.4 MB of matrix
values at n = 10 000. The factor `L` and `L.T` share them, so the solves run on int32 CSR.
The per-step gather positions stay int64 (int32 advanced-index gathers measured 10-40%
slower) and dominate the rest of the pattern's memory, so they bound further savings. The
reuse key `crow`/`col` are copies owned by the pattern (4 bytes per stored entry and per
row with int32 indices, 8 with int64, on top of the arrays above), so a later in-place edit
of the matrix's own indices cannot alias the key; `matches` also rejects a matrix whose
index dtypes differ from the analyzed one.

Reuse is instance-held and content-keyed, never memoized by `id()` or in a module-level
cache: a new matrix object with the same pattern would miss such a cache, and an in-place
mutation would hit it stale. Because the key holds copies of `crow`/`col`, an in-place edit
of the matrix's indices (or values mask) after `setup` makes `matches` false and the next
`setup` re-analyzes. `IC0Preconditioner` keeps the pattern of its last
`setup`; the next `setup` reuses it when `matches` is true and re-analyzes otherwise,
then factorizes, builds the `PreparedTriangularPair` (its transpose is a gather by
`transpose_perm`, not a COO round trip) and assigns all state only after every step
succeeded.

ILU(0) follows the same lifecycle with its own pattern: `analyze_ilu0(A)` returns a
frozen `ILU0Pattern` (`preconditioners/_ilu0_pattern.py`) holding the forward schedule's
elimination steps (storage positions of every `L[i,k]`/`U[k,j]` product, found masks, L-side
flags, positions of `U[j,j]`) and the structure-only split of the combined factor into the
unit-lower and upper triangles (`lower_source`, `upper_source` gather positions). ILU(0)'s
pattern is `A`'s own stored structure, with no zero mask, so `pattern.matches(A)` compares
`crow`, `col` and the index dtypes only (a stored zero stays in the pattern).
`sparse_ilu0(A, pattern)` keeps the exact numeric loop (padded gather, out-of-place
`scatter`), so factor values are bit-identical and autograd is unchanged; its numeric phase
makes no `lookup` call and no host read. `build_sparse_lu_solve_cache(factor, pattern)`
builds both triangles by gathers, with no argsort or boolean-mask indexing. The structural
arrays follow the IC(0) width rule (int32 when `max(n, nnz)` fits, via the shared
`structural_index` of `preconditioners/_pattern_index.py`, which also holds the `lookup` search both analyses use), and the factor's `crow`/`col` use that width too; the per-step
gather positions stay int64. `ILUPreconditioner` holds its pattern and applies the same
reuse and assign-after-success rules as `IC0Preconditioner`. The pattern's level schedule is
consumed only to order the steps: the library triangular solve builds none, so nothing is
passed to `PreparedTriangularSolve` on CPU/CUDA.

## Prepared smoothers

A sparse smoother object holds only configuration; `prepare(A)` does every
matrix-dependent one-off computation and returns the object that sweeps, so
`apply()` never estimates, schedules or factors anything.

- `JacobiSmoother.prepare(A)` extracts the diagonal once and resolves the
  damping eagerly (`1 / rho(D^-1 A)` from a fixed-seed Arnoldi estimate, or the
  explicit override); the prepared object holds `omega / diag` with a zero
  step for near-zero diagonals. `smooth` is `x + diag_inv * (rhs - A @ x)`
  with no clone; `smooth_from_zero` skips the first `A @ 0`. Both accept a
  vector `(n,)` or a block `(n, k)`.
- `GaussSeidelSmoother.prepare(A)` returns a prepared symmetric Gauss-Seidel in
  one of two formats with identical results. Each hop has one role:
  `_gauss_seidel_format.prepare_gauss_seidel` only picks the format, then
  `_gauss_seidel.prepare_csr_gauss_seidel` builds `PreparedSymmetricGaussSeidel`
  (level-scheduled CSR: both direction-specific schedules, the per-entry row index,
  the strict-triangle entries, each direction a `PreparedTriangularSolve`) or
  `_dense_gauss_seidel.prepare_dense_gauss_seidel` builds
  `PreparedDenseGaussSeidel` (the matrix densified once into `D + L` and
  `D + U` solved with `solve_triangular`); `PreparedGaussSeidel` is their union. A CSR sweep costs
  about 0.06 ms per dependency level and is flat in `n`; a dense sweep costs about
  `8e-7 ms * n^2` whatever the density. Rule (CPU, 6 threads): dense for
  `n <= 1500`, CSR for `n >= 4000`, and in between dense when
  `8e-7 * n^2 < 0.06 * levels` using the forward schedule's level count, which the
  CSR format then reuses instead of rebuilding it (the crossover
  is `n ~ 2500` at 5 nonzeros per row and `n ~ 2750` at 23, so density acts only through
  the level count). The constants are CPU estimates; CUDA is unmeasured and expected to
  favour dense longer. Everything else in the sparse tree stays CSR: below `n ~ 200` dense
  matvecs and Galerkin products save only microseconds per level. Both formats
  support `smooth`, `smooth_from_zero` (which skips the first strict-upper matvec) and
  `smooth(..., rows=)` (PyAMG's indexed sweep); a zero pivot freezes a row to the
  sweep's target in both.
- Every prepared smoother also implements `residual(rhs, x) = rhs - A x`, and the
  V- and W-cycles form level residuals through it, so the operator each level
  multiplies by is the one `prepare` stored. Sparse prepared smoothers and
  `SparseTransferOperator` store their CSR operators with int32 indices
  (`kernels.index_dtype.compact_csr_indices`): PyTorch's CPU sparse BLAS narrows
  int64 indices on every product, so narrowing once at setup makes SpMV
  1.5-2.5x faster and the AMG V-cycles 1.2-1.8x faster, bit-identical.
- `_jacobi_omega.jacobi_omega` / `jacobi_spectral_radius` are pure and
  uncached. Setup code that needs rho (prolongation smoothing in
  coarsening and adaptive setup, the Jacobi smoother's `prepare`) calls them
  once per level at setup. There is no module-level or `id()`-keyed cache.
- Setup algorithms that relax the same level matrices many times (compatible
  relaxation, test-vector relaxation, adaptive trial cycles, bootstrap
  improvement passes) own a `GaussSeidelMemo` and a `TransferOperatorMemo`
  for the duration of one setup run. Each is an identity memo (`*Memo`, not
  prepared state: the prepared objects it hands out are the frozen `Prepared*`
  results), stores its matrices beside them so ids cannot be reused, and is
  created fresh per `adaptive_sa_hierarchy` call or per
  `BootstrapAMGPreconditioner.setup()`, so each distinct level matrix gets its
  Gauss-Seidel prepared once and each prolongation is transposed once.
  `_presets.gs_setup_cycle(stage)` builds the GS V(1,1) setup cycle over a
  stage's smoother.
- `kernels.diagonal.sparse_diagonal` scatters masked values instead of
  boolean-indexing them, and `repeat_interleave` row-index expansion passes
  `output_size`, so neither reads a count back from the device.
