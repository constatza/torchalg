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
- Every public sparse preconditioner, smoother and coarsening (and the
  `vcycle_amg`/`wcycle_amg` factories) documents its cost in a `Cost:` docstring
  section: work `W`, bytes moved `Q`, depth `D`, kernel launches `L` and, where
  nonzero, host synchronisations `S`, as formulas in `nnz`, the dependency levels
  of the triangular solves and the hierarchy sizes, stated separately for `Setup`
  (the matrix-dependent one-off work) and `Apply` (one application, one sweep for a
  smoother). `tests/test_cost_docstrings.py` requires `W`, `D` and `L` in each
  phase. A setup that prepares a triangular solve pays four host reads per
  direction (one zero-pivot test and three count reads of the masked triangle
  extraction), so IC(0) setup reads the host 9 times and ILU(0) 8 times.

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
  call has no autograd. That fallback is built on the first such call and kept as a
  submodule (its schedule is an `O(n + nnz)` host recurrence), except when the triangle
  itself requires grad: the fallback captures the triangle's graph at construction, so it
  is then rebuilt per call. The triangle's `crow_indices` take the operand's
  `col_indices` dtype, so an int32 operand stays int32 end to end (MKL then does not
  narrow int64 indices on every call).
- `PreparedTriangularSolve`: level-scheduled substitution, the default for every
  unregistered device type. It handles zero pivots (row frozen to the target)
  and `active` masks. It is launch-bound (about 13 eager ops per dependency
  level). ROCm reports as `"cuda"` and is unsupported: its library solve faults
  on the second call (gfx1030, rocm7.2).

Both also expose `with_identity_rows(frozen)`: the same substitution with the `frozen`
rows replaced by identity rows (each solves to its target). The library triangle keeps
its pattern, with explicit zeros on the frozen rows' off-diagonals and a unit diagonal,
and shares its index arrays; the level-scheduled solve only widens its zero-pivot
freeze, which is the same row. Neither sorts, rebuilds or analyses anything, so a
caller whose mask changes every call pays one `O(nnz)` value pass.

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
read except one breakdown check after the last level (per-level pivots are collected and
tested together; the error still names the first failing row in level order, because
pivots after a breakdown turn `-inf` or `nan` but are concatenated after it); the
factor's input values are gathered by the integer `pattern.lower_positions`, not the
boolean mask, so the gather needs no `nonzero` host sync on an accelerator; each step
writes its positions in place unless grad mode is on and the matrix values require grad,
when the out-of-place `scatter` keeps autograd's saved versions valid (shared
`write_positions`/`tracks_autograd` in `preconditioners/_triangular.py`); `pattern=None` analyzes internally for
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
`sparse_ilu0(A, pattern)` keeps the exact numeric loop (padded gather; in-place writes into a private copy of the
values, out-of-place `scatter` when autograd tracks them), so factor values are bit-identical; its numeric phase
makes no `lookup` call and no host read. `build_sparse_lu_solve_cache(factor, pattern)`
builds both triangles by gathers, with no argsort or boolean-mask indexing. The structural
arrays follow the IC(0) width rule (int32 when `max(n, nnz)` fits, via the shared
`structural_index` of `preconditioners/_pattern_index.py`, which also holds the `lookup` search both analyses use), and the factor's `crow`/`col` use that width too; the per-step
gather positions stay int64. `ILUPreconditioner` holds its pattern and applies the same
reuse and assign-after-success rules as `IC0Preconditioner`. The pattern's level schedule is
consumed only to order the steps: the library triangular solve builds none, so nothing is
passed to `PreparedTriangularSolve` on CPU/CUDA.

## Prepared smoothers

Each smoother declares a `placement` (`Placement`): `JacobiSmoother` is `SPMV_BOUND`,
`GaussSeidelSmoother` and `GaussSeidelMemo` are `SEQUENTIAL_SWEEP`. Cycles expose the
smoother's, and the AMG/BootCMatch preconditioners take their device placement from it, so a
Jacobi-smoothed sparse AMG is placed like an SpMV and a Gauss-Seidel-smoothed one stays on the
CPU.

A sparse smoother object holds only configuration; `prepare(A)` does every
matrix-dependent one-off computation and returns the object that sweeps, so
`apply()` never estimates, schedules or factors anything.

- `JacobiSmoother.prepare(A)` extracts the diagonal once and resolves the
  damping eagerly (`1 / rho(D^-1 A)` from a fixed-seed Arnoldi estimate, or the
  explicit override); the prepared object holds `omega / diag` with a zero
  step for near-zero diagonals. `smooth` is `x + diag_inv * (rhs - A @ x)`
  with no clone; `smooth_from_zero` skips the first `A @ 0`. Both accept a
  vector `(n,)` or a block `(n, k)`.
- `GaussSeidelSmoother.prepare(A)` returns a prepared symmetric Gauss-Seidel
  (`_gauss_seidel.prepare_csr_gauss_seidel` -> `PreparedSymmetricGaussSeidel`) at every
  size; there is no dense-backed format. Preparation compacts `A` to int32 indices once,
  splits it into strict-upper and strict-lower CSR triangles and prepares `D + L` and
  `D + U` with `kernels.triangular.prepare_triangular_solve` from the compacted matrix,
  so the substitution is the vendor solve registered for the device
  (`TRIANGULAR_SOLVER_BY_DEVICE`: MKL on CPU, cuSPARSE on CUDA) with launches independent
  of the dependency depth, and MKL never narrows int64 triangle indices per call (161 to
  137 us per solve at lap2d n = 10 000). The caller chooses how a sweep gets its
  strict-triangle products (`SweepProducts`). The solve-time smoother takes
  `FROM_SOLVES`: when every pivot is nonzero the solves satisfy their systems, so the
  strict products come from them: the backward target is `U x_old + D x_f` and the next
  forward sweep's `U x_b` is `t_b - D x_b`. `smooth` therefore applies `U` once per call
  and `smooth_from_zero` (backward target `D x_f`) no strict product at all; results
  equal the explicit sweep to rounding (1e-16). Setup stages (`GaussSeidelMemo`:
  compatible relaxation, adaptive and bootstrap test-vector relaxation and setup cycles)
  take `EXPLICIT` and apply `L` and `U` every sweep, because coarsening and fitting
  amplify that rounding to 1e-10 in the hierarchy; explicit products keep every setup
  bit-identical to the textbook sweep. A matrix with a zero pivot gets the
  level-scheduled `PreparedTriangularSolve`, which freezes that row to the sweep's
  target; the freeze breaks both identities, so its sweeps apply `L` and `U` explicitly
  whatever the caller chose. `smooth(..., rows=)` (PyAMG's indexed sweep, used
  only by adaptive SA's general setup stage; compatible relaxation sweeps unmasked) also
  sweeps triangles with every inactive or zero-pivot row replaced by an identity row, and
  solves for the target `current` on the inactive rows, as the dense
  `_relaxation._prepare_sweep` does. The mask comes from the iterate (the nonzero rows of
  a prolongated candidate), so it changes every call and a per-mask cache would never
  hit; each call derives both solves from the prepared ones with `with_identity_rows`
  instead of building new triangles (a masked call on box3d n = 27 000 at 2 steps:
  6.6 ms against 65 ms rebuilding the triangles and 35 ms level-scheduled). Both `smooth` and `smooth_from_zero` accept a
  vector `(n,)` or a block `(n, k)`. Adaptive SA, bootstrap AMG and BootCMatch build
  identical hierarchies with identical PCG iteration counts on the `benchmarks/cases.py`
  cases.
- Every prepared smoother also implements `residual(rhs, x) = rhs - A x`, and the
  V- and W-cycles form level residuals through it, so the operator each level
  multiplies by is the one `prepare` stored. Sparse prepared smoothers and
  `SparseTransferOperator` store their CSR operators with int32 indices
  (`utils.csr_index.compact_csr_indices`): PyTorch's CPU sparse BLAS narrows
  int64 indices on every product, so narrowing once at setup makes SpMV
  1.5-2.5x faster and the AMG V-cycles 1.2-1.8x faster, bit-identical.
- `_jacobi_omega.jacobi_spectral_radius` is a pure function of the matrix,
  memoized per matrix object (`utils.weak_cache.tensor_memo`: entries die with
  the matrix, in-place edits invalidate, inference tensors are never cached).
  Prolongation smoothing in coarsening and the Jacobi smoother's `prepare`
  receive only the level matrix, so they share one estimate per level through
  its identity. Nothing estimates in `apply`. Bootstrap and BootCMatch estimate
  each distinct level matrix once (a later BootCMatch hierarchy on the same fine
  matrix object reuses the finest estimate). Adaptive setup calls
  `approximate_spectral_radius` itself, outside the memo, with the start vector
  drawn from its shared random stream: its estimates of one matrix start from
  different vectors and differ at the 1e-5 level, so merging them changes the
  hierarchy (iteration counts hold, a one-application result moves by 3e-7 to
  1e-5) and they stay separate.
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
  stage's smoother. Adaptive SA's general stage prepares its trial hierarchy once per
  candidate and swaps in only the level each nested sub-hierarchy solve refits, so the
  densified coarsest `pinv` is computed once per candidate instead of once per trial
  solve.
- `BootstrapAMGPreconditioner.setup()` also builds one `weak_memo` of
  `kernels.depth_neighborhood.sparse_depth_neighborhood` per setup run and hands
  it to every `BAMGCoarsening` as `neighborhood_of`. The finest matrix's
  LS-ring neighborhood depends on `A` alone, so it is built once across the
  initial and every improvement hierarchy; coarse matrices are new per
  hierarchy and are dropped with their matrix. `sparse_depth_neighborhood`
  itself runs CSR products (not COO) and sorts once at the end, because the
  CSR product leaves columns unsorted and keeps the diagonal.
- `BAMGCoarsening` resolves every empty-interpolatory-set row at once
  (`_strongest_candidates_sparse`, a searchsorted over the distance matrix's row-major
  keys), so the fallback costs one `nonzero` read and one extra row fit, never a host
  read per row.
- `form_sparse_sparse` stays on COO: a CSR `P.T @ (A @ P)` is `0.3 ms` faster
  per call at `n = 10 000` but returns unsorted columns, and restoring sorted
  rows costs what the COO path already spends.
- The dense coarsest-level inverse (`multigrid.coarse_solve.pseudo_inverse_coarse_solver`)
  uses a Cholesky inverse `X` for a symmetric SPD matrix of dimension
  `>= CHOLESKY_MIN_DIMENSION` when `||A||_F ||X||_F` (an upper bound on `kappa_2`)
  is below `0.1 / (c eps)`, which proves `pinv`'s default `rtol = c eps` would
  truncate no singular value, and `pinv` otherwise, so singular, near-singular,
  indefinite, nonsymmetric and small coarse operators are unchanged. The Cholesky
  inverse measured at least as accurate as `pinv` at every `kappa` (forward error
  2.6e-11 against 3.1e-11 at `kappa = 1e6`, c = 256), so the two differ by rounding.
- `kernels.diagonal.sparse_diagonal` scatters masked values instead of
  boolean-indexing them, and `repeat_interleave` row-index expansion passes
  `output_size`, so neither reads a count back from the device.
