# Sparse implementation

`torchalg.sparse` contains the CSR-specific implementation tree. Dense
`torchalg` modules must not import from it; callers select a dense or sparse
algorithm based on the matrix representation they already own.

- `kernels/` provides reusable sparse primitives such as diagonal extraction,
  Galerkin formation, graph operations, and triangular solves.
- `preconditioners/` contains complete sparse algorithms, including AMG,
  IC(0), ICholesky, ILU, and Jacobi implementations.
- `device_policy.py` supplies internal, advisory CPU/CUDA recommendations
  derived from a particular benchmark environment. It never moves tensors and
  is not a cross-hardware performance guarantee.

Re-run the opt-in experiments in `benchmarks/sparse_dense/` when a device,
PyTorch version, or matrix family changes materially.

## Triangular solve preparation

`kernels.triangular.prepare_triangular_solve(matrix, direction)` returns the exact
substitution registered for the matrix's device type in
`TRIANGULAR_SOLVER_BY_DEVICE`, the single place that maps devices to
implementations; callers never branch on the device, and `solve` never consults
it. Both implementations expose `solve(target)` with results equal to rounding:

- `LibraryTriangularSolve`: one `torch.triangular_solve` call on the CSR triangle
  (MKL). Registered for `"cpu"`, used when no pivot is within `diagonal_tol` of
  zero (checked once at setup). At n = 10000 (5-point Laplacian, 199 dependency
  levels) an IC(0) apply takes 0.28 ms against 14.7 ms level-scheduled. A target
  that requires grad falls back to the level-scheduled solve, since the library
  call has no autograd.
- `PreparedTriangularSolve`: level-scheduled substitution, the default for every
  unregistered device type. It handles zero pivots (row frozen to the target)
  and `active` masks. It is launch-bound (about 13 eager ops per dependency
  level). GPUs are unregistered until measured: on gfx1030 with the rocm7.2
  wheels the library solve faults on its second call.

`SparseTriangularSolveCache` keeps the factor pair as its canonical state and
re-resolves both halves through the registry when `.to()` moves it to another
device type, so a preconditioner set up on the CPU and moved to a GPU uses the
GPU's implementation.

Level scheduling is structural preprocessing. CPU scheduling uses a linear
native-integer recurrence; accelerator scheduling uses reverse adjacency and
wavefront traversal so each dependency edge is processed once without
per-row device scalar reads.

Sparse IC(0), ICholesky, and ILU preconditioners materialize their lower/upper
factors and prepare both substitutions during `setup()`. Their repeated
`apply()` path performs only the two numerical triangular solves and accepts
vector or matrix right-hand sides directly. The cached transpose or split
factors add `O(nnz)` storage in exchange for removing structural work from
every solver iteration.

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
  one of two formats with identical results (`_gauss_seidel_format.prepare_gauss_seidel`):
  `PreparedSymmetricGaussSeidel` (level-scheduled CSR: both direction-specific
  schedules, the per-entry row index, the strict-triangle entries) or
  `DenseBackedSymmetricGaussSeidel` (`_dense_gauss_seidel.py`: the matrix densified
  once into `D + L` and `D + U` solved with `solve_triangular`). A CSR sweep costs
  about 0.06 ms per dependency level and is flat in `n`; a dense sweep costs about
  `8e-7 ms * n^2` whatever the density. Rule (CPU, 6 threads): dense for
  `n <= 1500`, CSR for `n >= 4000`, and in between dense when
  `8e-7 * n^2 < 0.06 * levels` using the forward schedule's level count (the crossover
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
  improvement passes) own a `GaussSeidelStage` and a `TransferOperatorCache`
  for the duration of one setup run. Each caches by tensor identity, stores
  its matrices beside the prepared state so ids cannot be reused, and is created fresh per `adaptive_sa_hierarchy` call or
  per `BootstrapAMGPreconditioner.setup()`, so each distinct level matrix gets
  its Gauss-Seidel prepared once and each prolongation is transposed once.
  `_presets.gs_setup_cycle(stage)` builds the GS V(1,1) setup cycle over a
  stage's smoother.
- `kernels.diagonal.sparse_diagonal` scatters masked values instead of
  boolean-indexing them, and `repeat_interleave` row-index expansion passes
  `output_size`, so neither reads a count back from the device.
