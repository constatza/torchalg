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

Level scheduling is structural preprocessing. CPU scheduling uses a linear
native-integer recurrence; accelerator scheduling uses reverse adjacency and
wavefront traversal so each dependency edge is processed once without
per-row device scalar reads.

Sparse IC(0), ICholesky, and ILU preconditioners materialize their lower/upper
solve factors and both direction-specific schedules during `setup()`. Their
repeated `apply()` path performs only the two numerical triangular solves and
accepts vector or matrix right-hand sides directly. The cached transpose or
split factors add `O(nnz)` storage in exchange for removing structural work
from every solver iteration.

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
- `GaussSeidelSmoother.prepare(A)` returns a `PreparedSymmetricGaussSeidel`
  holding both direction-specific level schedules, the per-entry row index and
  the strict-triangle entries. `smooth` and `smooth_from_zero` (which skips the
  first strict-upper matvec) reuse them; `smooth(..., rows=)` is PyAMG's indexed
  sweep. `triangular_solve` receives the precomputed row index; its per-level
  masking is unchanged.
- `_jacobi_omega.jacobi_omega` / `jacobi_spectral_radius` are pure and
  uncached. Setup code that needs rho (prolongation smoothing in
  coarsening and adaptive setup, the Jacobi smoother's `prepare`) calls them
  once per level at setup. There is no module-level or `id()`-keyed cache.
- Setup algorithms that relax the same level matrices many times (compatible
  relaxation, test-vector relaxation, adaptive trial cycles, bootstrap
  improvement passes) own a `GaussSeidelStage` and a `TransferOperatorCache`
  for the duration of one setup run. Each caches by tensor identity, holds
  its matrices alive, and is created fresh per `adaptive_sa_hierarchy` call or
  per `BootstrapAMGPreconditioner.setup()`, so each distinct level matrix gets
  its schedules built once and each prolongation is transposed once.
  `_presets.gs_setup_cycle(stage)` builds the GS V(1,1) setup cycle over a
  stage's smoother.
- `kernels.diagonal.sparse_diagonal` scatters masked values instead of
  boolean-indexing them, and `repeat_interleave` row-index expansion passes
  `output_size`, so neither reads a count back from the device.
