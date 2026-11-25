# Lifecycle convention and sparse IC(0)/ILU(0) pattern reuse: status

Six stages are implemented in the working tree (uncommitted). The convention itself
(config / analyze / prepare / setup, caching rule) lives in
`src/torchalg/preconditioners/README.md` and `src/torchalg/sparse/README.md`; this file
records what each stage delivered, where it deviated, what was measured, what broke, and
what is still undecided.

## Delivered

1. **Convention and sparse IC(0) pattern reuse.** `analyze_ic0(A) -> IC0Pattern` (frozen,
   slots, values-free beyond the non-zero mask) holds the level schedule, the gather
   positions of every elimination step, the diagonal positions and the transpose
   permutation of `L`. `sparse_ic0(matrix, pattern=None)` only gathers by those positions.
   `IC0Preconditioner` keeps its last pattern, reuses it when `pattern.matches(A)` (content
   check on `crow`, `col` and the lower non-zero mask, copies held by the pattern) and
   assigns state only after every step succeeded. `threshold` is gone from both the dense
   and the sparse IC(0) (`|value| > 0` is pattern; NaN and zero drop out, as at HEAD).
2. **Naming clashes removed.** `.build` aliases became `prepare_*` functions, stage-scoped
   identity memos became `*Memo`, `HierarchyNotPreparedError` subclasses
   `PreconditionerNotReadyError`.
3. **Structure-only handles beyond IC(0).** `analyze_ilu0(A) -> ILU0Pattern` (key: `crow`,
   `col`, index dtypes; a stored zero stays in the pattern), the structure-only split of the
   combined factor into the two triangles, `ILUPreconditioner` pattern reuse. Structural
   index arrays are int32 when `max(n, nnz)` fits `INT32_INDEX_LIMIT`, int64 otherwise; the
   per-step gather positions stay int64 (int32 advanced-index gathers measured 10-40%
   slower).
4. **State, not flags.** `is_ready` is an abstract read-only property derived from the
   state `setup()` builds; `_mark_ready()` and the `_is_ready` flag are removed.
5. **AMG binds configuration only.** `AMGPreconditioner(coarsening, cycle, n_levels, ...)`
   takes no matrix; its only prepared state is a `PreparedHierarchy` whose finest level
   carries the matrix. `AdaptiveSAPreconditioner` and `BootstrapAMGPreconditioner` override
   `setup` and `_commit` their realized hierarchy.

6. **Structural hierarchy levels.** `MultigridHierarchy(fine, coarsest)` and
   `PreparedHierarchy(fine, coarsest)` hold a tuple of fine levels plus exactly one
   coarsest level, over distinct level types (`MultigridFineLevel`/`MultigridCoarsestLevel`
   unprepared, `PreparedFineLevel`/`PreparedCoarsestLevel` prepared; the `*Level` unions
   remain as aliases and `levels` is the read-only view). Dropping the coarsest level and a
   coarsest level in the middle are no longer representable; constructors raise `TypeError`
   for a mis-typed level, and the cycles recurse over `fine` with no `match`. A
   single-level hierarchy (empty `fine`) stays representable because BootCMatch can stop
   at one level; AMG rejects `n_levels < 2` itself. `assemble_hierarchy` is the shared
   builder for the adaptive and bootstrap setups.

## Deviations from the plan

- Stage 4 is a derived `is_ready` per class, not one `_prepared` state: collapsing the
  registered buffers into one object would break `.to()` and `state_dict`.
- `GaussSeidelMemo.prepare` keeps the name `prepare`.

## Measured (CPU, float64)

- Restored masked gather (`torch.where(found, values[pos], 0)` instead of `* found`):
  0-9% of the numeric loop at lap2d n = 10 000 (re-timed: IC(0) +1-16% run to run, median
  about +4%; ILU(0) +1-4%), bit-identical on finite input.
- IC(0) `setup` on new values with the same pattern: 2.2-3.7x faster; first `setup` within
  noise.
- ILU(0) `setup` on new values: 4.2-5.0x faster; first `setup` within noise.
- int32 structural arrays: -48% pattern memory (lap2d n = 10 000: 1.32 -> 0.69 MB).
- Pattern memory as a multiple of the matrix's own memory (reviewer): IC(0) 2.7x on a
  9-point 2-D stencil and 4.8x on a 27-point 3-D stencil; ILU(0) 5.3x and 12x on the same
  two.
- Factors, one application and PCG/FCG iteration counts equal the pre-change code.

## Breaking changes

- AMG constructors and the `vcycle_amg`/`wcycle_amg`/`pod2g_preconditioner`/Adaptive/
  Bootstrap factories no longer take the matrix; pass it to `setup(matrix)`.
- `IC0Preconditioner(threshold)`, `dense_ic0(matrix, threshold)` and
  `ic0_sparsity_mask(matrix, threshold)` lose `threshold` (dense and sparse);
  `sparse_ic0(matrix, pattern=None)`.
- Renames: `SparseTriangularSolveCache` -> `PreparedTriangularPair`, `GaussSeidelStage` ->
  `GaussSeidelMemo`, `TransferOperatorCache` -> `TransferOperatorMemo`,
  `NeighborhoodCache` -> `NeighborhoodMemo`, `.build` -> `prepare_*_gauss_seidel`,
  `SymmetricGaussSeidel` -> `PreparedGaussSeidel`, `DenseBackedSymmetricGaussSeidel` ->
  `PreparedDenseGaussSeidel`.
- `Identity` is defined in `torchalg.preconditioners.base` (the solvers' default, replacing
  `_IdentityPreconditioner`); `implementations.Identity` re-exports it.
- Sparse `IC0Preconditioner`/`ILUPreconditioner` extend `MatrixSetupPreconditioner` and
  define `is_ready` (`_operator is not None`); their dead `_compute_operator` and the
  duplicate `ILUPreconditioner._lower` buffer are gone.
- The shared index helpers moved from `_ic0_pattern` to `_pattern_index`
  (`structural_index`, `build_lookup`, `lookup`, `row_pointer`, public inside the private
  module).
- `_mark_ready` removed; `Preconditioner.is_ready` is abstract.
- `HierarchyNotPreparedError` subclasses `PreconditionerNotReadyError`.
- `MultigridLevel` loses its smoother and coarse-solve fields.
- `AMGPreconditioner.state_dict()` no longer has `_matrix`.
- `ScheduledPreconditioner` readiness is `primary.is_ready and fallback.is_ready`: a
  schedule over static components is ready before `setup`; a failed `setup` can leave the
  first component on the new matrix.

## Open decisions

- Pattern memory multiple: the gather arrays dominate (2.7x-12x the matrix).
- One `_prepared` state per class versus derived `is_ready` over registered buffers.
- `ScheduledPreconditioner` readiness and atomicity: a failed `setup` can leave the first
  component on the new matrix.
