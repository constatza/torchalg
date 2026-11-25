# Solver Package

`torchalg` contains the torch-native CG-family solver stack.

The solver layer is split into explicit dependency boundaries:

- `utils`, `analysis`, `models`, `monitoring`, and `strategies` provide
  low-level tensor primitives, optional numerical diagnostics, immutable
  records, telemetry, and algorithm policies.
- `preconditioners.base` defines the preconditioner abstraction.
- `base.py` and `conjugate_gradient.py` implement solver orchestration while
  depending only on preconditioner abstractions.
- `factories.py` is the composition root. It is the only solver module that
  imports concrete preconditioner implementations.
- `comparison/` separates result records, FCG execution, text presentation,
  and ranking policy. Its runner uses the factory default for the baseline,
  so it does not import concrete preconditioners directly.

Public solver entry points are `pcg()`, `flexible_cg()`, and
`run_cg_comparison()`.

`pcg()` and `flexible_cg()` run under inference mode by default; callers
request autograd through an unrolled solve with `differentiable=True`.
Returned `SolverResult` norms and histories are detached scalar telemetry,
while the returned solution retains its gradient graph in differentiable
mode.

A solve that passes no `device=` keeps a system matrix already on an accelerator
there, with `b`, `x0` and a module preconditioner moved to it; moving it to the CPU
would copy it back on every solve, so a caller solving many times on the GPU moves
`A` once. For a CPU-resident matrix, `placement.py` chooses the device: from the
preconditioner's `Placement` (sequential sweeps, SpMV-bound, dense-bound) and the matrix size,
CUDA only where a GPU is expected to beat the CPU. Sweep-bound preconditioners (IC(0), ILU,
ICholesky, and AMG or BootCMatch with a Gauss-Seidel smoother) stay on the CPU; Jacobi-smoothed
AMG, BootCMatch and POD-2G are SpMV-bound; the thresholds (`SPMV_GPU_MIN_NNZ`,
`DENSE_GPU_MIN_N`) are provisional ROCm measurements. A matvec callable for `A` keeps the
operands on `b`'s device. The cost of each building block is documented in its docstring's
`Cost:` section placed before `Args:`: flops `W`, bytes moved `Q`, depth `D`, kernel
launches `L` and host synchronisations `S`, as formulas, never measured times.

System validation accepts dense and PyTorch sparse tensors. For sparse
matrices it validates stored values, since implicit entries are finite zeros
and PyTorch does not implement `torch.isfinite` directly for CSR tensors.
Finiteness is one fused infinity-norm reduction (no `n^2` boolean mask), so a
`nan`/`inf` anywhere in `A` is still rejected at a fraction of the cost.

Work per solve: a `k`-iteration CG solve applies the preconditioner `k` times
and multiplies by `A` `k` times (`k + 1` with a nonzero `x0`, whose residual
needs `A x0`). The first iteration reuses the preconditioned residual,
direction and `A d` computed for the initial state, `(r, w)` is computed once
per iteration and shared with the direction strategy, and `(d, q)` once.
`stable_dot_product` is a plain `torch.dot` plus one finiteness test; only a
non-finite (overflowed) result falls back to the rescaled evaluation.

Host synchronizations per CG iteration: one, for PCG and FCG alike. The solver
step takes both plain dots and any orthogonalization projections unchecked,
computes the next residual and its stopping test speculatively, and reads the
finiteness flag of the dot pair (`pair_is_finite`), ANDed with the projection
flag `OrthogonalizationReport.projections_finite`, together with the stopping flag
(`IConvergenceCriterion.converged_flag_from_norm`) in one transfer
(`read_flags`); the result lands in `state.converged`, so `_check_stopping`
only evaluates the criterion for the initial state. A criterion with a
non-Euclidean norm needs the residual vector and keeps its own read in
`_check_stopping` (two syncs per PCG iteration). A non-finite pair sends the
iteration through `_recompute_overflow_safe`, which redoes it with
`stable_dot_product` and reads the stopping flag again (a non-finite `(r, w)`
recomputes the direction, so a stateful orthogonalization window counter
advances twice on such an iteration; non-finite projections retry the
direction with per-pair `stable_dot_product` projections and the same window).
`u`, `r` and the two-term
direction `w + beta d` update with fused `addcmul`, which may contract a
multiply-add, so results agree with the unfused formulation to a few ulp
rather than bit for bit.

System-matrix index width: `_place_on_device` narrows a CSR system matrix to
int32 indices (`utils.csr_index.compact_csr_indices`, a shared leaf so
`torchalg.base` need not import `torchalg.sparse`) before moving it, so a
host-to-device copy carries half the index bytes, and `_prepare_operator`
applies it through those indices (its own narrowing is then a no-op): the CPU
sparse BLAS works on 32-bit indices and otherwise narrows them on every
product. The narrowed
tensor shares the values and is local to the solve; the caller's matrix is
unchanged. A matrix that requires grad, a dense or COO matrix, and a matvec
callable are not touched.

Sparse AMG keeps its existing greedy and compatible-relaxation coarsening
algorithms as correctness baselines. Any tensor-parallel coarsening path is
a separate algorithm and strategy, not an alternate implementation expected
to reproduce the baseline's aggregate assignments.
