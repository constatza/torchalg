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

`placement.py` chooses the device of a solve that passes no `device=`: from the
preconditioner's `Placement` (sequential sweeps, SpMV-bound, dense-bound) and the matrix size,
CUDA only where a GPU is expected to beat the CPU. Sweep-bound preconditioners (IC(0), ILU,
ICholesky, and AMG or BootCMatch with a Gauss-Seidel smoother) stay on the CPU; Jacobi-smoothed
AMG, BootCMatch and POD-2G are SpMV-bound; the thresholds (`SPMV_GPU_MIN_NNZ`,
`DENSE_GPU_MIN_N`) are provisional ROCm measurements. A matvec callable for `A` keeps the
operands on `b`'s device. The cost of each building block is documented in its docstring's
`Cost:` section, defined in `docs/implementations/cost-model.md`.

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

Host synchronizations per CG iteration: one (PCG) or two (FCG). The solver
step takes both plain dots unchecked, computes the next residual and its
stopping test speculatively, and reads the finiteness flag of the dot pair
(`pair_is_finite`) together with the stopping flag
(`IConvergenceCriterion.converged_flag_from_norm`) in one transfer
(`read_flags`); the result lands in `state.converged`, so `_check_stopping`
only evaluates the criterion for the initial state. A criterion with a
non-Euclidean norm needs the residual vector and keeps its own read in
`_check_stopping` (two syncs per PCG iteration). A non-finite pair sends the
iteration through `_recompute_overflow_safe`, which redoes it with
`stable_dot_product` and reads the stopping flag again (a stateful
orthogonalization window counter advances twice on such an iteration). FCG
adds the window-projection finiteness read. `u`, `r` and the two-term
direction `w + beta d` update with fused `addcmul`, which may contract a
multiply-add, so results agree with the unfused formulation to a few ulp
rather than bit for bit.

System-matrix index width: `_prepare_operator` applies a CSR system matrix
through int32 indices (`utils.csr_index.compact_csr_indices`, a shared leaf so
`torchalg.base` need not import `torchalg.sparse`): the CPU sparse BLAS works
on 32-bit indices and otherwise narrows them on every product. The narrowed
tensor shares the values and is local to the solve; the caller's matrix is
unchanged. A matrix that requires grad, a dense or COO matrix, and a matvec
callable are not touched.

Sparse AMG keeps its existing greedy and compatible-relaxation coarsening
algorithms as correctness baselines. Any tensor-parallel coarsening path is
a separate algorithm and strategy, not an alternate implementation expected
to reproduce the baseline's aggregate assignments.
