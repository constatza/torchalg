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

Sparse AMG keeps its existing greedy and compatible-relaxation coarsening
algorithms as correctness baselines. Any tensor-parallel coarsening path is
a separate algorithm and strategy, not an alternate implementation expected
to reproduce the baseline's aggregate assignments.
