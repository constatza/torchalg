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

Sparse AMG keeps its existing greedy and compatible-relaxation coarsening
algorithms as correctness baselines. Any tensor-parallel coarsening path is
a separate algorithm and strategy, not an alternate implementation expected
to reproduce the baseline's aggregate assignments.
