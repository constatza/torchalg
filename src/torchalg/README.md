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
