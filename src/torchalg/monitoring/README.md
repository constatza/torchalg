# Solver Monitoring

`torchalg.monitoring` contains continuous per-iteration telemetry for
solver loops. It is intentionally separate from termination diagnostics:
`IterationHistory` records values collected over time, while convergence and
breakdown facts are stored once in `models.TerminationDiagnostics`.

The package depends only on `torchalg.utils` (the shared `energy_dot`
A-inner-product primitive used by `analysis.py`) - never on solver state,
strategies, or preconditioners.

## Components

- `TraceMode` controls the amount of *vector* telemetry collected:
  - `DISABLED`: no data is recorded at all.
  - `MINIMAL`: residual norms only.
  - `FULL`: residual norms plus residual, solution, and direction tensors.
- `ScalarHistory` stores immutable scalar sequences.
- `VectorHistory` stores immutable `torch.Tensor` sequences, always moved to
  CPU the moment each vector is added (`add`/`prepend`), regardless of the
  solve device - this bounds device memory during a `FULL`-traced solve to
  the algorithm's own working set; see
  `docs/bug-full-trace-history-exhausts-gpu-memory.md`. `to_tensor()` stacks
  the (CPU) sequence.
- `IterationHistory` is a small mutable container that applies trace-mode
  policy by replacing immutable histories with updated instances. Two scalar
  channels run independently of `TraceMode` (cheap - O(1) floats/iteration,
  never vectors):
  - `error_norms`: exact `||u_k - x_exact||_A`, populated whenever an
    `x_exact` (known solution) is supplied to the constructor, via a single
    dot product - no extra matvec, no stored vectors.
  - `energy_decrements`: CG's exact per-iteration `alpha_k * rho_k` (decrease
    in `||e_k||_A^2`), populated automatically whenever history is enabled at
    all - no ground truth required.
- `analysis.py` (`energy_norm_history`, `golub_meurant_error_bound`):
  explicit, opt-in post-hoc functions over a *completed* solve's recorded
  history. Never run automatically inside `solve()` - `energy_norm_history`
  needs `A` back on a device in caller-sized chunks, and
  `golub_meurant_error_bound`'s look-ahead `delay` is a caller tradeoff.
