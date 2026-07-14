# Solver Monitoring

`torchalg.monitoring` contains continuous per-iteration telemetry for
solver loops. It is intentionally separate from termination diagnostics:
`IterationHistory` records values collected over time, while convergence and
breakdown facts are stored once in `models.TerminationDiagnostics`.

The package has no internal `torchalg` dependencies and is a leaf in the
`tach` graph. Solver orchestration can depend on monitoring, but monitoring
must not depend on solver state, strategies, or preconditioners.

## Components

- `TraceMode` controls the amount of telemetry collected:
  - `DISABLED`: no data is recorded.
  - `MINIMAL`: residual norms only.
  - `FULL`: residual norms plus residual, solution, and direction tensors.
- `ScalarHistory` stores immutable scalar sequences.
- `VectorHistory` stores immutable cloned `torch.Tensor` sequences and stacks
  them with `to_tensor()`.
- `IterationHistory` is a small mutable container that applies trace-mode
  policy by replacing immutable histories with updated instances.
