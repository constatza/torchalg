# Solver Models

`torchalg.models` contains immutable records shared by solver
orchestration, factories, and comparison reporting.

Configuration and result objects are frozen dataclasses. Tuple storage is used
for result histories, and `SolverConfig.extra_params` is copied into a read-only
mapping at construction so callers cannot mutate configuration state through a
nested dictionary after creation.

This module is a low-level dependency. It must not import solver orchestration,
monitoring, concrete preconditioners, or strategy implementations that would
invert the dependency direction.

Execution outputs live in `result.py`; iteration-time inputs live separately
in `context.py`. Keeping the two records apart prevents result reporting from
coupling to adaptive iteration collaborators.

`DirectionHistory` (`history.py`) is the one deliberately mutable record: an
append-only accumulator, owned by a single solve, holding FCG's last `m`
`(d_j, q_j)` pairs and their curvatures `(d_j, q_j)` in preallocated
`(capacity, n)` buffers (grown geometrically up to `m` plus `m // 2` rows of
slack, window shifted to the front once per `m // 2` appends). `window()`
returns a `DirectionWindow` of contiguous views so orthogonalization is two
matrix-vector products instead of a Python loop; when autograd tracks the
stored pairs the views are cloned so later in-place appends cannot invalidate
saved tensors. `CGState` itself stays frozen and carries the first-breakdown
iteration as one device-side integer tensor.
