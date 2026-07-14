# Solver Models

`torchalg.models` contains immutable records shared by solver
orchestration, factories, and comparison reporting.

Configuration and result objects are frozen dataclasses. Tuple storage is used
for histories, and `SolverConfig.extra_params` is copied into a read-only
mapping at construction so callers cannot mutate configuration state through a
nested dictionary after creation.

This module is a low-level dependency. It must not import solver orchestration,
monitoring, concrete preconditioners, or strategy implementations that would
invert the dependency direction.
