# Preconditioners

`torchalg.preconditioners` defines the preconditioner abstraction,
predictor ports, and concrete torch-native preconditioner implementations.

`base.py` and `ports.py` are interface layers. Solver orchestration should
depend on those abstractions; `factories.py` is the composition root that may
instantiate concrete implementations such as `Identity`.

Implementation modules keep validation local to their own scheduling or
factorization rules. `ScheduledPreconditioner` rejects negative `start_iter`
and negative bounded windows, and `NeuralPreconditioner` binds extra tensors
against the same public `extra_input_names` contract used by solver factories.

`PODCoarseningStrategy` splits construction from fitting: `__init__(rank)`
stores the target rank without registering a basis buffer, and `fit(snapshots)`
runs the one-shot SVD and registers it. This lets an instance exist before
snapshot data does (e.g. a checkpointed training-job definition) and be
reconstructed from a `state_dict` without recomputing the SVD - the resolved
mode count is readable afterward via the `rank` property. `build_transfer`
and `rank` both raise `RuntimeError` if called before `fit()`.
`POD2GPreconditioner` still takes `snapshots` in one call and fits internally,
so its public signature is unchanged.

AMG presets currently expose one `omega` value for both weighted-Jacobi cycle
smoothing and smoothed-aggregation prolongation smoothing. These are distinct
Jacobi operations; a future AMG API should split them into separate parameters
once backward compatibility can be managed.

AMG hierarchy depth is counted as total levels, including the finest matrix.
`n_levels=2` is the minimum valid multigrid hierarchy and means one fine level
plus one coarse level; `n_levels=1` is rejected because it would bypass
coarsening and reduce to a direct dense solve on the original system.
