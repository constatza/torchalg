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

`compute_pod_basis`/`PODCoarseningStrategy.fit` take an optional `row_scales`
(shape `(n_samples,)`), multiplied into each snapshot row before the SVD.
Because row scaling only touches the sample axis, `snapshots'^T snapshots' =
sum_k row_scales_k^2 e_k e_k^T` - a weighted covariance - without ever
changing the SVD's (Euclidean) inner product or requiring a
back-transform of the returned basis. `pod.weighting` computes that vector
for three schemes: `power_norm_scales` (row scale `||e_k||^(-beta)`, `beta=0`
raw / `beta=1` fully normalized / in between interpolates, `metric="l2"` or
`"a"`), `smoother_persistence_scales` (row scale by how much of a snapshot's
norm survives `steps` weighted-Jacobi sweeps - directions the smoother
already handles well contribute little), and the `apply_jacobi_damping`
helper both build on top of (also usable directly by snapshot-generation
code that wants "algebraically smooth" probe vectors, reusing
`JacobiSmoother` rather than a second Jacobi-map implementation). `None`
(default) is the original unweighted behavior, unchanged.

AMG presets currently expose one `omega` value for both weighted-Jacobi cycle
smoothing and smoothed-aggregation prolongation smoothing. These are distinct
Jacobi operations; a future AMG API should split them into separate parameters
once backward compatibility can be managed.

AMG hierarchy depth is counted as total levels, including the finest matrix.
`n_levels=2` is the minimum valid multigrid hierarchy and means one fine level
plus one coarse level; `n_levels=1` is rejected because it would bypass
coarsening and reduce to a direct dense solve on the original system.

`TargetDimensionCoarsening` wraps `AggregationCoarsening` from the outside to
give it the same "set the coarse dimension directly" ergonomics
`PODCoarseningStrategy`'s `rank` already has: realized coarse dimension vs.
`theta` is an emergent, empirically step-function (not smooth, not
monotonic) response, so `_search` uses `adaptive_theta_scan`
(`_theta_search.py`) - a cheap dimension-only probe (strength +
aggregation, skipping prolongation smoothing and the Galerkin product) over
a log-spaced coarse pass plus bisection of every disagreeing interval - and
pays for a full `AggregationCoarsening.build_transfer` exactly once, at the
winning `theta`, caching it as `_theta`/`_realized_coarse_dim` afterward.
Deliberately adaptive sampling, not a plain fixed grid (a plateau narrower
than the grid spacing can sit between two agreeing samples and get missed
entirely), bisection alone (assumes monotonicity, which doesn't hold), or a
black-box optimizer like Optuna (built for expensive, smooth,
higher-dimensional objectives — none of which describes a single cheap
bounded scalar with a step-function response). `cache_candidates=True`
additionally shares that one full build across sibling instances searching
the same matrix object (e.g. several `target_coarse_dim` values in one
comparison sweep) via a module-level `functools.lru_cache`; off by default
since it's only a win when multiple instances against the same matrix are
expected.
