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
`"energy"`), `smoother_persistence_scales` (row scale by how much of a snapshot's
norm survives `steps` weighted-Jacobi sweeps - directions the smoother
already handles well contribute little), and the `apply_jacobi_damping`
helper both build on top of (also usable directly by snapshot-generation
code that wants "algebraically smooth" probe vectors, reusing
`JacobiSmoother` rather than a second Jacobi-map implementation). `None`
(default) is the original unweighted behavior, unchanged.

`amg.hierarchy.build_hierarchy(matrix, coarsening, n_levels)` is the pure
function `AMGPreconditioner` uses to build its levels, so setup code can build
trial hierarchies without instantiating a preconditioner.
`apply_jacobi_damping[_trajectory]` live in `amg/_test_vectors.py` (`pod`
depends on `amg`, never the reverse) and are re-exported by `pod.weighting`.

`AdaptiveSAPreconditioner` (`amg/adaptive.py`) is a step-by-step dense-torch
port of PyAMG 5.3.0's `adaptive_sa_solver` (`initial_setup_stage` =
Algorithm 3 and `general_setup_stage` = Algorithm 4 of Brezina et al. 2005).
Every kernel is a port of the PyAMG routine it calls and is tested against
it on identical inputs (`tests/.../test_adaptive_sa_port.py`): `_tentative.py`
(`fit_candidates`: modified Gram-Schmidt per aggregate with PyAMG's
`tol=1e-10` drop rule, always `m` columns per aggregate), `_node_strength.py`
(symmetric strength on the node graph; coarse levels carry `k` dofs per node,
strength uses block Frobenius norms), `_prolongation.py` (Jacobi prolongator
smoothing with `omega / rho(D^-1 A)` and the bridging prolongator),
`_spectral.py` (restarted-Arnoldi estimate of `rho`, random start, ~1 %
tolerance), and `_relaxation.py` (symmetric Gauss-Seidel as triangular solves;
PyAMG's block GS is point-wise, hence identical). The hierarchy is applied
with a V(1,1)-cycle, `GaussSeidelSmoother` and a pseudo-inverse coarse solve
(`cycle.pseudo_inverse_solve`), PyAMG's defaults. Defaults follow PyAMG
(`theta=0`, `omega=4/3`, `candidate_iters=5`, `max_levels=max_coarse=10`).
All random vectors (including the spectral-radius starts) come from one
injectable `draw` source, so a run can replay NumPy's stream and be compared
with PyAMG number for number. Not ported: `improvement_iters`,
`eliminate_local`, `epsilon`/`pdef` (only used by a branch PyAMG disables),
non-default smoothers/strength/aggregation/coarse solvers, complex and
nonsymmetric matrices, and the `work` counter. One documented difference:
`standard_aggregation`'s second pass depends on the stored column order of
the sparse coarse matrices in PyAMG (unsorted after sparse products); this
port uses ascending order, PyAMG's result on a sorted-index copy of the same
graph.

`AMGPreconditioner._make_hierarchy` is the hook subclasses override when their
levels are computed elsewhere (as `AdaptiveSAPreconditioner` does).

Weighted-Jacobi damping has two roles with separate parameters, both
defaulting to a spectral rule with `rho = rho(D^-1 A)` (`amg/_jacobi_omega.py`,
PyAMG's rules): **relaxation** (`JacobiSmoother`, presets' `smoother_omega`,
applied to the error every cycle) uses `omega = 1 / rho`; **prolongation
smoothing** (`AggregationCoarsening`, presets' `prolongation_omega`, one Jacobi
step on the tentative prolongator at setup) uses `omega = (4/3) / rho`
(Vanek, Mandel & Brezina 1996). `None` selects the rule, a float fixes the
value. `rho` is a seeded, hence deterministic, Arnoldi estimate
(`_spectral.approximate_spectral_radius`), cached per matrix object and
invalidated on in-place modification, so the smoother and the prolongation
smoothing of the same level, and every `apply`, share one estimate per level.
For `rho ~= 2` the rules give about 0.5 and 0.67; unlike a fixed 0.67 they keep
the smoother convergent when `rho > 3` (`omega` must stay below `2 / rho`).
The POD helpers `apply_jacobi_damping[_trajectory]` and
`smoother_persistence_scales` default to the relaxation rule as well.
Gauss-Seidel has no damping parameter.

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

`BootstrapAMGPreconditioner` (`amg/bootstrap.py`) is Bootstrap AMG (BAMG,
Brandt/Brannick/Kahl/Livshits 2011, `docs/bootstrap-amg.md`): unlike
`AggregationCoarsening`/`AdaptiveSAPreconditioner`, the C/F split, the
strength-of-connection measure and the interpolation weights are all
derived from test vectors instead of `A`'s raw entries or an M-matrix sign
assumption. `_compatible_relaxation.py` (compatible relaxation, choosing
`C`), `_algebraic_distance.py` (the algebraic-distance strength graph
`M_d`) and `_least_squares.py` (LS/LSR interpolation, greedy
caliber-bounded interpolatory-set selection) are pure per-kernel modules;
`bootstrap.py`'s `BAMGCoarsening` wires them into one `build_transfer(A)` =
one level of the paper's setup algorithm, `BootstrapSetup.run` is the outer
Sec. 4.1/5 loop (build the initial hierarchy from relaxed random test
vectors, then improve every level's vectors for `n_bootstrap_cycles` by
running the current partial hierarchy on `A x = 0` and rebuilding), and
`BootstrapAMGPreconditioner` is the `AdaptiveSAPreconditioner`-shaped preset
(`_PrebuiltCoarsening` + `_make_hierarchy` override) around it. Not built in
v1: the multigrid eigensolver (MGE, Sec. 4.2) - bootstrap cycles improve
test vectors by relaxation/cycling alone. `compatible_relaxation_coarsening`
takes an optional keyword-only `guidance_graph` overriding its default
plain-matrix-graph independent-set guide; `BAMGCoarsening` passes the
algebraic-distance strength graph there, computed once per level (before any
point is marked coarse) and held fixed through CR's outer loop, matching
that parameter's own fixed-for-the-whole-call shape.

Verified by property tests (`tests/solver/preconditioners/implementations/amg/test_algebraic_distance.py`,
`test_compatible_relaxation.py`, `test_least_squares.py`, `test_bootstrap.py`):
`algebraic_distance` is directional (`r_ij != r_ji` in general, by
design - a one-sided LS fit at node `i`, not a functional symmetric in
`i, j`), zero outside the depth neighborhood, and matches a direct
weighted-LS fit; `hcr_operator` is an idempotent projector on `F`, `cr_rate`
stays in `[0, 1)`; `ls_interpolation_row` reproduces the test vectors
exactly when the interpolatory set has full local rank, `lsr_correction`
never increases fit error; `BAMGCoarsening` always returns a PSD Galerkin
operator (with and without LSR) and, after a real bug was caught and fixed,
never appends a degenerate zero-size coarsest level (CR can legitimately
converge to an empty coarse set on an already-coarsened level -
`BootstrapSetup._build_levels` now discards that pass and keeps the
previous level as the final one); `BootstrapAMGPreconditioner` measurably
reduces PCG iterations vs. unpreconditioned PCG and is linear (plain `pcg`,
no `flexible_cg` required).

Unlike `AdaptiveSAPreconditioner`, **no PyAMG oracle exists for this
algorithm**. PyAMG does have a `pyamg.classical.cr.CR` compatible-relaxation
splitter and a `pyamg.strength.algebraic_distance` strength measure, but
neither matches this module's algorithm - `CR` is a different (non
AD11-guided) compatible-relaxation formulation, `algebraic_distance` is a
different, unrelated measure (Safro/Sanders/Schulz), and PyAMG's
`classical/interpolate.py` has no least-squares interpolation scheme at all
(only `classical_interpolation`/`direct_interpolation`/
`injection_interpolation`/`one_point_interpolation`/`local_air`) to wire
either of them into. PyAMG therefore has no complete, assembled
Bootstrap-AMG pipeline to validate the whole algorithm against. Correctness
instead rests on the property tests above plus a `benchmark`-marked paper-table reproduction
(`tests/benchmarks/preconditioners/test_bootstrap_amg.py`) checked against
[BAMG11]'s Tables 4.2/4.3 convergence-factor trends on this codebase's own
Poisson fixtures. That benchmark also surfaces a real, known accuracy gap,
stated plainly rather than swept under the rug: **this implementation seeds
no a priori near-null vector** (e.g. the constant vector `1`) into the
test-vector set - only `k_r` random draws - and [BAMG11] Sec. 6 itself
reports that without that seed, LSR's advantage over LS degrades with
problem size. This codebase's own measurements (averaged over independent
`BootstrapSetup` seeds) found LS and LSR statistically indistinguishable
here, sometimes with LSR measurably worse than LS for a given setup seed,
rather than the paper's reliably "LSR wins" trend; the benchmark demonstrates
LSR beating LS only at specific, documented seeds/problem sizes, not as a
general property. LSR's paper-reported advantage should not be assumed to
hold without near-null-vector seeding, which is not implemented here.
