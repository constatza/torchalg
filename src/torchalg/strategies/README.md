# Solver Strategies

`torchalg.strategies` contains stateless policy objects used by solver
orchestration:

- `norms.py` defines convergence norm functions.
- `convergence.py` defines tolerance criteria. `converged_flag_from_norm`
  is the stopping test as a device-side bool (no synchronization, no
  conversion launches when both norms are tensors of one dtype and device);
  `has_converged_from_norm` is the same test read to the host with one
  synchronization. The CG step reads the flag together with its finiteness
  flag in a single transfer.
- `orthogonalization.py` defines A-conjugacy orthogonalization strategies for
  FCG and optional PCG reorthogonalization. They consume a `DirectionWindow`
  (`models/history.py`): classical Gram-Schmidt (`PeriodicRestartOrthogonalization`,
  `TruncatedGramSchmidt`) takes every numerator from the original vector, so
  it is two matrix-vector products using the curvatures stored with the
  window. The projections are unchecked: their finiteness stays on the device
  as `OrthogonalizationReport.projections_finite`, which the CG step reads in
  its single per-iteration transfer. When it is false the caller repeats the
  call with `overflow_safe_retry=True`, which takes every projection with a
  per-pair `stable_dot_product`, reuses the same window and does not advance
  the periodic-restart counter. `ModifiedGramSchmidt` stays sequential (each
  numerator uses the updated vector), cannot be collapsed that way, and its
  numerators are already stable.
- `direction.py` defines CG search-direction policies: two-term recurrence,
  orthogonalization-based FCG, and composite PCG reorthogonalization. A
  strategy receives `(w, rw, state)` where `rw = (r, w)` is already computed
  by the solver step; at iteration 0 it must return a copy of `w` (the solver
  reuses the precomputed `A w`). A strategy returns the direction with its
  `OrthogonalizationReport` (breakdown and projection-finiteness flags), or
  `None` when it does not orthogonalize; `overflow_safe_retry` is passed
  through to the orthogonalization.

Strategies may depend on solver models and low-level utilities, but
they do not import preconditioners or solver factories.
