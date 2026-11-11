# Solver Strategies

`torchalg.strategies` contains stateless policy objects used by solver
orchestration:

- `norms.py` defines convergence norm functions.
- `convergence.py` defines tolerance criteria.
- `orthogonalization.py` defines A-conjugacy orthogonalization strategies for
  FCG and optional PCG reorthogonalization. They consume a `DirectionWindow`
  (`models/history.py`): classical Gram-Schmidt (`PeriodicRestartOrthogonalization`,
  `TruncatedGramSchmidt`) takes every numerator from the original vector, so
  it is two matrix-vector products using the curvatures stored with the
  window, with a per-pair `stable_dot_product` fallback if a projection
  overflows. `ModifiedGramSchmidt` stays sequential (each numerator uses the
  updated vector) and cannot be collapsed that way.
- `direction.py` defines CG search-direction policies: two-term recurrence,
  orthogonalization-based FCG, and composite PCG reorthogonalization. A
  strategy receives `(w, rw, state)` where `rw = (r, w)` is already computed
  by the solver step; at iteration 0 it must return a copy of `w` (the solver
  reuses the precomputed `A w`). Strategies without orthogonalization return
  `None` as the breakdown flag.

Strategies may depend on solver models and low-level utilities, but
they do not import preconditioners or solver factories.
