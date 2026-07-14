# Solver Strategies

`torchalg.strategies` contains stateless policy objects used by solver
orchestration:

- `norms.py` defines convergence norm functions.
- `convergence.py` defines tolerance criteria.
- `orthogonalization.py` defines A-conjugacy orthogonalization strategies for
  FCG and optional PCG reorthogonalization.
- `direction.py` defines CG search-direction policies: two-term recurrence,
  orthogonalization-based FCG, and composite PCG reorthogonalization.

Strategies may depend on immutable solver models and low-level utilities, but
they do not import preconditioners or solver factories.
