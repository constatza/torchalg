"""Small, stateless primitives shared by solvers and preconditioners.

The dependency floor of ``torchalg`` (see ``docs/plan.md``'s
"Dependency architecture (tach)" section): everything else may depend on
this package, but it depends on nothing else in ``torchalg``. Optional
operator diagnostics belong in :mod:`torchalg.analysis`, not here.
"""
