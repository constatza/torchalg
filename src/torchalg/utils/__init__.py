"""Small, stateless primitives shared by solvers and preconditioners.

The dependency floor of ``torchalg`` (see
``docs/dependency-architecture.md``; enforced by ``tach.toml``): everything else may depend on
this package, but it depends on nothing else in ``torchalg``. Optional
operator diagnostics belong in :mod:`torchalg.analysis`, not here.
"""
