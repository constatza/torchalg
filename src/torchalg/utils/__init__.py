"""Pure, stateless numerical helpers shared by every solver/preconditioner.

The dependency floor of ``torchalg`` (see ``docs/plan.md``'s
"Dependency architecture (tach)" section): everything else may depend on
this package, but it depends on nothing else in ``torchalg``.
"""
