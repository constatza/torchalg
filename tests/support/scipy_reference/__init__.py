"""Frozen scipy CG oracle, vendored from ``neuralls.domain.solver`` (Stage 0).

This package is a **correctness-baseline snapshot**, not new code: it exists
solely so later migration stages have a numpy/scipy ground truth to compare
the torch-native port against. It has zero dependency on ``torchalg`` or
``neuralls`` and must stay numpy/scipy-only forever (never import torch).

Provenance:
    Copied from ``neuralls.domain.solver`` at
    ``/home/archer/projects/dl-experiments/src/neuralls/domain/solver``.
    ``scipy_wrapper.py`` (``SciPyCGSolver``) and ``monitoring/callbacks.py``
    (``SciPyCallbackAdapter``, ``InitialStateComputer``) are copied verbatim,
    along with every module they transitively import at runtime.

Deviations from a literal verbatim copy (all mechanical, no logic changes):
    - Two absolute ``from neuralls.domain.solver...`` imports (in
      ``monitoring/event_log.py`` and ``monitoring/iteration_history.py``)
      were rewritten to relative imports, since the code was physically
      relocated and ``neuralls`` is not, and must never become, a dependency
      of this package.
    - Package ``__init__.py`` files are hand-written minimal re-exports
      rather than verbatim copies of the reference's aggregating
      ``__init__.py``. The reference's ``preconditioners/__init__.py`` and
      ``preconditioners/implementations/__init__.py`` eagerly import the
      entire preconditioner subsystem (AMG, ILU, IC0, ICholesky, POD,
      neural) — POD's ``basis.py`` already imports **torch**, which this
      package must never depend on, and ILU/IC0 pull in scipy/numba
      factorization backends unrelated to the CG oracle. Likewise
      ``models/__init__.py`` aggregates ``config.py``, which depends on
      ``neuralls.shared.types`` (an external package this snapshot cannot
      and must not depend on), and ``utils/__init__.py`` aggregates
      ``utils/export/*``, explicitly scope-trimmed by the migration plan.
      Only the leaf modules actually imported by ``SciPyCGSolver`` /
      ``SciPyCallbackAdapter`` (traced transitively) are present here:
      ``preconditioners.base.Preconditioner``/``PreconditionerContext``,
      ``preconditioners.implementations.identity.Identity``, and
      ``models.result.SolverResult`` (trimmed to just that one dataclass —
      the reference's other five dataclasses in the same file belong to the
      excluded ``comparison.py``/``utils/export`` subsystem and one of them
      has the same external ``neuralls.shared.types`` dependency above).
    - ``strategies/convergence.py`` + ``strategies/norms.py`` are included
      even though ``scipy_wrapper.py`` itself does not import them: they are
      needed by the exactness-benchmark test helpers
      (``tests/benchmarks/exactness/conftest.py``), which reuse
      ``CombinedToleranceCriterion`` exactly as the reference's own
      benchmark suite does.
"""
