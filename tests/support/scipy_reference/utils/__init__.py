"""Narrowed snapshot of ``neuralls.domain.solver.utils``.

Only ``validation.py`` is included: ``SciPyCGSolver`` uses
``check_solution_validity``/``record_breakdown_event`` from it.
``numerics.py`` is not transitively needed by the CG oracle and additionally
depends on the external ``neuralls.shared.constants`` package, so it is
intentionally omitted, as is ``export/`` (explicitly scope-trimmed by the
migration plan). See the package-level docstring in
``tests.support.scipy_reference``.
"""
