"""Narrowed snapshot of ``neuralls.domain.solver.models``.

Only ``SolverResult`` is re-exported: it is the sole model type
``SciPyCGSolver``/``SciPyCallbackAdapter`` need. See the package-level
docstring in ``tests.support.scipy_reference`` for why this ``__init__.py``
is hand-written rather than a verbatim copy of the reference's aggregator.
"""

from .result import SolverResult

__all__ = ["SolverResult"]
