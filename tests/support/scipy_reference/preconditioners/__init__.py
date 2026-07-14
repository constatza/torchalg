"""Narrowed snapshot of ``neuralls.domain.solver.preconditioners``.

Only the base ABC, ``Identity``, and ``CallablePreconditioner`` are
re-exported: they are the preconditioner types the CG oracle and its test
fixtures need (``CallablePreconditioner`` lets fixture-level ``scipy_cg``
wrappers accept a plain ``Callable[[NDArray], NDArray]``, mirroring the
reference's ``factories.py::scipy_cg``). See the package-level docstring in
``tests.support.scipy_reference`` for why this ``__init__.py`` is
hand-written rather than a verbatim copy of the reference's aggregator
(which eagerly imports the entire AMG/ILU/IC0/ICholesky/POD/neural
preconditioner subsystem, including a torch dependency via POD).
"""

from .base import Preconditioner, PreconditionerContext
from .callable import CallablePreconditioner
from .implementations.identity import Identity

__all__ = ["CallablePreconditioner", "Identity", "Preconditioner", "PreconditionerContext"]
