"""Narrowed snapshot of ``neuralls.domain.solver.preconditioners.implementations``.

Only ``Identity`` is re-exported here; see
``tests.support.scipy_reference.preconditioners``'s docstring.
"""

from .identity import Identity

__all__ = ["Identity"]
