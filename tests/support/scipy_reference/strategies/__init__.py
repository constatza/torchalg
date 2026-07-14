"""Narrowed snapshot of ``neuralls.domain.solver.strategies``.

Only ``convergence.py`` and ``norms.py`` are included: they are needed by
the exactness-benchmark test helpers (``tests/benchmarks/exactness/conftest.py``),
which reuse ``CombinedToleranceCriterion`` exactly as the reference's own
benchmark suite does. ``orthogonalization.py``/``direction.py`` are not
needed by anything in this snapshot and are intentionally omitted — see the
package-level docstring in ``tests.support.scipy_reference``.
"""

from .convergence import CombinedToleranceCriterion, IConvergenceCriterion
from .norms import Norm, energy_norm, euclidean_norm

__all__ = [
    "CombinedToleranceCriterion",
    "IConvergenceCriterion",
    "Norm",
    "energy_norm",
    "euclidean_norm",
]
