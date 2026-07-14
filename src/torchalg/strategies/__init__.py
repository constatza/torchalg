"""Strategy implementations for solver components."""

from .convergence import CombinedToleranceCriterion, IConvergenceCriterion
from .direction import (
    CompositeDirectionStrategy,
    DirectionStrategy,
    OrthogonalizationDirectionStrategy,
    TwoTermRecurrenceStrategy,
)
from .norms import Norm, energy_norm, euclidean_norm
from .orthogonalization import (
    ModifiedGramSchmidt,
    OrthogonalizationReport,
    OrthogonalizationStrategy,
    PeriodicRestartOrthogonalization,
    TruncatedGramSchmidt,
    create_fcg_orthogonalization,
)

__all__ = [
    # Norms
    "Norm",
    "euclidean_norm",
    "energy_norm",
    # Convergence
    "IConvergenceCriterion",
    "CombinedToleranceCriterion",
    # Direction
    "DirectionStrategy",
    "TwoTermRecurrenceStrategy",
    "OrthogonalizationDirectionStrategy",
    "CompositeDirectionStrategy",
    # Orthogonalization
    "OrthogonalizationStrategy",
    "OrthogonalizationReport",
    "PeriodicRestartOrthogonalization",
    "TruncatedGramSchmidt",
    "ModifiedGramSchmidt",
    "create_fcg_orthogonalization",
]
