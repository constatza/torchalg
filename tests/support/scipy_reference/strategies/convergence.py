"""Convergence criteria abstractions for iterative solvers.

This module provides convergence criterion implementations with injectable norms.
The norm is injected at construction time, following Dependency Inversion.

Design:
    - IConvergenceCriterion: Abstract base defining convergence interface
    - CombinedToleranceCriterion: SciPy-style ||r||_norm <= max(rtol * ||b||, atol)

The criterion is agnostic to the specific norm used. Different norms can be
injected to implement different convergence criteria (L2, A-norm, etc.).

Example:
    >>> from neuralls.domain.solver.strategies.convergence import CombinedToleranceCriterion
    >>> from neuralls.domain.solver.strategies.norms import euclidean_norm, energy_norm
    >>> import numpy as np
    >>>
    >>> # Default L2 norm criterion
    >>> criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
    >>>
    >>> # A-norm criterion for diagonal matrix
    >>> A_diag = np.array([2.0, 8.0])
    >>> criterion_anorm = CombinedToleranceCriterion(
    ...     rtol=1e-6, atol=1e-14, norm=energy_norm(A_diag)
    ... )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .norms import Norm, euclidean_norm

if TYPE_CHECKING:
    from numpy.typing import NDArray


class IConvergenceCriterion(ABC):
    """Abstract base class for convergence checks.

    Subclasses receive norm at construction time (dependency injection).
    The criterion delegates magnitude computation to the injected norm.
    """

    norm: Norm

    @abstractmethod
    def threshold(self, rhs_norm: float) -> float:
        """Return the stopping threshold for a given RHS norm.

        Args:
            rhs_norm: Norm of the right-hand side ||b||.

        Returns:
            Convergence threshold value.
        """

    @abstractmethod
    def has_converged(self, residual: NDArray, rhs_norm: float) -> bool:
        """Determine convergence given residual and RHS norm.

        Uses the injected norm function - criterion doesn't know which norm.

        Args:
            residual: Current residual vector r_k.
            rhs_norm: Norm of the right-hand side ||b||.

        Returns:
            True if converged, False otherwise.
        """


@dataclass(frozen=True)
class CombinedToleranceCriterion(IConvergenceCriterion):
    """SciPy-style criterion: ||r||_norm <= max(rtol * ||b||, atol).

    The norm function is injected at construction time via dependency injection.
    Default: euclidean_norm (L2). Can be replaced with energy_norm(A) for A-norm.

    Attributes:
        rtol: Relative tolerance (multiplied by ||b||).
        atol: Absolute tolerance (lower bound).
        norm: Norm function to use (default: euclidean_norm).

    Example:
        >>> # L2 norm convergence (default)
        >>> criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        >>>
        >>> # A-norm convergence for SPD matrix A
        >>> from neuralls.domain.solver.strategies.norms import energy_norm
        >>> criterion_A = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14, norm=energy_norm(A))
    """

    rtol: float
    atol: float
    norm: Norm = field(default=euclidean_norm)

    def threshold(self, rhs_norm: float) -> float:
        """Compute threshold: max(rtol * ||b||, atol).

        Args:
            rhs_norm: Norm of the right-hand side ||b||.

        Returns:
            Convergence threshold value.
        """
        return max(self.rtol * rhs_norm, self.atol)

    def has_converged(self, residual: NDArray, rhs_norm: float) -> bool:
        """Check: ||r||_norm <= threshold.

        Uses injected norm - criterion is agnostic to norm type.

        Args:
            residual: Current residual vector r_k.
            rhs_norm: Norm of the right-hand side ||b||.

        Returns:
            True if ||r||_norm <= max(rtol * ||b||, atol), False otherwise.
            Returns False if residual norm or rhs_norm is non-finite (NaN/Inf).
        """
        import numpy as np

        residual_norm = self.norm(residual)

        # Check for non-finite values - not converged if NaN/Inf present
        if not np.isfinite(residual_norm) or not np.isfinite(rhs_norm):
            return False

        return bool(residual_norm <= self.threshold(rhs_norm))
