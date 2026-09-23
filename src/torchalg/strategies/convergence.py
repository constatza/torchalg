"""Convergence criteria abstractions for iterative solvers.

Ported from ``neuralls.domain.solver.strategies.convergence`` (see
``docs/plan.md``) with ``numpy`` operations translated to ``torch``
equivalents. Mathematically identical to the frozen numpy reference
snapshot in ``tests/support/scipy_reference/strategies/convergence.py``
(Stage 0) - the formula is what the scipy-equivalence benchmarks
ultimately depend on being right, so no behavior changed in translation,
only the tensor type.

This module provides convergence criterion implementations with injectable
norms. The norm is injected at construction time, following Dependency
Inversion.

Design:
    - ``IConvergenceCriterion``: Abstract base defining convergence
      interface.
    - ``CombinedToleranceCriterion``: SciPy-style
      ``||r||_norm <= max(rtol * ||b||, atol)``.

The criterion is agnostic to the specific norm used. Different norms can be
injected to implement different convergence criteria (L2, A-norm, etc.).

Example:
    >>> from torchalg.strategies.convergence import CombinedToleranceCriterion
    >>> from torchalg.strategies.norms import euclidean_norm, energy_norm
    >>> import torch
    >>>
    >>> # Default L2 norm criterion
    >>> criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
    >>>
    >>> # A-norm criterion for diagonal matrix
    >>> a_diag = torch.tensor([2.0, 8.0])
    >>> criterion_anorm = CombinedToleranceCriterion(
    ...     rtol=1e-6, atol=1e-14, norm=energy_norm(a_diag)
    ... )
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

from .norms import Norm, euclidean_norm


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
            rhs_norm (float): Norm of the right-hand side ``||b||``.

        Returns:
            float: Convergence threshold value.
        """

    @abstractmethod
    def has_converged(self, residual: torch.Tensor, rhs_norm: float) -> bool:
        """Determine convergence given residual and RHS norm.

        Uses the injected norm function - criterion doesn't know which norm.

        Args:
            residual (torch.Tensor): Current residual vector r_k.
            rhs_norm (float): Norm of the right-hand side ``||b||``.

        Returns:
            bool: True if converged, False otherwise.
        """

    def has_converged_from_norm(self, residual_norm: float, rhs_norm: float) -> bool:
        """Same check as ``has_converged``, given an already-computed residual norm.

        Skips recomputing ``self.norm(residual)`` — and, for a CUDA residual,
        the blocking device sync that recompute would force — when the
        caller already has the value from elsewhere (e.g. a solver's
        per-iteration state). Callers must only pass a ``residual_norm``
        that was computed with this criterion's own ``norm`` function;
        otherwise the comparison is against the wrong quantity.

        Args:
            residual_norm (float): ``self.norm(residual)``, already computed.
            rhs_norm (float): Norm of the right-hand side ``||b||``.

        Returns:
            bool: True if ``residual_norm <= threshold(rhs_norm)``, False
                otherwise (including when either input is non-finite).
        """
        if not math.isfinite(residual_norm) or not math.isfinite(rhs_norm):
            return False
        return bool(residual_norm <= self.threshold(rhs_norm))


@dataclass(frozen=True, slots=True)
class CombinedToleranceCriterion(IConvergenceCriterion):
    """SciPy-style criterion: ``||r||_norm <= max(rtol * ||b||, atol)``.

    The norm function is injected at construction time via dependency
    injection. Default: ``euclidean_norm`` (L2). Can be replaced with
    ``energy_norm(A)`` for the A-norm.

    Attributes:
        rtol (float): Relative tolerance (multiplied by ``||b||``).
        atol (float): Absolute tolerance (lower bound).
        norm (Norm): Norm function to use (default: ``euclidean_norm``).

    Example:
        >>> # L2 norm convergence (default)
        >>> criterion = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14)
        >>>
        >>> # A-norm convergence for SPD matrix A
        >>> from torchalg.strategies.norms import energy_norm
        >>> criterion_a = CombinedToleranceCriterion(rtol=1e-6, atol=1e-14, norm=energy_norm(A))
    """

    rtol: float
    atol: float
    norm: Norm = field(default=euclidean_norm)

    def threshold(self, rhs_norm: float) -> float:
        """Compute threshold: ``max(rtol * ||b||, atol)``.

        Args:
            rhs_norm (float): Norm of the right-hand side ``||b||``.

        Returns:
            float: Convergence threshold value.
        """
        return max(self.rtol * rhs_norm, self.atol)

    def has_converged(self, residual: torch.Tensor, rhs_norm: float) -> bool:
        """Check: ``||r||_norm <= threshold``.

        Uses injected norm - criterion is agnostic to norm type.

        Args:
            residual (torch.Tensor): Current residual vector r_k.
            rhs_norm (float): Norm of the right-hand side ``||b||``.

        Returns:
            bool: True if ``||r||_norm <= max(rtol * ||b||, atol)``, False
                otherwise. Returns False if residual norm or rhs_norm is
                non-finite (NaN/Inf).
        """
        return self.has_converged_from_norm(self.norm(residual), rhs_norm)
