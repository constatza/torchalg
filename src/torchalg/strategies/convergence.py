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
    def threshold(self, rhs_norm: torch.Tensor | float) -> torch.Tensor | float:
        """Return the stopping threshold for a given RHS norm.

        When rhs_norm is a tensor (e.g. from solver state), may return a
        tensor to avoid forced device syncs. When rhs_norm is a Python float,
        returns a float.

        Args:
            rhs_norm (torch.Tensor | float): Norm of the right-hand side ``||b||``.

        Returns:
            torch.Tensor | float: Convergence threshold value (same type as input).
        """

    @abstractmethod
    def has_converged(self, residual: torch.Tensor, rhs_norm: torch.Tensor | float) -> bool:
        """Determine convergence given residual and RHS norm.

        Uses the injected norm function - criterion doesn't know which norm.

        Args:
            residual (torch.Tensor): Current residual vector r_k.
            rhs_norm (torch.Tensor | float): Norm of the right-hand side ``||b||``
                (0-d tensor or Python float).

        Returns:
            bool: True if converged, False otherwise.
        """

    def has_converged_from_norm(
        self, residual_norm: torch.Tensor | float, rhs_norm: torch.Tensor | float
    ) -> bool:
        """Same check as ``has_converged``, given an already-computed residual norm.

        Skips recomputing ``self.norm(residual)`` — and, for a CUDA residual,
        the blocking device sync that recompute would force — when the
        caller already has the value from elsewhere (e.g. a solver's
        per-iteration state). Callers must only pass a ``residual_norm``
        that was computed with this criterion's own ``norm`` function;
        otherwise the comparison is against the wrong quantity.

        When both inputs are tensors (e.g., from a solver's tensor-valued state),
        this method avoids redundant device syncs by performing a single combined
        tensor comparison and converting to bool only once at the very end —
        the irreducible cost of bridging tensor-land to Python's boolean value
        required by the calling loop's stopping condition.

        Args:
            residual_norm (torch.Tensor | float): ``self.norm(residual)``,
                already computed (0-d tensor or Python float).
            rhs_norm (torch.Tensor | float): Norm of the right-hand side ``||b||``
                (0-d tensor or Python float).

        Returns:
            bool: True if ``residual_norm <= threshold(rhs_norm)``, False
                otherwise (including when either input is non-finite).
        """
        threshold = self.threshold(rhs_norm)
        # If either input is a tensor, use tensor arithmetic for combined expression
        if isinstance(residual_norm, torch.Tensor) or isinstance(rhs_norm, torch.Tensor):
            finite = (
                torch.as_tensor(residual_norm).isfinite() & torch.as_tensor(rhs_norm).isfinite()
            )
            converged = finite & (torch.as_tensor(residual_norm) <= threshold)
            return bool(converged)  # The one irreducible sync to Python bool
        # Both are Python floats: use native float arithmetic
        if not (math.isfinite(residual_norm) and math.isfinite(rhs_norm)):
            return False
        return bool(residual_norm <= threshold)


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

    def threshold(self, rhs_norm: torch.Tensor | float) -> torch.Tensor | float:
        """Compute threshold: ``max(rtol * ||b||, atol)`` or tensor equivalent.

        Handles both float and tensor inputs without forced syncs. When
        rhs_norm is a tensor, uses torch.clamp_min for a single vectorized
        operation instead of Python's max() with implicit comparisons.

        Args:
            rhs_norm (torch.Tensor | float): Norm of the right-hand side ``||b||``.

        Returns:
            torch.Tensor | float: Convergence threshold value (tensor if input
                is tensor, float if input is float).
        """
        if isinstance(rhs_norm, torch.Tensor):
            return torch.clamp_min(self.rtol * rhs_norm, self.atol)
        return max(self.rtol * rhs_norm, self.atol)

    def has_converged(self, residual: torch.Tensor, rhs_norm: torch.Tensor | float) -> bool:
        """Check: ``||r||_norm <= threshold``.

        Uses injected norm - criterion is agnostic to norm type.

        Args:
            residual (torch.Tensor): Current residual vector r_k.
            rhs_norm (torch.Tensor | float): Norm of the right-hand side ``||b||``
                (0-d tensor or Python float).

        Returns:
            bool: True if ``||r||_norm <= max(rtol * ||b||, atol)``, False
                otherwise. Returns False if residual norm or rhs_norm is
                non-finite (NaN/Inf).
        """
        return self.has_converged_from_norm(self.norm(residual), rhs_norm)
