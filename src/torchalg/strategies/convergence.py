"""Convergence criteria abstractions for iterative solvers.

Ported from ``neuralls.domain.solver.strategies.convergence``
with ``numpy`` operations translated to ``torch``
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


def _match_tensor(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    """Return ``value`` as a tensor of ``reference``'s dtype and device.

    A tensor already matching ``reference`` is returned as is: ``torch.as_tensor``
    would dispatch a no-op ``aten.to`` for it on every solver iteration.
    """
    if (
        isinstance(value, torch.Tensor)
        and value.dtype == reference.dtype
        and value.device == reference.device
    ):
        return value
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


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

    def converged_flag_from_norm(
        self, residual_norm: torch.Tensor | float, rhs_norm: torch.Tensor | float
    ) -> torch.Tensor:
        """Return the stopping test of ``has_converged_from_norm`` as a 0-d bool tensor.

        Lets a solver read the flag together with other device-side flags in a
        single host transfer instead of one synchronization per flag.

        Cost:
            Device-side only, no host synchronization and, for two matching
            tensors, no conversion launches: the threshold, two finiteness
            tests, a comparison and two ``and`` launches.

        Args:
            residual_norm (torch.Tensor | float): ``self.norm(residual)`` (0-d tensor or float).
            rhs_norm (torch.Tensor | float): ``||b||`` (0-d tensor or float); at least
                one of the two inputs must be a tensor.

        Returns:
            torch.Tensor: 0-d bool tensor, ``True`` iff both inputs are finite and
                ``residual_norm <= threshold(rhs_norm)``.
        """
        threshold = self.threshold(rhs_norm)
        # `torch.as_tensor(a_float)` with no dtype/device falls back to the
        # ambient default (float32, CPU) — this repo never sets a global
        # default dtype, so converting the non-tensor operand blind would
        # silently downcast precision (or mismatch device) against the
        # other, genuinely-tensor operand. Match the real tensor's
        # dtype/device instead.
        reference = residual_norm if isinstance(residual_norm, torch.Tensor) else rhs_norm
        assert isinstance(reference, torch.Tensor)
        residual_norm_t = _match_tensor(residual_norm, reference)
        rhs_norm_t = _match_tensor(rhs_norm, reference)
        finite = residual_norm_t.isfinite() & rhs_norm_t.isfinite()
        return finite & (residual_norm_t <= threshold)

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

        Cost:
            One host synchronization and, for two matching tensors, no
            conversion launches (previously two no-op ``aten.to`` per call).

        Args:
            residual_norm (torch.Tensor | float): ``self.norm(residual)``,
                already computed (0-d tensor or Python float).
            rhs_norm (torch.Tensor | float): Norm of the right-hand side ``||b||``
                (0-d tensor or Python float).

        Returns:
            bool: True if ``residual_norm <= threshold(rhs_norm)``, False
                otherwise (including when either input is non-finite).
        """
        if isinstance(residual_norm, torch.Tensor) or isinstance(rhs_norm, torch.Tensor):
            return bool(self.converged_flag_from_norm(residual_norm, rhs_norm))  # the one sync
        # Both are Python floats: use native float arithmetic
        if not (math.isfinite(residual_norm) and math.isfinite(rhs_norm)):
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
