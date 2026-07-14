"""Callable preconditioner wrapper for custom functions."""

from __future__ import annotations

from collections.abc import Callable

from numpy.typing import NDArray

from .base import NonLinearPreconditioner, PreconditionerContext


class CallablePreconditioner(NonLinearPreconditioner):
    """Wrap arbitrary function as preconditioner.

    For custom/experimental preconditioners provided as functions.
    Replaces FunctionPreconditioner from previous design.

    Args:
        func: Function taking residual → preconditioned residual

    Example:
        >>> def my_precond(r):
        ...     return r * 0.5  # Simple damping
        >>> precond = CallablePreconditioner(my_precond)
        >>> z = precond.apply(r)
    """

    def __init__(self, func: Callable[[NDArray], NDArray]):
        """Initialize from function.

        Args:
            func: Function taking residual → preconditioned residual
        """
        self._func = func

    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        """Apply function to residual.

        Args:
            residual: Current residual vector r_k
            context: Ignored (callable preconditioners don't use context)

        Returns:
            Preconditioned residual z_k = f(r_k)
        """
        return self._func(residual)
