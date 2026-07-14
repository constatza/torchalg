"""Callable preconditioner wrapper for custom functions."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..base import NonLinearPreconditioner, PreconditionerContext


class CallablePreconditioner(NonLinearPreconditioner):
    """Wrap an arbitrary function as a preconditioner.

    For custom/experimental preconditioners provided as plain functions.

    Stateless wrapper: holds a Python callable, not a tensor buffer, so it
    is a plain class - no ``nn.Module`` needed.

    Args:
        func (Callable[[torch.Tensor], torch.Tensor]): Function taking a
            residual and returning the preconditioned residual.

    Example:
        >>> import torch
        >>> def my_precond(r: torch.Tensor) -> torch.Tensor:
        ...     return r * 0.5  # Simple damping
        >>> precond = CallablePreconditioner(my_precond)
        >>> z = precond.apply(torch.tensor([2.0, 4.0]))
    """

    def __init__(self, func: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Initialize from function.

        Args:
            func (Callable[[torch.Tensor], torch.Tensor]): Function taking a
                residual and returning the preconditioned residual.
        """
        self._func = func

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Apply the wrapped function to the residual.

        Args:
            residual (torch.Tensor): Current residual vector r_k.
            context (PreconditionerContext | None): Ignored (callable
                preconditioners don't use context).

        Returns:
            torch.Tensor: Preconditioned residual z_k = f(r_k).
        """
        return self._func(residual)
