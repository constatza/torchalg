"""Callable preconditioner wrapper for custom functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

import torch

from ..base import NonLinearPreconditioner, PreconditionerContext, StaticSetupPreconditioner


class CallablePreconditioner(NonLinearPreconditioner, StaticSetupPreconditioner):
    """Wrap an arbitrary function as a preconditioner.

    For custom/experimental preconditioners provided as plain functions.

    Stateless wrapper: holds a Python callable, not a tensor buffer, so it
    is a plain class - no ``nn.Module`` needed.

    Cost:
        Setup: none (``W = 0``, ``D = 0``, ``L = 0``, ``S = 0``).
        Apply: exactly the wrapped function's, ``W = W_f``, ``Q = Q_f``, ``D = D_f``,
        ``L = L_f`` and ``S = S_f``; the wrapper adds one Python call and no kernel.

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

    def setup(self, context: PreconditionerContext | None = None) -> Self:
        """No-op: the wrapped function is already usable after construction.

        Args:
            context (PreconditionerContext | None): Ignored.

        Returns:
            Self: This preconditioner, already ready.
        """
        return self

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
