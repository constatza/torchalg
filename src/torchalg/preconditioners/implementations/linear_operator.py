"""Callable-operator preconditioner wrapper.

Ported from ``neuralls.domain.solver.preconditioners.linear_operator``,
which wraps a ``scipy.sparse.linalg.LinearOperator``. ``torchalg`` has no
``LinearOperator``-equivalent type and no sparse dependency at all (see
``docs/plan.md``'s dense-only directive): here, a plain
``Callable[[torch.Tensor], torch.Tensor]`` *is* the linear-operator
abstraction, so this wraps that callable directly instead of a
scipy-specific wrapper type.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..base import Preconditioner, PreconditionerContext


def _ensure_tensor(
    value: torch.Tensor | None, template: torch.Tensor, *, name: str
) -> torch.Tensor:
    """Validate and reshape a preconditioner's raw output to match the input shape.

    Ensures the wrapped operator returned a usable tensor. Handles ``None``
    returns and automatic reshaping for flattened or incorrectly shaped
    outputs.

    Args:
        value (torch.Tensor | None): Operator output (may be ``None`` or
            misshapen).
        template (torch.Tensor): Reference tensor defining the expected
            shape.
        name (str): Preconditioner name for error messages.

    Returns:
        torch.Tensor: Validated tensor with shape matching ``template``.

    Raises:
        ValueError: If ``value`` is ``None`` or cannot be reshaped to match
            ``template``.
    """
    if value is None:
        raise ValueError(f"{name} returned None; expected torch.Tensor")
    if value.shape != template.shape:
        value = value.reshape(template.shape)
    return value


class LinearOperatorPreconditioner(Preconditioner):
    """Wrap an arbitrary linear-operator callable so it can be used by the solvers.

    Provides an adapter for any ``M^{-1}``-applying callable that isn't
    itself a ``Preconditioner`` - e.g. a matvec closure built elsewhere.

    Stateless wrapper: holds only the callable, not a tensor buffer, so it
    is a plain class - no ``nn.Module`` needed.

    Args:
        operator (Callable[[torch.Tensor], torch.Tensor]): Callable applying
            ``M^{-1}`` to a residual vector.

    Example:
        >>> import torch
        >>> diag = torch.tensor([2.0, 4.0, 1.0])
        >>> precond = LinearOperatorPreconditioner(lambda r: r / diag)
        >>> z = precond.apply(torch.tensor([2.0, 4.0, 1.0]))
    """

    def __init__(self, operator: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Initialize from a linear-operator callable.

        Args:
            operator (Callable[[torch.Tensor], torch.Tensor]): Callable
                applying ``M^{-1}`` to a residual vector.
        """
        self.operator = operator

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Apply the wrapped operator to the residual.

        Args:
            residual (torch.Tensor): Residual vector r.
            context (PreconditionerContext | None): Ignored (the wrapped
                operator doesn't need context).

        Returns:
            torch.Tensor: Preconditioned residual z = M^{-1}r.
        """
        return _ensure_tensor(
            self.operator(residual),
            residual,
            name="linear_operator_preconditioner",
        )
