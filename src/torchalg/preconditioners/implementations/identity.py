"""Identity preconditioner (no preconditioning)."""

from __future__ import annotations

from typing import Self

import torch

from ..base import Preconditioner, PreconditionerContext


class Identity(Preconditioner):
    """Identity preconditioner: z = r (no preconditioning).

    Returns the residual unchanged, equivalent to M = I. Used for standard
    unpreconditioned CG where the system matrix A is already
    well-conditioned, or for baseline comparisons.

    Stateless - holds no tensor buffers, so it is a plain class (no
    ``nn.Module``); there is nothing for ``.to(device/dtype)`` to move.

    Mathematical Properties:
        - M = I (identity matrix).
        - z = M^{-1}r = Ir = r.
        - Convergence rate: O(sqrt(kappa(A))) where
          kappa(A) = lambda_max(A) / lambda_min(A).
        - No computational overhead.

    Stateless, so it is always ready - construction alone satisfies the
    ``setup()``-before-``apply()`` contract (``setup()`` is still provided,
    as a no-op, so passing an explicit matrix through the uniform
    ``Preconditioner`` interface works too).

    Example:
        >>> import torch
        >>> precond = Identity()
        >>> r = torch.tensor([1.0, 2.0, 3.0])
        >>> z = precond.apply(r)
        >>> torch.equal(z, r)
        True
    """

    def __init__(self) -> None:
        """Mark ready immediately; identity has no matrix-dependent state."""
        self._mark_ready()

    def setup(
        self,
        matrix: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> Self:
        """No-op: identity has no matrix-dependent state to build.

        Args:
            matrix (torch.Tensor): Ignored.
            context (PreconditionerContext | None): Ignored.

        Returns:
            Self: This preconditioner, already ready.
        """
        self._mark_ready()
        return self

    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Return residual unchanged (identity preconditioning).

        Args:
            residual (torch.Tensor): Residual vector r.
            context (PreconditionerContext | None): Ignored (identity
                doesn't need context).

        Returns:
            torch.Tensor: Clone of the input residual vector (z = r).
        """
        return residual.clone()
