"""Identity preconditioner (no preconditioning)."""

from __future__ import annotations

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

    Example:
        >>> import torch
        >>> precond = Identity()
        >>> r = torch.tensor([1.0, 2.0, 3.0])
        >>> z = precond.apply(r)
        >>> torch.equal(z, r)
        True
    """

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
