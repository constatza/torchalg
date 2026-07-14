"""Identity preconditioner (no preconditioning)."""

from __future__ import annotations

from numpy.typing import NDArray

from ..base import Preconditioner, PreconditionerContext


class Identity(Preconditioner):
    """Identity preconditioner: z = r (no preconditioning).

    Returns the residual unchanged, equivalent to M = I. Used for standard
    unpreconditioned CG where the system matrix A is already well-conditioned
    or for baseline comparisons.

    Mathematical Properties:
        - M = I (identity matrix)
        - z = M^{-1}r = Ir = r
        - Convergence rate: O(√κ(A)) where κ(A) = λ_max(A) / λ_min(A)
        - No computational overhead

    Example:
        >>> precond = Identity()
        >>> r = np.array([1.0, 2.0, 3.0])
        >>> z = precond.apply(r)
        >>> assert np.array_equal(z, r)  # z = r for identity
    """

    def __init__(self):
        """No matrix needed - identity works for any system."""
        pass

    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        """Return residual unchanged (identity preconditioning).

        Args:
            residual: Residual vector r
            context: Ignored (identity doesn't need context)

        Returns:
            Copy of input residual vector (z = r)
        """
        return residual.copy()
