"""Base preconditioner abstractions.

This module defines abstract base classes for preconditioners.
All concrete implementations are in implementations.py.

Design Principles:
    - ABC-based: Clear inheritance hierarchy with fail-fast validation
    - Clean categories: Linear vs Non-linear vs Contextual
    - Minimal interfaces: Single apply() method is the only requirement

Mathematical Background:
    Preconditioners accelerate iterative solvers by transforming the linear system.
    At each iteration k, we compute:

        z_k = M^{-1}(r_k)  or  z_k = f(r_k)

    where:
    - r_k is the current residual
    - z_k is the preconditioned residual
    - M is an approximation to A (for linear preconditioners)
    - f is a learned/adaptive function (for non-linear preconditioners)

    Desirable properties:
    1. M ≈ A or f approximates A^{-1} (fast convergence)
    2. Application is computationally cheap
    3. For CG: M should be SPD (relaxed in Flexible CG)

    Flexible CG allows M_k to vary with iteration and be non-symmetric,
    enabling neural preconditioners and adaptive strategies.

References:
    - Notay, Y. (2000). Flexible Conjugate Gradients. SIAM J. Sci. Comput.
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. Ch. 9.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from numpy.typing import NDArray

if TYPE_CHECKING:
    pass  # NDArray moved to top-level for use in generics


@dataclass(frozen=True)
class PreconditionerContext:
    """Immutable context for iteration-dependent preconditioners.

    Provides iteration state without coupling to solver internals.
    Enables adaptive strategies, scheduling, and diagnostics.

    Attributes:
        iteration: Current solver iteration number (0-indexed)
        residual_norm: ||r_k|| at current iteration
        rhs_norm: ||b|| (constant throughout solve)
    """

    iteration: int
    residual_norm: float
    rhs_norm: float


class Preconditioner(ABC):
    """Base class for all preconditioners.

    All preconditioners must implement apply(residual) -> result.
    This is the ONLY required interface.

    Mathematical interpretation:
        z = M^{-1}(r)  [linear preconditioner]
        z = f_θ(r)     [non-linear preconditioner, e.g., neural network]

    where the goal is to accelerate convergence of the iterative solver.

    Examples:
        >>> # Direct instantiation - simple!
        >>> precond = JacobiPreconditioner(matrix)
        >>> z = precond.apply(residual)
        >>>
        >>> # Neural preconditioner (GPU cleanup automatic on GC)
        >>> precond = NeuralPreconditioner(checkpoint_path)
        >>> z = precond.apply(residual)
    """

    @abstractmethod
    def apply(self, residual: NDArray, context: PreconditionerContext | None = None) -> NDArray:
        """Apply preconditioner to residual vector.

        Computes z = M^{-1}(r) for linear preconditioners or z = f(r) for
        non-linear preconditioners. The output must match the input shape.

        Args:
            residual: Residual vector r ∈ R^n
            context: Optional iteration context for contextual preconditioners

        Returns:
            Preconditioned residual z ∈ R^n with shape matching input

        Examples:
            >>> # Identity preconditioner
            >>> z = precond.apply(r)  # z = r
            >>> assert z.shape == r.shape
            >>>
            >>> # Jacobi preconditioner
            >>> z = precond.apply(r)  # z = D^{-1}r
            >>>
            >>> # Neural preconditioner
            >>> z = precond.apply(r)  # z = network(r)
        """
        ...

    @property
    def requires_flexible_cg(self) -> bool:
        """Whether this preconditioner requires the Flexible CG variant.

        Linear static preconditioners (Jacobi, ILU, IC0) return False and
        work with standard PCG.  Non-linear and contextual preconditioners
        return True and require Flexible CG, which allows M_k to vary per
        iteration.

        Returns:
            False for linear preconditioners; subclasses override to True.
        """
        return False

    def __call__(self, residual: NDArray) -> NDArray:
        """Make preconditioner callable: precond(r) is alias for precond.apply(r).

        This convenience method allows preconditioners to be used with code
        expecting callable interface (functions, LinearOperators, etc.).

        Args:
            residual: Residual vector r ∈ R^n

        Returns:
            Preconditioned residual z ∈ R^n

        Note:
            Context parameter is not available when using callable syntax.
            Use .apply() directly if you need to pass context.

        Examples:
            >>> # Both syntaxes work
            >>> z1 = precond.apply(r)
            >>> z2 = precond(r)
            >>> assert np.allclose(z1, z2)
        """
        return self.apply(residual)


@runtime_checkable
class BindableInputs(Protocol):
    """Protocol for preconditioners that accept named extra inputs beyond the residual.

    Only preconditioners that declare this need it — base Preconditioner does not.
    Use isinstance(p, BindableInputs) before calling bind_inputs() or reading extra_input_names.
    """

    @property
    def extra_input_names(self) -> tuple[str, ...]:
        """Names of extra arrays this preconditioner expects beyond the residual."""
        ...

    def bind_inputs(self, **inputs: NDArray) -> None:
        """Pre-bind named extra arrays before the CG loop starts."""
        ...


class LinearPreconditioner[T_Operator](Preconditioner):
    """Base for matrix-based linear preconditioners.

    These compute M^{-1}r where M is derived from system matrix A.
    Subclasses override _compute_operator(matrix) to build M.

    The key design principle: User passes matrix, preconditioner computes
    what it needs internally.

    Type parameter T_Operator defines the internal operator type
    (e.g., NDArray for Jacobi, SuperLU for ILU).

    Example:
        >>> # User passes matrix - simple!
        >>> precond = JacobiPreconditioner(matrix)
        >>>
        >>> # Preconditioner extracts diagonal and inverts it internally
        >>> z = precond.apply(residual)  # z = D^{-1}r
    """

    def __init__(self, matrix: NDArray) -> None:
        """Initialize from system matrix.

        Args:
            matrix: System matrix A (will compute M from this)
        """
        self._operator: T_Operator = self._compute_operator(matrix)

    @abstractmethod
    def _compute_operator(self, matrix: NDArray) -> T_Operator:
        """Compute internal preconditioner operator from matrix.

        Subclasses implement this to extract diag, compute ILU, etc.

        Args:
            matrix: System matrix A

        Returns:
            Internal representation for fast application
        """
        ...


class NonLinearPreconditioner(Preconditioner):
    """Base for non-linear preconditioners.

    These compute z = f(r) where f is not necessarily linear.
    Examples: neural networks, adaptive strategies, custom functions.

    Subclasses just implement apply() - no matrix needed.
    """

    @property
    def requires_flexible_cg(self) -> bool:
        """Non-linear preconditioners require Flexible CG.

        Returns:
            True — non-linear preconditioners are not SPD-preserving.
        """
        return True
