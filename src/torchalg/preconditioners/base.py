"""Preconditioner interface and base abstractions.

Ported from ``neuralls.domain.solver.preconditioners.base`` (see
``docs/plan.md``) with ``NDArray`` translated to ``torch.Tensor``.

``Preconditioner`` stays a plain ``ABC`` here — just the interface. Concrete
subclasses that actually own precomputed tensor state (``JacobiPreconditioner``
onward: ILU/IC0/ICholesky/AMG/POD in later stages) additionally inherit
``torch.nn.Module`` and use ``register_buffer`` for that state, purely to
reuse ``nn.Module``'s ``.to(device/dtype)`` propagation — never
``register_parameter``, since none of this is learnable/autograd-tracked. See
``docs/plan.md``'s "``nn.Module`` usage" architecture decision for the full
rationale. This module itself has no ``nn.Module`` dependency: the interface
is framework-agnostic by design.

Design Principles:
    - ABC-based: Clear inheritance hierarchy with fail-fast validation.
    - Clean categories: Linear vs Non-linear vs Contextual.
    - Minimal interfaces: Single ``apply()`` method is the only requirement.

Mathematical Background:
    Preconditioners accelerate iterative solvers by transforming the linear
    system. At each iteration k, we compute::

        z_k = M^{-1}(r_k)  or  z_k = f(r_k)

    where:
    - r_k is the current residual.
    - z_k is the preconditioned residual.
    - M is an approximation to A (for linear preconditioners).
    - f is a learned/adaptive function (for non-linear preconditioners).

    Desirable properties:
    1. M ~= A or f approximates A^{-1} (fast convergence).
    2. Application is computationally cheap.
    3. For CG: M should be SPD (relaxed in Flexible CG).

    Flexible CG allows M_k to vary with iteration and be non-symmetric,
    enabling neural preconditioners and adaptive strategies.

References:
    - Notay, Y. (2000). Flexible Conjugate Gradients. SIAM J. Sci. Comput.
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems. Ch. 9.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass(frozen=True, slots=True)
class PreconditionerContext:
    """Immutable context for iteration-dependent preconditioners.

    Provides iteration state without coupling to solver internals. Enables
    adaptive strategies, scheduling, and diagnostics.

    Attributes:
        iteration (int): Current solver iteration number (0-indexed).
        residual_norm (float): ``||r_k||`` at the current iteration.
        rhs_norm (float): ``||b||`` (constant throughout the solve).
    """

    iteration: int
    residual_norm: float
    rhs_norm: float


class Preconditioner(ABC):
    """Base class for all preconditioners.

    All preconditioners must implement ``apply(residual) -> result``. This is
    the ONLY required interface.

    Mathematical interpretation::

        z = M^{-1}(r)  [linear preconditioner]
        z = f_theta(r) [non-linear preconditioner, e.g. neural network]

    where the goal is to accelerate convergence of the iterative solver.

    Example:
        >>> import torch
        >>> from torchalg.preconditioners.implementations import (
        ...     JacobiPreconditioner,
        ... )
        >>> matrix = torch.eye(3, dtype=torch.float64) * 2.0
        >>> precond = JacobiPreconditioner(matrix)
        >>> z = precond.apply(torch.ones(3, dtype=torch.float64))
    """

    @abstractmethod
    def apply(
        self,
        residual: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> torch.Tensor:
        """Apply preconditioner to residual vector.

        Computes ``z = M^{-1}(r)`` for linear preconditioners or ``z = f(r)``
        for non-linear preconditioners. The output must match the input
        shape.

        Args:
            residual (torch.Tensor): Residual vector r, shape ``(n,)``.
            context (PreconditionerContext | None): Optional iteration
                context for contextual preconditioners.

        Returns:
            torch.Tensor: Preconditioned residual z, shape matching
                ``residual``.
        """
        ...

    @property
    def requires_flexible_cg(self) -> bool:
        """Whether this preconditioner requires the Flexible CG variant.

        Linear static preconditioners (Jacobi, ILU, IC0) return ``False`` and
        work with standard PCG. Non-linear and contextual preconditioners
        return ``True`` and require Flexible CG, which allows M_k to vary
        per iteration.

        Returns:
            bool: ``False`` for linear preconditioners; subclasses override
                to ``True``.
        """
        return False

    def __call__(self, residual: torch.Tensor) -> torch.Tensor:
        """Make preconditioner callable: ``precond(r)`` aliases ``precond.apply(r)``.

        This convenience method allows preconditioners to be used with code
        expecting a callable interface (functions, wrapped operators, etc.).

        Args:
            residual (torch.Tensor): Residual vector r, shape ``(n,)``.

        Returns:
            torch.Tensor: Preconditioned residual z.

        Note:
            The context parameter is not available via callable syntax. Use
            ``.apply()`` directly if you need to pass context.
        """
        return self.apply(residual)


@runtime_checkable
class BindableInputs(Protocol):
    """Protocol for preconditioners that accept named extra inputs beyond the residual.

    Structural typing (``Protocol``, not ``ABC``) is deliberate: only the
    preconditioners that actually need extra bound inputs (e.g. a neural
    preconditioner needing auxiliary fields) declare this shape. The base
    ``Preconditioner`` does not, so ``isinstance(p, BindableInputs)`` cleanly
    distinguishes "needs extra inputs" from "doesn't", without forcing every
    preconditioner to carry an unused ``bind_inputs``/``extra_input_names``
    pair through inheritance. Use ``isinstance(p, BindableInputs)`` before
    calling ``bind_inputs()`` or reading ``extra_input_names``.
    """

    @property
    def extra_input_names(self) -> tuple[str, ...]:
        """Names of extra tensors this preconditioner expects beyond the residual."""
        ...

    def bind_inputs(self, **inputs: torch.Tensor) -> None:
        """Pre-bind named extra tensors before the CG loop starts."""
        ...


class LinearPreconditioner[T](Preconditioner):
    """Base for matrix-based linear preconditioners.

    These compute ``M^{-1}r`` where M is derived from system matrix A.
    Subclasses override ``_compute_operator(matrix)`` to build M.

    The key design principle: the user passes a matrix, and the
    preconditioner computes what it needs internally.

    Type parameter ``T`` defines the internal operator type (e.g.
    ``torch.Tensor`` for Jacobi).

    Example:
        >>> # User passes matrix - simple!
        >>> precond = JacobiPreconditioner(matrix)
        >>> z = precond.apply(residual)  # z = D^{-1}r
    """

    def __init__(self, matrix: torch.Tensor) -> None:
        """Initialize from system matrix.

        Args:
            matrix (torch.Tensor): System matrix A (will compute M from this).
        """
        self._operator: T = self._compute_operator(matrix)

    @abstractmethod
    def _compute_operator(self, matrix: torch.Tensor) -> T:
        """Compute internal preconditioner operator from matrix.

        Subclasses implement this to extract the diagonal, compute an ILU
        factorization, etc.

        Args:
            matrix (torch.Tensor): System matrix A.

        Returns:
            T: Internal representation for fast application.
        """
        ...


class NonLinearPreconditioner(Preconditioner):
    """Base for non-linear preconditioners.

    These compute ``z = f(r)`` where f is not necessarily linear. Examples:
    neural networks, adaptive strategies, custom functions.

    Subclasses just implement ``apply()`` - no matrix needed.
    """

    @property
    def requires_flexible_cg(self) -> bool:
        """Non-linear preconditioners require Flexible CG.

        Returns:
            bool: ``True`` - non-linear preconditioners are not
                SPD-preserving.
        """
        return True
