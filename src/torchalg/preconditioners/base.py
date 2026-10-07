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
    - Minimal interfaces: ``setup(matrix)`` then ``apply(residual)`` are the
      only two required methods, applied uniformly across every
      preconditioner - see ``Preconditioner``'s docstring for the two-phase
      contract.

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

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Self, runtime_checkable

import torch


class PreconditionerNotReadyError(RuntimeError):
    """Raised when ``apply()`` is called before ``setup()`` has run.

    Every ``Preconditioner`` follows an explicit two-phase contract:
    ``setup(matrix)`` builds whatever internal state ``apply()`` needs, and
    only then may ``apply()`` be called (possibly many times, e.g. once per
    solver iteration or across multiple right-hand sides against the same
    matrix). This error makes a caller that skips ``setup()`` fail fast
    with a clear message instead of hitting an attribute error deep inside
    a subclass's ``apply()`` implementation.
    """


@dataclass(frozen=True, slots=True)
class PreconditionerContext:
    """Immutable context for iteration-dependent preconditioners.

    Provides iteration state without coupling to solver internals. Enables
    adaptive strategies, scheduling, and diagnostics.

    Attributes:
        iteration (int): Current solver iteration number (0-indexed).
        residual_norm (torch.Tensor | float): ``||r_k||`` at the current
            iteration. A 0-d tensor or float; may be a tensor to avoid
            forced device syncs. Preconditioners reading the VALUE of this
            field (not just passing it through) will incur a sync cost at
            that point — that cost belongs to whichever preconditioner
            author actually needs the value, not to every solve.
        rhs_norm (torch.Tensor | float): ``||b||`` (constant throughout the
            solve; 0-d tensor or float). Same sync considerations as
            ``residual_norm``.
    """

    iteration: int
    residual_norm: torch.Tensor | float
    rhs_norm: torch.Tensor | float


class Preconditioner(ABC):
    """Base class for all preconditioners.

    Every preconditioner follows an explicit two-phase contract:
    ``setup(matrix)`` builds whatever internal state is needed (a
    factorization, a multigrid hierarchy, a loaded checkpoint, ...), then
    ``apply(residual) -> result`` applies it - callable many times after one
    ``setup()`` call (e.g. once per solver iteration, or across several
    right-hand sides against the same matrix). ``apply()`` raises
    ``PreconditionerNotReadyError`` if called before ``setup()``; every
    concrete subclass's ``apply`` is wrapped with this guard automatically
    (see ``__init_subclass__`` below), so no subclass needs to implement the
    check itself.

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
        >>> precond.setup(matrix)
        >>> z = precond.apply(torch.ones(3, dtype=torch.float64))
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Wrap a freshly-defined ``apply`` with the ``setup()``-readiness guard.

        Runs once per subclass, at class-definition time, only when that
        subclass itself defines ``apply`` (``"apply" in cls.__dict__``) -
        an intermediate subclass that merely inherits ``apply`` from a
        parent (whose own ``apply`` was already wrapped when that parent
        was defined) is left alone, so ``apply`` is wrapped exactly once
        per concrete implementation.

        Args:
            **kwargs (object): Forwarded to ``ABC.__init_subclass__``.
        """
        super().__init_subclass__(**kwargs)
        if "apply" not in cls.__dict__:
            return
        unguarded_apply = cls.__dict__["apply"]

        @functools.wraps(unguarded_apply)
        def guarded_apply(self: Preconditioner, *args: object, **inner_kwargs: object) -> object:
            if not getattr(self, "_is_ready", False):
                raise PreconditionerNotReadyError(
                    f"{type(self).__name__}.setup() must be called before apply()."
                )
            return unguarded_apply(self, *args, **inner_kwargs)

        cls.apply = guarded_apply  # ty: ignore[invalid-assignment]

    def _mark_ready(self) -> None:
        """Record that ``setup()`` has completed, unblocking ``apply()``.

        Every ``setup()`` implementation calls this as its last step.
        """
        self._is_ready = True

    @abstractmethod
    def setup(
        self,
        matrix: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> Self:
        """Build whatever internal state ``apply()`` needs for ``matrix``.

        Must be called once before the first ``apply()`` call, and again
        (discarding any previously built state) to rebind this
        preconditioner to a new or changed matrix. After ``setup()``
        returns, ``apply()`` may be called any number of times.

        Returns ``self``, matching ``nn.Module.to()``/``.cuda()``'s
        chaining convention (also already followed by
        ``AMGPreconditioner._apply``) - e.g. ``precond.setup(A).apply(r)``.

        Args:
            matrix (torch.Tensor): System matrix A that this preconditioner
                approximates/operates on.
            context (PreconditionerContext | None): Optional iteration
                context; ignored by preconditioners whose setup does not
                depend on solver iteration state.

        Returns:
            Self: This preconditioner, now ready for ``apply()``.
        """
        ...

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

        Raises:
            PreconditionerNotReadyError: If ``setup()`` has not been called.
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
    Subclasses override ``_compute_operator(matrix)`` to build M; ``setup()``
    is what actually calls it, per the two-phase ``Preconditioner`` contract
    - ``__init__`` is config-only.

    Type parameter ``T`` defines the internal operator type (e.g.
    ``torch.Tensor`` for Jacobi).

    Example:
        >>> # User passes matrix - simple!
        >>> precond = JacobiPreconditioner(matrix)
        >>> precond.setup(matrix)
        >>> z = precond.apply(residual)  # z = D^{-1}r
    """

    def setup(
        self,
        matrix: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> Self:
        """Compute the internal operator ``M`` from ``matrix``.

        Args:
            matrix (torch.Tensor): System matrix A.
            context (PreconditionerContext | None): Ignored.

        Returns:
            Self: This preconditioner, now ready for ``apply()``.
        """
        self._operator: T = self._compute_operator(matrix)
        self._mark_ready()
        return self

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
