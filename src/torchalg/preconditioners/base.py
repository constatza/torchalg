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
    - Minimal interfaces: ``apply(residual)`` is the one method every
      preconditioner shares; ``setup()`` is split by role (see
      ``MatrixSetupPreconditioner``/``StaticSetupPreconditioner`` below)
      since not every preconditioner's setup needs the system matrix.

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
from enum import Enum, auto
from typing import Protocol, Self, runtime_checkable

import torch


class Placement(Enum):
    """What bounds one application of a preconditioner, hence which device suits it.

    The cost audit (``docs/implementations/cost-model.md``) shows that the three
    classes behave differently on an accelerator, so the solver chooses the device
    from this closed set rather than moving everything to CUDA:

    - ``SEQUENTIAL_SWEEP``: a chain of dependent steps (triangular solves, level-scheduled
      Gauss-Seidel sweeps). The work per dependency level is tiny, so the GPU pays its
      launch latency per level and measured 30 to 165 times slower than the CPU.
    - ``SPMV_BOUND``: sparse matrix-vector products and vector updates. Bandwidth bound
      with a launch-latency floor, so the GPU wins only for large systems.
    - ``DENSE_BOUND``: dense products and factor solves. Compute or bandwidth bound with
      high arithmetic intensity, so the GPU wins from moderate sizes.
    """

    SEQUENTIAL_SWEEP = auto()
    SPMV_BOUND = auto()
    DENSE_BOUND = auto()


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

    Every preconditioner follows an explicit two-phase contract: ``setup()``
    builds whatever internal state is needed (a factorization, a multigrid
    hierarchy, a loaded checkpoint, ...), then ``apply(residual) -> result``
    applies it - callable many times after one ``setup()`` call (e.g. once
    per solver iteration, or across several right-hand sides against the
    same matrix). ``apply()`` raises ``PreconditionerNotReadyError`` if
    called before ``setup()``; every concrete subclass's ``apply`` is
    wrapped with this guard automatically (see ``__init_subclass__`` below),
    so no subclass needs to implement the check itself.

    Readiness is derived, never stored: each concrete class defines
    ``is_ready`` from the state its ``setup()`` builds (a registered buffer
    that is ``None`` until built, a prepared hierarchy, the readiness of the
    preconditioners it wraps, ...), so there is no flag to forget to set or
    to leave stale. ``setup()`` assigns that state only after every step
    succeeded, so a failed ``setup()`` leaves the previous complete state, and
    with it the previous readiness, untouched. The exception is a composite
    that sets up wrapped preconditioners one after the other
    (``ScheduledPreconditioner``): they mutate in place, so a failure after the
    first leaves that one on the new matrix.

    ``setup()`` itself is deliberately NOT declared here. Forcing every
    preconditioner to accept a ``matrix`` argument - even the ones whose
    setup never reads it (``NeuralPreconditioner``, ``CallablePreconditioner``,
    ``LinearOperatorPreconditioner``, ``Identity`` all used to carry a
    ``matrix: Ignored`` parameter for this reason alone) - is an Interface
    Segregation violation: those implementers depended on a parameter they
    never used. Pick the role interface that matches what your
    preconditioner's setup actually needs:

    - ``MatrixSetupPreconditioner``: setup genuinely depends on the system
      matrix (Jacobi, IC0, ILU, ICholesky, every AMG variant - anything
      that factorizes or builds a hierarchy from A's actual values).
    - ``StaticSetupPreconditioner``: setup never needs the matrix (a
      injected predictor or a wrapped callable that's already usable after
      construction, ...).

    This is a property of what a given concrete class's ``setup()`` body
    actually does, decided once per class - never inferred from a family
    label like "neural" vs "classical". A neural preconditioner whose setup
    builds something from the matrix (e.g. a GNN constructing message-passing
    edges from A's connectivity before it can run) is a
    ``MatrixSetupPreconditioner`` like Jacobi or AMG, not a
    ``StaticSetupPreconditioner`` - "data-driven" and "needs the matrix at
    setup" are independent axes.

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
        >>> precond = JacobiPreconditioner().setup(matrix)
        >>> z = precond.apply(torch.ones(3, dtype=torch.float64))
    """

    placement: Placement = Placement.SPMV_BOUND
    """What bounds one ``apply`` (see :class:`Placement`).

    Subclasses with sweeps override it on the class; a composite such as a multigrid
    preconditioner assigns it per instance from the smoother it is built around.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Wrap a freshly-defined ``apply`` with the ``is_ready`` guard.

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
            if not self.is_ready:
                raise PreconditionerNotReadyError(
                    f"{type(self).__name__}.setup() must be called before apply()."
                )
            return unguarded_apply(self, *args, **inner_kwargs)

        cls.apply = guarded_apply  # ty: ignore[invalid-assignment]

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Whether ``apply()`` may run, derived from the state ``setup()`` builds.

        A read-only view of that state, not a stored flag: it cannot disagree
        with what ``apply()`` actually consumes.

        Returns:
            bool: ``True`` once the state ``apply()`` consumes exists.
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


class MatrixSetupPreconditioner(Preconditioner, ABC):
    """Role interface for preconditioners whose setup depends on the matrix.

    Covers every preconditioner that factorizes, or builds a hierarchy
    from, the system matrix's actual values - Jacobi, IC0, ILU, ICholesky,
    every AMG/BootCMatch variant. Must be called again (discarding any
    previously built state) whenever the matrix changes.
    """

    @abstractmethod
    def setup(
        self,
        matrix: torch.Tensor,
        context: PreconditionerContext | None = None,
    ) -> Self:
        """Build whatever internal state ``apply()`` needs for ``matrix``.

        Must be called before the first ``apply()`` call, and again
        (discarding any previously built state) to rebind this
        preconditioner to a new or changed matrix. The one exception is a
        preconditioner that is ready by construction (a
        ``ScheduledPreconditioner`` over static components), whose
        ``is_ready`` is already ``True``. After ``setup()``
        returns, ``apply()`` may be called any number of times, including
        across several right-hand sides against the same matrix.

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


class StaticSetupPreconditioner(Preconditioner, ABC):
    """Role interface for preconditioners whose setup never needs the matrix.

    Covers anything whose setup is matrix-independent by construction - an
    injected predictor (``NeuralPreconditioner``), a wrapped callable that's
    already usable after ``__init__`` (``CallablePreconditioner``,
    ``LinearOperatorPreconditioner``), or a stateless no-op (``Identity``).
    Because the matrix never enters this contract at all, one ``setup()``
    call serves any number of systems, including systems of different
    dimensions or entirely different matrices - not just different
    right-hand sides against a fixed matrix.
    """

    @property
    def is_ready(self) -> bool:
        """Always ``True``: a static preconditioner is usable from construction.

        Returns:
            bool: ``True``.
        """
        return True

    @abstractmethod
    def setup(self, context: PreconditionerContext | None = None) -> Self:
        """Build whatever internal state ``apply()`` needs, independent of any matrix.

        A static preconditioner is ready from construction (``is_ready`` is
        ``True``), so ``apply()`` does not require this call; it builds
        whatever extra state the class defines. After ``setup()`` returns, ``apply()`` may be called any number of times,
        against any sequence of systems.

        Args:
            context (PreconditionerContext | None): Optional iteration
                context; ignored by preconditioners whose setup does not
                depend on solver iteration state.

        Returns:
            Self: This preconditioner, now ready for ``apply()``.
        """
        ...


class LinearPreconditioner[T](MatrixSetupPreconditioner):
    """Base for matrix-based linear preconditioners.

    These compute ``M^{-1}r`` where M is derived from system matrix A.
    Subclasses override ``_compute_operator(matrix)`` to build M; ``setup()``
    is what actually calls it, per the two-phase ``Preconditioner`` contract
    - ``__init__`` is config-only.

    Type parameter ``T`` defines the internal operator type (e.g.
    ``torch.Tensor`` for Jacobi). ``_operator`` is that operator, ``None``
    until ``setup()`` stores it; readiness is derived from it.

    Example:
        >>> # Config-only constructor; ``setup`` binds the matrix.
        >>> precond = JacobiPreconditioner().setup(matrix)
        >>> z = precond.apply(residual)  # z = D^{-1}r
    """

    _operator: T

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
        self._operator = self._compute_operator(matrix)
        return self

    @property
    def is_ready(self) -> bool:
        """Whether ``_operator`` has been built.

        Concrete subclasses initialise ``_operator`` to ``None`` (``nn.Module``
        ones via ``register_buffer("_operator", None)``) so ``setup()`` replaces
        it in one assignment.

        Returns:
            bool: ``True`` once ``setup()`` has stored the operator.
        """
        return self._operator is not None

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
    neural networks, adaptive strategies, custom functions. This base only
    fixes ``requires_flexible_cg = True``; it says nothing about whether
    setup needs the matrix - concrete subclasses additionally extend
    ``MatrixSetupPreconditioner`` or ``StaticSetupPreconditioner`` for that
    (most wrapper preconditioners here are matrix-agnostic, but
    ``ScheduledPreconditioner`` composes sub-preconditioners that may need
    the matrix, so it is not universal).
    """

    @property
    def requires_flexible_cg(self) -> bool:
        """Non-linear preconditioners require Flexible CG.

        Returns:
            bool: ``True`` - non-linear preconditioners are not
                SPD-preserving.
        """
        return True


class Identity(StaticSetupPreconditioner):
    """Identity preconditioner: z = r (no preconditioning).

    Returns the residual unchanged, equivalent to M = I. Used for standard
    unpreconditioned CG where the system matrix A is already
    well-conditioned, or for baseline comparisons. It lives with the
    abstractions, not with the concrete implementations, because it is the
    null object a solver falls back to when constructed without a
    preconditioner, and solvers may depend on this module only.

    Stateless - holds no tensor buffers, so it is a plain class (no
    ``nn.Module``); there is nothing for ``.to(device/dtype)`` to move.

    Mathematical Properties:
        - M = I (identity matrix).
        - z = M^{-1}r = Ir = r.
        - Convergence rate: O(sqrt(kappa(A))) where
          kappa(A) = lambda_max(A) / lambda_min(A).
        - No computational overhead.

    Cost:
        Setup: none. Apply: ``W = 0``, ``Q = 0``, ``D = 0``, ``L = 0``,
        ``S = 0``; the input tensor itself is returned. No solver writes ``r``
        or ``z`` in place (the CG updates are out of place, and the first
        direction and an empty FCG window still copy), so a defensive clone
        would only add a ``2Vn`` memory pass per iteration: measured 6-15% of
        a plain-CG iteration (129 to 120 us at ``n = 10^4``, 214 to 184 us at
        ``n = 9 * 10^4``), with bit-identical iterates.
        Roofline and work-depth: nothing to bound; the cost of an
        unpreconditioned solve is the PCG iteration alone, which the
        iteration count multiplies.

    Stateless, so it is always ready - construction alone satisfies the
    ``setup()``-before-``apply()`` contract (``setup()`` is still provided,
    as a no-op, for callers that always call it uniformly).

    Example:
        >>> import torch
        >>> precond = Identity()
        >>> r = torch.tensor([1.0, 2.0, 3.0])
        >>> z = precond.apply(r)
        >>> torch.equal(z, r)
        True
    """

    def setup(self, context: PreconditionerContext | None = None) -> Self:
        """No-op: identity has no state to build at all.

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
        """Return residual unchanged (identity preconditioning).

        Args:
            residual (torch.Tensor): Residual vector r.
            context (PreconditionerContext | None): Ignored (identity
                doesn't need context).

        Returns:
            torch.Tensor: The input tensor itself (z = r), not a copy; a
                caller that writes ``z`` in place writes ``r``.
        """
        return residual
