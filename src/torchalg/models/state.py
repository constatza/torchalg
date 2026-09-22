"""Typed, immutable state classes for iterative solvers.

Ported from ``neuralls.domain.solver.models.state`` (see ``docs/plan.md``)
with ``NDArray`` fields translated to ``torch.Tensor`` and every dataclass
made ``slots=True`` (the reference only used plain ``frozen=True``) per this
repo's immutability convention. This module defines a hierarchy of
immutable state dataclasses that eliminate the need for dynamic attributes.
All state is properly typed and validated.

Hierarchy:
    ``SolverState`` (base) - Common solver state.
    -> ``KrylovState`` - Krylov-specific state (adds working vectors).
    -> -> ``CGState`` - CG-specific state (adds direction history).

Design Principles:
    - Single Responsibility: Each class represents one level of abstraction.
    - Immutable (``frozen=True``): Prevents accidental mutation.
    - Typed: All fields properly typed, no dynamic attributes.
    - Open/Closed: Easy to extend for new solver types.

Theory (Notay 2000):
    Krylov methods maintain working vectors (u, r, w, d, q) that are updated
    each iteration. All fields are explicit and typed, avoiding dynamic
    attribute assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .history import DirectionHistory, ResidualHistory


@dataclass(frozen=True, slots=True)
class SolverState:
    """Base solver state common to all iterative solvers.

    Contains diagnostic information and counters shared by all solver
    types. Immutable (``frozen=True``) to prevent accidental mutation.

    Attributes:
        iteration (int): Current iteration number (0-indexed).
        converged (bool): Whether convergence criterion has been satisfied.
        breakdown (bool): Whether numerical breakdown occurred (NaN/Inf).
            Terminal condition.
        divergence (bool): Whether divergence detected
            (``||r|| > threshold``).
        residual_norm (float): Current residual norm ``||r_k||_2``.
        rhs_norm (float): Right-hand side norm ``||b||_2`` (constant).

    Example:
        >>> state = SolverState(
        ...     iteration=0,
        ...     converged=False,
        ...     breakdown=False,
        ...     divergence=False,
        ...     residual_norm=1.0,
        ...     rhs_norm=1.0,
        ... )
        >>> state.converged
        False
    """

    iteration: int
    """Current iteration number (0-indexed)."""

    converged: bool
    """Whether convergence criterion satisfied: ||r|| <= max(rtol*||b||, atol)."""

    breakdown: bool
    """Whether numerical breakdown occurred (NaN/Inf). Terminal condition."""

    divergence: bool
    """Whether divergence detected. Triggers residual recomputation."""

    residual_norm: float
    """Current residual norm ||r_k||_2."""

    rhs_norm: float
    """Right-hand side norm ||b||_2 (constant throughout solve)."""

    def __post_init__(self) -> None:
        """Validate state invariants.

        Raises:
            ValueError: If state has inconsistent fields.
        """
        if self.iteration < 0:
            raise ValueError(f"iteration must be >= 0, got {self.iteration}")
        if self.residual_norm < 0:
            raise ValueError(f"residual_norm must be >= 0, got {self.residual_norm}")
        if self.rhs_norm < 0:
            raise ValueError(f"rhs_norm must be >= 0, got {self.rhs_norm}")


@dataclass(frozen=True, slots=True, kw_only=True)
class KrylovState(SolverState):
    """Krylov solver state with working vectors.

    Extends ``SolverState`` with the working vectors (u, r, w, d, q) that
    Krylov methods maintain. All vectors are properly typed, no dynamic
    attributes.

    Note:
        ``kw_only=True`` ensures all fields must be passed as keyword
        arguments, preventing positional argument confusion with parent
        class fields.

    Attributes:
        u (torch.Tensor): Current solution vector u_i (Notay 2000).
        r (torch.Tensor): Current residual vector r_i = b - A*u_i.
        w (torch.Tensor): Preconditioned residual w_i = M^{-1} r_i
            (Notay 2000).
        d (torch.Tensor): Search direction d_i (Notay 2000).
        q (torch.Tensor): Matrix-vector product q_i = A d_i.

    Theory (Notay 2000):
        Krylov methods iteratively update these vectors:
        1. Preconditioning: w_i = M^{-1} r_i
        2. Direction: d_i = f(w_i, history)
        3. Matrix-vector: q_i = A d_i
        4. Step length: alpha_i = (d_i, r_i) / (d_i, q_i)
        5. Solution: u_{i+1} = u_i + alpha_i d_i
        6. Residual: r_{i+1} = r_i - alpha_i q_i

    Example:
        >>> import torch
        >>> state = KrylovState(
        ...     iteration=0,
        ...     converged=False,
        ...     breakdown=False,
        ...     divergence=False,
        ...     residual_norm=1.0,
        ...     rhs_norm=1.0,
        ...     u=torch.zeros(10),
        ...     r=torch.ones(10),
        ...     w=torch.ones(10),
        ...     d=torch.ones(10),
        ...     q=torch.ones(10),
        ... )
        >>> state.u.shape
        torch.Size([10])
    """

    u: torch.Tensor
    """Current solution vector u_i (Notay 2000)."""

    r: torch.Tensor
    """Current residual vector r_i = b - A*u_i."""

    w: torch.Tensor
    """Preconditioned residual w_i = M^{-1} r_i (Notay 2000)."""

    d: torch.Tensor
    """Search direction d_i (Notay 2000)."""

    q: torch.Tensor
    """Matrix-vector product q_i = A d_i."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CGState(KrylovState):
    """Conjugate Gradient solver state with history tracking.

    Extends ``KrylovState`` with CG-specific history buffers for direction
    orthogonalization and residual tracking.

    Attributes:
        direction_history (DirectionHistory): Sliding window of search
            directions and products. Used by FCG for truncated Gram-Schmidt
            orthogonalization.
        residual_history (ResidualHistory): Residual norms (absolute and
            relative). Used for convergence monitoring and post-analysis.
        w_prev (torch.Tensor | None): Previous preconditioned residual
            w_{k-1} (for two-term recurrence). Used by
            ``TwoTermRecurrenceStrategy`` to compute the beta coefficient.
        r_prev (torch.Tensor | None): Previous residual r_{k-1} (for
            two-term recurrence). Used by ``TwoTermRecurrenceStrategy`` to
            compute the beta coefficient.
        rw_prev (float): Previous inner product (r_{k-1}, w_{k-1}) (for
            two-term recurrence). Used by ``TwoTermRecurrenceStrategy`` to
            compute the beta coefficient.
        ortho_breakdown_at (int | None): Iteration at which FCG's
            orthogonalization first reported breakdown (near-zero result,
            loss of A-conjugacy), or ``None`` if it never occurred so far.
            Purely diagnostic - carried through to
            ``TerminationDiagnostics.ortho_breakdown_at`` and never gates
            ``_check_stopping`` (Notay 2000 treats truncated-orthogonalization
            quality loss as a convergence-rate signal, not a hard breakdown
            requiring termination).
        energy_decrement (float | None): This step's exact
            ``alpha_k * rho_k``, the decrease in ``||e_k||_A^2`` (Golub &
            Meurant 1994). ``None`` for the initial state, before any step.
            See ``torchalg.models.protocols.HasEnergyDecrement``.

    Theory (Notay 2000):
        FCG maintains a window of previous search directions to
        orthogonalize the new direction against::

            d_i = w_i - sum_{k=i-m}^{i-1} [(w_i, q_k) / (d_k, q_k)] d_k

        The ``direction_history`` stores the last ``m`` (d_k, q_k) pairs.

        PCG two-term recurrence needs only the previous iteration values::

            beta_k = (r_k, w_k) / (r_{k-1}, w_{k-1})
            d_k = w_k + beta_k * d_{k-1}

    Example:
        >>> import torch
        >>> from torchalg.models.history import DirectionHistory, ResidualHistory
        >>> state = CGState(
        ...     iteration=0,
        ...     converged=False,
        ...     breakdown=False,
        ...     divergence=False,
        ...     residual_norm=1.0,
        ...     rhs_norm=1.0,
        ...     u=torch.zeros(10),
        ...     r=torch.ones(10),
        ...     w=torch.ones(10),
        ...     d=torch.ones(10),
        ...     q=torch.ones(10),
        ...     direction_history=DirectionHistory.empty(max_size=10),
        ...     residual_history=ResidualHistory.empty(),
        ...     w_prev=None,
        ...     r_prev=None,
        ...     rw_prev=0.0,
        ... )
        >>> len(state.direction_history)
        0
    """

    direction_history: DirectionHistory
    """Sliding window of search directions and matrix-vector products."""

    residual_history: ResidualHistory
    """Residual norms (absolute and relative) across iterations."""

    w_prev: torch.Tensor | None
    """Previous preconditioned residual w_{k-1} (for two-term recurrence)."""

    r_prev: torch.Tensor | None
    """Previous residual r_{k-1} (for two-term recurrence)."""

    rw_prev: float
    """Previous inner product (r_{k-1}, w_{k-1}) (for two-term recurrence)."""

    ortho_breakdown_at: int | None = None
    """Iteration of first-detected orthogonalization breakdown, or None."""

    energy_decrement: float | None = None
    """This step's exact alpha_k * rho_k (decrease in ||e_k||_A^2), or None initially."""

    @classmethod
    def create_initial(
        cls,
        u: torch.Tensor,
        r: torch.Tensor,
        w: torch.Tensor,
        d: torch.Tensor,
        q: torch.Tensor,
        residual_norm: float,
        rhs_norm: float,
        max_history: int = 10,
    ) -> CGState:
        """Create initial CG state with empty histories.

        Args:
            u (torch.Tensor): Initial solution u_0 (Notay 2000).
            r (torch.Tensor): Initial residual r_0 = b - A*u_0.
            w (torch.Tensor): Initial preconditioned residual
                w_0 = M^{-1} r_0 (Notay 2000).
            d (torch.Tensor): Initial search direction d_0 = w_0
                (Notay 2000).
            q (torch.Tensor): Initial matrix-vector product q_0 = A d_0.
            residual_norm (float): Initial residual norm ``||r_0||_2``.
            rhs_norm (float): RHS norm ``||b||_2``.
            max_history (int): Maximum direction history size.

        Returns:
            CGState: New state at iteration 0 with empty histories.

        Example:
            >>> import torch
            >>> n = 10
            >>> b = torch.ones(n)
            >>> state = CGState.create_initial(
            ...     u=torch.zeros(n),
            ...     r=b.clone(),
            ...     w=b.clone(),
            ...     d=b.clone(),
            ...     q=torch.ones(n),
            ...     residual_norm=float(torch.linalg.norm(b)),
            ...     rhs_norm=float(torch.linalg.norm(b)),
            ...     max_history=10,
            ... )
            >>> state.iteration
            0
        """
        return cls(
            iteration=0,
            converged=False,
            breakdown=False,
            divergence=False,
            residual_norm=residual_norm,
            rhs_norm=rhs_norm,
            u=u,
            r=r,
            w=w,
            d=d,
            q=q,
            direction_history=DirectionHistory.empty(max_size=max_history),
            residual_history=ResidualHistory.empty().add(
                norm_abs=residual_norm,
                norm_rel=residual_norm / rhs_norm if rhs_norm > 0 else residual_norm,
            ),
            w_prev=None,
            r_prev=None,
            rw_prev=0.0,
            ortho_breakdown_at=None,
        )
