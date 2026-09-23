"""Structural type protocols for solver state fields and solver objects.

``HasVectors``/``HasDirectionHistory`` are ported from
``neuralls.domain.solver.models.protocols`` (see ``docs/plan.md``) with
``NDArray`` fields translated to ``torch.Tensor``.

``SolverProtocol`` is new - not a port. See ``docs/plan.md``'s
"Architectural corrections" section: the reference's
``SciPyCGSolver`` is duck-typed against ``IterativeSolverBase`` ("does not
inherit ``IterativeSolverBase``... compatible ``.solve()`` signature, not
enforced by an ABC"). Since the scipy oracle now lives in test-support only
(``tests/support/scipy_reference/scipy_wrapper.py::SciPyCGSolver``, not
production), ``SolverProtocol`` gives that structural compatibility an
actual, checkable shape: both future ``IterativeSolverBase`` subclasses
(Stage 9) and the scipy oracle satisfy it without either one inheriting
from the other, so the parametrized ``[fcg, pcg, scipy_cg]`` solver-family
tests can be typed instead of relying on untyped duck typing.

Design Principles:
    - Dependency Inversion: Depend on protocols (interfaces), not concrete
      classes.
    - Structural Typing: Type safety without inheritance constraints.
    - Single Responsibility: Each protocol defines one structural contract.
    - Open/Closed: Easy to add new protocols without modifying existing
      code.

Why Protocols Over Inheritance:
    1. No variance issues - ``isinstance()`` works at runtime.
    2. No ``type: ignore`` comments needed.
    3. Multiple protocols can be satisfied by the same class.
    4. Loose coupling - implementation can change freely.

Theory:
    Protocols implement structural subtyping (PEP 544). A class satisfies a
    Protocol if it has the required attributes/methods, regardless of
    inheritance. This is "duck typing" with compile-time type checking.

References:
    - PEP 544: Protocols (Structural Subtyping).
    - Liskov Substitution Principle (SOLID).
    - Interface Segregation Principle (SOLID).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

from .history import DirectionHistory, ResidualHistory
from .result import SolverResult


@runtime_checkable
class HasVectors(Protocol):
    """Protocol for states with Krylov iteration vectors.

    Classes satisfying this protocol have the five working vectors
    maintained during Krylov subspace iteration: solution, residual,
    preconditioned residual, search direction, and matrix-vector product.

    Satisfied By:
        - ``KrylovState``
        - ``CGState`` (inherits from ``KrylovState``)

    Mathematical Context (Notay 2000):
        Krylov methods maintain these vectors at each iteration i:
        - u_i: Current solution estimate (Notay 2000).
        - r_i: Residual r_i = b - A u_i.
        - w_i: Preconditioned residual w_i = M^{-1} r_i (Notay 2000).
        - d_i: Search direction (A-conjugate to previous directions)
          (Notay 2000).
        - q_i: Matrix-vector product q_i = A d_i.

    Usage:
        >>> def extract_residual(state: SolverState) -> torch.Tensor | None:
        ...     if isinstance(state, HasVectors):
        ...         return state.r  # Type-safe access.
        ...     return None

    Theory:
        The Krylov subspace ``K_i(A, r_0) = span{r_0, A*r_0, ..., A^(i-1)*r_0}``
        is implicitly represented by these vectors. The search direction
        ``d_i`` is constructed to be A-conjugate to all previous directions.

    References:
        - Notay, Y. (2000). Flexible Conjugate Gradients.
        - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems.
        - Greenbaum, A. (1997). Iterative Methods for Solving Linear Systems.
    """

    u: torch.Tensor
    """Solution vector u_i at current iteration (Notay 2000)."""

    r: torch.Tensor
    """Residual vector r_i = b - A u_i."""

    w: torch.Tensor
    """Preconditioned residual w_i = M^{-1} r_i (Notay 2000)."""

    d: torch.Tensor
    """Search direction d_i (A-conjugate to previous directions) (Notay 2000)."""

    q: torch.Tensor
    """Matrix-vector product q_i = A d_i."""


@runtime_checkable
class HasDirectionHistory(Protocol):
    """Protocol for states with CG direction and residual history tracking.

    Classes satisfying this protocol maintain sliding window histories of
    search directions and residuals for orthogonalization and diagnostics.

    Satisfied By:
        - ``CGState``

    Mathematical Context (Notay 2000):
        Flexible CG uses explicit orthogonalization of search directions::

            d_i (A-orthogonal to) d_j for j in sliding window [i-m, i-1]

        This requires storing recent direction vectors (d_j) and their
        matrix products (q_j = A d_j) for computing orthogonalization
        coefficients: beta_j = (w_i, q_j) / (d_j, q_j).

    Usage:
        >>> def orthogonalize(state: SolverState, w: torch.Tensor) -> torch.Tensor:
        ...     if isinstance(state, HasDirectionHistory):
        ...         history = state.direction_history
        ...         d_vectors = history.d_vectors
        ...         q_vectors = history.q_vectors
        ...         # Compute orthogonalization...
        ...     return w  # No orthogonalization if no history.

    Theory (Notay 2000):
        Truncated orthogonalization maintains approximate A-conjugacy:
        ``(d_i, A d_j) ~= 0`` for j in window [i-m, i-1]. This prevents
        numerical breakdown when the preconditioner M_i is not symmetric
        positive definite (SPD). Full orthogonalization would require
        O(i) storage; truncation reduces to O(m).

    References:
        - Notay, Y. (2000). Flexible Conjugate Gradients.
        - Saad, Y. (2003). Section 9.4: Flexible Variants.
    """

    direction_history: DirectionHistory
    """Sliding window history of search directions (d_i, q_i) for orthogonalization."""

    residual_history: ResidualHistory
    """History of residual norms (absolute and relative) for diagnostics."""


@runtime_checkable
class HasEnergyDecrement(Protocol):
    """Protocol for states exposing CG's exact per-iteration A-norm-squared decrement.

    Satisfied By:
        - ``CGState``

    Theory:
        For SPD ``A``, CG's error strictly decreases in the A-norm each
        iteration by exactly ``alpha_k * rho_k``, where
        ``rho_k = (r_k, w_k)`` (Golub & Meurant 1994; Strakoš & Tichý 2002)::

            ||e_k||_A^2 - ||e_{k+1}||_A^2 = alpha_k * rho_k

        Both ``alpha_k`` and ``rho_k`` are already computed by the CG
        recurrence, so this is a free byproduct of one iteration, not an
        extra computation. Summing a forward window of these decrements
        gives a ground-truth-free lower-bound estimate of ``||e_k||_A^2``
        (see ``torchalg.monitoring.analysis.golub_meurant_error_bound``).

    References:
        - Golub, G.H. & Meurant, G. (1994). Matrices, moments and quadrature.
        - Strakoš, Z. & Tichý, P. (2002). On error estimation in the
          conjugate gradient method. ETNA 13, 56-80.
    """

    energy_decrement: torch.Tensor | float | None
    """This iteration's ``alpha_k * rho_k`` (0-d tensor or float), or ``None`` before the first step."""


@runtime_checkable
class SolverProtocol(Protocol):
    """Minimal structural contract shared by every CG-family solver.

    Any object exposing a compatible ``solve(...) -> (x, SolverResult)``
    method structurally satisfies this protocol - no explicit inheritance
    required (PEP 544). Both the production ``IterativeSolverBase``
    subclasses (``PCGSolver``, ``FCGSolver``; Stage 9) and the test-support
    scipy oracle (``tests/support/scipy_reference/scipy_wrapper.py::SciPyCGSolver``)
    satisfy this shape, so tests parametrized over ``[fcg, pcg, scipy_cg]``
    can be typed against one interface rather than relying on untyped duck
    typing (see ``docs/plan.md``'s "Architectural corrections" section).

    Deliberately minimal: only the call shape needed to type the
    parametrized ``solver_factories`` test fixture, not a full solver
    interface specification (no strategy-injection constructor shape, no
    trace-mode/history wiring - those stay implementation details of each
    concrete solver).
    """

    def solve(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        x0: torch.Tensor | None = None,
        *,
        rtol: float = 1e-6,
        atol: float = 1e-14,
        maxiter: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, SolverResult]:
        """Solve ``A x = b`` iteratively.

        Args:
            A (torch.Tensor): System matrix, shape ``(n, n)``.
            b (torch.Tensor): Right-hand side vector, shape ``(n,)``.
            x0 (torch.Tensor | None): Initial guess; zero vector if
                ``None``.
            rtol (float): Relative convergence tolerance.
            atol (float): Absolute convergence tolerance.
            maxiter (int | None): Maximum iteration count.
            **kwargs (Any): Implementation-specific extra parameters.

        Returns:
            tuple[torch.Tensor, SolverResult]: The solution vector and the
                execution result/diagnostics.
        """
        ...
