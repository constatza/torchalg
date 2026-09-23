"""Public factory functions for CG-family solvers.

``pcg()`` and ``flexible_cg()`` are the two public entry points into this
package: thin, keyword-driven wrappers that assemble a
``PCGSolver``/``FCGSolver`` (``conjugate_gradient.py``) from a
``Preconditioner``, a ``ConvergenceCriterion``, and (for ``flexible_cg``)
an orthogonalization strategy, then call ``.solve()``.

References:
    - Hestenes, M.R. & Stiefel, E. (1952). Methods of Conjugate Gradients
      for Solving Linear Systems. J. Res. Natl. Bur. Stand. 49(6), 409-436.
    - Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd ed.
      SIAM. Ch. 6 (CG), Ch. 9 (preconditioned CG).
    - Notay, Y. (2000). Flexible Conjugate Gradients. SIAM J. Sci. Comput.
      22(4), 1444-1460.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

from torchalg.base import DEFAULT_ATOL, DEFAULT_RTOL, LinearSystemOperator
from torchalg.conjugate_gradient import FCGSolver, PCGSolver
from torchalg.models.result import SolverResult
from torchalg.monitoring import IterationHistory, TraceMode, coerce_trace_mode
from torchalg.preconditioners.base import BindableInputs, Preconditioner
from torchalg.preconditioners.implementations import (
    CallablePreconditioner,
    Identity,
)
from torchalg.strategies.convergence import CombinedToleranceCriterion
from torchalg.strategies.norms import Norm, euclidean_norm
from torchalg.strategies.orthogonalization import (
    DEFAULT_M_MAX,
    create_fcg_orthogonalization,
)


def flexible_cg(
    A: LinearSystemOperator,
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    *,
    preconditioner: Preconditioner | Callable[[torch.Tensor], torch.Tensor] | None = None,
    extra_inputs: Mapping[str, torch.Tensor] | None = None,
    m_max: int = DEFAULT_M_MAX,
    norm: Norm = euclidean_norm,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    maxiter: int | None = None,
    trace_mode: TraceMode | str = TraceMode.MINIMAL,
    x_exact: torch.Tensor | None = None,
    differentiable: bool = False,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, SolverResult]:
    """Solve ``A x = b`` with Flexible Conjugate Gradient (Notay 2000).

    Flexible CG generalizes PCG for preconditioners that vary between
    iterations or are not exactly SPD (e.g. a neural or adaptively-scheduled
    preconditioner): instead of the two-term recurrence
    ``d_k = w_k + beta_k d_{k-1}``, the new search direction is built by
    explicitly orthogonalizing the preconditioned residual ``w_k`` against
    up to ``m_max`` previous ``(d_j, q_j)`` pairs in the A-inner-product
    (Notay 2000, Algorithm 2.1), since PCG's short two-term recurrence is
    only exact when ``M`` is fixed and SPD.

    Mathematical method (per iteration):
        1. ``w_k = M_k^{-1} r_k`` (apply the possibly iteration-varying
           preconditioner).
        2. ``d_k = orthogonalize(w_k, {(d_j, q_j)})`` - A-conjugacy
           Gram-Schmidt over the last ``m_max`` stored directions
           (``m_max=-1``/``inf`` recovers full, untruncated orthogonalization).
        3. ``q_k = A d_k``, ``alpha_k = (r_k, w_k) / (d_k, q_k)``.
        4. ``x_{k+1} = x_k + alpha_k d_k``, ``r_{k+1} = r_k - alpha_k q_k``.

    Args:
        A (LinearSystemOperator): SPD system matrix, or a callable computing
            ``A @ v``.
        b (torch.Tensor): Right-hand side, shape ``(n,)``.
        x0 (torch.Tensor | None): Initial guess; defaults to the zero vector.
        preconditioner (Preconditioner | Callable | None): ``M_k^{-1}``
            applied each iteration; ``None`` uses the identity (unpreconditioned
            CG). A plain callable is wrapped in ``CallablePreconditioner``.
        extra_inputs (Mapping[str, torch.Tensor] | None): Named tensors bound
            onto preconditioners that declare ``BindableInputs``.
        m_max (int): Orthogonalization window - number of previous
            directions each new one is conjugated against. ``-1`` means
            unlimited (full FCG). Larger values cost more per iteration but
            more closely restore CG's optimality property under a varying
            preconditioner.
        norm (Norm): Norm used for the convergence criterion's ``||b||``.
        rtol (float): Relative residual tolerance.
        atol (float): Absolute residual tolerance.
        maxiter (int | None): Maximum iterations; defaults to ``10 * n``.
        trace_mode (TraceMode | str): Iteration-history verbosity.
        x_exact (torch.Tensor | None): Known exact solution, if available
            (e.g. a synthetic benchmark system). When given, the returned
            ``SolverResult.error_history_a_norm`` is populated with the
            exact ``||u_k - x_exact||_A`` per iteration, regardless of
            ``trace_mode`` - this is a single cheap dot product per
            iteration, not a stored vector history.
        differentiable (bool): When ``False`` (default), the solve runs
            under ``torch.inference_mode()`` since most callers use a
            frozen preconditioner and never backpropagate through the
            solve. Set ``True`` to keep normal autograd tracking, e.g. to
            train a preconditioner by unrolling the iteration.
        device (torch.device | None): Overrides automatic CUDA/CPU
            resolution when given. See ``IterativeSolverBase.solve``.

    Returns:
        tuple[torch.Tensor, SolverResult]: Final solution and diagnostics.
    """
    trace_mode_enum = coerce_trace_mode(trace_mode)
    precond_strategy = _coerce_preconditioner(preconditioner)
    _bind_extra_inputs(precond_strategy, extra_inputs)
    solver = FCGSolver(
        orthogonalization=create_fcg_orthogonalization(m_max=m_max),
        preconditioner=precond_strategy,
        convergence_criterion=CombinedToleranceCriterion(rtol=rtol, atol=atol, norm=norm),
        iteration_history=_iteration_history(trace_mode_enum, x_exact=x_exact),
        trace_mode=trace_mode_enum,
    )
    return solver.solve(
        A,
        b,
        x0,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        differentiable=differentiable,
        device=device,
    )


def pcg(
    A: LinearSystemOperator,
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    *,
    preconditioner: Preconditioner | Callable[[torch.Tensor], torch.Tensor] | None = None,
    extra_inputs: Mapping[str, torch.Tensor] | None = None,
    m_max: int | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    tol: float | None = None,
    maxiter: int | None = None,
    beta_formula: str = "fletcher_reeves",
    trace_mode: TraceMode | str = TraceMode.MINIMAL,
    x_exact: torch.Tensor | None = None,
    differentiable: bool = False,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, SolverResult]:
    """Solve ``A x = b`` with Preconditioned Conjugate Gradient.

    Standard Hestenes & Stiefel (1952) CG two-term recurrence, preconditioned
    per Saad (2003) Ch. 9 (Algorithm 9.1): assumes a single, fixed SPD
    preconditioner ``M`` for the whole solve. For a preconditioner that
    varies per iteration or is not exactly SPD, use ``flexible_cg()``
    instead - PCG's short recurrence is not valid in that case.

    Mathematical method (per iteration):
        1. ``w_k = M^{-1} r_k``.
        2. ``beta_k = (r_k, w_k) / (r_{k-1}, w_{k-1})`` (0 at ``k=0``),
           ``d_k = w_k + beta_k d_{k-1}``.
        3. ``q_k = A d_k``, ``alpha_k = (r_k, w_k) / (d_k, q_k)``.
        4. ``x_{k+1} = x_k + alpha_k d_k``, ``r_{k+1} = r_k - alpha_k q_k``.

    ``alpha_k`` is computed unconditionally every iteration, with no
    curvature/breakdown gate before the division - this matches Notay
    (2000)'s and Saad's published algorithm exactly; see
    ``.claude/plan.md`` for why an earlier draft's breakdown gate was
    reverted.

    Args:
        A (LinearSystemOperator): SPD system matrix, or a callable computing
            ``A @ v``.
        b (torch.Tensor): Right-hand side, shape ``(n,)``.
        x0 (torch.Tensor | None): Initial guess; defaults to the zero vector.
        preconditioner (Preconditioner | Callable | None): ``M^{-1}``
            applied each iteration; ``None`` uses the identity (unpreconditioned
            CG). A plain callable is wrapped in ``CallablePreconditioner``.
        extra_inputs (Mapping[str, torch.Tensor] | None): Named tensors bound
            onto preconditioners that declare ``BindableInputs``.
        m_max (int | None): If given, enables periodic reorthogonalization
            (Notay 2000 restart truncation) on top of the two-term
            recurrence, mitigating loss of A-conjugacy from rounding error
            over many iterations; ``None`` disables it (plain PCG).
        rtol (float): Relative residual tolerance.
        atol (float): Absolute residual tolerance.
        tol (float | None): Deprecated alias for ``rtol``; takes precedence
            over ``rtol`` when given.
        maxiter (int | None): Maximum iterations; defaults to ``10 * n``.
        beta_formula (str): Beta recurrence formula. Only
            ``"fletcher_reeves"`` is supported; any other value raises
            ``ValueError``.
        trace_mode (TraceMode | str): Iteration-history verbosity.
        x_exact (torch.Tensor | None): Known exact solution, if available
            (e.g. a synthetic benchmark system). When given, the returned
            ``SolverResult.error_history_a_norm`` is populated with the
            exact ``||u_k - x_exact||_A`` per iteration, regardless of
            ``trace_mode`` - this is a single cheap dot product per
            iteration, not a stored vector history.
        differentiable (bool): When ``False`` (default), the solve runs
            under ``torch.inference_mode()`` since most callers use a
            frozen preconditioner and never backpropagate through the
            solve. Set ``True`` to keep normal autograd tracking, e.g. to
            train a preconditioner by unrolling the iteration.
        device (torch.device | None): Overrides automatic CUDA/CPU
            resolution when given. See ``IterativeSolverBase.solve``.

    Returns:
        tuple[torch.Tensor, SolverResult]: Final solution and diagnostics.
    """
    trace_mode_enum = coerce_trace_mode(trace_mode)
    rtol_value = rtol if tol is None else tol
    precond_strategy = _coerce_preconditioner(preconditioner)
    _bind_extra_inputs(precond_strategy, extra_inputs)
    reorthogonalization = None
    if m_max is not None:
        reorthogonalization = create_fcg_orthogonalization(m_max=m_max)

    solver = PCGSolver(
        preconditioner=precond_strategy,
        convergence_criterion=CombinedToleranceCriterion(rtol=rtol_value, atol=atol),
        reorthogonalization=reorthogonalization,
        beta_formula=beta_formula,
        iteration_history=_iteration_history(trace_mode_enum, x_exact=x_exact),
        trace_mode=trace_mode_enum,
    )
    return solver.solve(
        A,
        b,
        x0,
        rtol=rtol_value,
        atol=atol,
        maxiter=maxiter,
        differentiable=differentiable,
        device=device,
    )


def _coerce_preconditioner(
    preconditioner: Preconditioner | Callable[[torch.Tensor], torch.Tensor] | None,
) -> Preconditioner:
    """Normalize public preconditioner inputs to the ``Preconditioner`` interface."""
    if preconditioner is None:
        return Identity()
    if isinstance(preconditioner, Preconditioner):
        return preconditioner
    return CallablePreconditioner(preconditioner)


def _bind_extra_inputs(
    preconditioner: Preconditioner,
    extra_inputs: Mapping[str, torch.Tensor] | None,
) -> None:
    """Bind declared extra tensors on preconditioners that support them."""
    if extra_inputs is None or not isinstance(preconditioner, BindableInputs):
        return
    declared = preconditioner.extra_input_names
    filtered = {name: extra_inputs[name] for name in declared if name in extra_inputs}
    preconditioner.bind_inputs(**filtered)


def _iteration_history(
    trace_mode: TraceMode, *, x_exact: torch.Tensor | None = None
) -> IterationHistory:
    """Create iteration history for the requested trace mode.

    Always returns a real ``IterationHistory`` object; for ``TraceMode.DISABLED``,
    the object is configured to be a no-op (its ``log_iteration()`` method returns
    early). This eliminates the need for None checks elsewhere.
    """
    return IterationHistory(mode=trace_mode, x_exact=x_exact)
