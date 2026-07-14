"""Public factory functions for CG-family solvers."""

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
    breakdown_tol: float | None = None,
    maxiter: int | None = None,
    trace_mode: TraceMode | str = TraceMode.MINIMAL,
) -> tuple[torch.Tensor, SolverResult]:
    """Solve ``A x = b`` with Flexible CG."""
    trace_mode_enum = coerce_trace_mode(trace_mode)
    precond_strategy = _coerce_preconditioner(preconditioner)
    _bind_extra_inputs(precond_strategy, extra_inputs)
    solver = FCGSolver(
        orthogonalization=create_fcg_orthogonalization(m_max=m_max),
        preconditioner=precond_strategy,
        convergence_criterion=CombinedToleranceCriterion(rtol=rtol, atol=atol, norm=norm),
        iteration_history=_iteration_history(trace_mode_enum),
        trace_mode=trace_mode_enum,
    )
    return solver.solve(
        A,
        b,
        x0,
        rtol=rtol,
        atol=atol,
        breakdown_tol=breakdown_tol,
        maxiter=maxiter,
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
    breakdown_tol: float | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    beta_formula: str = "fletcher_reeves",
    trace_mode: TraceMode | str = TraceMode.MINIMAL,
) -> tuple[torch.Tensor, SolverResult]:
    """Solve ``A x = b`` with Preconditioned CG."""
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
        iteration_history=_iteration_history(trace_mode_enum),
        trace_mode=trace_mode_enum,
    )
    return solver.solve(
        A,
        b,
        x0,
        rtol=rtol_value,
        atol=atol,
        breakdown_tol=breakdown_tol,
        maxiter=maxiter,
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


def _iteration_history(trace_mode: TraceMode) -> IterationHistory | None:
    """Create iteration history for enabled trace modes."""
    if trace_mode == TraceMode.DISABLED:
        return None
    return IterationHistory(mode=trace_mode)
