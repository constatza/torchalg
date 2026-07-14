"""Configuration models for solver execution.

Ported from ``neuralls.domain.solver.models.config`` (see ``docs/plan.md``),
made ``slots=True`` per this repo's immutability convention (the reference
was plain ``frozen=True``).

``ComparisonData``/``ComparisonGeneral`` are not ported here: both exist
solely to configure the multi-preconditioner comparison workflow
(``comparison.py``), which is Stage 10 (lowest priority) per
``docs/plan.md``'s scope trim - porting them now would be dead weight until
that workflow lands.

``SolverConfig.trace_mode`` is typed as the literal string values
``TraceMode`` uses (``"disabled" | "minimal" | "full"`` - see
``torchalg.monitoring.TraceMode``), rather than importing
``TraceMode`` itself: ``torchalg.models`` must not depend on
``torchalg.monitoring`` (see the dependency DAG in ``docs/plan.md``),
so this stays a plain string literal even though the real enum now exists.
Revisit when Stage 9 wiring lands.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Structured configuration for solver execution.

    This dataclass captures the parameters used to run a solver, enabling
    reproducible execution and structured export.

    Attributes:
        algorithm (str): Name of the algorithm (e.g., 'preconditioned_cg').
        rtol (float): Relative tolerance for convergence.
        atol (float): Absolute tolerance for convergence.
        maxiter (int): Maximum number of iterations allowed.
        trace_mode (Literal["disabled", "minimal", "full"]): Verbosity of
            iteration tracking.
        m_max (int | None): Orthogonalization window size (used by FCG).
        preconditioner (str | None): Name or description of the
            preconditioner used.
        extra_params (Mapping[str, Any]): Additional algorithm-specific
            parameters.
    """

    algorithm: str
    """Name of the algorithm (e.g., 'preconditioned_cg')."""

    rtol: float
    """Relative tolerance for convergence."""

    atol: float
    """Absolute tolerance for convergence."""

    maxiter: int
    """Maximum number of iterations allowed."""

    trace_mode: Literal["disabled", "minimal", "full"]
    """Verbosity of iteration tracking."""

    m_max: int | None = None
    """Orthogonalization window size (used by FCG)."""

    preconditioner: str | None = None
    """Name or description of the preconditioner used."""

    extra_params: Mapping[str, Any] = field(default_factory=dict)
    """Additional algorithm-specific parameters."""

    def __post_init__(self) -> None:
        """Freeze a copied view of ``extra_params`` for nested immutability."""
        object.__setattr__(self, "extra_params", MappingProxyType(dict(self.extra_params)))


@dataclass(frozen=True, slots=True)
class SolverParams:
    """Numerical solver parameters for comparison runs.

    Attributes:
        rtol (float): Relative convergence tolerance.
        atol (float): Absolute convergence tolerance.
        max_iterations (int): Maximum iterations allowed.
        stopping_criterion (Literal["residual_norm", "fixed_iterations"]):
            When to stop iteration.
        m_max (int): FCG orthogonalization restart parameter.
        breakdown_tol (float | None): Breakdown detection tolerance.
    """

    rtol: float
    atol: float
    max_iterations: int
    stopping_criterion: Literal["residual_norm", "fixed_iterations"]
    m_max: int
    breakdown_tol: float | None
