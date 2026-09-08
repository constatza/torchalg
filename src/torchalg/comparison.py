"""Comparison runner for multiple CG preconditioners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from torchalg.factories import flexible_cg
from torchalg.preconditioners.base import Preconditioner


@dataclass(frozen=True, slots=True)
class CGComparisonResult:
    """Result for one preconditioner in a CG comparison run."""

    x: torch.Tensor
    converged: bool
    iterations: int
    residual: float
    residual_abs: float
    residual_history_rel: tuple[float, ...]
    residual_history_abs: tuple[float, ...]
    preconditioner: str
    initial_guess: torch.Tensor
    exact_error: float | None
    rhs_norm: float
    breakdown: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RankedRecommendation:
    """Ranked recommendation entry for a converged preconditioner."""

    label: str
    iterations: int
    residual: float
    residual_abs: float
    breakdown: bool


@dataclass(frozen=True, slots=True)
class ComparisonRecommendations:
    """Ranked comparison recommendations."""

    ranked: tuple[RankedRecommendation, ...]
    overall_best: RankedRecommendation | None


def run_cg_comparison(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    preconditioners: Mapping[str, Preconditioner],
    x0: torch.Tensor | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-14,
    maxiter: int = 100,
    m_max: int = 20,
) -> dict[str, CGComparisonResult]:
    """Run Flexible CG with multiple preconditioners for comparison."""
    x0_base = torch.zeros_like(b) if x0 is None else x0.clone()
    preconditioners_eff = _with_identity_baseline(preconditioners)
    x_exact = torch.linalg.solve(A, b)

    results: dict[str, CGComparisonResult] = {}
    for name, preconditioner in preconditioners_eff.items():
        results[name] = _run_one_comparison(
            A=A,
            b=b,
            x0=x0_base,
            x_exact=x_exact,
            preconditioner_name=name,
            preconditioner=preconditioner,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            m_max=m_max,
        )

    return results


def format_results_summary(results: dict[str, CGComparisonResult]) -> str:
    """Format CG comparison results into a readable summary."""
    lines = ["Flexible CG results:"]
    for name, result in results.items():
        status = "ok" if result.converged else "fail"
        line = (
            f"- {name:<18} status={status:<4} iters={result.iterations:>3}  "
            f"rel_res={result.residual:.3e} (abs={result.residual_abs:.3e})"
        )
        if result.exact_error is not None:
            line += f"  exact_err={result.exact_error:.3e}"
        if result.error:
            line += f"  note={result.error}"
        elif result.breakdown and not result.converged:
            line += "  note=breakdown"
        elif result.breakdown and result.converged:
            line += "  note=breakdown_post_convergence"
        lines.append(line)
    return "\n".join(lines)


def summarize_best_combinations(
    results: dict[str, CGComparisonResult],
) -> ComparisonRecommendations:
    """Rank converged comparison results by final relative residual."""
    ranked = tuple(
        sorted(
            (
                RankedRecommendation(
                    label=label,
                    iterations=result.iterations,
                    residual=result.residual,
                    residual_abs=result.residual_abs,
                    breakdown=result.breakdown,
                )
                for label, result in results.items()
                if result.converged
            ),
            key=lambda entry: entry.residual,
        )
    )
    return ComparisonRecommendations(ranked=ranked, overall_best=ranked[0] if ranked else None)


def _with_identity_baseline(
    preconditioners: Mapping[str, Preconditioner],
) -> dict[str, Preconditioner | None]:
    """Return preconditioners with a factory-default ``none`` baseline included."""
    result: dict[str, Preconditioner | None] = dict(preconditioners)
    if "none" not in result:
        result["none"] = None
    return result


def _run_one_comparison(
    *,
    A: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor,
    x_exact: torch.Tensor,
    preconditioner_name: str,
    preconditioner: Preconditioner | None,
    rtol: float,
    atol: float,
    maxiter: int,
    m_max: int,
) -> CGComparisonResult:
    """Run one preconditioner comparison and capture failures as data."""
    try:
        x_sol, info = flexible_cg(
            A,
            b,
            x0,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            preconditioner=preconditioner,
            m_max=m_max,
        )
    except (ValueError, RuntimeError, torch.linalg.LinAlgError) as solver_exc:
        return _failed_result(preconditioner_name, x0, b, solver_exc)

    exact_norm = float(torch.linalg.norm(x_exact))
    exact_error_abs = float(torch.linalg.norm(x_sol - x_exact))
    exact_error = exact_error_abs / exact_norm if exact_norm > 0 else exact_error_abs
    return CGComparisonResult(
        x=x_sol,
        converged=info.converged,
        iterations=info.iterations,
        residual=info.residual,
        residual_abs=info.residual_abs,
        residual_history_rel=info.residual_history_rel or (),
        residual_history_abs=info.residual_history_abs or (),
        preconditioner=preconditioner_name,
        initial_guess=x0.clone(),
        exact_error=exact_error,
        rhs_norm=info.rhs_norm,
        breakdown=info.breakdown,
    )


def _failed_result(
    preconditioner_name: str,
    x0: torch.Tensor,
    b: torch.Tensor,
    solver_exc: Exception,
) -> CGComparisonResult:
    """Build failed comparison result from a solver exception."""
    return CGComparisonResult(
        x=x0.clone(),
        converged=False,
        iterations=0,
        residual=float("inf"),
        residual_abs=float("inf"),
        residual_history_rel=(),
        residual_history_abs=(),
        preconditioner=preconditioner_name,
        initial_guess=x0.clone(),
        exact_error=None,
        rhs_norm=float(torch.linalg.norm(b)),
        breakdown=False,
        error=f"CG solver failed: {solver_exc}",
    )
