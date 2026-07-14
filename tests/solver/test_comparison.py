"""Tests for ``torchalg.comparison``."""

from __future__ import annotations

import torch

from torchalg.comparison import (
    CGComparisonResult,
    format_results_summary,
    run_cg_comparison,
    summarize_best_combinations,
)
from torchalg.preconditioners.implementations.identity import Identity
from torchalg.preconditioners.implementations.jacobi import JacobiPreconditioner


def test_run_cg_comparison_adds_identity_baseline(
    diagonal_4x4_torch: torch.Tensor,
    rhs_progression_4_torch: torch.Tensor,
) -> None:
    """Comparison runner includes a ``none`` baseline when not supplied."""
    results = run_cg_comparison(
        diagonal_4x4_torch,
        rhs_progression_4_torch,
        preconditioners={"jacobi": JacobiPreconditioner(diagonal_4x4_torch)},
        rtol=1e-10,
        maxiter=20,
    )

    assert set(results) == {"jacobi", "none"}
    assert results["jacobi"].converged is True
    assert results["none"].converged is True
    assert results["jacobi"].iterations <= results["none"].iterations


def test_run_cg_comparison_records_failed_preconditioner(
    diagonal_4x4_torch: torch.Tensor,
    rhs_progression_4_torch: torch.Tensor,
) -> None:
    """Comparison runner captures solver failures as failed comparison results."""

    class _BrokenIdentity(Identity):
        def apply(self, residual: torch.Tensor, context=None) -> torch.Tensor:  # noqa: ANN001
            raise RuntimeError("broken preconditioner")

    results = run_cg_comparison(
        diagonal_4x4_torch,
        rhs_progression_4_torch,
        preconditioners={"broken": _BrokenIdentity()},
        maxiter=20,
    )

    assert results["broken"].converged is False
    assert results["broken"].error is not None
    assert "broken preconditioner" in results["broken"].error
    assert results["none"].converged is True


def test_format_results_summary_includes_status_and_error() -> None:
    """Summary formatting emits stable status, residual, and error details."""
    results = {
        "ok": CGComparisonResult(
            x=torch.ones(2),
            converged=True,
            iterations=3,
            residual=1e-8,
            residual_abs=2e-8,
            residual_history_rel=(1.0, 1e-8),
            residual_history_abs=(2.0, 2e-8),
            preconditioner="ok",
            initial_guess=torch.zeros(2),
            exact_error=1e-9,
            rhs_norm=2.0,
            breakdown=False,
        ),
        "bad": CGComparisonResult(
            x=torch.zeros(2),
            converged=False,
            iterations=0,
            residual=float("inf"),
            residual_abs=float("inf"),
            residual_history_rel=(),
            residual_history_abs=(),
            preconditioner="bad",
            initial_guess=torch.zeros(2),
            exact_error=None,
            rhs_norm=2.0,
            breakdown=False,
            error="failed",
        ),
    }

    summary = format_results_summary(results)

    assert "Flexible CG results:" in summary
    assert "ok" in summary
    assert "status=ok" in summary
    assert "bad" in summary
    assert "status=fail" in summary
    assert "note=failed" in summary


def test_summarize_best_combinations_ranks_converged_by_residual() -> None:
    """Recommendations ignore failures and rank converged entries by residual."""
    slow = CGComparisonResult(
        x=torch.zeros(1),
        converged=True,
        iterations=5,
        residual=1e-5,
        residual_abs=1e-5,
        residual_history_rel=(1e-5,),
        residual_history_abs=(1e-5,),
        preconditioner="slow",
        initial_guess=torch.zeros(1),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
    )
    fast = CGComparisonResult(
        x=torch.zeros(1),
        converged=True,
        iterations=2,
        residual=1e-8,
        residual_abs=1e-8,
        residual_history_rel=(1e-8,),
        residual_history_abs=(1e-8,),
        preconditioner="fast",
        initial_guess=torch.zeros(1),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
    )
    failed = CGComparisonResult(
        x=torch.zeros(1),
        converged=False,
        iterations=1,
        residual=1.0,
        residual_abs=1.0,
        residual_history_rel=(1.0,),
        residual_history_abs=(1.0,),
        preconditioner="failed",
        initial_guess=torch.zeros(1),
        exact_error=None,
        rhs_norm=1.0,
        breakdown=False,
    )

    recommendations = summarize_best_combinations(
        {"slow": slow, "fast": fast, "failed": failed},
    )

    assert recommendations.overall_best is not None
    assert recommendations.overall_best.label == "fast"
    assert [entry.label for entry in recommendations.ranked] == ["fast", "slow"]
