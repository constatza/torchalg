"""Immutable data returned by CG comparison workflows."""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
