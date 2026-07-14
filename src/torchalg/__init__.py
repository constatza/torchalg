"""Pure PyTorch iterative solver and preconditioner stack."""

from .comparison import (
    CGComparisonResult,
    ComparisonRecommendations,
    RankedRecommendation,
    format_results_summary,
    run_cg_comparison,
    summarize_best_combinations,
)
from .factories import flexible_cg, pcg

__all__ = [
    "CGComparisonResult",
    "ComparisonRecommendations",
    "RankedRecommendation",
    "flexible_cg",
    "format_results_summary",
    "pcg",
    "run_cg_comparison",
    "summarize_best_combinations",
]
