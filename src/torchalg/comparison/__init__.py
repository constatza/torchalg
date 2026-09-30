"""Compare preconditioners without coupling execution to presentation policy."""

from .models import CGComparisonResult, ComparisonRecommendations, RankedRecommendation
from .presentation import format_results_summary
from .recommendations import summarize_best_combinations
from .runner import run_cg_comparison

__all__ = [
    "CGComparisonResult",
    "ComparisonRecommendations",
    "RankedRecommendation",
    "format_results_summary",
    "run_cg_comparison",
    "summarize_best_combinations",
]
