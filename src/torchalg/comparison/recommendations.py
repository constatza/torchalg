"""Ranking policy for completed CG comparison runs."""

from .models import CGComparisonResult, ComparisonRecommendations, RankedRecommendation


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
