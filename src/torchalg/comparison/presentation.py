"""Text presentation for CG comparison results."""

from .models import CGComparisonResult


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
