"""Solution and system validation utilities for iterative solvers.

Ported from ``neuralls.domain.solver.utils.validation`` (see
``docs/plan.md``) with ``numpy`` operations translated to ``torch``
equivalents. ``record_breakdown_event`` is not ported as-is: its only job in
the reference is constructing and appending an ``EventLog``/``EventType``
entry, a system this migration deliberately drops (see ``docs/plan.md``'s
"Architectural corrections" section — replaced by a ``TerminationDiagnostics``
frozen dataclass landing in Stage 2, wired up in Stage 9). The one piece of
real, reusable logic inside it — classifying *why* a solution broke down
(NaN vs. Inf) into a reason string — is kept below as
``describe_breakdown_reason``, with the ``EventLog`` side effect dropped
entirely rather than stubbed against a type that doesn't exist yet.

Design:
    - Pure functions (no state).
    - Reusable across solver types.
"""

from __future__ import annotations

import torch


def validate_matrix(A: torch.Tensor) -> None:
    """Validate system matrix is square, 2-D, and finite.

    Args:
        A (torch.Tensor): Input matrix.

    Raises:
        ValueError: If matrix is not 2-D, not square, or contains
            non-finite values.
    """
    if A.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got {A.ndim}D")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {tuple(A.shape)}")
    if not torch.isfinite(A).all():
        raise ValueError("Matrix contains non-finite values")


def validate_rhs_vector(b: torch.Tensor | None, A: torch.Tensor | None = None) -> None:
    """Validate RHS vector dimensions and finiteness.

    Args:
        b (torch.Tensor | None): RHS vector.
        A (torch.Tensor | None): Optional system matrix for size checking.

    Raises:
        ValueError: If RHS is invalid.
    """
    if b is None:
        return
    if b.ndim > 2:
        raise ValueError(f"RHS must be 1D or 2D, got {b.ndim}D")
    if b.ndim == 2 and b.shape[1] != 1:
        raise ValueError(f"RHS must be a column vector, got shape {tuple(b.shape)}")
    if not torch.isfinite(b).all():
        raise ValueError("RHS contains non-finite values")
    if A is not None and len(b.flatten()) != A.shape[0]:
        raise ValueError(f"RHS length {len(b.flatten())} doesn't match matrix size {A.shape[0]}")


def validate_ax_equals_b(matrix: torch.Tensor, rhs: torch.Tensor, lhs: torch.Tensor) -> None:
    """Verify ``matrix @ lhs`` and ``rhs`` are consistent up to a positive scalar.

    Scale-agnostic (doesn't assume any particular multiplier baked into rhs):
    checks the two vectors are parallel via cosine similarity. Intended as a
    direct runtime proof that a loaded/normalized linear system is internally
    consistent whenever a known true solution (``lhs``) is available — a
    structural safety net against double-normalization or mismatched-scale
    bugs, not just careful-by-construction code.

    Args:
        matrix (torch.Tensor): System matrix A.
        rhs (torch.Tensor): Right-hand side vector b.
        lhs (torch.Tensor): Known true solution x.

    Raises:
        ValueError: If the Ax=b invariant is violated beyond floating
            tolerance.
    """
    predicted = matrix @ lhs
    predicted_norm = float(torch.linalg.norm(predicted))
    rhs_norm = float(torch.linalg.norm(rhs))
    if predicted_norm == 0.0 or rhs_norm == 0.0:
        return
    cosine = float(torch.dot(predicted, rhs) / (predicted_norm * rhs_norm))
    if cosine < 1.0 - 1e-6:
        raise ValueError(
            "Ax=b invariant violated: cosine similarity between (matrix @ lhs) "
            f"and rhs is {cosine:.10f} (expected ~1.0). This indicates a "
            "double-normalization or mismatched-scale bug in the linear system."
        )


def check_solution_validity(solution: torch.Tensor) -> bool:
    """Check if solution contains only finite values.

    Args:
        solution (torch.Tensor): Solution vector to validate.

    Returns:
        bool: ``True`` if solution is valid (all finite), ``False``
            otherwise.

    Example:
        >>> import torch
        >>> check_solution_validity(torch.tensor([1.0, 2.0, 3.0]))
        True
        >>> check_solution_validity(torch.tensor([1.0, float("nan"), 3.0]))
        False
        >>> check_solution_validity(torch.tensor([1.0, float("inf"), 3.0]))
        False
    """
    return bool(torch.isfinite(solution).all())


def describe_breakdown_reason(solution: torch.Tensor) -> str:
    """Classify why ``solution`` contains non-finite values.

    Pure extraction of the breakdown-classification logic from the
    reference's ``record_breakdown_event``, with its ``EventLog`` side effect
    dropped (see module docstring). Not wired to anything yet: Stage 9
    attaches this reason to
    ``models.diagnostics.TerminationDiagnostics.breakdown_reason`` when the
    solver loop that detects breakdown is ported.

    Args:
        solution (torch.Tensor): Solution vector suspected of breakdown.

    Returns:
        str: Comma-separated reason describing the non-finite values found,
            e.g. ``"nan_in_solution, inf_in_solution"``. Empty string if
            ``solution`` is actually finite everywhere.

    Example:
        >>> import torch
        >>> describe_breakdown_reason(torch.tensor([1.0, float("nan"), 3.0]))
        'nan_in_solution'
    """
    has_nan = bool(torch.isnan(solution).any())
    has_inf = bool(torch.isinf(solution).any())

    reason_parts = []
    if has_nan:
        reason_parts.append("nan_in_solution")
    if has_inf:
        reason_parts.append("inf_in_solution")

    return ", ".join(reason_parts)
