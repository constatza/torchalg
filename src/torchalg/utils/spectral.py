"""Condition-number diagnostics for dense symmetric matrices.

Replaces ``neuralls.domain.analysis.spectra.compute_condition_numbers``'
folded-spectrum shifted-power-iteration estimator, which was found (via a
synthetic stiffness matrix mimicking a stiff cube with a much softer
embedded region, e.g. a 1000x-softer sphere) to catastrophically fail on
exactly the matrices this diagnostic exists to characterize: those with a
wide, ill-conditioned eigenvalue spread and a cluster of near-zero modes.

Root cause of that failure (see the reference's
``_power_iteration_extremes``/``_power_iterate``):

1. **Slow, cluster-blind convergence.** Plain power iteration converges to
   the dominant eigenvector at the linear rate ``|lambda_2 / lambda_1|``.
   A heterogeneous-stiffness spectrum is exactly the case where the
   extremal eigenvalues are clustered (many near-zero modes from a soft
   inclusion, many near-``lambda_max`` modes from the stiff bulk), so this
   rate is close to 1 and the iteration has not converged after any
   practical iteration budget.
2. **Catastrophic cancellation in the shift.** The smallest eigenvalue is
   recovered as ``lambda_min = shift - power_iterate(shift*I - A)`` with
   ``shift ~= 1.01 * lambda_max``. Whenever the matrix is genuinely
   ill-conditioned (``lambda_min << lambda_max``, precisely when an
   accurate condition number matters most), this subtracts two numbers
   that agree to most of their significant digits, destroying the relative
   precision of the tiny result — the estimator's own
   ``if lambda_min <= 0: raise ValueError(...)`` guard exists because this
   routinely drives the estimate negative, which is the exact symptom
   reported against a cube/soft-sphere stiffness matrix.

This module replaces the approximate iterative estimator with the exact
textbook definition instead: ``torchalg`` is dense-only throughout (see
``docs/plan.md``), so there is no matrix-free constraint forcing an
iterative approximation in the first place. For a real symmetric matrix,
the 2-norm condition number is ``|lambda_max| / |lambda_min|`` computed
from the *full* eigendecomposition (Golub & Van Loan, "Matrix
Computations", 4th ed., SS2.7.2 and SS8.3) — backward-stable (LAPACK
``syevd``, via ``torch.linalg.eigvalsh``), with no iteration count,
convergence tolerance, or shift heuristic to tune, and therefore no
literature-undocumented threshold that could silently cap what the
diagnostic can resolve.

References:
    - Golub, G.H. & Van Loan, C.F. (2013). Matrix Computations, 4th ed.,
      Johns Hopkins University Press. SS2.7.2 (condition number via
      singular/eigen values), SS8.3 (symmetric eigenvalue problem).
"""

from __future__ import annotations

import math

import torch


def condition_number(matrix: torch.Tensor) -> float:
    """Compute the exact 2-norm condition number of a symmetric matrix.

    Uses the full eigendecomposition of ``matrix`` (``torch.linalg.eigvalsh``,
    LAPACK ``syevd``/``syevr``): the 2-norm condition number of a real
    symmetric matrix is ``max(|eigenvalues|) / min(|eigenvalues|)`` (Golub &
    Van Loan, "Matrix Computations", 4th ed., SS2.7.2). Only the lower
    triangle of ``matrix`` is read (``eigvalsh`` convention), so the result
    is well-defined even if the caller passes a matrix that is only
    numerically, not exactly, symmetric.

    Args:
        matrix (torch.Tensor): Square symmetric matrix, shape ``(n, n)``.

    Returns:
        float: ``max(|eigenvalues|) / min(|eigenvalues|)``. ``math.inf`` if
            ``matrix`` has an exact zero eigenvalue (singular).

    Raises:
        ValueError: If ``matrix`` is not square.

    Example:
        >>> import torch
        >>> condition_number(torch.eye(3, dtype=torch.float64))
        1.0
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got shape {tuple(matrix.shape)}")

    eigenvalues = torch.linalg.eigvalsh(matrix).abs()
    lambda_min = float(eigenvalues.min())
    lambda_max = float(eigenvalues.max())
    if lambda_min == 0.0:
        return math.inf
    return lambda_max / lambda_min
