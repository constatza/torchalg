"""Norm functions for convergence checking.

Design: Norm is a simple callable type alias. Factories create norm instances.
The convergence criterion receives the norm at construction time (DI).

This follows the Dependency Inversion principle: convergence criteria depend
on the abstract Norm type, not on specific norm implementations like scipy.linalg.norm.

Example:
    >>> from neuralls.domain.solver.strategies.norms import euclidean_norm, energy_norm
    >>> import numpy as np
    >>>
    >>> # L2 norm (default)
    >>> v = np.array([3.0, 4.0])
    >>> euclidean_norm(v)
    5.0
    >>>
    >>> # A-norm for diagonal matrix
    >>> A_diag = np.array([2.0, 8.0])
    >>> anorm = energy_norm(A_diag)
    >>> anorm(np.array([1.0, 1.0]))
    3.16...  # sqrt(2*1 + 8*1)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


# Type alias: norm takes vector, returns scalar
Norm = Callable[[NDArray], float]


def euclidean_norm(v: NDArray) -> float:
    """Compute L2 (Euclidean) norm of a vector.

    Formula: ||v||_2 = sqrt(v^T v)

    Args:
        v: Input vector of shape (n,).

    Returns:
        Euclidean norm of v as a float.

    Example:
        >>> euclidean_norm(np.array([3.0, 4.0]))
        5.0
    """
    return float(np.linalg.norm(v))


def energy_norm(A: NDArray) -> Norm:
    """Create energy norm (A-norm) function for SPD matrix A.

    The energy norm is induced by symmetric positive definite matrix A:
        ||v||_A = sqrt(v^T A v)

    For diagonal A (common in benchmarks), this uses O(n) computation
    instead of O(n^2) matrix multiplication.

    Args:
        A: SPD matrix as either:
            - 2D array of shape (n, n): Full matrix
            - 1D array of shape (n,): Diagonal entries only

    Returns:
        Norm function that computes ||v||_A for any vector v.

    Example:
        >>> # Full matrix
        >>> A = np.diag([2.0, 8.0])
        >>> anorm = energy_norm(A)
        >>> anorm(np.array([1.0, 1.0]))
        3.16...
        >>>
        >>> # Diagonal entries only (more efficient)
        >>> d = np.array([2.0, 8.0])
        >>> anorm_diag = energy_norm(d)
        >>> anorm_diag(np.array([1.0, 1.0]))
        3.16...

    Theory:
        The A-norm is central to CG convergence analysis. For SPD A,
        CG minimizes the A-norm of the error at each iteration:
            ||x* - x_k||_A = min_{y in x_0 + K_k} ||x* - y||_A
    """
    if A.ndim == 1:
        # Diagonal case: O(n) computation
        def _energy_norm_diag(v: NDArray) -> float:
            """Energy norm using diagonal entries: sqrt(sum(d_i * v_i^2))."""
            return float(np.sqrt(np.sum(A * v * v)))

        return _energy_norm_diag

    # Full matrix case: O(n^2) computation
    def _energy_norm_full(v: NDArray) -> float:
        """Energy norm using full matrix: sqrt(v^T A v)."""
        return float(np.sqrt(v @ A @ v))

    return _energy_norm_full
