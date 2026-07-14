"""Norm functions for convergence checking.

Ported from ``neuralls.domain.solver.strategies.norms`` (see
``docs/plan.md``) with ``numpy`` operations translated to
``torch.linalg.norm``/tensor ops.

Design: ``Norm`` is a simple callable type alias. Factories create norm
instances. The convergence criterion receives the norm at construction time
(dependency injection).

This follows the Dependency Inversion principle: convergence criteria
depend on the abstract ``Norm`` type, not on specific norm implementations.

Example:
    >>> from torchalg.strategies.norms import euclidean_norm, energy_norm
    >>> import torch
    >>>
    >>> # L2 norm (default)
    >>> v = torch.tensor([3.0, 4.0])
    >>> euclidean_norm(v)
    5.0
    >>>
    >>> # A-norm for diagonal matrix
    >>> a_diag = torch.tensor([2.0, 8.0])
    >>> anorm = energy_norm(a_diag)
    >>> anorm(torch.tensor([1.0, 1.0]))
    3.16...  # sqrt(2*1 + 8*1)
"""

from __future__ import annotations

from collections.abc import Callable

import torch

# Type alias: norm takes vector, returns scalar.
type Norm = Callable[[torch.Tensor], float]


def euclidean_norm(v: torch.Tensor) -> float:
    """Compute L2 (Euclidean) norm of a vector.

    Formula: ``||v||_2 = sqrt(v^T v)``.

    Args:
        v (torch.Tensor): Input vector of shape ``(n,)``.

    Returns:
        float: Euclidean norm of ``v``.

    Example:
        >>> euclidean_norm(torch.tensor([3.0, 4.0]))
        5.0
    """
    return float(torch.linalg.norm(v))


def energy_norm(A: torch.Tensor) -> Norm:
    """Create energy norm (A-norm) function for SPD matrix A.

    The energy norm is induced by symmetric positive definite matrix A::

        ||v||_A = sqrt(v^T A v)

    For diagonal A (common in benchmarks), this uses O(n) computation
    instead of O(n^2) matrix multiplication.

    Args:
        A (torch.Tensor): SPD matrix as either:
            - 2D tensor of shape ``(n, n)``: Full matrix.
            - 1D tensor of shape ``(n,)``: Diagonal entries only.

    Returns:
        Norm: Norm function that computes ``||v||_A`` for any vector v.

    Example:
        >>> # Full matrix
        >>> A = torch.diag(torch.tensor([2.0, 8.0]))
        >>> anorm = energy_norm(A)
        >>> anorm(torch.tensor([1.0, 1.0]))
        3.16...
        >>>
        >>> # Diagonal entries only (more efficient)
        >>> d = torch.tensor([2.0, 8.0])
        >>> anorm_diag = energy_norm(d)
        >>> anorm_diag(torch.tensor([1.0, 1.0]))
        3.16...

    Theory:
        The A-norm is central to CG convergence analysis. For SPD A, CG
        minimizes the A-norm of the error at each iteration::

            ||x* - x_k||_A = min_{y in x_0 + K_k} ||x* - y||_A
    """
    if A.ndim == 1:

        def _energy_norm_diag(v: torch.Tensor) -> float:
            """Energy norm using diagonal entries: sqrt(sum(d_i * v_i^2))."""
            return float(torch.sqrt(torch.sum(A * v * v)))

        return _energy_norm_diag

    def _energy_norm_full(v: torch.Tensor) -> float:
        """Energy norm using full matrix: sqrt(v^T A v)."""
        return float(torch.sqrt(v @ A @ v))

    return _energy_norm_full
