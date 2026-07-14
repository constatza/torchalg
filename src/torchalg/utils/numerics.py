"""Numerically stable primitives shared by every iterative solver.

Ported from ``neuralls.domain.solver.utils.numerics`` (see ``docs/plan.md``)
with ``numpy``/``scipy`` operations translated to their ``torch`` equivalents.
The algorithms and tolerances are unchanged from the reference; the only
deliberate deviation is that the overflow/breakdown thresholds below are
derived from the *actual* dtype of the tensors passed in (via
``torch.finfo``) rather than hardcoded to ``float64`` the way the reference's
call sites do — the reference's own docstring says "dtype-specific overflow
threshold" but then always calls ``get_overflow_threshold(np.float64)``
regardless of the input arrays' dtype, which reads as an oversight rather
than a genuine float64 requirement. Deriving from the input dtype keeps the
formulas mathematically identical for the ``float64`` case this repo runs by
default (see ``tests/conftest.py::torch_dtype``) while behaving sensibly for
other dtypes too.

Theory:
    Numerical stability is critical in iterative solvers where rounding
    errors accumulate. These utilities prevent catastrophic overflow/
    underflow while maintaining precision in normal ranges.
"""

from __future__ import annotations

import math

import torch


def _overflow_scale_threshold(dtype: torch.dtype) -> float:
    """Compute the magnitude past which ``stable_dot_product`` rescales.

    Mirrors ``neuralls.shared.constants.get_overflow_threshold``: uses
    ``0.1 * finfo(dtype).max`` as a practical overflow ceiling (10x safety
    margin for intermediate calculations), then returns its square root —
    the actual per-element magnitude at which a product ``a_i * b_i`` risks
    overflowing before balanced scaling is applied.

    Args:
        dtype (torch.dtype): Floating dtype to derive the threshold from.

    Returns:
        float: ``sqrt(0.1 * finfo(dtype).max)``, approximately ``1.3e154``
            for ``torch.float64``.
    """
    overflow_threshold = 0.1 * torch.finfo(dtype).max
    return math.sqrt(overflow_threshold)


def _breakdown_abs_tol(dtype: torch.dtype) -> float:
    """Compute the SciPy-convention absolute breakdown tolerance for a dtype.

    Mirrors ``neuralls.shared.constants.get_breakdown_tol``: SciPy's
    BICG/BICGSTAB/CGS solvers use ``eps**2`` as the vanishing-denominator
    threshold, since errors in squared quantities (norms, curvature) accumulate
    quadratically.

    Args:
        dtype (torch.dtype): Floating dtype to derive machine epsilon from.

    Returns:
        float: ``finfo(dtype).eps ** 2``, approximately ``4.93e-32`` for
            ``torch.float64``.
    """
    return torch.finfo(dtype).eps ** 2


def stable_dot_product(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute a dot product with overflow prevention via balanced scaling.

    Mathematically equivalent to ``torch.dot(a, b)`` but scales both vectors
    by the same factor to prevent overflow, then adjusts the result.

    Theory:
        For ``dot(a, b)``, computing ``dot(a/s, b/s)`` gives::

            sum((a_i/s) * (b_i/s)) = sum(a_i * b_i) / s^2 = dot(a, b) / s^2

        So ``dot(a, b) = s^2 * dot(a/s, b/s)``. This is mathematically exact,
        just reordered for numerical stability.

    Args:
        a (torch.Tensor): First vector for dot product, shape ``(n,)``.
        b (torch.Tensor): Second vector for dot product, shape ``(n,)``.

    Returns:
        float: Dot product ``a^T b``, computed stably to avoid overflow.

    Example:
        >>> import torch
        >>> stable_dot_product(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        11.0
    """
    max_a = a.abs().max()
    max_b = b.abs().max()

    sqrt_threshold = _overflow_scale_threshold(torch.result_type(a, b))

    if max_a > sqrt_threshold or max_b > sqrt_threshold:
        # At least one value is large - need to scale both vectors by the
        # same factor to keep the product safe:
        # dot(a, b) = (1 / scale^2) * dot(a * scale, b * scale)
        max_val = torch.maximum(max_a, max_b)
        target = math.sqrt(sqrt_threshold)
        scale = target / max_val
        return float(torch.dot(a * scale, b * scale) / (scale * scale))

    return float(torch.dot(a, b))


def compute_curvature(
    p: torch.Tensor,
    q: torch.Tensor,
    *,
    breakdown_tol: float = 1e-14,
) -> tuple[float, bool]:
    """Compute curvature ``d = p^T A p`` with breakdown detection.

    Args:
        p (torch.Tensor): Search direction vector.
        q (torch.Tensor): Matrix-vector product ``q = A @ p``.
        breakdown_tol (float): Relative threshold multiplier for breakdown
            detection (default: ``1e-14``).

    Returns:
        tuple[float, bool]: ``(curvature, breakdown)`` where ``curvature`` is
            the scalar ``d = p^T q`` and ``breakdown`` is ``True`` if
            breakdown was detected (``d`` too small, non-finite, or negative).

    Theory:
        The curvature ``d = p^T A p = p^T q`` must be positive for descent.
        Negative or tiny ``d`` signals numerical issues or non-SPD behavior.

        Breakdown conditions:
        - ``d <= 0``: non-positive definite (restart needed).
        - ``|d| < breakdown_tol * ||p||^2``: too small (numerical precision
          loss).

    Example:
        >>> import torch
        >>> p = torch.tensor([1.0, 2.0, 3.0])
        >>> q = torch.tensor([2.0, 4.0, 6.0])
        >>> d, breakdown = compute_curvature(p, q)
        >>> breakdown
        False
    """
    # Use stable dot product to prevent overflow.
    d = stable_dot_product(p, q)

    if not math.isfinite(d):
        return d, True

    abs_threshold = _breakdown_abs_tol(torch.result_type(p, q))
    if abs(d) < abs_threshold:
        return d, True

    p_norm_sq = float(torch.linalg.norm(p)) ** 2
    rel_threshold = breakdown_tol * p_norm_sq

    if d <= 0 or d < rel_threshold:
        return d, True

    return d, False


def check_breakdown(value: float, threshold: float = 1e-14) -> bool:
    """Check if a scalar value indicates numerical breakdown.

    Args:
        value (float): Value to check.
        threshold (float): Absolute threshold for breakdown detection
            (default: ``1e-14``).

    Returns:
        bool: ``True`` if breakdown detected (NaN, Inf, or too small).

    Example:
        >>> check_breakdown(1e-20)
        True
        >>> check_breakdown(1.0)
        False
    """
    return not math.isfinite(value) or abs(value) < threshold
