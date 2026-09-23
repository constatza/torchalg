"""Numerically stable primitives shared by every iterative solver.

Ported from ``neuralls.domain.solver.utils.numerics`` (see ``docs/plan.md``)
with ``numpy``/``scipy`` operations translated to their ``torch`` equivalents.
The algorithm and tolerance are unchanged from the reference; the only
deliberate deviation is that the overflow threshold below is derived from
the *actual* dtype of the tensors passed in (via ``torch.finfo``) rather
than hardcoded to ``float64`` the way the reference's call sites do — the
reference's own docstring says "dtype-specific overflow threshold" but then
always calls ``get_overflow_threshold(np.float64)`` regardless of the input
arrays' dtype, which reads as an oversight rather than a genuine float64
requirement. Deriving from the input dtype keeps the formula mathematically
identical for the ``float64`` case this repo runs by default (see
``tests/conftest.py::torch_dtype``) while behaving sensibly for other
dtypes too.

``compute_curvature``/``check_breakdown`` (a curvature-based CG breakdown
gate, ported from the reference's own dead code — see ``docs/plan.md``'s
"``ConjugateGradientSolver._iterate_step`` does not gate on
``compute_curvature``'s breakdown flag" for why it was never wired in) and
the ``breakdown_tol`` parameter that was threaded through the entire solver
API to feed it were removed entirely rather than left as unreachable dead
code: nothing in this package ever called them, and a parameter accepted by
``solve()``/``pcg()``/``flexible_cg()`` that silently does nothing is worse
than no parameter at all.

Theory:
    Numerical stability is critical in iterative solvers where rounding
    errors accumulate. This utility prevents catastrophic overflow while
    maintaining precision in normal ranges.
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


def stable_dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
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
        torch.Tensor: A 0-d tensor containing dot product ``a^T b``,
            computed stably to avoid overflow.

    Example:
        >>> import torch
        >>> stable_dot_product(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))
        tensor(11.)
    """
    max_a = a.abs().max()
    max_b = b.abs().max()

    sqrt_threshold = _overflow_scale_threshold(torch.result_type(a, b))
    target = math.sqrt(sqrt_threshold)

    max_val = torch.maximum(max_a, max_b)
    needs_scaling = max_val > sqrt_threshold

    # Guard the denominator to avoid NaN in untaken torch.where branch
    ones = torch.ones_like(max_val)
    safe_max_val = torch.where(needs_scaling, max_val, ones)
    scale = torch.where(needs_scaling, target / safe_max_val, ones)

    return torch.dot(a * scale, b * scale) / (scale * scale)
