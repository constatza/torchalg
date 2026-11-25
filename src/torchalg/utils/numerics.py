"""Numerically stable primitives shared by every iterative solver.

Ported from ``neuralls.domain.solver.utils.numerics``
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
gate, ported from the reference's own dead code, never wired in because
the reference never calls it either: it divides by the curvature unconditionally and
lets breakdown surface in ``check_solution_validity``) and
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


def is_finite(value: torch.Tensor) -> bool:
    """Return whether every entry of ``value`` is finite (one host synchronization)."""
    return bool(torch.isfinite(value).all().item())


def pair_is_finite(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return a 0-d bool tensor that is ``True`` only if two 0-d tensors are both finite.

    Cost:
        Two launches (add, isfinite) and no host synchronization: the caller
        reads the flag together with other flags in one transfer. The sum of
        two finite numbers is non-finite only when it overflows, and then this
        reports ``False`` conservatively: callers treat ``False`` as "take the
        slow, per-value path", which recomputes each value correctly, so a
        false negative costs time and never changes a result.

    Args:
        a (torch.Tensor): First 0-d tensor.
        b (torch.Tensor): Second 0-d tensor.

    Returns:
        torch.Tensor: 0-d bool tensor, ``True`` only if both ``a`` and ``b`` are finite.
    """
    return torch.isfinite(a + b)


def read_flags(first: torch.Tensor, second: torch.Tensor) -> tuple[bool, bool]:
    """Read two 0-d bool tensors with one host synchronization.

    Cost:
        Two launches (``first + 2 * second`` packs both into one integer) and
        one synchronization, against two synchronizations for two reads. On a
        GPU each synchronization drains the stream, so the packed read is the
        one that pays.

    Args:
        first (torch.Tensor): 0-d bool tensor.
        second (torch.Tensor): 0-d bool tensor.

    Returns:
        tuple[bool, bool]: ``(first, second)`` as Python bools.
    """
    packed = int((first + 2 * second).item())
    return bool(packed & 1), bool(packed & 2)


def both_finite(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return whether two 0-d tensors are both finite, with one host synchronization.

    Cost:
        ``pair_is_finite`` (two launches) plus one synchronization to read it,
        against two synchronizations for two ``is_finite`` calls.

    Args:
        a (torch.Tensor): First 0-d tensor.
        b (torch.Tensor): Second 0-d tensor.

    Returns:
        bool: ``True`` only if both ``a`` and ``b`` are finite.
    """
    return bool(pair_is_finite(a, b))


def stable_dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute a dot product with overflow prevention via balanced scaling.

    Mathematically equivalent to ``torch.dot(a, b)`` but scales both vectors
    by the same factor to prevent overflow, then adjusts the result.

    Theory:
        For ``dot(a, b)``, computing ``dot(a/s, b/s)`` gives::

            sum((a_i/s) * (b_i/s)) = sum(a_i * b_i) / s^2 = dot(a, b) / s^2

        So ``dot(a, b) = s^2 * dot(a/s, b/s)``. This is mathematically exact,
        just reordered for numerical stability.

    Cost:
        Overflow is absorbing in floating point (an infinite product or
        partial sum makes the result ``inf`` or ``nan``, never finite again),
        so a finite ``torch.dot(a, b)`` proves nothing overflowed and is the
        answer. That is the common path: one dot, one finiteness test and
        one host synchronization, with no ``n``-sized temporaries. Only a
        non-finite plain result takes the rescaled path (``max |a_i|, |b_i|``
        scaled to ``sqrt(sqrt(0.1 * finfo.max))``), which is the previous
        branch-free formulation; the two paths agree to rounding whenever
        both are finite.

        Common path, length ``n``, ``V = 8`` bytes per ``float64``: ``W = 2n``,
        ``Q = 2Vn`` (the finiteness test reads the 0-d result only, so no
        ``n``-sized pass is added), ``D = ceil(log2 n) + 1``, ``L = 4``
        (dot, ``isfinite``, ``all``, read) and ``S = 1``.
        Roofline: ``W/Q = 0.125`` flops per byte (memory-bound).
        Work-depth: ``W/D = 2n / (ceil(log2 n) + 1)`` available parallelism.
        The synchronization, not the flops, is the cost at small ``n``. The
        rescaled path adds two ``n``-sized ``abs`` passes, two maxima and a
        few scalar kernels.

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
    plain = torch.dot(a, b)
    if is_finite(plain):
        return plain
    return _scaled_dot_product(a, b)


def _scaled_dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Dot product evaluated after balanced scaling of both vectors (overflow-safe path).

    The scale is ``1`` unless ``max(|a|, |b|)`` exceeds the dtype's overflow
    scale threshold, in which case both vectors are scaled by
    ``target / max`` and the result is divided by the squared scale.
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
