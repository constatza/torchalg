"""Adaptive theta sampling for `TargetDimensionCoarsening`'s search.

Isolated here, separate from `TargetDimensionCoarsening` (`coarsening.py`),
per the project's helpers-isolated-in-their-own-file convention: the
branchy adaptive-refinement loop lives here, the strategy class stays
declarative orchestration.

**The problem.** `theta -> realized coarse dimension` is a step function
with plateaus of wildly varying width (a fraction of `step` to most of
`[theta_min, theta_max]`) and jumps of wildly varying size, in no
monotonic order - see `TargetDimensionCoarsening`'s docstring for the
empirical evidence and why this rules out plain bisection (which assumes
monotonicity). A fixed-step grid, linear or log, either wastes evaluations
re-sampling the flat interior of wide plateaus, or risks striding past a
narrow one entirely - silently handing back the same, coarser-than-
intended dimension for two genuinely different targets (this is what
motivated this module: see `.claude/plan.md`'s investigation notes on a
25x25 soft-inclusion grid whose true staircase has plateaus as narrow as
~0.003 in `theta` sitting between much wider ones).

**The algorithm**: adaptive boundary bisection, in two stages.

1. *Coarse pass* (`_log_spaced`): sample `theta` at the same budget a
   linear `step` grid would use, but log-spaced - denser near
   `theta_min`, where narrow plateaus empirically cluster on
   heterogeneous-contrast matrices.
2. *Refinement*: seed an explicit worklist with every adjacent pair from
   the coarse pass. Pop a pair; if its two `dim` values agree, the
   interval is (as far as this search can tell) one plateau - stop. If
   they disagree, split at the midpoint, evaluate there, and push both
   halves back onto the worklist. This is ordinary binary search
   generalized from "monotonic function, discard one half" to
   "non-monotonic step function, keep subdividing both halves until each
   one stops changing" - equivalent to a depth-first traversal of the
   implicit bisection tree, using an explicit list as the stack rather
   than the Python call stack (see *Termination* below for why that
   distinction matters). It closes the gap plain fixed-step grids leave:
   a plateau narrower than `step` but sandwiched between two disagreeing
   coarse points gets discovered regardless, because refinement continues
   past `step` down to `_RESOLUTION_FLOOR`, not stopping at `step` itself.

The one case no finite-budget grid can catch is a plateau strictly
narrower than the coarse pass's spacing whose neighbors happen to agree
on *both* sides (a "spike" invisible to every sample taken) - detecting
that would require exhaustive fine sampling of the whole domain.

**Termination.** The worklist loop is iterative (a `while` over a plain
list), not Python recursion, so it cannot raise `RecursionError` or grow
the call stack - the concern for a step function with many true
boundaries is unbounded *work* (evaluations, and the `samples` dict they
accumulate in), not unbounded call depth. Every interval on the worklist
provably shrinks: each split replaces one interval with two, each exactly
half the parent's width, and a popped interval is dropped for good once
its width is at or below `_RESOLUTION_FLOOR` or its endpoints agree - so
along any single branch, depth is bounded by
``log2((theta_max - theta_min) / _RESOLUTION_FLOOR)`` (~20 for a unit
range), which already rules out non-terminating recursion. The real risk
is *breadth*: a pathological `evaluate` that disagrees with its neighbor
at every scale (no plateau ever found) forces every branch to refine all
the way to the floor, and the total node count across all branches is
``O((theta_max - theta_min) / _RESOLUTION_FLOOR)`` - unbounded work, not
unbounded depth, but still a real cost/memory ceiling worth capping
directly. `max_total_samples` is that cap: refinement simply stops
taking new samples once it's hit, keeping both `evaluate` call count and
the `samples` dict's size bounded regardless of how adversarial the
function turns out to be. `max_coarse_samples` caps the *other* input to
that cost - a `step` far smaller than ``theta_max - theta_min`` - so a
single misconfigured call can't build an oversized coarse pass before
refinement even starts. Both are keyword parameters (defaulting to
`_MAX_COARSE_SAMPLES`/`_MAX_TOTAL_SAMPLES`) rather than baked-in
constants: they bound this function's own cost, not
`TargetDimensionCoarsening`'s public contract, so they live on the
low-level search primitive a caller (including this module's own tests)
can override, instead of on the higher-level class that has no reason to
know about them.
"""

from __future__ import annotations

import math
from collections.abc import Callable

_MIN_COARSE_SAMPLES = 2
"""Floor on the coarse pass's sample count, so a single-point range is
still a valid interval to (trivially) bisect."""

_MAX_COARSE_SAMPLES = 1_000
"""Ceiling on the coarse pass's sample count, independent of how small
`step` is relative to `theta_max - theta_min` - bounds the fixed cost
paid before refinement even starts."""

_RESOLUTION_FLOOR = 1e-6
"""Refinement stops once a disagreeing interval is narrower than this,
independent of `step` - see the module docstring. `theta` lives in
(0, 1), so this is far finer than any physically meaningful threshold
while still bounding each true boundary to ~20 extra evaluations."""

_MAX_TOTAL_SAMPLES = 5_000
"""Hard ceiling on `len(samples)` (coarse pass + all refinement),
regardless of how many boundaries a pathological `evaluate` appears to
have - see the module docstring's *Termination* section. Refinement
degrades gracefully past this point: it just stops taking new samples and
returns the best information gathered so far, rather than raising."""


def adaptive_theta_scan(
    theta_min: float,
    theta_max: float,
    step: float,
    evaluate: Callable[[float], int],
    *,
    max_coarse_samples: int = _MAX_COARSE_SAMPLES,
    max_total_samples: int = _MAX_TOTAL_SAMPLES,
) -> list[tuple[float, int]]:
    """Sample a `theta -> dim` step function: log-spaced coarse pass, then bisect jumps.

    See the module docstring for the algorithm and its termination
    argument. Bounded by construction: at most `max_coarse_samples`
    coarse samples and `max_total_samples` samples overall, so `evaluate`
    is called a finite, predictable number of times no matter how
    pathological it is.

    Args:
        theta_min (float): Lower bound of the search range.
        theta_max (float): Upper bound of the search range.
        step (float): Coarse-pass spacing budget (linear fallback if
            ``theta_min <= 0``). Disagreeing intervals are always refined
            past this down to `_RESOLUTION_FLOOR`, regardless of `step`.
        evaluate (Callable[[float], int]): Maps a `theta` to its realized
            coarse dimension.
        max_coarse_samples (int): Ceiling on the coarse pass's sample
            count, independent of how small `step` is relative to
            ``theta_max - theta_min``. Defaults to `_MAX_COARSE_SAMPLES`;
            override only if a caller has a demonstrated need for a
            larger (or tighter, e.g. for a fast deterministic test)
            budget than the default safety net.
        max_total_samples (int): Hard ceiling on samples taken overall
            (coarse pass + refinement), regardless of how many boundaries
            a pathological `evaluate` appears to have. Defaults to
            `_MAX_TOTAL_SAMPLES`; same override guidance as
            `max_coarse_samples`.

    Returns:
        list[tuple[float, int]]: ``(theta, dim)`` samples taken, sorted by
            `theta`, deduplicated.
    """
    n_coarse = min(
        max(round((theta_max - theta_min) / step) + 1, _MIN_COARSE_SAMPLES),
        max_coarse_samples,
    )
    coarse_thetas = (
        _log_spaced(theta_min, theta_max, n_coarse)
        if theta_min > 0
        else [theta_min + i * step for i in range(n_coarse)]
    )

    samples: dict[float, int] = {theta: evaluate(theta) for theta in coarse_thetas}
    frontier = list(zip(sorted(samples), sorted(samples)[1:]))
    while frontier and len(samples) < max_total_samples:
        left, right = frontier.pop()
        if right - left <= _RESOLUTION_FLOOR or samples[left] == samples[right]:
            continue
        mid = (left + right) / 2
        if mid <= left or mid >= right:
            continue  # float precision exhausted
        samples[mid] = evaluate(mid)
        frontier.append((left, mid))
        frontier.append((mid, right))

    return sorted(samples.items())


def _log_spaced(low: float, high: float, count: int) -> list[float]:
    """Build `count` log-uniformly spaced points in `[low, high]`.

    Args:
        low (float): Range lower bound, must be > 0.
        high (float): Range upper bound.
        count (int): Number of points, >= 2.

    Returns:
        list[float]: Log-uniformly spaced points, ascending, endpoints included.
    """
    log_low, log_high = math.log(low), math.log(high)
    step = (log_high - log_low) / (count - 1)
    return [math.exp(log_low + i * step) for i in range(count)]
