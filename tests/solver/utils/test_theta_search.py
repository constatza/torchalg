"""Tests for ``torchalg.utils.theta_search.adaptive_theta_scan``.

Moved out of ``tests/solver/preconditioners/implementations/test_amg.py``
when `adaptive_theta_scan` was promoted from
``torchalg.preconditioners.implementations.amg._theta_search`` to
``torchalg.utils.theta_search`` (a dependency-free leaf shared by both the
dense and sparse `TargetDimensionCoarsening` siblings) - mirrors
``test_spectral.py``'s location for the same reason
``torchalg.utils.spectral`` is tested there.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from torchalg.utils.theta_search import adaptive_theta_scan

# ---------------------------------------------------------------------------
# adaptive_theta_scan
# ---------------------------------------------------------------------------


class TestAdaptiveThetaScan:
    """Exercises the pure step-function sampler against synthetic `theta -> dim`
    functions, independent of any real AMG matrix, so the refinement behavior
    itself is pinned down deterministically.
    """

    def test_finds_narrow_plateau_sandwiched_between_disagreeing_coarse_points(self) -> None:
        """A plateau narrower than the coarse spacing is still found via bisection.

        ``dim`` is 100 on [0, 0.301), 55 on [0.301, 0.305) (width 0.004 - far
        narrower than the coarse step of 0.1), and 10 on [0.305, 1]. A plain
        linear grid at step=0.1 samples 0.3 and 0.4 (100 and 10) and never
        sees 55 at all; since those two coarse points disagree, adaptive
        refinement must bisect between them and discover the narrow plateau.
        """

        def step_function(theta: float) -> int:
            if theta < 0.301:
                return 100
            if theta < 0.305:
                return 55
            return 10

        samples = adaptive_theta_scan(
            theta_min=0.001, theta_max=1.0, step=0.1, evaluate=step_function
        )
        dims = {dim for _, dim in samples}
        assert 55 in dims, (
            "narrow plateau sandwiched between two differing coarse points was missed"
        )

    def test_refines_disagreeing_boundary_far_finer_than_step(self) -> None:
        """A true boundary between two coarse points is pinned down past `step`, not just to it.

        `step` only sizes the coarse pass (see the module docstring) - once
        two points disagree, refinement continues to `_RESOLUTION_FLOOR`
        regardless of `step`, which is what lets a plateau narrower than
        `step` (see the sandwiched-plateau test above) get discovered.
        """

        def step_function(theta: float) -> int:
            return 0 if theta < 0.30000005 else 1

        step = 0.2
        samples = adaptive_theta_scan(
            theta_min=0.001, theta_max=1.0, step=step, evaluate=step_function
        )
        thetas = [theta for theta, _ in samples]
        boundary_gaps = [
            right - left for left, right in pairwise(thetas) if left < 0.30000005 <= right
        ]
        assert boundary_gaps, "no sample straddled the true boundary"
        assert min(boundary_gaps) < step * 0.01, (
            f"boundary resolved only to {min(boundary_gaps)}, expected far finer than step={step}"
        )

    def test_coarse_pass_is_log_spaced_when_theta_min_positive(self) -> None:
        """Consecutive gaps must grow as theta grows (denser sampling near theta_min)."""
        samples = adaptive_theta_scan(theta_min=1e-4, theta_max=1.0, step=0.2, evaluate=lambda _: 0)
        thetas = [theta for theta, _ in samples]
        assert thetas[1] - thetas[0] < thetas[-1] - thetas[-2]

    def test_falls_back_to_linear_spacing_when_theta_min_is_zero(self) -> None:
        """theta_min=0 would make log-spacing undefined (log(0)); must not raise."""
        samples = adaptive_theta_scan(theta_min=0.0, theta_max=1.0, step=0.25, evaluate=lambda _: 0)
        thetas = [theta for theta, _ in samples]
        assert thetas[0] == 0.0
        assert thetas[-1] == pytest.approx(1.0)

    def test_pathological_evaluate_is_bounded_not_unbounded(self) -> None:
        """A function that disagrees with its neighbor at every scale must not blow up.

        No plateau ever appears, so every branch would refine all the way
        to `_RESOLUTION_FLOOR` without `max_total_samples` - this pins
        down that the cap actually stops it, bounding both the number of
        `evaluate` calls and the size of the returned sample set. Passed
        explicitly (small) rather than relying on the production default,
        so the test is fast and its intent - "the cap works, regardless
        of its value" - doesn't depend on the default staying 5000.
        """
        call_count = 0

        def adversarial(theta: float) -> int:
            nonlocal call_count
            call_count += 1
            return int(theta * 1e12) % 2  # flips on every finer bisection

        cap = 200
        samples = adaptive_theta_scan(
            theta_min=1e-6, theta_max=1.0, step=0.5, evaluate=adversarial, max_total_samples=cap
        )

        assert len(samples) <= cap
        assert call_count <= cap

    def test_max_coarse_samples_bounds_the_coarse_pass_regardless_of_step(self) -> None:
        """A degenerate `step` must not build an oversized coarse pass before refinement runs.

        `step=1e-6` over a unit range would ask for ~1e6 coarse points
        without `max_coarse_samples` - this confirms the cap on that
        *first* stage, distinct from `max_total_samples`'s cap on
        refinement.
        """
        coarse_cap = 10
        samples = adaptive_theta_scan(
            theta_min=0.001,
            theta_max=1.0,
            step=1e-6,
            evaluate=lambda _: (
                0
            ),  # every point agrees: no refinement to conflate with the coarse cap
            max_coarse_samples=coarse_cap,
        )
        assert len(samples) == coarse_cap

    def test_samples_are_sorted_and_deduplicated(self) -> None:
        samples = adaptive_theta_scan(
            theta_min=0.01, theta_max=1.0, step=0.1, evaluate=lambda theta: int(theta > 0.5)
        )
        thetas = [theta for theta, _ in samples]
        assert thetas == sorted(thetas)
        assert len(thetas) == len(set(thetas))
