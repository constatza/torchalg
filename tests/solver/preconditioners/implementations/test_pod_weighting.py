"""Tests for POD snapshot row-weighting helpers.

These implement the "how much does each snapshot vote in the SVD" axis from
the POD-2G snapshot-metrics design (``dl-experiments``'s
``docs/plan.md``/PR description for this feature): row-scaling a snapshot
before an ordinary Euclidean SVD reproduces a weighted covariance
``sum_k scale_k^2 e_k e_k^T`` without changing the SVD's inner product, so
these functions only ever compute a length-``n_samples`` scale vector -
``compute_pod_basis``'s ``row_scales`` parameter (see ``test_pod.py``) is
what actually applies it.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations.amg.smoothers import JacobiSmoother
from torchalg.preconditioners.implementations.pod.weighting import (
    apply_jacobi_damping,
    apply_jacobi_damping_trajectory,
    energy_row_norms,
    l2_row_norms,
    power_norm_scales,
    smoother_persistence_scales,
)

# ---------------------------------------------------------------------------
# l2_row_norms / energy_row_norms
# ---------------------------------------------------------------------------


class TestL2RowNorms:
    def test_matches_manual_norm(self, poisson_snapshots: torch.Tensor) -> None:
        """Must match ``torch.linalg.vector_norm`` per row."""
        norms = l2_row_norms(poisson_snapshots)
        expected = torch.linalg.vector_norm(poisson_snapshots, dim=1)
        torch.testing.assert_close(norms, expected)

    def test_zero_row_gives_zero_norm(self, snapshots_with_zero_row: torch.Tensor) -> None:
        """A zero snapshot has zero L2 norm - no clamping at this layer."""
        norms = l2_row_norms(snapshots_with_zero_row)
        assert norms[0].item() == 0.0


class TestEnergyRowNorms:
    def test_matches_manual_quadratic_form(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """Must match ``sqrt(e^T A e)`` computed row-by-row."""
        norms = energy_row_norms(poisson_snapshots, poisson_1d)
        expected = torch.tensor(
            [torch.sqrt(row @ poisson_1d @ row).item() for row in poisson_snapshots],
            dtype=poisson_snapshots.dtype,
        )
        torch.testing.assert_close(norms, expected, atol=1e-6, rtol=1e-6)

    def test_nonnegative_for_spd_matrix(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """SPD ``A`` must give strictly non-negative energy norms."""
        norms = energy_row_norms(poisson_snapshots, poisson_1d)
        assert torch.all(norms >= 0.0)


# ---------------------------------------------------------------------------
# power_norm_scales
# ---------------------------------------------------------------------------


class TestPowerNormScales:
    def test_beta_zero_is_raw_passthrough(self, poisson_snapshots: torch.Tensor) -> None:
        """``beta=0`` must leave every snapshot with scale 1 (no reweighting)."""
        scales = power_norm_scales(poisson_snapshots, beta=0.0)
        torch.testing.assert_close(scales, torch.ones_like(scales))

    def test_beta_one_l2_gives_full_normalization(self, poisson_snapshots: torch.Tensor) -> None:
        """``metric='l2', beta=1`` scale must equal ``1/||e_k||_2``."""
        scales = power_norm_scales(poisson_snapshots, metric="l2", beta=1.0)
        expected = 1.0 / l2_row_norms(poisson_snapshots)
        torch.testing.assert_close(scales, expected)

    def test_beta_one_energy_gives_energy_normalization(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """``metric='energy', beta=1`` scale must equal ``1/||e_k||_A``."""
        scales = power_norm_scales(poisson_snapshots, matrix=poisson_1d, metric="energy", beta=1.0)
        expected = 1.0 / energy_row_norms(poisson_snapshots, poisson_1d)
        torch.testing.assert_close(scales, expected, atol=1e-6, rtol=1e-6)

    def test_partial_beta_interpolates_between_raw_and_normalized(
        self, poisson_snapshots: torch.Tensor
    ) -> None:
        """``0 < beta < 1`` must sit strictly between the raw and fully-normalized scale."""
        raw = power_norm_scales(poisson_snapshots, beta=0.0)
        full = power_norm_scales(poisson_snapshots, beta=1.0)
        partial = power_norm_scales(poisson_snapshots, beta=0.5)
        norms = l2_row_norms(poisson_snapshots)
        # Every snapshot here has norm > 1, so scale strictly decreases as beta grows.
        assert torch.all(norms > 1.0)
        assert torch.all(partial < raw)
        assert torch.all(partial > full)

    def test_metric_energy_without_matrix_raises(self, poisson_snapshots: torch.Tensor) -> None:
        """``metric='energy'`` needs the system matrix - omitting it is a usage error."""
        with pytest.raises(ValueError, match="matrix"):
            power_norm_scales(poisson_snapshots, metric="energy", beta=1.0)

    def test_zero_row_does_not_produce_inf(self, snapshots_with_zero_row: torch.Tensor) -> None:
        """A zero snapshot must be clamped, not divide-by-zero into inf/nan."""
        scales = power_norm_scales(snapshots_with_zero_row, beta=1.0)
        assert torch.isfinite(scales).all()


# ---------------------------------------------------------------------------
# apply_jacobi_damping
# ---------------------------------------------------------------------------


class TestApplyJacobiDamping:
    def test_batched_matches_single_vector_smoother(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """Batched damping must match calling ``JacobiSmoother.smooth`` per row with rhs=0."""
        omega, steps = 0.67, 5
        damped = apply_jacobi_damping(poisson_snapshots, poisson_1d, omega=omega, steps=steps)
        smoother = JacobiSmoother(omega=omega)
        expected = torch.stack(
            [
                smoother.smooth(poisson_1d, torch.zeros_like(row), row, steps=steps)
                for row in poisson_snapshots
            ]
        )
        torch.testing.assert_close(damped, expected, atol=1e-6, rtol=1e-6)

    def test_output_shape_matches_input(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """Damping must not change the snapshot ensemble's shape."""
        damped = apply_jacobi_damping(poisson_snapshots, poisson_1d, omega=0.67, steps=5)
        assert damped.shape == poisson_snapshots.shape

    def test_zero_steps_is_identity(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """``steps=0`` must leave every snapshot unchanged."""
        damped = apply_jacobi_damping(poisson_snapshots, poisson_1d, omega=0.67, steps=0)
        torch.testing.assert_close(damped, poisson_snapshots)


# ---------------------------------------------------------------------------
# apply_jacobi_damping_trajectory
# ---------------------------------------------------------------------------


class TestApplyJacobiDampingTrajectory:
    def test_shape_is_steps_plus_one(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """Trajectory must have exactly steps + 1 rows per vector - no more, no fewer."""
        steps = 5
        trajectory = apply_jacobi_damping_trajectory(
            poisson_snapshots, poisson_1d, omega=0.67, steps=steps
        )
        assert trajectory.shape == (
            poisson_snapshots.shape[0],
            steps + 1,
            poisson_snapshots.shape[1],
        )

    def test_shape_tracks_steps_exactly(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """Increasing steps must grow the trajectory by exactly one row per step - proof no
        extra sweeps are silently run or dropped."""
        for steps in (0, 1, 3, 7):
            trajectory = apply_jacobi_damping_trajectory(
                poisson_snapshots, poisson_1d, omega=0.67, steps=steps
            )
            assert trajectory.shape[1] == steps + 1

    def test_first_row_is_untouched_input(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """Index 0 must be the input vector before any sweep is applied."""
        trajectory = apply_jacobi_damping_trajectory(
            poisson_snapshots, poisson_1d, omega=0.67, steps=5
        )
        torch.testing.assert_close(trajectory[:, 0, :], poisson_snapshots)

    def test_last_row_matches_apply_jacobi_damping(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """Row `steps` must equal apply_jacobi_damping's own final-iterate output exactly -
        the two functions must agree on what "steps sweeps" means."""
        omega, steps = 0.67, 6
        trajectory = apply_jacobi_damping_trajectory(
            poisson_snapshots, poisson_1d, omega=omega, steps=steps
        )
        final_only = apply_jacobi_damping(poisson_snapshots, poisson_1d, omega=omega, steps=steps)
        torch.testing.assert_close(trajectory[:, -1, :], final_only)

    def test_intermediate_row_matches_fewer_steps(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """Row k of a `steps`-sweep trajectory must equal a standalone k-sweep damping call -
        the trajectory isn't just bracketed correctly, every row is the right vector."""
        omega = 0.67
        trajectory = apply_jacobi_damping_trajectory(
            poisson_snapshots, poisson_1d, omega=omega, steps=6
        )
        for k in (2, 4):
            expected = apply_jacobi_damping(poisson_snapshots, poisson_1d, omega=omega, steps=k)
            torch.testing.assert_close(trajectory[:, k, :], expected, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# smoother_persistence_scales
# ---------------------------------------------------------------------------


class TestSmootherPersistenceScales:
    def test_smooth_mode_persists_more_than_oscillatory_mode(
        self, poisson_1d: torch.Tensor
    ) -> None:
        """The lowest-frequency mode must survive Jacobi damping better than the highest-frequency one.

        This is the write-up's central claim for this weighting scheme: the
        smoother eliminates high-frequency error quickly, so a
        smoother-resistant (low-frequency, "algebraically smooth") direction
        should retain a much larger fraction of its norm than a
        high-frequency one after the same number of sweeps.
        """
        n = poisson_1d.shape[0]
        positions = torch.arange(1, n + 1, dtype=poisson_1d.dtype)
        smooth_mode = torch.sin(torch.pi * positions / (n + 1))
        oscillatory_mode = torch.sin(torch.pi * n * positions / (n + 1))
        snapshots = torch.stack([smooth_mode, oscillatory_mode])

        scales = smoother_persistence_scales(snapshots, poisson_1d, omega=0.67, steps=10)

        assert scales[0] > scales[1]

    def test_scale_is_bounded_in_zero_one_for_spd_matrix(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """Damping can only remove energy, never add it, so scale must stay in [0, 1]."""
        scales = smoother_persistence_scales(poisson_snapshots, poisson_1d, omega=0.67, steps=5)
        assert torch.all(scales >= 0.0)
        assert torch.all(scales <= 1.0 + 1e-6)

    def test_zero_row_does_not_produce_nan(
        self, poisson_1d: torch.Tensor, snapshots_with_zero_row: torch.Tensor
    ) -> None:
        """A zero snapshot must be clamped, not divide-by-zero into nan."""
        scales = smoother_persistence_scales(
            snapshots_with_zero_row, poisson_1d, omega=0.67, steps=5
        )
        assert torch.isfinite(scales).all()
