"""Tests for ``torchalg.utils.pod_basis.compute_pod_basis``.

Promoted alongside the source module out of
``tests/solver/preconditioners/implementations/test_pod.py`` (see
``docs/plan.md``): ``compute_pod_basis`` is a dependency-free leaf that
never touches the fine-grid matrix, so its tests live here, mirroring
``tests/solver/utils/test_spectral.py``'s convention for the other promoted
leaf (``torchalg.utils.spectral``), rather than under the preconditioners
test tree.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.utils.pod_basis import compute_pod_basis


class TestComputePodBasis:
    def test_output_shape(self, poisson_snapshots: torch.Tensor) -> None:
        """Basis shape must be (n_dofs, rank)."""
        basis = compute_pod_basis(poisson_snapshots, rank=10)
        assert basis.shape == (20, 10)

    def test_columns_orthonormal(self, pod_basis: torch.Tensor) -> None:
        """Phi_r^T Phi_r must equal the identity (snapshot-method POD property)."""
        gram = pod_basis.T @ pod_basis
        torch.testing.assert_close(gram, torch.eye(10, dtype=pod_basis.dtype), atol=1e-8, rtol=0.0)

    def test_rank_exceeds_samples_raises(self, poisson_snapshots: torch.Tensor) -> None:
        """rank > n_samples must raise ValueError."""
        with pytest.raises(ValueError, match="rank"):
            compute_pod_basis(poisson_snapshots, rank=poisson_snapshots.shape[0] + 1)

    def test_dtype_preserved_float32(self, poisson_snapshots: torch.Tensor) -> None:
        """float32 input must produce a float32 basis (no silent upcast)."""
        basis = compute_pod_basis(poisson_snapshots.to(torch.float32), rank=10)
        assert basis.dtype == torch.float32

    def test_dtype_preserved_float64(self, poisson_snapshots: torch.Tensor) -> None:
        """float64 input must produce a float64 basis (no silent downcast)."""
        basis = compute_pod_basis(poisson_snapshots.to(torch.float64), rank=10)
        assert basis.dtype == torch.float64

    def test_explicit_dtype_override(self, poisson_snapshots: torch.Tensor) -> None:
        """An explicit dtype argument must override the input tensor's dtype."""
        basis = compute_pod_basis(poisson_snapshots.to(torch.float64), rank=10, dtype=torch.float32)
        assert basis.dtype == torch.float32

    def test_energy_threshold_selects_fewer_modes_than_full_rank(
        self, poisson_snapshots: torch.Tensor
    ) -> None:
        """A energy threshold < 1.0 must retain strictly fewer modes than the full rank."""
        basis = compute_pod_basis(poisson_snapshots, rank=0.999)
        assert 0 < basis.shape[1] < poisson_snapshots.shape[0]

    def test_energy_threshold_one_retains_full_rank(self, poisson_snapshots: torch.Tensor) -> None:
        """rank=1.0 (100% energy) must retain min(n_samples, n_dofs) modes."""
        basis = compute_pod_basis(poisson_snapshots, rank=1.0)
        assert basis.shape[1] == min(poisson_snapshots.shape)

    def test_energy_threshold_out_of_range_raises(self, poisson_snapshots: torch.Tensor) -> None:
        """An energy threshold outside (0, 1] must raise ValueError."""
        with pytest.raises(ValueError, match="energy threshold"):
            compute_pod_basis(poisson_snapshots, rank=1.5)

    def test_row_scales_none_matches_unweighted_basis(
        self, poisson_snapshots: torch.Tensor
    ) -> None:
        """``row_scales=None`` must reproduce today's exact unweighted basis."""
        weighted = compute_pod_basis(poisson_snapshots, rank=10, row_scales=None)
        unweighted = compute_pod_basis(poisson_snapshots, rank=10)
        torch.testing.assert_close(weighted, unweighted)

    def test_uniform_row_scales_leave_basis_unchanged(
        self, poisson_snapshots: torch.Tensor
    ) -> None:
        """A constant row scale rescales the covariance uniformly - same basis, up to sign."""
        uniform = torch.full((poisson_snapshots.shape[0],), 3.0, dtype=poisson_snapshots.dtype)
        weighted = compute_pod_basis(poisson_snapshots, rank=10, row_scales=uniform)
        unweighted = compute_pod_basis(poisson_snapshots, rank=10)
        torch.testing.assert_close(weighted.abs(), unweighted.abs(), atol=1e-6, rtol=1e-6)

    def test_row_scales_still_orthonormal(
        self, poisson_snapshots: torch.Tensor, snapshot_row_scales: torch.Tensor
    ) -> None:
        """Weighted-covariance POD must still return an orthonormal basis."""
        basis = compute_pod_basis(poisson_snapshots, rank=10, row_scales=snapshot_row_scales)
        gram = basis.T @ basis
        torch.testing.assert_close(gram, torch.eye(10, dtype=basis.dtype), atol=1e-6, rtol=0.0)

    def test_row_scales_reweight_the_svd(
        self, poisson_snapshots: torch.Tensor, single_dominant_row_scales: torch.Tensor
    ) -> None:
        """Concentrating weight on one snapshot must align the leading mode with it.

        Directly exercises the write-up's central claim: row scaling changes
        *which* directions the SVD favors (unlike normalizing a snapshot's own
        magnitude, which does not preferentially amplify any component).
        """
        basis = compute_pod_basis(poisson_snapshots, rank=1, row_scales=single_dominant_row_scales)
        dominant_direction = poisson_snapshots[0] / poisson_snapshots[0].norm()
        cosine = (basis[:, 0] @ dominant_direction).abs()
        assert cosine > 0.999

    def test_row_scales_shape_mismatch_raises(self, poisson_snapshots: torch.Tensor) -> None:
        """``row_scales`` with the wrong length must raise ValueError, not broadcast silently."""
        wrong_length = torch.ones(poisson_snapshots.shape[0] + 1, dtype=poisson_snapshots.dtype)
        with pytest.raises(ValueError, match="row_scales"):
            compute_pod_basis(poisson_snapshots, rank=10, row_scales=wrong_length)
