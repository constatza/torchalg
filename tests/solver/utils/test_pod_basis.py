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

from torchalg.preconditioners.implementations.pod import PODCoarseningStrategy
from torchalg.sparse.preconditioners.pod.coarsening import (
    PODCoarseningStrategy as SparsePODCoarseningStrategy,
)
from torchalg.utils.pod_basis import compute_pod_basis


class TestComputePodBasis:
    def test_output_shape(self, poisson_snapshots: torch.Tensor) -> None:
        """Basis shape must be (n_dofs, rank)."""
        basis = compute_pod_basis(poisson_snapshots, rank=10)
        assert basis.shape == (20, 10)

    def test_basis_is_contiguous_and_owns_only_its_modes(
        self, poisson_snapshots: torch.Tensor
    ) -> None:
        """The basis is a compact (n_dofs, rank) array, not a view into the SVD factor.

        Prolongation and restriction multiply by it every cycle; a transposed view
        measured 1.4-3x slower there and keeps the whole ``(m, n_dofs)`` factor alive.
        """
        basis = compute_pod_basis(poisson_snapshots, rank=5)

        assert basis.is_contiguous()
        assert basis.untyped_storage().nbytes() == basis.numel() * basis.element_size()

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


class TestWeightedPodBasis:
    """``inner_product=W``: POD in the ``W``-norm via ``W = L L^T`` and ``Phi_r = L^{-T} V_r``."""

    def test_identity_weight_reproduces_euclidean_basis(
        self, poisson_snapshots: torch.Tensor
    ) -> None:
        identity = torch.eye(poisson_snapshots.shape[1], dtype=poisson_snapshots.dtype)

        weighted = compute_pod_basis(poisson_snapshots, rank=5, inner_product=identity)

        assert torch.equal(weighted, compute_pod_basis(poisson_snapshots, rank=5))

    def test_modes_are_weight_orthonormal_and_contiguous(
        self, poisson_snapshots: torch.Tensor, pod_poisson_1d: torch.Tensor
    ) -> None:
        basis = compute_pod_basis(poisson_snapshots, rank=5, inner_product=pod_poisson_1d)

        assert basis.is_contiguous()
        gram = basis.T @ pod_poisson_1d @ basis
        torch.testing.assert_close(gram, torch.eye(5, dtype=gram.dtype))

    def test_weighted_projection_error_is_the_trailing_weighted_energy(
        self, poisson_snapshots: torch.Tensor, pod_poisson_1d: torch.Tensor
    ) -> None:
        """Optimality: the W-error equals sum_{i>r} sigma_i(S L)^2 and beats the Euclidean basis."""
        weight = pod_poisson_1d
        sigma = torch.linalg.svdvals(poisson_snapshots @ torch.linalg.cholesky(weight))

        def weighted_error(basis: torch.Tensor) -> torch.Tensor:
            coarse = basis.T @ weight @ basis
            projected = basis @ torch.linalg.solve(coarse, basis.T @ weight @ poisson_snapshots.T)
            residual = poisson_snapshots.T - projected
            return torch.einsum("ij,ik,kj->", residual, weight, residual)

        weighted = compute_pod_basis(poisson_snapshots, rank=5, inner_product=weight)
        euclidean = compute_pod_basis(poisson_snapshots, rank=5)

        torch.testing.assert_close(weighted_error(weighted), sigma[5:].pow(2).sum())
        assert weighted_error(weighted) <= weighted_error(euclidean)

    def test_energy_threshold_uses_weighted_singular_values(
        self, poisson_snapshots: torch.Tensor, pod_poisson_1d: torch.Tensor
    ) -> None:
        energy = torch.linalg.svdvals(
            poisson_snapshots @ torch.linalg.cholesky(pod_poisson_1d)
        ).pow(2)
        expected = int((torch.cumsum(energy, 0) / energy.sum() < 0.99).sum()) + 1

        basis = compute_pod_basis(poisson_snapshots, rank=0.99, inner_product=pod_poisson_1d)

        assert basis.shape[1] == expected

    def test_row_scales_combine_with_weight(
        self, poisson_snapshots: torch.Tensor, pod_poisson_1d: torch.Tensor
    ) -> None:
        scales = torch.linspace(0.5, 2.0, poisson_snapshots.shape[0], dtype=torch.float64)

        combined = compute_pod_basis(
            poisson_snapshots, rank=5, row_scales=scales, inner_product=pod_poisson_1d
        )
        prescaled = compute_pod_basis(
            poisson_snapshots * scales.unsqueeze(-1), rank=5, inner_product=pod_poisson_1d
        )

        torch.testing.assert_close(combined @ combined.T, prescaled @ prescaled.T)

    def test_wrong_weight_shape_is_rejected(self, poisson_snapshots: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="inner_product must have shape"):
            compute_pod_basis(poisson_snapshots, rank=5, inner_product=torch.eye(3))

    def test_sparse_weight_is_rejected(
        self, poisson_snapshots: torch.Tensor, pod_poisson_1d: torch.Tensor
    ) -> None:
        with pytest.raises(ValueError, match="dense"):
            compute_pod_basis(
                poisson_snapshots, rank=5, inner_product=pod_poisson_1d.to_sparse_csr()
            )

    def test_asymmetric_weight_is_rejected(
        self, poisson_snapshots: torch.Tensor, pod_poisson_1d: torch.Tensor
    ) -> None:
        asymmetric = pod_poisson_1d.clone()
        asymmetric[0, 5] = 100.0

        with pytest.raises(ValueError, match="symmetric"):
            compute_pod_basis(poisson_snapshots, rank=5, inner_product=asymmetric)

    def test_indefinite_weight_raises_linalg_error(
        self, poisson_snapshots: torch.Tensor, pod_poisson_1d: torch.Tensor
    ) -> None:
        with pytest.raises(torch.linalg.LinAlgError):
            compute_pod_basis(poisson_snapshots, rank=5, inner_product=-pod_poisson_1d)

    @pytest.mark.parametrize("strategy_cls", [PODCoarseningStrategy, SparsePODCoarseningStrategy])
    def test_strategy_fit_forwards_the_weight(
        self,
        strategy_cls: type[PODCoarseningStrategy | SparsePODCoarseningStrategy],
        poisson_snapshots: torch.Tensor,
        pod_poisson_1d: torch.Tensor,
    ) -> None:
        strategy = strategy_cls(rank=5)

        strategy.fit(poisson_snapshots, inner_product=pod_poisson_1d)

        torch.testing.assert_close(
            compute_pod_basis(poisson_snapshots, rank=5, inner_product=pod_poisson_1d),
            strategy.state_dict()["_basis"],
        )
