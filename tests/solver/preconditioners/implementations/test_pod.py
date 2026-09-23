"""Tests for POD-2G preconditioner components.

Ported from the reference (``dl-experiments``'s
``tests/solver/preconditioners/implementations/test_pod.py``), adapted from
numpy to torch tensors, using the shared ``poisson_1d``/``poisson_rhs``/
``poisson_snapshots``/``pod_basis`` fixtures
(``tests/solver/preconditioners/conftest.py``) instead of inline
construction.

Deviations from the reference:
    - ``TestDenseTransferOperator`` imports ``DenseTransferOperator`` from
      ``preconditioners.implementations.amg`` rather than
      ``preconditioners.implementations.pod``: the POD package does not
      define its own transfer-operator class (see
      ``preconditioners.implementations.pod``'s module docstring) - the POD
      basis is passed directly into the same dense P/R wrapper AMG uses.
    - ``TestPOD2GWithFCGSolver`` is active now that
      ``torchalg.flexible_cg`` has landed.
    - ``TestPOD2GConfigParsing``/``TestPOD2GFactory`` (TOML/Pydantic config
      parsing and the composition-layer factory) are not ported: both are
      part of the explicit scope trim documented in ``docs/plan.md``'s
      "Scope trim" section (the TOML-driven preconditioner factory is not
      part of this migration).
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations.amg import DenseTransferOperator, MultigridSmoother
from torchalg.preconditioners.implementations.pod import (
    POD2GPreconditioner,
    PODCoarseningStrategy,
    compute_pod_basis,
)

# ---------------------------------------------------------------------------
# compute_pod_basis
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# DenseTransferOperator (reused from amg, backed by a POD basis)
# ---------------------------------------------------------------------------


class TestDenseTransferOperator:
    def test_prolongate_shape(self, pod_basis: torch.Tensor) -> None:
        """Prolongate maps coarse (rank) -> fine (n_dofs)."""
        transfer = DenseTransferOperator(pod_basis)
        fine = transfer.prolongate(torch.ones(10, dtype=pod_basis.dtype))
        assert fine.shape == (20,)

    def test_restrict_shape(self, pod_basis: torch.Tensor) -> None:
        """Restrict maps fine (n_dofs) -> coarse (rank)."""
        transfer = DenseTransferOperator(pod_basis)
        coarse = transfer.restrict(torch.ones(20, dtype=pod_basis.dtype))
        assert coarse.shape == (10,)

    def test_restrict_prolongate_consistency(self, pod_basis: torch.Tensor) -> None:
        """restrict(prolongate(x)) == x since Phi_r^T Phi_r = I for orthonormal Phi_r."""
        transfer = DenseTransferOperator(pod_basis)
        x = torch.arange(10, dtype=pod_basis.dtype)
        roundtrip = transfer.restrict(transfer.prolongate(x))
        torch.testing.assert_close(roundtrip, x, atol=1e-8, rtol=0.0)


# ---------------------------------------------------------------------------
# PODCoarseningStrategy
# ---------------------------------------------------------------------------


class TestPODCoarseningStrategy:
    def test_returns_smaller_matrix(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """Coarse matrix must have fewer dofs than the fine matrix."""
        strategy = PODCoarseningStrategy(rank=10)
        strategy.fit(poisson_snapshots)
        a_c, _ = strategy.build_transfer(poisson_1d)
        assert a_c.shape == (10, 10)

    def test_coarse_matrix_spd(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """Galerkin coarse matrix (Phi_r^T A Phi_r) must be SPD for SPD fine matrix."""
        strategy = PODCoarseningStrategy(rank=10)
        strategy.fit(poisson_snapshots)
        a_c, _ = strategy.build_transfer(poisson_1d)
        eigenvalues = torch.linalg.eigvalsh(a_c)
        assert torch.all(eigenvalues > 0)

    def test_transfer_shapes_consistent(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """Prolongate/restrict sizes must match the coarse and fine matrix shapes."""
        strategy = PODCoarseningStrategy(rank=10)
        strategy.fit(poisson_snapshots)
        a_c, transfer = strategy.build_transfer(poisson_1d)
        n_fine = poisson_1d.shape[0]
        n_coarse = a_c.shape[0]
        assert transfer.prolongate(torch.ones(n_coarse, dtype=poisson_1d.dtype)).shape == (n_fine,)
        assert transfer.restrict(torch.ones(n_fine, dtype=poisson_1d.dtype)).shape == (n_coarse,)

    def test_to_dtype_moves_the_basis_buffer(self, poisson_snapshots: torch.Tensor) -> None:
        """The fitted POD basis is a registered buffer, so ``.to()`` must move it.

        Regression test for a gap flagged during Stage 6: ``PODCoarseningStrategy``
        owns real tensor state (the fitted basis) computed once at construction,
        the same shape of state that motivated ``nn.Module``+``register_buffer``
        for ``JacobiPreconditioner``/``AMGPreconditioner``. Without registering
        it as a buffer, ``.to(dtype=...)`` on an owning ``POD2GPreconditioner``
        would silently leave the basis at its original dtype, causing a dtype
        mismatch on the next ``apply()``.
        """
        strategy = PODCoarseningStrategy(rank=10)
        strategy.fit(poisson_snapshots.double())
        assert strategy._basis.dtype == torch.float64

        strategy.to(dtype=torch.float32)

        assert strategy._basis.dtype == torch.float32

    def test_owning_amg_preconditioner_moves_the_basis_too(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """``.to()`` on the owning ``POD2GPreconditioner`` must reach the basis buffer.

        ``AMGPreconditioner.__init__`` assigns ``self._coarsening = coarsening``
        as a plain attribute; since ``PODCoarseningStrategy`` is an ``nn.Module``,
        ``nn.Module.__setattr__`` auto-registers it as a submodule with no
        ``AMGPreconditioner`` changes required - this test proves that wiring
        actually works end-to-end, not just in isolation.
        """
        precond = POD2GPreconditioner(poisson_1d.double(), poisson_snapshots.double(), rank=10)

        precond.to(dtype=torch.float32)

        coarsening = precond._coarsening
        assert isinstance(coarsening, PODCoarseningStrategy)
        assert coarsening._basis.dtype == torch.float32

    def test_str_before_fit_says_not_yet_fit(self) -> None:
        """`str()` before `fit()` reports the unfit state, doesn't raise."""
        strategy = PODCoarseningStrategy(rank=10)
        assert str(strategy) == "POD-2G(not yet fit)"

    def test_str_after_fit_reports_resolved_rank(self, poisson_snapshots: torch.Tensor) -> None:
        """`str()` after `fit()` reports the actual resolved rank."""
        strategy = PODCoarseningStrategy(rank=10)
        strategy.fit(poisson_snapshots)
        assert str(strategy) == f"POD-2G(rank={strategy.rank})"


# ---------------------------------------------------------------------------
# PODCoarseningStrategy construct/fit lifecycle
# ---------------------------------------------------------------------------


class TestPODCoarseningStrategyLifecycle:
    def test_is_fitted_false_before_fit(self) -> None:
        """A freshly constructed, unfit strategy reports ``is_fitted() is False``."""
        strategy = PODCoarseningStrategy(rank=10)
        assert strategy.is_fitted() is False

    def test_is_fitted_true_after_fit(self, poisson_snapshots: torch.Tensor) -> None:
        """Calling ``fit()`` registers the basis buffer and flips ``is_fitted()``."""
        strategy = PODCoarseningStrategy(rank=10)
        strategy.fit(poisson_snapshots)
        assert strategy.is_fitted() is True

    def test_build_transfer_before_fit_raises(self, poisson_1d: torch.Tensor) -> None:
        """``build_transfer`` on an unfit strategy must raise, not crash obscurely."""
        strategy = PODCoarseningStrategy(rank=10)
        with pytest.raises(RuntimeError, match="fit"):
            strategy.build_transfer(poisson_1d)

    def test_rank_before_fit_raises(self) -> None:
        """The ``rank`` property is only meaningful once a basis has been fit."""
        strategy = PODCoarseningStrategy(rank=10)
        with pytest.raises(RuntimeError, match="fit"):
            _ = strategy.rank

    def test_rank_after_fixed_count_fit_matches_requested_rank(
        self, poisson_snapshots: torch.Tensor
    ) -> None:
        """A fixed-count fit resolves ``rank`` to exactly the requested count."""
        strategy = PODCoarseningStrategy(rank=10)
        strategy.fit(poisson_snapshots)
        assert strategy.rank == 10

    def test_rank_after_energy_threshold_fit_reflects_actual_mode_count(
        self, poisson_snapshots: torch.Tensor
    ) -> None:
        """An energy-threshold fit resolves ``rank`` to the actual (smaller) mode count."""
        strategy = PODCoarseningStrategy(rank=0.999)
        strategy.fit(poisson_snapshots)
        assert 0 < strategy.rank < poisson_snapshots.shape[0]

    def test_out_of_range_float_rank_raises_at_construction(self) -> None:
        """An invalid energy threshold is a self-contained error - fail at construction, not fit()."""
        with pytest.raises(ValueError, match="energy threshold"):
            PODCoarseningStrategy(rank=1.5)

    def test_rank_exceeding_samples_raises_only_at_fit(
        self, poisson_snapshots: torch.Tensor
    ) -> None:
        """An int rank too large for the data can only be caught once data exists, at ``fit()``."""
        strategy = PODCoarseningStrategy(rank=poisson_snapshots.shape[0] + 1)
        with pytest.raises(ValueError, match="rank"):
            strategy.fit(poisson_snapshots)

    def test_reconstruction_from_state_dict_without_refit(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """A fitted strategy's state must be reloadable into a fresh instance without recomputing the SVD.

        This is the exact reconstruction path the construct/fit split exists
        for: a caller who knows the resolved ``rank`` and ``n_dofs`` (both
        recoverable without the original snapshots) can rebuild the basis
        buffer at its exact final shape and ``load_state_dict(strict=True)``
        directly - never touching the ``None``/wrong-shape buffer edge cases
        that motivated this design.
        """
        original = PODCoarseningStrategy(rank=10)
        original.fit(poisson_snapshots)
        state = original.state_dict()
        resolved_rank = original.rank
        n_dofs = poisson_snapshots.shape[1]

        reconstructed = PODCoarseningStrategy(rank=resolved_rank)
        reconstructed.register_buffer(
            "_basis", torch.zeros(n_dofs, resolved_rank, dtype=poisson_snapshots.dtype)
        )
        reconstructed.load_state_dict(state)

        assert reconstructed.is_fitted() is True
        torch.testing.assert_close(reconstructed._basis, original._basis)

        a_c, _ = reconstructed.build_transfer(poisson_1d)
        assert a_c.shape == (resolved_rank, resolved_rank)

    def test_fit_forwards_row_scales_to_compute_pod_basis(
        self, poisson_snapshots: torch.Tensor, snapshot_row_scales: torch.Tensor
    ) -> None:
        """``fit(row_scales=...)`` must produce the same basis as calling ``compute_pod_basis`` directly."""
        strategy = PODCoarseningStrategy(rank=10)
        strategy.fit(poisson_snapshots, row_scales=snapshot_row_scales)
        expected = compute_pod_basis(poisson_snapshots, rank=10, row_scales=snapshot_row_scales)
        torch.testing.assert_close(strategy._basis, expected)


# ---------------------------------------------------------------------------
# POD2GPreconditioner
# ---------------------------------------------------------------------------


class TestPOD2GPreconditioner:
    def test_apply_returns_correct_shape(
        self,
        poisson_1d: torch.Tensor,
        poisson_rhs: torch.Tensor,
        poisson_snapshots: torch.Tensor,
    ) -> None:
        """apply() output must match residual shape."""
        precond = POD2GPreconditioner(poisson_1d, snapshots=poisson_snapshots, rank=10)
        z = precond.apply(poisson_rhs)
        assert z.shape == poisson_rhs.shape

    def test_apply_returns_expected_dtype(
        self,
        poisson_1d: torch.Tensor,
        poisson_rhs: torch.Tensor,
        poisson_snapshots: torch.Tensor,
    ) -> None:
        """apply() output dtype must match the working dtype (no float64 hardcode)."""
        precond = POD2GPreconditioner(poisson_1d, snapshots=poisson_snapshots, rank=10)
        assert precond.apply(poisson_rhs).dtype == poisson_1d.dtype

    def test_uses_configured_smoother(
        self,
        poisson_1d: torch.Tensor,
        poisson_rhs: torch.Tensor,
        poisson_snapshots: torch.Tensor,
        raising_smoother: MultigridSmoother,
    ) -> None:
        """POD-2G must delegate solve-time smoothing to the injected strategy."""
        precond = POD2GPreconditioner(
            poisson_1d,
            snapshots=poisson_snapshots,
            rank=10,
            smoother=raising_smoother,
        )
        with pytest.raises(RuntimeError, match="configured smoother used"):
            precond.apply(poisson_rhs)

    def test_requires_flexible_cg_false(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """POD-2G is a linear preconditioner and must not request FCG."""
        precond = POD2GPreconditioner(poisson_1d, snapshots=poisson_snapshots, rank=10)
        assert precond.requires_flexible_cg is False

    def test_rejects_single_level_hierarchy(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """POD-2G must keep its two-grid contract explicit."""
        with pytest.raises(ValueError, match="n_levels"):
            POD2GPreconditioner(
                poisson_1d,
                snapshots=poisson_snapshots,
                rank=10,
                n_levels=1,
            )


# ---------------------------------------------------------------------------
# Integration: POD-2G inside FCG solver
# ---------------------------------------------------------------------------


class TestPOD2GWithFCGSolver:
    def test_pod2g_preconditioned_fcg_converges(
        self,
        poisson_1d: torch.Tensor,
        poisson_rhs: torch.Tensor,
        poisson_snapshots: torch.Tensor,
    ) -> None:
        """POD-2G-preconditioned FCG must converge on 1D Poisson."""
        from torchalg import flexible_cg

        precond = POD2GPreconditioner(poisson_1d, snapshots=poisson_snapshots, rank=15)
        x, info = flexible_cg(poisson_1d, poisson_rhs, preconditioner=precond, rtol=1e-8)
        assert info.converged, f"FCG+POD-2G did not converge: {info}"
        torch.testing.assert_close(poisson_1d @ x, poisson_rhs, rtol=1e-6, atol=1e-6)
