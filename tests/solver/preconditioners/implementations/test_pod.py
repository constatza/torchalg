"""Tests for POD-2G preconditioner components.

Ported from the reference (``dl-experiments``'s
``tests/solver/preconditioners/implementations/test_pod.py``), adapted from
numpy to torch tensors, using the shared ``poisson_1d``/``poisson_rhs``/
``poisson_snapshots``/``pod_basis`` fixtures
(``tests/solver/preconditioners/conftest.py``) instead of inline
construction.

Deviations from the reference:
    - ``TestComputePodBasis`` and ``TestDenseTransferOperator`` have moved to
      ``tests/solver/utils/test_pod_basis.py`` and
      ``tests/solver/utils/test_dense_transfer.py`` respectively, mirroring
      ``compute_pod_basis``/``DenseTransferOperator`` themselves being
      promoted to dependency-free leaves under ``torchalg.utils`` (see
      ``docs/plan.md``). This file keeps only the POD-2G-specific tests
      (``PODCoarseningStrategy``, ``pod2g_preconditioner``).
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

from torchalg.preconditioners.implementations.amg import CoarseningStrategy, MultigridSmoother
from torchalg.preconditioners.implementations.pod import (
    PODCoarseningStrategy,
    pod2g_preconditioner,
)

# ---------------------------------------------------------------------------
# PODCoarseningStrategy
# ---------------------------------------------------------------------------


class TestPODCoarseningStrategy:
    def test_matches_dense_galerkin_for_either_layout(
        self,
        poisson_1d: torch.Tensor,
        poisson_1d_any_layout: torch.Tensor,
        pod_basis: torch.Tensor,
        pod_transfer_vectors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Strided and CSR ``A`` give ``Phi^T A Phi`` and the ``Phi`` / ``Phi^T`` transfers."""
        # Annotated to pin protocol conformance under ``ty check``.
        strategy: CoarseningStrategy = PODCoarseningStrategy(pod_basis)
        coarse, fine = pod_transfer_vectors

        a_coarse, transfer = strategy.build_transfer(poisson_1d_any_layout)

        torch.testing.assert_close(a_coarse, pod_basis.T @ poisson_1d @ pod_basis)
        torch.testing.assert_close(transfer.prolongate(coarse), pod_basis @ coarse)
        torch.testing.assert_close(transfer.restrict(fine), pod_basis.T @ fine)

    def test_coarse_matrix_spd(self, poisson_1d: torch.Tensor, pod_basis: torch.Tensor) -> None:
        """Galerkin coarse matrix (Phi_r^T A Phi_r) must be SPD for SPD fine matrix."""
        a_c, _ = PODCoarseningStrategy(pod_basis).build_transfer(poisson_1d)
        eigenvalues = torch.linalg.eigvalsh(a_c)
        assert torch.all(eigenvalues > 0)

    def test_invalid_basis_raises(self, invalid_pod_basis: torch.Tensor) -> None:
        """Only a strided ``(n_dofs, r)`` basis with ``r >= 1`` is accepted."""
        with pytest.raises(ValueError, match="basis"):
            PODCoarseningStrategy(invalid_pod_basis)

    def test_str_reports_rank(self, pod_basis: torch.Tensor) -> None:
        """``str()`` reports the basis column count."""
        assert str(PODCoarseningStrategy(pod_basis)) == "POD-2G(rank=10)"

    def test_state_dict_round_trip(self, pod_basis: torch.Tensor) -> None:
        """A checkpointed basis loads into a strategy built from a placeholder of its shape."""
        reconstructed = PODCoarseningStrategy(torch.empty_like(pod_basis))

        reconstructed.load_state_dict(PODCoarseningStrategy(pod_basis).state_dict())

        torch.testing.assert_close(reconstructed.basis, pod_basis)

    def test_to_dtype_moves_the_basis_buffer(self, pod_basis: torch.Tensor) -> None:
        """The basis is a registered buffer, so ``.to()`` must move it.

        Without registering it as a buffer, ``.to(dtype=...)`` on an owning
        ``POD2GPreconditioner`` would silently leave the basis at its original
        dtype, causing a dtype mismatch on the next ``apply()``.
        """
        strategy = PODCoarseningStrategy(pod_basis.double())

        strategy.to(dtype=torch.float32)

        assert strategy.basis.dtype == torch.float32

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
        precond = pod2g_preconditioner(poisson_snapshots.double(), rank=10)

        precond.to(dtype=torch.float32)

        coarsening = precond._coarsening
        assert isinstance(coarsening, PODCoarseningStrategy)
        assert coarsening.basis.dtype == torch.float32


# ---------------------------------------------------------------------------
# POD2GPreconditioner
# ---------------------------------------------------------------------------


class TestPOD2GPreconditioner:
    def test_string_uses_generic_amg_format(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """pod2g_preconditioner returns a plain AMGPreconditioner, so its
        __str__ is the generic AMG format embedding the POD coarsening's
        own string, not a POD-specific override.
        """
        precond = pod2g_preconditioner(snapshots=poisson_snapshots, rank=10)
        assert str(precond) == "AMG(n_levels=2, POD-2G(rank=10))"

    def test_apply_returns_correct_shape(
        self,
        poisson_1d: torch.Tensor,
        poisson_rhs: torch.Tensor,
        poisson_snapshots: torch.Tensor,
    ) -> None:
        """apply() output must match residual shape."""
        precond = pod2g_preconditioner(snapshots=poisson_snapshots, rank=10)
        precond.setup(poisson_1d)
        z = precond.apply(poisson_rhs)
        assert z.shape == poisson_rhs.shape

    def test_apply_returns_expected_dtype(
        self,
        poisson_1d: torch.Tensor,
        poisson_rhs: torch.Tensor,
        poisson_snapshots: torch.Tensor,
    ) -> None:
        """apply() output dtype must match the working dtype (no float64 hardcode)."""
        precond = pod2g_preconditioner(snapshots=poisson_snapshots, rank=10)
        precond.setup(poisson_1d)
        assert precond.apply(poisson_rhs).dtype == poisson_1d.dtype

    def test_uses_configured_smoother(
        self,
        poisson_1d: torch.Tensor,
        poisson_snapshots: torch.Tensor,
        raising_smoother: MultigridSmoother,
    ) -> None:
        """POD-2G must delegate solve-time smoothing to the injected strategy."""
        precond = pod2g_preconditioner(
            snapshots=poisson_snapshots,
            rank=10,
            smoother=raising_smoother,
        )
        with pytest.raises(RuntimeError, match="configured smoother used"):
            precond.setup(poisson_1d)

    def test_requires_flexible_cg_false(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """POD-2G is a linear preconditioner and must not request FCG."""
        precond = pod2g_preconditioner(snapshots=poisson_snapshots, rank=10)
        assert precond.requires_flexible_cg is False

    def test_rejects_single_level_hierarchy(
        self, poisson_1d: torch.Tensor, poisson_snapshots: torch.Tensor
    ) -> None:
        """POD-2G must keep its two-grid contract explicit."""
        with pytest.raises(ValueError, match="n_levels"):
            pod2g_preconditioner(
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

        precond = pod2g_preconditioner(snapshots=poisson_snapshots, rank=15)
        precond.setup(poisson_1d)
        x, info = flexible_cg(poisson_1d, poisson_rhs, preconditioner=precond, rtol=1e-8)
        assert info.converged, f"FCG+POD-2G did not converge: {info}"
        torch.testing.assert_close(poisson_1d @ x, poisson_rhs, rtol=1e-6, atol=1e-6)
