"""Tests for ``torchalg.utils.dense_transfer.DenseTransferOperator``.

Promoted alongside the source class out of
``tests/solver/preconditioners/implementations/test_pod.py``'s
``TestDenseTransferOperator`` (see ``docs/plan.md``): the class was already
fully format-agnostic (only ever requires a dense ``P``), so its tests live
here, mirroring ``tests/solver/utils/test_spectral.py``'s convention for the
other promoted leaf (``torchalg.utils.spectral``). Reuses the ``pod_basis``
fixture as a convenient dense ``(n_dofs, rank)`` matrix - ``DenseTransferOperator``
itself has no POD-specific behavior.
"""

from __future__ import annotations

import torch

from torchalg.utils.dense_transfer import DenseTransferOperator


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
