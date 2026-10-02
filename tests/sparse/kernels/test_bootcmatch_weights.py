"""Tests for ``torchalg.sparse.kernels.bootcmatch_weights.bootcmatch_edge_weights``.

Dense-as-oracle, this project's standing rule: checked against
``preconditioners.implementations.amg._bootcmatch_weights.bootcmatch_edge_weights``
on shared fixtures.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations.amg._bootcmatch_weights import (
    bootcmatch_edge_weights as dense_bootcmatch_edge_weights,
)
from torchalg.sparse.kernels.bootcmatch_weights import bootcmatch_edge_weights


@pytest.fixture
def tridiagonal_4x4(torch_dtype: torch.dtype) -> torch.Tensor:
    """4x4 symmetric tridiagonal SPD matrix, diagonal 4, off-diagonal 1."""
    return torch.tensor(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 4.0, 1.0, 0.0],
            [0.0, 1.0, 4.0, 1.0],
            [0.0, 0.0, 1.0, 4.0],
        ],
        dtype=torch_dtype,
    )


@pytest.fixture
def tridiagonal_4x4_csr(tridiagonal_4x4: torch.Tensor) -> torch.Tensor:
    """``tridiagonal_4x4`` converted to sparse CSR."""
    return tridiagonal_4x4.to_sparse_csr()


@pytest.fixture
def smooth_vector_4(torch_dtype: torch.dtype) -> torch.Tensor:
    """Smooth vector ``w = [1, 2, 3, 4]`` for ``tridiagonal_4x4``."""
    return torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch_dtype)


class TestSparseBootcmatchEdgeWeights:
    def test_hand_computed_edge_weights(
        self, tridiagonal_4x4_csr: torch.Tensor, smooth_vector_4: torch.Tensor
    ) -> None:
        """``a_hat_01`` and ``a_hat_12`` match the hand-computed eq. 10 values."""
        weights, fine_only_mask = bootcmatch_edge_weights(smooth_vector_4, tridiagonal_4x4_csr)
        dense = weights.to_dense()

        assert dense[0, 1] == pytest.approx(0.8)
        assert dense[1, 2] == pytest.approx(1.0 - 12.0 / 52.0)
        assert not fine_only_mask.any()

    def test_matches_dense_on_poisson_1d(
        self, poisson_1d_dense: torch.Tensor, poisson_1d_csr: torch.Tensor, torch_dtype: torch.dtype
    ) -> None:
        """Sparse result matches the dense oracle at every shared stored-edge position."""
        torch.manual_seed(0)
        n = poisson_1d_dense.shape[0]
        w = torch.randn(n, dtype=torch_dtype)

        dense_weights, dense_fine_only = dense_bootcmatch_edge_weights(w, poisson_1d_dense)
        sparse_weights, sparse_fine_only = bootcmatch_edge_weights(w, poisson_1d_csr)

        off_diagonal = ~torch.eye(n, dtype=torch.bool)
        shared_edges = (poisson_1d_dense != 0) & off_diagonal

        torch.testing.assert_close(
            sparse_weights.to_dense()[shared_edges], dense_weights[shared_edges]
        )
        assert torch.equal(dense_fine_only, sparse_fine_only)

    def test_degenerate_edge_is_structurally_absent(self, torch_dtype: torch.dtype) -> None:
        """A degenerate edge is absent from the returned CSR pattern entirely."""
        matrix = torch.tensor([[4.0, 1.0], [1.0, 4.0]], dtype=torch_dtype).to_sparse_csr()
        w = torch.tensor([0.0, 0.0], dtype=torch_dtype)

        weights, _ = bootcmatch_edge_weights(w, matrix)

        assert weights._nnz() == 0

    def test_fine_only_mask_flags_near_zero_smooth_vector_entries(
        self, tridiagonal_4x4_csr: torch.Tensor, torch_dtype: torch.dtype
    ) -> None:
        """``fine_only_mask`` is ``True`` only where ``|w_i| < TOL`` (machine epsilon)."""
        tol = torch.finfo(torch_dtype).eps
        w = torch.tensor([tol / 10.0, 1.0, 2.0, 3.0], dtype=torch_dtype)

        _, fine_only_mask = bootcmatch_edge_weights(w, tridiagonal_4x4_csr)

        assert fine_only_mask.tolist() == [True, False, False, False]

    def test_rejects_non_csr_input(
        self, tridiagonal_4x4: torch.Tensor, smooth_vector_4: torch.Tensor
    ) -> None:
        with pytest.raises(ValueError, match="sparse CSR"):
            bootcmatch_edge_weights(smooth_vector_4, tridiagonal_4x4)
