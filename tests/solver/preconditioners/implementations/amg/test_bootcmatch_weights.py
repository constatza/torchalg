"""BootCMatch edge-weight tests ([BootCMatch18] eq. 10, Bootstrap AMG graph matching).

Fixtures live in this module, matching ``test_algebraic_distance.py``'s
precedent: each ``implementations/amg/`` test module keeps its fixtures
local rather than pre-emptively factoring them into a shared conftest before
a second consumer exists.
"""

from __future__ import annotations

import math

import pytest
import torch

from torchalg.preconditioners.implementations.amg._bootcmatch_weights import (
    bootcmatch_edge_weights,
)


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
def smooth_vector_4(torch_dtype: torch.dtype) -> torch.Tensor:
    """Smooth vector ``w = [1, 2, 3, 4]`` for ``tridiagonal_4x4``."""
    return torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch_dtype)


class TestBootcmatchEdgeWeightsDense:
    def test_hand_computed_edge_weights(
        self, tridiagonal_4x4: torch.Tensor, smooth_vector_4: torch.Tensor
    ) -> None:
        """``a_hat_01`` and ``a_hat_12`` match the hand-computed eq. 10 values."""
        weights, fine_only_mask = bootcmatch_edge_weights(smooth_vector_4, tridiagonal_4x4)

        # a_hat_01 = 1 - 2*1*1*2 / (4*1**2 + 4*2**2) = 1 - 4/20 = 0.8
        assert weights[0, 1] == pytest.approx(0.8)
        # a_hat_12 = 1 - 2*1*2*3 / (4*2**2 + 4*3**2) = 1 - 12/52
        assert weights[1, 2] == pytest.approx(1.0 - 12.0 / 52.0)
        assert not fine_only_mask.any()

    def test_non_edges_and_diagonal_are_negative_infinity(
        self, tridiagonal_4x4: torch.Tensor, smooth_vector_4: torch.Tensor
    ) -> None:
        """Non-edge positions (including the diagonal) are ``-inf``, never selectable by argmax."""
        weights, _ = bootcmatch_edge_weights(smooth_vector_4, tridiagonal_4x4)

        assert torch.isinf(torch.diagonal(weights)).all()
        assert weights[0, 2] == float("-inf")
        assert weights[0, 3] == float("-inf")

    def test_degenerate_edge_is_excluded(self, torch_dtype: torch.dtype) -> None:
        """An edge with ``sqrt(w_i**2/a_ii + w_j**2/a_jj) < TOL`` is ``-inf``, never selectable."""
        matrix = torch.tensor(
            [[4.0, 1.0], [1.0, 4.0]],
            dtype=torch_dtype,
        )
        w = torch.tensor([0.0, 0.0], dtype=torch_dtype)

        weights, _ = bootcmatch_edge_weights(w, matrix)

        assert weights[0, 1] == float("-inf")
        assert weights[1, 0] == float("-inf")

    def test_fine_only_mask_flags_near_zero_smooth_vector_entries(
        self, tridiagonal_4x4: torch.Tensor, torch_dtype: torch.dtype
    ) -> None:
        """``fine_only_mask`` is ``True`` only where ``|w_i| < TOL`` (machine epsilon)."""
        tol = torch.finfo(torch_dtype).eps
        w = torch.tensor([tol / 10.0, 1.0, 2.0, 3.0], dtype=torch_dtype)

        _, fine_only_mask = bootcmatch_edge_weights(w, tridiagonal_4x4)

        assert fine_only_mask.tolist() == [True, False, False, False]

    def test_dense_sparse_parity(
        self, tridiagonal_4x4: torch.Tensor, smooth_vector_4: torch.Tensor
    ) -> None:
        """Dense and sparse kernels agree at every shared stored-edge position."""
        from torchalg.sparse.kernels.bootcmatch_weights import (
            bootcmatch_edge_weights as sparse_bootcmatch_edge_weights,
        )

        dense_weights, dense_fine_only = bootcmatch_edge_weights(smooth_vector_4, tridiagonal_4x4)
        sparse_weights, sparse_fine_only = sparse_bootcmatch_edge_weights(
            smooth_vector_4, tridiagonal_4x4.to_sparse_csr()
        )

        sparse_dense = sparse_weights.to_dense()
        edge_mask = tridiagonal_4x4 != 0
        off_diagonal = ~torch.eye(4, dtype=torch.bool)
        shared_edges = edge_mask & off_diagonal

        torch.testing.assert_close(
            sparse_dense[shared_edges],
            dense_weights[shared_edges],
        )
        assert torch.equal(dense_fine_only, sparse_fine_only)

    def test_no_nan_in_finite_weights(
        self, tridiagonal_4x4: torch.Tensor, smooth_vector_4: torch.Tensor
    ) -> None:
        """Finite entries of the weight matrix are never NaN."""
        weights, _ = bootcmatch_edge_weights(smooth_vector_4, tridiagonal_4x4)
        assert not torch.isnan(weights[torch.isfinite(weights)]).any()
        assert math.isfinite(float(weights[0, 1]))
