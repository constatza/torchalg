"""Tests for ``torchalg.sparse.kernels.row_max.sparse_row_max``.

Spec: ``sparse_row_max(matrix)`` must equal ``matrix.to_dense().max(dim=1)
.values`` for any sparse CSR matrix - ``to_dense()`` fills every
non-stored position with 0, so a row's dense max already includes 0 as an
implicit candidate alongside its real stored entries, the same semantics
``_algebraic_distance.strength_graph``'s ``distance.max(dim=1).values``
relies on (every non-neighborhood position is densely zero-filled there
too, not merely absent).
"""

from __future__ import annotations

import pytest
import torch

from torchalg.sparse.kernels.row_max import sparse_row_max


@pytest.fixture
def mixed_sign_matrix(torch_dtype: torch.dtype) -> torch.Tensor:
    """4x4 matrix with positive, negative, zero-valued, and empty rows/positions.

    Row 0: all positive stored entries.
    Row 1: all negative stored entries (max should be the implicit 0, not
        the least-negative entry) - exercises the "0 is always a candidate"
        semantics directly.
    Row 2: a single positive entry.
    Row 3: no stored entries at all (fully empty row).
    """
    dense = torch.tensor(
        [
            [0.0, 2.0, 0.0, 5.0],
            [-3.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 7.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch_dtype,
    )
    return dense


class TestSparseRowMax:
    def test_matches_dense_max_over_rows(self, mixed_sign_matrix: torch.Tensor) -> None:
        expected = mixed_sign_matrix.max(dim=1).values
        actual = sparse_row_max(mixed_sign_matrix.to_sparse_csr())
        torch.testing.assert_close(actual, expected)

    def test_matches_dense_on_poisson_1d(self, poisson_1d_dense: torch.Tensor) -> None:
        expected = poisson_1d_dense.max(dim=1).values
        actual = sparse_row_max(poisson_1d_dense.to_sparse_csr())
        torch.testing.assert_close(actual, expected)

    def test_rejects_non_csr_input(self, poisson_1d_dense: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="sparse CSR"):
            sparse_row_max(poisson_1d_dense)
