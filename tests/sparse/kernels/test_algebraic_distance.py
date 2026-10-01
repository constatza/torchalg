"""Tests for ``torchalg.sparse.kernels.algebraic_distance.sparse_algebraic_distance``.

Dense-as-oracle, this project's standing rule: checked against
``preconditioners.implementations.amg._algebraic_distance.algebraic_distance``
on shared fixtures, including a deliberately non-symmetric matrix - ``r_ij
!= r_ji`` in general (the dense function's own documented directionality),
so a symmetric-only fixture would hide a row/column transpose bug.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations.amg._algebraic_distance import (
    algebraic_distance as dense_algebraic_distance,
)
from torchalg.sparse.kernels.algebraic_distance import sparse_algebraic_distance


@pytest.fixture
def asymmetric_matrix(torch_dtype: torch.dtype) -> torch.Tensor:
    """5x5 non-symmetric matrix, positive diagonal, with one bidirectional edge (0,1)/(1,0).

    The bidirectional edge (both ``(0,1)`` and ``(1,0)`` stored, with
    different magnitudes) lets a test compare two *real computed* ``r``
    values in opposite directions at the same pair - not just "stored
    position vs. absent position", which wouldn't actually exercise the
    directional formula.
    """
    dense = torch.tensor(
        [
            [4.0, 1.0, 0.0, 0.0, 0.0],
            [2.0, 5.0, 2.0, 0.0, 0.0],
            [1.0, 0.0, 6.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 3.0],
            [2.0, 0.0, 0.0, 0.0, 5.0],
        ],
        dtype=torch_dtype,
    )
    assert not torch.equal(dense, dense.T)
    return dense


class TestSparseAlgebraicDistance:
    def test_matches_dense_on_asymmetric_matrix(self, asymmetric_matrix: torch.Tensor) -> None:
        torch.manual_seed(0)
        n = asymmetric_matrix.shape[0]
        test_vectors = torch.randn(n, 3, dtype=asymmetric_matrix.dtype)

        expected = dense_algebraic_distance(test_vectors, asymmetric_matrix, depth=1)
        actual = sparse_algebraic_distance(test_vectors, asymmetric_matrix.to_sparse_csr(), depth=1)
        torch.testing.assert_close(actual.to_dense(), expected)

    def test_not_symmetric_r_ij_differs_from_r_ji(self, asymmetric_matrix: torch.Tensor) -> None:
        torch.manual_seed(1)
        n = asymmetric_matrix.shape[0]
        test_vectors = torch.randn(n, 3, dtype=asymmetric_matrix.dtype)

        r = sparse_algebraic_distance(
            test_vectors, asymmetric_matrix.to_sparse_csr(), depth=1
        ).to_dense()
        # (0,1) and (1,0) are both stored (the fixture's deliberate
        # bidirectional edge) - compare two real computed values, not a
        # stored-vs-absent position.
        assert not torch.isclose(r[0, 1], r[1, 0])

    def test_matches_dense_on_poisson_1d(self, poisson_1d_dense: torch.Tensor) -> None:
        torch.manual_seed(2)
        n = poisson_1d_dense.shape[0]
        test_vectors = torch.randn(n, 4, dtype=poisson_1d_dense.dtype)

        expected = dense_algebraic_distance(test_vectors, poisson_1d_dense, depth=1)
        actual = sparse_algebraic_distance(test_vectors, poisson_1d_dense.to_sparse_csr(), depth=1)
        torch.testing.assert_close(actual.to_dense(), expected)

    def test_matches_dense_at_depth_2(self, poisson_1d_dense: torch.Tensor) -> None:
        torch.manual_seed(3)
        n = poisson_1d_dense.shape[0]
        test_vectors = torch.randn(n, 3, dtype=poisson_1d_dense.dtype)

        expected = dense_algebraic_distance(test_vectors, poisson_1d_dense, depth=2)
        actual = sparse_algebraic_distance(test_vectors, poisson_1d_dense.to_sparse_csr(), depth=2)
        torch.testing.assert_close(actual.to_dense(), expected)

    def test_matches_dense_on_poisson_2d(self, poisson_2d_dense: torch.Tensor) -> None:
        torch.manual_seed(4)
        n = poisson_2d_dense.shape[0]
        test_vectors = torch.randn(n, 3, dtype=poisson_2d_dense.dtype)

        expected = dense_algebraic_distance(test_vectors, poisson_2d_dense, depth=1)
        actual = sparse_algebraic_distance(test_vectors, poisson_2d_dense.to_sparse_csr(), depth=1)
        torch.testing.assert_close(actual.to_dense(), expected)

    def test_rejects_non_csr_input(self, poisson_1d_dense: torch.Tensor) -> None:
        test_vectors = torch.randn(poisson_1d_dense.shape[0], 2, dtype=poisson_1d_dense.dtype)
        with pytest.raises(ValueError, match="sparse CSR"):
            sparse_algebraic_distance(test_vectors, poisson_1d_dense, depth=1)
