"""Tests for ``torchalg.sparse.aggregation``.

Cross-checks the sparse-native strength-of-connection/aggregation kernels
against the existing dense ones (``amg._aggregation``) on the same matrix -
same algorithm, different storage, so the two must agree exactly.
"""

from __future__ import annotations

import torch

from torchalg.preconditioners.implementations.amg._aggregation import (
    piecewise_constant_prolongation,
    smoothed_prolongation,
    standard_aggregation,
    strength_of_connection,
)
from torchalg.sparse.aggregation import (
    sparse_piecewise_constant_prolongation,
    sparse_smoothed_prolongation,
    sparse_standard_aggregation,
    sparse_strength_of_connection,
)

_THETA = 0.25
_OMEGA = 2.0 / 3.0


class TestSparseStrengthOfConnection:
    """``sparse_strength_of_connection`` matches the dense reference exactly."""

    def test_matches_dense_reference(
        self, poisson_1d_dense: torch.Tensor, poisson_1d_csr: torch.Tensor
    ) -> None:
        """The sparse strength graph, densified, equals the dense strength matrix."""
        expected = strength_of_connection(poisson_1d_dense, _THETA)
        actual = sparse_strength_of_connection(poisson_1d_csr, _THETA)
        torch.testing.assert_close(actual.to_dense(), expected)

    def test_returns_sparse_csr(self, poisson_1d_csr: torch.Tensor) -> None:
        """The result stays sparse CSR - no dense (n, n) intermediate."""
        actual = sparse_strength_of_connection(poisson_1d_csr, _THETA)
        assert actual.layout == torch.sparse_csr

    def test_excludes_diagonal(self, poisson_1d_csr: torch.Tensor) -> None:
        """No diagonal entry is ever marked strong, matching the dense convention."""
        actual = sparse_strength_of_connection(poisson_1d_csr, _THETA).to_dense()
        assert not torch.diagonal(actual).any()


class TestSparseStandardAggregation:
    """``sparse_standard_aggregation`` matches the dense reference exactly."""

    def test_matches_dense_reference(
        self, poisson_1d_dense: torch.Tensor, poisson_1d_csr: torch.Tensor
    ) -> None:
        """Aggregate assignment is identical whether built from dense or sparse strength."""
        expected = standard_aggregation(strength_of_connection(poisson_1d_dense, _THETA))
        strength_sparse = sparse_strength_of_connection(poisson_1d_csr, _THETA)
        actual = sparse_standard_aggregation(strength_sparse)
        torch.testing.assert_close(actual, expected)


class TestSparsePiecewiseConstantProlongation:
    """``sparse_piecewise_constant_prolongation`` matches the dense reference exactly."""

    def test_matches_dense_reference(
        self, poisson_1d_dense: torch.Tensor, poisson_1d_csr: torch.Tensor
    ) -> None:
        """The sparse tentative prolongation, densified, equals the dense one."""
        aggregate = standard_aggregation(strength_of_connection(poisson_1d_dense, _THETA))
        expected = piecewise_constant_prolongation(aggregate, dtype=poisson_1d_dense.dtype)
        actual = sparse_piecewise_constant_prolongation(aggregate, dtype=poisson_1d_dense.dtype)
        torch.testing.assert_close(actual.to_dense(), expected)

    def test_returns_sparse_csr(self, poisson_1d_dense: torch.Tensor) -> None:
        """The tentative prolongation is returned sparse CSR."""
        aggregate = standard_aggregation(strength_of_connection(poisson_1d_dense, _THETA))
        actual = sparse_piecewise_constant_prolongation(aggregate, dtype=poisson_1d_dense.dtype)
        assert actual.layout == torch.sparse_csr


class TestSparseSmoothedProlongation:
    """``sparse_smoothed_prolongation`` matches the dense reference exactly."""

    def test_matches_dense_reference(
        self, poisson_1d_dense: torch.Tensor, poisson_1d_csr: torch.Tensor
    ) -> None:
        """The sparse smoothed prolongation, densified, equals the dense one."""
        aggregate = standard_aggregation(strength_of_connection(poisson_1d_dense, _THETA))
        tentative_dense = piecewise_constant_prolongation(aggregate, dtype=poisson_1d_dense.dtype)
        expected = smoothed_prolongation(poisson_1d_dense, tentative_dense, _OMEGA)

        tentative_sparse = sparse_piecewise_constant_prolongation(
            aggregate, dtype=poisson_1d_dense.dtype
        )
        actual = sparse_smoothed_prolongation(poisson_1d_csr, tentative_sparse, _OMEGA)
        torch.testing.assert_close(actual.to_dense(), expected)

    def test_returns_sparse_csr(
        self, poisson_1d_dense: torch.Tensor, poisson_1d_csr: torch.Tensor
    ) -> None:
        """The smoothed prolongation is returned sparse CSR."""
        aggregate = standard_aggregation(strength_of_connection(poisson_1d_dense, _THETA))
        tentative_sparse = sparse_piecewise_constant_prolongation(
            aggregate, dtype=poisson_1d_dense.dtype
        )
        actual = sparse_smoothed_prolongation(poisson_1d_csr, tentative_sparse, _OMEGA)
        assert actual.layout == torch.sparse_csr
