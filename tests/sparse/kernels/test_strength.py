"""Tests for ``torchalg.sparse.kernels.strength``."""

from __future__ import annotations

import torch

from torchalg.preconditioners.implementations.amg._aggregation import strength_of_connection
from torchalg.preconditioners.implementations.amg._algebraic_distance import (
    strength_graph as dense_strength_graph,
)
from torchalg.sparse.kernels.strength import sparse_strength_graph, sparse_strength_of_connection

_THETA = 0.25


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


class TestSparseStrengthGraph:
    """``sparse_strength_graph`` matches the dense reference exactly, given the same ``r``.

    Fed the identical ``distance`` tensor (one dense, one its sparse CSR
    twin) so this isolates the pruning step's own correctness from any
    upstream ``algebraic_distance`` computation - per this project's
    standing dense-as-oracle rule.
    """

    @staticmethod
    def _distance_fixture(torch_dtype: torch.dtype) -> torch.Tensor:
        """4x4 directional ``r``-like matrix: positive values, zero diagonal, asymmetric."""
        return torch.tensor(
            [
                [0.0, 3.0, 0.0, 1.0],
                [0.5, 0.0, 2.0, 0.0],
                [0.0, 4.0, 0.0, 6.0],
                [2.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch_dtype,
        )

    def test_matches_dense_reference(self, torch_dtype: torch.dtype) -> None:
        distance = self._distance_fixture(torch_dtype)
        fine_mask = torch.tensor([True, True, False, True])
        expected = dense_strength_graph(distance, fine_mask, theta_ad=0.5)
        actual = sparse_strength_graph(distance.to_sparse_csr(), fine_mask, theta_ad=0.5)
        torch.testing.assert_close(actual.to_dense(), expected)

    def test_matches_dense_reference_all_fine(self, torch_dtype: torch.dtype) -> None:
        distance = self._distance_fixture(torch_dtype)
        fine_mask = torch.ones(4, dtype=torch.bool)
        expected = dense_strength_graph(distance, fine_mask, theta_ad=0.25)
        actual = sparse_strength_graph(distance.to_sparse_csr(), fine_mask, theta_ad=0.25)
        torch.testing.assert_close(actual.to_dense(), expected)

    def test_returns_sparse_csr(self, torch_dtype: torch.dtype) -> None:
        distance = self._distance_fixture(torch_dtype)
        fine_mask = torch.ones(4, dtype=torch.bool)
        actual = sparse_strength_graph(distance.to_sparse_csr(), fine_mask)
        assert actual.layout == torch.sparse_csr
        assert actual.dtype == torch.bool
