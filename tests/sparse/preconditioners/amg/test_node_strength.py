"""Tests for ``torchalg.sparse.preconditioners.amg._node_strength.sparse_node_strength``.

Dense-as-oracle, this project's standing rule: both the scalar
(``dofs_per_node=1``) and block (``dofs_per_node>1``) cases are checked
against the dense ``node_strength`` on a shared fixture.
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations.amg._node_strength import (
    node_strength as dense_node_strength,
)
from torchalg.sparse.preconditioners.amg._node_strength import sparse_node_strength


class TestSparseNodeStrengthScalar:
    """``dofs_per_node=1`` - every stored entry is its own trivial 1x1 block."""

    @pytest.mark.parametrize("theta", [0.0, 0.25, 0.5])
    def test_matches_dense_on_poisson_1d(
        self, poisson_1d_dense: torch.Tensor, theta: float
    ) -> None:
        expected = dense_node_strength(poisson_1d_dense, 1, theta)
        actual = sparse_node_strength(poisson_1d_dense.to_sparse_csr(), 1, theta)
        torch.testing.assert_close(actual.to_dense(), expected)

    @pytest.mark.parametrize("theta", [0.0, 0.3])
    def test_matches_dense_on_poisson_2d(
        self, poisson_2d_dense: torch.Tensor, theta: float
    ) -> None:
        expected = dense_node_strength(poisson_2d_dense, 1, theta)
        actual = sparse_node_strength(poisson_2d_dense.to_sparse_csr(), 1, theta)
        torch.testing.assert_close(actual.to_dense(), expected)

    def test_rejects_negative_theta(self, poisson_1d_csr: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="positive theta"):
            sparse_node_strength(poisson_1d_csr, 1, -0.1)

    def test_rejects_non_csr_input(self, poisson_1d_dense: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="sparse CSR"):
            sparse_node_strength(poisson_1d_dense, 1, 0.0)


class TestSparseNodeStrengthBlock:
    """``dofs_per_node>1`` - the block-Frobenius-norm case."""

    @pytest.fixture
    def block_matrix(self, torch_dtype: torch.dtype) -> torch.Tensor:
        """30x30 SPD matrix with a thresholded sparsity pattern, 3 dofs/node.

        Mirrors ``test_adaptive_sa_port.py::TestNodeStrength.test_block_matches_pyamg``'s
        fixture-construction approach (random SPD + threshold + fixed
        diagonal), kept local per this project's convention of per-module
        fixtures in ``tests/sparse/preconditioners/amg/``.
        """
        generator = torch.Generator().manual_seed(24)
        raw = torch.randn(30, 30, generator=generator, dtype=torch_dtype)
        matrix = raw @ raw.T + 30 * torch.eye(30, dtype=torch_dtype)
        matrix[torch.abs(matrix) < 4.0] = 0.0
        matrix.fill_diagonal_(30.0)
        return matrix

    @pytest.mark.parametrize("theta", [0.0, 0.3])
    def test_matches_dense(self, block_matrix: torch.Tensor, theta: float) -> None:
        expected = dense_node_strength(block_matrix, 3, theta)
        actual = sparse_node_strength(block_matrix.to_sparse_csr(), 3, theta)
        torch.testing.assert_close(actual.to_dense(), expected)
