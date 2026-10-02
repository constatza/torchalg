"""Parity tests for the sparse ``make_bridge`` sibling.

Sparse-CSR counterpart of the dense
``preconditioners.implementations.amg._prolongation.make_bridge`` - for CSR
input every inserted row is structurally empty, so this reduces to a pure
index remap of ``crow_indices`` (no dense intermediate).
"""

from __future__ import annotations

import pytest
import torch

from torchalg.preconditioners.implementations.amg._prolongation import (
    make_bridge as dense_make_bridge,
)
from torchalg.sparse.preconditioners.amg._prolongation import sparse_make_bridge


class TestSparseMakeBridgeMatchesDense:
    @pytest.mark.parametrize("dofs_per_node", [1, 2, 3])
    def test_matches_dense_bridge(self, torch_dtype: torch.dtype, dofs_per_node: int) -> None:
        torch.manual_seed(0)
        n_nodes = 5
        n_coarse = 4
        n = n_nodes * dofs_per_node
        dense = torch.zeros(n, n_coarse, dtype=torch_dtype)
        # Sparse structure: each row has a couple of nonzero entries so the
        # test actually exercises nonzero-preserving index remap, not just
        # zero blocks.
        for row in range(n):
            cols = torch.randperm(n_coarse)[: min(2, n_coarse)]
            dense[row, cols] = torch.randn(cols.numel(), dtype=torch_dtype)
        sparse = dense.to_sparse_csr()

        dense_result = dense_make_bridge(dense, dofs_per_node)
        sparse_result = sparse_make_bridge(sparse, dofs_per_node)

        torch.testing.assert_close(sparse_result.to_dense(), dense_result)

    def test_rejects_non_csr_layout(self, torch_dtype: torch.dtype) -> None:
        dense = torch.zeros(4, 2, dtype=torch_dtype)
        with pytest.raises(ValueError, match="sparse CSR"):
            sparse_make_bridge(dense, 2)
