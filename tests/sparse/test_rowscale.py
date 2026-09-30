"""Tests for ``torchalg.sparse.rowscale``."""

from __future__ import annotations

import torch

from torchalg.sparse.rowscale import sparse_row_scale


class TestSparseRowScale:
    """``sparse_row_scale`` matches dense per-row broadcasting multiply."""

    def test_matches_dense_reference(
        self, poisson_1d_dense: torch.Tensor, poisson_1d_csr: torch.Tensor
    ) -> None:
        """Row-scaling the sparse matrix matches scaling its dense twin."""
        scale = torch.arange(1, poisson_1d_dense.shape[0] + 1, dtype=poisson_1d_dense.dtype)
        expected = scale.unsqueeze(1) * poisson_1d_dense
        actual = sparse_row_scale(poisson_1d_csr, scale)
        torch.testing.assert_close(actual.to_dense(), expected)

    def test_returns_sparse_csr(self, poisson_1d_csr: torch.Tensor) -> None:
        """The scaled result stays sparse CSR."""
        scale = torch.ones(poisson_1d_csr.shape[0], dtype=poisson_1d_csr.dtype)
        assert sparse_row_scale(poisson_1d_csr, scale).layout == torch.sparse_csr
