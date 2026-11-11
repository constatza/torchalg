"""Tests for ``torchalg.sparse.kernels.index_dtype``."""

from __future__ import annotations

import torch

from torchalg.sparse.kernels.index_dtype import compact_csr_indices


class TestCompactCsrIndices:
    def test_narrows_indices_and_shares_values(self, poisson_2d_csr: torch.Tensor) -> None:
        compact = compact_csr_indices(poisson_2d_csr)

        assert compact.crow_indices().dtype == torch.int32
        assert compact.col_indices().dtype == torch.int32
        assert compact.values().data_ptr() == poisson_2d_csr.values().data_ptr()

    def test_products_are_bit_identical(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_dense: torch.Tensor
    ) -> None:
        vector = torch.linspace(-1.0, 2.0, poisson_2d_dense.shape[0], dtype=poisson_2d_dense.dtype)
        block = torch.stack([vector, vector.flip(0)], dim=1)
        compact = compact_csr_indices(poisson_2d_csr)

        assert torch.equal(compact @ vector, poisson_2d_csr @ vector)
        assert torch.equal(compact @ block, poisson_2d_csr @ block)

    def test_non_csr_and_already_compact_are_returned_unchanged(
        self, poisson_2d_csr: torch.Tensor, poisson_2d_dense: torch.Tensor
    ) -> None:
        compact = compact_csr_indices(poisson_2d_csr)

        assert compact_csr_indices(poisson_2d_dense) is poisson_2d_dense
        assert compact_csr_indices(compact) is compact
