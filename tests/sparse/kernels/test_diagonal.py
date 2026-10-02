"""Tests for ``torchalg.sparse.diagonal``."""

from __future__ import annotations

import pytest
import torch

from torchalg.sparse.kernels.diagonal import sparse_diagonal


class TestSparseDiagonal:
    """``sparse_diagonal`` matches ``torch.diagonal`` on the dense twin."""

    def test_matches_dense_diagonal(
        self, poisson_1d_dense: torch.Tensor, poisson_1d_csr: torch.Tensor
    ) -> None:
        """Extracted diagonal equals the dense matrix's diagonal exactly."""
        torch.testing.assert_close(
            sparse_diagonal(poisson_1d_csr), torch.diagonal(poisson_1d_dense)
        )

    def test_zero_diagonal_entry_is_zero(self, torch_dtype: torch.dtype) -> None:
        """A structurally-absent diagonal entry (no stored zero) reads back as 0."""
        matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch_dtype).to_sparse_csr()
        torch.testing.assert_close(sparse_diagonal(matrix), torch.zeros(2, dtype=torch_dtype))

    def test_rejects_non_csr_layout(self, poisson_1d_dense: torch.Tensor) -> None:
        """A dense (strided) tensor is rejected rather than silently misread."""
        with pytest.raises(ValueError, match="sparse CSR"):
            sparse_diagonal(poisson_1d_dense)

    def test_differentiable_through_values(self, torch_dtype: torch.dtype) -> None:
        """Gradients flow from the diagonal output back to the CSR ``values`` tensor."""
        dense = torch.eye(3, dtype=torch_dtype) * torch.tensor([2.0, 3.0, 4.0], dtype=torch_dtype)
        matrix = dense.to_sparse_csr()
        values = matrix.values().clone().requires_grad_(True)
        matrix = torch.sparse_csr_tensor(
            matrix.crow_indices(),
            matrix.col_indices(),
            values,
            size=matrix.shape,
            check_invariants=False,
        )
        sparse_diagonal(matrix).sum().backward()
        assert values.grad is not None
        torch.testing.assert_close(values.grad, torch.ones_like(values))
