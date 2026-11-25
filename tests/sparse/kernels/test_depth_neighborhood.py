"""Tests for ``torchalg.sparse.kernels.depth_neighborhood.sparse_depth_neighborhood``.

Dense-as-oracle, this project's standing rule: checked against
``preconditioners.implementations.amg._graph.depth_neighborhood`` on shared
fixtures for every depth it's actually used at (BAMG's default depth=1 for
the algebraic-distance neighborhood itself, and depth+2 for the LS-ring
neighborhood).
"""

from __future__ import annotations

import warnings

import pytest
import torch

from torchalg.preconditioners.implementations.amg._graph import (
    depth_neighborhood as dense_depth_neighborhood,
)
from torchalg.sparse.kernels.depth_neighborhood import sparse_depth_neighborhood

_ASYMMETRIC_SEED = 11
_ASYMMETRIC_SIZE = 40
_ASYMMETRIC_DENSITY = 0.08


@pytest.fixture
def asymmetric_dense() -> torch.Tensor:
    """Random nonsymmetric matrix with a full stored diagonal, so reach has cycles."""
    generator = torch.Generator().manual_seed(_ASYMMETRIC_SEED)
    mask = torch.rand(_ASYMMETRIC_SIZE, _ASYMMETRIC_SIZE, generator=generator) < _ASYMMETRIC_DENSITY
    matrix = torch.randn(
        _ASYMMETRIC_SIZE, _ASYMMETRIC_SIZE, generator=generator, dtype=torch.float64
    )
    return matrix * mask + torch.eye(_ASYMMETRIC_SIZE, dtype=torch.float64)


class TestSparseDepthNeighborhood:
    @pytest.mark.parametrize("depth", [1, 2, 3, 4])
    def test_matches_dense_on_asymmetric_pattern(
        self, asymmetric_dense: torch.Tensor, depth: int
    ) -> None:
        expected = dense_depth_neighborhood(asymmetric_dense, depth)
        actual = sparse_depth_neighborhood(asymmetric_dense.to_sparse_csr(), depth)
        torch.testing.assert_close(actual.to_dense(), expected)

    @pytest.mark.parametrize("depth", [1, 3])
    def test_columns_ascend_within_every_row(
        self, asymmetric_dense: torch.Tensor, depth: int
    ) -> None:
        """Consumers look rows up with ``searchsorted``; the CSR product emits unsorted columns."""
        result = sparse_depth_neighborhood(asymmetric_dense.to_sparse_csr(), depth)
        n = result.shape[0]
        row = torch.repeat_interleave(torch.arange(n), result.crow_indices().diff())
        keys = row * n + result.col_indices()
        assert bool((keys[1:] > keys[:-1]).all())

    def test_accepts_compact_int32_indices(self, asymmetric_dense: torch.Tensor) -> None:
        csr = asymmetric_dense.to_sparse_csr()
        compact = torch.sparse_csr_tensor(
            csr.crow_indices().int(),
            csr.col_indices().int(),
            csr.values(),
            size=csr.shape,
            check_invariants=False,
        )
        expected = sparse_depth_neighborhood(csr, 3)
        torch.testing.assert_close(
            sparse_depth_neighborhood(compact, 3).to_dense(), expected.to_dense()
        )

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_matches_dense_on_poisson_1d(self, poisson_1d_dense: torch.Tensor, depth: int) -> None:
        expected = dense_depth_neighborhood(poisson_1d_dense, depth)
        actual = sparse_depth_neighborhood(poisson_1d_dense.to_sparse_csr(), depth)
        torch.testing.assert_close(actual.to_dense(), expected)

    @pytest.mark.parametrize("depth", [1, 2, 3])
    def test_matches_dense_on_poisson_2d(self, poisson_2d_dense: torch.Tensor, depth: int) -> None:
        expected = dense_depth_neighborhood(poisson_2d_dense, depth)
        actual = sparse_depth_neighborhood(poisson_2d_dense.to_sparse_csr(), depth)
        torch.testing.assert_close(actual.to_dense(), expected)

    def test_result_is_boolean_sparse_csr(self, poisson_1d_csr: torch.Tensor) -> None:
        result = sparse_depth_neighborhood(poisson_1d_csr, 1)
        assert result.layout == torch.sparse_csr
        assert result.dtype == torch.bool

    def test_explicitly_disables_costly_sparse_invariant_checks(
        self, poisson_1d_csr: torch.Tensor
    ) -> None:
        """Construction selects the documented unchecked fast path without warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = sparse_depth_neighborhood(poisson_1d_csr, 2)
        assert result.layout == torch.sparse_csr

    def test_rejects_non_csr_input(self, poisson_1d_dense: torch.Tensor) -> None:
        with pytest.raises(ValueError, match="sparse CSR"):
            sparse_depth_neighborhood(poisson_1d_dense, 1)
