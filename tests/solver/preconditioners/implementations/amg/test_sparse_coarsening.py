"""``AggregationCoarsening.build_transfer`` sparse dispatch (``A.is_sparse_csr``).

Cross-checks the sparse branch against the existing dense branch on the same
matrix - same algorithm, different storage/transfer-operator type, so the
two must agree on the coarse operator within tolerance. See
``docs/plan.md``'s "Wiring into existing preconditioners" and its
strength-of-connection/aggregation correction.
"""

from __future__ import annotations

import torch

from torchalg.preconditioners.implementations.amg.coarsening import (
    AggregationCoarsening,
    TargetDimensionCoarsening,
)
from torchalg.preconditioners.implementations.amg.transfer import DenseTransferOperator
from torchalg.sparse import SparseTransferOperator

_THETA = 0.25


class TestAggregationCoarseningSparseDispatch:
    """Sparse ``A`` in produces a sparse transfer operator matching the dense path."""

    def test_returns_sparse_transfer_operator(self, poisson_1d: torch.Tensor) -> None:
        """A sparse CSR ``A`` yields a ``SparseTransferOperator``, not the dense one."""
        _, transfer = AggregationCoarsening(theta=_THETA).build_transfer(poisson_1d.to_sparse_csr())
        assert isinstance(transfer, SparseTransferOperator)

    def test_dense_input_still_returns_dense_transfer_operator(
        self, poisson_1d: torch.Tensor
    ) -> None:
        """Dense ``A`` in is completely unaffected by the new sparse branch."""
        _, transfer = AggregationCoarsening(theta=_THETA).build_transfer(poisson_1d)
        assert isinstance(transfer, DenseTransferOperator)

    def test_coarse_matrix_matches_dense_path(self, poisson_1d: torch.Tensor) -> None:
        """The coarse operator is the same (within tolerance) whether A is dense or sparse."""
        expected_coarse, _ = AggregationCoarsening(theta=_THETA).build_transfer(poisson_1d)
        actual_coarse, _ = AggregationCoarsening(theta=_THETA).build_transfer(
            poisson_1d.to_sparse_csr()
        )
        torch.testing.assert_close(actual_coarse, expected_coarse)

    def test_prolongate_matches_dense_path(self, poisson_1d: torch.Tensor) -> None:
        """Prolongating the same coarse vector gives the same fine-grid vector either way."""
        coarse_matrix, dense_transfer = AggregationCoarsening(theta=_THETA).build_transfer(
            poisson_1d
        )
        _, sparse_transfer = AggregationCoarsening(theta=_THETA).build_transfer(
            poisson_1d.to_sparse_csr()
        )
        n_coarse = coarse_matrix.shape[0]
        coarse_vector = torch.arange(n_coarse, dtype=poisson_1d.dtype) + 1.0
        torch.testing.assert_close(
            sparse_transfer.prolongate(coarse_vector), dense_transfer.prolongate(coarse_vector)
        )


class TestTargetDimensionCoarseningSparseDispatch:
    """``TargetDimensionCoarsening``'s search probe also works against sparse ``A``."""

    def test_realized_dimension_matches_dense_path(self, poisson_1d: torch.Tensor) -> None:
        """The realized coarse dimension is identical whether A is dense or sparse."""

        def search() -> TargetDimensionCoarsening:
            return TargetDimensionCoarsening(
                target_coarse_dim=5, theta_min=0.01, theta_max=0.9, step=0.05
            )

        dense_coarse, _ = search().build_transfer(poisson_1d)
        sparse_coarse, sparse_transfer = search().build_transfer(poisson_1d.to_sparse_csr())
        assert isinstance(sparse_transfer, SparseTransferOperator)
        assert sparse_coarse.shape == dense_coarse.shape
