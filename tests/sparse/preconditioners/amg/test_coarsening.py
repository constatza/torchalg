"""``torchalg.sparse.preconditioners.amg.coarsening.AggregationCoarsening`` tests.

Cross-checks the standalone sparse ``AggregationCoarsening`` against the
dense sibling (``torchalg.preconditioners.implementations.amg.coarsening
.AggregationCoarsening``) on the same matrix - same algorithm, two classes
now instead of one class's two branches (``docs/plan.md``'s "Correction:
dense and sparse must be separate implementations, not an internal
branch"). Replaces the deleted
``tests/solver/preconditioners/implementations/amg/test_sparse_coarsening.py``.
"""

from __future__ import annotations

from typing import TypedDict

import torch

from torchalg.preconditioners.implementations.amg.coarsening import (
    AggregationCoarsening as DenseAggregationCoarsening,
)
from torchalg.preconditioners.implementations.amg.coarsening import (
    TargetDimensionCoarsening as DenseTargetDimensionCoarsening,
)
from torchalg.preconditioners.implementations.amg.transfer import DenseTransferOperator
from torchalg.sparse.preconditioners.amg.coarsening import (
    AggregationCoarsening,
    TargetDimensionCoarsening,
)
from torchalg.sparse.preconditioners.amg.transfer import SparseTransferOperator

_THETA = 0.25


class _TargetDimensionSearchKwargs(TypedDict):
    """Keyword parameters shared by dense and sparse theta-search tests."""

    theta_min: float
    theta_max: float
    step: float
    omega: float


def _target_dimension_search_kwargs() -> _TargetDimensionSearchKwargs:
    """Return the validated adaptive-theta parameters shared by both siblings."""
    return {"theta_min": 0.01, "theta_max": 0.5, "step": 0.01, "omega": 0.67}


class TestAggregationCoarseningSparseMatchesDense:
    """Sparse ``AggregationCoarsening`` agrees with the dense sibling."""

    def test_returns_sparse_transfer_operator(self, poisson_1d_dense: torch.Tensor) -> None:
        """Building from sparse CSR ``A`` yields a ``SparseTransferOperator``."""
        _, transfer = AggregationCoarsening(theta=_THETA).build_transfer(
            poisson_1d_dense.to_sparse_csr()
        )
        assert isinstance(transfer, SparseTransferOperator)

    def test_dense_sibling_still_returns_dense_transfer_operator(
        self, poisson_1d_dense: torch.Tensor
    ) -> None:
        """The dense sibling class is unaffected by the sparse class existing."""
        _, transfer = DenseAggregationCoarsening(theta=_THETA).build_transfer(poisson_1d_dense)
        assert isinstance(transfer, DenseTransferOperator)

    def test_coarse_matrix_matches_dense_sibling(self, poisson_1d_dense: torch.Tensor) -> None:
        """The coarse operator is the same (within tolerance) whether built dense or sparse."""
        expected_coarse, _ = DenseAggregationCoarsening(theta=_THETA).build_transfer(
            poisson_1d_dense
        )
        actual_coarse, _ = AggregationCoarsening(theta=_THETA).build_transfer(
            poisson_1d_dense.to_sparse_csr()
        )
        assert actual_coarse.layout == torch.sparse_csr
        torch.testing.assert_close(actual_coarse.to_dense(), expected_coarse)

    def test_prolongate_matches_dense_sibling(self, poisson_1d_dense: torch.Tensor) -> None:
        """Prolongating the same coarse vector gives the same fine-grid vector either way."""
        coarse_matrix, dense_transfer = DenseAggregationCoarsening(theta=_THETA).build_transfer(
            poisson_1d_dense
        )
        _, sparse_transfer = AggregationCoarsening(theta=_THETA).build_transfer(
            poisson_1d_dense.to_sparse_csr()
        )
        n_coarse = coarse_matrix.shape[0]
        coarse_vector = torch.arange(n_coarse, dtype=poisson_1d_dense.dtype) + 1.0
        torch.testing.assert_close(
            sparse_transfer.prolongate(coarse_vector), dense_transfer.prolongate(coarse_vector)
        )


class TestTargetDimensionCoarseningSparseMatchesDense:
    """Sparse `TargetDimensionCoarsening` agrees with the dense sibling.

    ``_target_dimension_search_kwargs`` reuses the ``theta_min=0.01, theta_max=0.5,
    step=0.01, omega=0.67`` parameters already validated by
    `tests/solver/preconditioners/implementations/test_amg.py
    ::TestTargetDimensionCoarsening` against `poisson_1d`, rather than
    inventing new, unvalidated ones.
    """

    def test_returns_sparse_transfer_operator(self, poisson_1d_dense: torch.Tensor) -> None:
        """Building from sparse CSR ``A`` yields a ``SparseTransferOperator``."""
        coarsening = TargetDimensionCoarsening(
            target_coarse_dim=5, **_target_dimension_search_kwargs()
        )
        _, transfer = coarsening.build_transfer(poisson_1d_dense.to_sparse_csr())
        assert isinstance(transfer, SparseTransferOperator)

    def test_dense_sibling_still_returns_dense_transfer_operator(
        self, poisson_1d_dense: torch.Tensor
    ) -> None:
        """The dense sibling class is unaffected by the sparse class existing."""
        coarsening = DenseTargetDimensionCoarsening(
            target_coarse_dim=5, **_target_dimension_search_kwargs()
        )
        _, transfer = coarsening.build_transfer(poisson_1d_dense)
        assert isinstance(transfer, DenseTransferOperator)

    def test_realized_coarse_dim_matches_dense_sibling(
        self, poisson_1d_dense: torch.Tensor
    ) -> None:
        """Both siblings land on the same realized coarse dimension for the same target."""
        dense_coarsening = DenseTargetDimensionCoarsening(
            target_coarse_dim=5, **_target_dimension_search_kwargs()
        )
        sparse_coarsening = TargetDimensionCoarsening(
            target_coarse_dim=5, **_target_dimension_search_kwargs()
        )
        dense_coarsening.build_transfer(poisson_1d_dense)
        sparse_coarsening.build_transfer(poisson_1d_dense.to_sparse_csr())
        assert sparse_coarsening.realized_coarse_dim(
            poisson_1d_dense.to_sparse_csr()
        ) == dense_coarsening.realized_coarse_dim(poisson_1d_dense)

    def test_coarse_matrix_matches_dense_sibling(self, poisson_1d_dense: torch.Tensor) -> None:
        """The coarse operator is the same (within tolerance) whether built dense or sparse."""
        expected_coarse, _ = DenseTargetDimensionCoarsening(
            target_coarse_dim=5, **_target_dimension_search_kwargs()
        ).build_transfer(poisson_1d_dense)
        actual_coarse, _ = TargetDimensionCoarsening(
            target_coarse_dim=5, **_target_dimension_search_kwargs()
        ).build_transfer(poisson_1d_dense.to_sparse_csr())
        assert actual_coarse.layout == torch.sparse_csr
        torch.testing.assert_close(actual_coarse.to_dense(), expected_coarse)

    def test_prolongate_matches_dense_sibling(self, poisson_1d_dense: torch.Tensor) -> None:
        """Prolongating the same coarse vector gives the same fine-grid vector either way."""
        coarse_matrix, dense_transfer = DenseTargetDimensionCoarsening(
            target_coarse_dim=5, **_target_dimension_search_kwargs()
        ).build_transfer(poisson_1d_dense)
        _, sparse_transfer = TargetDimensionCoarsening(
            target_coarse_dim=5, **_target_dimension_search_kwargs()
        ).build_transfer(poisson_1d_dense.to_sparse_csr())
        n_coarse = coarse_matrix.shape[0]
        coarse_vector = torch.arange(n_coarse, dtype=poisson_1d_dense.dtype) + 1.0
        torch.testing.assert_close(
            sparse_transfer.prolongate(coarse_vector), dense_transfer.prolongate(coarse_vector)
        )
