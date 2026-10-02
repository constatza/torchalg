"""``torchalg.sparse.preconditioners.pod.coarsening.PODCoarseningStrategy`` tests.

Cross-checks the standalone sparse ``PODCoarseningStrategy`` against the
dense sibling (``torchalg.preconditioners.implementations.pod.coarsening
.PODCoarseningStrategy``) on the same fitted basis and fine-grid matrix -
same algorithm, two classes now instead of one class's two branches
(``docs/plan.md``'s "Correction: dense and sparse must be separate
implementations, not an internal branch"). Mirrors
``tests/sparse/preconditioners/amg/test_coarsening.py``'s pattern.
"""

from __future__ import annotations

import torch

from torchalg.preconditioners.implementations.pod.coarsening import (
    PODCoarseningStrategy as DensePODCoarseningStrategy,
)
from torchalg.sparse.preconditioners.pod.coarsening import PODCoarseningStrategy
from torchalg.utils.dense_transfer import DenseTransferOperator

_RANK = 5


class TestPODCoarseningSparseMatchesDense:
    """Sparse ``PODCoarseningStrategy`` agrees with the dense sibling."""

    def test_returns_dense_transfer_operator(
        self, poisson_1d_dense: torch.Tensor, pod_snapshots: torch.Tensor
    ) -> None:
        """Building from sparse CSR ``A`` still yields a ``DenseTransferOperator``.

        Unlike AMG's sparse sibling, POD's basis Phi_r is always dense, so
        the transfer operator is unaffected by A's format - both siblings
        return the exact same class.
        """
        strategy = PODCoarseningStrategy(rank=_RANK)
        strategy.fit(pod_snapshots)
        _, transfer = strategy.build_transfer(poisson_1d_dense.to_sparse_csr())
        assert isinstance(transfer, DenseTransferOperator)

    def test_coarse_matrix_matches_dense_sibling(
        self, poisson_1d_dense: torch.Tensor, pod_snapshots: torch.Tensor
    ) -> None:
        """The coarse operator is the same (within tolerance) whether built dense or sparse."""
        dense_strategy = DensePODCoarseningStrategy(rank=_RANK)
        dense_strategy.fit(pod_snapshots)
        expected_coarse, _ = dense_strategy.build_transfer(poisson_1d_dense)

        sparse_strategy = PODCoarseningStrategy(rank=_RANK)
        sparse_strategy.fit(pod_snapshots)
        actual_coarse, _ = sparse_strategy.build_transfer(poisson_1d_dense.to_sparse_csr())

        torch.testing.assert_close(actual_coarse, expected_coarse)

    def test_prolongate_matches_dense_sibling(
        self, poisson_1d_dense: torch.Tensor, pod_snapshots: torch.Tensor
    ) -> None:
        """Prolongating the same coarse vector gives the same fine-grid vector either way."""
        dense_strategy = DensePODCoarseningStrategy(rank=_RANK)
        dense_strategy.fit(pod_snapshots)
        coarse_matrix, dense_transfer = dense_strategy.build_transfer(poisson_1d_dense)

        sparse_strategy = PODCoarseningStrategy(rank=_RANK)
        sparse_strategy.fit(pod_snapshots)
        _, sparse_transfer = sparse_strategy.build_transfer(poisson_1d_dense.to_sparse_csr())

        n_coarse = coarse_matrix.shape[0]
        coarse_vector = torch.arange(n_coarse, dtype=poisson_1d_dense.dtype) + 1.0
        torch.testing.assert_close(
            sparse_transfer.prolongate(coarse_vector), dense_transfer.prolongate(coarse_vector)
        )

    def test_restrict_matches_dense_sibling(
        self, poisson_1d_dense: torch.Tensor, pod_snapshots: torch.Tensor
    ) -> None:
        """Restricting the same fine-grid vector gives the same coarse vector either way."""
        dense_strategy = DensePODCoarseningStrategy(rank=_RANK)
        dense_strategy.fit(pod_snapshots)
        _, dense_transfer = dense_strategy.build_transfer(poisson_1d_dense)

        sparse_strategy = PODCoarseningStrategy(rank=_RANK)
        sparse_strategy.fit(pod_snapshots)
        _, sparse_transfer = sparse_strategy.build_transfer(poisson_1d_dense.to_sparse_csr())

        n_fine = poisson_1d_dense.shape[0]
        fine_vector = torch.arange(n_fine, dtype=poisson_1d_dense.dtype) + 1.0
        torch.testing.assert_close(
            sparse_transfer.restrict(fine_vector), dense_transfer.restrict(fine_vector)
        )
