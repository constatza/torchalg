"""Dense/sparse parity tests for ``BootCMatchCoarsening.build_transfer``.

Same oracle-testing discipline used throughout this session: dense and
sparse ``BootCMatchCoarsening`` run on the same matrix and the same smooth
vector ``w`` must produce the same ``A_coarse`` and the same ``P`` (sparse
``.to_dense()`` vs dense), via ``torch.testing.assert_close``.
"""

from __future__ import annotations

import torch

from torchalg.preconditioners.implementations.amg.bootcmatch_coarsening import (
    BootCMatchCoarsening as DenseBootCMatchCoarsening,
)
from torchalg.sparse.preconditioners.amg.bootcmatch_coarsening import (
    BootCMatchCoarsening as SparseBootCMatchCoarsening,
)


def test_dense_and_sparse_build_transfer_agree(
    poisson_1d_dense: torch.Tensor, poisson_1d_csr: torch.Tensor
) -> None:
    n = poisson_1d_dense.shape[0]
    generator = torch.Generator().manual_seed(0)
    w = torch.rand(n, dtype=poisson_1d_dense.dtype, generator=generator)

    dense_coarsening = DenseBootCMatchCoarsening(w.clone())
    sparse_coarsening = SparseBootCMatchCoarsening(w.clone())

    dense_coarse, _ = dense_coarsening.build_transfer(poisson_1d_dense)
    sparse_coarse, _ = sparse_coarsening.build_transfer(poisson_1d_csr)

    torch.testing.assert_close(sparse_coarse.to_dense(), dense_coarse)
    torch.testing.assert_close(
        sparse_coarsening.last_prolongation.to_dense(), dense_coarsening.last_prolongation
    )


def test_smooth_vector_restricted_consistently_across_trees(
    poisson_1d_dense: torch.Tensor, poisson_1d_csr: torch.Tensor
) -> None:
    n = poisson_1d_dense.shape[0]
    generator = torch.Generator().manual_seed(1)
    w = torch.rand(n, dtype=poisson_1d_dense.dtype, generator=generator)

    dense_coarsening = DenseBootCMatchCoarsening(w.clone())
    sparse_coarsening = SparseBootCMatchCoarsening(w.clone())

    dense_coarse, _ = dense_coarsening.build_transfer(poisson_1d_dense)
    sparse_coarse, _ = sparse_coarsening.build_transfer(poisson_1d_csr)

    dense_w_coarse = dense_coarsening.smooth_vector_for(dense_coarse)
    sparse_w_coarse = sparse_coarsening.smooth_vector_for(sparse_coarse)

    torch.testing.assert_close(sparse_w_coarse, dense_w_coarse)
