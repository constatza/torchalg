"""Tests for ``torchalg.sparse.preconditioners.amg.adaptive``.

Dense-as-oracle, this project's standing rule: the sparse
``adaptive_sa_hierarchy``/``AdaptiveSAPreconditioner`` are checked against
their dense siblings given the same matrix (one dense, one its sparse CSR
twin) and the same seeded draw - full "two implementations, same result"
parity, matching ``tests/sparse/preconditioners/amg/test_bootstrap.py``'s
existing pattern for ``BAMGCoarsening``.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from torchalg import pcg
from torchalg.preconditioners.implementations.amg.adaptive import (
    AdaptiveSAPreconditioner as DenseAdaptiveSAPreconditioner,
)
from torchalg.preconditioners.implementations.amg.adaptive import (
    adaptive_sa_hierarchy as dense_adaptive_sa_hierarchy,
)
from torchalg.sparse.preconditioners.amg.adaptive import (
    AdaptiveSAPreconditioner,
    adaptive_sa_hierarchy,
)


@pytest.fixture
def seeded_draw_factory(
    torch_dtype: torch.dtype,
) -> Callable[[int], Callable[[int], torch.Tensor]]:
    def _factory(seed: int) -> Callable[[int], torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)

        def _draw(n: int) -> torch.Tensor:
            return torch.rand(n, generator=generator, dtype=torch_dtype)

        return _draw

    return _factory


class TestAdaptiveSAHierarchyMatchesDense:
    def test_matches_dense_on_poisson_1d(
        self,
        poisson_1d_large_dense: torch.Tensor,
        seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
    ) -> None:
        sparse_matrix = poisson_1d_large_dense.to_sparse_csr()

        dense_result = dense_adaptive_sa_hierarchy(
            poisson_1d_large_dense,
            num_candidates=2,
            candidate_iters=2,
            max_levels=3,
            max_coarse=4,
            draw=seeded_draw_factory(0),
        )
        sparse_result = adaptive_sa_hierarchy(
            sparse_matrix,
            num_candidates=2,
            candidate_iters=2,
            max_levels=3,
            max_coarse=4,
            draw=seeded_draw_factory(0),
        )

        assert len(sparse_result.matrices) == len(dense_result.matrices)
        for sparse_matrix_level, dense_matrix_level in zip(
            sparse_result.matrices, dense_result.matrices, strict=True
        ):
            torch.testing.assert_close(
                sparse_matrix_level.to_dense(), dense_matrix_level, atol=1e-8, rtol=1e-6
            )
        for sparse_p, dense_p in zip(
            sparse_result.prolongations, dense_result.prolongations, strict=True
        ):
            torch.testing.assert_close(sparse_p.to_dense(), dense_p, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(
            sparse_result.candidates, dense_result.candidates, atol=1e-8, rtol=1e-6
        )

    def test_matches_dense_on_poisson_2d(
        self,
        poisson_2d_dense: torch.Tensor,
        seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
    ) -> None:
        sparse_matrix = poisson_2d_dense.to_sparse_csr()

        dense_result = dense_adaptive_sa_hierarchy(
            poisson_2d_dense,
            num_candidates=2,
            candidate_iters=2,
            max_levels=2,
            max_coarse=2,
            draw=seeded_draw_factory(1),
        )
        sparse_result = adaptive_sa_hierarchy(
            sparse_matrix,
            num_candidates=2,
            candidate_iters=2,
            max_levels=2,
            max_coarse=2,
            draw=seeded_draw_factory(1),
        )

        assert len(sparse_result.matrices) == len(dense_result.matrices)
        for sparse_matrix_level, dense_matrix_level in zip(
            sparse_result.matrices, dense_result.matrices, strict=True
        ):
            torch.testing.assert_close(
                sparse_matrix_level.to_dense(), dense_matrix_level, atol=1e-8, rtol=1e-6
            )


class TestAdaptiveSAPreconditionerMatchesDense:
    def test_apply_matches_dense(
        self,
        poisson_1d_large_dense: torch.Tensor,
        seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
    ) -> None:
        torch.manual_seed(0)
        sparse_matrix = poisson_1d_large_dense.to_sparse_csr()
        residual = torch.randn(poisson_1d_large_dense.shape[0], dtype=poisson_1d_large_dense.dtype)

        dense_preconditioner = DenseAdaptiveSAPreconditioner(
            num_candidates=2,
            candidate_iters=2,
            max_levels=3,
            max_coarse=4,
            draw=seeded_draw_factory(0),
        )
        dense_preconditioner.setup(poisson_1d_large_dense)
        sparse_preconditioner = AdaptiveSAPreconditioner(
            num_candidates=2,
            candidate_iters=2,
            max_levels=3,
            max_coarse=4,
            draw=seeded_draw_factory(0),
        )
        sparse_preconditioner.setup(sparse_matrix)

        dense_out = dense_preconditioner.apply(residual)
        sparse_out = sparse_preconditioner.apply(residual)
        torch.testing.assert_close(sparse_out, dense_out, atol=1e-6, rtol=1e-5)

    def test_pcg_matches_dense(
        self,
        poisson_1d_large_dense: torch.Tensor,
        poisson_1d_large_rhs: torch.Tensor,
        seeded_draw_factory: Callable[[int], Callable[[int], torch.Tensor]],
    ) -> None:
        """The full adaptive-SA-preconditioned PCG solve matches its dense sibling."""
        dense_preconditioner = DenseAdaptiveSAPreconditioner(
            num_candidates=2,
            candidate_iters=2,
            max_levels=3,
            max_coarse=4,
            draw=seeded_draw_factory(0),
        )
        dense_preconditioner.setup(poisson_1d_large_dense)
        sparse_preconditioner = AdaptiveSAPreconditioner(
            num_candidates=2,
            candidate_iters=2,
            max_levels=3,
            max_coarse=4,
            draw=seeded_draw_factory(0),
        )
        sparse_preconditioner.setup(poisson_1d_large_dense.to_sparse_csr())

        dense_solution, dense_info = pcg(
            poisson_1d_large_dense,
            poisson_1d_large_rhs,
            preconditioner=dense_preconditioner,
            rtol=1e-8,
            maxiter=100,
        )
        sparse_solution, sparse_info = pcg(
            poisson_1d_large_dense.to_sparse_csr(),
            poisson_1d_large_rhs,
            preconditioner=sparse_preconditioner,
            rtol=1e-8,
            maxiter=100,
        )

        assert dense_info.converged and sparse_info.converged
        torch.testing.assert_close(sparse_solution, dense_solution, rtol=1e-5, atol=1e-6)

    def test_raises_on_single_level(self, poisson_1d_dense: torch.Tensor) -> None:
        sparse_matrix = poisson_1d_dense.to_sparse_csr()
        preconditioner = AdaptiveSAPreconditioner(max_coarse=poisson_1d_dense.shape[0])
        with pytest.raises(ValueError, match="single level"):
            preconditioner.setup(sparse_matrix)
