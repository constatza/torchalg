"""``torchalg.sparse.preconditioners.amg.variants`` tests - the load-bearing
multi-level sparsity-parity check.

This is the test that proves the ``form_sparse_sparse`` fix
(``torchalg.sparse.kernels.galerkin``) actually matters: before the fix, the
kernel always densified its output, so every call to ``AggregationCoarsening
.build_transfer`` beyond the first in a hierarchy silently operated on a
dense matrix - harmless for ``n_levels=2`` (no existing test built a deeper
hierarchy), but reproducing the exact cubic-blowup problem
``torchalg.sparse`` exists to avoid for ``n_levels > 2``. Reverting the
``.to_dense()`` removal in ``galerkin.py`` makes
``test_every_level_is_sparse_csr_densified_only_transiently_at_solve_time``
below fail (level 1's matrix comes back dense), which is the concrete
evidence this fix addresses.
"""

from __future__ import annotations

import pytest
import torch

from torchalg import pcg
from torchalg.preconditioners.implementations.amg import variants as dense_variants
from torchalg.sparse.preconditioners.amg import variants as sparse_variants

_N_LEVELS = 3


@pytest.fixture(
    params=[
        pytest.param("vcycle_amg", id="vcycle_amg"),
        pytest.param("wcycle_amg", id="wcycle_amg"),
    ]
)
def preset_name(request: pytest.FixtureRequest) -> str:
    """Parametrize over both preset factory names, dense and sparse alike."""
    return request.param


class TestSparseVCycleAMGMultiLevel:
    def test_every_level_is_sparse_csr_densified_only_transiently_at_solve_time(
        self, preset_name: str, poisson_1d_large_csr: torch.Tensor
    ) -> None:
        """Every hierarchy level, including the coarsest, must stay sparse CSR in storage.

        Regression coverage for the ``form_sparse_sparse`` bug: before the
        fix, level 1's matrix (the output of the *first* coarsening step,
        fed back in as the *second* coarsening step's input) was already
        dense, defeating the sparse hierarchy one level deep - every level
        from 1 onward silently lost the sparse advantage for ``n_levels >
        2``. Densification at the coarsest level happens only transiently,
        inside ``dense_coarse_solve`` at solve time (``coarse_solve.py``) -
        the hierarchy itself never stores a dense matrix.
        """
        factory = getattr(sparse_variants, preset_name)
        precond = factory(poisson_1d_large_csr, n_levels=_N_LEVELS)
        precond.setup(poisson_1d_large_csr)
        precond.apply(torch.ones(poisson_1d_large_csr.shape[0], dtype=poisson_1d_large_csr.dtype))

        hierarchy = precond._hierarchy
        assert hierarchy is not None
        assert len(hierarchy.levels) == _N_LEVELS
        for level in hierarchy.levels:
            assert level.matrix.is_sparse_csr, "every stored level must stay sparse CSR"

    def test_matches_equivalent_dense_preset(
        self,
        preset_name: str,
        poisson_1d_large_dense: torch.Tensor,
        poisson_1d_large_csr: torch.Tensor,
    ) -> None:
        """Sparse and dense presets on the same underlying matrix must agree."""
        rhs = torch.ones(poisson_1d_large_dense.shape[0], dtype=poisson_1d_large_dense.dtype)

        dense_factory = getattr(dense_variants, preset_name)
        sparse_factory = getattr(sparse_variants, preset_name)

        dense_precond = dense_factory(poisson_1d_large_dense, n_levels=_N_LEVELS)
        dense_precond.setup(poisson_1d_large_dense)
        sparse_precond = sparse_factory(poisson_1d_large_csr, n_levels=_N_LEVELS)
        sparse_precond.setup(poisson_1d_large_csr)

        expected = dense_precond.apply(rhs)
        actual = sparse_precond.apply(rhs)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_pcg_matches_equivalent_dense_preset(
        self,
        preset_name: str,
        poisson_1d_large_dense: torch.Tensor,
        poisson_1d_large_csr: torch.Tensor,
        poisson_1d_large_rhs: torch.Tensor,
    ) -> None:
        """A full PCG solve agrees when the hierarchy is dense or sparse CSR."""
        dense_preconditioner = getattr(dense_variants, preset_name)(
            poisson_1d_large_dense, n_levels=_N_LEVELS
        )
        dense_preconditioner.setup(poisson_1d_large_dense)
        sparse_preconditioner = getattr(sparse_variants, preset_name)(
            poisson_1d_large_csr, n_levels=_N_LEVELS
        )
        sparse_preconditioner.setup(poisson_1d_large_csr)

        dense_solution, dense_info = pcg(
            poisson_1d_large_dense,
            poisson_1d_large_rhs,
            preconditioner=dense_preconditioner,
            rtol=1e-8,
            maxiter=100,
        )
        sparse_solution, sparse_info = pcg(
            poisson_1d_large_csr,
            poisson_1d_large_rhs,
            preconditioner=sparse_preconditioner,
            rtol=1e-8,
            maxiter=100,
        )

        assert dense_info.converged and sparse_info.converged
        torch.testing.assert_close(sparse_solution, dense_solution, rtol=1e-5, atol=1e-6)
